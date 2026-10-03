import pytest
import torch
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights, ResNet18_Weights, ResNet50_Weights, ViT_B_16_Weights
from torchvision.models.vision_transformer import VisionTransformer

from scripts import eval_perf


def test_large_crop_resizes_instead_of_padding():
    image = Image.new("RGB", (480, 320), color="white")
    output = eval_perf._build_transform(384)(image)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    assert output.shape == (3, 384, 384)
    # A white image must stay white: the old pipeline padded one sixth of it black.
    torch.testing.assert_close(output * std + mean, torch.ones_like(output))


def test_default_crop_preserves_historical_preprocessing():
    generator = torch.Generator().manual_seed(12)
    pixels = torch.randint(256, (320, 480, 3), dtype=torch.uint8, generator=generator)
    image = Image.fromarray(pixels.numpy())

    torch.testing.assert_close(
        eval_perf._build_transform(224)(image), ResNet18_Weights.IMAGENET1K_V1.transforms()(image)
    )


def test_weights_keep_historical_checkpoints_and_allow_override():
    assert eval_perf._resolve_weights("resnet18", None) is ResNet18_Weights.IMAGENET1K_V1
    assert eval_perf._resolve_weights("mobilenet_v3_large", None) is MobileNet_V3_Large_Weights.IMAGENET1K_V2
    assert eval_perf._resolve_weights("mobilenet_v3_large", "IMAGENET1K_V1") is MobileNet_V3_Large_Weights.IMAGENET1K_V1
    assert eval_perf._resolve_weights("resnet50", None) is ResNet50_Weights.DEFAULT
    with pytest.raises(ValueError, match="unknown weights"):
        eval_perf._resolve_weights("resnet18", "invalid")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (("data", "UnknownCAM"), "invalid choice"),
        (("data", "GradCAM", "--size", "0"), "expected a positive integer"),
        (("data", "GradCAM", "--batch-size", "0"), "expected a positive integer"),
        (("data", "GradCAM", "--di-steps", "0"), "expected a positive integer"),
        (("data", "GradCAM", "--di-batch-size", "-1"), "expected a positive integer"),
        (("data", "GradCAM", "--workers", "-1"), "expected a non-negative integer"),
        (("data", "GradCAM", "--seed", "-1"), "expected a non-negative integer"),
    ],
)
def test_cli_rejects_invalid_values(argv, message, capsys):
    with pytest.raises(SystemExit):
        eval_perf._build_parser().parse_args(argv)
    assert message in capsys.readouterr().err


def test_seeded_benchmark_reports_resolved_protocol(tmp_path, monkeypatch, capsys):
    image_dir = tmp_path / "val" / "class"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (24, 16), color="white").save(image_dir / "first.png")
    Image.new("RGB", (24, 16), color="red").save(image_dir / "second.png")
    initial_weights = []

    def make_model(_arch, *, weights):
        assert weights is ResNet18_Weights.IMAGENET1K_V1
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(4, 2),
        )
        initial_weights.append(model[0].weight.detach().clone())
        return model

    monkeypatch.setattr(eval_perf, "get_model", make_model)
    argv = [
        str(tmp_path),
        "SmoothGradCAMpp",
        "--arch",
        "resnet18",
        "--size",
        "16",
        "--target",
        "1",
        "--workers",
        "0",
        "--device",
        "cpu",
        "--seed",
        "12",
        "--deletion-insertion",
        "--di-steps",
        "2",
    ]
    eval_perf.main(eval_perf._build_parser().parse_args(argv))
    first_output = capsys.readouterr().out
    first_rng = torch.get_rng_state()
    eval_perf.main(eval_perf._build_parser().parse_args(argv))

    assert capsys.readouterr().out == first_output
    torch.testing.assert_close(torch.get_rng_state(), first_rng)
    torch.testing.assert_close(initial_weights[0], initial_weights[1])
    for detail in (
        "seed=12",
        "weights=ResNet18_Weights.IMAGENET1K_V1",
        "target_layers=1",
        "checkpoint=https://download.pytorch.org/models/resnet18-",
        f"dataset={tmp_path / 'val'} samples=2",
        "python=",
        "torch=",
        "torchvision=",
        "torchcam=",
        "resize=18",
        "crop=16",
        "target=original-predicted-class",
        "masking=normalized-input",
        "steps=2",
        "baseline=normalized-zero-for-both-curves",
        "cam_draws=separate-per-metric",
        "Valid 2 samples, Skipped 0 samples",
    ):
        assert detail in first_output


@pytest.mark.parametrize(
    "target", ["encoder.layers.encoder_layer_1", "encoder.layers.encoder_layer_0,encoder.layers.encoder_layer_1"]
)
def test_legrad_benchmark_with_explicit_transformer_blocks(tmp_path, monkeypatch, capsys, target):
    image_dir = tmp_path / "val" / "class"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (48, 32), color="white").save(image_dir / "first.png")
    Image.new("RGB", (48, 32), color="red").save(image_dir / "second.png")

    def make_model(arch, *, weights):
        assert arch == "vit_b_16"
        assert weights is ViT_B_16_Weights.IMAGENET1K_V1
        model = VisionTransformer(
            image_size=32,
            patch_size=8,
            num_layers=2,
            num_heads=2,
            hidden_dim=32,
            mlp_dim=64,
            num_classes=3,
        )
        torch.nn.init.normal_(model.heads.head.weight)
        return model

    monkeypatch.setattr(eval_perf, "get_model", make_model)
    args = eval_perf._build_parser().parse_args([
        str(tmp_path),
        "LeGrad",
        "--arch",
        "vit_b_16",
        "--target",
        target,
        "--size",
        "32",
        "--device",
        "cpu",
        "--workers",
        "0",
        "--deletion-insertion",
        "--di-steps",
        "2",
    ])
    eval_perf.main(args)
    output = capsys.readouterr().out

    assert f"target_layers={target}" in output
    assert "Average Drop" in output
    assert "Deletion AUC" in output
    assert output.count("Valid 2 samples, Skipped 0 samples") == 2
