import pytest
import torch
from torchvision.models.vision_transformer import VisionTransformer

from torchcam.methods import LayerCAM, ScoreCAM


def _build_vit() -> VisionTransformer:
    torch.manual_seed(0)
    model = VisionTransformer(
        image_size=32,
        patch_size=8,
        num_layers=2,
        num_heads=2,
        hidden_dim=32,
        mlp_dim=64,
        num_classes=10,
    )
    torch.nn.init.normal_(model.heads.head.weight)
    return model.eval()


def _reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    return tensor[:, 1:].reshape(tensor.shape[0], 4, 4, tensor.shape[-1]).permute(0, 3, 1, 2)


def test_reshape_transform_validation():
    model = _build_vit()
    target_layer = "encoder.layers.encoder_layer_1.ln_1"

    with pytest.raises(TypeError, match="reshape_transform"):
        LayerCAM(model, target_layer, reshape_transform=1)
    with pytest.raises(ValueError, match="target_layer"):
        LayerCAM(model, reshape_transform=_reshape_transform)


def test_layercam_with_reshape_transform():
    model = _build_vit()
    input_tensor = torch.randn(1, 3, 32, 32)

    with LayerCAM(
        model,
        "encoder.layers.encoder_layer_1.ln_1",
        reshape_transform=_reshape_transform,
    ) as extractor:
        scores = model(input_tensor)
        cam = extractor(scores[0].argmax().item(), scores)[0]

        assert extractor.hook_a[0].shape == (1, 32, 4, 4)
        assert extractor.hook_g[0].shape == (1, 32, 4, 4)
        assert cam.shape == (1, 4, 4)
        assert torch.isfinite(cam).all()
        assert cam.std() > 0


def test_scorecam_with_reshape_transform():
    model = _build_vit()
    input_tensor = torch.randn(1, 3, 32, 32)

    with ScoreCAM(
        model,
        "encoder.layers.encoder_layer_1.ln_1",
        batch_size=16,
        reshape_transform=_reshape_transform,
    ) as extractor:
        scores = model(input_tensor)
        cam = extractor(scores[0].argmax().item())[0]

        assert extractor.hook_a[0].shape == (1, 32, 4, 4)
        assert cam.shape == (1, 4, 4)
        assert torch.isfinite(cam).all()
        assert cam.std() > 0
