from copy import deepcopy
from functools import partial
from operator import itemgetter

import pytest
import torch
from torch import nn
from torchvision.models import get_model

from torchcam.methods import activation, gradient
from torchcam.metrics import ClassificationMetric


def _verify_cam(activation_map, output_size):
    # Simple verifications
    assert isinstance(activation_map, torch.Tensor)
    assert activation_map.shape == output_size
    assert not torch.isnan(activation_map).any()


def _tiny_model(num_classes=3):
    return nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(4, num_classes),
    ).eval()


class _StructuredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(1, 2, 1)

    def forward(self, input_tensor):
        scores = self.features(input_tensor).flatten(2).mean(-1)
        return [{"primary": row[0], "secondary": row[1]} for row in scores]


def _contrastive_scores(scores, class_idx, comparison_idx, gamma):
    batch_size = scores.shape[0]
    targets = [class_idx] * batch_size if isinstance(class_idx, int) else class_idx
    if isinstance(comparison_idx, int):
        references = [[comparison_idx]] * batch_size
    else:
        references = [comparison_idx] * batch_size if isinstance(comparison_idx[0], int) else comparison_idx
    target_indices = torch.tensor(targets, device=scores.device).view(-1, 1)
    reference_indices = torch.tensor(references, device=scores.device)
    return scores.gather(1, target_indices) - gamma * scores.gather(1, reference_indices).mean(1, keepdim=True)


@pytest.mark.parametrize(
    ("cam_name", "target_layer", "output_size", "batch_size"),
    [
        ("GradCAM", "features.18.0", (7, 7), 1),
        ("GradCAMpp", "features.18.0", (7, 7), 1),
        ("SmoothGradCAMpp", lambda m: m.features[18][0], (7, 7), 1),
        ("SmoothGradCAMpp", "features.18.0", (7, 7), 1),
        ("XGradCAM", "features.18.0", (7, 7), 1),
        ("LayerCAM", "features.18.0", (7, 7), 1),
        ("LayerCAM", "features.18.0", (7, 7), 2),
    ],
)
def test_img_cams(cam_name, target_layer, output_size, batch_size, mock_img_tensor):
    model = get_model("mobilenet_v2", weights=None).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    target_layer = target_layer(model) if callable(target_layer) else target_layer
    # Hook the corresponding layer in the model
    with gradient.__dict__[cam_name](model, target_layer) as extractor:
        scores = model(mock_img_tensor.repeat((batch_size,) + (1,) * (mock_img_tensor.ndim - 1)))
        # Use the hooked data to compute activation map
        _verify_cam(
            extractor(scores[0].argmax().item(), scores, retain_graph=True)[0],
            (batch_size, *output_size),
        )
        # Multiple class indices
        _verify_cam(extractor(list(range(batch_size)), scores)[0], (batch_size, *output_size))

    # Inplace model
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(8, 10),
    )
    for p in model.parameters():
        p.requires_grad_(False)

    # Hook before the inplace ops
    with gradient.__dict__[cam_name](model, "2") as extractor:
        scores = model(mock_img_tensor)
        # Use the hooked data to compute activation map
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], (1, 224, 224))


@pytest.mark.parametrize(
    ("cam_name", "target_layer", "output_size"),
    [
        ("GradCAM", "0.3", (1, 8, 16, 16)),
        ("GradCAMpp", "0.3", (1, 8, 16, 16)),
        ("SmoothGradCAMpp", "0.3", (1, 8, 16, 16)),
        ("XGradCAM", "0.3", (1, 8, 16, 16)),
        ("LayerCAM", "0.3", (1, 8, 16, 16)),
    ],
)
def test_video_cams(cam_name, target_layer, output_size, mock_video_model, mock_video_tensor):
    model = mock_video_model.eval()
    # Hook the corresponding layer in the model
    with gradient.__dict__[cam_name](model, target_layer) as extractor:
        scores = model(mock_video_tensor)
        # Use the hooked data to compute activation map
        _verify_cam(extractor(scores[0].argmax().item(), scores)[0], output_size)


def test_smoothgradcampp_repr():
    model = get_model("mobilenet_v2", weights=None).eval()

    # Hook the corresponding layer in the model
    with gradient.SmoothGradCAMpp(model, "features.18.0") as extractor:
        assert repr(extractor) == "SmoothGradCAMpp(target_layer=['features.18.0'], num_samples=4, std=0.3)"


def test_gradcam_supports_per_sample_output_targets_without_touching_parameter_grads():
    model = _StructuredModel().eval()
    input_tensor = torch.rand(2, 1, 4, 4)

    with gradient.GradCAM(model, "features") as extractor:
        output = model(input_tensor)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        cams = extractor(
            scores=output,
            targets=[itemgetter("primary"), itemgetter("secondary")],
        )

    _verify_cam(cams[0], (2, 4, 4))
    assert all(torch.equal(parameter.grad, torch.ones_like(parameter)) for parameter in model.parameters())


def test_smoothgradcampp_supports_output_targets():
    model = _StructuredModel().eval()

    with gradient.SmoothGradCAMpp(model, "features", num_samples=2) as extractor:
        model(torch.rand(2, 1, 4, 4))
        cams = extractor(targets=itemgetter("primary"))

    _verify_cam(cams[0], (2, 4, 4))


def test_gradcam_reports_disconnected_output_target():
    model = _StructuredModel().eval()

    with gradient.GradCAM(model, "features") as extractor:
        output = model(torch.rand(1, 1, 4, 4))

        def detached_target(sample):
            return sample["primary"].detach()

        with pytest.raises(RuntimeError, match="not connected to every target layer"):
            extractor(scores=output, targets=detached_target)


def test_output_target_must_return_a_scalar_tensor():
    model = _tiny_model()

    with gradient.GradCAM(model, "2") as extractor:
        scores = model(torch.rand(1, 3, 4, 4))
        with pytest.raises(ValueError, match="scalar tensor"):
            extractor(scores=scores, targets=itemgetter(slice(2)))


def test_refinecam_supports_output_targets():
    model = _tiny_model()

    with gradient.RefineCAM(model, ["0", "2"], base_method=gradient.LayerCAM) as extractor:
        scores = model(torch.rand(2, 3, 4, 4))
        cams = extractor(scores=scores, targets=itemgetter(1))

    _verify_cam(cams[0], (2, 4, 4))


def test_finercam_rejects_output_targets():
    model = _tiny_model()

    with gradient.FinerCAM(model, "2") as extractor:
        scores = model(torch.rand(1, 3, 4, 4))
        with pytest.raises(ValueError, match="does not support"):
            extractor(0, scores, targets=itemgetter(0))


def test_layercam_fuse_cams():
    with pytest.raises(TypeError):
        gradient.LayerCAM.fuse_cams(torch.zeros((3, 32, 32)))

    with pytest.raises(ValueError):
        gradient.LayerCAM.fuse_cams([])

    cams = [torch.rand((1, 32, 32)), torch.rand((1, 16, 16))]

    # Single CAM
    assert torch.equal(cams[0], gradient.LayerCAM.fuse_cams(cams[:1]))

    # Fusion
    cam = gradient.LayerCAM.fuse_cams(cams)
    assert isinstance(cam, torch.Tensor)
    assert cam.ndim == cams[0].ndim
    assert cam.shape == (1, 32, 32)

    # Specify target shape
    cam = gradient.LayerCAM.fuse_cams(cams, (16, 16))
    assert isinstance(cam, torch.Tensor)
    assert cam.ndim == cams[0].ndim
    assert cam.shape == (1, 16, 16)


def test_refinecam_fuse_cams():
    cams = [
        torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]),
        torch.tensor([[[3.0, 2.0], [1.0, 0.0]]]),
    ]
    originals = [cam.clone() for cam in cams]

    assert torch.allclose(
        gradient.RefineCAM.fuse_cams(cams, normalized=False),
        torch.tensor([[[0.0, 2 / 9], [2 / 9, 0.0]]]),
    )
    assert torch.allclose(
        gradient.RefineCAM.fuse_cams(cams),
        torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),
    )
    assert all(map(torch.equal, cams, originals))

    larger_cam = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    assert gradient.RefineCAM.fuse_cams([cams[0], larger_cam]).shape == (1, 4, 4)
    assert gradient.RefineCAM.fuse_cams([cams[0], larger_cam], (2, 2)).shape == (1, 2, 2)
    assert gradient.RefineCAM.fuse_cams(cams[:1]).shape == cams[0].shape

    volume = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
    assert gradient.RefineCAM.fuse_cams([volume, volume.flip(-1)]).shape == volume.shape

    with pytest.raises(TypeError):
        gradient.RefineCAM.fuse_cams(torch.zeros((1, 2, 2)))
    with pytest.raises(ValueError):
        gradient.RefineCAM.fuse_cams([])


@pytest.mark.parametrize("base_method", [gradient.GradCAMpp, gradient.LayerCAM, activation.ScoreCAM])
def test_refinecam_base_methods(base_method):
    model = _tiny_model()
    input_tensor = torch.rand((2, 3, 8, 8))

    with gradient.RefineCAM(model, ["0", "2"], base_method=base_method) as extractor:
        scores = model(input_tensor)
        cams = extractor([0, 1], scores)
        assert len(cams) == 1
        _verify_cam(cams[0], (2, 8, 8))
        assert extractor.model is model
        assert extractor.target_names == ["0", "2"]
        assert repr(extractor) == f"RefineCAM(base_method={base_method.__name__}, target_layer=['0', '2'])"

    assert extractor.base_cam.hook_handles == []


def test_refinecam_metric_compatibility():
    model = _tiny_model()
    input_tensor = torch.rand((1, 3, 8, 8))

    with gradient.RefineCAM(model, ["0", "2"]) as extractor:
        metric = ClassificationMetric(extractor, partial(torch.softmax, dim=-1))
        metric.update(input_tensor)
        summary = metric.summary()

    assert set(summary) == {"avg_drop", "conf_increase"}


def test_refinecam_rejects_invalid_configuration():
    model = _tiny_model()

    with pytest.raises(ValueError, match="at least two target layers"):
        gradient.RefineCAM(model, ["0"])
    with pytest.raises(TypeError, match="CAM extractor class"):
        gradient.RefineCAM(model, ["0", "2"], base_method=nn.Module)


@pytest.mark.parametrize("base_method", [gradient.GradCAM, gradient.GradCAMpp, gradient.LayerCAM])
@pytest.mark.parametrize(
    ("class_idx", "comparison_idx"),
    [(0, 2), ([0, 1], [2]), ([0, 1], [[1, 2], [0, 2]])],
)
@pytest.mark.parametrize("normalized", [False, True])
def test_finercam_matches_manual_contrastive_score(base_method, class_idx, comparison_idx, normalized):
    model = _tiny_model()
    manual_model = deepcopy(model)
    input_tensor = torch.rand((2, 3, 8, 8))
    gamma = 0.6

    with gradient.FinerCAM(model, "2", base_method=base_method, gamma=gamma) as extractor:
        scores = model(input_tensor)
        original_scores = scores.detach().clone()
        cams = extractor(class_idx, scores, comparison_idx, normalized)

    with base_method(manual_model, "2") as base_cam:
        manual_scores = manual_model(input_tensor)
        contrastive_scores = _contrastive_scores(manual_scores, class_idx, comparison_idx, gamma)
        expected = base_cam([0, 0], contrastive_scores, normalized)

    assert len(cams) == len(expected) == 1
    torch.testing.assert_close(cams[0], expected[0])
    assert torch.equal(scores.detach(), original_scores)
    assert scores.shape == (2, 3)


def test_finercam_automatic_references_cap_and_use_closest_scores(monkeypatch):
    model = _tiny_model(num_classes=3)
    input_tensor = torch.rand((2, 3, 8, 8))

    with gradient.FinerCAM(model, "2", num_references=10) as extractor:
        scores = model(input_tensor)
        class_idx = [1, 2]
        target_indices = torch.tensor(class_idx).view(-1, 1)
        distances = (scores.detach() - scores.detach().gather(1, target_indices)).abs()
        distances.scatter_(1, target_indices, float("inf"))
        references = distances.topk(2, dim=1, largest=False).indices
        expected = scores.gather(1, target_indices) - 0.6 * scores.gather(1, references).mean(1, keepdim=True)
        calls = []
        backprop = extractor.base_cam._backprop

        def capture_backprop(contrastive_scores, objective_idx, **kwargs):
            calls.append((contrastive_scores, objective_idx))
            return backprop(contrastive_scores, objective_idx, **kwargs)

        monkeypatch.setattr(extractor.base_cam, "_backprop", capture_backprop)
        extractor(class_idx, scores)

    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0], expected)
    assert calls[0][1] == [0, 0]


@pytest.mark.parametrize("num_classes", [2, 3])
def test_finercam_caps_automatic_references_for_small_outputs(num_classes):
    model = _tiny_model(num_classes)
    input_tensor = torch.rand((1, 3, 8, 8))

    with gradient.FinerCAM(model, "2") as extractor:
        scores = model(input_tensor)
        cams = extractor(0, scores, normalized=False)

    _verify_cam(cams[0], (1, 8, 8))


def test_finercam_gamma_zero_matches_base_cam():
    model = _tiny_model()
    manual_model = deepcopy(model)
    input_tensor = torch.rand((2, 3, 8, 8))
    class_idx = [0, 1]

    with gradient.FinerCAM(model, "2", gamma=0) as extractor:
        scores = model(input_tensor)
        cams = extractor(class_idx, scores, normalized=False)

    with gradient.GradCAM(manual_model, "2") as base_cam:
        scores = manual_model(input_tensor)
        expected = base_cam(class_idx, scores, normalized=False)

    torch.testing.assert_close(cams[0], expected[0])


def test_finercam_aggregates_before_relu():
    class ContrastiveModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Conv2d(1, 1, 1, bias=False)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Linear(1, 3, bias=False)
            with torch.no_grad():
                self.features.weight.fill_(1)
                self.classifier.weight.copy_(torch.tensor([[1.0], [3.0], [-1.0]]))

        def forward(self, input_tensor):
            return self.classifier(self.pool(self.features(input_tensor)).flatten(1))

    model = ContrastiveModel().eval()
    input_tensor = torch.ones((1, 1, 2, 2))

    with gradient.FinerCAM(model, "features", gamma=1) as extractor:
        scores = model(input_tensor)
        aggregated = extractor(0, scores, [1, 2], normalized=False)[0]

    pairwise_cams = []
    for reference in [1, 2]:
        pair_model = deepcopy(model)
        with gradient.GradCAM(pair_model, "features") as extractor:
            scores = pair_model(input_tensor)
            contrastive_scores = _contrastive_scores(scores, 0, [reference], 1)
            pairwise_cams.append(extractor(0, contrastive_scores, normalized=False)[0])

    assert torch.count_nonzero(aggregated) == 0
    assert torch.stack(pairwise_cams).mean(0).max() > 0


def test_finercam_multiple_layers_and_compute_cams():
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 4, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(1),
        nn.Linear(4, 3),
    ).eval()
    input_tensor = torch.rand((1, 3, 8, 8))

    with gradient.FinerCAM(model, ["0", "2"]) as extractor:
        scores = model(input_tensor)
        cams = extractor.compute_cams(0, scores, [1, 2], normalized=False)

    assert [cam.shape for cam in cams] == [(1, 8, 8), (1, 4, 4)]


def test_finercam_video_cam(mock_video_tensor):
    model = nn.Sequential(
        nn.Conv3d(3, 4, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool3d((1, 1, 1)),
        nn.Flatten(1),
        nn.Linear(4, 3),
    ).eval()

    with gradient.FinerCAM(model, "0", base_method=gradient.LayerCAM) as extractor:
        scores = model(mock_video_tensor)
        cams = extractor(0, scores, comparison_idx=[1, 2])

    _verify_cam(cams[0], (1, 8, 16, 16))


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(torch.device("cuda"), marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
    ],
)
def test_finercam_preserves_dtype_and_device(device):
    model = _tiny_model().to(device=device, dtype=torch.float64)
    input_tensor = torch.rand((1, 3, 8, 8), device=device, dtype=torch.float64)

    with gradient.FinerCAM(model, "2") as extractor:
        scores = model(input_tensor)
        cams = extractor(0, scores, comparison_idx=[1, 2], normalized=False)

    assert cams[0].dtype == scores.dtype == torch.float64
    assert cams[0].device == scores.device == device


def test_finercam_lifecycle_repr_and_metric_compatibility():
    model = _tiny_model()
    input_tensor = torch.rand((1, 3, 8, 8))
    target_layer = model[2]
    assert len(target_layer._forward_hooks) == 0

    with gradient.FinerCAM(model, "2") as extractor:
        assert len(target_layer._forward_hooks) == 2
        assert extractor.model is model
        assert extractor.target_names == ["2"]
        assert repr(extractor) == ("FinerCAM(base_method=GradCAM, target_layer=['2'], gamma=0.6, num_references=3)")
        metric = ClassificationMetric(extractor, partial(torch.softmax, dim=-1))
        metric.update(input_tensor)
        assert set(metric.summary()) == {"avg_drop", "conf_increase"}

    assert len(target_layer._forward_hooks) == 0
    assert extractor.base_cam.hook_handles == []


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"base_method": activation.ScoreCAM}, TypeError, "GradCAM, GradCAMpp, or LayerCAM"),
        ({"base_method": gradient.SmoothGradCAMpp}, TypeError, "GradCAM, GradCAMpp, or LayerCAM"),
        ({"base_method": nn.Identity()}, TypeError, "GradCAM, GradCAMpp, or LayerCAM"),
        ({"gamma": True}, TypeError, "real number"),
        ({"gamma": "0.6"}, TypeError, "real number"),
        ({"gamma": -0.1}, ValueError, "finite and non-negative"),
        ({"gamma": float("inf")}, ValueError, "finite and non-negative"),
        ({"gamma": float("nan")}, ValueError, "finite and non-negative"),
        ({"num_references": True}, TypeError, "integer"),
        ({"num_references": 1.5}, TypeError, "integer"),
        ({"num_references": 0}, ValueError, "positive"),
    ],
)
def test_finercam_rejects_invalid_configuration(kwargs, error, match):
    with pytest.raises(error, match=match):
        gradient.FinerCAM(_tiny_model(), "2", **kwargs)


def test_finercam_rejects_invalid_scores_targets_and_references():
    model = _tiny_model()
    input_tensor = torch.rand((2, 3, 8, 8))

    with gradient.FinerCAM(model, "2") as extractor:
        with pytest.raises(ValueError, match="scores is required"):
            extractor(0)
        with pytest.raises(TypeError, match="scores must be a tensor"):
            extractor(0, [[1.0, 2.0]])
        with pytest.raises(ValueError, match="shape"):
            extractor(0, torch.rand(3))
        with pytest.raises(ValueError, match="comparison class"):
            extractor(0, torch.rand((2, 1)))

        scores = model(input_tensor)
        with pytest.raises(TypeError, match="class_idx"):
            extractor(True, scores)
        with pytest.raises(TypeError, match="contain integers"):
            extractor([0, True], scores)
        with pytest.raises(ValueError, match="batch size"):
            extractor([0], scores)
        with pytest.raises(ValueError, match="out-of-range"):
            extractor([0, 3], scores)
        with pytest.raises(ValueError, match="cannot be empty"):
            extractor([0, 1], scores, [])
        with pytest.raises(TypeError, match="integer, a list, or a nested list"):
            extractor([0, 1], scores, (2,))
        with pytest.raises(ValueError, match="target class"):
            extractor([0, 1], scores, [0, 2])
        with pytest.raises(ValueError, match="duplicates"):
            extractor([0, 1], scores, [2, 2])
        with pytest.raises(ValueError, match="out-of-range"):
            extractor([0, 1], scores, [2, 3])
        with pytest.raises(TypeError, match="contain integers"):
            extractor([0, 1], scores, [True])
        with pytest.raises(ValueError, match="batch size"):
            extractor([0, 1], scores, [[1, 2]])
        with pytest.raises(ValueError, match="rows cannot be empty"):
            extractor([0, 1], scores, [[], []])
        with pytest.raises(ValueError, match="equal lengths"):
            extractor([0, 1], scores, [[1], [0, 2]])
        with pytest.raises(TypeError, match="contain integers"):
            extractor([0, 1], scores, [[True], [0]])
        with pytest.raises(TypeError, match="integers or lists"):
            extractor([0, 1], scores, [2, [0]])


def test_refinecam_shared_wrapper_regression():
    model = _tiny_model()
    input_tensor = torch.rand((1, 3, 8, 8))
    extractor = gradient.RefineCAM(model, ["0", "2"])

    with extractor as entered:
        assert entered is extractor
        extractor.disable_hooks()
        assert not extractor.base_cam._hooks_enabled
        extractor.enable_hooks()
        scores = model(input_tensor)
        _verify_cam(extractor(0, scores)[0], (1, 8, 8))
        assert repr(extractor) == "RefineCAM(base_method=GradCAMpp, target_layer=['0', '2'])"

    assert extractor.base_cam.hook_handles == []


def test_gradcam_does_not_accumulate_hook_handles(mock_img_tensor):
    model = get_model("mobilenet_v2", weights=None).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with gradient.GradCAM(model, "features.18.0") as extractor:
        initial_handles = len(extractor.hook_handles)
        for _ in range(3):
            scores = model(mock_img_tensor)
            extractor(scores[0].argmax().item(), scores, retain_graph=True)
        assert len(extractor.hook_handles) == initial_handles
        assert len(extractor._hook_outputs) == len(extractor.target_names)


def test_smoothgradcampp_restores_input_hook_on_error(mock_img_tensor, monkeypatch):
    model = get_model("mobilenet_v2", weights=None).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    with gradient.SmoothGradCAMpp(model, "features.18.0", num_samples=1) as extractor:
        scores = model(mock_img_tensor)
        monkeypatch.setattr(
            extractor,
            "_backprop",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
        )

        with pytest.raises(RuntimeError):
            extractor(scores[0].argmax().item(), scores)

        assert extractor._ihook_enabled
