from contextlib import contextmanager
from functools import partial

import pytest
import torch
from torch import nn

from torchcam import metrics
from torchcam.methods import FinerCAM, LayerCAM, RefineCAM


class _SumClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.identity = nn.Identity()
        self.calls = 0
        self.fail_on_call = None
        self.seen = []

    def forward(self, input_tensor):
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("scoring failed")
        self.seen.append(input_tensor.detach().clone())
        score = self.identity(input_tensor).flatten(1).sum(dim=1)
        return torch.stack((score, 1 - score), dim=1)


class _ConvClassifier(nn.Module):
    def __init__(self, spatial_dims=2):
        super().__init__()
        conv = nn.Conv2d if spatial_dims == 2 else nn.Conv3d
        self.conv1 = conv(1, 1, 1, bias=False)
        self.conv2 = conv(1, 1, 1, bias=False)
        with torch.no_grad():
            self.conv1.weight.fill_(1)
            self.conv2.weight.fill_(1)

    def forward(self, input_tensor):
        score = self.conv2(self.conv1(input_tensor)).flatten(1).sum(dim=1)
        return torch.stack((score, 1 - score), dim=1)


class _FixedExtractor:
    def __init__(self, model, cam):
        self.model = model
        self.cam = cam
        self._hooks_enabled = True
        self.class_idx = None

    def __call__(self, class_idx, scores=None, _normalized=True, **_kwargs):
        self.class_idx = class_idx
        cam = self.cam.to(scores)
        if cam.shape[0] == 1 and scores.shape[0] > 1:
            cam = cam.expand(scores.shape[0], *cam.shape[1:])
        return [cam.clone()]

    @staticmethod
    def fuse_cams(cams):
        return torch.stack(cams).max(dim=0).values

    @contextmanager
    def _hooks_off(self):
        previous = self._hooks_enabled
        self._hooks_enabled = False
        try:
            yield
        finally:
            self._hooks_enabled = previous


def _exact_input():
    return torch.tensor([[[[0.4, 0.3], [0.2, 0.1]]]])


def _fixed_metric(*, steps=2, batch_size=32, baseline=None):
    model = _SumClassifier()
    extractor = _FixedExtractor(model, _exact_input().squeeze(1))
    return metrics.DeletionInsertionMetric(
        extractor,
        steps=steps,
        batch_size=batch_size,
        baseline=baseline,
    )


def test_classification_metric():
    model = _ConvClassifier()
    model.train()
    model.conv1.eval()
    modes = [module.training for module in model.modules()]
    input_tensor = _exact_input().expand(2, -1, -1, -1).clone().requires_grad_(True)

    with LayerCAM(model, "conv2") as extractor:
        metric = metrics.ClassificationMetric(extractor, partial(torch.softmax, dim=-1))
        metric.update(input_tensor, class_idx=0)
        metric.update(input_tensor, class_idx=[0, 0])
        metric.update(input_tensor)
        assert extractor._hooks_enabled

    assert metric.summary().keys() == {"avg_drop", "conf_increase"}
    assert all(0 <= value <= 1 for value in metric.summary().values())
    assert metric.total == 6
    assert metric.nan_count == 0
    assert [module.training for module in model.modules()] == modes


def test_classification_metric_exact_value():
    model = _SumClassifier()
    extractor = _FixedExtractor(model, torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]))
    metric = metrics.ClassificationMetric(extractor)

    metric.update(_exact_input(), class_idx=0)

    assert metric.summary() == pytest.approx({"avg_drop": 0.6 / (1 + 1e-7), "conf_increase": 0})


def test_deletion_insertion_complete_curves_and_auc():
    metric = _fixed_metric(steps=2)
    input_tensor = _exact_input()
    original_scores = metric.cam_extractor.model(input_tensor)[:, 0]
    fractions, deletion, insertion = metric._compute_curves(
        input_tensor,
        torch.zeros_like(input_tensor),
        torch.arange(4).unsqueeze(0),
        torch.tensor([0]),
        original_scores,
    )

    torch.testing.assert_close(fractions, torch.tensor([0.0, 0.5, 1.0]))
    torch.testing.assert_close(deletion, torch.tensor([[1.0, 0.3, 0.0]]))
    torch.testing.assert_close(insertion, torch.tensor([[0.0, 0.7, 1.0]]))
    assert torch.trapezoid(deletion, fractions).item() == pytest.approx(0.4)
    assert torch.trapezoid(insertion, fractions).item() == pytest.approx(0.6)

    metric.update(input_tensor, class_idx=0)
    assert metric.summary() == pytest.approx({"deletion_auc": 0.4, "insertion_auc": 0.6})


def test_deletion_insertion_non_divisible_steps():
    metric = _fixed_metric(steps=3)
    input_tensor = torch.tensor([[[[0.3, 0.25, 0.2, 0.15, 0.1]]]])
    original_scores = metric.cam_extractor.model(input_tensor)[:, 0]
    fractions, deletion, insertion = metric._compute_curves(
        input_tensor,
        torch.zeros_like(input_tensor),
        torch.arange(5).unsqueeze(0),
        torch.tensor([0]),
        original_scores,
    )

    torch.testing.assert_close(fractions, torch.tensor([0.0, 0.4, 0.8, 1.0]))
    torch.testing.assert_close(deletion, torch.tensor([[1.0, 0.45, 0.1, 0.0]]))
    torch.testing.assert_close(insertion, torch.tensor([[0.0, 0.55, 0.9, 1.0]]))
    assert torch.trapezoid(deletion, fractions).item() == pytest.approx(0.41)
    assert torch.trapezoid(insertion, fractions).item() == pytest.approx(0.59)


def test_deletion_insertion_chunked_parity():
    input_tensor = _exact_input().expand(2, -1, -1, -1).clone()
    chunked = _fixed_metric(steps=4, batch_size=1)
    unchunked = _fixed_metric(steps=4, batch_size=100)

    chunked.update(input_tensor, class_idx=[0, 0])
    unchunked.update(input_tensor, class_idx=[0, 0])

    assert chunked.summary() == pytest.approx(unchunked.summary())


def test_deletion_insertion_custom_baselines_and_non_mutation():
    input_tensor = _exact_input()
    original_input = input_tensor.clone()
    baseline = torch.tensor(0.05)
    original_baseline = baseline.clone()
    calls = 0

    def baseline_fn(_tensor):
        nonlocal calls
        calls += 1
        return baseline

    tensor_metric = _fixed_metric(baseline=baseline)
    callable_metric = _fixed_metric(baseline=baseline_fn)
    tensor_metric.update(input_tensor, class_idx=0)
    callable_metric.update(input_tensor, class_idx=0)

    assert tensor_metric.summary() == pytest.approx(callable_metric.summary())
    assert calls == 1
    torch.testing.assert_close(input_tensor, original_input)
    torch.testing.assert_close(baseline, original_baseline)


@pytest.mark.parametrize(
    ("class_idx", "expected"),
    [
        (0, 0),
        ([0, 1], [0, 1]),
        (None, [0, 1]),
    ],
)
def test_deletion_insertion_class_indices(class_idx, expected):
    model = _SumClassifier()
    extractor = _FixedExtractor(model, _exact_input().squeeze(1))
    metric = metrics.DeletionInsertionMetric(extractor, steps=2)
    input_tensor = torch.cat((_exact_input(), _exact_input() / 5))

    metric.update(input_tensor, class_idx=class_idx)

    assert extractor.class_idx == expected
    assert set(metric.summary()) == {"deletion_auc", "insertion_auc"}


@pytest.mark.parametrize(
    ("class_idx", "error"),
    [
        (2, ValueError),
        ("0", TypeError),
        ([0], ValueError),
        ([0, 2], ValueError),
        ([False, 0], TypeError),
    ],
)
def test_metrics_reject_invalid_class_indices(class_idx, error):
    extractor = _FixedExtractor(_SumClassifier(), _exact_input().squeeze(1))
    metric = metrics.ClassificationMetric(extractor)

    with pytest.raises(error):
        metric.update(_exact_input().expand(2, -1, -1, -1), class_idx=class_idx)


def test_metrics_reject_unsupported_spatial_rank():
    extractor = _FixedExtractor(_SumClassifier(), _exact_input().squeeze(1))

    with pytest.raises(ValueError, match="1D"):
        metrics.ClassificationMetric(extractor).update(torch.ones((1, 1)), class_idx=0)


@pytest.mark.parametrize("metric_cls", [metrics.ClassificationMetric, metrics.DeletionInsertionMetric])
def test_metrics_skip_nan_cams(metric_cls):
    model = _SumClassifier()
    cam = torch.stack((_exact_input().squeeze(0).squeeze(0), torch.full((2, 2), float("nan"))))
    extractor = _FixedExtractor(model, cam)
    input_tensor = _exact_input().expand(2, -1, -1, -1).clone()
    metric = (
        metric_cls(extractor)
        if metric_cls is metrics.ClassificationMetric
        else metric_cls(extractor, steps=2, baseline=torch.zeros_like(input_tensor))
    )

    metric.update(input_tensor, class_idx=[0, 0])

    assert metric.total == 1
    assert metric.nan_count == 1


@pytest.mark.parametrize("metric_cls", [metrics.ClassificationMetric, metrics.DeletionInsertionMetric])
def test_metrics_all_nan_cams(metric_cls):
    model = _SumClassifier()
    extractor = _FixedExtractor(model, torch.full((1, 2, 2), float("nan")))
    metric = metric_cls(extractor) if metric_cls is metrics.ClassificationMetric else metric_cls(extractor, steps=2)

    metric.update(_exact_input())

    assert metric.nan_count == 1
    with pytest.raises(AssertionError, match="update"):
        metric.summary()


@pytest.mark.parametrize("metric_cls", [metrics.ClassificationMetric, metrics.DeletionInsertionMetric])
@pytest.mark.parametrize("spatial_dims", [2, 3])
def test_metrics_spatial_inputs(metric_cls, spatial_dims):
    model = _ConvClassifier(spatial_dims)
    shape = (1, 1, 2, 2) if spatial_dims == 2 else (1, 1, 2, 2, 2)
    input_tensor = torch.full(shape, 1 / (2**spatial_dims))

    with LayerCAM(model, "conv2") as extractor:
        metric = (
            metric_cls(extractor)
            if metric_cls is metrics.ClassificationMetric
            else metric_cls(extractor, steps=2, batch_size=2)
        )
        metric.update(input_tensor, class_idx=0)

    assert all(torch.isfinite(torch.tensor(list(metric.summary().values()))))


@pytest.mark.parametrize(
    "extractor_factory",
    [
        pytest.param(lambda model: LayerCAM(model, "conv2"), id="ordinary"),
        pytest.param(lambda model: LayerCAM(model, ["conv1", "conv2"]), id="multi-layer"),
        pytest.param(
            lambda model: RefineCAM(model, ["conv1", "conv2"], base_method=LayerCAM),
            id="refinecam-wrapper",
        ),
        pytest.param(
            lambda model: FinerCAM(model, "conv2", base_method=LayerCAM, num_references=1),
            id="finercam-wrapper",
        ),
    ],
)
def test_deletion_insertion_extractor_protocols(extractor_factory):
    model = _ConvClassifier()
    input_tensor = torch.full((1, 1, 2, 2), 0.25)

    with extractor_factory(model) as extractor:
        metric = metrics.DeletionInsertionMetric(extractor, partial(torch.softmax, dim=-1), steps=1)
        metric.update(input_tensor)
        hook_owner = getattr(extractor, "base_cam", extractor)
        hooks_enabled = hook_owner._hooks_enabled

    assert hooks_enabled
    assert set(metric.summary()) == {"deletion_auc", "insertion_auc"}


@pytest.mark.parametrize(
    ("device", "dtype"),
    [
        (torch.device("cpu"), torch.float64),
        pytest.param(
            torch.device("cuda"),
            torch.float64,
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
        ),
        pytest.param(
            torch.device("mps"),
            torch.float32,
            marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS"),
        ),
    ],
)
def test_deletion_insertion_preserves_dtype_and_device(device, dtype):
    model = _SumClassifier().to(device=device, dtype=dtype)
    extractor = _FixedExtractor(model, _exact_input().squeeze(1))
    metric = metrics.DeletionInsertionMetric(extractor, steps=2, batch_size=1)
    input_tensor = _exact_input().to(device=device, dtype=dtype)

    metric.update(input_tensor, class_idx=0)

    assert all(tensor.device.type == device.type and tensor.dtype == dtype for tensor in model.seen)


@pytest.mark.parametrize("metric_cls", [metrics.ClassificationMetric, metrics.DeletionInsertionMetric])
def test_metrics_restore_model_and_hooks_on_scoring_error(metric_cls):
    model = _SumClassifier()
    model.train()
    model.identity.eval()
    modes = [module.training for module in model.modules()]
    model.fail_on_call = 2
    extractor = _FixedExtractor(model, _exact_input().squeeze(1))
    metric = metric_cls(extractor) if metric_cls is metrics.ClassificationMetric else metric_cls(extractor, steps=2)

    with pytest.raises(RuntimeError, match="scoring failed"):
        metric.update(_exact_input(), class_idx=0)

    assert [module.training for module in model.modules()] == modes
    assert extractor._hooks_enabled
    assert metric.total == 0


def test_deletion_insertion_reset():
    metric = _fixed_metric()
    metric.update(_exact_input(), class_idx=0)

    metric.reset()

    assert metric.deletion == metric.insertion == 0
    assert metric.total == metric.nan_count == 0
    with pytest.raises(AssertionError, match="update"):
        metric.summary()


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"steps": 0}, ValueError),
        ({"steps": 1.5}, TypeError),
        ({"batch_size": 0}, ValueError),
        ({"batch_size": 1.5}, TypeError),
        ({"baseline": 0}, TypeError),
    ],
)
def test_deletion_insertion_rejects_invalid_configuration(kwargs, error):
    with pytest.raises(error):
        metrics.DeletionInsertionMetric(_FixedExtractor(_SumClassifier(), _exact_input().squeeze(1)), **kwargs)


@pytest.mark.parametrize(
    ("baseline", "error"),
    [
        (lambda _input: 0, TypeError),
        (torch.zeros((3, 3)), ValueError),
    ],
)
def test_deletion_insertion_rejects_invalid_baseline_result(baseline, error):
    metric = _fixed_metric(baseline=baseline)

    with pytest.raises(error):
        metric.update(_exact_input(), class_idx=0)
