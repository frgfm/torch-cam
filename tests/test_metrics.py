from functools import partial

import torch
from torchvision.models import mobilenet_v3_small

from torchcam import metrics
from torchcam.methods import LayerCAM


def test_classification_metric():
    model = mobilenet_v3_small(weights=None)
    with LayerCAM(model, "features.12") as extractor:
        metric = metrics.ClassificationMetric(extractor, partial(torch.softmax, dim=-1))

        # Fixed class
        metric.update(torch.rand((2, 3, 224, 224), dtype=torch.float32), class_idx=0)
        # Top predicted class
        metric.update(torch.rand((2, 3, 224, 224), dtype=torch.float32))
    out = metric.summary()

    assert len(out) == 2
    assert all(0 <= v <= 1 for k, v in out.items())
