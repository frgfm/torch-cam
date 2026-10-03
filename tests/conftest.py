import pytest
import torch
from torch import nn
from torchvision.transforms.functional import normalize


@pytest.fixture(scope="session")
def mock_img_tensor():
    generator = torch.Generator().manual_seed(0)
    return normalize(
        torch.rand((1, 3, 224, 224), generator=generator),
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ).requires_grad_(True)


@pytest.fixture(scope="session")
def mock_video_tensor():
    return torch.rand((1, 3, 8, 16, 16), requires_grad=True)


def _spatial_classifier(conv, pool):
    return nn.Sequential(
        nn.Sequential(
            conv(3, 8, 3, padding=1),
            nn.ReLU(),
            conv(8, 16, 3, padding=1),
            nn.ReLU(),
            pool(1),
        ),
        nn.Flatten(1),
        nn.Linear(16, 1),
    )


@pytest.fixture(scope="session")
def mock_video_model():
    return _spatial_classifier(nn.Conv3d, nn.AdaptiveAvgPool3d).requires_grad_(False)


@pytest.fixture(scope="session")
def mock_img_model():
    return _spatial_classifier(nn.Conv2d, nn.AdaptiveAvgPool2d).requires_grad_(False)


@pytest.fixture(scope="session")
def mock_fullyconv_model():
    model = nn.Sequential(
        nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        ),
        nn.Conv2d(16, 1, 1),
        nn.Flatten(1),
    )
    model.requires_grad_(False)
    return model
