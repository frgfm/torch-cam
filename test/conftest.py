import pytest
import requests
from io import BytesIO
from PIL import Image
import torch
from torch import nn
from torchvision.transforms.functional import normalize, resize, to_tensor


@pytest.fixture(scope="session")
def mock_img_tensor():
    try:
        # Get a dog image
        URL = 'https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg'
        response = requests.get(URL)

        # Forward an image
        pil_img = Image.open(BytesIO(response.content), mode='r').convert('RGB')
        img_tensor = normalize(to_tensor(resize(pil_img, (224, 224))),
                               [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)
    except ConnectionError:
        img_tensor = torch.rand((1, 3, 224, 224))

    return img_tensor


@pytest.fixture(scope="session")
def mock_video_tensor():
    return torch.rand((1, 3, 8, 16, 16))


@pytest.fixture(scope="session")
def mock_video_model():
    return nn.Sequential(
        nn.Sequential(
            nn.Conv3d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        ),
        nn.Flatten(1),
        nn.Linear(16, 1)
    )


@pytest.fixture(scope="session")
def mock_img_model():
    return nn.Sequential(
        nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        ),
        nn.Flatten(1),
        nn.Linear(16, 1)
    )