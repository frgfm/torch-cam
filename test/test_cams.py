import unittest
from io import BytesIO

import requests
import torch
from PIL import Image
from torch import nn
from torchvision.models import mobilenet_v2, resnet18
from torchvision.transforms.functional import normalize, resize, to_tensor

from torchcam import cams


class CAMCoreTester(unittest.TestCase):
    def _verify_cam(self, cam):
        # Simple verifications
        self.assertIsInstance(cam, torch.Tensor)
        self.assertEqual(cam.shape, (7, 7))
        self.assertFalse(torch.any(torch.isnan(cam)))

    @staticmethod
    def _get_img_tensor():

        # Get a dog image
        URL = 'https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg'
        response = requests.get(URL)

        # Forward an image
        pil_img = Image.open(BytesIO(response.content), mode='r').convert('RGB')
        img_tensor = normalize(to_tensor(resize(pil_img, (224, 224))),
                               [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return img_tensor

    def _test_extractor(self, extractor, model):

        # Check missing forward raises Error
        self.assertRaises(AssertionError, extractor, 0)

        # Get a dog image
        img_tensor = self._get_img_tensor()

        # Check that a batch of 2 cannot be accepted
        _ = model(torch.stack((img_tensor, img_tensor)))
        self.assertRaises(ValueError, extractor, 0)

        # Correct forward
        scores = model(img_tensor.unsqueeze(0))

        # Check incorrect class index
        self.assertRaises(ValueError, extractor, -1)

        # Check missing score
        if extractor._score_used:
            self.assertRaises(ValueError, extractor, 0)

        # Use the hooked data to compute activation map
        self._verify_cam(extractor(scores[0].argmax().item(), scores))

    def _test_cam(self, name):
        # Get a pretrained model
        model = mobilenet_v2(pretrained=False).eval()
        conv_layer = None if name == "CAM" else 'features.16.conv.3'

        kwargs = {}
        # Speed up testing by reducing the number of samples
        if name in ['SSCAM', 'ISCAM']:
            kwargs['num_samples'] = 4
        # Hook the corresponding layer in the model
        extractor = cams.__dict__[name](model, conv_layer, **kwargs)

        with torch.no_grad():
            self._test_extractor(extractor, model)

    def _test_gradcam(self, name):

        # Get a pretrained model
        model = mobilenet_v2(pretrained=False).eval()
        conv_layer = 'features.18.0'

        # Hook the corresponding layer in the model
        extractor = cams.__dict__[name](model, conv_layer)

        self._test_extractor(extractor, model)

    def test_smooth_gradcampp(self):

        # Get a pretrained model
        model = mobilenet_v2(pretrained=False).eval()

        # Hook the corresponding layer in the model
        extractor = cams.SmoothGradCAMpp(model)

        self._test_extractor(extractor, model)


class CAMUtilsTester(unittest.TestCase):

    @staticmethod
    def _get_custom_module():

        mod = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(3, 8, 3, 1),
                nn.ReLU(),
                nn.Conv2d(8, 16, 3, 1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1))
            ),
            nn.Flatten(1),
            nn.Linear(16, 1)
        )
        return mod

    def test_locate_candidate_layer(self):

        # ResNet-18
        mod = resnet18().eval()
        self.assertEqual(cams.utils.locate_candidate_layer(mod), 'layer4')

        # Custom model
        mod = self._get_custom_module()

        self.assertEqual(cams.utils.locate_candidate_layer(mod), '0.3')
        # Check that the model is switched back to its origin mode afterwards
        self.assertTrue(mod.training)

    def test_locate_linear_layer(self):

        # ResNet-18
        mod = resnet18().eval()
        self.assertEqual(cams.utils.locate_linear_layer(mod), 'fc')

        # Custom model
        mod = self._get_custom_module()
        self.assertEqual(cams.utils.locate_linear_layer(mod), '2')


for cam_extractor in ['CAM', 'ScoreCAM', 'SSCAM', 'ISCAM']:
    def do_test(self, cam_extractor=cam_extractor):
        self._test_cam(cam_extractor)

    setattr(CAMCoreTester, "test_" + cam_extractor.lower(), do_test)


for cam_extractor in ['GradCAM', 'GradCAMpp']:
    def do_test(self, cam_extractor=cam_extractor):
        self._test_gradcam(cam_extractor)

    setattr(CAMCoreTester, "test_" + cam_extractor.lower(), do_test)


if __name__ == '__main__':
    unittest.main()
