import unittest
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision.models import mobilenet_v2, resnet18
from torchvision.transforms.functional import normalize, resize, to_tensor

from torchcam import cams


def _forward(model, input_tensor):
    if model.training:
        scores = model(input_tensor)
    else:
        with torch.no_grad():
            scores = model(input_tensor)

    return scores


class Tester(unittest.TestCase):
    def _verify_cam(self, cam):
        # Simple verifications
        self.assertIsInstance(cam, torch.Tensor)
        self.assertEqual(cam.shape, (7, 7))

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
        _ = _forward(model, torch.stack((img_tensor, img_tensor)))
        self.assertRaises(ValueError, extractor, 0)

        # Correct forward
        scores = _forward(model, img_tensor.unsqueeze(0))

        # Check incorrect class index
        self.assertRaises(ValueError, extractor, -1)

        # Check missing score
        if extractor._score_used:
            self.assertRaises(ValueError, extractor, 0)

        # Use the hooked data to compute activation map
        self._verify_cam(extractor(scores[0].argmax().item(), scores))

    def _test_cam(self, name):
        # Get a pretrained model
        model = resnet18(pretrained=True).eval()
        conv_layer = 'layer4'
        input_layer = 'conv1'
        fc_layer = 'fc'

        # Hook the corresponding layer in the model
        extractor = cams.__dict__[name](model, conv_layer, fc_layer if name == 'CAM' else input_layer)

        self._test_extractor(extractor, model)

    def _test_gradcam(self, name):

        # Get a pretrained model
        model = mobilenet_v2(pretrained=True)
        conv_layer = 'features'

        # Hook the corresponding layer in the model
        extractor = cams.__dict__[name](model, conv_layer)

        self._test_extractor(extractor, model)

    def test_smooth_gradcampp(self):

        # Get a pretrained model
        model = mobilenet_v2(pretrained=True)
        conv_layer = 'features'
        input_layer = 'features'

        # Hook the corresponding layer in the model
        extractor = cams.SmoothGradCAMpp(model, conv_layer, input_layer)

        self._test_extractor(extractor, model)


for cam_extractor in ['CAM', 'ScoreCAM', 'SSCAM', 'ISSCAM']:
    def do_test(self, cam_extractor=cam_extractor):
        self._test_cam(cam_extractor)

    setattr(Tester, "test_" + cam_extractor.lower(), do_test)


for cam_extractor in ['GradCAM', 'GradCAMpp']:
    def do_test(self, cam_extractor=cam_extractor):
        self._test_gradcam(cam_extractor)

    setattr(Tester, "test_" + cam_extractor.lower(), do_test)


if __name__ == '__main__':
    unittest.main()
