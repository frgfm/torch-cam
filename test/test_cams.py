import unittest
from io import BytesIO

import requests
import torch
from PIL import Image
from torchvision.models import mobilenet_v2, resnet18
from torchvision.transforms.functional import normalize, resize, to_tensor

from torchcam import cams


class Tester(unittest.TestCase):
    def _verify_cam(self, cam):
        # Simple verifications
        self.assertIsInstance(cam, torch.Tensor)
        self.assertEqual(cam.shape, (1, 7, 7))

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

    def test_cam(self):
        # Get a pretrained model
        model = resnet18(pretrained=True).eval()
        conv_layer = 'layer4'
        fc_layer = 'fc'

        # Hook the corresponding layer in the model
        extractor = cams.CAM(model, conv_layer, fc_layer)

        # Get a dog image
        img_tensor = self._get_img_tensor()
        # Forward it
        with torch.no_grad():
            out = model(img_tensor.unsqueeze(0))

        # Use the hooked data to compute activation map
        self._verify_cam(extractor(out[0].argmax().item()))

    def _test_gradcam(self, name):

        # Get a pretrained model
        model = mobilenet_v2(pretrained=True)
        conv_layer = 'features'

        # Hook the corresponding layer in the model
        extractor = cams.__dict__[name](model, conv_layer)

        # Get a dog image
        img_tensor = self._get_img_tensor()

        # Forward an image
        out = model(img_tensor.unsqueeze(0))

        # Use the hooked data to compute activation map
        self._verify_cam(extractor(out, out[0].argmax().item()))

    def test_smooth_gradcampp(self):

        # Get a pretrained model
        model = mobilenet_v2(pretrained=True)
        conv_layer = 'features'
        first_layer = 'features'

        # Hook the corresponding layer in the model
        extractor = cams.SmoothGradCAMpp(model, conv_layer, first_layer)

        # Get a dog image
        img_tensor = self._get_img_tensor()

        # Forward an image
        out = model(img_tensor.unsqueeze(0))

        # Use the hooked data to compute activation map
        self._verify_cam(extractor(out, out[0].argmax().item()))


for cam_extractor in ['GradCAM', 'GradCAMpp']:
    def do_test(self, cam_extractor=cam_extractor):
        self._test_gradcam(cam_extractor)

    setattr(Tester, "test_" + cam_extractor.lower(), do_test)


if __name__ == '__main__':
    unittest.main()
