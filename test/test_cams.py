import unittest
import requests
from io import BytesIO
from PIL import Image
import torch
from torchvision.models import resnet18, mobilenet_v2
from torchvision.transforms.functional import resize, to_tensor, normalize

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
        # Border collie index in ImageNet
        class_idx = 232

        # Hook the corresponding layer in the model
        extractor = cams.CAM(model, conv_layer, fc_layer)

        # Get a dog image
        img_tensor = self._get_img_tensor()
        # Forward it
        with torch.no_grad():
            _ = model(img_tensor.unsqueeze(0))

        # Use the hooked data to compute activation map
        self._verify_cam(extractor(class_idx))

    def _test_gradcam(self, name):

        # Get a pretrained model
        model = mobilenet_v2(pretrained=True)
        conv_layer = 'features'
        # Border collie index in ImageNet
        class_idx = 232

        # Hook the corresponding layer in the model
        extractor = cams.__dict__[name](model, conv_layer)

        # Get a dog image
        img_tensor = self._get_img_tensor()

        # Forward an image
        out = model(img_tensor.unsqueeze(0))

        # Use the hooked data to compute activation map
        self._verify_cam(extractor(out, class_idx))


for cam_extractor in ['GradCAM', 'GradCAMpp']:
    def do_test(self, cam_extractor=cam_extractor):
        self._test_gradcam(cam_extractor)

    setattr(Tester, "test_" + cam_extractor.lower(), do_test)


if __name__ == '__main__':
    unittest.main()
