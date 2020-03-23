import unittest
import requests
from io import BytesIO
from PIL import Image
import torch
from torchvision.models import mobilenet_v2
from torchvision.transforms import transforms

from torchcam import gradcam


class Tester(unittest.TestCase):

    def _test_gradcam(self, name):

        # Get a pretrained model
        model = mobilenet_v2(pretrained=True)
        conv_layer = 'features'

        # Hook the corresponding layer in the model
        extractor = gradcam.__dict__[name](model, conv_layer)

        # Get a dog image
        URL = 'https://www.woopets.fr/assets/races/000/066/big-portrait/border-collie.jpg'
        response = requests.get(URL)

        # Forward an image
        pil_img = Image.open(BytesIO(response.content), mode='r').convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img_tensor = preprocess(pil_img)
        out = model(img_tensor.unsqueeze(0))

        # Border collie index in ImageNet
        class_idx = 232

        # Use the hooked data to compute activation map
        activation_map = extractor.get_activation_maps(out, class_idx)

        self.assertIsInstance(activation_map, torch.Tensor)
        self.assertEqual(activation_map.shape, (1, 7, 7))


for cam_extractor in ['GradCAM', 'GradCAM++']:
    def do_test(self, cam_extractor=cam_extractor):
        self._test_gradcam(cam_extractor)

    setattr(Tester, "test_" + cam_extractor.lower(), do_test)


if __name__ == '__main__':
    unittest.main()
