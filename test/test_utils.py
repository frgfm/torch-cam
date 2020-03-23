import unittest
from PIL import Image
import numpy as np

from torchcam import utils


class Tester(unittest.TestCase):

    def test_overlay_mask(self):

        img = Image.fromarray(np.zeros((4, 4, 3)).astype(np.uint8))
        mask = Image.fromarray(255 * np.ones((4, 4)).astype(np.uint8))

        overlayed = utils.overlay_mask(img, mask, alpha=0.7)

        # Check object type
        self.assertIsInstance(overlayed, Image.Image)
        # Verify value
        self.assertTrue(np.all(np.asarray(overlayed)[..., 0] == 0))
        self.assertTrue(np.all(np.asarray(overlayed)[..., 1] == 39))
        self.assertTrue(np.all(np.asarray(overlayed)[..., 2] == 76))


if __name__ == '__main__':
    unittest.main()
