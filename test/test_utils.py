import unittest
from PIL import Image
import numpy as np

from torcham import utils


class Tester(unittest.TestCase):

    def test_overlay_mask(self):

        img = Image.fromarray(np.zeros((4, 4, 3)).astype(np.uint8))
        mask = Image.fromarray(np.ones((4, 4)).astype(np.uint8))

        overlayed = utils.overlay_mask(img, mask, alpha=0.7)

        # Check object type
        self.assertIsInstance(overlayed, Image)
        # Verify value
        self.assertTrue(np.all(np.asarray(overlayed) == 0.3))


if __name__ == '__main__':
    unittest.main()
