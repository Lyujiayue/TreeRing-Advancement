import unittest

import numpy as np
from PIL import Image

from significance_detection import get_saliency_weight_from_tensor


class SaliencyDetectionTests(unittest.TestCase):
    def test_pil_batch_returns_normalized_float32_maps(self):
        image_a = np.zeros((32, 32, 3), dtype=np.uint8)
        image_a[8:24, 8:24] = 255
        image_b = np.zeros((32, 32, 3), dtype=np.uint8)
        image_b[:, 16:] = 180
        maps = get_saliency_weight_from_tensor([
            Image.fromarray(image_a), Image.fromarray(image_b)
        ])
        self.assertEqual(maps.shape, (2, 1, 32, 32))
        self.assertEqual(maps.dtype, np.float32)
        self.assertGreaterEqual(float(maps.min()), 0.0)
        self.assertLessEqual(float(maps.max()), 1.0)
        for saliency_map in maps:
            if float(saliency_map.max()) > 0:
                self.assertAlmostEqual(float(saliency_map.max()), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
