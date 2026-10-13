import tempfile
import unittest
from pathlib import Path

from PIL import Image

from image_artifacts import save_image_pair, verify_image_pair


class ImageArtifactTests(unittest.TestCase):
    def test_round_trip_is_lossless_and_does_not_mutate_original(self):
        image = Image.new('RGB', (8, 8), (10, 20, 30))
        original = image.tobytes()
        with tempfile.TemporaryDirectory() as directory:
            artifacts = save_image_pair(directory, 3, image, image)
            hashes = verify_image_pair(directory, 3, artifacts, expected_size=8)
            self.assertEqual(len(hashes), 2)
            self.assertEqual(artifacts['no_watermark']['pixel_sha256'], artifacts['watermarked']['pixel_sha256'])
            self.assertEqual(image.tobytes(), original)
            with self.assertRaises(FileExistsError):
                save_image_pair(directory, 3, image, image)
            self.assertEqual(hashes, verify_image_pair(directory, 3, artifacts, expected_size=8))

    def test_tamper_wrong_mapping_and_mode_are_rejected(self):
        for corruption in ('file', 'pixels', 'filename', 'size'):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as directory:
                image = Image.new('RGB', (8, 8))
                artifacts = save_image_pair(directory, 0, image, image)
                info = artifacts['no_watermark']
                if corruption == 'file':
                    (Path(directory) / info['filename']).write_bytes(b'corrupt PNG')
                elif corruption == 'pixels':
                    info['pixel_sha256'] = '0' * 64
                elif corruption == 'filename':
                    info['filename'] = '../outside.png'
                else:
                    info['width'] = 9
                with self.assertRaises(ValueError):
                    verify_image_pair(directory, 0, artifacts, expected_size=8)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            save_image_pair(directory, 0, Image.new('L', (8, 8)), Image.new('RGB', (8, 8)))
