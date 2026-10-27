import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from PIL import Image

from image_artifacts import save_image_pair
from quality_metrics import (
    ALEXNET_CHECKPOINT_SHA256,
    CLIP_CHECKPOINT_SHA256,
    CLIP_IMAGE_MEAN,
    CLIP_IMAGE_STD,
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED_TAG,
    LPIPS_LINEAR_CHECKPOINT_SHA256,
    ClipScorer,
    LpipsScorer,
    psnr_rgb,
    psnr_summary,
    scalar_summary,
    ssim_rgb,
)
from scripts.quality_validation import evaluate
from scripts.validate_quality_assets import ASSETS, validate as validate_assets


class FakeClipScorer:
    provenance = {"implementation": "test-double"}

    def score_pair(self, no_watermark_image, watermarked_image, prompt):
        return 0.4, 0.35


class FakeLpipsScorer:
    provenance = {"implementation": "test-double"}

    def score(self, reference, candidate):
        return float(np.mean(np.abs(reference - candidate)))


class QualityMetricTests(unittest.TestCase):
    def test_clip_protocol_identity_and_preprocessing_are_frozen(self):
        self.assertEqual(CLIP_MODEL_NAME, "ViT-g-14")
        self.assertEqual(CLIP_PRETRAINED_TAG, "laion2b_s12b_b42k")
        self.assertEqual(CLIP_CHECKPOINT_SHA256, "6aac683f899159946bc4ca15228bb7016f3cbb1a2c51f365cba0b23923f344da")
        self.assertEqual(CLIP_IMAGE_MEAN, (0.48145466, 0.4578275, 0.40821073))
        self.assertEqual(CLIP_IMAGE_STD, (0.26862954, 0.26130258, 0.27577711))
        self.assertEqual(LPIPS_LINEAR_CHECKPOINT_SHA256, "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0")
        self.assertEqual(ALEXNET_CHECKPOINT_SHA256, "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02")
        with self.assertRaises(ValueError):
            ClipScorer("ViT-B-32", Path("missing.bin"))

    def test_identical_pair_has_exact_psnr_and_unit_ssim(self):
        image = np.linspace(0, 1, 16 * 16 * 3, dtype=np.float64).reshape(16, 16, 3)
        psnr = psnr_rgb(image, image.copy())
        self.assertIsNone(psnr["value_db"])
        self.assertEqual(psnr["status"], "exact_match_positive_infinity")
        self.assertEqual(psnr["mse"], 0.0)
        self.assertAlmostEqual(ssim_rgb(image, image.copy()), 1.0, places=12)

    def test_maximally_different_constant_pair_has_zero_db_and_low_ssim(self):
        black = np.zeros((16, 16, 3), dtype=np.float64)
        white = np.ones((16, 16, 3), dtype=np.float64)
        result = psnr_rgb(black, white)
        self.assertEqual(result["status"], "finite")
        self.assertAlmostEqual(result["value_db"], 0.0, places=12)
        self.assertLess(ssim_rgb(black, white), 0.001)

    def test_noise_ordering_and_summaries(self):
        rng = np.random.RandomState(7)
        reference = rng.rand(32, 32, 3)
        mild = np.clip(reference + 0.01, 0, 1)
        strong = np.clip(reference + 0.1, 0, 1)
        self.assertGreater(psnr_rgb(reference, mild)["value_db"], psnr_rgb(reference, strong)["value_db"])
        self.assertGreater(ssim_rgb(reference, mild), ssim_rgb(reference, strong))
        summary = scalar_summary([1, 2, 3])
        self.assertEqual(summary["mean"], 2.0)
        psnr = psnr_summary([
            psnr_rgb(reference, reference),
            psnr_rgb(reference, mild),
        ])
        self.assertEqual(psnr["exact_match_count"], 1)
        self.assertEqual(psnr["finite"]["count"], 1)

    def test_invalid_inputs_are_rejected(self):
        valid = np.zeros((16, 16, 3), dtype=np.float64)
        for invalid in (
            np.zeros((16, 16), dtype=np.float64),
            np.zeros((10, 16, 3), dtype=np.float64),
            np.full((16, 16, 3), np.nan),
            np.full((16, 16, 3), 2.0),
        ):
            with self.subTest(shape=invalid.shape), self.assertRaises(ValueError):
                psnr_rgb(valid, invalid)

    def test_quality_asset_manifest_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory) / "asset.bin"
            asset.write_bytes(b"fixture")
            import hashlib
            specification = {
                "fixture": {
                    "size": 7,
                    "sha256": hashlib.sha256(b"fixture").hexdigest(),
                    "source": "test",
                }
            }
            with patch.dict(ASSETS, specification, clear=True):
                report = validate_assets({"fixture": asset})
                self.assertEqual(report["status"], "STAGE_F_QUALITY_ASSETS_PASS")
                asset.write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    validate_assets({"fixture": asset})

    def test_model_metrics_reject_missing_weights_before_import_or_download(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.pth"
            with self.assertRaises(ValueError):
                ClipScorer("ViT-g-14", missing)
            with self.assertRaises(ValueError):
                LpipsScorer(missing, missing)

    def test_weight_hash_format_is_rejected_before_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pth"
            checkpoint.write_bytes(b"fixture")
            with self.assertRaises(ValueError):
                ClipScorer("ViT-g-14", checkpoint, expected_sha256="not-a-sha")
            with self.assertRaises(ValueError):
                LpipsScorer(
                    checkpoint, checkpoint,
                    expected_linear_sha256="0" * 64,
                    expected_alexnet_sha256="0" * 64,
                )

    def test_offline_evaluator_preserves_sources_and_writes_exclusive_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "source.images"
            images.mkdir()
            no_watermark = Image.new("RGB", (512, 512), (10, 20, 30))
            watermarked = Image.new("RGB", (512, 512), (11, 20, 30))
            artifacts = save_image_pair(images, 0, no_watermark, watermarked)
            row = {
                "sample_id": "run:0", "sample_index": 0, "source_row_id": 1200,
                "prompt_split": "development", "method_name": "original_tree_ring",
                "replicate_id": 1, "protocol_version": "v1", "attack_name": "clean",
                "watermark_key_id": "key", "prompt": "fixture prompt",
                "generation_seed": 0, "watermark_seed": 999999,
                "image_artifacts": artifacts,
            }
            summary_path = root / "source.json"
            samples_path = root / "source.samples.jsonl"
            summary_path.write_text(json.dumps({"run_name": "run", "num_samples": 1, "results": [row]}), encoding="utf-8")
            samples_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            before = {path: path.read_bytes() for path in (summary_path, samples_path, *images.iterdir())}
            output = root / "quality"
            report = evaluate(summary_path, samples_path, images, output, FakeClipScorer(), FakeLpipsScorer())
            self.assertEqual(report["source_run"]["sample_count"], 1)
            self.assertEqual(report["metrics"]["fid"]["status"], "omitted_by_approved_protocol_decision")
            self.assertEqual(len((output / "quality.samples.jsonl").read_text(encoding="utf-8").splitlines()), 1)
            self.assertTrue((output / "quality.summary.json").is_file())
            self.assertTrue((output / "SHA256SUMS.json").is_file())
            self.assertEqual(before, {path: path.read_bytes() for path in before})
            with self.assertRaises(FileExistsError):
                evaluate(summary_path, samples_path, images, output, FakeClipScorer(), FakeLpipsScorer())


if __name__ == "__main__":
    unittest.main()
