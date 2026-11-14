import hashlib
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pilot_selection as pilot
from run_naming import build_run_identity
from scripts.pilot_validation import strict_json, write_new


def constant(value):
    return [value] * pilot.PILOT_PROMPT_COUNT


TEST_COMMIT = "1" * 40
NO_WATERMARK_HASHES = [
    hashlib.sha256(f"no-watermark:{index}".encode()).hexdigest()
    for index in range(pilot.PILOT_PROMPT_COUNT)
]
NO_WATERMARK_DISTANCES = [100.0 + index / 100 for index in range(200)]


def observation(candidate, *, distance=20.0, clip=-0.003, lpips=0.5,
                psnr=15.0, ssim=0.5, rms=0.19, l2=24.0):
    return {
        "label": candidate["label"],
        "run_name": candidate["run_name"],
        "method_name": candidate["method_name"],
        "method_parameters": candidate["method_parameters"],
        "git_commit": TEST_COMMIT,
        "git_dirty": False,
        "source_row_ids": list(pilot.PILOT_SOURCE_ROW_IDS),
        "no_watermark_distances": list(NO_WATERMARK_DISTANCES),
        "no_watermark_image_sha256": list(NO_WATERMARK_HASHES),
        "watermarked_image_sha256": [
            hashlib.sha256(f"{candidate['label']}:{index}".encode()).hexdigest()
            for index in range(pilot.PILOT_PROMPT_COUNT)
        ],
        "watermarked_distances": constant(distance),
        "clip_delta": constant(clip),
        "lpips": constant(lpips),
        "psnr_db": constant(psnr),
        "ssim": constant(ssim),
        "latent_residual_rms": constant(rms),
        "latent_residual_l2": constant(l2),
    }


def complete_fixture():
    plan = pilot.candidate_plan()
    original = observation(plan[0])
    global_candidates = [observation(item, lpips=0.45) for item in plan[1:4]]
    global_candidates[0]["watermarked_distances"][-20:] = constant(200.0)[:20]
    global_candidates[1]["lpips"] = constant(0.40)
    saliency_candidates = [observation(item, lpips=0.48) for item in plan[4:]]
    saliency_candidates[3]["lpips"] = constant(0.35)
    return {
        "schema_version": pilot.PILOT_OBSERVATION_SCHEMA,
        "prompt_sha256": pilot.PILOT_PROMPT_SHA256,
        "implementation_commit": TEST_COMMIT,
        "source_row_ids": list(pilot.PILOT_SOURCE_ROW_IDS),
        "no_watermark_distances": list(NO_WATERMARK_DISTANCES),
        "no_watermark_image_sha256": list(NO_WATERMARK_HASHES),
        "original": original,
        "global_candidates": global_candidates,
        "saliency_candidates": saliency_candidates,
    }


class PilotSelectionTests(unittest.TestCase):
    def test_frozen_grid_and_boundaries(self):
        self.assertEqual(pilot.GLOBAL_ALPHA_GRID, (0.25, 0.5, 0.75))
        self.assertEqual(len(pilot.SALIENCY_GRID), 6)
        self.assertEqual(pilot.PILOT_PROMPT_COUNT, 200)
        self.assertEqual(pilot.PILOT_SOURCE_ROW_IDS, tuple(range(1000, 1200)))
        self.assertEqual(pilot.TPR_NONINFERIORITY_MARGIN, 0.05)
        self.assertEqual(pilot.CLIP_DELTA_MARGIN, 0.01)
        self.assertEqual(pilot.BOOTSTRAP_RESAMPLES, 10_000)
        specification = pilot.pilot_specification()
        self.assertFalse(specification["boundaries"]["formal_authorized"])
        self.assertTrue(specification["boundaries"]["formal_data_may_not_be_read"])
        self.assertEqual(len(specification["runs"]), 10)

    def test_all_frozen_run_names_match_runner_identity(self):
        for candidate in pilot.candidate_plan():
            parameters = candidate["method_parameters"]
            args = SimpleNamespace(
                run_name=candidate["run_name"], run_purpose="full",
                prompt_split="pilot", start=0, end=200,
                num_inference_steps=50, test_num_inference_steps=50,
                method_name=candidate["method_name"],
                global_alpha=parameters.get("alpha"),
                saliency_preview_steps=parameters.get("preview_steps"),
                saliency_background_strength=parameters.get("background_strength"),
                saliency_foreground_strength=parameters.get("foreground_strength"),
                saliency_power=parameters.get("power"),
                replicate_id=1, attack_name="clean",
            )
            identity = build_run_identity(args)
            self.assertEqual(identity["run_purpose"], "full")
            self.assertEqual(identity["sample_range"]["count"], 200)

    def test_threshold_has_at_most_one_percent_empirical_fpr(self):
        values = [100.0 + index / 100 for index in range(200)]
        result = pilot.calibrate_shared_threshold(values)
        self.assertEqual(result["false_positives"], 2)
        self.assertEqual(result["empirical_fpr"], 0.01)
        self.assertEqual(result["order_statistic_one_based"], 198)
        self.assertGreater(result["threshold"], -100.02)

    def test_threshold_is_conservative_at_ties(self):
        values = [100.0] * 200
        result = pilot.calibrate_shared_threshold(values)
        self.assertEqual(result["false_positives"], 0)
        self.assertEqual(result["empirical_fpr"], 0.0)

    def test_complete_selection_uses_constraints_and_lpips(self):
        fixture = complete_fixture()
        with patch.object(pilot, "BOOTSTRAP_RESAMPLES", 100):
            decision = pilot.select_pilot(fixture)
        self.assertEqual(decision["status"], "PILOT_SELECTION_PASS")
        self.assertEqual(decision["global_selection"]["selected_label"], "global_a0p5")
        self.assertEqual(
            decision["saliency_selection"]["selected_label"],
            pilot.candidate_plan()[7]["label"],
        )
        self.assertFalse(decision["formal_authorized"])
        self.assertFalse(decision["global_candidates"][0]["feasibility"]["tpr_noninferior"])

    def test_no_quality_benefit_stops_instead_of_forcing_selection(self):
        fixture = complete_fixture()
        for candidate in fixture["global_candidates"]:
            candidate["lpips"] = constant(0.6)
        with patch.object(pilot, "BOOTSTRAP_RESAMPLES", 50):
            decision = pilot.select_pilot(fixture)
        self.assertEqual(decision["status"], "PILOT_SELECTION_STOP_NO_FEASIBLE_CANDIDATE")
        self.assertEqual(decision["global_selection"]["status"], "NO_FEASIBLE_CANDIDATE_STOP")
        self.assertIsNone(decision["global_selection"]["selected_label"])

    def test_rejects_wrong_prompt_nonfinite_and_missing_candidates(self):
        cases = []
        wrong_prompt = complete_fixture()
        wrong_prompt["prompt_sha256"] = "0" * 64
        cases.append(wrong_prompt)
        nonfinite = complete_fixture()
        nonfinite["global_candidates"][0]["lpips"][0] = math.nan
        cases.append(nonfinite)
        missing = complete_fixture()
        missing["saliency_candidates"].pop()
        cases.append(missing)
        mismatched_pair = complete_fixture()
        mismatched_pair["global_candidates"][0]["no_watermark_image_sha256"][0] = "f" * 64
        cases.append(mismatched_pair)
        for value in cases:
            with self.subTest(case=len(value.get("saliency_candidates", []))):
                with self.assertRaises(ValueError):
                    pilot.select_pilot(value)

    def test_plan_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            write_new(output, pilot.pilot_specification())
            self.assertTrue(output.is_file())
            with self.assertRaises(FileExistsError):
                write_new(output, pilot.pilot_specification())

    def test_strict_json_rejects_duplicate_and_nonfinite_tokens(self):
        with self.assertRaises(ValueError):
            strict_json('{"value": 1, "value": 2}')
        with self.assertRaises(ValueError):
            strict_json('{"value": NaN}')


if __name__ == "__main__":
    unittest.main()
