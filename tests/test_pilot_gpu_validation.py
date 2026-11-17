import shutil
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pilot_selection
from scripts import pilot_gpu_validation as pilot_gpu


class PilotGpuValidationTests(unittest.TestCase):
    def make_repo(self, root):
        repo = Path(root) / "repo"
        prompt = repo / ".local_data/prompts/splits/sdp_pilot_rows_1000_1199.txt"
        prompt.parent.mkdir(parents=True)
        source = pilot_gpu.REPO / ".local_data/prompts/splits/sdp_pilot_rows_1000_1199.txt"
        shutil.copyfile(source, prompt)
        (repo / ".local_outputs/runs").mkdir(parents=True)
        return repo

    def git_state(self):
        return {
            "branch": "stage-c-record-identity",
            "commit": pilot_gpu.EXPECTED_HEAD,
            "dirty": False,
        }

    def write_manifest(self, path, root, relative_files):
        entries = []
        for relative in relative_files:
            source = Path(root) / relative
            entries.append({
                "path": relative,
                "size": source.stat().st_size,
                "sha256": pilot_gpu.file_sha256(source),
            })
        value = {
            "schema_version": "treering-file-manifest-v1",
            "root": str(Path(root).resolve()),
            "files": entries,
        }
        path.write_text(json.dumps(value), encoding="utf-8")
        return pilot_gpu.file_sha256(path)

    def fake_gpu_probe(self):
        return {
            "nvidia_smi": {"name": "NVIDIA B200"},
            "torch": {"cuda_available": True, "device_count": 1, "gpu_name": "NVIDIA B200"},
            "model_loaded": False,
            "image_evaluation_started": False,
        }

    def test_plan_binds_order_commands_and_budget_without_authorization(self):
        plan = pilot_gpu.build_plan()
        self.assertEqual(plan["status"], "PILOT_GPU_PLAN_PASS")
        self.assertEqual(len(plan["runs"]), 10)
        self.assertEqual(plan["runs"][0]["label"], "original")
        self.assertEqual(plan["runs"][1]["label"], "global_a0p25")
        self.assertEqual(plan["runs"][-1]["ordinal"], 10)
        self.assertEqual(plan["budget"]["aggregate_gpu_seconds"], 15 * 60 * 60)
        self.assertEqual(plan["budget"]["conditional_readiness_seconds"], 10 * 60)
        self.assertFalse(plan["execution_started"])
        self.assertFalse(plan["execution_authorized"])
        self.assertFalse(plan["formal_authorized"])
        for run in plan["runs"]:
            command = run["generation_command"]
            self.assertIn("--require_cuda", command)
            self.assertIn("--save_paired_images", command)
            self.assertNotIn("--overwrite_output", command)
            self.assertEqual(run["quality_command"][-2:], ["--device", "cuda"])

    def test_static_preflight_passes_without_live_or_gpu_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = self.make_repo(root)
            transfer = root / "transfer"
            transfer.mkdir()
            evidence = transfer / "pilot-gpu-execution_7e8ee07_20261116T120000Z"
            report = pilot_gpu.static_preflight(
                repo, transfer, evidence,
                git_state=self.git_state(), min_free_bytes=0,
                disk_usage=SimpleNamespace(total=1, used=0, free=1),
            )
            self.assertEqual(report["status"], "PILOT_GPU_STATIC_PREFLIGHT_PASS")
            self.assertFalse(report["gpu_checked"])
            self.assertFalse(report["execution_authorized"])
            self.assertFalse(report["execution_started"])
            self.assertFalse(report["formal_authorized"])
            self.assertFalse(evidence.exists())
            self.assertEqual(len(report["canonical_output_targets"]), 30)

    def test_static_preflight_rejects_occupied_output_and_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = self.make_repo(root)
            transfer = root / "transfer"
            transfer.mkdir()
            evidence = transfer / "pilot-gpu-execution_7e8ee07_20261116T120000Z"
            occupied = pilot_gpu.output_paths(
                repo, pilot_gpu.candidate_plan()[0]["run_name"]
            )[0]
            occupied.write_text("preserve", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "already occupied"):
                pilot_gpu.static_preflight(
                    repo, transfer, evidence,
                    git_state=self.git_state(), min_free_bytes=0,
                    disk_usage=SimpleNamespace(total=1, used=0, free=1),
                )
            occupied.unlink()
            evidence.mkdir()
            with self.assertRaisesRegex(ValueError, "already occupied"):
                pilot_gpu.static_preflight(
                    repo, transfer, evidence,
                    git_state=self.git_state(), min_free_bytes=0,
                    disk_usage=SimpleNamespace(total=1, used=0, free=1),
                )

    def test_static_preflight_rejects_identity_and_noncanonical_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = self.make_repo(root)
            transfer = root / "transfer"
            transfer.mkdir()
            wrong_state = {**self.git_state(), "dirty": True}
            evidence = transfer / "pilot-gpu-execution_7e8ee07_20261116T120000Z"
            with self.assertRaisesRegex(ValueError, "dirty"):
                pilot_gpu.static_preflight(
                    repo, transfer, evidence,
                    git_state=wrong_state, min_free_bytes=0,
                    disk_usage=SimpleNamespace(total=1, used=0, free=1),
                )
            with self.assertRaisesRegex(ValueError, "not canonical"):
                pilot_gpu.static_preflight(
                    repo, transfer, transfer / "pilot-output",
                    git_state=self.git_state(), min_free_bytes=0,
                    disk_usage=SimpleNamespace(total=1, used=0, free=1),
                )

    def test_live_preflight_replays_manifests_without_authorizing_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = self.make_repo(root)
            transfer = root / "transfer"
            transfer.mkdir()
            evidence = transfer / "pilot-gpu-execution_7e8ee07_20261116T120000Z"
            model = root / "model"
            model.mkdir()
            (model / "weight.bin").write_bytes(b"frozen-model")
            model_manifest = root / "model-manifest.json"
            model_sha = self.write_manifest(model_manifest, model, ["weight.bin"])
            protected_manifest = root / "protected-manifest.json"
            protected_relative = ".local_data/prompts/splits/sdp_pilot_rows_1000_1199.txt"
            protected_sha = self.write_manifest(protected_manifest, repo, [protected_relative])
            report = pilot_gpu.live_preflight(
                repo, transfer, evidence,
                model_manifest=model_manifest, model_manifest_sha256=model_sha,
                protected_manifest=protected_manifest,
                protected_manifest_sha256=protected_sha,
                quality_asset_paths={}, gpu_probe=self.fake_gpu_probe,
                git_state=self.git_state(), min_free_bytes=0,
                disk_usage=SimpleNamespace(total=1, used=0, free=1),
                model_root=model,
                quality_validator=lambda paths: {"status": "STAGE_F_QUALITY_ASSETS_PASS", "assets": {}},
            )
            self.assertEqual(report["status"], "PILOT_GPU_LIVE_PREFLIGHT_PASS")
            self.assertTrue(report["gpu_checked"])
            self.assertTrue(report["protected_snapshot_checked"])
            self.assertFalse(report["execution_authorized"])
            self.assertFalse(report["execution_started"])
            self.assertFalse(evidence.exists())

    def make_generation_artifacts(self, root, candidate):
        images = Path(root) / "images"
        images.mkdir()
        rows = []
        fixed = "a" * 64
        for index in range(200):
            for role in ("no_watermark", "watermarked"):
                (images / f"{index:04d}.{role}.png").write_bytes(b"")
            rows.append({
                "sample_id": f"{candidate['run_name']}:{index}",
                "sample_index": index,
                "run_purpose": "full",
                "run_naming_version": "treering-run-name-v2",
                "source_row_id": 1000 + index,
                "prompt_split": "pilot",
                "method_name": candidate["method_name"],
                "method_parameters": candidate["method_parameters"],
                "attack_name": "clean",
                "attack_parameters": {},
                "replicate_id": 1,
                "protocol_version": "v1.1-pilot",
                "watermark_key_id": "treering-rand-wseed-999999",
                "generation_seed": index,
                "watermark_seed": 999999,
                "prompt": f"prompt {index}",
                "latent_residual_rms": 0.1,
                "latent_residual_l2": 1.0,
                "no_w_metric": 100.0 + index,
                "w_metric": 20.0,
                "no_w_score": -100.0 - index,
                "w_score": -20.0,
                "w_no_sim": 0.0,
                "w_sim": 0.0,
                "sample_duration_seconds": 1.0,
                "paired_inputs": {
                    "base_latent_sha256": fixed,
                    "watermark_key_sha256": "b" * 64,
                    "watermark_mask_sha256": "c" * 64,
                },
                "image_artifacts": {
                    "contract": "treering-paired-rgb-png-v1",
                    "no_watermark": {"sha256": f"{index:064x}"},
                    "watermarked": {"sha256": f"{index + 1000:064x}"},
                },
            })
        summary = {
            "schema_version": "treering-experiment-v2",
            "run_name": candidate["run_name"],
            "num_samples": 200,
            "git": {"commit": pilot_gpu.EXPECTED_HEAD, "dirty": False},
            "run_identity": {
                "run_purpose": "full",
                "sample_range": {"start": 0, "end_exclusive": 200, "count": 200},
                "prompt_split": "pilot",
                "method_name": candidate["method_name"],
                "method_parameters": candidate["method_parameters"],
                "attack_name": "clean",
                "attack_parameters": {},
                "replicate_id": 1,
                "protocol_version": "v1.1-pilot",
                "watermark_key_id": "treering-rand-wseed-999999",
            },
            "prompt_source": {"sha256": pilot_gpu.PILOT_PROMPT_SHA256, "nonempty_line_count": 200},
            "environment": {
                "cuda_available": True, "device": "cuda", "gpu_name": "NVIDIA B200",
                "torch_dtype": "torch.float16",
            },
            "parameters": {
                **pilot_gpu.generation_config(candidate),
                "prompt_file": "/server/sdp_pilot_rows_1000_1199.txt",
                "model_id": "/server/stable-diffusion-2-1-base",
            },
            "results": rows,
        }
        summary_path = Path(root) / "summary.json"
        samples_path = Path(root) / "samples.jsonl"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return summary_path, samples_path, images

    def test_generation_structure_acceptance_hides_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = pilot_gpu.candidate_plan()[0]
            summary, samples, images = self.make_generation_artifacts(directory, candidate)

            def verifier(image_dir, index, artifacts, expected_size):
                return {
                    str(Path(image_dir) / f"{index:04d}.no_watermark.png"): artifacts["no_watermark"]["sha256"],
                    str(Path(image_dir) / f"{index:04d}.watermarked.png"): artifacts["watermarked"]["sha256"],
                }

            report = pilot_gpu.validate_generation_run(
                summary, samples, images, candidate, image_verifier=verifier
            )
            self.assertEqual(report["status"], "PILOT_GENERATION_STRUCTURE_PASS")
            self.assertEqual(report["record_count"], 200)
            self.assertEqual(report["png_count"], 400)
            self.assertFalse(report["outcomes_exposed"])
            self.assertNotIn("tpr", report)
            self.assertNotIn("mean", report)
            (images / "unexpected-directory").mkdir()
            with self.assertRaisesRegex(ValueError, "non-regular"):
                pilot_gpu.validate_generation_run(
                    summary, samples, images, candidate, image_verifier=verifier
                )

    def test_ten_run_acceptance_requires_shared_no_watermark_identity(self):
        reports = []
        common_hashes = [f"{index:064x}" for index in range(200)]
        paired = [{
            "base_latent_sha256": "a" * 64,
            "watermark_key_sha256": "b" * 64,
            "watermark_mask_sha256": "c" * 64,
        }] * 200
        for candidate in pilot_gpu.candidate_plan():
            reports.append({
                "status": "PILOT_GENERATION_STRUCTURE_PASS",
                "label": candidate["label"], "run_name": candidate["run_name"],
                "record_count": 200, "png_count": 400,
                "summary_sha256": "1" * 64, "samples_sha256": "2" * 64,
                "png_manifest_sha256": "3" * 64,
                "no_watermark_image_sha256": list(common_hashes),
                "no_watermark_distance_vector_sha256": "4" * 64,
                "paired_inputs": paired,
                "outcomes_exposed": False, "formal_authorized": False,
            })
        accepted = pilot_gpu.accept_generation_runs(reports)
        self.assertEqual(accepted["status"], "PILOT_GENERATION_ACCEPT_PASS")
        self.assertFalse(accepted["outcomes_exposed"])
        reports[-1]["no_watermark_image_sha256"][0] = "f" * 64
        with self.assertRaisesRegex(ValueError, "PNG hashes differ"):
            pilot_gpu.accept_generation_runs(reports)

    def make_quality_artifacts(self, root, candidate, generation):
        rows = []
        for index in range(200):
            no_w = 0.3 + index / 100000
            watermarked = no_w - 0.003
            rows.append({
                "sample_id": f"{candidate['run_name']}:{index}",
                "sample_index": index,
                "source_row_id": 1000 + index,
                "prompt_split": "pilot",
                "method_name": candidate["method_name"],
                "replicate_id": 1,
                "protocol_version": "v1.1-pilot",
                "attack_name": "clean",
                "watermark_key_id": "treering-rand-wseed-999999",
                "prompt": f"prompt {index}",
                "generation_seed": index,
                "watermark_seed": 999999,
                "source_image_sha256": {
                    "no_watermark": generation["no_watermark_image_sha256"][index],
                    "watermarked": generation["watermarked_image_sha256"][index],
                },
                "quality_metrics": {
                    "clip_text_image_cosine_no_watermark": no_w,
                    "clip_text_image_cosine_watermarked": watermarked,
                    "clip_text_image_cosine_delta": -0.003,
                    "lpips_alex_v0_1": 0.4,
                    "psnr_rgb": {"value_db": 16.0, "status": "finite", "mse": 0.02},
                    "ssim_rgb": 0.6,
                },
            })
        summary = {
            "schema_version": "treering-quality-evaluation-v1",
            "metric_spec_version": "treering-quality-v1",
            "git": {"commit": pilot_gpu.EXPECTED_HEAD, "dirty": False},
            "source_run": {
                "run_name": candidate["run_name"], "sample_count": 200,
                "summary_sha256": generation["summary_sha256"],
                "samples_sha256": generation["samples_sha256"],
            },
            "metrics": {
                "clip": {
                    "model_name": "ViT-g-14", "device": "cuda", "dtype": "float32",
                    "checkpoint_sha256": "6aac683f899159946bc4ca15228bb7016f3cbb1a2c51f365cba0b23923f344da",
                },
                "lpips": {
                    "version": "treering-lpips-alex-v0.1", "device": "cuda", "dtype": "float32",
                    "linear_checkpoint_sha256": "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
                    "alexnet_checkpoint_sha256": "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
                },
                "fid": {"status": "omitted_by_approved_protocol_decision"},
            },
            "results": rows,
        }
        summary_path = Path(root) / "quality.summary.json"
        samples_path = Path(root) / "quality.samples.jsonl"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        samples_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return summary_path, samples_path

    def test_quality_structure_binds_generation_and_hides_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = pilot_gpu.candidate_plan()[0]
            generation = {
                "summary_sha256": "1" * 64, "samples_sha256": "2" * 64,
                "no_watermark_image_sha256": [f"{index:064x}" for index in range(200)],
                "watermarked_image_sha256": [f"{index + 1000:064x}" for index in range(200)],
            }
            summary, samples = self.make_quality_artifacts(root, candidate, generation)
            report = pilot_gpu.validate_quality_run(summary, samples, candidate, generation)
            self.assertEqual(report["status"], "PILOT_QUALITY_STRUCTURE_PASS")
            self.assertEqual(report["record_count"], 200)
            self.assertFalse(report["outcomes_exposed"])
            self.assertNotIn("_vectors", report)
            altered = {**generation, "summary_sha256": "f" * 64}
            with self.assertRaisesRegex(ValueError, "source hashes"):
                pilot_gpu.validate_quality_run(summary, samples, candidate, altered)

    def test_observation_assembly_and_final_selection_keep_formal_false(self):
        no_w_distances = [100.0 + index / 100 for index in range(200)]
        no_w_hashes = [f"{index:064x}" for index in range(200)]
        clip_no_w = [0.3 + index / 100000 for index in range(200)]
        generations = []
        qualities = []
        for position, candidate in enumerate(pilot_gpu.candidate_plan()):
            generations.append({
                "status": "PILOT_GENERATION_STRUCTURE_PASS",
                "label": candidate["label"],
                "no_watermark_image_sha256": list(no_w_hashes),
                "watermarked_image_sha256": [f"{position * 1000 + index + 1:064x}" for index in range(200)],
                "_vectors": {
                    "no_watermark_distances": list(no_w_distances),
                    "watermarked_distances": [20.0] * 200,
                    "latent_residual_rms": [0.2] * 200,
                    "latent_residual_l2": [24.0] * 200,
                },
            })
            qualities.append({
                "status": "PILOT_QUALITY_STRUCTURE_PASS",
                "label": candidate["label"],
                "_vectors": {
                    "clip_no_watermark": list(clip_no_w),
                    "clip_delta": [-0.003] * 200,
                    "lpips": ([0.5] if position == 0 else [0.4]) * 200,
                    "psnr_db": [16.0] * 200,
                    "ssim": [0.6] * 200,
                },
            })
        observations = pilot_gpu.build_observation_document(generations, qualities)
        self.assertEqual(observations["schema_version"], pilot_selection.PILOT_OBSERVATION_SCHEMA)
        generation_accept = {
            "status": "PILOT_GENERATION_ACCEPT_PASS",
            "outcomes_exposed": False,
            "formal_authorized": False,
        }
        assembly = {
            "status": "PILOT_OBSERVATIONS_ASSEMBLED_PASS",
            "observation_semantic_sha256": pilot_gpu.hashlib.sha256(
                json.dumps(observations, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
            ).hexdigest(),
            "formal_authorized": False,
        }
        with patch.object(pilot_selection, "BOOTSTRAP_RESAMPLES", 20):
            accepted = pilot_gpu.final_selection_accept(observations, generation_accept, assembly)
        self.assertEqual(accepted["status"], "PILOT_GPU_ACCEPT_AND_SELECTION_PASS")
        self.assertEqual(accepted["decision"]["status"], "PILOT_SELECTION_PASS")
        self.assertFalse(accepted["formal_authorized"])
        self.assertFalse(accepted["decision"]["formal_authorized"])
        assembly["observation_semantic_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "assembly evidence"):
            pilot_gpu.final_selection_accept(observations, generation_accept, assembly)

    def test_new_reports_are_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plan.json"
            pilot_gpu.write_new(output, pilot_gpu.build_plan())
            with self.assertRaises(FileExistsError):
                pilot_gpu.write_new(output, pilot_gpu.build_plan())
            second = Path(directory) / "second.json"
            with self.assertRaisesRegex(ValueError, "occupied"):
                pilot_gpu.check_new_outputs(output, second)
            with self.assertRaisesRegex(ValueError, "distinct"):
                pilot_gpu.check_new_outputs(second, second)

    def test_cli_has_no_gpu_execution_mode(self):
        parser = pilot_gpu.build_parser()
        subparsers = next(
            action for action in parser._actions
            if action.__class__.__name__ == "_SubParsersAction"
        )
        self.assertEqual(
            set(subparsers.choices),
            {
                "plan", "static-preflight", "live-preflight", "inspect-generation",
                "accept-generation", "inspect-quality", "assemble-observations", "final-accept",
            },
        )
        self.assertNotIn("run-one", subparsers.choices)
        self.assertNotIn("quality-one", subparsers.choices)


if __name__ == "__main__":
    unittest.main()
