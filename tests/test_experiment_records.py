import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiment_records import (
    SCHEMA_VERSION,
    append_jsonl,
    build_sample_identity,
    describe_prompt_file,
    expected_run_name,
    namespace_to_dict,
    source_row_id_for_sample,
    validate_attack_configuration,
    validate_method_configuration,
    validate_prompt_source,
    validate_run_name,
    validate_sample_range,
    write_json,
    write_jsonl,
)


class ExperimentRecordTests(unittest.TestCase):
    def test_method_parameters_require_explicit_global_alpha(self):
        self.assertEqual(validate_method_configuration("original_tree_ring"), {})
        for alpha in (0, 0.25, 1):
            with self.subTest(alpha=alpha):
                self.assertEqual(
                    validate_method_configuration("globally_weaker_tree_ring", alpha),
                    {"alpha": float(alpha)},
                )

    def test_method_parameters_reject_invalid_alpha_and_unimplemented_methods(self):
        for alpha in (None, True, False, "0.5", 1j, -0.1, 1.1,
                      float("nan"), float("inf"), -float("inf"), 10 ** 400):
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    validate_method_configuration("globally_weaker_tree_ring", alpha)
        with self.assertRaises(ValueError):
            validate_method_configuration("original_tree_ring", 0.5)
        for method in ("saliency_aware_tree_ring", "no_watermark", "unknown"):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    validate_method_configuration(method)

    def test_global_sample_identity_and_run_name(self):
        identity = build_sample_identity(
            prompt_split="development", sample_index=7,
            method_name="globally_weaker_tree_ring", attack_name="clean",
            replicate_id=1, protocol_version="v1",
            watermark_key_id="treering-rand-wseed-999999",
        )
        self.assertEqual(identity["source_row_id"], 1207)
        self.assertEqual(identity["method_name"], "globally_weaker_tree_ring")
        self.assertEqual(
            validate_run_name("global_development_r1_clean",
                              "globally_weaker_tree_ring", "development", 1, "clean"),
            "global_development_r1_clean",
        )

    def test_schema_version_is_v2(self):
        self.assertEqual(
            SCHEMA_VERSION,
            "treering-experiment-v2",
        )

    def test_describe_prompt_file_records_hash_and_count(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "prompts.txt"
            content = "first prompt\nsecond prompt\n"
            path.write_text(content, encoding="utf-8", newline="\n")

            description = describe_prompt_file(path)

            self.assertEqual(description["nonempty_line_count"], 2)
            self.assertEqual(
                description["sha256"],
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
            self.assertEqual(description["size_bytes"], len(content.encode("utf-8")))

    def test_namespace_conversion_is_json_serializable(self):
        args = SimpleNamespace(
            run_name="development",
            prompt_file=Path("prompts.txt"),
            values=(1, 2),
        )

        converted = namespace_to_dict(args)

        self.assertEqual(converted["run_name"], "development")
        self.assertEqual(converted["prompt_file"], "prompts.txt")
        self.assertEqual(converted["values"], [1, 2])
        json.dumps(converted)

    def test_source_row_id_for_frozen_splits(self):
        self.assertEqual(
            source_row_id_for_sample("formal", 0),
            0,
        )
        self.assertEqual(
            source_row_id_for_sample("formal", 999),
            999,
        )
        self.assertEqual(
            source_row_id_for_sample("pilot", 0),
            1000,
        )
        self.assertEqual(
            source_row_id_for_sample("pilot", 199),
            1199,
        )
        self.assertEqual(
            source_row_id_for_sample("development", 0),
            1200,
        )
        self.assertEqual(
            source_row_id_for_sample("development", 31),
            1231,
        )

        with self.assertRaises(ValueError):
            source_row_id_for_sample("unknown", 0)

        with self.assertRaises(ValueError):
            source_row_id_for_sample("formal", -1)
        with self.assertRaises(ValueError):
            source_row_id_for_sample("development", 32)

    def test_validate_sample_range_uses_frozen_split_bounds(self):
        self.assertEqual(
            validate_sample_range("development", 0, 1),
            {
                "start": 0,
                "end_exclusive": 1,
                "count": 1,
            },
        )
        self.assertEqual(
            validate_sample_range("development", 0, 32),
            {
                "start": 0,
                "end_exclusive": 32,
                "count": 32,
            },
        )

        for start, end in ((-1, 1), (0, 0), (2, 1), (0, 33)):
            with self.subTest(start=start, end=end):
                with self.assertRaises(ValueError):
                    validate_sample_range("development", start, end)

    def test_run_name_matches_explicit_identity(self):
        expected = expected_run_name(
            "original_tree_ring",
            "development",
            1,
            "clean",
        )
        self.assertEqual(
            expected,
            "original_development_r1_clean",
        )
        self.assertEqual(
            validate_run_name(
                expected,
                "original_tree_ring",
                "development",
                1,
                "clean",
            ),
            expected,
        )

        with self.assertRaises(ValueError):
            validate_run_name(
                "wrong_name",
                "original_tree_ring",
                "development",
                1,
                "clean",
            )

    def test_build_sample_identity_records_required_fields(self):
        identity = build_sample_identity(
            prompt_split="development",
            sample_index=7,
            method_name="original_tree_ring",
            replicate_id=1,
            protocol_version="v1",
            watermark_key_id="treering-rand-wseed-999999",
            attack_name="clean",
        )

        self.assertEqual(
            identity,
            {
                "source_row_id": 1207,
                "prompt_split": "development",
                "method_name": "original_tree_ring",
                "attack_name": "clean",
                "replicate_id": 1,
                "protocol_version": "v1",
                "watermark_key_id": "treering-rand-wseed-999999",
            },
        )

    def test_build_sample_identity_rejects_invalid_metadata(self):
        common = {
            "prompt_split": "development",
            "sample_index": 0,
            "method_name": "original_tree_ring",
            "attack_name": "clean",
            "replicate_id": 1,
            "protocol_version": "v1",
            "watermark_key_id": "treering-rand-wseed-999999",
        }

        for field_name, invalid_value in (
            ("method_name", "original"),
            ("replicate_id", 0),
            ("protocol_version", ""),
            ("watermark_key_id", ""),
            ("attack_name", "unknown"),
        ):
            invalid = dict(common)
            invalid[field_name] = invalid_value

            with self.subTest(field_name=field_name):
                with self.assertRaises(ValueError):
                    build_sample_identity(**invalid)

    def test_validate_attack_configuration_rejects_mismatch(self):
        valid_configurations = (
            ("clean", {}, {}),
            ("rotation", {"r_degree": 75}, {"r_degree": 75}),
            ("jpeg", {"jpeg_ratio": 25}, {"jpeg_ratio": 25}),
            (
                "crop",
                {"crop_scale": 0.75, "crop_ratio": 0.75},
                {"crop_scale": 0.75, "crop_ratio": 0.75},
            ),
            (
                "gaussian_blur",
                {"gaussian_blur_r": 4},
                {"gaussian_blur_r": 4},
            ),
            (
                "gaussian_noise",
                {"gaussian_std": 0.1},
                {"gaussian_std": 0.1},
            ),
            (
                "brightness",
                {"brightness_factor": 6},
                {"brightness_factor": 6},
            ),
        )
        for attack_name, arguments, expected in valid_configurations:
            with self.subTest(attack_name=attack_name):
                self.assertEqual(
                    validate_attack_configuration(
                        attack_name,
                        **arguments,
                    ),
                    expected,
                )

        with self.assertRaises(ValueError):
            validate_attack_configuration(
                "clean",
                r_degree=75,
            )

        with self.assertRaises(ValueError):
            validate_attack_configuration(
                "rotation",
                r_degree=30,
            )

        with self.assertRaises(ValueError):
            validate_attack_configuration(
                "clean",
                rand_aug=1,
            )

    def test_json_and_jsonl_writers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            json_path = output / "summary.json"
            jsonl_path = output / "samples.jsonl"

            write_json(
                json_path,
                {"schema_version": SCHEMA_VERSION, "count": 2},
            )
            write_jsonl(
                jsonl_path,
                [{"sample_id": 0}, {"sample_id": 1}],
            )
            append_jsonl(jsonl_path, {"sample_id": 2})

            summary = json.loads(json_path.read_text(encoding="utf-8"))
            samples = [
                json.loads(line)
                for line in jsonl_path.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
            self.assertEqual(summary["count"], 2)
            self.assertEqual(
                [sample["sample_id"] for sample in samples],
                [0, 1, 2],
            )

    def test_validate_prompt_source_matches_frozen_split(self):
        development = {
            "path": "sdp_dev_rows_1200_1231.txt",
            "size_bytes": 8072,
            "sha256": (
                "39ed02810d78fdec8ab535935ac8f2e3819043c7db"
                "15a7585d2a999990b57d5f"
            ),
            "nonempty_line_count": 32,
        }

        validated = validate_prompt_source(
            "development",
            development,
        )
        self.assertEqual(validated, development)

        wrong_hash = dict(development)
        wrong_hash["sha256"] = "0" * 64

        wrong_count = dict(development)
        wrong_count["nonempty_line_count"] = 31

        with self.assertRaises(ValueError):
            validate_prompt_source(
                "development",
                wrong_hash,
            )

        with self.assertRaises(ValueError):
            validate_prompt_source(
                "pilot",
                development,
            )

        with self.assertRaises(ValueError):
            validate_prompt_source(
                "development",
                wrong_count,
            )

        with self.assertRaises(ValueError):
            validate_prompt_source(
                "development",
                None,
            )


if __name__ == "__main__":
    unittest.main()
