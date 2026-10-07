import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiment_records import (
    SCHEMA_VERSION,
    append_jsonl,
    describe_prompt_file,
    namespace_to_dict,
    write_json,
    write_jsonl,
)


class ExperimentRecordTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
