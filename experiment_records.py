from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from numbers import Real
from pathlib import Path


SCHEMA_VERSION = "treering-experiment-v2"
FROZEN_PROMPT_SPLITS = {
    "formal": {
        "source_row_start": 0,
        "count": 1000,
        "sha256": (
            "949e3e7acb3c7b8bce874add0d822823e58826a1a28"
            "c9390b6f712eb7bdf57ce"
        ),
    },
    "pilot": {
        "source_row_start": 1000,
        "count": 200,
        "sha256": (
            "1455baf5b73b143aa07227c2dcfab96d5658c29e816"
            "fd9ddea33b591e7b70f3b"
        ),
    },
    "development": {
        "source_row_start": 1200,
        "count": 32,
        "sha256": (
            "39ed02810d78fdec8ab535935ac8f2e3819043c7db15"
            "a7585d2a999990b57d5f"
        ),
    },
}
METHOD_RUN_TOKENS = {
    "no_watermark": "no_watermark",
    "original_tree_ring": "original",
    "globally_weaker_tree_ring": "global",
    "saliency_aware_tree_ring": "saliency",
}
FROZEN_METHOD_NAMES = frozenset(METHOD_RUN_TOKENS)
IMPLEMENTED_METHOD_NAMES = frozenset({
    "original_tree_ring",
    "globally_weaker_tree_ring",
})
FROZEN_ATTACK_PARAMETERS = {
    "clean": {},
    "rotation": {
        "r_degree": 75,
    },
    "jpeg": {
        "jpeg_ratio": 25,
    },
    "crop": {
        "crop_scale": 0.75,
        "crop_ratio": 0.75,
    },
    "gaussian_blur": {
        "gaussian_blur_r": 4,
    },
    "gaussian_noise": {
        "gaussian_std": 0.1,
    },
    "brightness": {
        "brightness_factor": 6,
    },
}
FROZEN_ATTACK_NAMES = frozenset(FROZEN_ATTACK_PARAMETERS)


def validate_method_configuration(method_name, global_alpha=None):
    """Return explicit parameters only for methods implemented by the runner."""
    if method_name not in IMPLEMENTED_METHOD_NAMES:
        raise ValueError(f"Method is not implemented: {method_name!r}")

    if method_name == "original_tree_ring":
        if global_alpha is not None:
            raise ValueError("global_alpha is only valid for globally_weaker_tree_ring")
        return {}

    if (
        isinstance(global_alpha, bool)
        or not isinstance(global_alpha, Real)
        or not 0 <= global_alpha <= 1
        or not math.isfinite(global_alpha)
    ):
        raise ValueError("global_alpha must be an explicit finite number in [0, 1]")

    return {"alpha": float(global_alpha)}


def source_row_id_for_sample(prompt_split, sample_index):
    if prompt_split not in FROZEN_PROMPT_SPLITS:
        raise ValueError(
            f"Unknown prompt split: {prompt_split!r}"
        )

    if not isinstance(sample_index, int) or isinstance(sample_index, bool):
        raise TypeError("sample_index must be an integer")

    split = FROZEN_PROMPT_SPLITS[prompt_split]

    if sample_index < 0 or sample_index >= split["count"]:
        raise ValueError(
            f"sample_index {sample_index} is outside the frozen "
            f"{prompt_split!r} split range 0-{split['count'] - 1}"
        )

    return split["source_row_start"] + sample_index


def validate_sample_range(prompt_split, start, end):
    if prompt_split not in FROZEN_PROMPT_SPLITS:
        raise ValueError(
            f"Unknown prompt split: {prompt_split!r}"
        )

    for field_name, value in (("start", start), ("end", end)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{field_name} must be an integer")

    split_count = FROZEN_PROMPT_SPLITS[prompt_split]["count"]
    if start < 0 or end <= start or end > split_count:
        raise ValueError(
            f"Sample range [{start}, {end}) is outside the frozen "
            f"{prompt_split!r} split range [0, {split_count})"
        )

    return {
        "start": start,
        "end_exclusive": end,
        "count": end - start,
    }


def expected_run_name(
    method_name,
    prompt_split,
    replicate_id,
    attack_name,
):
    if method_name not in METHOD_RUN_TOKENS:
        raise ValueError(
            f"Unknown method_name: {method_name!r}"
        )
    if prompt_split not in FROZEN_PROMPT_SPLITS:
        raise ValueError(
            f"Unknown prompt split: {prompt_split!r}"
        )
    if attack_name not in FROZEN_ATTACK_NAMES:
        raise ValueError(
            f"Unknown attack_name: {attack_name!r}"
        )
    if (
        not isinstance(replicate_id, int)
        or isinstance(replicate_id, bool)
        or replicate_id < 1
    ):
        raise ValueError(
            "replicate_id must be a positive integer"
        )

    return (
        f"{METHOD_RUN_TOKENS[method_name]}_{prompt_split}_"
        f"r{replicate_id}_{attack_name}"
    )


def validate_run_name(
    run_name,
    method_name,
    prompt_split,
    replicate_id,
    attack_name,
):
    expected = expected_run_name(
        method_name,
        prompt_split,
        replicate_id,
        attack_name,
    )
    if run_name != expected:
        raise ValueError(
            f"run_name must be {expected!r}, received {run_name!r}"
        )
    return run_name


def validate_prompt_source(prompt_split, prompt_source):
    if prompt_split not in FROZEN_PROMPT_SPLITS:
        raise ValueError(
            f"Unknown prompt split: {prompt_split!r}"
        )

    if not isinstance(prompt_source, dict):
        raise ValueError(
            "A frozen prompt file is required"
        )

    expected = FROZEN_PROMPT_SPLITS[prompt_split]
    observed_count = prompt_source.get("nonempty_line_count")
    observed_sha256 = prompt_source.get("sha256")

    if observed_count != expected["count"]:
        raise ValueError(
            f"Prompt count for {prompt_split!r} must be "
            f"{expected['count']}, received {observed_count!r}"
        )

    if observed_sha256 != expected["sha256"]:
        raise ValueError(
            f"Prompt SHA-256 for {prompt_split!r} must be "
            f"{expected['sha256']}, received {observed_sha256!r}"
        )

    return prompt_source


def build_sample_identity(
    prompt_split,
    sample_index,
    method_name,
    attack_name,
    replicate_id,
    protocol_version,
    watermark_key_id,
):
    source_row_id = source_row_id_for_sample(
        prompt_split,
        sample_index,
    )

    if method_name not in FROZEN_METHOD_NAMES:
        raise ValueError(
            f"Unknown method_name: {method_name!r}"
        )
    if attack_name not in FROZEN_ATTACK_NAMES:
        raise ValueError(
            f"Unknown attack_name: {attack_name!r}"
        )

    if (
        not isinstance(replicate_id, int)
        or isinstance(replicate_id, bool)
        or replicate_id < 1
    ):
        raise ValueError(
            "replicate_id must be a positive integer"
        )

    for field_name, value in (
        ("protocol_version", protocol_version),
        ("watermark_key_id", watermark_key_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"{field_name} must be a non-empty string"
            )

    return {
        "source_row_id": source_row_id,
        "prompt_split": prompt_split,
        "method_name": method_name,
        "attack_name": attack_name,
        "replicate_id": replicate_id,
        "protocol_version": protocol_version,
        "watermark_key_id": watermark_key_id,
    }


def validate_attack_configuration(
    attack_name,
    *,
    r_degree=None,
    jpeg_ratio=None,
    crop_scale=None,
    crop_ratio=None,
    gaussian_blur_r=None,
    gaussian_std=None,
    brightness_factor=None,
    rand_aug=0,
):
    if attack_name not in FROZEN_ATTACK_PARAMETERS:
        raise ValueError(
            f"Unknown attack_name: {attack_name!r}"
        )

    if rand_aug != 0:
        raise ValueError(
            "rand_aug must remain 0 under the frozen protocol"
        )

    provided = {
        key: value
        for key, value in {
            "r_degree": r_degree,
            "jpeg_ratio": jpeg_ratio,
            "crop_scale": crop_scale,
            "crop_ratio": crop_ratio,
            "gaussian_blur_r": gaussian_blur_r,
            "gaussian_std": gaussian_std,
            "brightness_factor": brightness_factor,
        }.items()
        if value is not None
    }

    expected = FROZEN_ATTACK_PARAMETERS[attack_name]

    if provided != expected:
        raise ValueError(
            f"Attack parameters for {attack_name!r} must be "
            f"{expected!r}, received {provided!r}"
        )

    return dict(expected)

def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path):
    path = Path(path)
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return str(value)


def namespace_to_dict(args):
    return {
        key: json_safe(value)
        for key, value in sorted(vars(args).items())
    }


def get_git_state(repo_dir):
    repo_dir = Path(repo_dir)

    def run_git(*arguments):
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = run_git("rev-parse", "HEAD")
    branch = run_git("branch", "--show-current")
    status = run_git("status", "--porcelain")

    return {
        "commit": commit,
        "branch": branch,
        "dirty": None if status is None else bool(status),
    }


def describe_prompt_file(prompt_file):
    if prompt_file is None:
        return None

    path = Path(prompt_file).resolve()
    line_count = sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )

    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "nonempty_line_count": line_count,
    }


def collect_environment(torch_module=None):
    information = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
    }

    if torch_module is not None:
        cuda_available = bool(torch_module.cuda.is_available())
        information.update({
            "torch_version": torch_module.__version__,
            "cuda_available": cuda_available,
            "torch_cuda_version": torch_module.version.cuda,
            "gpu_name": (
                torch_module.cuda.get_device_name(0)
                if cuda_available
                else None
            ),
        })

    return information


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(text, encoding="utf-8", newline="\n")
    temporary_path.replace(path)


def write_json(path, value):
    text = json.dumps(
        json_safe(value),
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    atomic_write_text(path, text)


def write_jsonl(path, records):
    text = "".join(
        json.dumps(json_safe(record), ensure_ascii=False) + "\n"
        for record in records
    )
    atomic_write_text(path, text)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(
            json.dumps(json_safe(record), ensure_ascii=False) + "\n"
        )
        file.flush()
