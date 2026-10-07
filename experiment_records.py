from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "treering-experiment-v1"


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
