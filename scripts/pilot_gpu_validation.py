"""Outcome-blind Pilot GPU planning and static preflight validation.

This module deliberately contains no GPU execution mode.  It cannot start the
diffusion runner or quality evaluator.  Live GPU preflight, per-run execution,
and final acceptance are added only after separate review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiment_records import get_git_state, utc_now_iso
from image_artifacts import verify_image_pair
from pilot_selection import (
    PILOT_OBSERVATION_SCHEMA, PILOT_PROMPT_SHA256, PILOT_SOURCE_ROW_IDS,
    candidate_plan, pilot_specification, select_pilot,
)
from scripts.validate_quality_assets import validate as validate_quality_assets


PLAN_VERSION = "treering-pilot-gpu-execution-plan-v1"
PILOT_FREEZE_BASE_HEAD = "7e8ee072e59ecf59449beb076716617361e8b30d"
# Backward-compatible test/default identity. Live and artifact-validation modes
# bind an explicit reviewed runtime commit instead of attempting an impossible
# self-referential commit constant.
EXPECTED_HEAD = PILOT_FREEZE_BASE_HEAD
SERVER_REPO = Path("/public/home/rxa-jj-ljq/projects/TreeRing-Advancement")
SERVER_TRANSFER = Path("/public/home/rxa-jj-ljq/TR_Transfer")
SERVER_MODEL = Path("/public/home/rxa-jj-ljq/models/stable-diffusion-2-1-base")
SERVER_PYTHON = Path("/public/home/rxa-jj-ljq/venvs/treering/bin/python")
SERVER_QUALITY_ASSETS = Path(
    "/public/home/rxa-jj-ljq/projects/TreeRing-Advancement_stage_f_assets/v1_1"
)
EVIDENCE_NAME = re.compile(r"^pilot-gpu-execution_7e8ee07_\d{8}T\d{6}Z$")
MIN_FREE_BYTES = 10 * 1024 ** 3

GENERATION_TIMEOUTS = {
    "original_tree_ring": 45 * 60,
    "globally_weaker_tree_ring": 45 * 60,
    "saliency_aware_tree_ring": 60 * 60,
}
QUALITY_TIMEOUT_SECONDS = 35 * 60
AGGREGATE_GPU_BUDGET_SECONDS = 15 * 60 * 60


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(text):
    def reject_duplicates(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"Duplicate JSON key: {key}")
            value[key] = item
        return value

    return json.loads(
        text,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON token: {value}")
        ),
        object_pairs_hook=reject_duplicates,
    )


def write_new(path, value):
    path = Path(path).resolve()
    require(path.parent.is_dir() and not path.parent.is_symlink(),
            "Output parent must be an existing regular directory")
    rendered = json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)


def check_new_outputs(*paths):
    """Precheck a multi-file exclusive write set without creating any member."""
    resolved = [Path(path).resolve() for path in paths]
    require(len(set(resolved)) == len(resolved), "Output paths must be distinct")
    for path in resolved:
        require(path.parent.is_dir() and not path.parent.is_symlink(),
                "Output parent must be an existing regular directory")
        require(not path.exists() and not path.is_symlink(), "Output path is already occupied")


def generation_config(candidate):
    parameters = candidate["method_parameters"]
    return {
        "run_name": candidate["run_name"],
        "run_purpose": "full",
        "save_paired_images": True,
        "require_cuda": True,
        "prompt_file": str(SERVER_REPO / ".local_data/prompts/splits/sdp_pilot_rows_1000_1199.txt"),
        "prompt_split": "pilot",
        "method_name": candidate["method_name"],
        "global_alpha": parameters.get("alpha"),
        "saliency_preview_steps": parameters.get("preview_steps"),
        "saliency_background_strength": parameters.get("background_strength"),
        "saliency_foreground_strength": parameters.get("foreground_strength"),
        "saliency_power": parameters.get("power"),
        "attack_name": "clean",
        "replicate_id": 1,
        "protocol_version": "v1.1-pilot",
        "watermark_key_id": "treering-rand-wseed-999999",
        "start": 0,
        "end": 200,
        "image_length": 512,
        "model_id": str(SERVER_MODEL),
        "num_images": 1,
        "guidance_scale": 7.5,
        "num_inference_steps": 50,
        "test_num_inference_steps": 50,
        "gen_seed": 0,
        "w_seed": 999999,
        "w_channel": 0,
        "w_pattern": "rand",
        "w_mask_shape": "circle",
        "w_radius": 10,
        "w_measurement": "l1_complex",
        "w_injection": "complex",
        "rand_aug": 0,
    }


def command_from_config(config):
    command = [str(SERVER_PYTHON), "-B", str(SERVER_REPO / "run_tree_ring_watermark.py")]
    for key, value in config.items():
        if value is True:
            command.append("--" + key)
        elif value is not None and value is not False:
            command.extend(("--" + key, str(value)))
    return command


def output_paths(repo, run_name):
    root = Path(repo).resolve() / ".local_outputs/runs"
    return [
        root / f"{run_name}.json",
        root / f"{run_name}.samples.jsonl",
        root / f"{run_name}.images",
    ]


def quality_command(candidate, evidence_root):
    run_name = candidate["run_name"]
    source_root = SERVER_REPO / ".local_outputs/runs"
    output = Path(evidence_root) / "quality" / (
        f"{candidate['ordinal']:02d}_{candidate['label']}"
    )
    return [
        str(SERVER_PYTHON), "-B", str(SERVER_REPO / "scripts/quality_validation.py"),
        "--summary", str(source_root / f"{run_name}.json"),
        "--samples", str(source_root / f"{run_name}.samples.jsonl"),
        "--images", str(source_root / f"{run_name}.images"),
        "--output-dir", str(output),
        "--clip-checkpoint", str(SERVER_QUALITY_ASSETS / "open_clip_pytorch_model.bin"),
        "--clip-checkpoint-sha256", "6aac683f899159946bc4ca15228bb7016f3cbb1a2c51f365cba0b23923f344da",
        "--lpips-linear-checkpoint", str(SERVER_QUALITY_ASSETS / "alex.pth"),
        "--lpips-linear-checkpoint-sha256", "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0",
        "--alexnet-checkpoint", str(SERVER_QUALITY_ASSETS / "alexnet-owt-7be5be79.pth"),
        "--alexnet-checkpoint-sha256", "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
        "--device", "cuda",
    ]


def _validate_full_sha(value, name="expected head"):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{40}", value),
            f"Invalid {name}")
    return value


def build_plan(evidence_root=None, expected_head=EXPECTED_HEAD):
    expected_head = _validate_full_sha(expected_head)
    evidence_root = Path(evidence_root or (
        SERVER_TRANSFER / "pilot-gpu-execution_7e8ee07_<UTC-YYYYMMDDTHHMMSSZ>"
    ))
    runs = []
    generation_budget = 0
    for ordinal, candidate in enumerate(candidate_plan(), start=1):
        timeout = GENERATION_TIMEOUTS[candidate["method_name"]]
        generation_budget += timeout
        enriched = {**candidate, "ordinal": ordinal}
        runs.append({
            **enriched,
            "generation_timeout_seconds": timeout,
            "quality_timeout_seconds": QUALITY_TIMEOUT_SECONDS,
            "generation_command": command_from_config(generation_config(candidate)),
            "quality_command": quality_command(enriched, evidence_root),
        })
    quality_budget = len(runs) * QUALITY_TIMEOUT_SECONDS
    reserved_readiness = AGGREGATE_GPU_BUDGET_SECONDS - generation_budget - quality_budget
    require(reserved_readiness == 10 * 60, "Frozen aggregate budget arithmetic changed")
    return {
        "status": "PILOT_GPU_PLAN_PASS",
        "plan_version": PLAN_VERSION,
        "expected_head": expected_head,
        "pilot_freeze_base_head": PILOT_FREEZE_BASE_HEAD,
        "execution_started": False,
        "execution_authorized": False,
        "formal_authorized": False,
        "specification": pilot_specification(),
        "evidence_root_template": str(evidence_root),
        "runs": runs,
        "budget": {
            "generation_seconds": generation_budget,
            "quality_seconds": quality_budget,
            "conditional_readiness_seconds": reserved_readiness,
            "aggregate_gpu_seconds": AGGREGATE_GPU_BUDGET_SECONDS,
        },
    }


def _prompt_check(repo):
    path = Path(repo).resolve() / ".local_data/prompts/splits/sdp_pilot_rows_1000_1199.txt"
    require(path.is_file() and not path.is_symlink(), "Pilot prompt is missing, non-regular, or symlinked")
    require(file_sha256(path) == PILOT_PROMPT_SHA256, "Pilot prompt SHA-256 mismatch")
    lines = path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 200, "Pilot prompt count mismatch")
    return {"path": str(path), "count": len(lines), "sha256": PILOT_PROMPT_SHA256}


def static_preflight(
    repo, transfer_root, evidence_root, *, expected_head=EXPECTED_HEAD,
    min_free_bytes=MIN_FREE_BYTES, git_state=None, disk_usage=None,
):
    """Validate immutable identity and exclusive paths without probing or using GPU."""
    repo = Path(repo).resolve()
    transfer_root = Path(transfer_root).resolve()
    evidence_root = Path(evidence_root).resolve()
    expected_head = _validate_full_sha(expected_head)
    require(repo.is_dir() and not repo.is_symlink(), "Repository path is invalid")
    require(transfer_root.is_dir() and not transfer_root.is_symlink(), "Transfer root is invalid")
    require(evidence_root.parent == transfer_root, "Evidence root must be a direct child of transfer root")
    require(EVIDENCE_NAME.fullmatch(evidence_root.name) is not None, "Evidence root name is not canonical")
    require(not evidence_root.exists(), "Evidence root is already occupied")

    state = git_state if git_state is not None else get_git_state(repo)
    require(state.get("branch") == "stage-c-record-identity", "Pilot branch mismatch")
    require(state.get("commit") == expected_head, "Pilot Git commit mismatch")
    require(state.get("dirty") is False, "Pilot worktree is dirty")
    if git_state is None:
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", PILOT_FREEZE_BASE_HEAD, expected_head],
            cwd=repo, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        require(ancestry.returncode == 0, "Runtime commit does not descend from Pilot freeze")
    prompt = _prompt_check(repo)

    occupied = []
    expected_targets = []
    for candidate in candidate_plan():
        for path in output_paths(repo, candidate["run_name"]):
            expected_targets.append(str(path))
            if path.exists() or path.is_symlink():
                occupied.append(str(path))
    require(not occupied, "Canonical Pilot output target is already occupied")

    usage = disk_usage if disk_usage is not None else shutil.disk_usage(transfer_root)
    require(usage.free >= min_free_bytes, "Insufficient persistent storage headroom")
    plan = build_plan(evidence_root, expected_head)
    return {
        "status": "PILOT_GPU_STATIC_PREFLIGHT_PASS",
        "plan_version": PLAN_VERSION,
        "created_at_utc": utc_now_iso(),
        "expected_head": expected_head,
        "pilot_freeze_base_head": PILOT_FREEZE_BASE_HEAD,
        "git": state,
        "prompt": prompt,
        "evidence_root": str(evidence_root),
        "evidence_root_exists": False,
        "canonical_output_targets": expected_targets,
        "canonical_output_targets_absent": True,
        "storage": {"free_bytes": usage.free, "required_free_bytes": min_free_bytes},
        "run_count": len(plan["runs"]),
        "budget": plan["budget"],
        "live_environment_checked": False,
        "gpu_checked": False,
        "model_assets_checked": False,
        "quality_assets_checked": False,
        "protected_snapshot_checked": False,
        "execution_authorized": False,
        "execution_started": False,
        "formal_authorized": False,
        "next_required_gate": "reviewed live server preflight; this static report cannot authorize execution",
    }


def validate_file_manifest(path, expected_sha256, expected_root):
    """Replay a reviewed path/size/hash manifest without changing its sources."""
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink(), "Reviewed file manifest is invalid")
    require(file_sha256(path) == expected_sha256, "Reviewed file manifest SHA-256 mismatch")
    manifest = strict_json(path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == "treering-file-manifest-v1",
            "File manifest schema mismatch")
    root = Path(manifest.get("root", "")).resolve()
    require(root == Path(expected_root).resolve(), "File manifest root mismatch")
    entries = manifest.get("files")
    require(isinstance(entries, list) and entries, "File manifest must be nonempty")
    seen = set()
    for entry in entries:
        require(isinstance(entry, dict), "Invalid file manifest entry")
        relative = entry.get("path")
        require(isinstance(relative, str) and relative not in seen,
                "Duplicate or invalid manifest path")
        relative_path = Path(relative)
        require(not relative_path.is_absolute() and ".." not in relative_path.parts,
                "Manifest path must be a safe relative path")
        seen.add(relative)
        unresolved = root / relative_path
        require(not unresolved.is_symlink(), "Manifest source must not be symlinked")
        candidate = unresolved.resolve()
        require(candidate.is_relative_to(root), "Manifest path escapes its root")
        require(candidate.is_file(), "Manifest source is missing or non-regular")
        require(candidate.stat().st_size == entry.get("size"), "Manifest file size mismatch")
        require(file_sha256(candidate) == entry.get("sha256"), "Manifest file SHA-256 mismatch")
    return {"manifest_sha256": expected_sha256, "root": str(root), "file_count": len(entries)}


def probe_gpu():
    """Query one CUDA device without loading any model or evaluating an image."""
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=30,
    )
    rows = [row.strip() for row in query.stdout.splitlines() if row.strip()]
    require(len(rows) == 1, "Live preflight requires exactly one GPU")
    parts = [value.strip() for value in rows[0].split(",")]
    require(len(parts) == 4 and parts[0] == "NVIDIA B200", "Live preflight requires NVIDIA B200")
    environment = dict(
        os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1", WANDB_MODE="disabled", PYTHONDONTWRITEBYTECODE="1",
    )
    code = (
        "import json,torch; print(json.dumps({"
        "'torch_version':torch.__version__,'torch_cuda_version':torch.version.cuda,"
        "'cuda_available':torch.cuda.is_available(),'device_count':torch.cuda.device_count(),"
        "'gpu_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
    )
    torch_query = subprocess.run(
        [sys.executable, "-B", "-c", code], check=True, capture_output=True,
        text=True, timeout=60, env=environment,
    )
    torch_state = strict_json(torch_query.stdout)
    require(torch_state.get("cuda_available") is True, "CUDA is unavailable")
    require(torch_state.get("device_count") == 1, "CUDA device count mismatch")
    require(torch_state.get("gpu_name") == "NVIDIA B200", "PyTorch GPU identity mismatch")
    return {
        "nvidia_smi": {
            "name": parts[0], "uuid": parts[1], "driver_version": parts[2],
            "memory_total_mib": int(parts[3]),
        },
        "torch": torch_state,
        "model_loaded": False,
        "image_evaluation_started": False,
    }


def live_preflight(
    repo, transfer_root, evidence_root, *, model_manifest, model_manifest_sha256,
    protected_manifest, protected_manifest_sha256, quality_asset_paths,
    gpu_probe=None, git_state=None, disk_usage=None, model_root=SERVER_MODEL,
    quality_validator=validate_quality_assets, min_free_bytes=MIN_FREE_BYTES,
    expected_head=EXPECTED_HEAD,
):
    """Complete the reviewed live gate without loading diffusion/metric models."""
    static = static_preflight(
        repo, transfer_root, evidence_root, git_state=git_state, disk_usage=disk_usage,
        min_free_bytes=min_free_bytes, expected_head=expected_head,
    )
    model = validate_file_manifest(model_manifest, model_manifest_sha256, model_root)
    protected = validate_file_manifest(protected_manifest, protected_manifest_sha256, repo)
    quality = quality_validator(quality_asset_paths)
    gpu = (gpu_probe or probe_gpu)()
    require(gpu["nvidia_smi"]["name"] == "NVIDIA B200", "GPU probe model mismatch")
    require(gpu["torch"]["cuda_available"] is True, "GPU probe CUDA mismatch")
    require(gpu["torch"]["device_count"] == 1, "GPU probe device-count mismatch")
    require(gpu.get("model_loaded") is False, "Preflight must not load a model")
    require(gpu.get("image_evaluation_started") is False, "Preflight must not evaluate images")
    return {
        **static,
        "status": "PILOT_GPU_LIVE_PREFLIGHT_PASS",
        "model_manifest": model,
        "protected_manifest": protected,
        "quality_assets": quality,
        "gpu": gpu,
        "live_environment_checked": True,
        "gpu_checked": True,
        "model_assets_checked": True,
        "quality_assets_checked": True,
        "protected_snapshot_checked": True,
        "execution_authorized": False,
        "execution_started": False,
        "next_required_gate": "separate explicit user approval for Pilot GPU execution",
    }


def _finite_number(value, name):
    require(not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value),
            f"Non-finite or non-numeric {name}")


def validate_generation_run(
    summary_path, samples_path, images_dir, candidate, *, expected_head=EXPECTED_HEAD,
    image_verifier=verify_image_pair, include_vectors=False,
):
    """Validate one completed run structurally and return hashes, not outcomes."""
    summary_path = Path(summary_path).resolve()
    samples_path = Path(samples_path).resolve()
    images_dir = Path(images_dir).resolve()
    for path in (summary_path, samples_path):
        require(path.is_file() and not path.is_symlink(), "Generation record is invalid")
    require(images_dir.is_dir() and not images_dir.is_symlink(), "Generation image directory is invalid")
    summary = strict_json(summary_path.read_text(encoding="utf-8"))
    lines = samples_path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 200 and all(line.strip() for line in lines),
            "Generation JSONL must contain exactly 200 records")
    rows = [strict_json(line) for line in lines]
    require(summary.get("run_name") == candidate["run_name"], "Generation run-name mismatch")
    require(summary.get("schema_version") == "treering-experiment-v2",
            "Generation schema mismatch")
    require(summary.get("num_samples") == 200 and summary.get("results") == rows,
            "Generation summary/JSONL mismatch")
    git = summary.get("git", {})
    require(git.get("commit") == expected_head and git.get("dirty") is False,
            "Generation Git identity mismatch")
    identity = summary.get("run_identity", {})
    require(identity.get("run_purpose") == "full", "Generation purpose mismatch")
    require(identity.get("sample_range") == {"start": 0, "end_exclusive": 200, "count": 200},
            "Generation range mismatch")
    require(identity.get("prompt_split") == "pilot", "Generation prompt split mismatch")
    require(identity.get("method_name") == candidate["method_name"], "Generation method mismatch")
    require(identity.get("method_parameters") == candidate["method_parameters"],
            "Generation parameters mismatch")
    for key, value in {
        "attack_name": "clean", "attack_parameters": {}, "replicate_id": 1,
        "protocol_version": "v1.1-pilot",
        "watermark_key_id": "treering-rand-wseed-999999",
    }.items():
        require(identity.get(key) == value, f"Generation identity mismatch: {key}")
    prompt = summary.get("prompt_source", {})
    require(prompt.get("sha256") == PILOT_PROMPT_SHA256 and prompt.get("nonempty_line_count") == 200,
            "Generation prompt provenance mismatch")
    environment = summary.get("environment", {})
    require(environment.get("cuda_available") is True and environment.get("device") == "cuda" and
            environment.get("gpu_name") == "NVIDIA B200" and
            environment.get("torch_dtype") == "torch.float16",
            "Generation GPU provenance mismatch")
    parameters = summary.get("parameters", {})
    expected_parameters = generation_config(candidate)
    for key, value in expected_parameters.items():
        if key in {"prompt_file", "model_id"}:
            continue
        require(parameters.get(key) == value, f"Generation parameter mismatch: {key}")
    require(Path(parameters.get("prompt_file", "")).name == "sdp_pilot_rows_1000_1199.txt",
            "Generation prompt path mismatch")
    require(Path(parameters.get("model_id", "")).name == SERVER_MODEL.name,
            "Generation model path mismatch")

    expected_pngs = set()
    no_watermark_hashes = []
    watermarked_hashes = []
    no_watermark_distances = []
    watermarked_distances = []
    residual_rms = []
    residual_l2 = []
    paired_inputs = []
    all_hashes = {}
    for index, row in enumerate(rows):
        require(row.get("sample_index") == index and row.get("source_row_id") == 1000 + index,
                "Generation sample/source-row mismatch")
        require(row.get("sample_id") == f"{candidate['run_name']}:{index}",
                "Generation sample-id mismatch")
        require(row.get("prompt_split") == "pilot" and row.get("method_name") == candidate["method_name"],
                "Generation row identity mismatch")
        require(row.get("run_purpose") == "full" and
                row.get("run_naming_version") == "treering-run-name-v2",
                "Generation row naming identity mismatch")
        require(row.get("method_parameters") == candidate["method_parameters"] and
                row.get("attack_name") == "clean" and row.get("attack_parameters") == {},
                "Generation row method/attack mismatch")
        require(row.get("replicate_id") == 1 and row.get("protocol_version") == "v1.1-pilot" and
                row.get("watermark_key_id") == "treering-rand-wseed-999999",
                "Generation row protocol identity mismatch")
        require(row.get("generation_seed") == index and row.get("watermark_seed") == 999999,
                "Generation seed mismatch")
        require(isinstance(row.get("prompt"), str) and row["prompt"], "Generation prompt is empty")
        for field in (
            "latent_residual_rms", "latent_residual_l2", "no_w_metric", "w_metric",
            "no_w_score", "w_score", "w_no_sim", "w_sim", "sample_duration_seconds",
        ):
            _finite_number(row.get(field), field)
        artifacts = row.get("image_artifacts")
        hashes = image_verifier(images_dir, index, artifacts, expected_size=512)
        all_hashes.update(hashes)
        no_watermark_hashes.append(artifacts["no_watermark"]["sha256"])
        watermarked_hashes.append(artifacts["watermarked"]["sha256"])
        no_watermark_distances.append(float(row["no_w_metric"]))
        watermarked_distances.append(float(row["w_metric"]))
        residual_rms.append(float(row["latent_residual_rms"]))
        residual_l2.append(float(row["latent_residual_l2"]))
        paired = row.get("paired_inputs")
        require(isinstance(paired, dict) and set(paired) == {
            "base_latent_sha256", "watermark_key_sha256", "watermark_mask_sha256"
        }, "Generation paired-input identity mismatch")
        require(all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in paired.values()), "Generation paired-input hash is invalid")
        paired_inputs.append(paired)
        expected_pngs.update((f"{index:04d}.no_watermark.png", f"{index:04d}.watermarked.png"))
    image_members = list(images_dir.iterdir())
    require(all(path.is_file() and not path.is_symlink() for path in image_members),
            "Generation image directory contains a non-regular or symlinked member")
    actual_pngs = {path.name for path in image_members}
    require(actual_pngs == expected_pngs, "Generation PNG member set mismatch")
    vector_digest = hashlib.sha256(
        json.dumps(no_watermark_distances, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    report = {
        "status": "PILOT_GENERATION_STRUCTURE_PASS",
        "label": candidate["label"],
        "run_name": candidate["run_name"],
        "record_count": 200,
        "png_count": 400,
        "summary_sha256": file_sha256(summary_path),
        "samples_sha256": file_sha256(samples_path),
        "png_manifest_sha256": hashlib.sha256(
            json.dumps(all_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "no_watermark_image_sha256": no_watermark_hashes,
        "watermarked_image_sha256": watermarked_hashes,
        "no_watermark_distance_vector_sha256": vector_digest,
        "paired_inputs": paired_inputs,
        "outcomes_exposed": False,
        "formal_authorized": False,
    }
    if include_vectors:
        report["_vectors"] = {
            "no_watermark_distances": no_watermark_distances,
            "watermarked_distances": watermarked_distances,
            "latent_residual_rms": residual_rms,
            "latent_residual_l2": residual_l2,
        }
    return report


def accept_generation_runs(run_reports):
    """Accept the exact ten structural reports and shared paired inputs."""
    plan = candidate_plan()
    require(isinstance(run_reports, list) and len(run_reports) == len(plan),
            "Generation acceptance requires exactly ten reports")
    canonical = run_reports[0]
    for expected, report in zip(plan, run_reports):
        require(report.get("status") == "PILOT_GENERATION_STRUCTURE_PASS",
                "Generation structure report did not pass")
        require(report.get("label") == expected["label"] and
                report.get("run_name") == expected["run_name"],
                "Generation report order/identity mismatch")
        require(report.get("record_count") == 200 and report.get("png_count") == 400,
                "Generation report count mismatch")
        require(report.get("no_watermark_image_sha256") == canonical["no_watermark_image_sha256"],
                "No-Watermark PNG hashes differ across candidates")
        require(report.get("no_watermark_distance_vector_sha256") ==
                canonical["no_watermark_distance_vector_sha256"],
                "No-Watermark distances differ across candidates")
        require(report.get("paired_inputs") == canonical["paired_inputs"],
                "Paired-input identities differ across candidates")
        require(report.get("outcomes_exposed") is False and report.get("formal_authorized") is False,
                "Generation report crossed the outcome/Formal boundary")
        for key in ("summary_sha256", "samples_sha256", "png_manifest_sha256"):
            require(isinstance(report.get(key), str) and
                    re.fullmatch(r"[0-9a-f]{64}", report[key]),
                    "Generation evidence SHA-256 is invalid")
    evidence = {
        report["label"]: {
            key: report[key] for key in (
                "summary_sha256", "samples_sha256", "png_manifest_sha256"
            )
        }
        for report in run_reports
    }
    return {
        "status": "PILOT_GENERATION_ACCEPT_PASS",
        "run_count": 10,
        "record_count_per_run": 200,
        "png_count_per_run": 400,
        "shared_no_watermark_pngs": True,
        "shared_no_watermark_distances": True,
        "shared_paired_inputs": True,
        "evidence": evidence,
        "outcomes_exposed": False,
        "execution_started_by_acceptance": False,
        "formal_authorized": False,
    }


def validate_quality_run(
    quality_summary_path, quality_samples_path, candidate, generation_report,
    *, expected_head=EXPECTED_HEAD, include_vectors=False,
):
    """Validate one frozen quality result and expose outcomes only when requested."""
    summary_path = Path(quality_summary_path).resolve()
    samples_path = Path(quality_samples_path).resolve()
    for path in (summary_path, samples_path):
        require(path.is_file() and not path.is_symlink(), "Quality record is invalid")
    summary = strict_json(summary_path.read_text(encoding="utf-8"))
    lines = samples_path.read_text(encoding="utf-8").splitlines()
    require(len(lines) == 200 and all(line.strip() for line in lines),
            "Quality JSONL must contain exactly 200 records")
    rows = [strict_json(line) for line in lines]
    require(summary.get("schema_version") == "treering-quality-evaluation-v1" and
            summary.get("metric_spec_version") == "treering-quality-v1",
            "Quality schema/specification mismatch")
    require(summary.get("results") == rows, "Quality summary/JSONL mismatch")
    git = summary.get("git", {})
    require(git.get("commit") == expected_head and git.get("dirty") is False,
            "Quality Git identity mismatch")
    source = summary.get("source_run", {})
    require(source.get("run_name") == candidate["run_name"] and source.get("sample_count") == 200,
            "Quality source-run identity mismatch")
    require(source.get("summary_sha256") == generation_report.get("summary_sha256") and
            source.get("samples_sha256") == generation_report.get("samples_sha256"),
            "Quality source hashes do not bind the generation records")
    metrics = summary.get("metrics", {})
    clip = metrics.get("clip", {})
    lpips = metrics.get("lpips", {})
    require(clip.get("model_name") == "ViT-g-14" and clip.get("device") == "cuda" and
            clip.get("dtype") == "float32" and
            clip.get("checkpoint_sha256") ==
            "6aac683f899159946bc4ca15228bb7016f3cbb1a2c51f365cba0b23923f344da",
            "Quality CLIP identity mismatch")
    require(lpips.get("version") == "treering-lpips-alex-v0.1" and
            lpips.get("device") == "cuda" and lpips.get("dtype") == "float32" and
            lpips.get("linear_checkpoint_sha256") ==
            "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0" and
            lpips.get("alexnet_checkpoint_sha256") ==
            "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02",
            "Quality LPIPS identity mismatch")
    require(metrics.get("fid", {}).get("status") == "omitted_by_approved_protocol_decision",
            "Quality FID boundary mismatch")

    vectors = {name: [] for name in (
        "clip_no_watermark", "clip_delta", "lpips", "psnr_db", "ssim"
    )}
    expected_no_w = generation_report.get("no_watermark_image_sha256")
    expected_w = generation_report.get("watermarked_image_sha256")
    require(isinstance(expected_no_w, list) and isinstance(expected_w, list) and
            len(expected_no_w) == len(expected_w) == 200,
            "Generation image hashes are incomplete")
    for index, row in enumerate(rows):
        require(row.get("sample_id") == f"{candidate['run_name']}:{index}" and
                row.get("sample_index") == index and row.get("source_row_id") == 1000 + index,
                "Quality sample identity mismatch")
        require(row.get("prompt_split") == "pilot" and
                row.get("method_name") == candidate["method_name"] and
                row.get("replicate_id") == 1 and row.get("protocol_version") == "v1.1-pilot" and
                row.get("attack_name") == "clean" and
                row.get("watermark_key_id") == "treering-rand-wseed-999999",
                "Quality row protocol identity mismatch")
        require(row.get("generation_seed") == index and row.get("watermark_seed") == 999999,
                "Quality row seed mismatch")
        image_hashes = row.get("source_image_sha256", {})
        require(image_hashes == {
            "no_watermark": expected_no_w[index], "watermarked": expected_w[index]
        }, "Quality source-image hashes mismatch")
        values = row.get("quality_metrics", {})
        for field in (
            "clip_text_image_cosine_no_watermark",
            "clip_text_image_cosine_watermarked",
            "clip_text_image_cosine_delta", "lpips_alex_v0_1", "ssim_rgb",
        ):
            _finite_number(values.get(field), field)
        require(math.isclose(
            values["clip_text_image_cosine_watermarked"] -
            values["clip_text_image_cosine_no_watermark"],
            values["clip_text_image_cosine_delta"], rel_tol=0.0, abs_tol=1e-12,
        ), "Quality CLIP delta mismatch")
        psnr = values.get("psnr_rgb", {})
        require(psnr.get("status") == "finite", "Pilot PSNR must be finite")
        _finite_number(psnr.get("value_db"), "psnr_rgb.value_db")
        _finite_number(psnr.get("mse"), "psnr_rgb.mse")
        vectors["clip_no_watermark"].append(float(values["clip_text_image_cosine_no_watermark"]))
        vectors["clip_delta"].append(float(values["clip_text_image_cosine_delta"]))
        vectors["lpips"].append(float(values["lpips_alex_v0_1"]))
        vectors["psnr_db"].append(float(psnr["value_db"]))
        vectors["ssim"].append(float(values["ssim_rgb"]))
    report = {
        "status": "PILOT_QUALITY_STRUCTURE_PASS",
        "label": candidate["label"],
        "run_name": candidate["run_name"],
        "record_count": 200,
        "quality_summary_sha256": file_sha256(summary_path),
        "quality_samples_sha256": file_sha256(samples_path),
        "source_generation_summary_sha256": generation_report["summary_sha256"],
        "source_generation_samples_sha256": generation_report["samples_sha256"],
        "clip_no_watermark_vector_sha256": hashlib.sha256(
            json.dumps(vectors["clip_no_watermark"], separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "outcomes_exposed": False,
        "formal_authorized": False,
    }
    if include_vectors:
        report["_vectors"] = vectors
    return report


def build_observation_document(
    generation_reports, quality_reports, *, expected_head=EXPECTED_HEAD,
):
    """Assemble the exact frozen observation schema after all structural gates pass."""
    plan = candidate_plan()
    expected_head = _validate_full_sha(expected_head)
    require(len(generation_reports) == len(quality_reports) == len(plan),
            "Observation assembly requires exact ten-run inputs")
    canonical_no_w = generation_reports[0]["_vectors"]["no_watermark_distances"]
    canonical_hashes = generation_reports[0]["no_watermark_image_sha256"]
    canonical_clip = quality_reports[0]["_vectors"]["clip_no_watermark"]
    candidates = []
    for expected, generation, quality in zip(plan, generation_reports, quality_reports):
        require(generation.get("status") == "PILOT_GENERATION_STRUCTURE_PASS" and
                quality.get("status") == "PILOT_QUALITY_STRUCTURE_PASS",
                "Observation input did not pass structural acceptance")
        require(generation.get("label") == quality.get("label") == expected["label"],
                "Observation candidate order/identity mismatch")
        require(generation["_vectors"]["no_watermark_distances"] == canonical_no_w and
                generation["no_watermark_image_sha256"] == canonical_hashes,
                "Observation No-Watermark generation mismatch")
        require(quality["_vectors"]["clip_no_watermark"] == canonical_clip,
                "Observation No-Watermark CLIP mismatch")
        candidates.append({
            "label": expected["label"],
            "run_name": expected["run_name"],
            "method_name": expected["method_name"],
            "method_parameters": expected["method_parameters"],
            "git_commit": expected_head,
            "git_dirty": False,
            "source_row_ids": list(PILOT_SOURCE_ROW_IDS),
            "no_watermark_distances": list(canonical_no_w),
            "no_watermark_image_sha256": list(canonical_hashes),
            "watermarked_image_sha256": list(generation["watermarked_image_sha256"]),
            "watermarked_distances": list(generation["_vectors"]["watermarked_distances"]),
            "clip_delta": list(quality["_vectors"]["clip_delta"]),
            "lpips": list(quality["_vectors"]["lpips"]),
            "psnr_db": list(quality["_vectors"]["psnr_db"]),
            "ssim": list(quality["_vectors"]["ssim"]),
            "latent_residual_rms": list(generation["_vectors"]["latent_residual_rms"]),
            "latent_residual_l2": list(generation["_vectors"]["latent_residual_l2"]),
        })
    return {
        "schema_version": PILOT_OBSERVATION_SCHEMA,
        "prompt_sha256": PILOT_PROMPT_SHA256,
        "implementation_commit": expected_head,
        "source_row_ids": list(PILOT_SOURCE_ROW_IDS),
        "no_watermark_distances": list(canonical_no_w),
        "no_watermark_image_sha256": list(canonical_hashes),
        "original": candidates[0],
        "global_candidates": candidates[1:4],
        "saliency_candidates": candidates[4:],
    }


def final_selection_accept(observations, generation_accept_report, assembly_report):
    require(generation_accept_report.get("status") == "PILOT_GENERATION_ACCEPT_PASS",
            "Final selection requires generation acceptance")
    require(generation_accept_report.get("outcomes_exposed") is False and
            generation_accept_report.get("formal_authorized") is False,
            "Generation acceptance boundary mismatch")
    observation_sha256 = hashlib.sha256(
        json.dumps(observations, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")
    ).hexdigest()
    require(assembly_report.get("status") == "PILOT_OBSERVATIONS_ASSEMBLED_PASS" and
            assembly_report.get("observation_semantic_sha256") == observation_sha256 and
            assembly_report.get("formal_authorized") is False,
            "Observation assembly evidence mismatch")
    decision = select_pilot(observations)
    require(decision.get("formal_authorized") is False, "Selection cannot authorize Formal")
    return {
        "status": "PILOT_GPU_ACCEPT_AND_SELECTION_PASS",
        "observation_sha256": observation_sha256,
        "decision": decision,
        "all_candidates_reported": decision.get("all_candidates_reported") is True,
        "formal_authorized": False,
    }


def assemble_from_source_manifest(path):
    """Revalidate ten raw generation/quality sources named by one strict manifest."""
    path = Path(path).resolve()
    require(path.is_file() and not path.is_symlink(), "Source manifest is invalid")
    manifest = strict_json(path.read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == "treering-pilot-source-manifest-v1",
            "Source manifest schema mismatch")
    sources = manifest.get("runs")
    expected_head = _validate_full_sha(manifest.get("expected_head"))
    require(manifest.get("pilot_freeze_base_head") == PILOT_FREEZE_BASE_HEAD,
            "Source manifest Pilot freeze base mismatch")
    plan = candidate_plan()
    require(isinstance(sources, list) and len(sources) == len(plan),
            "Source manifest requires exactly ten runs")
    generation_reports = []
    quality_reports = []
    for expected, source in zip(plan, sources):
        require(isinstance(source, dict) and source.get("label") == expected["label"],
                "Source manifest order/identity mismatch")
        generation = validate_generation_run(
            source.get("generation_summary"), source.get("generation_samples"),
            source.get("generation_images"), expected, include_vectors=True,
            expected_head=expected_head,
        )
        quality = validate_quality_run(
            source.get("quality_summary"), source.get("quality_samples"),
            expected, generation, include_vectors=True, expected_head=expected_head,
        )
        generation_reports.append(generation)
        quality_reports.append(quality)
    accept_generation_runs(generation_reports)
    return build_observation_document(
        generation_reports, quality_reports, expected_head=expected_head,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Outcome-blind Pilot GPU plan and static preflight; no execution mode exists"
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--output")
    static = subparsers.add_parser("static-preflight")
    static.add_argument("--repo", default=str(REPO))
    static.add_argument("--transfer-root", required=True)
    static.add_argument("--evidence-root", required=True)
    static.add_argument("--expected-head", required=True)
    static.add_argument("--output", required=True)
    live = subparsers.add_parser("live-preflight")
    live.add_argument("--repo", default=str(REPO))
    live.add_argument("--transfer-root", required=True)
    live.add_argument("--evidence-root", required=True)
    live.add_argument("--expected-head", required=True)
    live.add_argument("--model-manifest", required=True)
    live.add_argument("--model-manifest-sha256", required=True)
    live.add_argument("--protected-manifest", required=True)
    live.add_argument("--protected-manifest-sha256", required=True)
    live.add_argument("--clip-checkpoint", required=True)
    live.add_argument("--lpips-linear-checkpoint", required=True)
    live.add_argument("--alexnet-checkpoint", required=True)
    inspect = subparsers.add_parser("inspect-generation")
    inspect.add_argument("--ordinal", required=True, type=int)
    inspect.add_argument("--expected-head", required=True)
    inspect.add_argument("--summary", required=True)
    inspect.add_argument("--samples", required=True)
    inspect.add_argument("--images", required=True)
    inspect.add_argument("--output", required=True)
    accept = subparsers.add_parser("accept-generation")
    accept.add_argument("--run-report", action="append", required=True)
    accept.add_argument("--output", required=True)
    quality = subparsers.add_parser("inspect-quality")
    quality.add_argument("--ordinal", required=True, type=int)
    quality.add_argument("--expected-head", required=True)
    quality.add_argument("--quality-summary", required=True)
    quality.add_argument("--quality-samples", required=True)
    quality.add_argument("--generation-report", required=True)
    quality.add_argument("--output", required=True)
    assemble = subparsers.add_parser("assemble-observations")
    assemble.add_argument("--source-manifest", required=True)
    assemble.add_argument("--output", required=True)
    assemble.add_argument("--report", required=True)
    final = subparsers.add_parser("final-accept")
    final.add_argument("--observations", required=True)
    final.add_argument("--assembly-report", required=True)
    final.add_argument("--generation-accept-report", required=True)
    final.add_argument("--decision-output", required=True)
    final.add_argument("--output", required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "plan":
        report = build_plan()
        if args.output:
            write_new(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))
        return
    if args.mode == "static-preflight":
        report = static_preflight(
            args.repo, args.transfer_root, args.evidence_root,
            expected_head=args.expected_head,
        )
        write_new(args.output, report)
        print(report["status"])
        return
    if args.mode == "live-preflight":
        report = live_preflight(
            args.repo, args.transfer_root, args.evidence_root,
            model_manifest=args.model_manifest,
            model_manifest_sha256=args.model_manifest_sha256,
            protected_manifest=args.protected_manifest,
            protected_manifest_sha256=args.protected_manifest_sha256,
            quality_asset_paths={
                "clip_vit_g_14_laion2b_s12b_b42k": args.clip_checkpoint,
                "lpips_v0_1_alex_linear": args.lpips_linear_checkpoint,
                "torchvision_alexnet_imagenet1k_v1": args.alexnet_checkpoint,
            },
            expected_head=args.expected_head,
        )
        evidence_root = Path(args.evidence_root).resolve()
        evidence_root.mkdir()
        preflight_dir = evidence_root / "preflight"
        preflight_dir.mkdir()
        write_new(preflight_dir / "live-preflight.json", report)
        print(report["status"])
        return
    if args.mode == "inspect-generation":
        require(1 <= args.ordinal <= 10, "Generation ordinal must be 1-10")
        report = validate_generation_run(
            args.summary, args.samples, args.images, candidate_plan()[args.ordinal - 1],
            expected_head=_validate_full_sha(args.expected_head),
        )
        write_new(args.output, report)
        print(report["status"])
        return
    if args.mode == "accept-generation":
        reports = []
        for path in args.run_report:
            source = Path(path).resolve()
            require(source.is_file() and not source.is_symlink(), "Generation report path is invalid")
            reports.append(strict_json(source.read_text(encoding="utf-8")))
        report = accept_generation_runs(reports)
        write_new(args.output, report)
        print(report["status"])
        return
    if args.mode == "inspect-quality":
        require(1 <= args.ordinal <= 10, "Quality ordinal must be 1-10")
        source = Path(args.generation_report).resolve()
        require(source.is_file() and not source.is_symlink(), "Generation report path is invalid")
        generation = strict_json(source.read_text(encoding="utf-8"))
        report = validate_quality_run(
            args.quality_summary, args.quality_samples,
            candidate_plan()[args.ordinal - 1], generation,
            expected_head=_validate_full_sha(args.expected_head),
        )
        write_new(args.output, report)
        print(report["status"])
        return
    if args.mode == "assemble-observations":
        observations = assemble_from_source_manifest(args.source_manifest)
        check_new_outputs(args.output, args.report)
        write_new(args.output, observations)
        report = {
            "status": "PILOT_OBSERVATIONS_ASSEMBLED_PASS",
            "source_manifest_sha256": file_sha256(args.source_manifest),
            "observation_sha256": file_sha256(args.output),
            "observation_semantic_sha256": hashlib.sha256(
                json.dumps(observations, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8")
            ).hexdigest(),
            "candidate_count": 10,
            "formal_authorized": False,
        }
        write_new(args.report, report)
        print(report["status"])
        return
    observations_path = Path(args.observations).resolve()
    assembly_path = Path(args.assembly_report).resolve()
    generation_path = Path(args.generation_accept_report).resolve()
    for source in (observations_path, assembly_path, generation_path):
        require(source.is_file() and not source.is_symlink(), "Final acceptance input is invalid")
    observations = strict_json(observations_path.read_text(encoding="utf-8"))
    assembly = strict_json(assembly_path.read_text(encoding="utf-8"))
    generation = strict_json(generation_path.read_text(encoding="utf-8"))
    require(assembly.get("observation_sha256") == file_sha256(observations_path),
            "Observation file changed after assembly")
    report = final_selection_accept(observations, generation, assembly)
    check_new_outputs(args.decision_output, args.output)
    write_new(args.decision_output, report["decision"])
    report = {**report, "decision_sha256": file_sha256(args.decision_output)}
    del report["decision"]
    write_new(args.output, report)
    print(report["status"])


if __name__ == "__main__":
    main()
