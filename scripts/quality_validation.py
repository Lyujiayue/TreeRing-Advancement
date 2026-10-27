"""Read-only Stage F evaluation over preserved paired PNG artifacts."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import sys
from pathlib import Path

from PIL import Image


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiment_records import get_git_state, utc_now_iso
from image_artifacts import verify_image_pair
from quality_metrics import (
    CLIP_MODEL_NAME,
    QUALITY_METRIC_SPEC_VERSION,
    ClipScorer,
    LpipsScorer,
    file_sha256,
    load_rgb_float64,
    psnr_rgb,
    psnr_summary,
    scalar_summary,
    ssim_rgb,
)


IDENTITY_FIELDS = (
    "sample_id", "sample_index", "source_row_id", "prompt_split",
    "method_name", "replicate_id", "protocol_version", "attack_name",
    "watermark_key_id", "prompt", "generation_seed", "watermark_seed",
)


def strict_json(text):
    return json.loads(text, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Non-finite JSON token: {value}")
    ))


def _write_new(path, text):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_inputs(summary_path, samples_path, images_dir):
    summary_path = Path(summary_path)
    samples_path = Path(samples_path)
    images_dir = Path(images_dir)
    for path in (summary_path, samples_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing/non-regular input: {path}")
    if images_dir.is_symlink() or not images_dir.is_dir():
        raise ValueError(f"Missing/non-directory image input: {images_dir}")

    summary = strict_json(summary_path.read_text(encoding="utf-8"))
    lines = samples_path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("Samples JSONL is empty or contains blank rows")
    rows = [strict_json(line) for line in lines]
    if summary.get("results") != rows or summary.get("num_samples") != len(rows):
        raise ValueError("Summary and JSONL records disagree")

    indices = [row.get("sample_index") for row in rows]
    if len(set(indices)) != len(rows) or any(type(index) is not int for index in indices):
        raise ValueError("Sample indices are missing, duplicated, or invalid")
    expected_names = {
        f"{index:04d}.{role}.png"
        for index in indices for role in ("no_watermark", "watermarked")
    }
    if {path.name for path in images_dir.iterdir()} != expected_names:
        raise ValueError("Image directory contains missing or unexpected entries")
    for row in rows:
        if any(field not in row for field in IDENTITY_FIELDS):
            raise ValueError("A required sample identity field is missing")
        verify_image_pair(images_dir, row["sample_index"], row.get("image_artifacts"))
    return summary, rows


def evaluate(summary_path, samples_path, images_dir, output_dir, clip_scorer, lpips_scorer):
    summary_path = Path(summary_path).resolve()
    samples_path = Path(samples_path).resolve()
    images_dir = Path(images_dir).resolve()
    output_dir = Path(output_dir).resolve()
    source_summary, rows = validate_inputs(summary_path, samples_path, images_dir)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    quality_rows = []
    try:
        for row in rows:
            index = row["sample_index"]
            no_path = images_dir / f"{index:04d}.no_watermark.png"
            watermarked_path = images_dir / f"{index:04d}.watermarked.png"
            no_array = load_rgb_float64(no_path)
            watermarked_array = load_rgb_float64(watermarked_path)
            with Image.open(no_path) as no_image, Image.open(watermarked_path) as watermarked_image:
                no_image.load()
                watermarked_image.load()
                clip_no, clip_w = clip_scorer.score_pair(no_image, watermarked_image, row["prompt"])
            lpips_value = lpips_scorer.score(no_array, watermarked_array)
            psnr = psnr_rgb(no_array, watermarked_array)
            ssim = ssim_rgb(no_array, watermarked_array)
            numeric = [clip_no, clip_w, clip_w - clip_no, lpips_value, ssim, psnr["mse"]]
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError("A quality result is non-finite")
            quality_rows.append({
                **{field: row[field] for field in IDENTITY_FIELDS},
                "source_image_sha256": {
                    "no_watermark": file_sha256(no_path),
                    "watermarked": file_sha256(watermarked_path),
                },
                "quality_metrics": {
                    "clip_text_image_cosine_no_watermark": clip_no,
                    "clip_text_image_cosine_watermarked": clip_w,
                    "clip_text_image_cosine_delta": clip_w - clip_no,
                    "lpips_alex_v0_1": lpips_value,
                    "psnr_rgb": psnr,
                    "ssim_rgb": ssim,
                },
            })

        def values(name):
            return [row["quality_metrics"][name] for row in quality_rows]

        aggregate = {
            "clip_text_image_cosine_no_watermark": scalar_summary(values("clip_text_image_cosine_no_watermark")),
            "clip_text_image_cosine_watermarked": scalar_summary(values("clip_text_image_cosine_watermarked")),
            "clip_text_image_cosine_delta": scalar_summary(values("clip_text_image_cosine_delta")),
            "lpips_alex_v0_1": scalar_summary(values("lpips_alex_v0_1")),
            "psnr_rgb": psnr_summary(values("psnr_rgb")),
            "ssim_rgb": scalar_summary(values("ssim_rgb")),
        }
        report = {
            "schema_version": "treering-quality-evaluation-v1",
            "metric_spec_version": QUALITY_METRIC_SPEC_VERSION,
            "created_at_utc": utc_now_iso(),
            "git": get_git_state(REPO),
            "source_run": {
                "run_name": source_summary.get("run_name"),
                "summary_path": str(summary_path),
                "summary_sha256": file_sha256(summary_path),
                "samples_path": str(samples_path),
                "samples_sha256": file_sha256(samples_path),
                "images_path": str(images_dir),
                "sample_count": len(rows),
            },
            "metrics": {
                "clip": clip_scorer.provenance,
                "lpips": lpips_scorer.provenance,
                "psnr": {
                    "input": "preserved RGB PNG pixels converted to float64 [0,1]",
                    "formula": "10*log10(1/MSE); exact matches use a null value plus an explicit status",
                },
                "ssim": {
                    "implementation": "quality_metrics.ssim_rgb",
                    "input": "preserved RGB PNG pixels converted to float64 [0,1]",
                    "window": "Gaussian 11x11, sigma=1.5, truncate=3.5, reflect padding; valid-region mean",
                    "constants": {"K1": 0.01, "K2": 0.03, "data_range": 1.0},
                    "channel_aggregation": "arithmetic mean of three RGB channel scores",
                },
                "fid": {
                    "status": "omitted_by_approved_protocol_decision",
                    "reason": "No pre-frozen, verifiable 5,000-image MS-COCO reference set",
                },
            },
            "packages": {
                name: _package_version(name)
                for name in (
                    "numpy", "scipy", "Pillow", "torch", "torchvision",
                    "timm", "ftfy", "regex", "huggingface-hub",
                )
            },
            "aggregation": aggregate,
            "results": quality_rows,
        }
        samples_output = output_dir / "quality.samples.jsonl"
        summary_output = output_dir / "quality.summary.json"
        _write_new(samples_output, "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in quality_rows))
        _write_new(summary_output, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        manifest = {
            samples_output.name: file_sha256(samples_output),
            summary_output.name: file_sha256(summary_output),
        }
        _write_new(output_dir / "SHA256SUMS.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return report
    except Exception:
        # Preserve partial evidence and the exclusive output directory for diagnosis.
        raise


def main():
    parser = argparse.ArgumentParser(description="Offline Stage F quality evaluation")
    parser.add_argument("--summary", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clip-model", default=CLIP_MODEL_NAME, choices=(CLIP_MODEL_NAME,))
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--clip-checkpoint-sha256", required=True)
    parser.add_argument("--lpips-linear-checkpoint", required=True)
    parser.add_argument("--lpips-linear-checkpoint-sha256", required=True)
    parser.add_argument("--alexnet-checkpoint", required=True)
    parser.add_argument("--alexnet-checkpoint-sha256", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    clip = ClipScorer(
        args.clip_model, args.clip_checkpoint, args.device,
        expected_sha256=args.clip_checkpoint_sha256,
    )
    lpips = LpipsScorer(
        args.lpips_linear_checkpoint, args.alexnet_checkpoint, args.device,
        expected_linear_sha256=args.lpips_linear_checkpoint_sha256,
        expected_alexnet_sha256=args.alexnet_checkpoint_sha256,
    )
    evaluate(args.summary, args.samples, args.images, args.output_dir, clip, lpips)
    print("STAGE_F_QUALITY_EVALUATION_PASS")


if __name__ == "__main__":
    main()
