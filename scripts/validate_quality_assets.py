"""Fail-closed validation for the three frozen Stage F model assets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from quality_metrics import (
    ALEXNET_CHECKPOINT_SHA256,
    CLIP_CHECKPOINT_SHA256,
    LPIPS_LINEAR_CHECKPOINT_SHA256,
    file_sha256,
)


ASSETS = {
    "clip_vit_g_14_laion2b_s12b_b42k": {
        "size": 5467006745,
        "sha256": CLIP_CHECKPOINT_SHA256,
        "source": "huggingface.co/laion/CLIP-ViT-g-14-laion2B-s12B-b42K@923c8d71576624933b27ba2cdea739f192ddef48",
    },
    "lpips_v0_1_alex_linear": {
        "size": 6009,
        "sha256": LPIPS_LINEAR_CHECKPOINT_SHA256,
        "source": "github.com/richzhang/PerceptualSimilarity@c33f89e9f46522a584cf41d8880eb0afa982708b",
    },
    "torchvision_alexnet_imagenet1k_v1": {
        "size": 244408911,
        "sha256": ALEXNET_CHECKPOINT_SHA256,
        "source": "download.pytorch.org/models/alexnet-owt-7be5be79.pth",
    },
}


def validate(paths):
    results = {}
    for name, path in paths.items():
        path = Path(path).resolve()
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing, non-regular, or symlinked asset: {path}")
        expected = ASSETS[name]
        size = path.stat().st_size
        digest = file_sha256(path)
        if size != expected["size"]:
            raise ValueError(f"Asset size mismatch for {name}")
        if digest != expected["sha256"]:
            raise ValueError(f"Asset SHA-256 mismatch for {name}")
        results[name] = {
            **expected,
            "path": str(path),
        }
    return {"status": "STAGE_F_QUALITY_ASSETS_PASS", "assets": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-checkpoint", required=True)
    parser.add_argument("--lpips-linear-checkpoint", required=True)
    parser.add_argument("--alexnet-checkpoint", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = validate({
        "clip_vit_g_14_laion2b_s12b_b42k": args.clip_checkpoint,
        "lpips_v0_1_alex_linear": args.lpips_linear_checkpoint,
        "torchvision_alexnet_imagenet1k_v1": args.alexnet_checkpoint,
    })
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)


if __name__ == "__main__":
    main()
