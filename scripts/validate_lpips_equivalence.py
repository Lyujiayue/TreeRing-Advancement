"""Compare the frozen LPIPS implementation against an official wheel offline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from frozen_lpips import create_frozen_lpips_alex
from quality_metrics import _require_sha256, file_sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-wheel", required=True)
    parser.add_argument("--linear-checkpoint", required=True)
    parser.add_argument("--linear-checkpoint-sha256", required=True)
    parser.add_argument("--alexnet-checkpoint", required=True)
    parser.add_argument("--alexnet-checkpoint-sha256", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--output")
    args = parser.parse_args()

    wheel = Path(args.official_wheel).resolve()
    linear = Path(args.linear_checkpoint).resolve()
    alexnet_checkpoint = Path(args.alexnet_checkpoint).resolve()
    if wheel.is_symlink() or not wheel.is_file():
        raise ValueError("Official LPIPS wheel must be an existing regular file")
    _require_sha256(linear, args.linear_checkpoint_sha256)
    _require_sha256(alexnet_checkpoint, args.alexnet_checkpoint_sha256)

    import torch
    import torchvision.models as tv

    original_alexnet = tv.alexnet

    def explicit_alexnet(*_args, **_kwargs):
        model = original_alexnet(weights=None)
        try:
            state = torch.load(alexnet_checkpoint, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(alexnet_checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
        return model

    tv.alexnet = explicit_alexnet
    sys.path.insert(0, str(wheel))
    try:
        import lpips

        official = lpips.LPIPS(
            net="alex", version="0.1", lpips=True, spatial=False,
            pretrained=True, pnet_rand=False, model_path=str(linear),
            eval_mode=True, verbose=False,
        ).eval().float()
    finally:
        tv.alexnet = original_alexnet

    frozen = create_frozen_lpips_alex(linear, alexnet_checkpoint).eval().float()
    generator = torch.Generator(device="cpu").manual_seed(20261027)
    cases = [
        ("identical", torch.rand((1, 3, 64, 64), generator=generator) * 2 - 1, None),
        (
            "deterministic_random_pair",
            torch.rand((1, 3, 64, 64), generator=generator) * 2 - 1,
            torch.rand((1, 3, 64, 64), generator=generator) * 2 - 1,
        ),
        ("black_white", -torch.ones((1, 3, 64, 64)), torch.ones((1, 3, 64, 64))),
    ]
    results = []
    with torch.no_grad():
        for name, first, second in cases:
            second = first.clone() if second is None else second
            official_value = float(official(first, second).reshape(-1)[0])
            frozen_value = float(frozen(first, second).reshape(-1)[0])
            results.append({
                "case": name,
                "official": official_value,
                "frozen": frozen_value,
                "absolute_difference": abs(official_value - frozen_value),
            })
    maximum = max(result["absolute_difference"] for result in results)
    report = {
        "status": "LPIPS_EQUIVALENCE_PASS" if maximum <= args.tolerance else "LPIPS_EQUIVALENCE_FAIL",
        "official_wheel": str(wheel),
        "official_wheel_sha256": file_sha256(wheel),
        "linear_checkpoint_sha256": file_sha256(linear),
        "alexnet_checkpoint_sha256": file_sha256(alexnet_checkpoint),
        "tolerance": args.tolerance,
        "maximum_absolute_difference": maximum,
        "results": results,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
    if maximum > args.tolerance:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
