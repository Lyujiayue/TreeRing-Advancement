"""Plan-only Pilot freeze output and deterministic post-Pilot selection."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pilot_selection import pilot_specification, select_pilot
from experiment_records import get_git_state, utc_now_iso


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
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("Output parent must be an existing regular directory")
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(rendered)


def command_plan(args):
    report = {
        "status": "PILOT_FREEZE_PLAN_PASS",
        "execution_started": False,
        "created_at_utc": utc_now_iso(),
        "git": get_git_state(REPO),
        "specification": pilot_specification(),
    }
    if args.output:
        write_new(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def command_select(args):
    input_path = Path(args.input).resolve()
    if input_path.is_symlink() or not input_path.is_file():
        raise ValueError("Pilot observation input must be an existing regular file")
    observations = strict_json(input_path.read_text(encoding="utf-8"))
    decision = select_pilot(observations)
    write_new(args.output, decision)
    print(decision["status"])


def build_parser():
    parser = argparse.ArgumentParser(
        description="Frozen TreeRing-Advancement Pilot planning and selection"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser(
        "plan", help="Print the outcome-independent freeze; never starts an experiment"
    )
    plan.add_argument("--output")
    plan.set_defaults(handler=command_plan)
    select = subparsers.add_parser(
        "select", help="Apply the frozen rule to a complete audited Pilot observation file"
    )
    select.add_argument("--input", required=True)
    select.add_argument("--output", required=True)
    select.set_defaults(handler=command_select)
    return parser


def main():
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
