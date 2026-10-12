"""Explicit new-run identity, kept separate from historical v1 name readers."""

from experiment_records import (
    expected_run_name, validate_method_configuration, validate_sample_range,
)


RUN_NAMING_VERSION = "treering-run-name-v2"


def alpha_token(alpha):
    """Lossless round-trip float token; never round distinct candidate values."""
    alpha = validate_method_configuration("globally_weaker_tree_ring", alpha)["alpha"]
    if alpha == 0:
        alpha = 0.0  # Canonicalize signed zero, which has identical semantics.
    return repr(alpha).replace(".", "p").replace("-", "m").replace("+", "p")


def build_run_identity(args):
    """Validate purpose/range/steps before any output deletion or model I/O."""
    from experiment_records import FROZEN_PROMPT_SPLITS

    purpose = getattr(args, "run_purpose", None)
    if purpose not in ("smoke", "full"):
        raise ValueError("run_purpose must be explicitly smoke or full")
    sample_range = validate_sample_range(args.prompt_split, args.start, args.end)
    for field in ("num_inference_steps", "test_num_inference_steps"):
        value = getattr(args, field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{field} must be a positive integer")
    if purpose == "smoke" and args.prompt_split != "development":
        raise ValueError("Smoke must use the frozen development split")
    if purpose == "full" and (args.start, args.end) != (
        0, FROZEN_PROMPT_SPLITS[args.prompt_split]["count"]
    ):
        raise ValueError("Full runs must cover the entire frozen split")

    method_parameters = validate_method_configuration(
        args.method_name, getattr(args, "global_alpha", None)
    )
    name = expected_run_name(
        args.method_name, args.prompt_split, args.replicate_id, args.attack_name
    )
    if method_parameters:
        name += "_a" + alpha_token(method_parameters["alpha"])
    name += f"_{purpose}"
    # Steps distinguish one-step format checks from normal-step E2E probes.
    name += (f"_s{args.start}_e{args.end}"
             f"_g{args.num_inference_steps}_i{args.test_num_inference_steps}")
    if args.run_name != name:
        raise ValueError(f"run_name must be {name!r}, received {args.run_name!r}")
    return {
        "run_naming_version": RUN_NAMING_VERSION,
        "run_purpose": purpose,
        "sample_range": sample_range,
    }
