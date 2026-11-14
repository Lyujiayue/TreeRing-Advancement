"""Pre-registered Pilot grids, threshold calibration, and selection rules."""
from __future__ import annotations

import hashlib
import json
import math
import statistics

import numpy as np


PILOT_SPEC_VERSION = "treering-pilot-freeze-v1"
PILOT_OBSERVATION_SCHEMA = "treering-pilot-observations-v1"
PILOT_DECISION_SCHEMA = "treering-pilot-decision-v1"
PILOT_PROMPT_SHA256 = "1455baf5b73b143aa07227c2dcfab96d5658c29e816fd9ddea33b591e7b70f3b"
PILOT_PROMPT_COUNT = 200
PILOT_SOURCE_ROW_IDS = tuple(range(1000, 1200))
PILOT_BASE_GENERATION_SEED = 0
PILOT_WATERMARK_SEED = 999999
PILOT_WATERMARK_KEY_ID = "treering-rand-wseed-999999"
PILOT_PROTOCOL_VERSION = "v1.1-pilot"

GLOBAL_ALPHA_GRID = (0.25, 0.5, 0.75)
SALIENCY_GRID = (
    (10, 1.5, 0.2, 2.0),  # Validated Development anchor.
    (10, 1.0, 0.2, 2.0),
    (10, 0.5, 0.2, 2.0),
    (10, 1.0, 0.0, 2.0),
    (10, 0.5, 0.0, 2.0),
    (10, 1.0, 0.2, 1.0),
)

TARGET_EMPIRICAL_FPR = 0.01
TPR_NONINFERIORITY_MARGIN = 0.05
CLIP_DELTA_MARGIN = 0.01
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20_261_114
CONFIDENCE_LEVEL = 0.95

OBSERVATION_ARRAY_FIELDS = (
    "watermarked_distances",
    "clip_delta",
    "lpips",
    "psnr_db",
    "ssim",
    "latent_residual_rms",
    "latent_residual_l2",
)


def _float_token(value):
    return repr(float(value)).replace(".", "p").replace("-", "m").replace("+", "p")


def _saliency_parameters(values):
    preview, background, foreground, power = values
    return {
        "preview_steps": preview,
        "background_strength": float(background),
        "foreground_strength": float(foreground),
        "power": float(power),
    }


def _saliency_label(parameters):
    return (
        f"saliency_pv{parameters['preview_steps']}"
        f"_bg{_float_token(parameters['background_strength'])}"
        f"_fg{_float_token(parameters['foreground_strength'])}"
        f"_pw{_float_token(parameters['power'])}"
    )


def candidate_plan():
    """Return the exact ten pre-registered Pilot watermarked runs."""
    suffix = "_full_s0_e200_g50_i50"
    plan = [{
        "label": "original",
        "method_name": "original_tree_ring",
        "method_parameters": {},
        "run_name": "original_pilot_r1_clean" + suffix,
    }]
    for alpha in GLOBAL_ALPHA_GRID:
        parameters = {"alpha": float(alpha)}
        plan.append({
            "label": f"global_a{_float_token(alpha)}",
            "method_name": "globally_weaker_tree_ring",
            "method_parameters": parameters,
            "run_name": f"global_pilot_r1_clean_a{_float_token(alpha)}" + suffix,
        })
    for values in SALIENCY_GRID:
        parameters = _saliency_parameters(values)
        label = _saliency_label(parameters)
        plan.append({
            "label": label,
            "method_name": "saliency_aware_tree_ring",
            "method_parameters": parameters,
            "run_name": label.replace("saliency_", "saliency_pilot_r1_clean_", 1) + suffix,
        })
    return plan


def pilot_specification():
    """Return the outcome-independent Pilot freeze record."""
    return {
        "schema_version": PILOT_SPEC_VERSION,
        "prompt": {
            "split": "pilot",
            "path": ".local_data/prompts/splits/sdp_pilot_rows_1000_1199.txt",
            "sha256": PILOT_PROMPT_SHA256,
            "count": PILOT_PROMPT_COUNT,
            "source_row_ids": list(PILOT_SOURCE_ROW_IDS),
        },
        "identity": {
            "replicate_id": 1,
            "attack_name": "clean",
            "protocol_version": PILOT_PROTOCOL_VERSION,
            "base_generation_seed": PILOT_BASE_GENERATION_SEED,
            "watermark_seed": PILOT_WATERMARK_SEED,
            "watermark_key_id": PILOT_WATERMARK_KEY_ID,
            "run_purpose": "full",
            "generation_steps": 50,
            "inversion_steps": 50,
        },
        "runs": candidate_plan(),
        "shared_threshold": {
            "score": "negative watermark distance; larger is stronger evidence",
            "calibration_population": "200 unique paired No-Watermark Pilot observations",
            "prediction": "positive when score >= threshold",
            "rule": (
                "Sort 200 No-Watermark scores ascending and set threshold to "
                "math.nextafter(the 198th order statistic, +infinity), giving "
                "at most two false positives and empirical FPR <= 1%."
            ),
        },
        "selection": {
            "tpr_noninferiority_margin": TPR_NONINFERIORITY_MARGIN,
            "clip_delta_margin": CLIP_DELTA_MARGIN,
            "quality_benefit": "mean LPIPS must be strictly lower than Original",
            "primary_objective": "minimum mean LPIPS among feasible candidates",
            "tie_breakers": [
                "higher mean CLIP delta",
                "higher mean SSIM",
                "higher mean PSNR",
                "lower mean latent residual RMS",
                "lexicographically smaller candidate label",
            ],
            "failure_rule": "If either method family has no feasible candidate, stop without selecting or starting Formal.",
        },
        "uncertainty": {
            "proportions": "two-sided Wilson 95% interval",
            "paired_mean_differences": "10,000 prompt-level paired bootstrap resamples",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "quantile_method": "NumPy linear 2.5th and 97.5th percentiles",
            "multiplicity": "Pilot selection is deterministic and descriptive; report every candidate without selective omission.",
        },
        "boundaries": {
            "development_values_are_not_selection_results": True,
            "formal_data_may_not_be_read": True,
            "formal_authorized": False,
            "fid": "omitted_by_approved_protocol_decision",
        },
    }


def calibrate_shared_threshold(no_watermark_distances):
    distances = _finite_vector(no_watermark_distances, "no_watermark_distances")
    scores = [-value for value in distances]
    ordered = sorted(scores)
    threshold = math.nextafter(ordered[197], math.inf)
    false_positives = sum(score >= threshold for score in scores)
    return {
        "threshold": threshold,
        "false_positives": false_positives,
        "count": PILOT_PROMPT_COUNT,
        "empirical_fpr": false_positives / PILOT_PROMPT_COUNT,
        "fpr_wilson_95": list(wilson_interval(false_positives, PILOT_PROMPT_COUNT)),
        "order_statistic_one_based": 198,
        "nextafter_direction": "+infinity",
    }


def wilson_interval(successes, count, z=1.959963984540054):
    if not isinstance(successes, int) or not isinstance(count, int) or not 0 <= successes <= count or count <= 0:
        raise ValueError("Invalid binomial counts")
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    spread = z * math.sqrt(proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)) / denominator
    return max(0.0, center - spread), min(1.0, center + spread)


def _finite_vector(values, name):
    if not isinstance(values, list) or len(values) != PILOT_PROMPT_COUNT:
        raise ValueError(f"{name} must contain exactly {PILOT_PROMPT_COUNT} values")
    result = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite or non-numeric value")
        result.append(float(value))
    return result


def _sha256_vector(values, name):
    if not isinstance(values, list) or len(values) != PILOT_PROMPT_COUNT:
        raise ValueError(f"{name} must contain exactly {PILOT_PROMPT_COUNT} hashes")
    for value in values:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} contains an invalid SHA-256")
    return list(values)


def _validate_observation(
    observation, expected, implementation_commit,
    canonical_no_watermark_distances, canonical_no_watermark_hashes,
):
    if not isinstance(observation, dict):
        raise ValueError("Candidate observation must be an object")
    for field in (
        "label", "run_name", "method_name", "method_parameters",
        "git_commit", "git_dirty", "source_row_ids",
        "no_watermark_distances", "no_watermark_image_sha256",
        "watermarked_image_sha256", *OBSERVATION_ARRAY_FIELDS,
    ):
        if field not in observation:
            raise ValueError(f"Candidate is missing {field}")
    if observation["label"] != expected["label"]:
        raise ValueError("Candidate label mismatch")
    if observation["run_name"] != expected["run_name"]:
        raise ValueError(f"Candidate run name mismatch: {expected['label']}")
    if observation["method_name"] != expected["method_name"]:
        raise ValueError(f"Candidate method mismatch: {expected['label']}")
    if observation["method_parameters"] != expected["method_parameters"]:
        raise ValueError(f"Candidate parameters mismatch: {expected['label']}")
    if observation["git_commit"] != implementation_commit or observation["git_dirty"] is not False:
        raise ValueError(f"Candidate Git identity mismatch: {expected['label']}")
    if observation["source_row_ids"] != list(PILOT_SOURCE_ROW_IDS):
        raise ValueError(f"Candidate source rows mismatch: {expected['label']}")
    no_watermark_distances = _finite_vector(
        observation["no_watermark_distances"],
        f"{expected['label']}.no_watermark_distances",
    )
    no_watermark_hashes = _sha256_vector(
        observation["no_watermark_image_sha256"],
        f"{expected['label']}.no_watermark_image_sha256",
    )
    _sha256_vector(
        observation["watermarked_image_sha256"],
        f"{expected['label']}.watermarked_image_sha256",
    )
    if no_watermark_distances != canonical_no_watermark_distances:
        raise ValueError(f"Candidate No-Watermark distances differ: {expected['label']}")
    if no_watermark_hashes != canonical_no_watermark_hashes:
        raise ValueError(f"Candidate No-Watermark PNG hashes differ: {expected['label']}")
    return {
        field: _finite_vector(observation[field], f"{expected['label']}.{field}")
        for field in OBSERVATION_ARRAY_FIELDS
    }


def _summarize(values, threshold):
    detections = [(-distance) >= threshold for distance in values["watermarked_distances"]]
    positives = sum(detections)
    return {
        "fixed_threshold_tpr": positives / PILOT_PROMPT_COUNT,
        "fixed_threshold_tpr_wilson_95": list(wilson_interval(positives, PILOT_PROMPT_COUNT)),
        "positive_count": positives,
        "watermark_distance_mean": statistics.fmean(values["watermarked_distances"]),
        "clip_delta_mean": statistics.fmean(values["clip_delta"]),
        "lpips_mean": statistics.fmean(values["lpips"]),
        "psnr_db_mean": statistics.fmean(values["psnr_db"]),
        "ssim_mean": statistics.fmean(values["ssim"]),
        "latent_residual_rms_mean": statistics.fmean(values["latent_residual_rms"]),
        "latent_residual_l2_mean": statistics.fmean(values["latent_residual_l2"]),
        "detections": detections,
    }


def _bootstrap_ci(differences, indices):
    values = np.asarray(differences, dtype=np.float64)
    estimates = values[indices].mean(axis=1, dtype=np.float64)
    lower, upper = np.quantile(estimates, [0.025, 0.975], method="linear")
    return [float(lower), float(upper)]


def _candidate_analysis(candidate, original, indices):
    summary = candidate["summary"]
    original_summary = original["summary"]
    feasibility = {
        "tpr_noninferior": summary["fixed_threshold_tpr"] >= (
            original_summary["fixed_threshold_tpr"] - TPR_NONINFERIORITY_MARGIN
        ),
        "clip_guard": summary["clip_delta_mean"] >= (
            original_summary["clip_delta_mean"] - CLIP_DELTA_MARGIN
        ),
        "lpips_benefit": summary["lpips_mean"] < original_summary["lpips_mean"],
    }
    paired = {}
    fields = {
        "fixed_threshold_tpr": (
            [float(value) for value in summary["detections"]],
            [float(value) for value in original_summary["detections"]],
        ),
        "clip_delta": (candidate["values"]["clip_delta"], original["values"]["clip_delta"]),
        "lpips": (candidate["values"]["lpips"], original["values"]["lpips"]),
        "psnr_db": (candidate["values"]["psnr_db"], original["values"]["psnr_db"]),
        "ssim": (candidate["values"]["ssim"], original["values"]["ssim"]),
        "latent_residual_rms": (
            candidate["values"]["latent_residual_rms"], original["values"]["latent_residual_rms"]
        ),
        "latent_residual_l2": (
            candidate["values"]["latent_residual_l2"], original["values"]["latent_residual_l2"]
        ),
    }
    for name, (candidate_values, original_values) in fields.items():
        differences = [first - second for first, second in zip(candidate_values, original_values)]
        paired[name] = {
            "mean_difference_candidate_minus_original": statistics.fmean(differences),
            "paired_bootstrap_95": _bootstrap_ci(differences, indices),
        }
    return {
        "label": candidate["label"],
        "method_name": candidate["method_name"],
        "method_parameters": candidate["method_parameters"],
        "summary": {key: value for key, value in summary.items() if key != "detections"},
        "feasibility": feasibility,
        "feasible": all(feasibility.values()),
        "paired_vs_original": paired,
    }


def _select_family(analyses):
    feasible = [candidate for candidate in analyses if candidate["feasible"]]
    if not feasible:
        return {
            "status": "NO_FEASIBLE_CANDIDATE_STOP",
            "selected_label": None,
            "selected_parameters": None,
        }
    selected = min(
        feasible,
        key=lambda candidate: (
            candidate["summary"]["lpips_mean"],
            -candidate["summary"]["clip_delta_mean"],
            -candidate["summary"]["ssim_mean"],
            -candidate["summary"]["psnr_db_mean"],
            candidate["summary"]["latent_residual_rms_mean"],
            candidate["label"],
        ),
    )
    return {
        "status": "SELECTED",
        "selected_label": selected["label"],
        "selected_parameters": selected["method_parameters"],
    }


def select_pilot(observations):
    """Validate complete Pilot observations and apply the frozen decision rule."""
    if not isinstance(observations, dict):
        raise ValueError("Pilot observations must be an object")
    if observations.get("schema_version") != PILOT_OBSERVATION_SCHEMA:
        raise ValueError("Pilot observation schema mismatch")
    if observations.get("prompt_sha256") != PILOT_PROMPT_SHA256:
        raise ValueError("Pilot prompt SHA-256 mismatch")
    if observations.get("source_row_ids") != list(PILOT_SOURCE_ROW_IDS):
        raise ValueError("Pilot source-row identity mismatch")
    implementation_commit = observations.get("implementation_commit")
    if (
        not isinstance(implementation_commit, str)
        or len(implementation_commit) != 40
        or any(character not in "0123456789abcdef" for character in implementation_commit)
    ):
        raise ValueError("Pilot implementation commit must be a full Git SHA")
    canonical_no_watermark_distances = _finite_vector(
        observations.get("no_watermark_distances"), "no_watermark_distances"
    )
    canonical_no_watermark_hashes = _sha256_vector(
        observations.get("no_watermark_image_sha256"), "no_watermark_image_sha256"
    )
    calibration = calibrate_shared_threshold(canonical_no_watermark_distances)
    require_fpr = calibration["empirical_fpr"] <= TARGET_EMPIRICAL_FPR
    if not require_fpr:
        raise ValueError("Calibrated threshold exceeds the frozen FPR target")

    plan = candidate_plan()
    expected_original = plan[0]
    expected_global = plan[1:1 + len(GLOBAL_ALPHA_GRID)]
    expected_saliency = plan[1 + len(GLOBAL_ALPHA_GRID):]
    original_values = _validate_observation(
        observations.get("original"), expected_original, implementation_commit,
        canonical_no_watermark_distances, canonical_no_watermark_hashes,
    )

    def validate_family(name, expected):
        supplied = observations.get(name)
        if not isinstance(supplied, list) or len(supplied) != len(expected):
            raise ValueError(f"{name} candidate count mismatch")
        by_label = {item.get("label"): item for item in supplied if isinstance(item, dict)}
        if len(by_label) != len(supplied) or set(by_label) != {item["label"] for item in expected}:
            raise ValueError(f"{name} candidate identity mismatch")
        return [
            {
                **candidate,
                "values": _validate_observation(
                    by_label[candidate["label"]], candidate, implementation_commit,
                    canonical_no_watermark_distances, canonical_no_watermark_hashes,
                ),
            }
            for candidate in expected
        ]

    global_candidates = validate_family("global_candidates", expected_global)
    saliency_candidates = validate_family("saliency_candidates", expected_saliency)
    threshold = calibration["threshold"]
    original = {
        **expected_original,
        "values": original_values,
        "summary": _summarize(original_values, threshold),
    }
    for candidate in (*global_candidates, *saliency_candidates):
        candidate["summary"] = _summarize(candidate["values"], threshold)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, PILOT_PROMPT_COUNT, size=(BOOTSTRAP_RESAMPLES, PILOT_PROMPT_COUNT))
    global_analyses = [_candidate_analysis(candidate, original, indices) for candidate in global_candidates]
    saliency_analyses = [_candidate_analysis(candidate, original, indices) for candidate in saliency_candidates]
    global_selection = _select_family(global_analyses)
    saliency_selection = _select_family(saliency_analyses)
    complete = (
        global_selection["status"] == "SELECTED"
        and saliency_selection["status"] == "SELECTED"
    )
    return {
        "schema_version": PILOT_DECISION_SCHEMA,
        "pilot_spec_version": PILOT_SPEC_VERSION,
        "implementation_commit": implementation_commit,
        "status": "PILOT_SELECTION_PASS" if complete else "PILOT_SELECTION_STOP_NO_FEASIBLE_CANDIDATE",
        "threshold": calibration,
        "original": {
            "label": original["label"],
            "method_name": original["method_name"],
            "method_parameters": original["method_parameters"],
            "summary": {key: value for key, value in original["summary"].items() if key != "detections"},
        },
        "global_candidates": global_analyses,
        "saliency_candidates": saliency_analyses,
        "global_selection": global_selection,
        "saliency_selection": saliency_selection,
        "formal_authorized": False,
        "all_candidates_reported": True,
    }
