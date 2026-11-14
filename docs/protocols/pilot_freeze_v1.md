# Pilot Freeze v1

## Status and boundary

This document freezes the TreeRing-Advancement Pilot design before any Pilot
image is generated or any Pilot outcome is observed. It does not authorize GPU
execution, Formal execution, dependency changes, model downloads, or instance
start/stop operations. Development results are used only to bound a compact
candidate set; they are not Pilot selections or paper conclusions.

The machine-readable source of truth is `pilot_selection.py`, specification
version `treering-pilot-freeze-v1`. `scripts/pilot_validation.py plan` prints
the complete outcome-independent freeze and never starts an experiment.

## Frozen Pilot population and identity

| Item | Frozen value |
|---|---|
| Prompt split | Pilot, source rows 1000-1199 |
| Prompt count | 200 |
| Prompt SHA-256 | `1455baf5b73b143aa07227c2dcfab96d5658c29e816fd9ddea33b591e7b70f3b` |
| Replicate | 1 |
| Base generation seed | 0; realized seed is `base + sample_index` |
| Watermark seed/key | `999999` / `treering-rand-wseed-999999` |
| Protocol record | `v1.1-pilot` |
| Condition | Clean |
| Generation/inversion steps | 50 / 50 |
| Run purpose/range | `full`, indices 0-199 |

Every candidate must use the same prompt order, base latent, generation seed,
watermark key, model, scheduler, generation/inversion settings, and paired
No-Watermark reference for a given sample. Before selection, the acceptance
gate must prove that the 200 No-Watermark PNGs and identities are identical
across all candidate runs. The duplicated No-Watermark observations are then
deduplicated to exactly 200 calibration observations.

## Frozen candidate runs

The runner produces paired No-Watermark and watermarked images in each listed
invocation, so a separate No-Watermark generation run is not required.

### Original reference

- `original_pilot_r1_clean_full_s0_e200_g50_i50`

### Globally Weaker grid

The candidate alpha values are exactly `0.25`, `0.5`, and `0.75`. No-Watermark
and Original provide the alpha 0 and alpha 1 endpoints without duplicate
Global generation.

- `global_pilot_r1_clean_a0p25_full_s0_e200_g50_i50`
- `global_pilot_r1_clean_a0p5_full_s0_e200_g50_i50`
- `global_pilot_r1_clean_a0p75_full_s0_e200_g50_i50`

### Saliency-Aware grid

Preview steps remain fixed at 10 because that path passed the real B200
Development smoke and 32-sample gate. The six candidates include the validated
Development anchor and a compact, outcome-independent sweep toward lower
background strength, zero salient-region residual, and a softer saliency-map
exponent:

| Preview | Background | Foreground | Power |
|---:|---:|---:|---:|
| 10 | 1.5 | 0.2 | 2.0 |
| 10 | 1.0 | 0.2 | 2.0 |
| 10 | 0.5 | 0.2 | 2.0 |
| 10 | 1.0 | 0.0 | 2.0 |
| 10 | 0.5 | 0.0 | 2.0 |
| 10 | 1.0 | 0.2 | 1.0 |

These correspond to six canonical run names emitted by
`pilot_selection.candidate_plan()`. Values outside this table are not Pilot
candidates under this freeze.

## Shared fixed-threshold calibration

The detection score is negative watermark distance, so larger values indicate
stronger evidence. Use exactly the 200 unique No-Watermark Pilot scores. Sort
them in ascending order as `s_(1), ..., s_(200)` and set:

```text
threshold = math.nextafter(s_(198), +infinity)
```

A sample is positive when `score >= threshold`. This admits at most the two
scores strictly greater than the 198th order statistic, so empirical Pilot FPR
is at most 1%. Ties are handled conservatively and can reduce, but never
increase, the empirical FPR. The one shared threshold applies to Original,
Global, and Saliency candidates. Report false-positive count, empirical FPR,
and a two-sided Wilson 95% interval.

## Deterministic candidate selection

Original is the detection and quality reference. A candidate is feasible only
when all 200 paired observations are present and finite and all three rules
hold:

1. Fixed-threshold TPR is at least `Original TPR - 0.05`.
2. Mean paired CLIP delta is at least `Original mean CLIP delta - 0.01`.
3. Mean LPIPS is strictly lower than Original mean LPIPS.

Select Global and Saliency independently. Among feasible candidates, select
the one with the lowest mean LPIPS. Resolve an exact tie by, in order: higher
mean CLIP delta, higher mean SSIM, higher mean PSNR, lower mean latent residual
RMS, then lexicographically smaller candidate label.

If either family has no feasible candidate, record
`NO_FEASIBLE_CANDIDATE_STOP`, select no parameters for that family, and stop
before Formal. Do not relax a margin after observing outcomes.

## Uncertainty, completeness, and reporting

- Proportion intervals: two-sided Wilson 95% intervals.
- Paired mean-difference intervals: 10,000 prompt-level paired bootstrap
  resamples, NumPy seed `20261114`, linear 2.5th/97.5th percentiles.
- Report every candidate, including failed or dominated candidates. Pilot
  selection is deterministic and descriptive; it is not a family of hidden
  hypothesis tests.
- Report fixed-threshold TPR, CLIP delta, LPIPS, PSNR, SSIM, latent residual
  RMS/L2, paired differences versus Original, and detection-quality Pareto
  views.
- Any missing, duplicated, mispaired, non-finite, wrong-prompt, wrong-commit,
  or CPU-fallback observation invalidates that candidate. Do not impute or
  silently omit it.
- `oracle_max_balanced_accuracy` remains descriptive and must not be used for
  fixed-threshold selection or reporting.

Even after a successful Pilot selection, `formal_authorized` remains false
until the evidence is archived, amendment v1.1 is completed and approved, all
remaining implementation/robustness/statistical decisions are frozen, and
the formal commands are placeholder-free.
