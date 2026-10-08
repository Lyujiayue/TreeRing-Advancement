# Protocol Amendment v1.1 — Completion Template

> **Control rule:** Complete every field, remove every bracketed instruction and placeholder, attach the cited evidence, and obtain all approvals before any formal run. If an item is not applicable, state `Not applicable`, explain why, and obtain approval. Formal commands must contain no placeholders.

## 1. Document control

| Field | Final entry |
|---|---|
| Amendment title | [Enter final title] |
| Protocol amended | Baseline Experiment Protocol v1 |
| Amendment version | v1.1 |
| Status | [Draft / Approved] |
| Author(s) | [Names] |
| Date created | [YYYY-MM-DD] |
| Date approved | [YYYY-MM-DD] |
| Effective date | [YYYY-MM-DD] |
| Superseded documents | [List or state none] |
| Evidence package location and SHA-256 manifest | [Persistent path and hash] |

### 1.1 Reason and scope

[State why the amendment is required, which previously unresolved decisions it freezes, and whether any v1 language is clarified or changed.]

### 1.2 Impact assessment

[Describe effects on comparability, data, code, schema, compute, statistical analysis, risks, and prior development/pilot results. Confirm that no formal-set outcomes informed the change.]

## 2. Formal replicate seed and key allocation

The current rule is `sample_seed = base_generation_seed + sample_index`, with formal `sample_index` 0-999. Fill all rows and verify pairwise non-overlap.

| Replicate | Base generation seed | Inclusive seed range | Watermark seed | Watermark key ID | Verified non-overlap by / date |
|---:|---:|---|---:|---|---|
| 1 | [value] | [base to base+999] | [value] | [ID] | [name / date] |
| 2 | [value] | [base to base+999] | [value] | [ID] | [name / date] |
| 3 | [value] | [base to base+999] | [value] | [ID] | [name / date] |
| 4 | [value] | [base to base+999] | [value] | [ID] | [name / date] |
| 5 | [value] | [base to base+999] | [value] | [ID] | [name / date] |

**Non-overlap verification method and result:** [Provide reproducible check, output, and evidence path.]

**Pairing rule:** [Confirm how the same prompt, base latent, generation seed, key, settings, and attack randomness are shared across methods within each replicate.]

## 3. Record identity implementation

| Required field | Exact definition and allowed values | Storage location | Implementation commit | Test/evidence |
|---|---|---|---|---|
| `source_row_id` | [Complete] | [Complete] | [40-character commit] | [Complete] |
| `prompt_split` | [Complete] | [Complete] | [40-character commit] | [Complete] |
| `method_name` | [Complete] | [Complete] | [40-character commit] | [Complete] |
| `replicate_id` | [Complete] | [Complete] | [40-character commit] | [Complete] |
| `protocol_version` | [Complete] | [Complete] | [40-character commit] | [Complete] |

**Additional required fields:** Document `watermark_key_id`, fixed-threshold provenance/outcome, attack identity and realized randomness, quality metrics, latent residual RMS/L2, method-specific parameters, and model manifest/hash in the same manner.

**Final schema version and compatibility decision:** [Complete.]

## 4. Globally Weaker Tree-Ring

Frozen definition: `z_global = z + alpha * (z_original - z)`.

| Item | Final entry |
|---|---|
| Alpha candidate grid | [Enumerate every value] |
| Pilot selection rule | [Pre-specified deterministic rule, tie-breaker, and metric] |
| Final alpha | [Value] |
| Perturbation budget definition | [Exact RMS/L2 formula, axes, precision, and aggregation] |
| Pilot realized RMS/L2 | [Estimate and interval/distribution] |
| Detection result at selected alpha | [Pilot evidence] |
| Quality result at selected alpha | [Pilot evidence] |
| Implementation commit | [40-character commit] |
| Validation evidence | [Path/hash] |

## 5. Saliency-Aware Tree-Ring

### 5.1 Final algorithm

[Specify the complete algorithm in execution order, including preview generation if any, saliency estimator and preprocessing, map normalization, spatial weighting, residual construction, frequency-domain re-projection, key consistency, numerical precision, and deterministic behavior.]

| Item | Final entry |
|---|---|
| Candidate grid | [Enumerate the full Cartesian or staged grid] |
| Pilot selection rule | [Pre-specified deterministic rule and tie-breaker] |
| Final parameter values | [All values] |
| Perturbation budget definition | [Exact RMS/L2 formula] |
| Pilot realized RMS/L2 | [Estimate and interval/distribution] |
| Detection result | [Pilot evidence] |
| Quality result | [Pilot evidence] |
| Implementation commit | [40-character commit] |
| End-to-end validation | [Path/hash and result] |

## 6. Fixed detection threshold

| Item | Final entry |
|---|---|
| Calibration split | Pilot, source rows 1000-1199 |
| Detection score | Negative watermark distance; larger is stronger evidence |
| Selection rule | [Exact rule, target, interpolation convention, and tie-breaker] |
| Final threshold | [Numeric value with precision] |
| Pilot FPR | [Value and 95% CI] |
| Pilot TPR | [Value and 95% CI] |
| Pilot accuracy | [Value and definition] |
| Applicability | [Methods, attacks, keys, and whether one shared or multiple thresholds] |
| Implementation commit | [40-character commit] |
| Evidence | [Path/hash] |

Confirm that formal fixed-threshold results will not use `oracle_max_balanced_accuracy` as accuracy: [Yes, with reviewer initials/date].

## 7. Frozen quality metrics

For every metric, specify the exact callable/algorithm, input range, color space, resize/crop, normalization, device/dtype, batch size, per-sample versus aggregate behavior, and missing/non-finite handling.

| Metric | Exact implementation and preprocessing | Package/version | Weights or reference data | SHA-256 | Aggregation |
|---|---|---|---|---|---|
| CLIP text-image similarity | [Complete] | [Complete] | [Model/pretraining tag and file] | [Hash] | [Complete] |
| LPIPS vs paired No-Watermark | [Complete] | [Complete] | [Network/weights] | [Hash] | [Complete] |
| PSNR vs paired No-Watermark | [Complete] | [Complete] | N/A or [Complete] | [Hash or N/A] | [Complete] |
| SSIM vs paired No-Watermark | [Complete] | [Complete] | [Window/weights if any] | [Hash or N/A] | [Complete] |
| FID, fixed 5,000-sample MS-COCO protocol | [Complete] | [Complete] | [Reference set and Inception weights] | [All hashes] | [Complete] |

**FID feasibility decision:** [Include final protocol, or an approved, pilot-independent justification for omission.]

**Metric validation evidence:** [Controlled examples, expected values/tolerances, test commit, and artifact hashes.]

## 8. Robustness implementation verification

| Condition | Protocol value | Actual implementation verified | Verification method and evidence | Approved |
|---|---|---|---|---|
| Clean | No attack | [Complete] | [Complete] | [Initial/date] |
| Rotation | 75 degrees | [Direction, interpolation, fill, output size] | [Complete] | [Initial/date] |
| JPEG | 25 | [Library and exact meaning of 25] | [Complete] | [Initial/date] |
| Crop | scale 0.75, ratio 0.75 | [Crop/resize algorithm and randomness] | [Complete] | [Initial/date] |
| Gaussian blur | `gaussian_blur_r=4` | [Actual kernel size, sigma, padding] | [Mandatory empirical/code verification] | [Initial/date] |
| Gaussian noise | sigma 0.1 | [Value range, clipping, RNG] | [Complete] | [Initial/date] |
| Brightness | factor 6 | [Library operation, range, clipping] | [Complete] | [Initial/date] |

**Matched attack-randomness mechanism:** [Complete.]

## 9. Final implementation commits

| Component | Final 40-character commit | Clean-tree verification | Tests/evidence |
|---|---|---|---|
| Original Tree-Ring | [commit] | [Complete] | [Complete] |
| Globally Weaker Tree-Ring | [commit] | [Complete] | [Complete] |
| Saliency-Aware Tree-Ring | [commit] | [Complete] | [Complete] |
| Quality metrics | [commit] | [Complete] | [Complete] |
| Detection threshold and record schema | [commit] | [Complete] | [Complete] |
| Statistical analysis and confidence intervals | [commit] | [Complete] | [Complete] |

**Model manifest/hash and code bundle hash:** [Complete.]

## 10. Pilot evidence and locked decisions

| Evidence item | Final entry |
|---|---|
| Pilot run names | [All method/condition run names] |
| Prompt SHA-256 | `1455baf5b73b143aa07227c2dcfab96d5658c29e816fd9ddea33b591e7b70f3b` |
| Expected/completed samples | [Complete] |
| Git commit and `git.dirty` | [Commit; must be false] |
| GPU | NVIDIA B200 |
| Output and log hashes | [Manifest path/hash] |
| Pairing/missingness audit | [Result and evidence] |
| Candidate-grid results | [Evidence path/hash] |
| Selected global alpha | [Value and rule result] |
| Selected saliency parameters | [Values and rule result] |
| Selected threshold | [Value and rule result] |
| Detection-quality Pareto evidence | [Path/hash or justified N/A] |
| Deviations/anomalies | [Complete; state none only if verified] |

Confirm that the formal split was not used for any selection: [Yes, reviewer initials/date].

## 11. Frozen statistical analysis

[Specify the analysis population, paired estimands, replicate aggregation, 95% confidence-interval method, resampling unit and seed if applicable, distribution summaries, multiple-comparison treatment, missing-data rule, failure reporting, Pareto construction, and software/commit.]

## 12. Final placeholder-free formal commands

Insert the exact copyable commands for all four methods, five replicates, the clean condition, and every frozen robustness condition. Commands must include concrete run names, prompt/model paths, counts, base seeds, watermark seeds/key IDs or their resolved configuration, method parameters, attack parameters, protocol version, and record-identity arguments. Do not use shell variables, angle-bracket tokens, ellipses, or references such as “same as above.”

### Replicate 1

```bash
[Replace this entire block with complete commands; no placeholders may remain.]
```

### Replicate 2

```bash
[Replace this entire block with complete commands; no placeholders may remain.]
```

### Replicate 3

```bash
[Replace this entire block with complete commands; no placeholders may remain.]
```

### Replicate 4

```bash
[Replace this entire block with complete commands; no placeholders may remain.]
```

### Replicate 5

```bash
[Replace this entire block with complete commands; no placeholders may remain.]
```

## 13. Approval checklist

- [ ] Reason, scope, and impact are complete and scientifically justified.
- [ ] No formal data informed parameter, threshold, seed/key, metric, or attack selection.
- [ ] Five base seeds and inclusive ranges are frozen and proven non-overlapping.
- [ ] Watermark seeds and stable key IDs are frozen.
- [ ] Required per-sample identity fields are implemented, tested, and recorded.
- [ ] Globally Weaker algorithm, grid, rule, alpha, and budget are frozen.
- [ ] Saliency-Aware algorithm, grid, rule, parameters, and budget are frozen.
- [ ] Fixed threshold, selection rule, pilot FPR, and pilot TPR are frozen.
- [ ] CLIP, LPIPS, PSNR, SSIM, and FID implementations, versions, weights/data, and hashes are frozen, or an explicit FID omission is approved.
- [ ] Every robustness transformation, including the actual blur kernel, is verified.
- [ ] Final commits for Original, Global, Saliency, metrics, records/threshold, and statistics are immutable and listed.
- [ ] Seven unit tests and all new tests pass locally and on the B200 server.
- [ ] Development and pilot gates in the runbook pass.
- [ ] Pilot evidence and all artifact hashes are archived.
- [ ] Statistical methods and 95% confidence intervals are pre-specified.
- [ ] Every formal command is complete, concrete, and placeholder-free.
- [ ] Pre-run and post-run checklists have named owners.
- [ ] Formal replicate 1 audit and stop/go authority are assigned.
- [ ] Artifact preservation locations and access controls are approved.
- [ ] All deviations are resolved or explicitly approved.

## 14. Sign-off

| Role | Name | Decision | Signature/record | Date |
|---|---|---|---|---|
| Principal investigator / protocol owner | [Complete] | [Approve / Reject] | [Complete] | [YYYY-MM-DD] |
| Implementation reviewer | [Complete] | [Approve / Reject] | [Complete] | [YYYY-MM-DD] |
| Experimental execution owner | [Complete] | [Approve / Reject] | [Complete] | [YYYY-MM-DD] |
| Statistical analysis reviewer | [Complete] | [Approve / Reject] | [Complete] | [YYYY-MM-DD] |
| Independent artifact/audit reviewer | [Complete] | [Approve / Reject] | [Complete] | [YYYY-MM-DD] |

Formal execution authorization: [Approved / Not approved].
