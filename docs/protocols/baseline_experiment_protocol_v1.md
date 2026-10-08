# Baseline Experiment Protocol v1

## 1. Scope and research objective

This protocol defines the paper-level experimental design for TreeRing-Advancement. The objective is to determine whether saliency-aware spatial redistribution of the watermark perturbation improves visual quality relative to both Original Tree-Ring and a uniformly weakened watermark while preserving, as far as possible, detection performance and robustness.

The formal comparison comprises exactly these methods:

1. No Watermark.
2. Original Tree-Ring.
3. Globally Weaker Tree-Ring.
4. Saliency-Aware Tree-Ring.

No formal-set observation may be used to tune a method, choose a threshold, select a seed or key, or decide which attacks to report.

## 2. Reproducibility boundary

The official GitHub runner currently generates images with DPM-Solver. This project uses DDIM for both generation and inversion. Therefore, results from this project must not be described as a parameter-for-parameter reproduction of the authors' current repository. The primary baseline must be described as:

> Original Tree-Ring (Tree-RingRand) implemented in our frozen DDIM-based Stable Diffusion 2.1 pipeline.

Relevant repository milestones are:

| Commit | Meaning |
|---|---|
| `107b662651259deb24bb96426c67271721d1999b` | Offline SD2.1 Tree-Ring baseline reproduction. |
| `61aa2373e63e5feb60828d8b0718fa7f15800c6a` | Local experiment data excluded from Git. |
| `296a6225c9bb2db2c7373086f2b15c8deb192fb9` | Reproducible experiment metadata recording. |

Final method and analysis commits must be frozen in amendment v1.1.

## 3. Frozen generation and inversion configuration

| Setting | Frozen value |
|---|---|
| Model | `sd2-community/stable-diffusion-2-1-base` |
| Server model path | `/public/home/rxa-jj-ljq/models/stable-diffusion-2-1-base` |
| Resolution | 512 x 512 |
| Generation scheduler | DDIM |
| Inversion scheduler | DDIM |
| Generation steps | 50 |
| Inversion steps | 50 |
| Guidance scale | 7.5 |
| Inversion prompt | Empty string |
| Images per prompt | 1 |
| GPU dtype | `torch.float16` |
| Weights & Biases | Disabled by default |

All four methods must use the same prompt, base latent, generation seed, watermark key where applicable, generation settings, inversion settings, and attack randomness for a given paired sample.

## 4. Frozen Original Tree-Ring configuration

| Setting | Frozen value |
|---|---|
| `w_pattern` | `rand` |
| `w_radius` | `10` |
| `w_channel` | `0` |
| `w_seed` | `999999` |
| `w_mask_shape` | `circle` |
| `w_measurement` | `l1_complex` |
| `w_injection` | `complex` |

The original paper studies both Tree-RingRand and Tree-RingRings. The repository README example uses `ring` with channel `3`, whereas the official parser defaults and this project's validated baseline use `rand` with channel `0`. The latter is the primary baseline. A `ring` plus channel `3` comparison may be reported as a supplemental cross-pattern/channel experiment, but it must not replace the primary baseline.

## 5. Frozen prompt dataset and provenance

| Item | Frozen value |
|---|---|
| Dataset | `Gustavosta/Stable-Diffusion-Prompts` |
| Repository revision | `f40484704ca5bdeb389483123ba017897f1bfcdc` |
| Source split | `test` |
| Source file | `data/eval.parquet` |
| Source rows | 8,192 |
| Source size | 1,026,077 bytes |
| Source SHA-256 | `e2df7c8d0b1b7bff9bd0afcf61b841181876920ab55c23d1791c6be99b5e9e25` |

### 5.1 Frozen disjoint subsets

| Split | Source rows, inclusive | Count | SHA-256 |
|---|---:|---:|---|
| Formal | 0-999 | 1,000 | `949e3e7acb3c7b8bce874add0d822823e58826a1a28c9390b6f712eb7bdf57ce` |
| Pilot | 1000-1199 | 200 | `1455baf5b73b143aa07227c2dcfab96d5658c29e816fd9ddea33b591e7b70f3b` |
| Development | 1200-1231 | 32 | `39ed02810d78fdec8ab535935ac8f2e3819043c7db15a7585d2a999990b57d5f` |

### 5.2 Supporting files

| File | Count | SHA-256 |
|---|---:|---|
| `sdp_rows_0000_1231.jsonl` | 1,232 | `c51a2f04f0da0acf7a9de8aa6dcd4ae4fa03af9d4bb861013ff6553236d9a7a8` |
| `dataset_manifest.json` | N/A | `3ee2444fbfb39d579c0cf815193338c7324b2bce5dcd144809df8397b96d158c` |

The split files and manifest are controlled inputs. Their hashes must be verified before execution; neither line ordering nor prompt normalization may be changed after freezing.

## 6. Experimental scales and separation of purpose

| Stage | Scale | Permitted use |
|---|---|---|
| Development | 32 prompts x 1 seed sequence x 1 watermark key | Implementation, debugging, and end-to-end validation only; never paper conclusions. |
| Pilot | 200 prompts x 1 seed sequence x 1 watermark key | Selection of global alpha, saliency parameters, and the detection threshold. |
| Formal | 1,000 prompts per replicate x 5 independent replicates | Final locked evaluation. Complete and audit replicate 1 before running the remaining four. |

The current per-sample seed rule is `sample_seed = gen_seed + sample_index`, where `sample_index` is local to the selected prompt file. Amendment v1.1 must freeze five non-overlapping formal base seeds and seed ranges, along with watermark seed and key identifiers.

## 7. Method definitions and fairness

### 7.1 No Watermark

Generate from the shared base latent without watermark injection. This image is the paired visual-quality reference for the three watermarked methods.

### 7.2 Original Tree-Ring

Apply the frozen Tree-RingRand configuration in Section 4 to the shared base latent.

### 7.3 Globally Weaker Tree-Ring

Let `z` be the unwatermarked latent and `z_original` the latent produced by Original Tree-Ring. Define:

```text
z_global = z + alpha * (z_original - z)
```

The alpha candidate grid, selection rule, final alpha, and achieved perturbation budget must be frozen from pilot data only in amendment v1.1.

### 7.4 Saliency-Aware Tree-Ring

The algorithm, saliency construction, candidate grid, selection rule, final parameters, and perturbation budget must be frozen from development and pilot work in amendment v1.1. End-to-end integration must be validated before pilot execution.

### 7.5 Required fairness views

At minimum, the analysis must include:

- A comparison matched on latent residual RMS or L2 budget.
- An image-quality comparison matched on detection performance.
- Detection-quality Pareto curves when the candidate sweeps contain enough valid operating points.

For every sample, record the method-specific perturbation parameters and the realized latent residual RMS and L2 norm. Paired methods must share all non-method inputs and randomness.

## 8. Detection outcomes

Report all of the following with 95% confidence intervals appropriate to the paired, replicated design:

- ROC-AUC.
- TPR at 1% FPR.
- Fixed-threshold accuracy.
- Fixed-threshold TPR and FPR.
- Watermark-distance distributions.

Smaller watermark distance means that the recovered latent is closer to the watermark key. The detection score is the negative distance, so a larger score means stronger watermark evidence.

The legacy `acc` value maximizes balanced accuracy on the same test ROC curve. It must be named `oracle_max_balanced_accuracy`; it is descriptive only and is not formal fixed-threshold accuracy. Formal accuracy, TPR, and FPR must use a threshold calibrated on the pilot split under a rule pre-specified and frozen in amendment v1.1.

## 9. Image-quality outcomes

The required image-quality measures are:

- CLIP text-image similarity.
- LPIPS against the paired No-Watermark image.
- PSNR against the paired No-Watermark image.
- SSIM against the paired No-Watermark image.
- FID under a fixed 5,000-sample MS-COCO protocol, if feasible.

Exact implementations, preprocessing, aggregation, library versions, weights, hashes, and the FID reference set must be frozen in amendment v1.1 before formal use. If FID proves infeasible, the reason and the decision to omit it require explicit approval in that amendment; formal-set results may not drive the decision.

## 10. Frozen robustness conditions

Evaluate the un-attacked condition and each frozen attack independently, using matched attack randomness across paired methods.

| Condition | Frozen runner parameter |
|---|---|
| Rotation | `r_degree = 75` |
| JPEG | `jpeg_ratio = 25` |
| Crop | `crop_scale = 0.75`, `crop_ratio = 0.75` |
| Gaussian blur | `gaussian_blur_r = 4` |
| Gaussian noise | `gaussian_std = 0.1` |
| Brightness | `brightness_factor = 6` |

Before formal execution, the implementation must be inspected and empirically verified to establish the actual kernel size represented by `gaussian_blur_r=4`. That result must be recorded in amendment v1.1.

## 11. Statistical and reporting principles

- Treat prompt-level observations as paired across methods and conditions.
- Preserve replicate identity; do not pool observations as if all 5,000 prompts were independent when estimating uncertainty.
- Pre-specify aggregation, confidence-interval, and multiple-comparison procedures in amendment v1.1.
- Report failures and missing samples; do not silently drop or replace them.
- Preserve both distances and signed detection scores so directionality is auditable.
- Clearly label development, pilot, supplemental, and formal results.

## 12. B200 format validation

At commit `296a6225c9bb2db2c7373086f2b15c8deb192fb9`, run `manifest_v1_b200_smoke` used one development prompt, one generation step, and one inversion step on an NVIDIA B200. It recorded `no_w_metric=74.5`, `w_metric=24.953125`, `git.dirty=false`, total duration `108.99201348703355` seconds, and sample duration `3.396694979048334` seconds. The record schema was verified.

This execution validates format and plumbing only. Because it contains one sample and one step, its AUC, accuracy, and TPR are not valid paper results.

## 13. Change control

Any change to a frozen input, method definition, metric, threshold, seed/key allocation, attack, or analysis rule requires a versioned amendment that states the reason, evidence, impact, implementation commit, and approvals. Formal execution is gated by a completed [Protocol Amendment v1.1](protocol_amendment_v1_1_template.md) and the [runbook](development_and_execution_runbook_v1.md).
