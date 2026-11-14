# Stage F Offline Quality Metrics

## Scope

Stage F evaluates preserved paired RGB PNGs without invoking the diffusion
pipeline or changing any source run artifact. The evaluator writes a separate,
exclusive output directory and binds every result to the source summary,
JSONL, PNG hashes, metric implementation, package versions, and model weights.

FID is omitted from protocol v1.1 by the protocol owner's 2026-10-21 decision.
The decision preceded Pilot and Formal results. The server has no pre-frozen,
verifiable 5,000-image MS-COCO reference set, and the 32-image Development set
is not a valid substitute. A future FID study must be separately amended and
identified as supplemental before its reference data or generated images are
examined.

## Frozen metric definitions

- CLIP: OpenCLIP `ViT-g-14`, raw cosine similarity between L2-normalized image
  and prompt embeddings, no `logit_scale` multiplier. The model checkpoint is
  an explicit existing local file for `laion2b_s12b_b42k`; evaluation uses a
  224px bicubic resize/center crop, RGB conversion, tensor conversion, and the
  fixed OpenAI mean `(0.48145466, 0.4578275, 0.40821073)` and standard deviation
  `(0.26862954, 0.26130258, 0.27577711)`. Its SHA-256 and full transform
  representation are recorded. No download fallback is allowed.
- LPIPS: learned LPIPS v0.1 with AlexNet, FP32, spatial reduction disabled,
  paired Watermarked versus No-Watermark RGB input mapped from `[0,1]` to
  `[-1,1]`. The repository-frozen inference implementation follows upstream
  commit `c33f89e9f46522a584cf41d8880eb0afa982708b`; both the LPIPS linear weights
  and AlexNet weights must be explicit local files before execution and are
  hashed. No `lpips` installation or torch-cache mutation is required.
- PSNR: paired RGB pixels converted to float64 `[0,1]`; MSE covers all pixels
  and channels; finite values use `10*log10(1/MSE)`. Exact matches record a
  null numeric value and explicit positive-infinity status rather than invalid
  JSON or an arbitrary cap.
- SSIM: per-channel Gaussian SSIM with an 11x11 window, sigma 1.5, truncate
  3.5, reflect padding, population covariance, K1 0.01, K2 0.03, data range
  1.0, valid-region mean, followed by the arithmetic mean of RGB channels.

Each per-sample record contains both CLIP values and their paired delta,
LPIPS, PSNR, SSIM, immutable sample identity, and the two source PNG hashes.
CLIP evaluates the two paired images together with one prompt; LPIPS evaluates
one image pair; samples are processed sequentially for deterministic memory
use. Any missing weight, wrong frozen hash, altered input, or non-finite output
fails the run rather than falling back to a download or substitute metric.
Run aggregation reports count, arithmetic mean, sample standard deviation,
median, minimum, and maximum. Exact-match PSNR observations are counted and
reported separately. Formal analysis must retain replicate identity rather
than pooling five replicates as 5,000 independent observations.

## Execution and evidence boundary

`scripts/quality_validation.py` requires explicit input paths, explicit local
weight paths, and a new output directory. It first verifies summary/JSONL
agreement and the existing paired-image contract. Missing, extra, altered,
non-RGB, non-PNG, non-finite, or identity-incomplete inputs fail closed.

Server execution remains a separately approved step. No dependency
installation or automatic model download is permitted, and quality evaluation
must never regenerate diffusion images.

The reviewed Stage F execution is complete at implementation commit
`084658ed1d3bc6d39094f7faaa07816a938cfb25`. The local and server CPU suites
passed 67 tests, the three fixed assets passed size and SHA-256 validation, and
the repository-frozen LPIPS implementation matched official `lpips==0.1.4` on
three fixtures with maximum absolute difference 0.0. A one-pair B200 smoke
then passed before the preserved Development runs were evaluated.

The final read-only Development evaluation used one NVIDIA B200, driver
590.48.01, and the existing server environment. Each of the Original,
Globally Weaker alpha=0.5, and Saliency-Aware runs contributed exactly 32
paired images. All three reports record CUDA/FP32 CLIP and LPIPS execution,
the frozen weight hashes, immutable source-summary/JSONL/PNG hashes, and the
same No-Watermark reference images. The only console warning was the already
reviewed timm import deprecation.

| Development run | CLIP No-W mean | CLIP W mean | Paired delta mean | LPIPS mean | PSNR mean (dB) | SSIM mean |
|---|---:|---:|---:|---:|---:|---:|
| Original | 0.3668062855 | 0.3635695009 | -0.0032367846 | 0.5279564308 | 13.6259966542 | 0.4710933691 |
| Global alpha=0.5 | 0.3668062855 | 0.3633200470 | -0.0034862384 | 0.4037110256 | 16.9494370310 | 0.5810611632 |
| Saliency | 0.3668062855 | 0.3639491461 | -0.0028571393 | 0.5281274477 | 13.6421690542 | 0.4707778081 |

These are descriptive Development values, not parameter-selection results or
paper conclusions. The 200-sample Pilot has not started.

The server strict status is `STAGE_F_32X3_STRICT_READONLY_ACCEPT_PASS`; the
independent local package status is
`STAGE_F_32X3_LOCAL_PACKAGE_REVIEW_PASS`. The evidence archive contains 20
regular, hash-verified members. Its tar.gz SHA-256 is
`12188ffdbd8adc75a26bc8b99e69d954ff4b5561956f6b44ed02e108db78ba45`,
and its package-manifest SHA-256 is
`fcabf09bc5acb541a07ffc9ed3f4d1b0c39165c543687cfca2f4039c0d80741e`.
