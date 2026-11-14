# Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust

<img src=scripts/teaser.png  width="80%" height="60%">

This code is the official implementation of [Tree-Ring Watermarks](http://arxiv.org/abs/2305.20030).

If you have any questions, feel free to email Yuxin (<ywen@umd.edu>).

## TreeRing-Advancement status

The active advancement work is on branch `stage-c-record-identity`. Commit
`084658ed1d3bc6d39094f7faaa07816a938cfb25` implements the frozen Stage F
offline quality metrics on top of the verified pre-Stage-F baseline
`1a6c191f6e8d87958788cd6d615e63e07d21d1da`.

- Stages B, C, D, and E are complete.
- The 32-sample Development image-generation and identity checks are complete
  for the No-Watermark reference, Original Tree-Ring, Globally Weaker
  Tree-Ring, and Saliency-Aware Tree-Ring.
- Stage F is complete. The frozen CLIP, LPIPS, PSNR, and SSIM implementation
  passed the local and server CPU gates, a one-pair B200 smoke, and strict
  read-only evaluation of the preserved 32-pair Original, Global alpha=0.5,
  and Saliency Development runs. The final 32x3 evidence package was reviewed
  and archived locally on 2026-11-14.
- Existing paired RGB PNGs must be reused for Stage F. Diffusion generation
  must not be repeated merely to calculate CLIP, LPIPS, PSNR, or SSIM.
- Development parameters, including the current Saliency probe values, are
  implementation checks rather than final parameters or paper conclusions.

The project server does not contain the fixed 5,000-image MS-COCO reference
set required by the baseline FID protocol. The 32-sample Development images
are not a valid substitute for that reference set, and FID must not be
estimated from them. The protocol owner approved omitting FID on 2026-10-21
because adding it would require importing an unverified reference package and
running a separate large-scale generation experiment. The decision and its
pilot-independent rationale are recorded in protocol amendment v1.1.

The shortest controlled path to completion is now:

1. Freeze the Pilot parameter grids and decision rules before examining Pilot
   outcomes.
2. Run and audit the 200-sample Pilot, then complete and approve amendment
   v1.1, including the FID decision and selected parameters/threshold.
3. Run the formal and robustness experiments without changing the frozen
   protocol, then perform the pre-specified paired analysis.

No 200-sample Pilot run has started. The Development values are descriptive
implementation evidence only and must not be used as final parameter choices
or paper conclusions.

See `docs/protocols/` for the controlled protocol, record schema, execution
runbook, and amendment template. The original usage examples below are kept
for upstream reproducibility; they do not authorize Advancement experiments.

## About
We propose Tree-Ring Watermarking to watermark diffusion model outputs. Tree-Ring Watermarking chooses the initial noise array so that its Fourier transform contains a carefully constructed pattern near its center. This pattern is called the key. This initial noise vector is then converted into an image using the standard diffusion pipeline with no modifications. To detect the watermark in an image, the diffusion model is inverted to retrieve the original noise array used for generation. This array is then checked to see whether the key is present.

## Dependencies
- Python 3.8-3.10
- PyTorch == 1.13.0
- transformers == 4.23.1
- diffusers == 0.11.1
- datasets

Note: higher diffusers version may not be compatible with the DDIM inversion code.

## Usage

### Perform main experiments and calculate CLIP Score
For non-adversarial case, you can simply run:
```
python run_tree_ring_watermark.py --run_name no_attack --w_channel 3 --w_pattern ring --start 0 --end 1000 --with_tracking --reference_model ViT-g-14 --reference_model_pretrain laion2b_s12b_b42k
```

You can modify arguments to perform attack. For example, for rotation with 75 degrees:
```
python run_tree_ring_watermark.py --run_name rotation --w_channel 3 --w_pattern ring --r_degree 75 --start 0 --end 1000 --with_tracking
```

For more adversarial cases, see [here](scripts/tree_ring.sh).

For other watermark types mentioned in the paper, you can check [scripts/](scripts/).

### Legacy upstream FID workflow

The original repository used 5,000 COCO examples available
[here](https://drive.google.com/drive/folders/1saWx-B3vJxzspJ-LaXSEn5Qjm8NIs3r0?usp=sharing), following the format of
`fid_outputs/coco/meta_data.json`.

This legacy command generates new images and may fetch Inception weights. It
is not the frozen TreeRing-Advancement Stage F evaluator and must not be run on
the project server under the current constraints. In particular, do not use
the 32-sample Development PNGs to produce a replacement FID value.

Then, to calculate FID, you may run:
```
python run_tree_ring_watermark_fid.py --run_name fid_run --w_channel 3 --w_pattern ring --start 0 --end 5000 --with_tracking --run_no_w
```

### Perform main experiments for Imagenet Models
You can get the pre-trained models [here](https://github.com/openai/guided-diffusion). For example, the link of the model used by the paper is [256x256 diffusion](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt). Then, you may follow the [script](scripts/tree_ring_imagenet.sh) to run the experiments.

The checkpoint is intentionally excluded from Git. Place it outside the repository or in the repository root, then pass its location explicitly when needed:

```
python run_tree_ring_watermark_imagenet.py --model_path /path/to/256x256_diffusion.pt --w_pattern ring --w_channel 2
```

The ImageNet runner enables saliency-adaptive injection by default. It creates a short DDIM preview, computes one saliency map per image, attenuates the watermark residual in salient regions, and re-projects the Tree-Ring key before generation so that embedding and detection use the same frequency target. The main adaptive parameters are:

- `saliency_preview_steps`: number of DDIM steps used for the preview (default: 10).
- `saliency_background_strength`: residual strength in non-salient regions (default: 1.5).
- `saliency_foreground_strength`: residual strength in salient regions (default: 0.2).
- `saliency_power`: saliency-map shaping exponent (default: 2.0).
- `disable_saliency`: run the standard Tree-Ring injection as an ablation baseline.

The runner reports watermark AUC and TPR together with PSNR and saliency-weighted foreground/background MSE. OpenCV's saliency implementation requires `opencv-contrib-python`.

## Parameters
Crucial hyperparameters for Tree-Ring:

- `w_channel`: the index of the watermarked channel. If set as -1, watermark all channels.
- `w_pattern`: watermark type: zeros, rand, ring.
- `w_radius`: watermark radius.

## Suggestions and Pull Requests are welcome!
