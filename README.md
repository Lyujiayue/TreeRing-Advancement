# Tree-Ring Watermarks: Fingerprints for Diffusion Images that are Invisible and Robust

<img src=scripts/teaser.png  width="80%" height="60%">

This code is the official implementation of [Tree-Ring Watermarks](http://arxiv.org/abs/2305.20030).

If you have any questions, feel free to email Yuxin (<ywen@umd.edu>).

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

### Calculate FID
You can download 5000 COCO examples used in the paper [here](https://drive.google.com/drive/folders/1saWx-B3vJxzspJ-LaXSEn5Qjm8NIs3r0?usp=sharing). Feel free to add more data or other datasets according to the format of `fid_outputs/coco/meta_data.json`.

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
