import torch
from torchvision import transforms
import torch.nn.functional as F

from PIL import Image, ImageFilter
import random
import numpy as np
import copy
from typing import Any, Mapping
import json
import scipy.stats
from experiment_records import validate_method_configuration

def read_json(filename: str) -> Mapping[str, Any]:
    """Returns a Python dict representation of JSON object at input file."""
    with open(filename) as fp:
        return json.load(fp)


def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def latents_to_imgs(pipe, latents):
    x = pipe.decode_image(latents)
    x = pipe.torch_to_numpy(x)
    x = pipe.numpy_to_pil(x)
    return x


def image_distortion(img1, img2, seed, args):
    if args.r_degree is not None:
        img1 = transforms.RandomRotation((args.r_degree, args.r_degree))(img1)
        img2 = transforms.RandomRotation((args.r_degree, args.r_degree))(img2)

    if args.jpeg_ratio is not None:
        img1.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img1 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")
        img2.save(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img2 = Image.open(f"tmp_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(seed)
        img1 = transforms.RandomResizedCrop(img1.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img1)
        set_random_seed(seed)
        img2 = transforms.RandomResizedCrop(img2.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img2)

    if args.gaussian_blur_r is not None:
        img1 = img1.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))
        img2 = img2.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img1).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img1 = Image.fromarray(np.clip(np.array(img1) + g_noise, 0, 255))
        img2 = Image.fromarray(np.clip(np.array(img2) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img1 = transforms.ColorJitter(brightness=args.brightness_factor)(img1)
        img2 = transforms.ColorJitter(brightness=args.brightness_factor)(img2)

    return img1, img2


def measure_similarity(images, prompt, model, clip_preprocess, tokenizer, device):
    with torch.no_grad():
        img_batch = [clip_preprocess(i).unsqueeze(0) for i in images]
        img_batch = torch.concatenate(img_batch).to(device)
        image_features = model.encode_image(img_batch)

        text = tokenizer([prompt]).to(device)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        return (image_features @ text_features.T).mean(-1)


def get_dataset(args):
    prompt_file = getattr(args, 'prompt_file', None)

    if prompt_file:
        with open(prompt_file, encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]

        if not prompts:
            raise ValueError(f'No prompts found in {prompt_file}')

        dataset = [{'Prompt': prompt} for prompt in prompts]
        return dataset, 'Prompt'

    from datasets import load_dataset

    if 'laion' in args.dataset:
        dataset = load_dataset(args.dataset)['train']
        prompt_key = 'TEXT'
    elif 'coco' in args.dataset:
        with open('fid_outputs/coco/meta_data.json', encoding='utf-8') as f:
            dataset = json.load(f)
            dataset = dataset['annotations']
            prompt_key = 'caption'
    else:
        dataset = load_dataset(args.dataset)['test']
        prompt_key = 'Prompt'

    return dataset, prompt_key

def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]

    return ((x - x0)**2 + (y-y0)**2)<= r**2


def get_watermarking_mask(init_latents_w, args, device):
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if args.w_mask_shape == 'circle':
        np_mask = circle_mask(init_latents_w.shape[-1], r=args.w_radius)
        torch_mask = torch.tensor(np_mask).to(device)

        if args.w_channel == -1:
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, args.w_channel] = torch_mask
    elif args.w_mask_shape == 'square':
        anchor_p = init_latents_w.shape[-1] // 2
        if args.w_channel == -1:
            watermarking_mask[:, :, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
        else:
            watermarking_mask[:, args.w_channel, anchor_p-args.w_radius:anchor_p+args.w_radius, anchor_p-args.w_radius:anchor_p+args.w_radius] = True
    elif args.w_mask_shape == 'no':
        pass
    else:
        raise NotImplementedError(f'w_mask_shape: {args.w_mask_shape}')
    return watermarking_mask


def get_watermarking_pattern(pipe, args, device, shape=None):
    set_random_seed(args.w_seed)
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        gt_init = pipe.get_random_latents()

    if 'seed_ring' in args.w_pattern:
        gt_patch = gt_init
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    elif 'seed_zeros' in args.w_pattern:
        gt_patch = gt_init * 0
    elif 'seed_rand' in args.w_pattern:
        gt_patch = gt_init
    elif 'rand' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch[:] = gt_patch[0]
    elif 'zeros' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
    elif 'const' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2)) * 0
        gt_patch += args.w_pattern_const
    elif 'ring' in args.w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(args.w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()
    else:
        raise NotImplementedError(f'w_pattern: {args.w_pattern}')

    return gt_patch


def _hermitian_partner(values):
    values = torch.flip(values, dims=(-2, -1))
    return torch.roll(values, shifts=(1, 1), dims=(-2, -1))




def inject_watermark(latents, mask, gt_patch, args, saliency_map=None):
    """Inject a Tree-Ring key, optionally modulating its spatial residual."""
    if args.w_injection == 'complex':
        latents_fft = torch.fft.fftshift(torch.fft.fft2(latents), dim=(-1, -2))
        latents_fft_wm = latents_fft.clone()
        latents_fft_wm[mask] = gt_patch[mask].clone()
        latents_wm_standard = torch.fft.ifft2(torch.fft.ifftshift(latents_fft_wm, dim=(-1, -2))).real
    elif args.w_injection == 'seed':
        latents_wm_standard = latents.clone()
        latents_wm_standard[mask] = gt_patch[mask].to(latents.dtype).clone()
    else:
        raise NotImplementedError(f'w_injection: {args.w_injection}')

    if saliency_map is None:
        return latents_wm_standard.to(latents.dtype)

    if saliency_map.ndim != 4 or saliency_map.shape[1] != 1:
        raise ValueError('saliency_map must have shape (B, 1, H, W)')
    if saliency_map.shape[0] not in (1, latents.shape[0]):
        raise ValueError('saliency_map batch size must be 1 or match the latent batch size')

    saliency_map = saliency_map.to(device=latents.device, dtype=latents.dtype)
    if saliency_map.shape[-2:] != latents.shape[-2:]:
        saliency_map = F.interpolate(
            saliency_map,
            size=latents.shape[-2:],
            mode='bilinear',
            align_corners=False,
        )
    saliency_map = saliency_map.clamp(0, 1)

    background_strength = getattr(args, 'saliency_background_strength', 1.5)
    foreground_strength = getattr(args, 'saliency_foreground_strength', 0.2)
    saliency_power = getattr(args, 'saliency_power', 2.0)
    if background_strength < 0 or foreground_strength < 0 or saliency_power <= 0:
        raise ValueError('saliency strengths must be non-negative and saliency_power must be positive')

    saliency_map = saliency_map.pow(saliency_power)
    adaptive_weight = (
        background_strength * (1.0 - saliency_map)
        + foreground_strength * saliency_map
    )
    residual = latents_wm_standard - latents
    latents_final = latents + residual * adaptive_weight

    # Spatial modulation spreads energy across frequencies. Re-project the
    # standard key and its Hermitian partners so taking the real component
    # cannot leak adaptive energy back into the detector's frequency mask.
    if args.w_injection == 'complex':
        standard_fft = torch.fft.fftshift(
            torch.fft.fft2(latents_wm_standard), dim=(-1, -2)
        )
        latents_final_fft = torch.fft.fftshift(
            torch.fft.fft2(latents_final), dim=(-1, -2)
        )
        projection_mask = mask | _hermitian_partner(mask)
        latents_final_fft[projection_mask] = standard_fft[projection_mask]
        latents_final = torch.fft.ifft2(
            torch.fft.ifftshift(latents_final_fft, dim=(-1, -2))
        ).real
    else:
        latents_final[mask] = gt_patch[mask].to(latents.dtype).clone()

    return latents_final.to(latents.dtype)

def inject_watermark_for_method(latents, mask, gt_patch, args):
    """Apply Original or uniformly scale its residual without changing inputs."""
    parameters = validate_method_configuration(
        args.method_name, getattr(args, 'global_alpha', None)
    )
    original = inject_watermark(latents, mask, gt_patch, args)
    if args.method_name == 'original_tree_ring':
        return original

    alpha = parameters['alpha']
    # Exact endpoints avoid cancellation/rounding in z + (z_original - z).
    if alpha == 0:
        return latents.clone()
    if alpha == 1:
        return original

    # Half-precision inputs use float32 arithmetic, then return to the input
    # dtype. The recorded budget is measured after this final conversion.
    working_dtype = (
        torch.float32
        if latents.dtype in (torch.float16, torch.bfloat16)
        else latents.dtype
    )
    base = latents.to(working_dtype)
    original = original.to(working_dtype)
    return (base + alpha * (original - base)).to(latents.dtype)


def measure_latent_residual(base_latents, injected_latents):
    """Measure realized RMS/L2 over all elements of one sample's latent."""
    if base_latents.shape != injected_latents.shape:
        raise ValueError('Base and injected latents must have the same shape')
    if base_latents.numel() == 0:
        raise ValueError('Latents must not be empty')
    if base_latents.device != injected_latents.device:
        raise ValueError('Base and injected latents must be on the same device')
    if not base_latents.is_floating_point() or not injected_latents.is_floating_point():
        raise ValueError('Latents must be real floating-point tensors')

    # Promote before subtraction, so the budget describes the actual tensor
    # values rather than a half-precision approximation to their difference.
    residual = (
        injected_latents.detach().to(torch.float64)
        - base_latents.detach().to(torch.float64)
    )
    if not torch.isfinite(residual).all():
        raise ValueError('Latent residual must contain only finite values')
    squared_l2 = residual.square().sum()
    if not torch.isfinite(squared_l2):
        raise ValueError('Latent residual budget must be finite')
    return {
        'latent_residual_rms': (squared_l2 / residual.numel()).sqrt().item(),
        'latent_residual_l2': squared_l2.sqrt().item(),
    }


def eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    if 'complex' in args.w_measurement:
        reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))
        reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))
        target_patch = gt_patch
    elif 'seed' in args.w_measurement:
        reversed_latents_no_w_fft = reversed_latents_no_w
        reversed_latents_w_fft = reversed_latents_w
        target_patch = gt_patch
    else:
        raise NotImplementedError(f'w_measurement: {args.w_measurement}')

    if 'l1' in args.w_measurement:
        no_w_metric = torch.abs(reversed_latents_no_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
        w_metric = torch.abs(reversed_latents_w_fft[watermarking_mask] - target_patch[watermarking_mask]).mean().item()
    else:
        raise NotImplementedError(f'w_measurement: {args.w_measurement}')

    return no_w_metric, w_metric


def get_p_value(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args):
    reversed_latents_no_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2))[watermarking_mask].flatten()
    reversed_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents_w), dim=(-1, -2))[watermarking_mask].flatten()
    target_patch = gt_patch[watermarking_mask].flatten()

    target_patch = torch.concatenate([target_patch.real, target_patch.imag])

    # no_w
    reversed_latents_no_w_fft = torch.concatenate([reversed_latents_no_w_fft.real, reversed_latents_no_w_fft.imag])
    sigma_no_w = reversed_latents_no_w_fft.std()
    lambda_no_w = (target_patch ** 2 / sigma_no_w ** 2).sum().item()
    x_no_w = (((reversed_latents_no_w_fft - target_patch) / sigma_no_w) ** 2).sum().item()
    p_no_w = scipy.stats.ncx2.cdf(x=x_no_w, df=len(target_patch), nc=lambda_no_w)

    # w
    reversed_latents_w_fft = torch.concatenate([reversed_latents_w_fft.real, reversed_latents_w_fft.imag])
    sigma_w = reversed_latents_w_fft.std()
    lambda_w = (target_patch ** 2 / sigma_w ** 2).sum().item()
    x_w = (((reversed_latents_w_fft - target_patch) / sigma_w) ** 2).sum().item()
    p_w = scipy.stats.ncx2.cdf(x=x_w, df=len(target_patch), nc=lambda_w)

    return p_no_w, p_w
