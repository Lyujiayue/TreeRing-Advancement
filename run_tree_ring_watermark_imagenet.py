import argparse
import wandb
import copy
from pathlib import Path
from tqdm import tqdm
from statistics import mean
from sklearn import metrics
import torch
import numpy as np

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    create_gaussian_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from optim_utils import *
from io_utils import *


def measure_image_quality(reference_image, watermarked_image, saliency_map):
    reference = np.asarray(reference_image, dtype=np.float32) / 255.0
    watermarked = np.asarray(watermarked_image, dtype=np.float32) / 255.0
    squared_error = np.mean((reference - watermarked) ** 2, axis=-1)
    mse = float(squared_error.mean())
    psnr = float('inf') if mse == 0 else float(10 * np.log10(1.0 / mse))

    saliency = np.clip(np.asarray(saliency_map, dtype=np.float32), 0, 1)
    salient_weight = float(saliency.sum())
    background = 1.0 - saliency
    background_weight = float(background.sum())
    salient_mse = (
        float((squared_error * saliency).sum() / salient_weight)
        if salient_weight > 1e-8 else mse
    )
    background_mse = (
        float((squared_error * background).sum() / background_weight)
        if background_weight > 1e-8 else mse
    )
    return psnr, salient_mse, background_mse


def main(args):
    if args.end <= args.start:
        raise ValueError('--end must be greater than --start')
    if args.num_images <= 0:
        raise ValueError('--num_images must be positive')
    if args.saliency_preview_steps <= 0:
        raise ValueError('--saliency_preview_steps must be positive')

    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['latent_watermark_fourier_openai'])
        wandb.config.update(args)
        table = wandb.Table(columns=[
            'gen_no_w', 'gen_w', 'no_w_metric', 'w_metric', 'psnr',
            'salient_mse', 'background_mse',
        ])

    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.timestep_respacing = f"ddim{args.num_inference_steps}"
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    diffusion_fast = None
    if not args.disable_saliency:
        diffusion_fast = create_gaussian_diffusion(
            steps=args.diffusion_steps,
            learn_sigma=args.learn_sigma,
            noise_schedule=args.noise_schedule,
            use_kl=args.use_kl,
            predict_xstart=args.predict_xstart,
            rescale_timesteps=args.rescale_timesteps,
            rescale_learned_sigmas=args.rescale_learned_sigmas,
            timestep_respacing=f"ddim{args.saliency_preview_steps}",
        )

    model_path = Path(args.model_path).expanduser()
    if not model_path.is_file():
        raise FileNotFoundError(
            f'Model checkpoint not found: {model_path}. '
            'Pass --model_path or update the model JSON file.'
        )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    shape = (args.num_images, 3, args.image_size, args.image_size)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(None, args, device, shape)

    no_w_metrics = []
    w_metrics = []
    psnr_values = []
    salient_mse_values = []
    background_mse_values = []

    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed

        ### generation
        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.num_images,), device=device
            )
            model_kwargs["y"] = classes

        set_random_seed(seed)
        init_latents_no_w = torch.randn(*shape, device=device)
        outputs_no_w = diffusion.ddim_sample_loop(
                    model=model,
                    shape=shape,
                    noise=init_latents_no_w,
                    model_kwargs=model_kwargs,
                    device=device,
                    return_image=True,
                )
        init_latents_w = copy.deepcopy(init_latents_no_w)

        saliency_map_np = None
        saliency_map = None
        if diffusion_fast is not None:
            from significance_detection import get_saliency_weight_from_tensor

            with torch.no_grad():
                outputs_tmp = diffusion_fast.ddim_sample_loop(
                    model=model,
                    shape=shape,
                    noise=init_latents_w,
                    model_kwargs=model_kwargs,
                    device=device,
                    clip_denoised=True,
                    return_image=True,
                )
            saliency_map_np = get_saliency_weight_from_tensor(outputs_tmp)
            saliency_map = torch.from_numpy(saliency_map_np).to(device=device)

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args, saliency_map=saliency_map)

        outputs_w = diffusion.ddim_sample_loop(
                    model=model,
                    shape=shape,
                    noise=init_latents_w,
                    model_kwargs=model_kwargs,
                    device=device,
                    return_image=True,
                )
        for image_index, (orig_image_no_w, orig_image_w) in enumerate(zip(outputs_no_w, outputs_w)):
            orig_image_no_w_auged, orig_image_w_auged = image_distortion(
                orig_image_no_w, orig_image_w, seed + image_index, args
            )
            reverse_shape = (1, *shape[1:])
            reverse_model_kwargs = {
                key: value[image_index:image_index + 1]
                for key, value in model_kwargs.items()
            }

            reversed_latents_no_w = diffusion.ddim_reverse_sample_loop(
                model=model,
                shape=reverse_shape,
                image=orig_image_no_w_auged,
                model_kwargs=reverse_model_kwargs,
                device=device,
            )

            reversed_latents_w = diffusion.ddim_reverse_sample_loop(
                model=model,
                shape=reverse_shape,
                image=orig_image_w_auged,
                model_kwargs=reverse_model_kwargs,
                device=device,
            )

            sample_mask = watermarking_mask[image_index:image_index + 1]
            sample_patch = gt_patch[image_index:image_index + 1]
            no_w_metric, w_metric = eval_watermark(
                reversed_latents_no_w, reversed_latents_w,
                sample_mask, sample_patch, args,
            )
            no_w_metrics.append(-no_w_metric)
            w_metrics.append(-w_metric)

            quality_map = (
                saliency_map_np[image_index, 0]
                if saliency_map_np is not None
                else np.zeros(shape[-2:], dtype=np.float32)
            )
            psnr, salient_mse, background_mse = measure_image_quality(
                orig_image_no_w, orig_image_w, quality_map
            )
            psnr_values.append(psnr)
            salient_mse_values.append(salient_mse)
            background_mse_values.append(background_mse)

            if args.with_tracking:
                log_images = (
                    args.reference_model is not None
                    and len(w_metrics) <= args.max_num_log_image
                )
                table.add_data(
                    wandb.Image(orig_image_no_w) if log_images else None,
                    wandb.Image(orig_image_w) if log_images else None,
                    no_w_metric, w_metric, psnr, salient_mse, background_mse,
                )

    # ROC analysis
    preds = no_w_metrics + w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)
    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    if args.with_tracking:
        wandb.log({
            'Table': table,
            'auc': auc,
            'acc': acc,
            'TPR@1%FPR': low,
            'psnr': mean(psnr_values),
            'salient_mse': mean(salient_mse_values),
            'background_mse': mean(background_mse_values),
        })

    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')
    print(
        f'psnr: {mean(psnr_values)}, '
        f'salient_mse: {mean(salient_mse_values)}, '
        f'background_mse: {mean(background_mse_values)}'
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='256x256_diffusion')
    parser.add_argument('--model_path', default=None)
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    parser.add_argument('--disable_saliency', action='store_true')
    parser.add_argument('--saliency_preview_steps', default=10, type=int)
    parser.add_argument('--saliency_background_strength', default=1.5, type=float)
    parser.add_argument('--saliency_foreground_strength', default=0.2, type=float)
    parser.add_argument('--saliency_power', default=2.0, type=float)
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    args = parser.parse_args()
    cli_model_path = args.model_path
    args.__dict__.update(model_and_diffusion_defaults())
    args.__dict__.update(read_json(f'{args.model_id}.json'))
    if cli_model_path is not None:
        args.model_path = cli_model_path

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps

    main(args)
