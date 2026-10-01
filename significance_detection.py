import cv2
import numpy as np
import torch
from PIL import Image


def _to_bgr_uint8(image):
    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert('RGB'))
    elif torch.is_tensor(image):
        if image.ndim != 3 or image.shape[0] not in (1, 3, 4):
            raise ValueError('tensor images must have shape (C, H, W)')
        image = image.detach().cpu().float()
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        elif image.shape[0] == 4:
            image = image[:3]
        image_min = image.min().item()
        image_max = image.max().item()
        if image_min >= -1.0 and image_max <= 1.0 and image_min < 0:
            image = (image + 1.0) / 2.0
        elif image_min < 0.0 or image_max > 1.0:
            raise ValueError('tensor image values must be in [-1, 1] or [0, 1]')
        rgb = (image.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    else:
        array = np.asarray(image)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError('array images must have shape (H, W, 3) or (H, W, 4)')
        rgb = array[:, :, :3].clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _compute_saliency(image):
    if not hasattr(cv2, 'saliency'):
        raise RuntimeError(
            'OpenCV saliency is unavailable; install opencv-contrib-python, '
            'not opencv-python.'
        )

    bgr = _to_bgr_uint8(image)
    detector = cv2.saliency.StaticSaliencySpectralResidual_create()
    success, saliency_map = detector.computeSaliency(bgr)
    if not success:
        return np.zeros(bgr.shape[:2], dtype=np.float32)

    saliency_map = cv2.GaussianBlur(saliency_map, (15, 15), 0)
    saliency_map = saliency_map.astype(np.float32)
    saliency_min = float(saliency_map.min())
    saliency_range = float(saliency_map.max()) - saliency_min
    if saliency_range <= np.finfo(np.float32).eps:
        return np.zeros_like(saliency_map, dtype=np.float32)
    return (saliency_map - saliency_min) / saliency_range


def get_saliency_weight_from_tensor(img_input):
    """Return normalized saliency maps with shape ``(B, 1, H, W)``."""
    if torch.is_tensor(img_input) and img_input.ndim == 4:
        images = list(img_input)
    elif isinstance(img_input, (list, tuple)):
        if not img_input:
            raise ValueError('img_input must contain at least one image')
        images = list(img_input)
    else:
        images = [img_input]

    saliency_maps = [_compute_saliency(image) for image in images]
    expected_shape = saliency_maps[0].shape
    if any(saliency_map.shape != expected_shape for saliency_map in saliency_maps):
        raise ValueError('all images in a batch must have the same spatial size')
    return np.stack(saliency_maps, axis=0)[:, None, :, :].astype(np.float32)
