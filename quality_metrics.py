"""Frozen, offline image-quality primitives for preserved RGB PNG evidence."""
from __future__ import annotations

import hashlib
import math
import statistics
from pathlib import Path

import numpy as np
from PIL import Image


QUALITY_METRIC_SPEC_VERSION = "treering-quality-v1"
CLIP_MODEL_NAME = "ViT-g-14"
CLIP_PRETRAINED_TAG = "laion2b_s12b_b42k"
CLIP_CHECKPOINT_SHA256 = "6aac683f899159946bc4ca15228bb7016f3cbb1a2c51f365cba0b23923f344da"
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
LPIPS_LINEAR_CHECKPOINT_SHA256 = "df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0"
ALEXNET_CHECKPOINT_SHA256 = "7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02"
SSIM_SIGMA = 1.5
SSIM_TRUNCATE = 3.5
SSIM_K1 = 0.01
SSIM_K2 = 0.03


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(path, expected_sha256):
    if expected_sha256 is None:
        return file_sha256(path)
    expected_sha256 = str(expected_sha256).lower()
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdef" for character in expected_sha256):
        raise ValueError("Expected SHA-256 must be 64 lowercase hexadecimal characters")
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"Weight SHA-256 mismatch for {path}")
    return actual


def load_rgb_float64(path):
    """Load an RGB image without resizing and return values in [0, 1]."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Image is missing, non-regular, or a symlink: {path}")
    with Image.open(path) as image:
        image.load()
        if image.format != "PNG" or image.mode != "RGB":
            raise ValueError(f"Expected an RGB PNG: {path}")
        array = np.asarray(image, dtype=np.float64) / 255.0
    if array.ndim != 3 or array.shape[2] != 3 or not np.isfinite(array).all():
        raise ValueError(f"Invalid RGB pixels: {path}")
    return array


def _validate_pair(reference, candidate):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise ValueError("Quality inputs must be same-shaped HxWx3 RGB arrays")
    if reference.shape[0] < 11 or reference.shape[1] < 11:
        raise ValueError("SSIM requires images at least 11x11")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("Quality inputs must be finite")
    if np.any(reference < 0) or np.any(reference > 1) or np.any(candidate < 0) or np.any(candidate > 1):
        raise ValueError("Quality inputs must be in [0, 1]")
    return reference, candidate


def psnr_rgb(reference, candidate):
    """Return finite PSNR metadata without serializing JSON Infinity."""
    reference, candidate = _validate_pair(reference, candidate)
    mse = float(np.mean(np.square(reference - candidate), dtype=np.float64))
    if mse == 0.0:
        return {"value_db": None, "status": "exact_match_positive_infinity", "mse": 0.0}
    value = float(10.0 * math.log10(1.0 / mse))
    if not math.isfinite(value):
        raise ValueError("PSNR is non-finite for a non-identical pair")
    return {"value_db": value, "status": "finite", "mse": mse}


def ssim_rgb(reference, candidate):
    """Gaussian-window RGB SSIM matching the frozen v1 definition."""
    reference, candidate = _validate_pair(reference, candidate)
    try:
        from scipy.ndimage import gaussian_filter
    except ImportError as error:
        raise RuntimeError("SciPy is required for frozen SSIM") from error

    c1 = SSIM_K1 ** 2
    c2 = SSIM_K2 ** 2
    scores = []
    radius = int(SSIM_TRUNCATE * SSIM_SIGMA + 0.5)
    for channel in range(3):
        x = reference[:, :, channel]
        y = candidate[:, :, channel]
        ux = gaussian_filter(x, sigma=SSIM_SIGMA, truncate=SSIM_TRUNCATE, mode="reflect")
        uy = gaussian_filter(y, sigma=SSIM_SIGMA, truncate=SSIM_TRUNCATE, mode="reflect")
        uxx = gaussian_filter(x * x, sigma=SSIM_SIGMA, truncate=SSIM_TRUNCATE, mode="reflect")
        uyy = gaussian_filter(y * y, sigma=SSIM_SIGMA, truncate=SSIM_TRUNCATE, mode="reflect")
        uxy = gaussian_filter(x * y, sigma=SSIM_SIGMA, truncate=SSIM_TRUNCATE, mode="reflect")
        vx = np.maximum(uxx - ux * ux, 0.0)
        vy = np.maximum(uyy - uy * uy, 0.0)
        vxy = uxy - ux * uy
        numerator = (2.0 * ux * uy + c1) * (2.0 * vxy + c2)
        denominator = (ux * ux + uy * uy + c1) * (vx + vy + c2)
        score_map = numerator / denominator
        if radius:
            score_map = score_map[radius:-radius, radius:-radius]
        score = float(np.mean(score_map, dtype=np.float64))
        if not math.isfinite(score):
            raise ValueError("SSIM is non-finite")
        scores.append(score)
    return float(statistics.fmean(scores))


def scalar_summary(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("Cannot aggregate empty or non-finite values")
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": float(statistics.fmean(values)),
        "sample_std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "median": float(statistics.median(ordered)),
        "min": ordered[0],
        "max": ordered[-1],
    }


def psnr_summary(records):
    finite = [record["value_db"] for record in records if record["status"] == "finite"]
    exact = sum(record["status"] == "exact_match_positive_infinity" for record in records)
    if len(finite) + exact != len(records):
        raise ValueError("Unknown PSNR status")
    return {
        "count": len(records),
        "finite": scalar_summary(finite) if finite else None,
        "exact_match_count": exact,
        "exact_match_fraction": exact / len(records) if records else 0.0,
    }


class ClipScorer:
    """OpenCLIP scorer that requires an explicit local checkpoint."""

    def __init__(self, model_name, checkpoint_path, device="cpu", expected_sha256=None):
        if model_name != CLIP_MODEL_NAME:
            raise ValueError(f"Frozen CLIP model must be {CLIP_MODEL_NAME}")
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
            raise ValueError("CLIP checkpoint must be an existing regular file")
        if expected_sha256 != CLIP_CHECKPOINT_SHA256:
            raise ValueError("CLIP checkpoint SHA-256 must equal the frozen protocol hash")
        checkpoint_sha256 = _require_sha256(checkpoint_path, expected_sha256)
        import open_clip
        import torch

        try:
            from open_clip.version import __version__ as open_clip_version
        except (ImportError, AttributeError):
            open_clip_version = getattr(open_clip, "__version__", "unknown")

        self.torch = torch
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=str(checkpoint_path), device=device,
            image_mean=CLIP_IMAGE_MEAN, image_std=CLIP_IMAGE_STD,
        )
        self.model = self.model.eval().float()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.provenance = {
            "implementation": "open_clip",
            "version": open_clip_version,
            "model_name": model_name,
            "pretrained_tag": CLIP_PRETRAINED_TAG,
            "checkpoint_path": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "preprocess": repr(self.preprocess),
            "image_size": 224,
            "image_mean": list(CLIP_IMAGE_MEAN),
            "image_std": list(CLIP_IMAGE_STD),
            "device": device,
            "dtype": "float32",
            "batching": "two images and one prompt per sample; samples processed sequentially",
            "similarity": "L2-normalized image/text embedding cosine; no scale factor",
        }

    def score_pair(self, no_watermark_image, watermarked_image, prompt):
        torch = self.torch
        with torch.no_grad():
            images = torch.stack([
                self.preprocess(no_watermark_image),
                self.preprocess(watermarked_image),
            ]).to(self.device, dtype=torch.float32)
            text = self.tokenizer([prompt]).to(self.device)
            image_features = self.model.encode_image(images)
            text_features = self.model.encode_text(text)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            values = (image_features @ text_features.T).squeeze(1).cpu().tolist()
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("CLIP returned invalid scores")
        return float(values[0]), float(values[1])


class LpipsScorer:
    """LPIPS-v0.1 AlexNet scorer using two explicit local weight files."""

    def __init__(self, linear_checkpoint, alexnet_checkpoint, device="cpu",
                 expected_linear_sha256=None, expected_alexnet_sha256=None):
        linear_checkpoint = Path(linear_checkpoint)
        alexnet_checkpoint = Path(alexnet_checkpoint)
        for path in (linear_checkpoint, alexnet_checkpoint):
            if path.is_symlink() or not path.is_file():
                raise ValueError("LPIPS weights must be existing regular files")
        if expected_linear_sha256 != LPIPS_LINEAR_CHECKPOINT_SHA256:
            raise ValueError("LPIPS linear checkpoint SHA-256 must equal the frozen protocol hash")
        if expected_alexnet_sha256 != ALEXNET_CHECKPOINT_SHA256:
            raise ValueError("AlexNet checkpoint SHA-256 must equal the frozen protocol hash")
        linear_sha256 = _require_sha256(linear_checkpoint, expected_linear_sha256)
        alexnet_sha256 = _require_sha256(alexnet_checkpoint, expected_alexnet_sha256)
        import torch
        from frozen_lpips import (
            LPIPS_IMPLEMENTATION_VERSION,
            LPIPS_UPSTREAM_COMMIT,
            create_frozen_lpips_alex,
        )

        self.torch = torch
        self.device = device
        self.model = create_frozen_lpips_alex(
            linear_checkpoint, alexnet_checkpoint
        ).to(device).eval().float()
        self.provenance = {
            "implementation": "frozen_lpips.create_frozen_lpips_alex",
            "version": LPIPS_IMPLEMENTATION_VERSION,
            "upstream_commit": LPIPS_UPSTREAM_COMMIT,
            "network": "alex",
            "lpips_version": "0.1",
            "linear_checkpoint_path": str(linear_checkpoint.resolve()),
            "linear_checkpoint_sha256": linear_sha256,
            "alexnet_checkpoint_path": str(alexnet_checkpoint.resolve()),
            "alexnet_checkpoint_sha256": alexnet_sha256,
            "device": device,
            "dtype": "float32",
            "batching": "one image pair per sample; samples processed sequentially",
            "input": "RGB [0,1] mapped to [-1,1]",
            "spatial": False,
        }

    def score(self, reference, candidate):
        reference, candidate = _validate_pair(reference, candidate)
        torch = self.torch
        tensors = []
        for array in (reference, candidate):
            tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0)
            tensors.append((tensor.to(self.device, dtype=torch.float32) * 2.0) - 1.0)
        with torch.no_grad():
            value = float(self.model(tensors[0], tensors[1]).reshape(-1)[0].cpu().item())
        if not math.isfinite(value):
            raise ValueError("LPIPS returned a non-finite score")
        return value
