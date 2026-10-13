"""Lossless, exclusive paired image evidence; no model or RNG operations."""
import hashlib
from pathlib import Path

from PIL import Image

from experiment_records import file_sha256

IMAGE_CONTRACT = 'treering-paired-rgb-png-v1'


def pixel_sha256(image):
    digest = hashlib.sha256()
    digest.update(f'{image.mode}:{image.width}:{image.height}:'.encode('ascii'))
    digest.update(image.tobytes())
    return digest.hexdigest()


def save_image_pair(directory, sample_index, no_watermark, watermarked):
    """Save untouched RGB images before attacks; retain partial files on failure."""
    directory = Path(directory)
    result = {}
    for role, image in (('no_watermark', no_watermark), ('watermarked', watermarked)):
        if not isinstance(image, Image.Image) or image.mode != 'RGB':
            raise ValueError('Image evidence requires original PIL RGB images')
        path = directory / f'{sample_index:04d}.{role}.png'
        pixels = pixel_sha256(image)
        with path.open('xb') as stream:
            image.save(stream, format='PNG')
        with Image.open(path) as saved:
            saved.load()
            if pixel_sha256(saved) != pixels:
                raise ValueError('PNG round trip changed generated pixels')
        result[role] = dict(filename=path.name, sha256=file_sha256(path),
                            pixel_sha256=pixels, mode=image.mode,
                            width=image.width, height=image.height)
    return dict(contract=IMAGE_CONTRACT, **result)


def verify_image_pair(directory, sample_index, artifacts, expected_size=512):
    """Use canonical paths, never paths supplied by an untrusted record."""
    if set(artifacts) != {'contract', 'no_watermark', 'watermarked'} or artifacts['contract'] != IMAGE_CONTRACT:
        raise ValueError('Image evidence contract mismatch')
    hashes = {}
    for role in ('no_watermark', 'watermarked'):
        info = artifacts[role]
        name = f'{sample_index:04d}.{role}.png'
        path = Path(directory) / name
        if path.is_symlink() or not path.is_file() or info['filename'] != name:
            raise ValueError('Missing or noncanonical image evidence')
        if info['mode'] != 'RGB' or (info['width'], info['height']) != (expected_size, expected_size):
            raise ValueError('Recorded image mode/size mismatch')
        actual_hash = file_sha256(path)
        if info['sha256'] != actual_hash:
            raise ValueError('Image file hash mismatch')
        with Image.open(path) as image:
            image.load()
            if image.format != 'PNG' or image.mode != 'RGB' or image.size != (expected_size, expected_size):
                raise ValueError('Actual image format/mode/size mismatch')
            if info['pixel_sha256'] != pixel_sha256(image):
                raise ValueError('Image pixel hash mismatch')
        hashes[str(path)] = actual_hash
    return hashes
