import unittest
from types import SimpleNamespace

import torch

from optim_utils import (
    get_watermarking_mask,
    get_watermarking_pattern,
    inject_watermark,
)


def make_args(**overrides):
    values = {
        'w_channel': 1,
        'w_mask_shape': 'circle',
        'w_radius': 3,
        'w_seed': 123,
        'w_pattern': 'rand',
        'w_pattern_const': 0,
        'w_injection': 'complex',
        'saliency_background_strength': 1.5,
        'saliency_foreground_strength': 0.2,
        'saliency_power': 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class WatermarkInjectionTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')
        self.latents = torch.randn(2, 3, 16, 16, device=self.device)
        self.args = make_args()
        self.mask = get_watermarking_mask(self.latents, self.args, self.device)
        self.patch = get_watermarking_pattern(
            None, self.args, self.device, self.latents.shape
        )

    def test_standard_complex_injection_only_replaces_mask(self):
        injected = inject_watermark(
            self.latents, self.mask, self.patch, self.args
        )
        expected_fft = torch.fft.fftshift(
            torch.fft.fft2(self.latents), dim=(-1, -2)
        )
        expected_fft[self.mask] = self.patch[self.mask]
        expected = torch.fft.ifft2(
            torch.fft.ifftshift(expected_fft, dim=(-1, -2))
        ).real
        self.assertTrue(torch.allclose(injected, expected, atol=1e-6, rtol=1e-6))

    def test_adaptive_injection_matches_standard_key(self):
        saliency = torch.rand(2, 1, 16, 16)
        standard = inject_watermark(
            self.latents, self.mask, self.patch, self.args
        )
        injected = inject_watermark(
            self.latents, self.mask, self.patch, self.args, saliency
        )
        standard_fft = torch.fft.fftshift(
            torch.fft.fft2(standard), dim=(-1, -2)
        )
        injected_fft = torch.fft.fftshift(
            torch.fft.fft2(injected), dim=(-1, -2)
        )
        self.assertEqual(injected.shape, self.latents.shape)
        self.assertEqual(injected.dtype, self.latents.dtype)
        self.assertTrue(torch.allclose(
            injected_fft[self.mask], standard_fft[self.mask], atol=1e-4, rtol=1e-4
        ))

    def test_original_pattern_modes_are_preserved(self):
        zeros_args = make_args(w_pattern='zeros')
        zeros = get_watermarking_pattern(
            None, zeros_args, self.device, self.latents.shape
        )
        self.assertEqual(torch.count_nonzero(zeros).item(), 0)

        const_args = make_args(w_pattern='const', w_pattern_const=2.5)
        const = get_watermarking_pattern(
            None, const_args, self.device, self.latents.shape
        )
        self.assertTrue(torch.all(const == 2.5))


if __name__ == '__main__':
    unittest.main()
