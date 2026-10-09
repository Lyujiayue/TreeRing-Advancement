import unittest
from types import SimpleNamespace

import torch

from optim_utils import (
    get_watermarking_mask,
    get_watermarking_pattern,
    inject_watermark,
    inject_watermark_for_method,
    measure_latent_residual,
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
        'method_name': 'original_tree_ring',
        'global_alpha': None,
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

    def test_method_dispatch_preserves_original_complex_and_seed_injection(self):
        for injection in ('complex', 'seed'):
            args = make_args(w_injection=injection)
            patch = self.patch if injection == 'complex' else self.patch.real
            with self.subTest(injection=injection):
                expected = inject_watermark(self.latents, self.mask, patch, args)
                actual = inject_watermark_for_method(self.latents, self.mask, patch, args)
                self.assertTrue(torch.equal(actual, expected))

    def test_global_endpoints_and_intermediate_values_across_dtypes(self):
        # CPU FFT does not support float16/bfloat16; use real seed injection
        # for those dtypes, while testing the frozen complex path at float32/64.
        for dtype, injection in ((torch.float32, 'complex'), (torch.float64, 'complex'),
                                 (torch.float16, 'seed'), (torch.bfloat16, 'seed')):
            base = self.latents.to(dtype)
            args = make_args(w_injection=injection, method_name='globally_weaker_tree_ring')
            key = (self.patch.to(torch.complex128 if dtype == torch.float64 else torch.complex64)
                   if injection == 'complex' else self.patch.real.to(dtype))
            original = inject_watermark(base, self.mask, key, args)
            working_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
            for alpha in (0.0, 0.25, 0.5, 1.0):
                with self.subTest(dtype=dtype, injection=injection, alpha=alpha):
                    args.global_alpha = alpha
                    actual = inject_watermark_for_method(base, self.mask, key, args)
                    if alpha == 0:
                        expected = base
                    elif alpha == 1:
                        expected = original
                    else:
                        expected = (base.to(working_dtype) + alpha * (
                            original.to(working_dtype) - base.to(working_dtype)
                        )).to(dtype)
                    self.assertTrue(torch.equal(actual, expected))
                    self.assertEqual(actual.dtype, dtype)
                    self.assertEqual(actual.device, base.device)
                    self.assertEqual(actual.shape, base.shape)

    def test_injection_does_not_mutate_or_alias_shared_inputs_or_consume_rng(self):
        base = self.latents.clone()
        mask = self.mask.clone()
        patch = self.patch.clone()
        rng_state = torch.get_rng_state().clone()
        for method, alpha in (('original_tree_ring', None),
                              ('globally_weaker_tree_ring', 0),
                              ('globally_weaker_tree_ring', 0.4),
                              ('globally_weaker_tree_ring', 1)):
            args = make_args(method_name=method, global_alpha=alpha)
            with self.subTest(method=method, alpha=alpha):
                actual = inject_watermark_for_method(base, mask, patch, args)
                self.assertTrue(torch.equal(base, self.latents))
                self.assertTrue(torch.equal(mask, self.mask))
                self.assertTrue(torch.equal(patch, self.patch))
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_state))
                actual.add_(10)
                self.assertTrue(torch.equal(base, self.latents))
                self.assertTrue(torch.equal(patch, self.patch))

    def test_global_invalid_configuration_never_mutates_inputs(self):
        base = self.latents.clone()
        for alpha in (None, -1, 2, float('nan'), float('inf'), True, '0.5'):
            args = make_args(method_name='globally_weaker_tree_ring', global_alpha=alpha)
            with self.subTest(alpha=alpha):
                with self.assertRaises(ValueError):
                    inject_watermark_for_method(base, self.mask, self.patch, args)
                self.assertTrue(torch.equal(base, self.latents))

    def test_residual_budget_has_known_rms_l2_and_zero_endpoint(self):
        base = torch.tensor([1., 2., 3., 4.])
        injected = base + torch.tensor([1., -2., 3., -4.])
        budget = measure_latent_residual(base, injected)
        self.assertAlmostEqual(budget['latent_residual_l2'], 30 ** 0.5)
        self.assertAlmostEqual(budget['latent_residual_rms'], (30 / 4) ** 0.5)
        self.assertEqual(measure_latent_residual(base, base.clone()),
                         {'latent_residual_rms': 0.0, 'latent_residual_l2': 0.0})

    def test_global_realized_budget_scales_original_residual(self):
        original = inject_watermark(self.latents, self.mask, self.patch, self.args)
        original_budget = measure_latent_residual(self.latents, original)
        args = make_args(method_name='globally_weaker_tree_ring', global_alpha=0.25)
        injected = inject_watermark_for_method(self.latents, self.mask, self.patch, args)
        budget = measure_latent_residual(self.latents, injected)
        for field in budget:
            self.assertAlmostEqual(budget[field], 0.25 * original_budget[field], places=6)

    def test_residual_budget_uses_actual_half_values_without_overflow(self):
        base = torch.tensor([1000., 0.001], dtype=torch.float16)
        injected = torch.tensor([-1000., 0.005], dtype=torch.float16)
        difference = injected.double() - base.double()
        budget = measure_latent_residual(base, injected)
        self.assertEqual(budget['latent_residual_l2'], difference.square().sum().sqrt().item())
        self.assertEqual(budget['latent_residual_rms'], difference.square().mean().sqrt().item())

    def test_residual_budget_rejects_invalid_or_nonfinite_tensors(self):
        for base, injected in ((torch.zeros(2), torch.zeros(3)),
                               (torch.empty(0), torch.empty(0)),
                               (torch.zeros(2), torch.tensor([float('nan'), 0.])),
                               (torch.zeros(2), torch.tensor([float('inf'), 0.])),
                               (torch.zeros(2, dtype=torch.int64), torch.ones(2)),
                               (torch.zeros(2, dtype=torch.complex64), torch.ones(2)),
                               (torch.zeros(1, dtype=torch.float64),
                                torch.tensor([1e308], dtype=torch.float64))):
            with self.subTest(base=base, injected=injected):
                with self.assertRaises(ValueError):
                    measure_latent_residual(base, injected)


if __name__ == '__main__':
    unittest.main()
