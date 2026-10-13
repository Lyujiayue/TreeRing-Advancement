import math
import unittest
from types import SimpleNamespace

from run_naming import alpha_token, build_run_identity


class RunNamingTests(unittest.TestCase):
    def args(self, **overrides):
        values = dict(
            run_name='original_development_r1_clean_smoke_s0_e1_g1_i1',
            run_purpose='smoke', method_name='original_tree_ring', global_alpha=None,
            saliency_preview_steps=None, saliency_background_strength=None,
            saliency_foreground_strength=None, saliency_power=None,
            prompt_split='development', replicate_id=1, attack_name='clean',
            start=0, end=1, num_inference_steps=1, test_num_inference_steps=1,
        )
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_old_occupied_name_cannot_identify_new_smoke_or_full(self):
        for purpose, end in [('smoke', 1), ('full', 32)]:
            with self.subTest(purpose=purpose), self.assertRaises(ValueError):
                build_run_identity(self.args(
                    run_name='original_development_r1_clean', run_purpose=purpose, end=end
                ))
        self.assertEqual(build_run_identity(self.args())['run_purpose'], 'smoke')
        self.assertEqual(build_run_identity(self.args(
            run_purpose='full', end=32, num_inference_steps=50, test_num_inference_steps=50,
            run_name='original_development_r1_clean_full_s0_e32_g50_i50'
        ))['sample_range']['count'], 32)

    def test_alpha_tokens_do_not_merge_adjacent_values_or_scientific_notation(self):
        values = [0., .5, math.nextafter(.5, 1.), 1e-7, 1.]
        self.assertEqual(len({alpha_token(v) for v in values}), len(values))
        self.assertEqual(alpha_token(-0.), alpha_token(0.))
        for v in values:
            token = alpha_token(v)
            self.assertEqual(float(token.replace('p', '.').replace('m', '-')), v)
            args = self.args(method_name='globally_weaker_tree_ring', global_alpha=v,
                             run_name=f'global_development_r1_clean_a{token}_smoke_s0_e1_g1_i1')
            build_run_identity(args)
            args.global_alpha = math.nextafter(v, 0.5 if v != .5 else 1.)
            with self.assertRaises(ValueError):
                build_run_identity(args)

    def test_partial_full_unclassified_runs_and_nonpositive_steps_are_rejected(self):
        for override in [dict(run_purpose='full'), dict(run_purpose=None),
                         dict(num_inference_steps=0), dict(test_num_inference_steps=-1),
                         dict(prompt_split='pilot')]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                build_run_identity(self.args(**override))

    def test_saliency_parameters_are_part_of_the_run_name(self):
        values = dict(
            method_name='saliency_aware_tree_ring',
            saliency_preview_steps=10,
            saliency_background_strength=1.5,
            saliency_foreground_strength=0.2,
            saliency_power=2.0,
            run_name=(
                'saliency_development_r1_clean_pv10_bg1p5_fg0p2_pw2p0_'
                'smoke_s0_e1_g1_i1'
            ),
        )
        identity = build_run_identity(self.args(**values))
        self.assertEqual(identity['run_purpose'], 'smoke')
        values['saliency_power'] = 2.1
        with self.assertRaises(ValueError):
            build_run_identity(self.args(**values))

    def test_saliency_signed_zero_has_one_canonical_run_name(self):
        values = dict(
            method_name='saliency_aware_tree_ring',
            saliency_preview_steps=10,
            saliency_background_strength=-0.0,
            saliency_foreground_strength=-0.0,
            saliency_power=2.0,
            run_name=(
                'saliency_development_r1_clean_pv10_bg0p0_fg0p0_pw2p0_'
                'smoke_s0_e1_g1_i1'
            ),
        )
        build_run_identity(self.args(**values))
        values['run_name'] = (
            'saliency_development_r1_clean_pv10_bgm0p0_fgm0p0_pw2p0_'
            'smoke_s0_e1_g1_i1'
        )
        with self.assertRaises(ValueError):
            build_run_identity(self.args(**values))

    def test_range_and_steps_are_part_of_the_name(self):
        for override in [dict(end=2), dict(start=1, end=2), dict(num_inference_steps=50),
                         dict(test_num_inference_steps=50)]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                build_run_identity(self.args(**override))
