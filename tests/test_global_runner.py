"""Exercise real runner dispatch/records using a CPU-only synthetic pipeline."""

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

import run_tree_ring_watermark as runner
from experiment_records import FROZEN_PROMPT_SPLITS, SCHEMA_VERSION
from optim_utils import get_watermarking_mask, inject_watermark, measure_latent_residual


class SyntheticPipeline:
    """No weights, model I/O, real prompt data, network, or GPU operations."""

    def __init__(self):
        self.calls = []
        self.random_latents = []

    def to(self, device):
        if device != 'cpu':
            raise AssertionError('This test must stay on CPU')
        return self

    def get_text_embedding(self, prompt):
        return torch.zeros(1)

    def get_random_latents(self):
        latent = torch.randn(1, 4, 16, 16)
        self.random_latents.append(latent)
        return latent

    def __call__(self, prompt, **kwargs):
        latent = kwargs.pop('latents')
        self.calls.append({
            'prompt': prompt, 'parameters': kwargs,
            'latent': latent.clone(), 'rng_state': torch.get_rng_state().clone(),
        })
        image = latent[0].clone()
        # An adversarial in-place pipeline operation must not corrupt the
        # shared base latent or the residual budget measured before generation.
        latent.add_(100)
        torch.rand(2)
        return Mock(images=[image])

    def get_image_latents(self, image, sample=False):
        return image.clone()

    def forward_diffusion(self, latents, **kwargs):
        return latents.clone()


class GlobalRunnerTests(unittest.TestCase):
    def make_args(self, method='original_tree_ring', alpha=None):
        token = 'original' if method == 'original_tree_ring' else 'global'
        arguments = [
            '--run_name', f'{token}_development_r1_clean',
            '--prompt_file', 'synthetic-prompts.txt',
            '--prompt_split', 'development', '--method_name', method,
            '--attack_name', 'clean', '--replicate_id', '1',
            '--protocol_version', 'v1',
            '--watermark_key_id', 'treering-rand-wseed-999999',
            '--start', '7', '--end', '8', '--num_inference_steps', '1',
            '--test_num_inference_steps', '1',
        ]
        if alpha is not None:
            arguments.extend(['--global_alpha', str(alpha)])
        return runner.build_parser().parse_args(arguments)

    def prompt_description(self):
        return {
            'path': 'synthetic-prompts.txt',
            'size_bytes': 1,
            'sha256': FROZEN_PROMPT_SPLITS['development']['sha256'],
            'nonempty_line_count': 32,
        }

    def run_synthetic_sample(self, args):
        pipe = SyntheticPipeline()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(runner, '__file__', str(root / 'runner.py')))
                stack.enter_context(patch.object(runner.torch.cuda, 'is_available', return_value=False))
                scheduler = stack.enter_context(patch.object(runner.DDIMScheduler, 'from_pretrained'))
                model = stack.enter_context(patch.object(
                    runner.InversableStableDiffusionPipeline, 'from_pretrained', return_value=pipe
                ))
                stack.enter_context(patch.object(runner, 'describe_prompt_file',
                                                 return_value=self.prompt_description()))
                stack.enter_context(patch.object(runner, 'get_dataset', return_value=(
                    [{'Prompt': f'synthetic {i}'} for i in range(32)], 'Prompt'
                )))
                stack.enter_context(patch.object(runner, 'get_git_state',
                                                 return_value={'commit': 'test', 'dirty': True}))
                stack.enter_context(patch.object(runner, 'image_distortion',
                                                 side_effect=lambda a, b, seed, config: (a, b)))
                stack.enter_context(patch.object(runner, 'transform_img', side_effect=lambda image: image))
                stack.enter_context(patch.object(runner, 'tqdm', side_effect=lambda values: values))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                runner.main(args)
                scheduler.assert_called_once()
                model.assert_called_once()
                self.assertTrue(scheduler.call_args.kwargs['local_files_only'])
                self.assertTrue(model.call_args.kwargs['local_files_only'])

            outputs = root / '.local_outputs' / 'runs'
            summary = json.loads((outputs / f'{args.run_name}.json').read_text(encoding='utf-8'))
            samples = [json.loads(line) for line in
                       (outputs / f'{args.run_name}.samples.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(summary['results'], samples)
        self.assertEqual(summary['schema_version'], SCHEMA_VERSION)
        self.assertEqual(summary['num_samples'], 1)
        self.assertEqual(summary['run_identity']['sample_range'],
                         {'start': 7, 'end_exclusive': 8, 'count': 1})
        self.assertEqual(samples[0]['source_row_id'], 1207)
        self.assertEqual(samples[0]['generation_seed'], args.gen_seed + 7)
        self.assertEqual(samples[0]['watermark_seed'], args.w_seed)
        self.assertEqual(samples[0]['method_name'], args.method_name)
        self.assertEqual(samples[0]['attack_parameters'], {})
        self.assertEqual(samples[0]['no_w_score'], -samples[0]['no_w_metric'])
        self.assertEqual(samples[0]['w_score'], -samples[0]['w_metric'])
        for value in samples[0].values():
            if isinstance(value, float):
                self.assertTrue(math.isfinite(value))
        # The pipeline modifies its own input, while the sampled base survives.
        self.assertTrue(torch.equal(pipe.random_latents[-1], pipe.calls[0]['latent']))
        budget = measure_latent_residual(pipe.calls[0]['latent'], pipe.calls[1]['latent'])
        for field, value in budget.items():
            self.assertEqual(samples[0][field], value)
        return pipe, summary, samples[0]

    def test_runner_original_and_global_share_inputs_and_record_realized_budgets(self):
        original_args = self.make_args()
        original, original_summary, original_sample = self.run_synthetic_sample(original_args)
        self.assertEqual(original_sample['method_parameters'], {})
        self.assertEqual(original_summary['run_identity']['method_parameters'], {})
        base = original.calls[0]['latent']
        key = original.random_latents[0]
        mask = get_watermarking_mask(base, original_args, 'cpu')
        key_fft = torch.fft.fftshift(torch.fft.fft2(key), dim=(-1, -2))
        expected_original = inject_watermark(base, mask, key_fft, original_args)
        self.assertTrue(torch.equal(original.calls[1]['latent'], expected_original))

        for alpha in (0.0, 0.25, 1.0):
            with self.subTest(alpha=alpha):
                args = self.make_args('globally_weaker_tree_ring', alpha)
                global_pipe, summary, sample = self.run_synthetic_sample(args)
                self.assertTrue(torch.equal(global_pipe.random_latents[0], key))
                self.assertTrue(torch.equal(global_pipe.calls[0]['latent'], base))
                for baseline_call, global_call in zip(original.calls, global_pipe.calls):
                    self.assertEqual(baseline_call['prompt'], global_call['prompt'])
                    self.assertEqual(baseline_call['parameters'], global_call['parameters'])
                    self.assertTrue(torch.equal(baseline_call['rng_state'], global_call['rng_state']))
                expected = (base if alpha == 0 else expected_original if alpha == 1
                            else base + alpha * (expected_original - base))
                self.assertTrue(torch.equal(global_pipe.calls[1]['latent'], expected))
                self.assertEqual(sample['method_parameters'], {'alpha': alpha})
                self.assertEqual(summary['run_identity']['method_parameters'], {'alpha': alpha})
                self.assertEqual(summary['parameters']['global_alpha'], alpha)
                if alpha == 0:
                    self.assertEqual(sample['latent_residual_l2'], 0.0)
                    self.assertEqual(sample['latent_residual_rms'], 0.0)
                    self.assertEqual(sample['w_metric'], sample['no_w_metric'])
                if alpha == 1:
                    self.assertEqual(sample['w_metric'], original_sample['w_metric'])
                    self.assertEqual(sample['latent_residual_l2'], original_sample['latent_residual_l2'])

    def test_parser_rejects_unimplemented_methods(self):
        for method in ('saliency_aware_tree_ring', 'no_watermark'):
            with self.subTest(method=method), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.make_args(method)

    def test_invalid_method_parameters_fail_before_model_io_or_output_deletion(self):
        configurations = [('globally_weaker_tree_ring', alpha) for alpha in
                          (None, -0.1, 1.1, float('nan'), float('inf'))]
        configurations.append(('original_tree_ring', 0.5))
        for method, alpha in configurations:
            args = self.make_args(method, alpha)
            args.overwrite_output = True
            with self.subTest(method=method, alpha=alpha), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                outputs = root / '.local_outputs' / 'runs'
                outputs.mkdir(parents=True)
                sentinel = outputs / f'{args.run_name}.json'
                sentinel.write_text('preserved evidence', encoding='utf-8')
                with patch.object(runner, '__file__', str(root / 'runner.py')), \
                     patch.object(runner.InversableStableDiffusionPipeline, 'from_pretrained') as model, \
                     patch.object(runner.DDIMScheduler, 'from_pretrained') as scheduler:
                    with self.assertRaises(ValueError):
                        runner.main(args)
                    model.assert_not_called()
                    scheduler.assert_not_called()
                self.assertEqual(sentinel.read_text(encoding='utf-8'), 'preserved evidence')
                self.assertEqual(list(outputs.iterdir()), [sentinel])

    def test_existing_output_is_rejected_before_model_loading(self):
        args = self.make_args('globally_weaker_tree_ring', 0.25)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs = root / '.local_outputs' / 'runs'
            outputs.mkdir(parents=True)
            sentinel = outputs / f'{args.run_name}.samples.jsonl'
            sentinel.write_text('preserved evidence', encoding='utf-8')
            with patch.object(runner, '__file__', str(root / 'runner.py')), \
                 patch.object(runner, 'describe_prompt_file', return_value=self.prompt_description()), \
                 patch.object(runner.InversableStableDiffusionPipeline, 'from_pretrained') as model:
                with self.assertRaises(FileExistsError):
                    runner.main(args)
                model.assert_not_called()
            self.assertEqual(sentinel.read_text(encoding='utf-8'), 'preserved evidence')

    def test_existing_identity_guards_still_fail_before_model_io_or_outputs(self):
        cases = (
            ('attack', {'r_degree': 75}, None),
            ('split', {'prompt_split': 'pilot'}, None),
            ('hash', {}, {'sha256': '0' * 64}),
            ('range', {'end': 33}, None),
            ('name', {'run_name': 'wrong_name'}, None),
        )
        for name, overrides, prompt_overrides in cases:
            args = self.make_args('globally_weaker_tree_ring', 0.25)
            for key, value in overrides.items():
                setattr(args, key, value)
            description = self.prompt_description()
            description.update(prompt_overrides or {})
            with self.subTest(case=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(runner, '__file__', str(root / 'runner.py')), \
                     patch.object(runner, 'describe_prompt_file', return_value=description), \
                     patch.object(runner.DDIMScheduler, 'from_pretrained') as scheduler, \
                     patch.object(runner.InversableStableDiffusionPipeline, 'from_pretrained') as model:
                    with self.assertRaises(ValueError):
                        runner.main(args)
                    scheduler.assert_not_called()
                    model.assert_not_called()
                self.assertFalse((root / '.local_outputs').exists())


if __name__ == '__main__':
    unittest.main()
