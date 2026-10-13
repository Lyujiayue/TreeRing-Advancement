import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from image_artifacts import save_image_pair
from scripts import development_validation as dev
from tests import test_global_gpu_validation as smoke_fixtures


class DevelopmentValidationTests(unittest.TestCase):
    head = 'a' * 40

    def fixture(self, label):
        summary, _ = smoke_fixtures.GlobalGPUAcceptanceTests().fixture(label)
        cfg = dev.config(label)
        summary.update(run_name=cfg['run_name'], parameters=cfg, num_samples=32, duration_seconds=70.)
        summary['run_identity'].update(run_purpose='full', sample_range=dict(start=0, end_exclusive=32, count=32))
        rows = []
        for i in range(32):
            row = copy.deepcopy(summary['results'][0])
            row.update(sample_index=i, source_row_id=1200 + i, prompt=f'fixture {i}', generation_seed=i,
                       sample_id=cfg['run_name'] + ':' + str(i), run_purpose='full',
                       clip_similarity_available=False, image_artifacts={})
            rows.append(row)
        summary['results'] = rows
        detected = label != 'global-a0'
        summary['metrics'].update(auc=1. if detected else .5,
                                  oracle_max_balanced_accuracy=1. if detected else .5,
                                  tpr_at_1pct_fpr=1. if detected else 0.)
        summary.update(summary['metrics'])
        summary['acc'] = summary['oracle_max_balanced_accuracy']
        return summary, copy.deepcopy(rows)

    def test_32_records_require_complete_ordered_paired_identity(self):
        prompts = [f'fixture {i}' for i in range(32)]
        edits = [None, lambda s: s['results'].pop(),
                 lambda s: s['results'].__setitem__(1, copy.deepcopy(s['results'][0])),
                 lambda s: s['results'][31].update(generation_seed=0),
                 lambda s: s['results'][5].update(source_row_id=1200),
                 lambda s: s['results'][2].update(no_w_score=0),
                 lambda s: s['results'][3].update(latent_residual_l2=float('nan')),
                 lambda s: s['environment'].update(device='cpu'),
                 lambda s: s['results'][2].update(clip_similarity_available=True),
                 lambda s: s['parameters'].update(save_paired_images=False),
                 lambda s: s['metrics'].update(auc=.123),
                 lambda s: s.update(acc=.123),
                 lambda s: s['results'][4].pop('image_artifacts')]
        for label in dev.RUNS:
            for edit in edits:
                with self.subTest(label=label, edit=edit):
                    summary, _ = self.fixture(label)
                    if edit:
                        edit(summary)
                        with self.assertRaises((ValueError, KeyError)):
                            dev.validate_output(summary, copy.deepcopy(summary['results']), label, self.head, prompts)
                    else:
                        dev.validate_output(summary, copy.deepcopy(summary['results']), label, self.head, prompts)

    def write_outputs(self, root):
        prompt_path = root / '.local_data/prompts/splits' / dev.smoke.PROMPT_FILES['development']
        prompt_path.parent.mkdir(parents=True)
        prompt_path.write_text(''.join(f'fixture {i}\n' for i in range(32)), encoding='utf-8')
        for label in dev.RUNS:
            summary, rows = self.fixture(label)
            summary_path, jsonl_path, images = dev.paths(label)
            images.mkdir(parents=True)
            for i, row in enumerate(rows):
                no_w = Image.new('RGB', (8, 8), (i, 0, 0))
                color = 0 if label == 'global-a0' else 1 if label == 'global-a05' else 2
                marked = Image.new('RGB', (8, 8), (i, color, 0))
                row['image_artifacts'] = save_image_pair(images, i, no_w, marked)
            summary['results'] = rows
            summary_path.write_text(json.dumps(summary), encoding='utf-8')
            jsonl_path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        return prompt_path

    def test_accept_checks_all_images_and_each_method_endpoints(self):
        # Small synthetic RGB images; no real prompt/model/GPU data.
        from image_artifacts import verify_image_pair
        edits = (None, 'pairing', 'endpoint', 'missing-image', 'changed-image', 'extra-image')
        for edit in edits:
            with self.subTest(edit=edit), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(dev, 'REPO', root), \
                     patch.object(dev, 'verify_image_pair', side_effect=lambda d, i, a: verify_image_pair(d, i, a, 8)):
                    prompt_path = self.write_outputs(root)
                    with patch.dict(dev.smoke.FROZEN_PROMPT_SPLITS['development'],
                                    sha256=dev.smoke.file_sha256(prompt_path)):
                        # Update fixtures to the synthetic prompt identity.
                        for label in dev.RUNS:
                            p, j, _ = dev.paths(label)
                            summary = json.loads(p.read_text(encoding='utf-8'))
                            summary['prompt_source']['sha256'] = dev.smoke.file_sha256(prompt_path)
                            if label == 'global-a1' and edit in ('pairing', 'endpoint'):
                                row = summary['results'][31]
                                if edit == 'pairing':
                                    row['paired_inputs']['base_latent_sha256'] = 'c' * 64
                                else:
                                    row.update(w_metric=4., w_score=-4.)
                            p.write_text(json.dumps(summary), encoding='utf-8')
                            j.write_text(''.join(json.dumps(row) + '\n' for row in summary['results']), encoding='utf-8')
                        _, _, images = dev.paths('global-a1')
                        if edit == 'missing-image':
                            (images / '0031.watermarked.png').unlink()
                        if edit == 'changed-image':
                            (images / '0031.watermarked.png').write_bytes(b'corrupted')
                        if edit == 'extra-image':
                            (images / 'extra.png').write_bytes(b'extra')
                        if edit:
                            with self.assertRaises(ValueError):
                                dev.accept_outputs(self.head)
                        else:
                            report = dev.accept_outputs(self.head)
                            self.assertEqual(report['status'], 'ORIGINAL_GLOBAL_32_DEVELOPMENT_GATE_PASS')
                            self.assertEqual(len(report['output_hashes']), 264)
                            self.assertFalse(report['full_development_gate'])
                            self.assertFalse(report['quality_metrics_validated'])

    def test_plan_and_unapproved_execution_do_not_launch_processes(self):
        for arguments in (['dev'], ['dev', 'run-one', '--run', 'original'], ['dev', 'cpu-gate']):
            with patch('sys.argv', arguments), patch.object(dev.subprocess, 'run') as run, \
                 patch.object(dev, 'run_one') as execute, contextlib.redirect_stdout(io.StringIO()):
                if len(arguments) == 1:
                    dev.main()
                else:
                    with self.assertRaises(ValueError):
                        dev.main()
                run.assert_not_called()
                execute.assert_not_called()

    def test_timeout_and_failed_process_preserve_console_and_status(self):
        for outcome in (subprocess.TimeoutExpired(['synthetic'], 1), OSError('synthetic spawn failure'),
                        subprocess.CompletedProcess(['synthetic'], 7)):
            with tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / 'failed.log'
                with patch.object(dev.subprocess, 'run') as run:
                    if isinstance(outcome, Exception):
                        run.side_effect = outcome
                    else:
                        run.return_value = outcome
                    report = dev.run_logged(['synthetic'], log, 1)
                self.assertTrue(log.is_file())
                self.assertEqual(report['console_sha256'], dev.smoke.file_sha256(log))
                self.assertNotEqual(report['returncode'], 0)
                with self.assertRaises(FileExistsError):
                    dev.run_logged(['synthetic'], log, 1)

    def test_console_prior_evidence_and_command_contract_are_strict(self):
        for warning in ('UserWarning: unknown', 'FutureWarning: The decode_latents method is deprecated but changed',
                        dev.ALLOWED_WARNINGS[0] + ' additional unreviewed warning',
                        'UserWarning: ' + dev.ALLOWED_WARNINGS[0].split(': ', 1)[1],
                        'Traceback (most recent call last)'):
            with self.assertRaises(ValueError):
                dev.console_review(warning)
        for label in dev.RUNS:
            cmd = dev.command(label)
            self.assertIn('--save_paired_images', cmd)
            self.assertIn('--require_cuda', cmd)
            self.assertNotIn('True', cmd)
            self.assertNotIn('--overwrite_output', cmd)
        with tempfile.TemporaryDirectory() as directory:
            prior = Path(directory) / 'prior.json'
            prior.write_text('{}', encoding='utf-8')
            with self.assertRaises(ValueError):
                dev.verify_prior_smoke(prior, self.head)

    def test_test_inventory_binds_sources_and_normalizes_python_test_display(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory, patch.object(dev, 'REPO', Path(directory)):
            source = Path(directory) / 'tests/test_example.py'
            source.parent.mkdir()
            source.write_bytes(b'# reviewed\r\n')
            inventory = dict(source_sha256_lf={'tests/test_example.py': hashlib.sha256(b'# reviewed\n').hexdigest()})
            dev.check_inventory_sources(inventory)
            source.write_bytes(b'# changed\n')
            with self.assertRaises(ValueError):
                dev.check_inventory_sources(inventory)
            with self.assertRaises(ValueError):
                dev.check_inventory_sources(dict(source_sha256_lf={}))
        old = 'test_example (test_example.ExampleTests) ... ok'
        new = 'test_example (test_example.ExampleTests.test_example) ... ok'
        self.assertEqual(dev.observed_test_ids(old), dev.observed_test_ids(new))

    def test_process_reports_bind_console_outputs_and_baseline(self):
        for anomaly in (None, 'tamper', 'warning', 'baseline', 'duplicate'):
            with self.subTest(anomaly=anomaly), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                baseline = root / 'baseline.json'
                baseline.write_text('{}', encoding='utf-8')
                reports = []
                with patch.object(dev.smoke, 'TRANSFER', root), patch.object(dev, 'snapshot_run', return_value={'image': 'a' * 64}):
                    for label in dev.RUNS:
                        log = root / (label + '.log')
                        log.write_text('UserWarning: unknown' if label == 'original' and anomaly == 'warning' else '', encoding='utf-8')
                        path = root / (label + '.json')
                        dev.smoke.write_report(path, dict(status='DEVELOPMENT_PROCESS_PASS', returncode=0, run=label,
                            plan_version=dev.PLAN_VERSION, expected_head=self.head, command=dev.command(label),
                            baseline_sha256='0' * 64 if label == 'original' and anomaly == 'baseline' else dev.smoke.file_sha256(baseline),
                            console_log=str(log), console_sha256=dev.smoke.file_sha256(log), output_hashes={'image': 'a' * 64}))
                        reports.append(path)
                        if label == 'original' and anomaly == 'tamper':
                            log.write_text('changed', encoding='utf-8')
                    if anomaly == 'duplicate':
                        reports[-1] = reports[0]
                    if anomaly:
                        with self.assertRaises(ValueError):
                            dev.review_process_reports(reports, self.head, baseline)
                    else:
                        self.assertEqual(len(dev.review_process_reports(reports, self.head, baseline)), 8)
