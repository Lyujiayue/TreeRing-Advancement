"""CPU-only adversarial checks for the prepared server acceptance gate."""
import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import global_gpu_validation as gate


class GlobalGPUAcceptanceTests(unittest.TestCase):
    head = 'a' * 40

    def fixture(self, label):
        cfg = gate.config(label)
        alpha = cfg['global_alpha']
        residual = 4. if alpha is None else 4. * alpha
        metric = 3. if alpha is None or alpha == 1 else 10. if alpha == 0 else 6.
        fields = ('prompt_split', 'method_name', 'attack_name', 'replicate_id',
                  'protocol_version', 'watermark_key_id', 'run_purpose')
        identity = {k: cfg[k] for k in fields}
        identity.update(run_naming_version=gate.RUN_NAMING_VERSION,
                        method_parameters={} if alpha is None else dict(alpha=alpha),
                        attack_parameters={})
        row = dict(identity, sample_id=cfg['run_name'] + ':0', sample_index=0,
                   source_row_id=1200, prompt='fixture prompt', generation_seed=0,
                   watermark_seed=999999, no_w_metric=10., w_metric=metric,
                   no_w_score=-10., w_score=-metric, w_no_sim=0, w_sim=0,
                   latent_residual_l2=residual, latent_residual_rms=residual / 128,
                   sample_duration_seconds=2.,
                   paired_inputs={key: 'b' * 64 for key in (
                       'base_latent_sha256', 'watermark_key_sha256', 'watermark_mask_sha256')})
        identity['sample_range'] = dict(start=0, end_exclusive=1, count=1)
        summary = dict(schema_version='treering-experiment-v2', run_name=cfg['run_name'],
                       run_identity=identity, num_samples=1, results=[row],
                       git=dict(commit=self.head, branch=gate.BRANCH, dirty=False),
                       prompt_source=dict(path=cfg['prompt_file'], nonempty_line_count=32,
                                          sha256=gate.FROZEN_PROMPT_SPLITS['development']['sha256']),
                       parameters=cfg, environment=dict(
                           cuda_available=True, device='cuda', torch_dtype='torch.float16',
                           gpu_name='NVIDIA B200', torch_version='2.7.0a0+ecf3bae40a.nv25.02',
                           torch_cuda_version='12.8', python_executable=str(gate.VENV / 'bin/python'),
                           python_version='3.12.3 fixture'), duration_seconds=4.,
                       started_at_utc='2026-10-10T01:00:00+00:00',
                       completed_at_utc='2026-10-10T01:00:04+00:00',
                       metrics=dict(clip_score_mean=0, w_clip_score_mean=0, auc=.5,
                                    oracle_max_balanced_accuracy=.5, tpr_at_1pct_fpr=0))
        return summary, [copy.deepcopy(row)]

    def test_each_planned_record_passes_without_gpu_or_model(self):
        for label in gate.RUNS:
            summary, records = self.fixture(label)
            gate.validate_output(summary, records, label, self.head, 'fixture prompt')

    def test_identity_parameter_device_and_missing_numeric_corruptions_fail(self):
        edits = [lambda s: s['git'].update(dirty=True),
                 lambda s: s['parameters'].update(global_alpha=.25),
                 lambda s: s['environment'].update(device='cpu'),
                 lambda s: s['prompt_source'].update(sha256='c' * 64),
                 lambda s: s['run_identity'].update(run_purpose='full'),
                 lambda s: s['results'][0].update(latent_residual_l2=None),
                 lambda s: s['results'][0].update(latent_residual_rms=float('nan')),
                 lambda s: s['results'][0]['paired_inputs'].pop('base_latent_sha256'),
                 lambda s: s['results'][0].update(source_row_id=1201),
                 lambda s: s['metrics'].update(auc=None),
                 lambda s: s.update(num_samples=32),
                 lambda s: s['results'].append(copy.deepcopy(s['results'][0]))]
        for index, edit in enumerate(edits):
            summary, _ = self.fixture('global-a05')
            edit(summary)
            records = copy.deepcopy(summary['results'])
            with self.subTest(index=index), self.assertRaises((ValueError, KeyError)):
                gate.validate_output(summary, records, 'global-a05', self.head, 'fixture prompt')

    def test_strict_json_rejects_nonfinite_constants_and_duplicate_keys(self):
        for text in ('{"x": NaN}', '{"x": Infinity}', '{"x":1,"x":2}'):
            with self.assertRaises(ValueError):
                gate.strict_json(text)

    def write_fixtures(self, root):
        prompts = root / '.local_data/prompts/splits'
        prompts.mkdir(parents=True)
        (prompts / gate.PROMPT_FILES['development']).write_text('fixture prompt\n', encoding='utf-8')
        outputs = root / '.local_outputs/runs'
        outputs.mkdir(parents=True)
        for label in gate.RUNS:
            summary, records = self.fixture(label)
            (outputs / (summary['run_name'] + '.json')).write_text(json.dumps(summary), encoding='utf-8')
            (outputs / (summary['run_name'] + '.samples.jsonl')).write_text(json.dumps(records[0]) + '\n', encoding='utf-8')

    def test_cross_method_pairing_and_endpoints_are_required(self):
        edits = [None, lambda row: row['paired_inputs'].update(base_latent_sha256='c' * 64),
                 lambda row: row.update(no_w_metric=11., no_w_score=-11.),
                 lambda row: row.update(w_metric=4., w_score=-4.)]
        for index, edit in enumerate(edits):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.write_fixtures(root)
                with patch.object(gate, 'REPO', root):
                    if edit:
                        summary_path, jsonl_path = gate.paths('global-a1')
                        summary = json.loads(summary_path.read_text(encoding='utf-8'))
                        edit(summary['results'][0])
                        summary_path.write_text(json.dumps(summary), encoding='utf-8')
                        jsonl_path.write_text(json.dumps(summary['results'][0]) + '\n', encoding='utf-8')
                        with self.assertRaises(ValueError):
                            gate.accept_outputs(self.head)
                    else:
                        result = gate.accept_outputs(self.head)
                        self.assertEqual(result['status'], 'GLOBAL_GPU_SMALL_SMOKE_GATE_PASS')
                        self.assertFalse(result['full_development_gate'])

    def test_default_mode_never_calls_gpu_or_launches_processes(self):
        with patch('sys.argv', ['gate']), patch.object(gate, 'numeric_sanity') as sanity, \
             patch.object(gate.subprocess, 'run') as run, contextlib.redirect_stdout(io.StringIO()):
            gate.main()
            sanity.assert_not_called()
            run.assert_not_called()

    def test_process_evidence_requires_matching_hashes_and_explained_console(self):
        for anomaly in (None, 'warning', 'traceback', 'tamper'):
            with self.subTest(anomaly=anomaly), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                baseline, sanity = root / 'baseline.json', root / 'sanity.json'
                baseline.write_text('{}', encoding='utf-8')
                sanity.write_text('{}', encoding='utf-8')
                reports = []
                with patch.object(gate, 'TRANSFER', root):
                    for label in gate.RUNS:
                        log = root / (label + '.log')
                        content = 'FutureWarning: Importing from timm.models.layers is deprecated\n'
                        if label == 'original' and anomaly == 'warning':
                            content += 'UserWarning: unexplained computation warning\n'
                        if label == 'original' and anomaly == 'traceback':
                            content += 'Traceback (most recent call last)\n'
                        log.write_text(content, encoding='utf-8')
                        path = root / (label + '.json')
                        gate.write_report(path, dict(
                            status='SMOKE_PROCESS_PASS', returncode=0, run=label,
                            expected_head=self.head, command=gate.command(label),
                            baseline_sha256=gate.file_sha256(baseline),
                            sanity_sha256=gate.file_sha256(sanity), console_log=str(log),
                            console_sha256=gate.file_sha256(log)))
                        reports.append(path)
                        if label == 'original' and anomaly == 'tamper':
                            log.write_text('changed after recorded hash', encoding='utf-8')
                    if anomaly:
                        with self.assertRaises(ValueError):
                            gate.review_process_reports(reports, self.head, baseline, sanity)
                    else:
                        hashes = gate.review_process_reports(reports, self.head, baseline, sanity)
                        self.assertEqual(len(hashes), 8)
                        with self.assertRaises(ValueError):
                            gate.review_process_reports(reports[:-1], self.head, baseline, sanity)

    def test_execution_modes_require_explicit_flags(self):
        for mode in ('sanity', 'run-one'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                baseline = root / 'baseline.json'
                baseline.write_text('{}', encoding='utf-8')
                with patch.object(gate, 'TRANSFER', root), \
                     patch('sys.argv', ['gate', mode, '--report', str(root / 'new.json'),
                                       '--baseline', str(baseline), '--sanity-report', str(baseline)]), \
                     patch.object(gate, 'load_report', return_value=dict(
                         status='GPU_NUMERIC_SANITY_PASS', expected_head=None,
                         baseline_sha256=gate.file_sha256(baseline))), \
                     patch.object(gate, 'verify_baseline'), \
                     patch.object(gate, 'numeric_sanity') as sanity, \
                     patch.object(gate.subprocess, 'run') as run:
                    with self.assertRaises(ValueError):
                        gate.main()
                    sanity.assert_not_called()
                    run.assert_not_called()
