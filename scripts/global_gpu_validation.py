#!/usr/bin/env python3
"""Prepared B200 gate. Default is plan-only; execution requires explicit flags.

Uses only the existing environment/model. Never installs, fetches, overwrites,
commits, starts/stops instances, or launches Development/Pilot/Formal runs.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from experiment_records import FROZEN_PROMPT_SPLITS, file_sha256
from run_naming import RUN_NAMING_VERSION, alpha_token, build_run_identity

SERVER_REPO = Path('/public/home/rxa-jj-ljq/projects/TreeRing-Advancement')
TRANSFER = Path('/public/home/rxa-jj-ljq/TR_Transfer')
VENV = Path('/public/home/rxa-jj-ljq/venvs/treering')
MODEL = '/public/home/rxa-jj-ljq/models/stable-diffusion-2-1-base'
BRANCH = 'stage-c-record-identity'
PARENT_GATE = '6c44c0a85f625f5e2b3bb1799162a51103df5550'
PLAN_VERSION = 'global-b200-smoke-plan-v1'
RUNS = {'original': ('original_tree_ring', None),
        'global-a0': ('globally_weaker_tree_ring', 0.),
        'global-a05': ('globally_weaker_tree_ring', .5),
        'global-a1': ('globally_weaker_tree_ring', 1.)}
PROMPT_FILES = {'development': 'sdp_dev_rows_1200_1231.txt',
                'pilot': 'sdp_pilot_rows_1000_1199.txt',
                'formal': 'sdp_formal_rows_0000_0999.txt'}
STAGE_C = {
    SERVER_REPO / '.local_outputs/runs/original_development_r1_clean.json':
    '5c5a227e3b5ffdeede3e0c0636047cbc794389ee36861556d3aa6be92e16d8d4',
    SERVER_REPO / '.local_outputs/runs/original_development_r1_clean.samples.jsonl':
    '5803fd2044fd704df55a8ec2baaf6501a8e50470db409db7b9af539e74b377df',
    TRANSFER / 'stage-c-52bcdea-b200-smoke.console.log':
    '0ba3bb0f616059f735389270d4915539069119126947b1d9ebc5d170da337a1c',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def git(*args):
    result = subprocess.run(['git', *args], cwd=REPO, capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=60)
    require(result.returncode == 0, f'Git check failed: {args}: {result.stderr}')
    return result.stdout.strip()


def write_report(path, report):
    # Exclusive creation: evidence is never replaced, including a failed report.
    path = Path(path).resolve()
    require(path.parent == TRANSFER.resolve(), 'Write reports directly in server TR_Transfer')
    checksum = Path(str(path) + '.sha256')
    require(not path.exists() and not checksum.exists(), 'Report/checksum already exists')
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    with checksum.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(f'{file_sha256(path)}  {path.name}\n')


def load_report(path):
    path = Path(path)
    checksum = Path(str(path) + '.sha256')
    require(checksum.read_text(encoding='ascii').strip() ==
            f'{file_sha256(path)}  {path.name}', 'Evidence checksum mismatch')
    return strict_json(path.read_text(encoding='utf-8'))


def strict_json(text):
    def reject(value):
        raise ValueError(f'Non-finite JSON constant: {value}')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    return json.loads(text, parse_constant=reject, object_pairs_hook=unique)


def config(label):
    method, alpha = RUNS[label]
    token = 'original' if alpha is None else 'global'
    name = f'{token}_development_r1_clean'
    if alpha is not None:
        name += '_a' + alpha_token(alpha)
    name += '_smoke_s0_e1_g50_i50'
    return dict(run_name=name, run_purpose='smoke', prompt_split='development',
                method_name=method, global_alpha=alpha, attack_name='clean',
                replicate_id=1, protocol_version='v1',
                watermark_key_id='treering-rand-wseed-999999', start=0, end=1,
                model_id=MODEL, image_length=512, num_images=1, guidance_scale=7.5,
                num_inference_steps=50, test_num_inference_steps=50,
                gen_seed=0, w_seed=999999, w_pattern='rand', w_radius=10,
                w_channel=0, w_mask_shape='circle', w_measurement='l1_complex',
                w_injection='complex', w_pattern_const=0, rand_aug=0,
                prompt_file=str(SERVER_REPO / '.local_data/prompts/splits' /
                                PROMPT_FILES['development']),
                reference_model=None, reference_model_pretrain=None,
                with_tracking=False, overwrite_output=False,
                r_degree=None, jpeg_ratio=None, crop_scale=None, crop_ratio=None,
                gaussian_blur_r=None, gaussian_std=None, brightness_factor=None)


def command(label):
    result = [sys.executable, '-B', str(REPO / 'run_tree_ring_watermark.py')]
    for key, value in config(label).items():
        if value is not None and value is not False:
            result.extend(['--' + key, str(value)])
    return result


def paths(label):
    name = config(label)['run_name']
    return tuple(REPO / '.local_outputs/runs' / (name + suffix)
                 for suffix in ('.json', '.samples.jsonl'))


def code_preflight(head):
    require(REPO.resolve() == SERVER_REPO.resolve(), 'Use the persistent server checkout')
    require(Path(sys.prefix).resolve() == VENV.resolve(), 'Use the existing treering venv')
    require(bool(re.fullmatch('[0-9a-f]{40}', head or '')), 'Supply the explicitly approved full HEAD')
    require(head != PARENT_GATE, 'Naming changes require a newly reviewed and approved commit')
    require(git('rev-parse', 'HEAD') == head, 'HEAD differs from approved commit')
    require(git('branch', '--show-current') == BRANCH, 'Unexpected branch')
    require(git('status', '--porcelain') == '', 'Worktree and index must be clean')
    git('merge-base', '--is-ancestor', PARENT_GATE, head)
    require(Path(MODEL).is_dir(), 'Persistent model directory is missing')


def frozen_inputs():
    results = {}
    for split, spec in FROZEN_PROMPT_SPLITS.items():
        path = REPO / '.local_data/prompts/splits' / PROMPT_FILES[split]
        require(file_sha256(path) == spec['sha256'], f'Frozen {split} prompt hash mismatch')
        require(len([s for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]) ==
                spec['count'], f'Frozen {split} prompt count mismatch')
        results[str(path)] = spec['sha256']
    for path, expected in STAGE_C.items():
        require(file_sha256(path) == expected, f'Historical Stage C evidence changed: {path}')
        results[str(path)] = expected
    return results


def model_snapshot():
    # Full weight identity for this validation only; no download or model load.
    files = sorted(p for p in Path(MODEL).rglob('*') if p.is_file())
    require(files, 'Model directory is empty')
    return {str(p): file_sha256(p) for p in files}


def preflight(head):
    code_preflight(head)
    protected = frozen_inputs()
    protected.update({str(p): file_sha256(p) for p in
                      sorted((REPO / '.local_outputs').rglob('*')) if p.is_file()})
    for label in RUNS:
        build_run_identity(SimpleNamespace(**config(label)))
        require(not any(p.exists() for p in paths(label)), f'Output identity already occupied: {label}')
    return dict(status='PREFLIGHT_PASS', plan_version=PLAN_VERSION, expected_head=head,
                protected=protected, model_files=model_snapshot(),
                prepared_commands={label: command(label) for label in RUNS},
                gpu_experiment_started=False)


def verify_baseline(baseline, head, verify_model=False):
    code_preflight(head)
    require(baseline['status'] == 'PREFLIGHT_PASS' and baseline['plan_version'] == PLAN_VERSION,
            'Use a passing preflight from this plan')
    require(baseline['expected_head'] == head, 'Baseline HEAD mismatch')
    expected_commands = {label: command(label) for label in RUNS}
    require(baseline['prepared_commands'] == expected_commands, 'Prepared commands changed')
    for path, expected in baseline['protected'].items():
        require(file_sha256(path) == expected, f'Protected input/evidence changed: {path}')
    frozen_inputs()
    if verify_model:
        require(model_snapshot() == baseline['model_files'], 'Persistent model contents changed')


def numeric_sanity():
    # Imported ONLY in explicitly authorized GPU mode; no pipeline/weights loaded.
    import torch
    from optim_utils import (get_watermarking_mask, get_watermarking_pattern,
                             inject_watermark, inject_watermark_for_method,
                             measure_latent_residual, set_random_seed)
    require(torch.cuda.is_available(), 'CUDA must be available; no CPU fallback')
    require(torch.cuda.device_count() == 1, 'Expose exactly one approved GPU')
    require('B200' in torch.cuda.get_device_name(0), 'Expected NVIDIA B200')
    require(torch.__version__ == '2.7.0a0+ecf3bae40a.nv25.02' and
            torch.version.cuda == '12.8', 'Environment differs from CPU gate')
    args = SimpleNamespace(**config('original'))
    observations = []
    class LatentsOnly:
        def get_random_latents(self):
            return torch.randn(1, 4, 64, 64, device='cuda', dtype=torch.float16)
    for dtype in (torch.float16, torch.float32):
        for seed in (0, 1):
            key_pipe = LatentsOnly()
            if dtype == torch.float32:
                key_pipe.get_random_latents = lambda: torch.randn(
                    1, 4, 64, 64, device='cuda', dtype=dtype)
            key = get_watermarking_pattern(key_pipe, args, 'cuda')
            set_random_seed(seed)
            base = torch.randn(1, 4, 64, 64, device='cuda', dtype=dtype)
            mask = get_watermarking_mask(base, args, 'cuda')
            original = inject_watermark(base, mask, key, args)
            require(original.dtype == dtype, 'Original injection changed dtype')
            originals = [t.clone() for t in (base, key, mask)]
            rng_cpu = torch.get_rng_state().clone()
            rng_cuda = torch.cuda.get_rng_state().clone()
            for alpha in (0., .5, 1.):
                args.method_name = 'globally_weaker_tree_ring'
                args.global_alpha = alpha
                actual = inject_watermark_for_method(base, mask, key, args)
                expected = (base if alpha == 0 else original if alpha == 1 else
                            (base.float() + alpha * (original.float() - base.float())).to(dtype))
                require(torch.equal(actual, expected), 'Global differs from rounded frozen formula')
                require(actual.dtype == dtype and actual.device == base.device, 'Device/dtype mismatch')
                require(all(torch.equal(t, old) for t, old in zip((base, key, mask), originals)),
                        'Shared base/key/mask mutated')
                require(torch.equal(torch.get_rng_state(), rng_cpu) and
                        torch.equal(torch.cuda.get_rng_state(), rng_cuda), 'Injection consumed RNG')
                if alpha == 0:
                    require(actual.data_ptr() != base.data_ptr(), 'Alpha zero must return an independent clone')
                delta = actual.double() - base.double()
                l2 = delta.square().sum().sqrt().item()
                rms = l2 / math.sqrt(base.numel())
                budget = measure_latent_residual(base, actual)
                require(math.isclose(budget['latent_residual_l2'], l2, rel_tol=1e-12, abs_tol=1e-12),
                        'L2 differs from independent float64 reference')
                require(math.isclose(budget['latent_residual_rms'], rms, rel_tol=1e-12, abs_tol=1e-12),
                        'RMS differs from independent float64 reference')
                require(all(math.isfinite(x) for x in budget.values()), 'Non-finite realized budget')
                observations.append(dict(dtype=str(dtype), seed=seed, alpha=alpha, **budget))
            args.method_name, args.global_alpha = 'original_tree_ring', None
    torch.cuda.synchronize()
    return dict(status='GPU_NUMERIC_SANITY_PASS', plan_version=PLAN_VERSION,
                gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                cuda_version=torch.version.cuda, observations=observations,
                model_loaded=False, image_generation_started=False)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), 'Non-finite value in output')
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def validate_output(summary, records, label, head, prompt):
    expected = config(label)
    finite_tree(summary)
    finite_tree(records)
    require(summary['schema_version'] == 'treering-experiment-v2', 'Schema mismatch')
    require(summary['run_name'] == expected['run_name'], 'Run name mismatch')
    require(summary['num_samples'] == len(records) == 1 and summary['results'] == records,
            'Summary/JSONL missing, duplicated, or inconsistent samples')
    require(summary['git'] == dict(commit=head, branch=BRANCH, dirty=False), 'Recorded Git mismatch')
    require(summary['prompt_source']['sha256'] == FROZEN_PROMPT_SPLITS['development']['sha256'] and
            summary['prompt_source']['nonempty_line_count'] == 32 and
            summary['prompt_source']['path'] == expected['prompt_file'], 'Prompt identity mismatch')
    for field, value in expected.items():
        require(summary['parameters'].get(field) == value, f'Parameter mismatch: {field}')
    env = summary['environment']
    require(env['cuda_available'] is True and env['device'] == 'cuda' and
            env['torch_dtype'] == 'torch.float16' and 'B200' in env['gpu_name'], 'GPU fallback/dtype mismatch')
    require(env['torch_version'] == '2.7.0a0+ecf3bae40a.nv25.02' and
            env['torch_cuda_version'] == '12.8', 'Recorded environment mismatch')
    require(env['python_executable'] == str(VENV / 'bin/python') and
            env['python_version'].startswith('3.12.3'), 'Recorded Python/venv mismatch')
    identity = summary['run_identity']
    require(identity['sample_range'] == dict(start=0, end_exclusive=1, count=1), 'Sample range mismatch')
    method_params = {} if expected['global_alpha'] is None else {'alpha': expected['global_alpha']}
    row = records[0]
    require(row['sample_index'] == 0 and row['source_row_id'] == 1200 and
            row['prompt'] == prompt and row['sample_id'] == expected['run_name'] + ':0', 'Sample/source identity mismatch')
    for field in ('prompt_split', 'method_name', 'attack_name', 'replicate_id',
                  'protocol_version', 'watermark_key_id', 'run_purpose'):
        require(identity[field] == row[field] == expected[field], f'Record identity mismatch: {field}')
    require(identity['run_naming_version'] == row['run_naming_version'] == RUN_NAMING_VERSION,
            'Naming version mismatch')
    require(identity['method_parameters'] == row['method_parameters'] == method_params and
            identity['attack_parameters'] == row['attack_parameters'] == {}, 'Method/attack mismatch')
    require(row['generation_seed'] == 0 and row['watermark_seed'] == 999999, 'Realized seed mismatch')
    require(row['w_no_sim'] == row['w_sim'] == 0, 'Unexpected reference-model similarities')
    require(row['no_w_score'] == -row['no_w_metric'] and row['w_score'] == -row['w_metric'], 'Score sign mismatch')
    for key in ('no_w_metric', 'w_metric', 'no_w_score', 'w_score',
                'latent_residual_rms', 'latent_residual_l2', 'sample_duration_seconds'):
        require(type(row[key]) in (int, float) and math.isfinite(row[key]), f'Invalid required number: {key}')
    require(row['latent_residual_rms'] >= 0 and row['latent_residual_l2'] >= 0, 'Negative budget')
    require(math.isclose(row['latent_residual_l2'], row['latent_residual_rms'] * 128,
                         rel_tol=1e-12, abs_tol=1e-12), 'RMS/L2 normalization mismatch')
    require(summary['duration_seconds'] >= row['sample_duration_seconds'] > 0, 'Invalid durations')
    started = datetime.fromisoformat(summary['started_at_utc'])
    completed = datetime.fromisoformat(summary['completed_at_utc'])
    require(started.tzinfo is not None and completed.tzinfo is not None and completed > started,
            'Invalid timestamps')
    require(set(row['paired_inputs']) == {'base_latent_sha256', 'watermark_key_sha256',
                                         'watermark_mask_sha256'}, 'Missing pairing fingerprints')
    require(all(re.fullmatch('[0-9a-f]{64}', s) for s in row['paired_inputs'].values()),
            'Malformed pairing fingerprints')
    for metric in ('clip_score_mean', 'w_clip_score_mean', 'auc',
                   'oracle_max_balanced_accuracy', 'tpr_at_1pct_fpr'):
        value = summary['metrics'][metric]
        require(type(value) in (int, float) and math.isfinite(value), f'Invalid required metric: {metric}')
    return row


def accept_outputs(head):
    prompt_path = REPO / '.local_data/prompts/splits' / PROMPT_FILES['development']
    prompt = [s.strip() for s in prompt_path.read_text(encoding='utf-8').splitlines() if s.strip()][0]
    rows, hashes = {}, {}
    for label in RUNS:
        summary_path, jsonl_path = paths(label)
        summary = strict_json(summary_path.read_text(encoding='utf-8'))
        lines = jsonl_path.read_text(encoding='utf-8').splitlines()
        require(len(lines) == 1 and lines[0].strip(), 'JSONL must contain exactly one record')
        rows[label] = validate_output(summary, [strict_json(lines[0])], label, head, prompt)
        for p in (summary_path, jsonl_path):
            hashes[str(p)] = file_sha256(p)
    original = rows['original']
    for row in rows.values():
        require(row['paired_inputs'] == original['paired_inputs'], 'Cross-method base/key/mask mismatch')
        require(row['no_w_metric'] == original['no_w_metric'], 'Paired No-Watermark distances differ')
    zero, one = rows['global-a0'], rows['global-a1']
    require(zero['latent_residual_l2'] == zero['latent_residual_rms'] == 0 and
            zero['w_metric'] == zero['no_w_metric'], 'Alpha zero endpoint mismatch')
    for field in ('w_metric', 'latent_residual_l2', 'latent_residual_rms'):
        require(one[field] == original[field], f'Alpha one endpoint mismatch: {field}')
    require(original['latent_residual_l2'] > 0 and
            0 < rows['global-a05']['latent_residual_l2'] < original['latent_residual_l2'],
            'Intermediate realized budget is not strictly between endpoints')
    return dict(status='GLOBAL_GPU_SMALL_SMOKE_GATE_PASS', plan_version=PLAN_VERSION,
                output_hashes=hashes, paired_inputs=original['paired_inputs'],
                paper_result=False, full_development_gate=False,
                alpha_grid_selected=False, protocol_v1_1_approved=False)


def review_process_reports(report_paths, head, baseline_path, sanity_path):
    require(len(report_paths) == len(RUNS), 'Supply all four --process-report paths')
    found, hashes = set(), {}
    for path in report_paths:
        report = load_report(path)
        label = report['run']
        require(label in RUNS and label not in found, 'Duplicate/unexpected process report')
        found.add(label)
        require(report['status'] == 'SMOKE_PROCESS_PASS' and report['returncode'] == 0 and
                report['expected_head'] == head and report['command'] == command(label) and
                report['baseline_sha256'] == file_sha256(baseline_path) and
                report['sanity_sha256'] == file_sha256(sanity_path), 'Process report/provenance mismatch')
        log_path = Path(report['console_log'])
        require(file_sha256(log_path) == report['console_sha256'], 'Console log changed')
        log = log_path.read_text(encoding='utf-8', errors='replace')
        require('Traceback (most recent call last)' not in log, 'Traceback in smoke console')
        allowed = ('Importing from timm.models.layers is deprecated',
                   'ComplexHalf support is experimental')
        warnings_found = [line.strip() for line in log.splitlines()
                          if re.search(r'\b\w*Warning:', line)]
        require(all(any(message in line for message in allowed) for line in warnings_found),
                'Unexpected warning: review before accepting the GPU gate')
        hashes[str(path)] = file_sha256(path)
        hashes[str(log_path)] = report['console_sha256']
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='plan',
                        choices=['plan', 'preflight', 'sanity', 'run-one', 'accept'])
    parser.add_argument('--expected-head')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--sanity-report', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--run', choices=RUNS)
    parser.add_argument('--execute-gpu-sanity', action='store_true')
    parser.add_argument('--execute-smoke', action='store_true')
    parser.add_argument('--process-report', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.mode == 'plan':
        print(json.dumps(dict(plan_version=PLAN_VERSION, execution_authorized=False,
                              runs={label: config(label) for label in RUNS}), indent=2))
        return
    require(args.report is not None, 'Supply a new report path in TR_Transfer')
    # Check report identity before any GPU work or model loading.
    require(args.report.resolve().parent == TRANSFER.resolve() and not args.report.exists() and
            not Path(str(args.report) + '.sha256').exists(), 'Report path invalid or already occupied')
    if args.mode == 'preflight':
        report = preflight(args.expected_head)
    else:
        require(args.baseline is not None, 'Supply the preflight baseline')
        baseline = load_report(args.baseline)
        verify_baseline(baseline, args.expected_head, verify_model=True)
        if args.mode == 'sanity':
            require(args.execute_gpu_sanity, 'GPU sanity requires explicit execution authorization/flag')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                report = numeric_sanity()
            report['warnings'] = [str(w.message) for w in caught]
            require(all('Importing from timm.models.layers is deprecated' in str(w.message) or
                        'ComplexHalf support is experimental' in str(w.message) for w in caught),
                    'Unexpected numeric sanity warning; retain console and review')
        else:
            require(args.sanity_report is not None, 'Supply a passing numeric sanity report')
            sanity = load_report(args.sanity_report)
            require(sanity['status'] == 'GPU_NUMERIC_SANITY_PASS' and
                    sanity['expected_head'] == args.expected_head and
                    sanity['baseline_sha256'] == file_sha256(args.baseline), 'Sanity provenance mismatch')
            if args.mode == 'run-one':
                require(args.execute_smoke and args.run is not None, 'Explicit smoke execution flag/run required')
                require(not any(p.exists() for p in paths(args.run)), 'Preserve existing/partial run outputs')
                import torch
                require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
                        'B200' in torch.cuda.get_device_name(0), 'Expected one B200; no fallback')
                require(torch.__version__ == '2.7.0a0+ecf3bae40a.nv25.02' and
                        torch.version.cuda == '12.8', 'Unexpected existing torch/CUDA environment')
                log = Path(str(args.report) + '.console.log')
                require(not log.exists(), 'Console log already exists')
                environment = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                                   HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1')
                started = time.perf_counter()
                with log.open('x', encoding='utf-8') as stream:
                    completed = subprocess.run(command(args.run), cwd=REPO, env=environment,
                                               stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
                report = dict(status='SMOKE_PROCESS_PASS' if completed.returncode == 0 else 'SMOKE_PROCESS_FAIL',
                              run=args.run, returncode=completed.returncode, command=command(args.run),
                              duration_seconds=time.perf_counter() - started,
                              console_log=str(log), console_sha256=file_sha256(log))
            else:
                process_hashes = review_process_reports(
                    args.process_report, args.expected_head, args.baseline, args.sanity_report)
                report = accept_outputs(args.expected_head)
                report['process_evidence_hashes'] = process_hashes
            report['sanity_sha256'] = file_sha256(args.sanity_report)
        verify_baseline(baseline, args.expected_head, verify_model=True)
        report['baseline_sha256'] = file_sha256(args.baseline)
    report.update(plan_version=PLAN_VERSION, expected_head=args.expected_head,
                  created_at_utc=datetime.now(timezone.utc).isoformat())
    write_report(args.report, report)
    print(report['status'])
    if report.get('returncode', 0):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
