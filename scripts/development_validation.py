#!/usr/bin/env python3
"""Original/Global 32-prompt Development preparation. Default: plan only.

CPU verification and GPU execution are separate explicit modes. This does not
implement Saliency, choose parameters, approve v1.1, or pass full Development.
"""
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import global_gpu_validation as smoke
from image_artifacts import verify_image_pair
from run_naming import build_run_identity
from types import SimpleNamespace

PLAN_VERSION = 'original-global-development-plan-v1'
RUNS = smoke.RUNS
require = smoke.require
PRIOR_HEAD = '6da7a016d0d619c00e1c2643b2efd6fb739eb9d0'
PRIOR_REVIEW_SHA = 'c74bd24670a2e488047613d8714c5ab787dad70988495c862046c9510bd5da36'
ALLOWED_WARNINGS = (
    'FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers',
    "UserWarning: ComplexHalf support is experimental and many operators don't support it yet. "
    '(Triggered internally at /opt/pytorch/pytorch/aten/src/ATen/EmptyTensor.cpp:50.)',
    'FutureWarning: The decode_latents method is deprecated and will be removed '
    'in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead',
)


def config(label):
    cfg = smoke.config(label)
    cfg.update(run_name=cfg['run_name'].replace('_smoke_s0_e1_', '_full_s0_e32_'),
               run_purpose='full', end=32, save_paired_images=True, require_cuda=True)
    build_run_identity(SimpleNamespace(**cfg))
    return cfg


def command(label):
    result = [sys.executable, '-B', str(REPO / 'run_tree_ring_watermark.py')]
    for key, value in config(label).items():
        if value is True:
            result.append('--' + key)
        elif value is not None and value is not False:
            result.extend(['--' + key, str(value)])
    return result


def paths(label):
    name = config(label)['run_name']
    root = REPO / '.local_outputs/runs'
    return tuple(root / (name + suffix) for suffix in ('.json', '.samples.jsonl', '.images'))


def console_review(log):
    require('Traceback (most recent call last)' not in log, 'Traceback in console')
    found = [line.strip() for line in log.splitlines() if re.search(r'\b\w*Warning:', line)]
    require(all(re.search(r'\b\w*Warning:.*$', line).group() in ALLOWED_WARNINGS for line in found),
            'Unexplained warning; preserve evidence and review')
    return found


def check_report_path(path):
    require(path is not None, 'Supply a new --report in TR_Transfer')
    path = Path(path)
    require(path.resolve().parent == smoke.TRANSFER.resolve(), 'Invalid report directory')
    for suffix in ('', '.sha256', '.console.log'):
        require(not Path(str(path) + suffix).exists(), 'Report/log/checksum already occupied')


def verify_prior_smoke(path, head):
    require(smoke.file_sha256(path) == PRIOR_REVIEW_SHA, 'Use the archived, reviewed GPU smoke evidence')
    report = smoke.load_report(path)
    require(report['status'] == 'GLOBAL_GPU_SMALL_SMOKE_REVIEWED_GATE_PASS' and
            report['expected_head'] == PRIOR_HEAD, 'Prior GPU smoke identity mismatch')
    for flag in ('paper_result', 'full_development_gate', 'alpha_grid_selected', 'protocol_v1_1_approved'):
        require(report[flag] is False, 'Unexpected prior smoke scope')
    for group in ('output_hashes', 'process_evidence_hashes'):
        for evidence, digest in report[group].items():
            require(smoke.file_sha256(evidence) == digest, 'Prior smoke evidence changed')
    smoke.git('merge-base', '--is-ancestor', PRIOR_HEAD, head)
    return smoke.file_sha256(path)


def check_inventory_sources(inventory):
    """Bind the reviewed test inventory to source contents across CRLF/LF hosts."""
    import hashlib
    sources = inventory['source_sha256_lf']
    actual_paths = {p.relative_to(REPO).as_posix() for p in (REPO / 'tests').glob('test_*.py')}
    require(actual_paths <= set(sources), 'Test inventory omits current test sources')
    for name, expected in sources.items():
        path = REPO / name
        require(path.resolve().is_relative_to(REPO.resolve()), 'Invalid inventory source path')
        digest = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        require(digest == expected, f'Reviewed source changed: {name}')


def observed_test_ids(log):
    result = []
    for name, location in re.findall(r'^(test_\w+) \(([^)]+)\)', log, re.M):
        result.append(location if location.endswith('.' + name) else f'{location}.{name}')
    return result


def offline_environment(cpu=False):
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', WANDB_MODE='disabled', PYTHONDONTWRITEBYTECODE='1')
    if cpu:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def run_logged(cmd, log_path, timeout, cpu=False):
    """Persist failure/timeout console without retry or deleting partial output."""
    started = time.perf_counter()
    with Path(log_path).open('x', encoding='utf-8') as stream:
        try:
            completed = subprocess.run(cmd, cwd=REPO, env=offline_environment(cpu),
                                       stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            returncode, failure = completed.returncode, None
        except subprocess.TimeoutExpired:
            returncode, failure = None, 'timeout'
        except OSError as error:
            returncode, failure = None, f'{type(error).__name__}: {error}'
    return dict(returncode=returncode, failure=failure,
                duration_seconds=time.perf_counter() - started,
                console_log=str(log_path), console_sha256=smoke.file_sha256(log_path))


def cpu_gate(head, expected_tests, report_path):
    smoke.code_preflight(head)
    require(expected_tests, 'Supply the locally reviewed test inventory')
    inventory = smoke.load_report(expected_tests)
    check_inventory_sources(inventory)
    expected = inventory['test_ids']
    require(expected and len(expected) == len(set(expected)) and all(isinstance(x, str) for x in expected),
            'Invalid expected test inventory')
    protected = smoke.frozen_inputs()
    protected.update({str(p): smoke.file_sha256(p) for p in sorted((REPO / '.local_outputs').rglob('*')) if p.is_file()})
    probe = (
        "import json, os, unittest, torch; "
        "from experiment_records import collect_environment; "
        "print('CPU_ENVIRONMENT=' + json.dumps(dict(collect_environment(torch), "
        "cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))), flush=True); "
        "result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests')); "
        "raise SystemExit(0 if result.wasSuccessful() else 1)"
    )
    cmd = [sys.executable, '-B', '-c', probe]
    report = run_logged(cmd, Path(str(report_path) + '.console.log'), 600, cpu=True)
    log = Path(report['console_log']).read_text(encoding='utf-8', errors='replace')
    observed = observed_test_ids(log)
    try:
        probes = re.findall(r'^CPU_ENVIRONMENT=(.*)$', log, re.M)
        require(len(probes) == 1, 'Missing CPU environment evidence')
        environment = smoke.strict_json(probes[0])
        require(environment['cuda_visible_devices'] == '' and environment['cuda_available'] is False and
                environment['torch_version'] == '2.7.0a0+ecf3bae40a.nv25.02' and
                environment['torch_cuda_version'] == '12.8' and
                environment['python_version'].startswith('3.12.3') and
                environment['python_executable'] == str(smoke.VENV / 'bin/python'), 'CPU environment changed')
        report['environment'] = environment
        warnings = console_review(log)
        valid = (report['returncode'] == 0 and sorted(observed) == sorted(expected) and
                 len(observed) == len(set(observed)) and
                 re.search(rf'^Ran {len(expected)} tests? in ', log, re.M) is not None and
                 re.search(r'^OK\s*$', log, re.M) is not None and
                 not re.search(r'\bskipped\b', log, re.I))
        for path, digest in protected.items():
            require(smoke.file_sha256(path) == digest, 'CPU gate changed protected evidence')
        smoke.code_preflight(head)
    except (ValueError, OSError, KeyError, TypeError) as error:
        valid, warnings = False, []
        report['review_failure'] = str(error)
    report.update(status='DEVELOPMENT_SERVER_CPU_GATE_PASS' if valid else 'DEVELOPMENT_SERVER_CPU_GATE_FAIL',
                  command=cmd, test_ids=observed, expected_test_inventory_sha256=smoke.file_sha256(expected_tests),
                  protected=protected, warnings=warnings, gpu_hidden=True, experiment_started=False)
    return report


def preflight(head, cpu_path, prior_path, timeout):
    smoke.code_preflight(head)
    require(cpu_path is not None and prior_path is not None, 'Supply CPU gate and archived prior smoke')
    cpu = smoke.load_report(cpu_path)
    require(cpu['status'] == 'DEVELOPMENT_SERVER_CPU_GATE_PASS' and cpu['expected_head'] == head and
            cpu['returncode'] == 0 and cpu['plan_version'] == PLAN_VERSION, 'Current CPU gate missing/mismatched')
    require(cpu['environment']['cuda_available'] is False and
            cpu['environment']['cuda_visible_devices'] == '', 'CPU gate did not hide GPU')
    prior_sha = verify_prior_smoke(prior_path, head)
    protected = smoke.frozen_inputs()
    protected.update({str(p): smoke.file_sha256(p) for p in sorted((REPO / '.local_outputs').rglob('*')) if p.is_file()})
    for evidence in (cpu_path, prior_path):
        for p in (Path(evidence), Path(str(evidence) + '.sha256')):
            protected[str(p)] = smoke.file_sha256(p)
    prior = smoke.load_report(prior_path)
    for group in ('output_hashes', 'process_evidence_hashes'):
        protected.update(prior[group])
    cpu_log = Path(cpu['console_log'])
    require(smoke.file_sha256(cpu_log) == cpu['console_sha256'], 'CPU console changed')
    protected[str(cpu_log)] = cpu['console_sha256']
    for label in RUNS:
        require(not any(p.exists() for p in paths(label)), f'Preserve occupied run: {label}')
    require(type(timeout) is int and 1 <= timeout <= 14400, 'Explicit per-run budget must be 1–14400 seconds')
    return dict(status='DEVELOPMENT_PREFLIGHT_PASS', protected=protected, model_files=smoke.model_snapshot(),
                prepared_commands={label: command(label) for label in RUNS}, run_timeout_seconds=timeout,
                cpu_gate_sha256=smoke.file_sha256(cpu_path), prior_reviewed_smoke_sha256=prior_sha,
                prior_smoke_head=PRIOR_HEAD, prior_evidence_relabelled=False, experiment_started=False)


def verify_baseline(baseline, head):
    smoke.code_preflight(head)
    require(baseline['status'] == 'DEVELOPMENT_PREFLIGHT_PASS' and
            baseline['plan_version'] == PLAN_VERSION and baseline['expected_head'] == head, 'Baseline mismatch')
    require(baseline['prepared_commands'] == {label: command(label) for label in RUNS}, 'Commands changed')
    timeout = baseline['run_timeout_seconds']
    require(type(timeout) is int and 1 <= timeout <= 14400, 'Invalid recorded run budget')
    for path, digest in baseline['protected'].items():
        require(smoke.file_sha256(path) == digest, f'Protected input/evidence changed: {path}')
    require(smoke.model_snapshot() == baseline['model_files'], 'Persistent model changed')


def validate_output(summary, rows, label, head, prompts):
    cfg = config(label)
    smoke.finite_tree(summary)
    smoke.finite_tree(rows)
    require(len(prompts) == len(rows) == summary['num_samples'] == 32 and summary['results'] == rows,
            'Expected exactly 32 consistent sample records')
    require(summary['schema_version'] == 'treering-experiment-v2' and summary['run_name'] == cfg['run_name'],
            'Schema/run mismatch')
    require(summary['git'] == dict(commit=head, branch=smoke.BRANCH, dirty=False), 'Recorded Git mismatch')
    source = summary['prompt_source']
    require(source['sha256'] == smoke.FROZEN_PROMPT_SPLITS['development']['sha256'] and
            source['nonempty_line_count'] == 32 and source['path'] == cfg['prompt_file'], 'Prompt identity mismatch')
    for field, value in cfg.items():
        require(summary['parameters'].get(field) == value, f'Parameter mismatch: {field}')
    env = summary['environment']
    require(env['cuda_available'] is True and env['device'] == 'cuda' and env['torch_dtype'] == 'torch.float16' and
            'B200' in env['gpu_name'] and env['torch_version'] == '2.7.0a0+ecf3bae40a.nv25.02' and
            env['torch_cuda_version'] == '12.8' and env['python_executable'] == str(smoke.VENV / 'bin/python') and
            env['python_version'].startswith('3.12.3'), 'Recorded GPU/environment mismatch')
    identity = summary['run_identity']
    require(identity['sample_range'] == dict(start=0, end_exclusive=32, count=32), 'Range mismatch')
    params = {} if cfg['global_alpha'] is None else dict(alpha=cfg['global_alpha'])
    identity_fields = ('prompt_split', 'method_name', 'attack_name', 'replicate_id',
                       'protocol_version', 'watermark_key_id', 'run_purpose')
    total = 0.
    for i, row in enumerate(rows):
        require(type(row['sample_index']) is int and row['sample_index'] == i and
                row['source_row_id'] == 1200 + i and row['prompt'] == prompts[i] and
                row['sample_id'] == cfg['run_name'] + ':' + str(i), 'Sample/source/order mismatch')
        for field in identity_fields:
            require(identity[field] == row[field] == cfg[field], f'Identity mismatch: {field}')
        require(identity['run_naming_version'] == row['run_naming_version'] == smoke.RUN_NAMING_VERSION and
                identity['method_parameters'] == row['method_parameters'] == params and
                identity['attack_parameters'] == row['attack_parameters'] == {}, 'Method/attack/naming mismatch')
        require(row['generation_seed'] == i and row['watermark_seed'] == 999999, 'Seed mismatch')
        require(row['clip_similarity_available'] is False and row['w_no_sim'] == row['w_sim'] == 0,
                'CLIP must be marked unavailable in this stage')
        for key in ('no_w_metric', 'w_metric', 'no_w_score', 'w_score', 'latent_residual_rms',
                    'latent_residual_l2', 'sample_duration_seconds'):
            require(type(row[key]) in (float, int) and math.isfinite(row[key]), f'Invalid number: {key}')
        require(row['no_w_score'] == -row['no_w_metric'] and row['w_score'] == -row['w_metric'], 'Score sign mismatch')
        require(row['latent_residual_rms'] >= 0 and row['latent_residual_l2'] >= 0 and
                math.isclose(row['latent_residual_l2'], 128 * row['latent_residual_rms'],
                             rel_tol=1e-12, abs_tol=1e-12), 'RMS/L2 mismatch')
        require(row['sample_duration_seconds'] > 0, 'Invalid sample duration')
        total += row['sample_duration_seconds']
        fingerprints = row['paired_inputs']
        require(set(fingerprints) == {'base_latent_sha256', 'watermark_key_sha256', 'watermark_mask_sha256'} and
                all(isinstance(x, str) and re.fullmatch('[0-9a-f]{64}', x) for x in fingerprints.values()),
                'Pairing fingerprints missing/invalid')
        require('image_artifacts' in row, 'Missing paired images')
    require(type(summary['duration_seconds']) in (float, int) and summary['duration_seconds'] >= total > 0,
            'Invalid total duration')
    start = datetime.fromisoformat(summary['started_at_utc'])
    end = datetime.fromisoformat(summary['completed_at_utc'])
    require(start.tzinfo is not None and end.tzinfo is not None and end > start, 'Invalid timestamps')
    for name in ('clip_score_mean', 'w_clip_score_mean', 'auc', 'oracle_max_balanced_accuracy', 'tpr_at_1pct_fpr'):
        value = summary['metrics'][name]
        require(type(value) in (float, int) and math.isfinite(value), f'Invalid aggregate: {name}')
    # Recompute descriptive detection metrics; never choose a fixed threshold here.
    from sklearn import metrics
    fpr, tpr, _ = metrics.roc_curve([0] * 32 + [1] * 32,
                                   [r['no_w_score'] for r in rows] + [r['w_score'] for r in rows])
    expected = dict(clip_score_mean=0., w_clip_score_mean=0., auc=float(metrics.auc(fpr, tpr)),
                    oracle_max_balanced_accuracy=float(max((1 + tpr - fpr) / 2)),
                    tpr_at_1pct_fpr=float(tpr[fpr < .01][-1]))
    for name, value in expected.items():
        require(math.isclose(summary['metrics'][name], value, rel_tol=1e-12, abs_tol=1e-12),
                f'Aggregate inconsistent with sample records: {name}')
        require(summary[name] == summary['metrics'][name], f'Top-level aggregate mismatch: {name}')
    require(summary['acc'] == expected['oracle_max_balanced_accuracy'], 'Legacy acc mismatch')
    return rows


def accept_outputs(head):
    prompt_path = REPO / '.local_data/prompts/splits' / smoke.PROMPT_FILES['development']
    require(smoke.file_sha256(prompt_path) == smoke.FROZEN_PROMPT_SPLITS['development']['sha256'], 'Prompt hash changed')
    prompts = [s.strip() for s in prompt_path.read_text(encoding='utf-8').splitlines() if s.strip()]
    all_rows, hashes = {}, {}
    for label in RUNS:
        summary_path, jsonl_path, image_dir = paths(label)
        summary = smoke.strict_json(summary_path.read_text(encoding='utf-8'))
        lines = jsonl_path.read_text(encoding='utf-8').splitlines()
        require(len(lines) == 32 and all(line.strip() for line in lines), 'Invalid JSONL count/blank rows')
        rows = validate_output(summary, [smoke.strict_json(line) for line in lines], label, head, prompts)
        require(not image_dir.is_symlink() and image_dir.is_dir(), 'Invalid image directory')
        expected_names = {f'{i:04d}.{role}.png' for i in range(32) for role in ('no_watermark', 'watermarked')}
        require({p.name for p in image_dir.iterdir()} == expected_names, 'Missing/unexpected image files')
        for i, row in enumerate(rows):
            hashes.update(verify_image_pair(image_dir, i, row['image_artifacts']))
        all_rows[label] = rows
        for p in (summary_path, jsonl_path):
            hashes[str(p)] = smoke.file_sha256(p)
    for i, original in enumerate(all_rows['original']):
        for rows in all_rows.values():
            row = rows[i]
            require(row['paired_inputs'] == original['paired_inputs'] and row['no_w_metric'] == original['no_w_metric'],
                    f'Cross-method input/distance mismatch at {i}')
            require(row['image_artifacts']['no_watermark']['pixel_sha256'] ==
                    original['image_artifacts']['no_watermark']['pixel_sha256'], f'No-Watermark image mismatch at {i}')
        zero, middle, one = (all_rows[x][i] for x in ('global-a0', 'global-a05', 'global-a1'))
        require(zero['latent_residual_l2'] == zero['latent_residual_rms'] == 0 and
                zero['w_metric'] == zero['no_w_metric'] and
                zero['image_artifacts']['watermarked']['pixel_sha256'] ==
                zero['image_artifacts']['no_watermark']['pixel_sha256'], f'Zero endpoint mismatch at {i}')
        for field in ('w_metric', 'latent_residual_l2', 'latent_residual_rms'):
            require(one[field] == original[field], f'One endpoint mismatch at {i}: {field}')
        require(one['image_artifacts']['watermarked']['pixel_sha256'] ==
                original['image_artifacts']['watermarked']['pixel_sha256'], f'One endpoint image mismatch at {i}')
        require(0 < middle['latent_residual_l2'] < original['latent_residual_l2'], f'Intermediate budget mismatch at {i}')
    return dict(status='ORIGINAL_GLOBAL_32_DEVELOPMENT_GATE_PASS', output_hashes=hashes,
                paper_result=False, full_development_gate=False, saliency_validated=False,
                quality_metrics_validated=False, alpha_grid_selected=False, protocol_v1_1_approved=False)


def review_process_reports(report_paths, head, baseline_path):
    require(len(report_paths) == 4, 'Supply four distinct process reports')
    hashes, found = {}, set()
    for path in report_paths:
        report = smoke.load_report(path)
        label = report['run']
        require(label in RUNS and label not in found, 'Duplicate/unexpected process report')
        found.add(label)
        require(report['plan_version'] == PLAN_VERSION and report['status'] == 'DEVELOPMENT_PROCESS_PASS' and
                report['returncode'] == 0 and report['expected_head'] == head and report['command'] == command(label) and
                report['baseline_sha256'] == smoke.file_sha256(baseline_path), 'Process provenance mismatch')
        log_path = Path(report['console_log'])
        require(smoke.file_sha256(log_path) == report['console_sha256'], 'Console changed')
        console_review(log_path.read_text(encoding='utf-8', errors='replace'))
        require(report['output_hashes'] == snapshot_run(label), 'Process output/image evidence changed')
        hashes[str(path)] = smoke.file_sha256(path)
        hashes[str(log_path)] = report['console_sha256']
    return hashes


def snapshot_run(label):
    summary, jsonl, images = paths(label)
    files = [p for p in (summary, jsonl) if p.is_file()]
    if images.is_dir():
        files.extend(sorted(p for p in images.rglob('*') if p.is_file()))
    return {str(p): smoke.file_sha256(p) for p in files}


def run_one(label, head, baseline, report_path):
    require(label in RUNS, 'Supply --run')
    require(not any(p.exists() for p in paths(label)), 'Preserve existing/partial run outputs')
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and
            'B200' in torch.cuda.get_device_name(0), 'Expected one B200; no fallback')
    require(torch.__version__ == '2.7.0a0+ecf3bae40a.nv25.02' and torch.version.cuda == '12.8', 'Environment changed')
    report = run_logged(command(label), Path(str(report_path) + '.console.log'), baseline['run_timeout_seconds'])
    valid = report['returncode'] == 0
    try:
        console_review(Path(report['console_log']).read_text(encoding='utf-8', errors='replace'))
        prompt_path = REPO / '.local_data/prompts/splits' / smoke.PROMPT_FILES['development']
        prompts = [s.strip() for s in prompt_path.read_text(encoding='utf-8').splitlines() if s.strip()]
        summary_path, jsonl_path, images = paths(label)
        if valid:
            lines = jsonl_path.read_text(encoding='utf-8').splitlines()
            rows = validate_output(smoke.strict_json(summary_path.read_text(encoding='utf-8')),
                                   [smoke.strict_json(line) for line in lines], label, head, prompts)
            require(len(list(images.iterdir())) == 64, 'Unexpected image count')
            for i, row in enumerate(rows):
                verify_image_pair(images, i, row['image_artifacts'])
        verify_baseline(baseline, head)
    except (ValueError, KeyError, OSError, TypeError) as error:
        valid = False
        report['review_failure'] = f'{type(error).__name__}: {error}'
    report.update(status='DEVELOPMENT_PROCESS_PASS' if valid else 'DEVELOPMENT_PROCESS_FAIL',
                  run=label, command=command(label), output_hashes=snapshot_run(label))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='plan', choices=['plan', 'cpu-gate', 'preflight', 'run-one', 'accept'])
    parser.add_argument('--expected-head')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--expected-tests', type=Path)
    parser.add_argument('--execute-cpu-tests', action='store_true')
    parser.add_argument('--cpu-report', type=Path)
    parser.add_argument('--prior-smoke-report', type=Path)
    parser.add_argument('--run-timeout-seconds', type=int)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--run', choices=RUNS)
    parser.add_argument('--execute-development', action='store_true')
    parser.add_argument('--process-report', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.mode == 'plan':
        print(json.dumps(dict(plan_version=PLAN_VERSION, execution_authorized=False,
                              runs={label: config(label) for label in RUNS},
                              complete_development=False, parameter_selection=False), indent=2))
        return
    if args.mode == 'run-one':
        require(args.execute_development, 'Development needs separate explicit execution approval/flag')
        require(args.run is not None, 'Supply --run')
    if args.mode == 'cpu-gate':
        require(args.execute_cpu_tests, 'CPU tests need explicit execution flag')
    check_report_path(args.report)
    if args.mode == 'cpu-gate':
        report = cpu_gate(args.expected_head, args.expected_tests, args.report)
    elif args.mode == 'preflight':
        report = preflight(args.expected_head, args.cpu_report, args.prior_smoke_report, args.run_timeout_seconds)
    else:
        require(args.baseline is not None, 'Supply a passing preflight')
        baseline = smoke.load_report(args.baseline)
        verify_baseline(baseline, args.expected_head)
        if args.mode == 'run-one':
            report = run_one(args.run, args.expected_head, baseline, args.report)
        else:
            hashes = review_process_reports(args.process_report, args.expected_head, args.baseline)
            report = accept_outputs(args.expected_head)
            verify_baseline(baseline, args.expected_head)
            report['process_evidence_hashes'] = hashes
        report['baseline_sha256'] = smoke.file_sha256(args.baseline)
    report.update(plan_version=PLAN_VERSION, expected_head=args.expected_head,
                  created_at_utc=datetime.now(timezone.utc).isoformat())
    smoke.write_report(args.report, report)
    print(report['status'])
    if report['status'].endswith('_FAIL'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
