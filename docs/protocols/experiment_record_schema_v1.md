# Experiment Record Schema v1

## 1. Identity and scope

| Item | Value |
|---|---|
| Schema version | `treering-experiment-v1` |
| Implementation commit | `296a6225c9bb2db2c7373086f2b15c8deb192fb9` |

The schema records reproducibility metadata for the current DDIM-based SD2.1 runner. It has been validated by a one-sample B200 smoke test, but it is not yet sufficient for formal experiments. Required extensions are listed in Section 7.

## 2. Output contract

For run name `<run_name>`, the repository-local outputs are:

```text
.local_outputs/runs/<run_name>.json
.local_outputs/runs/<run_name>.samples.jsonl
```

The JSONL file is initialized at run start. Each completed sample is appended and flushed immediately. The summary JSON is written only after the full run completes successfully. If either output exists, the runner rejects the run by default. Overwriting is allowed only when `--overwrite_output` is supplied explicitly; formal runs must use unique names and must not use that option.

## 3. Summary JSON

The current summary contains:

```text
schema_version
run_name
num_samples
started_at_utc
completed_at_utc
duration_seconds

git.commit
git.branch
git.dirty

environment.python_executable
environment.python_version
environment.platform
environment.torch_version
environment.cuda_available
environment.torch_cuda_version
environment.gpu_name
environment.device
environment.torch_dtype

prompt_source.path
prompt_source.size_bytes
prompt_source.sha256
prompt_source.nonempty_line_count

parameters
metrics
results
```

`parameters` is the serialized runner argument namespace. `metrics` contains `clip_score_mean`, `w_clip_score_mean`, `auc`, `oracle_max_balanced_accuracy`, and `tpr_at_1pct_fpr` in the current implementation. `results` duplicates the per-sample records in the completed summary. Compatibility aliases may exist, but analysis must use the explicitly named fields.

For every formal run, `git.dirty` must be `false`, `git.commit` must equal the approved implementation commit, and the recorded environment and parameters must match the frozen protocol.

## 4. Per-sample JSONL

Each completed sample currently records:

```text
sample_id
sample_index
prompt
generation_seed
watermark_seed
no_w_metric
w_metric
no_w_score
w_score
w_no_sim
w_sim
sample_duration_seconds
```

Field semantics:

| Field | Meaning |
|---|---|
| `sample_id` | Current run-local identifier in the form `<run_name>:<sample_index>`. |
| `sample_index` | Index inside the selected prompt file; it is not the source dataset row. |
| `generation_seed` | Realized sample seed under `gen_seed + sample_index`. |
| `watermark_seed` | Numeric seed used to construct the watermark key. |
| `no_w_metric`, `w_metric` | Watermark distances for the unwatermarked and watermarked images. Smaller means closer to the key. |
| `no_w_score`, `w_score` | Negative distances. Larger means stronger watermark evidence. |
| `w_no_sim`, `w_sim` | Current similarity outputs for the paired unwatermarked and watermarked images; values are zero when no reference model is configured. |
| `sample_duration_seconds` | Elapsed wall-clock time for the sample. |

All numeric fields used in analysis must be finite. A missing, malformed, or non-finite observation is a run anomaly and must not be silently omitted.

## 5. Source-row mapping limitation

The current `sample_index` is local to the prompt file, not the original dataset `source_row_id`. Until explicit source identity is implemented, the mapping is:

```text
Formal:      source_row_id = sample_index
Pilot:       source_row_id = 1000 + sample_index
Development: source_row_id = 1200 + sample_index
```

This arithmetic mapping is valid only for the frozen contiguous split files. It must not be inferred for another prompt file or reordered data. Formal execution requires an explicit recorded `source_row_id` rather than post-hoc inference.

## 6. Current B200 validation record

The record-writing path was verified at commit `296a6225c9bb2db2c7373086f2b15c8deb192fb9` using run `manifest_v1_b200_smoke`: development prompt source, one sample, one generation step, one inversion step, NVIDIA B200, `git.dirty=false`, `no_w_metric=74.5`, `w_metric=24.953125`, total duration `108.99201348703355` seconds, and sample duration `3.396694979048334` seconds. `record_schema_verified` was true.

This was a single-sample, single-step format check. AUC, accuracy, and TPR from it are undefined or degenerate for scientific purposes and must not be reported as paper results.

## 7. Required extension before formal execution

The per-sample and/or summary contract, as appropriate, must explicitly add and validate:

```text
source_row_id
prompt_split
method_name
replicate_id
protocol_version
watermark_key_id
fixed threshold fields
attack identity
quality metrics
latent residual RMS/L2
method-specific parameters
model manifest/hash
```

At minimum, each per-sample record must directly carry `source_row_id`, `prompt_split`, `method_name`, `replicate_id`, and `protocol_version`. Threshold fields must distinguish the threshold value, calibration rule/version, predicted label, and fixed-threshold outcome. Attack identity must distinguish the clean condition from every attack and record realized parameters/randomness. Quality metric and perturbation fields must be available at sample level when the metric admits per-sample values.

The implementing commit and tests for these fields must be recorded in protocol amendment v1.1. A schema-version change should be used if consumers cannot safely interpret the extended records as `treering-experiment-v1`.

## 8. Validation invariants

For a completed run:

- Both files exist.
- JSONL row count equals `num_samples` and the expected split count.
- Each expected `source_row_id` occurs exactly once per method, replicate, and condition.
- Summary `results` and JSONL agree for shared fields.
- Seeds follow the approved, non-overlapping allocation.
- Prompt hash, Git state, environment, model identity, and parameters match the approved protocol.
- All numeric analysis values are finite.
- Start, completion, run duration, and sample durations are present and internally plausible.

See the [execution runbook](development_and_execution_runbook_v1.md) for operational checks and recovery rules.
