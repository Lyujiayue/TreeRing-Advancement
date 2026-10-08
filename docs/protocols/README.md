# TreeRing-Advancement Experimental Protocols

## Purpose and authority

This directory is the controlled index for the TreeRing-Advancement experimental protocol. The documents separate the frozen scientific design, the current record format, the operating procedure, and the amendment that must be approved before formal execution. In a conflict, the latest approved protocol amendment takes precedence; implementation behavior must never silently override an approved protocol.

## Document index

| Document | Purpose | Status |
|---|---|---|
| [Baseline Experiment Protocol v1](baseline_experiment_protocol_v1.md) | Freezes the research question, methods, model and generation settings, data splits, scales, fairness rules, metrics, and robustness conditions. | Complete; amendment v1.1 is still required before formal execution. |
| [Experiment Record Schema v1](experiment_record_schema_v1.md) | Defines the current summary JSON and per-sample JSONL records, their semantics, write behavior, and known omissions. | Implemented and smoke-tested; required identity and formal-analysis fields remain to be added. |
| [Development and Execution Runbook v1](development_and_execution_runbook_v1.md) | Defines safe development, offline deployment, execution, validation, recovery, artifact preservation, and stage gates. | Complete for the current baseline; commands for new methods must be frozen in amendment v1.1. |
| [Protocol Amendment v1.1 Template](protocol_amendment_v1_1_template.md) | Captures all remaining implementation, seed, tuning, threshold, metric, robustness, commit, evidence, command, and approval decisions. | Template only; must be completed and approved before any formal experiment. |
| `README.md` | Provides the protocol map, readiness summary, and outstanding work. | Current. |

## Verified completed work

- Offline Stable Diffusion 2.1 baseline.
- End-to-end baseline smoke test on an NVIDIA B200.
- Baseline Experiment Protocol v1.
- Frozen prompt data and server deployment.
- Seven local and server unit tests.
- Experiment record schema.
- One-sample B200 record-format smoke test.

## Work required before formal experiments

- Add explicit `source_row_id`, `method_name`, `replicate_id`, and `protocol_version` to every per-sample record. The full required field set is listed in the schema document.
- Implement Globally Weaker Tree-Ring.
- Complete end-to-end validation of Saliency-Aware Tree-Ring.
- Implement and freeze CLIP, LPIPS, PSNR, SSIM, and FID.
- Calibrate and freeze the detection threshold using the pilot split only.
- Complete and approve protocol amendment v1.1.
- Complete the 32-sample development experiment and the 200-sample pilot.

Formal runs are not authorized until every applicable stage gate in the runbook is satisfied and amendment v1.1 contains no unresolved placeholders.

## Verified B200 record-format smoke test

The following execution verified the record-writing path at implementation commit `296a6225c9bb2db2c7373086f2b15c8deb192fb9`:

| Field | Value |
|---|---|
| Run name | `manifest_v1_b200_smoke` |
| Prompt source | Development split |
| Samples | 1 |
| Generation / inversion steps | 1 / 1 |
| `no_w_metric` | 74.5 |
| `w_metric` | 24.953125 |
| Record schema verified | `true` |
| `git.dirty` | `false` |
| GPU | NVIDIA B200 |
| Run duration | 108.99201348703355 seconds |
| Sample duration | 3.396694979048334 seconds |

This was a single-sample, single-step format validation only. Its AUC, accuracy, and TPR are invalid as paper results and must not be cited as experimental evidence.
