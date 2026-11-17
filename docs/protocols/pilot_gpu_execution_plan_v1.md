# Pilot GPU Execution Plan v1

## Status and authorization boundary

This document defines the outcome-independent execution budget, order,
exclusive paths, failure preservation, and acceptance sequence for the frozen
200-prompt Pilot at commit
`7e8ee072e59ecf59449beb076716617361e8b30d`.

It is a review document only. It does not authorize GPU execution, Formal
execution, instance start or stop, dependency or environment changes, model or
asset downloads, server writes outside the new Pilot evidence root and the ten
canonical run outputs, Git commits, or pushes. A later execution approval must
name this plan and remains limited to Pilot.

The candidate grid and selection rule remain governed by
`docs/protocols/pilot_freeze_v1.md` and `pilot_selection.py`. This document does
not add, remove, reorder based on outcomes, or change any candidate.

## Bound identity

| Item | Required value |
|---|---|
| Branch | `stage-c-record-identity` |
| Commit | `7e8ee072e59ecf59449beb076716617361e8b30d` |
| Worktree | clean |
| Pilot prompts | source rows 1000-1199, exactly 200 |
| Prompt SHA-256 | `1455baf5b73b143aa07227c2dcfab96d5658c29e816fd9ddea33b591e7b70f3b` |
| Protocol | `v1.1-pilot` |
| Replicate | 1 |
| Generation seed | base 0; realized as base plus sample index |
| Watermark seed/key | `999999` / `treering-rand-wseed-999999` |
| Attack | clean |
| Steps | 50 generation / 50 inversion |
| Range | full, indices 0-199 |
| Device | existing single NVIDIA B200 only |
| Model | existing local SD2.1 directory only |
| Environment | existing `treering` environment only |

The run must be offline. No fallback to CPU, alternate GPU model, remote
checkpoint, dependency installation, cache download, or changed model asset is
permitted.

## Frozen execution order

The order is fixed before outcomes are observed. It is the order emitted by the
frozen Pilot specification:

1. `original_pilot_r1_clean_full_s0_e200_g50_i50`
2. `global_pilot_r1_clean_a0p25_full_s0_e200_g50_i50`
3. `global_pilot_r1_clean_a0p5_full_s0_e200_g50_i50`
4. `global_pilot_r1_clean_a0p75_full_s0_e200_g50_i50`
5. `saliency_pilot_r1_clean_pv10_bg1p5_fg0p2_pw2p0_full_s0_e200_g50_i50`
6. `saliency_pilot_r1_clean_pv10_bg1p0_fg0p2_pw2p0_full_s0_e200_g50_i50`
7. `saliency_pilot_r1_clean_pv10_bg0p5_fg0p2_pw2p0_full_s0_e200_g50_i50`
8. `saliency_pilot_r1_clean_pv10_bg1p0_fg0p0_pw2p0_full_s0_e200_g50_i50`
9. `saliency_pilot_r1_clean_pv10_bg0p5_fg0p0_pw2p0_full_s0_e200_g50_i50`
10. `saliency_pilot_r1_clean_pv10_bg1p0_fg0p2_pw1p0_full_s0_e200_g50_i50`

Only operational checks may be performed between runs: process status,
identity, record count, file count and format, finite-value validity, warning
allowlist, device provenance, and hashes. Detection or quality values must not
be printed, ranked, compared, summarized, or used to change the remaining
order.

After runs 1-4, an operational checkpoint must confirm that all completed
Global runs have the same 200 No-Watermark identities and PNG hashes as
Original. Each Saliency run must pass the same check against Original before
the next run starts. A mismatch stops the Pilot without examining comparative
performance.

## GPU budget

The estimate is based only on preserved Development execution timing, not on
Pilot outcomes:

| Work | Development basis | Expected Pilot time | Per-run hard timeout |
|---|---:|---:|---:|
| Original generation | about 121.6 s fixed plus 7.15 s/sample | about 26 min | 45 min |
| Each Global generation | about 125.2 s fixed plus 7.60 s/sample | about 28 min | 45 min |
| Each Saliency generation | about 130.0 s fixed plus 10.34 s/sample | about 37 min | 60 min |
| Each quality evaluation | about 3.5 min for 32 sequential samples | about 22 min | 35 min |

Expected generation occupancy is approximately 5.5 GPU-hours. Expected quality
evaluation occupancy is approximately 3.6 GPU-hours. The expected total is
therefore 9-10 GPU-hours.

The hard aggregate child-process budget is 15 GPU-hours:

- four Original/Global generation processes: at most 3 hours total;
- six Saliency generation processes: at most 6 hours total;
- ten quality processes: at most 5 hours 50 minutes total;
- the remaining 10 minutes is reserved only for the separately approved
  single-pair quality readiness check if the preflight cannot prove that the
  already validated Stage F assets and environment are unchanged.

The readiness check must be skipped when unchanged Stage F evidence can be
strictly bound. It must never regenerate a diffusion image. Acceptance,
hashing, and evidence packaging are not GPU work and do not extend the GPU
budget. Reaching any per-process or aggregate limit stops execution; unused
budget from one process must not silently enlarge another limit.

## Exclusive output paths

The runner-owned outputs remain the canonical paths required by the frozen run
identity:

```text
<repo>/.local_outputs/runs/<run_name>.json
<repo>/.local_outputs/runs/<run_name>.samples.jsonl
<repo>/.local_outputs/runs/<run_name>.images/
```

Before execution, all 30 targets for the ten runs must be absent. Symlinks are
not accepted. The existing runner overwrite flag remains forbidden.

All process reports, console logs, manifests, quality outputs, and acceptance
reports must be contained in one new server evidence root:

```text
/public/home/rxa-jj-ljq/TR_Transfer/
  pilot-gpu-execution_7e8ee07_<UTC-YYYYMMDDTHHMMSSZ>/
```

The resolved root must be recorded in preflight and must not exist before the
exclusive creation step. Its internal layout is fixed:

```text
preflight/
generation/<01-10>_<label>/
quality/<01-10>_<label>/
accept/
package/
```

Each quality evaluator receives its corresponding new `quality/<ordinal>_<label>`
directory and must create it exclusively. No quality result may be written into
or modify a source run directory.

Preflight must require at least 10 GiB of usable project/quota headroom. This is
a safety floor for approximately 4,000 generated PNGs, records, partial-failure
preservation, quality reports, and one evidence package; it is not permission
to consume unrelated storage.

## Preflight gate

Preflight is read-only except for exclusive creation of the evidence root and
its report. It must pass all of the following before a GPU child process may be
started:

- exact branch, full commit, and clean worktree;
- frozen plan JSON, Pilot prompt count/hash/order, and all ten canonical
  identities match the committed specification;
- no Pilot process is active and all 30 canonical output targets are absent;
- existing B200, driver, Python, PyTorch/CUDA, repository source, SD2.1 model,
  Stage F metric assets, and their frozen hashes match reviewed evidence;
- CUDA is available to the intended child process and no CPU fallback is
  possible;
- offline variables are set and the commands contain no download or install
  action;
- evidence-root path is new, non-symlinked, and on the intended persistent
  filesystem, with at least 10 GiB usable headroom;
- protected pre-Pilot files are snapshotted by path, size, and SHA-256;
- `formal_authorized=false` and no Formal path is accepted by the tool.

The preflight report must contain the ten complete commands but must not run
them without a separate explicit execution flag and user approval.

## Per-run operational acceptance

After each generation process ends, the tool must write a new process report
and check, without exposing comparative outcomes:

- return code zero, no timeout, no traceback, and only the frozen warning set;
- exact Git, environment, GPU, run name, method parameters, prompt, seed, key,
  attack, step, and range identity;
- exactly 200 strict JSONL records and exactly 400 paired RGB 512x512 PNGs;
- summary/JSONL equality, unique sample indices 0-199 and source rows
  1000-1199, complete finite required fields, and valid per-file SHA-256;
- 200 unique No-Watermark and 200 unique Watermarked file records;
- console, summary, JSONL, and PNG manifest hashes recorded after the process;
- protected pre-Pilot files unchanged.

The process report may record opaque hashes and validity booleans. It must not
print detection distances, TPR, quality metrics, candidate ranking, or a
selection-oriented summary while the grid is incomplete.

## Quality phase

Quality evaluation begins only after all ten generation runs pass generation
acceptance and the 200 No-Watermark identities and PNG hashes are identical
across all candidates.

The ten quality evaluations run in the same fixed order. Each uses the frozen
local CLIP and LPIPS/AlexNet assets, CUDA/FP32, sequential sample processing,
and a new exclusive directory. It must not regenerate or alter images.

Each quality result must bind the source summary, JSONL, all 400 PNG hashes,
full Git commit, metric specification, package versions, asset paths and
hashes, device, and exactly 200 finite per-sample observations. The shared
No-Watermark CLIP results must be identical across candidates because their
prompts and PNG bytes are identical.

No metric values are compared or selected until all ten quality evaluations
pass structural acceptance.

## Final acceptance and selection

Final acceptance proceeds in this order:

1. Recheck all generation and quality process evidence and all source hashes.
2. Recheck exact ten-candidate completeness and cross-candidate paired-input
   identity, including the deduplicated 200 No-Watermark calibration records.
3. Build one strict, immutable Pilot observation JSON without duplicate keys,
   missing values, non-finite values, or unlisted candidates.
4. Record the observation JSON SHA-256, then invoke the committed frozen
   selection implementation exactly once into a new output file.
5. Verify the selected decision against the observation hash, specification,
   threshold rule, margins, tie-breakers, bootstrap seed, and complete report
   set.
6. Package raw process reports, consoles, summaries, JSONL files, PNG hash
   manifests, quality reports, observations, selection decision, and internal
   checksums without rewriting the source evidence.

The acceptance report must continue to record `formal_authorized=false`.
Success does not authorize Formal. If either candidate family has no feasible
candidate, the frozen `NO_FEASIBLE_CANDIDATE_STOP` rule applies and the project
stops before Formal.

## Failure preservation and recovery boundary

Any occupied target, identity mismatch, asset mismatch, warning mismatch,
timeout, nonzero return code, missing or malformed record, non-finite value,
hash change, No-Watermark mismatch, quality failure, or budget exhaustion stops
the sequence before another child process starts.

All existing and partial files remain in place. The tool must not delete,
truncate, overwrite, repair, rename, or automatically retry them. It writes a
new failure report when possible and records the last completed phase.

Recovery is outside this plan. It requires a separate read-only diagnosis and
explicit approval. If a later recovery needs to move a partial canonical path,
that move must first be archived and hash-bound; it cannot be inferred from the
original GPU approval.
