# Pilot GPU Validation Tool Design

## Purpose

This is the review design for a future `scripts/pilot_gpu_validation.py` tool.
The reviewed implementation currently contains `plan`, `static-preflight`,
`live-preflight`, `inspect-generation`, `accept-generation`, `inspect-quality`,
`assemble-observations`, and `final-accept`. It has no GPU or quality execution
subcommand. The live mode only probes one existing B200, replays reviewed file
manifests, validates frozen quality assets, and creates a new preflight report.
Generation and quality inspection modes only inspect already completed
artifacts. Outcome arrays are emitted only into the complete ten-candidate
observation artifact; the assembly report binds that file and its semantic JSON
hash before frozen selection. Server synchronization and GPU use require later,
separately scoped approvals.

The tool must consume the committed Pilot specification rather than carry an
independent grid. Its plan, preflight, execution reports, acceptance, and
observation assembly must all bind commit
`7e8ee072e59ecf59449beb076716617361e8b30d` and
`treering-pilot-freeze-v1`.

## Proposed modes

### Current `static-preflight`

This deliberately weaker local/review gate checks the exact branch/commit,
clean state, frozen Pilot prompt, ten-run plan, absence of all 30 canonical
output targets, canonical unused evidence-root name, storage floor, and budget
arithmetic. It does not create the evidence root, probe CUDA, hash model or
quality assets, snapshot all protected files, or authorize execution. Its
distinct status is `PILOT_GPU_STATIC_PREFLIGHT_PASS`, and its report explicitly
records every deferred live check as false.

This mode cannot substitute for the proposed live `preflight` below.

### Current `live-preflight` and generation inspection

`live-preflight` completes the non-experimental gate: one B200/CUDA probe with
no model load, reviewed SD2.1 and protected-file manifest replay, frozen Stage F
asset validation, prompt/Git/output-path/storage checks, and exclusive evidence
root/report creation. A PASS still records `execution_authorized=false` and
cannot start a child process.

`inspect-generation` validates one already completed canonical run: exact Git,
GPU, prompt, method, parameter, seed and protocol identity; 200 strict records;
400 canonical RGB PNGs; summary/JSONL equality; finite required fields; paired
input fingerprints; and opaque evidence hashes. `accept-generation` requires
the exact ten ordered structural reports and equality of all 200 No-Watermark
PNG hashes, distance-vector digest, and paired-input fingerprints. Neither mode
prints detection or quality outcomes or invokes selection.

`inspect-quality` binds each quality result to the accepted source summary,
JSONL, and 400 image hashes; requires the frozen CUDA/FP32 CLIP and LPIPS asset
identities; validates 200 finite paired metric records; and emits only evidence
hashes. `assemble-observations` replays all raw generation and quality sources
in frozen order, requires shared No-Watermark distances, PNGs, paired inputs,
and CLIP reference values, then writes the strict observation schema and a
separate hash-binding assembly report. `final-accept` refuses an observation
whose file or semantic hash differs from that report, applies the committed
selection exactly once, and keeps `formal_authorized=false` for both selection
success and no-feasible-candidate stop.

### `plan`

Safe default. Prints the complete ten-run order, commands, budgets, expected
paths, quality commands, and acceptance requirements. It sets
`execution_started=false`. With `--output`, it may write only a new report into
an already existing review directory using exclusive creation.

It never imports the diffusion pipeline, initializes CUDA, loads model assets,
creates run output paths, or starts a child process.

### `preflight`

Performs the gate in `pilot_gpu_execution_plan_v1.md`. It requires a new
`--evidence-root`, verifies all identities and assets, snapshots protected
files, and writes one exclusive preflight report. It may query CUDA/GPU
identity but must not load SD2.1, CLIP, or LPIPS or generate/evaluate an image.

Preflight has no execution flag and cannot fall through into another mode.

### `run-one`

Reserved for a later implementation and disabled without all of:

- `--execute-pilot-gpu`;
- a user-approved preflight report;
- exact `--ordinal` and frozen `--label` agreement;
- successful acceptance of every earlier ordinal;
- a new process-report path inside the bound evidence root.

It starts exactly one frozen generation child process with the plan's timeout.
The parent persists console output and a process report on success, failure, or
timeout. It does not choose the next run automatically.

### `accept-generation`

Reads one or more completed process reports and source outputs. It performs
only operational/structural checks until all ten candidates are present. It
must support an opaque progress report that contains hashes and booleans but no
comparative performance values.

When all ten runs are present, it additionally verifies the shared 200
No-Watermark identities and PNG hashes and writes a new generation-accept
report. It never starts a child process.

### `quality-one`

Reserved for later implementation and disabled without
`--execute-pilot-quality`. It accepts only the next frozen ordinal after the
ten-run generation-accept report passes. It calls the committed offline quality
evaluator once, with frozen asset hashes and a new exclusive quality directory.
It preserves console and partial outputs on failure and never regenerates an
image.

### `accept`

Read-only with respect to generation and quality sources. It requires all ten
generation reports, the generation-accept report, ten quality reports, and the
preflight report. It replays structural acceptance, constructs one new strict
observation JSON, invokes `pilot_selection.select_pilot` once, and writes new
acceptance and decision files by exclusive creation.

The mode must not accept a Formal input path or set Formal authorization.

## Required arguments and binding

Every non-plan mode requires:

- `--expected-head` with the full 40-character commit;
- `--plan-report` and its checksum;
- `--evidence-root` resolved inside server `TR_Transfer`;
- a mode-specific new `--report` path inside that root.

Execution modes additionally require the reviewed preflight report and its
checksum. Acceptance modes require explicit repeated report paths rather than
directory globbing, so an unexpected file cannot silently enter the evidence
set.

All JSON is strict: reject duplicate keys and non-finite tokens. All new files
use exclusive creation. Parent directories must be real directories, not
symlinks. Resolved source and destination paths must remain within their
declared roots.

## State machine

```text
PLAN_ONLY
  -> PREFLIGHT_PASS
  -> GENERATION_01_PASS ... GENERATION_10_PASS
  -> GENERATION_ACCEPT_PASS
  -> QUALITY_01_PASS ... QUALITY_10_PASS
  -> PILOT_ACCEPT_AND_SELECTION_PASS
```

Any transition may instead enter a terminal preserved-failure state. There is
no automatic retry transition and no transition to Formal.

The next ordinal is derived from accepted reports, never from directory
presence alone. A later report must hash-bind all prerequisite reports so that
the chain cannot be silently substituted.

## Report separation

Each phase writes a distinct status namespace to prevent a partial result from
being mistaken for final acceptance. Proposed statuses include:

- `PILOT_GPU_PLAN_PASS`
- `PILOT_GPU_PREFLIGHT_PASS` / `PILOT_GPU_PREFLIGHT_FAIL`
- `PILOT_GENERATION_PROCESS_PASS` / `PILOT_GENERATION_PROCESS_FAIL`
- `PILOT_GENERATION_PROGRESS_PASS`
- `PILOT_GENERATION_ACCEPT_PASS` / `PILOT_GENERATION_ACCEPT_FAIL`
- `PILOT_QUALITY_PROCESS_PASS` / `PILOT_QUALITY_PROCESS_FAIL`
- `PILOT_GPU_ACCEPT_AND_SELECTION_PASS` / `PILOT_GPU_ACCEPT_FAIL`

A process PASS is not a candidate-selection result. A progress PASS is not a
generation-accept PASS. A final acceptance PASS is not Formal authorization.

## Outcome blinding before completeness

Generation progress reports expose only:

- identity and provenance;
- return code, timing, timeout, and warning classification;
- expected/observed counts;
- opaque SHA-256 values;
- validity booleans and failure text.

They do not expose individual or aggregate watermark distances, scores, TPR,
quality metrics, Pareto status, feasibility, or rankings. Quality process
reports similarly expose only completion, source binding, count, device,
asset, and output hashes until all ten quality runs pass.

Raw source files necessarily contain observations, but the orchestration tool
must not print them during the incomplete grid. Selection-oriented reading is
deferred to final `accept`.

## Process and timeout behavior

Child processes stream combined stdout/stderr to a new console file. On normal
exit, timeout, interruption, or exception, the parent records start/end time,
duration, command, return code, failure category, console path/hash, and any
source paths/hashes that exist. Timeout terminates only the recorded child
process group and does not delete partial evidence.

The tool checks both the per-run timeout and cumulative approved GPU budget
before starting a child. It refuses to borrow unused time across limits. It
also refuses to start when another Pilot process is active or the ordinal's
canonical output target is occupied.

## Protected-file model

Preflight snapshots:

- tracked repository files and Git identity;
- all pre-existing `.local_outputs` files;
- frozen prompts;
- SD2.1 model files;
- Stage F metric assets;
- prior Pilot CPU evidence named by the handoff;
- plan and expected-test package evidence.

Later reports verify that this snapshot is unchanged, excluding only the
explicitly declared new canonical Pilot outputs and files under the new
evidence root. The exclusion list is derived from the frozen ten-run plan and
cannot be widened by a command-line wildcard.

## Acceptance data flow

```text
committed Pilot specification
  + preflight identity and protected snapshot
  + 10 generation process reports and source artifacts
  + generation cross-run acceptance
  + 10 quality process reports and quality artifacts
  -> strict observation document
  -> frozen `select_pilot`
  -> decision + final acceptance + immutable evidence manifest
```

The observation document records source hashes for every derived array. Final
acceptance independently recomputes the threshold inputs, completeness,
pairing, metric aggregates, and candidate set before selection. It must not
trust a process summary merely because that summary reports PASS.

## Implementation review requirements

Before any server synchronization or GPU approval, a future implementation
must be reviewed for:

- default plan-only behavior and explicit execution flags;
- exact command identity for all ten candidates;
- path containment, symlink rejection, and exclusive creation;
- fail-closed strict JSON and finite-number handling;
- preserved timeout/interruption evidence and absence of automatic retry;
- outcome blinding during incomplete generation and quality phases;
- cross-run No-Watermark identity and PNG equality;
- complete observation assembly and deterministic frozen selection;
- protected-file invariance and `formal_authorized=false`;
- tests using synthetic CPU fixtures only, with no model load or GPU access.
