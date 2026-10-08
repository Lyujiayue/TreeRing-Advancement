# Development and Execution Runbook v1

## 1. Purpose

This runbook defines the controlled path from local development through offline server deployment, staged experiments, validation, recovery, and preservation. It does not authorize formal execution by itself; the baseline protocol and completed amendment v1.1 are also required.

## 2. Controlled server locations

```text
project:       /public/home/rxa-jj-ljq/projects/TreeRing-Advancement
model:         /public/home/rxa-jj-ljq/models/stable-diffusion-2-1-base
venv:          /public/home/rxa-jj-ljq/venvs/treering
transfer:      /public/home/rxa-jj-ljq/TR_Transfer
prompt splits: /public/home/rxa-jj-ljq/projects/TreeRing-Advancement/.local_data/prompts/splits
outputs:       /public/home/rxa-jj-ljq/projects/TreeRing-Advancement/.local_outputs/runs
```

`/workspace` is not the declared persistent project directory. Do not treat files placed there as authoritative or durable.

## 3. Prohibited operations

Do not perform any of the following in this workflow:

```text
pip install -r requirements.txt
replace or downgrade platform PyTorch
overwrite torchvision, CUDA, the GPU driver, or triton
rebuild the environment
download SD2.1 again
delete the model
git reset --hard
git checkout --
```

Do not use `--overwrite_output` for pilot or formal runs. Do not alter frozen prompt files, model files, or completed formal artifacts.

## 4. Known non-fatal warnings

The following are known and may be ignored only when their text and context match the already observed behavior and execution otherwise succeeds:

- Legacy `.egg` pip warning.
- `timm.models.layers` `FutureWarning`.
- ComplexHalf experimental warning.
- `decode_latents` deprecation warning.

Any new warning, traceback, non-finite output, device fallback, or changed warning severity requires investigation and a recorded disposition.

## 5. Required implementation order

1. Add explicit source-row and method identity.
2. Implement Globally Weaker Tree-Ring.
3. Validate Original and Globally Weaker on development data.
4. Complete Saliency-Aware end-to-end integration.
5. Implement and freeze quality metrics.
6. Freeze pilot candidate grids.
7. Run the 200-sample pilot.
8. Complete and approve protocol amendment v1.1.
9. Run and audit one 1,000-sample formal replicate.
10. Run the remaining four formal replicates.
11. Run the frozen robustness suite.
12. Aggregate paired statistics and confidence intervals.

Do not advance past a failed stage gate.

## 6. Local development and Git procedure

From the local repository:

```bash
git branch --show-current
git rev-parse HEAD
git status --short
python -m unittest discover -s tests -v
```

Confirm the intended branch, inspect every diff, and verify that no local data, outputs, model weights, credentials, or unrelated files are staged. Commit one coherent change at a time with the exact intended paths. Do not amend, submit, or deploy a change until its tests pass. The protocol documents themselves should be reviewed and committed only after explicit approval.

After review and approval, stage only the named implementation or documentation paths, re-check the staged diff, and create a descriptive commit:

```bash
git add <EXACT_REVIEWED_PATHS>
git diff --cached --check
git diff --cached
git commit -m "<DESCRIPTIVE_MESSAGE>"
git status --short
git log -1 --format='%H %s'
```

The angle-bracket tokens above are procedural placeholders, not commands to copy unchanged. Never use a broad staging command when unrelated files are present.

Before creating a deployment bundle, record:

```bash
git status --short
git log -1 --format='%H %s'
git diff --check
python -m unittest discover -s tests -v
```

The working tree must be clean for any commit used in pilot or formal work.

## 7. Offline deployment with a Git bundle

### 7.1 Create and verify locally

Replace the branch and bundle name only with reviewed values:

```bash
git bundle create TreeRing-Advancement-stage-b-manifest-v1.bundle stage-b-manifest-v1
git bundle verify TreeRing-Advancement-stage-b-manifest-v1.bundle
sha256sum TreeRing-Advancement-stage-b-manifest-v1.bundle
```

Transfer the bundle through the approved offline channel into `/public/home/rxa-jj-ljq/TR_Transfer`. Record its SHA-256 on both sides and require an exact match.

### 7.2 Apply on the server

```bash
cd /public/home/rxa-jj-ljq/projects/TreeRing-Advancement
git status --short
git fetch /public/home/rxa-jj-ljq/TR_Transfer/TreeRing-Advancement-stage-b-manifest-v1.bundle stage-b-manifest-v1
git merge --ff-only FETCH_HEAD
git rev-parse HEAD
git status --short
```

Stop if the server worktree is dirty, the merge is not fast-forward, the commit is unexpected, or the bundle hash differs. Never force the update.

## 8. Server unit-test procedure

```bash
cd /public/home/rxa-jj-ljq/projects/TreeRing-Advancement
source /public/home/rxa-jj-ljq/venvs/treering/bin/activate
python -m unittest discover -s tests -v
```

The expected current suite contains seven tests. Record the commit, Python executable/version, PyTorch/CUDA versions, GPU, test count, result, and timestamp. A skipped or newly discovered test requires review; do not equate it with the validated seven-test result.

## 9. Naming and command templates

### 9.1 Run names

Use:

```text
<method>_<split>_r<replicate>_<condition>
```

Use stable lowercase method and condition tokens, for example `original_formal_r1_clean`. A run name must identify exactly one method, split, replicate, and condition and must be unique for all preserved outputs.

### 9.2 Common frozen arguments

The current baseline runner command uses the following frozen arguments. Do not add `--with_tracking`.

```bash
python run_tree_ring_watermark.py \
  --run_name <RUN_NAME> \
  --prompt_file <PROMPT_FILE> \
  --start 0 \
  --end <COUNT> \
  --model_id /public/home/rxa-jj-ljq/models/stable-diffusion-2-1-base \
  --image_length 512 \
  --num_images 1 \
  --guidance_scale 7.5 \
  --num_inference_steps 50 \
  --test_num_inference_steps 50 \
  --gen_seed <APPROVED_BASE_SEED> \
  --w_seed <APPROVED_WATERMARK_SEED> \
  --w_pattern rand \
  --w_radius 10 \
  --w_channel 0 \
  --w_mask_shape circle \
  --w_measurement l1_complex \
  --w_injection complex
```

The current runner creates paired No-Watermark and Original Tree-Ring observations within one invocation. After method-identity implementation, method-specific commands and any changed output contract must be frozen in amendment v1.1. The templates below are readiness templates, not authorization to run code that lacks required method and record fields.

### 9.3 Development template

```bash
cd /public/home/rxa-jj-ljq/projects/TreeRing-Advancement
source /public/home/rxa-jj-ljq/venvs/treering/bin/activate
python run_tree_ring_watermark.py \
  --run_name <method>_development_r1_clean \
  --prompt_file .local_data/prompts/splits/sdp_dev_rows_1200_1231.txt \
  --start 0 --end 32 \
  --model_id /public/home/rxa-jj-ljq/models/stable-diffusion-2-1-base \
  --image_length 512 --num_images 1 --guidance_scale 7.5 \
  --num_inference_steps 50 --test_num_inference_steps 50 \
  --gen_seed <DEVELOPMENT_BASE_SEED> --w_seed <WATERMARK_SEED> \
  --w_pattern rand --w_radius 10 --w_channel 0 \
  --w_mask_shape circle --w_measurement l1_complex --w_injection complex \
  <APPROVED_METHOD_ARGUMENTS>
```

### 9.4 Pilot template

```bash
python run_tree_ring_watermark.py \
  --run_name <method>_pilot_r1_clean \
  --prompt_file .local_data/prompts/splits/sdp_pilot_rows_1000_1199.txt \
  --start 0 --end 200 \
  --model_id /public/home/rxa-jj-ljq/models/stable-diffusion-2-1-base \
  --image_length 512 --num_images 1 --guidance_scale 7.5 \
  --num_inference_steps 50 --test_num_inference_steps 50 \
  --gen_seed <PILOT_BASE_SEED> --w_seed <WATERMARK_SEED> \
  --w_pattern rand --w_radius 10 --w_channel 0 \
  --w_mask_shape circle --w_measurement l1_complex --w_injection complex \
  <APPROVED_METHOD_ARGUMENTS>
```

### 9.5 Formal template

```bash
python run_tree_ring_watermark.py \
  --run_name <method>_formal_r<replicate>_<condition> \
  --prompt_file .local_data/prompts/splits/sdp_formal_rows_0000_0999.txt \
  --start 0 --end 1000 \
  --model_id /public/home/rxa-jj-ljq/models/stable-diffusion-2-1-base \
  --image_length 512 --num_images 1 --guidance_scale 7.5 \
  --num_inference_steps 50 --test_num_inference_steps 50 \
  --gen_seed <APPROVED_FORMAL_BASE_SEED> --w_seed <APPROVED_WATERMARK_SEED> \
  --w_pattern rand --w_radius 10 --w_channel 0 \
  --w_mask_shape circle --w_measurement l1_complex --w_injection complex \
  <APPROVED_METHOD_AND_CONDITION_ARGUMENTS>
```

Do not execute a formal template with placeholders. Amendment v1.1 must contain the complete, copyable, placeholder-free commands for every method, replicate, and condition.

Attack flags are `--r_degree 75`, `--jpeg_ratio 25`, `--crop_scale 0.75 --crop_ratio 0.75`, `--gaussian_blur_r 4`, `--gaussian_std 0.1`, and `--brightness_factor 6`. Use only the flag for the named condition unless the approved amendment defines a combination.

## 10. Pre-run checklist

Before every pilot or formal run, record affirmative evidence that:

- [ ] Git working tree is clean.
- [ ] HEAD equals the approved implementation commit.
- [ ] All required unit and stage-specific tests pass.
- [ ] Prompt file path, count, and SHA-256 are correct.
- [ ] Model path and frozen model manifest/hash are correct.
- [ ] Run name follows the convention and is unique.
- [ ] Neither expected output file already exists.
- [ ] GPU is NVIDIA B200 and CUDA is available.
- [ ] All paired methods use matched prompts, base latents, seeds, watermark key, settings, and attack randomness.
- [ ] Parameters match the frozen baseline protocol and approved amendment.
- [ ] Seed range does not overlap another formal replicate.
- [ ] Threshold and method parameters came only from the approved pilot decision.
- [ ] W&B remains disabled unless a later approved amendment explicitly changes this.

## 11. Post-run checklist

After every run, verify and record:

- [ ] Summary JSON exists.
- [ ] Per-sample JSONL exists.
- [ ] JSONL row count is correct.
- [ ] Summary `num_samples` equals the JSONL row count and expected count.
- [ ] Recorded Git commit is correct.
- [ ] Recorded `git.dirty` is `false`.
- [ ] Prompt hash is correct.
- [ ] GPU is NVIDIA B200.
- [ ] Recorded parameters are correct.
- [ ] No expected sample is silently missing or duplicated.
- [ ] All numeric outputs required for analysis are finite.
- [ ] Run and sample durations are present.
- [ ] Explicit source row, split, method, replicate, protocol, key, attack, threshold, quality, perturbation, and model fields are present as required.
- [ ] Output hashes and validation report have been captured.

## 12. Interrupted-run recovery

The JSONL file is append-and-flush, so it can show the last completed sample after interruption; the summary JSON is absent unless the run completed. The current runner has no approved resume mode.

1. Stop and preserve the console log and partial JSONL unchanged.
2. Record run name, commit, host/GPU, timestamps, last valid row, row count, output hashes, and interruption cause.
3. Quarantine the partial artifacts under a clearly labeled failure directory outside the active run-name namespace; do not treat them as complete data.
4. Diagnose and validate the correction on development data.
5. Restart the entire affected method/replicate/condition with a new unique run name and the same approved inputs. Do not use `--overwrite_output` and do not splice partial and restarted files.
6. Link the failed and replacement run identifiers in the audit log.

If exact regeneration cannot be demonstrated, stop the stage and amend the protocol before proceeding.

## 13. Formal artifact preservation

Immediately after validation, preserve as one immutable logical package:

- Summary JSON and per-sample JSONL.
- Console/stdout and scheduler logs.
- Pre-run and post-run checklists.
- Git commit and clean-status evidence.
- Protocol and amendment versions.
- Prompt, model, code/bundle, and output SHA-256 manifests.
- Environment and GPU metadata.
- Exact commands and any job scripts.
- Validation and anomaly reports.

Create at least two approved persistent copies, verify their hashes after copying, and restrict later processing to read-only inputs. Derived analysis files must retain provenance to the immutable raw package. Never rename an artifact in a way that breaks its run identity.

## 14. Stage gates

### 14.1 Development gate

- All seven current tests pass locally and on the server.
- Required record identities are implemented and tested.
- Original and Globally Weaker complete all 32 prompts with matched inputs.
- Saliency-Aware completes end-to-end and its perturbation is measured.
- Quality metrics pass controlled sanity tests.
- No unexplained missing/non-finite values or device fallbacks remain.

### 14.2 Pilot gate

- Development gate passes.
- Candidate grids and selection rules were frozen before observing pilot outcomes.
- All four methods complete 200 matched prompts under the intended pilot conditions.
- Global alpha, saliency parameters, and threshold are selected strictly by the pre-specified rules.
- Pilot evidence, sensitivity results, realized budgets, and threshold FPR/TPR are archived.
- Protocol amendment v1.1 is completed, contains placeholder-free formal commands, and is approved.

### 14.3 Formal replicate 1 gate

- Pilot gate passes and formal data remain unused for selection.
- Replicate 1 completes 1,000 matched prompts for all required methods/conditions.
- All post-run checks and cross-method pairing audits pass.
- Raw artifacts are preserved and independently hash-verified.
- Any anomaly is resolved without outcome-driven changes; otherwise stop and amend.

### 14.4 Remaining formal gate

- Formal replicate 1 is explicitly approved after audit.
- Replicates 2-5 use their pre-frozen, non-overlapping seed ranges and the identical approved protocol.
- The frozen robustness suite completes without selective omission.
- Aggregation uses the pre-specified paired statistics and confidence intervals.
