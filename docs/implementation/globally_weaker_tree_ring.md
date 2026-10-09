# Globally Weaker Tree-Ring: local implementation

This note describes the local implementation following Stage C record identity.
It is implementation evidence, not an approved protocol amendment or permission
to run Development, Pilot, or Formal experiments. The historical v1 schema
document and seven-test counts in `docs/protocols/` describe earlier milestones;
the current code declares `treering-experiment-v2`.

## Method and parameters

The main SD2.1 runner supports `original_tree_ring` and
`globally_weaker_tree_ring`. Saliency and standalone No Watermark remain rejected
as runner method names. Every run still includes a paired No-Watermark reference.

`optim_utils.inject_watermark_for_method` uses the existing `inject_watermark`
function to construct `z_original`, then applies the frozen definition:

```text
z_global = z + alpha * (z_original - z)
```

Global requires an explicit `--global_alpha` with a finite numeric value in
`[0, 1]`. Booleans and non-numeric Python values are rejected by the shared
validator. Original rejects this parameter. There is no default alpha and no
selected operating point. Method validation precedes model loading, output
initialization, and any explicitly requested overwrite.

At alpha 0 the result is an independent copy of `z`; at alpha 1 it is exactly
the existing Original result. Intermediate values use float32 arithmetic for
float16/bfloat16 inputs, preserve float32/float64 arithmetic otherwise, and
return the input dtype. Global does not re-project the key after weakening.
Neither the base latent, mask, nor key is modified; injection consumes no RNG.
The Original and existing saliency injection functions retain their behavior.

## Pairing

The runner samples one base latent under `gen_seed + sample_index`, saves it
before generation, and sends separate copies to the paired generation paths.
Both method invocations use the same watermark construction, seed rule, prompt,
generation/inversion settings, detector key, and attack path. Method dispatch
does not add random draws. The shared base remains intact even if a pipeline
modifies a supplied generation tensor in place.

Separate Original/Global runs must still use identical non-method arguments,
prompt hashes, environment, and model. CPU tests check this runner path with a
synthetic pipeline; actual B200 end-to-end pairing remains a later stage gate.
Frozen clean/attack configurations and run-name guards remain in effect.

## Additive v2 records and realized budget

The schema remains `treering-experiment-v2`; the following additions are
compatible with readers that use the existing identity and score fields:

| Location | Fields |
|---|---|
| `summary.run_identity` | `method_parameters`: `{}` for Original, `{"alpha": value}` for Global |
| Each sample in JSONL and `summary.results` | The same `method_parameters`, plus `latent_residual_rms` and `latent_residual_l2` |
| `summary.parameters` | Explicit parsed `global_alpha`, null for Original |

Let `delta` be the difference between the final injected tensor and the shared
base tensor, as actually supplied before diffusion generation. Values are
detached and promoted to float64 **before subtraction**, then accumulated in
float64 over every element of the latent for that prompt (the frozen runner
configuration uses a singleton batch, one image, with C/H/W elements):

```text
L2  = sqrt(sum(delta_i ** 2))
RMS = sqrt(sum(delta_i ** 2) / N)
```

L2 is unnormalized; RMS divides by the number of latent elements, including
unmodified channels/locations. These are latent-space measures before the
generation call, not image metrics or detector distances. Half-precision
rounding is therefore included in the realized budget. Finite-precision Global
budgets may differ slightly from alpha times the Original budget. Empty,
shape/device-mismatched, non-real, or non-finite inputs/budgets are rejected.
There is no frozen target budget yet. Historical Stage C records lack these
new additive fields and remain valid historical evidence; do not rewrite them.

## Local validation and pending gates

Use the existing environment without installing or changing dependencies:

```text
python -B -m unittest discover -s tests -v
```

Tests cover alpha endpoints/intermediate values, invalid parameters, input
protection, Original regression, RNG preservation, analytic and realized
RMS/L2, and real runner record serialization with a CPU synthetic pipeline.
CPU half/bfloat16 arithmetic tests use seed injection because CPU FFT does not
support those dtypes; the frozen complex injection path is tested at
float32/float64. GPU complex-half validation is pending and must not be inferred
from these tests. No real image-generation experiment is part of local tests.

Alpha grid, pre-specified selection rule, final alpha, and achieved budget
remain Pilot/amendment decisions. Test alpha values are implementation probes,
not a proposed grid. Existing canonical names identify method/split/replicate/
attack and do not distinguish alpha candidates or smoke/full runs. The default
collision guard prevents overwriting preserved evidence. A reviewed naming
extension is needed before candidate sweeps or reuse of an occupied identity;
do not bypass this with `--overwrite_output`.

In particular, the preserved Stage C server run
`original_development_r1_clean` contains one sample and one step. It is not the
32-sample Development result and must not be overwritten. Local Global tests
do not complete the Development gate. The next authorized step is review of
the uncommitted local diff; a commit/push requires explicit user confirmation.
