# Distillation development experiments

These are local 4-bit teacher/student ablations, not official BF16 distillation,
not final acceptance, and not natural-image generalization evidence. The same
repeatedly observed synthetic development set is used; test inference is excluded.

## Trial A: positives plus negative protocol positions

Preregistered in `configs/distillation-trial.json`. Relative to positive-weight-3,
the only objective change is adding masked `KL(teacher || student)` with lambda 1,
temperature 1. Positive CE remains weighted 3, negative CE weighted 1; KL is not
multiplied by the positive weight. All positive answer/EOS positions are preserved;
negative payload positions between spatial markers are excluded from KL, but the
negative prefix, closing marker and EOS are included. CE still supervises all
answer/EOS positions in both classes.

The teacher is the same frozen original 4-bit model, cached at zero-delta LoRA
before any optimizer update. Full-vocabulary FP32 log probabilities are detached.
Vision, original weights and all previous adapters remain unchanged.

Same eight training scenes / 32 questions, 64 shuffled batch-1 Adam updates at
`1e-4`, seed42, last-two-layer q/v rank8 scale2, image budget64. Same complete ten
validation scenes / 40 questions. Initial CE, all initial baseline outputs and the
64-example sample schedule exactly match the previous positive-weight-3 trial.

| Step | Positive localized /20 | Positive false rejection /20 | Negative correct rejection /20 | Negative false positive /20 | Invalid /40 |
|---|---:|---:|---:|---:|---:|
| Untrained | 19 | 0 | 0 | 20 | 0 |
| 16 | 19 | 0 | 0 | 11 | 9 |
| 32 | 20 | 0 | 4 | 9 | 7 |
| 48 | 19 | 0 | 8 | 2 | 10 |
| 64 | 8 | 1 | 9 | 1 | 20 |

**All four checkpoints fail** the prespecified development screen: positive
localization >=19/20, negative FP <20/20, invalid=0. At 32/48 steps positive
localization is better preserved than the no-KL trial (17/13), but negative
rejection and protocol failures still prevent adoption. At 64 steps degeneration
is severe. No parser repair, constrained decoding or relabeling of invalids is used.

The invalids are real strict-GAM failures: curly quotes at 16 steps; missing closing
markers and duplicated `NoneNone` at 32; missing markers/duplicates at 48; at 64,
15 outputs contain `None` followed by an incorrect object-reference ending and
another spatial block, plus other incomplete/duplicated payloads. Detail is retained
in the local `invalid-analysis.json`.

Run A: **103.47 seconds** including load/cache/checks/before-and-after evaluation;
peak MLX allocation **3,895,811,964 bytes**, not process RSS. Teacher cache is
**195,417,600 bytes**. Additional fresh-process evaluations are outside that time.
Unweighted same-training-set mean CE: **2.683087 -> 0.666195**. Initial positive and
negative KL are both exactly zero; training KL is finite, range 0 to 0.588395.
All four checkpoints were evaluated in separate processes; all 40 final raw outputs
match reloaded checkpoint64. Base sampled frozen arrays remain unchanged.

Local artifacts: `outputs/mac-lora-distill1-positive3-001/` and
`outputs/mac-lora-distill1-positive3-step*-eval/`. No adapter is promoted.

## Teacher diagnostic and follow-up hypothesis

A read-only diagnostic on the **16 training negatives only** queried the original
zero-delta teacher after the gold `None` payload. Its next-token argmax matched
the required `<|box_end|>` in only **4/16** examples. The other 12 preferred a wrong
object-reference marker, punctuation, a coordinate or ordinary text. Gold closing
marker probability ranged from approximately 0.0000146 to 0.974.

Thus preserving the teacher on negative suffix positions is not a reliable
protocol teacher. The observed malformed rollout resembles some of its wrong
preferences. This establishes a teacher/supervision conflict, not proof that it
uniquely causes every rollout error. Teacher forcing is also not a guarantee of
correct free-running autoregressive output.

The follow-up hypothesis was **positive-only distillation**: retain positive KL,
remove all negative KL, and leave full negative CE supervision unchanged. Lambda1,
positive weight3, the same 64-step schedule and development screen were retained.

## Trial B: positive-only distillation

Preregistered before execution in `configs/distillation-positive-only-trial.json`.
The only change relative to A is disabling all negative KL. Negative labels, payload,
closing marker and EOS still receive their original CE supervision. Positive
teacher distributions and all other training settings stay unchanged.

| Step | Positive localized /20 | Positive false rejection /20 | Negative correct rejection /20 | Negative false positive /20 | Invalid /40 |
|---|---:|---:|---:|---:|---:|
| Untrained | 19 | 0 | 0 | 20 | 0 |
| 16 | 19 | 0 | 0 | 11 | 9 |
| 32 | **20** | **0** | **8** | 9 | 3 |
| 48 | 18 | 1 | 13 | 2 | 5 |
| 64 | 8 | 11 | 19 | 0 | 1 |

**All four evaluated checkpoints still fail** the unchanged screen. Step32 is a
partial development improvement: all 20 positives are localized, eight negatives
are correctly rejected, but three malformed negatives remain and nine negatives
still produce coordinates. It must not be declared accepted or generalized.

Relative to A, negative correct rejections at step32 rise 4 -> 8 and invalids fall
7 -> 3 while positive localization stays 20/20. At step48 invalids fall 10 -> 5 but
positive localization falls 19 -> 18. At step64 invalids fall 20 -> 1, yet positive
localization remains only 8/20 and false rejection increases to 11/20. Removing the
unreliable teacher terms improves protocol counts in this single-seed comparison,
but does not solve later over-refusal or establish a unique causal explanation.

Step32 malformed responses are missing box-start markers (two negatives) and
`NoneNone` (one negative). Step48 has missing markers / duplicated payloads; step64
has one absent-box `NoneNone` response. No invalid was counted as correct rejection.
No constrained decoder or output repair was used in either trial.

Run B: **96.43 seconds**, excluding the four extra fresh-process evaluations;
peak MLX allocation **3,807,731,004 bytes**, not RSS. Teacher cache is
**107,479,680 bytes**, containing positives only. Unweighted same-training-set CE:
**2.683087 -> 0.623030**. Initialization, baseline outputs and all 64 sampled IDs
match A exactly. Every negative update was checked to have KL exactly zero;
all positive target positions remain preserved. All 40 final raw outputs exactly
match independent-process checkpoint64 reload. Frozen sampled base arrays remain
unchanged. Original model and all earlier adapters were preserved.

Local artifacts: `outputs/mac-lora-distill1-positive-only-001/` and
`outputs/mac-lora-distill1-positive-only-step*-eval/`. The former contains reports,
per-checkpoint comparison, invalid diagnostics and `acceptance.json` (false).
No test inference, paid cloud call, upload or deployment occurred.

## Interpretation and next candidate

These experiments provide limited evidence for positive preservation at intermediate
steps and reveal an unreliable-teacher suffix conflict. They do not show distillation
cannot work in general: only lambda1 and the recorded masks/checkpoints were tested.
Likewise, no unobserved in-between checkpoint can be declared good or bad from this curve.

A next architectural candidate is separating target-existence classification from
coordinate generation: keep the original localizer frozen and train a small
query/image-conditioned existence component. A subsequent
[existence-head trial](existence-head-development.md) now implements this and passes
a permissive development screen with limited box-only rejection. Localization weights
are isolated, but classifier false rejection and generalization still need joint
measurement; this is not a production acceptance. Do not manufacture
confidence scores or relabel errors as `None`. A final unseen evaluation is required
after choices are fixed; the repeatedly used development set is not that evaluation.

## Reproduction

Use the same dataset/model/environment described in `mac-training.md` and
`multiscene-development.md`, and a new output directory for each run:

```sh
BASELINE_REPO=/path/to/mlx-grounding-pi
PYTHON="$BASELINE_REPO/.venv/bin/python"
"$PYTHON" -m rejection_lab.train_mac \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-4bit" \
  --dataset data/multiscene-dev-v1 --train-scenes 8 --shuffle --seed 42 \
  --steps 64 --max-seconds 240 --image-tokens 64 --positive-weight 3 \
  --distillation-strength 1 --distillation-scope positive-only \
  --checkpoint-every 16 --output outputs/distill-positive-only-repro
```

For A use `--distillation-scope positive-and-protocol` and another output directory.
Evaluate each checkpoint with `rejection_lab.baseline --split validation --adapter
<run>/checkpoints/step-0032` (and the other steps), as in the positive-weight report.
KL defaults to disabled (`--distillation-strength 0`). Checkpoints save adapter
weights, not optimizer state. Teacher cache is transient and is not saved in adapters.

At the distillation stage there were 44 passing local tests, including 22 optional
MLX cases; the current suite count is in the README.
Tests cover masked-KL agreement with an independent NumPy calculation, zero teacher
and excluded-payload gradients, detached cache stability, initial zero KL, later
positive KL, mask validation and exact no-distillation value/gradient equivalence.
Passing these implementation tests is not passing the model-quality screen.
