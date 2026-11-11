# Frozen localizer + independent existence head

**Historical task-specific-feature trial: first permissive development-screen pass,
not deployment acceptance.** A later [canonical-query follow-up](canonical-existence-development.md)
now improves point/box rejection on reused development data; that does not change the
limited results recorded below. Original
GroundingPI weights and its coordinate outputs remain unchanged. The best observed
cost-3 checkpoint preserves 19/20 positive localization and correctly rejects 5/20
negatives, but **15/20 negatives still false-positive**. Improvements are restricted
to box queries; point rejection and missing attribute combinations remain unsolved.

## Architecture and attribution

1. Run the frozen original 4-bit model on **image + input prompt only**. No answer,
   ground-truth coordinates/presence or generated tokens enter feature extraction.
2. Take the final normalized language hidden vector at the last prompt position:
   2,560 dimensions, FP32, stop-gradient, independently L2-normalized per example.
   No train/validation fitted normalization statistics or validation-label fitting.
3. Train a zero-initialized FP32 linear head, two classes `[absent, present]`, with
   **5,122 parameters**. The whole original localizer is frozen and stays in eval mode.
4. Use fixed argmax, no validation-selected threshold. Scores are uncalibrated logits,
   not confidence probabilities. An exact tie selects absent (class0).
5. A present decision passes through the exact original localizer response, including
   any invalid response. An absent decision renders a validated, canonical GAM `None`
   with `origin=existence_gate` and `finish_reason=gate_constructed`.

The original model has **not** learned to output `None` here. The semantic decision
comes from a separately supervised classifier, with an explicit output formatter.
This is not post-hoc relabeling of malformed responses: the gate does not inspect
localizer validity/text or the ground-truth label. False gate rejection of a real
target is counted as failure. All-absent and all-present policies cannot pass the
joint screen. Constructed answers are checked with the unchanged strict GAM parser.

For controlled evaluation, all 40 original localizer outputs are generated once,
then routing is replayed against them. The code does not claim operational skipping,
KV-cache reuse, or inference latency savings from this evaluation protocol.

## Fixed feasibility setup

Same original local 4-bit/BF16-vision base and image budget64 as the LoRA trials.
Same eight training scenes / 32 balanced questions, same full ten-scene / 40-question
synthetic development split (`multiscene-dev-v1`, generation seed2025). Test is not
inferred. All splits are scene/image isolated, but paired questions are not independent
images. Seed42. Adam learning rate0.05; full-batch128 updates; no early stopping.
Checkpoints at 32/64/128 are specified before execution.

The head objective is CE plus `0.5 * 0.01 * sum(weight**2)`; bias is unpenalized.
There is no LoRA or backbone gradient. 128 full-batch steps mean 4,096 label
presentations on the same 32 examples: this is **not** an optimization-budget-matched
comparison with 64 single-example LoRA steps, nor an accuracy/speed benchmark.

First trial: equal class costs, preregistered in `configs/existence-head-trial.json`.
After observing gate false rejection, a single follow-up fixes positive cost3,
negative cost1 in `configs/existence-head-cost3-trial.json`. Weighted CE is normalized
by the sum of class weights, so the balanced batch's CE mixture is 3/4 positive and
1/4 negative. Other features, data, regularizer, initialization, optimizer, checkpoint
steps and argmax rule are unchanged. No threshold or multi-parameter search was run.
This follow-up is development-guided, not a blind evaluation.

The unchanged permissive screen is: positive localization >=19/20, negative false
positives <20/20, invalid=0. It only requires **some** correct rejection while preserving
the observed positive count; it is explicitly not sufficient for production.

## Complete curves

| Setting | Step | Positive localized /20 | Positive false rejection /20 | Negative correct rejection /20 | Negative FP /20 | Invalid /40 | Dev screen |
|---|---:|---:|---:|---:|---:|---:|---|
| Original, no gate | — | 19 | 0 | 0 | 20 | 0 | Fail |
| Equal cost | 32 | 16 | 3 | 10 | 10 | 0 | Fail |
| Equal cost | 64 | 13 | 6 | 17 | 3 | 0 | Fail |
| Equal cost | 128 | 14 | 5 | 16 | 4 | 0 | Fail |
| Positive cost3 | **32** | **19** | **0** | **5** | **15** | **0** | **Pass** |
| Positive cost3 | 64 | 19 | 0 | 2 | 18 | 0 | Pass |
| Positive cost3 | 128 | 19 | 0 | 3 | 17 | 0 | Pass |

The equal-cost classifier rejects real objects despite unchanged localization
weights. Preserving the localizer is not a guarantee of preserving joint accuracy.
Cost3 trades negative coverage for positive recall on this development set. Step32
has the most correct rejections among the three cost3 checkpoints; highlighting it
is development-set selection, not evidence on an independent test.

### Important subgroup failures at cost3 / step32

| Task | Positive localized /10 | Correct negative rejection /10 | Negative FP /10 |
|---|---:|---:|---:|
| Box | 9 | 5 | 5 |
| Point | 10 | **0** | **10** |

By absent type (both tasks): color3/10, shape2/6, attribute-combination**0/4** correct
rejections. All five successful rejections are box questions, with three absent
colors and two absent shapes. None of the point negatives is solved. The feature
uses the original task-specific query; this exposes possible instruction/task bias.

A post-hoc descriptive diagnostic finds 26 overlapping positive/negative image pairs
with exactly the same query string, of which seven pairs have both gate decisions
correct. This suggests some image sensitivity but is not an independent benchmark,
a proof of robust attribute reasoning, or an extra model-selection criterion.
The local `matched-query-diagnostic.json` records the pairs; repeated images/queries
must not be treated as 26 independent test samples.

## Execution, isolation and reloading

| Measurement | Equal cost | Cost3 |
|---|---:|---:|
| Full run, load/features/baseline/training/evaluation included | 50.06 s | 49.87 s |
| Peak MLX allocation (not RSS) | 3,702,152,780 bytes | 3,702,152,780 bytes |
| Train objective before -> after | 0.693147 -> 0.579863 | 0.693147 -> 0.488452 |
| Final train TP / TN / FP / FN | 16 / 13 / 3 / 0 | 16 / 2 / 14 / 0 |
| Saved head weight file | 20,662 bytes | 20,662 bytes |

The two after-training objectives use different class costs and are not directly
comparable quality metrics. Full training/validation feature tensors are **elementwise
identical** across runs, and all 40 baseline raw outputs/finish reasons match each
other and the preceding original-model baseline. Head states have no reference to
backbone parameters; only the head is passed to the optimizer. Tiny tests check all
frozen arrays; real runs check an empty backbone trainable tree and sampled frozen
vision/language arrays.

**All six checkpoints were loaded in separate fresh processes**, with features
recomputed from images/prompts and original baseline outputs regenerated, not just
loaded from cache. Every gate decision, baseline response and full routed output
matches the training-process evaluation for all 40 questions at each checkpoint.
Those six additional evaluations are outside the roughly50-second run timings.

Saved-head loading validates feature version, two-class ordering, input dimension,
base metadata fingerprint, baseline commit, image-token budget, parameter names,
shapes, dtype and finite weights. The fingerprint covers model metadata/index, not
a fresh hash of all converted tensors; the base directory must be trusted/unmodified.
Head checkpoints do not include resumable optimizer state. Output directories cannot
be overwritten. The wall/memory limits are soft checks between operations, not OS
isolation or hard preemptive limits.

Both `acceptance.json` files have `deployment_accepted=false`. Cost3 separately records
its three development-screen passes. No original baseline edits, previous adapter
replacement, test inference, cloud task, model upload or deployment occurred.

## Reproduce from a trusted source checkout on Mac

```sh
BASELINE_REPO=/path/to/mlx-grounding-pi
PYTHON="$BASELINE_REPO/.venv/bin/python"
# Same seed2025/100-scene data preparation as multiscene-development.md.
"$PYTHON" -m rejection_lab.train_existence \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-4bit" \
  --dataset data/multiscene-dev-v1 --train-scenes 8 --steps 128 \
  --seed 42 --image-tokens 64 --learning-rate 0.05 --regularization 0.01 \
  --positive-weight 3 --max-seconds 240 --output outputs/existence-cost3-repro
# For equal cost use --positive-weight 1 and a different output directory.
"$PYTHON" -m rejection_lab.train_existence \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-4bit" \
  --dataset data/multiscene-dev-v1 --image-tokens 64 --max-seconds 240 \
  --head outputs/existence-cost3-repro/checkpoints/step-0032 \
  --output outputs/existence-cost3-reloaded
```

The CLI is bounded to <=16 training scenes, <=64 validation questions, and the pinned
clean baseline commit. Defaults are cost1 and 128 full-batch updates. It is a source
checkout research CLI, not a released production inference wrapper. Current local
artifacts are under `outputs/mac-existence-linear-001/` and
`outputs/mac-existence-linear-cost3-001/`, with separate reload directories.

There are **59 passing local tests**, including27 optional MLX cases. New tests cover
image/query feature influence, stop-gradient/full frozen-array isolation, a toy
linearly separable probe, weighted CE, safe save/reload, metadata/finite-value checks,
explicit routing attribution, no ground-truth use, no invalid-output repair and
all-absent gate failure. Implementation tests do not establish model generalization.

## Follow-up status and remaining questions

- Task-independent features were subsequently implemented with a separate feature
  version, keeping the old default/load compatibility. See the
  [canonical-query cost3/cost1 curves and independent reloads](canonical-existence-development.md).
  Cost1/steps64 and128 reject20/20 negatives while retaining19/20 positive localization
  on this reused split; cost3 and an early all-absent checkpoint remain failures.
- Add more independent training scenes and task/absence-type coverage. Eight scenes
  are far too small to establish robust image/query presence classification.
- After fixing a design, evaluate unseen data and independently labeled real photos.
  Do not tune repeatedly on the current 40 development questions or consume the
  locked test while choosing changes. No final-test or natural-image claim is made.
