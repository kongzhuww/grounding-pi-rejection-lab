# Task-independent existence queries — development improvement only

**2026-10-28: canonical-query cost1 checkpoints64/128 preserve 19/20 positive
localizations, correctly reject 20/20 negatives, and have zero false rejection and
invalid outputs on the reused synthetic development split. This is not final or
production acceptance. All artifacts stay local.** A subsequent
[fixed-candidate fresh same-generator holdout](independent-synthetic-evaluation.md)
now checks50 new images without training or selecting another checkpoint; that
stronger within-generator evidence is separate from the development curve below.

This follows the [task-specific existence-head trial](existence-head-development.md),
whose positive-cost3 candidate rejected only five box negatives and zero point
negatives. The original localizer is unchanged in both experiments.

## Hypothesis and controlled changes

The old feature used the original localization instruction. Therefore the same
image/phrase could have different features and gate decisions for point vs box.
The new existence-only feature uses:

```text
Is the following target present in the image? {phrase}. Answer yes or no.
```

`phrase` is the target description already supplied in the original user query,
not an answer or presence label. No gold coordinates, output text, generated tokens
or row IDs enter the feature. Row IDs appear only in audit files. The original
localization prompt remains unchanged when generating baseline coordinates.

The backbone still provides a 2,560-dimensional last-prompt hidden state, detached,
cast to FP32 and per-example L2 normalized. The same FP32 linear head has 5,122
parameters, zero initialization, fixed argmax and uncalibrated logits. Absent
answers are explicitly constructed by the external gate, not newly learned
GroundingPI outputs. Present decisions retain original outputs, including invalids.

Canonical point/box features are identical **by construction**, through the same
image and query, not a newly learned invariance. Caching duplicate features only
avoids redundant computation; both paired labels remain in the 32-question batch.

### Fixed common setup

- Original trusted 4-bit/BF16-vision base, image budget64, seed42.
- Same eight training scenes /32 questions, positive16/negative16, from
  `data/multiscene-dev-v1` (generation seed2025).
- Same complete ten-scene /40-question development split, positive20/negative20.
  Scene and image hashes are disjoint from training. No test inference.
- Adam learning rate0.05, L2 coefficient0.01, bias unpenalized, 128 full-batch updates
  and checkpoints32/64/128. Same 4,096 label presentations as the old head trials.
- Unchanged permissive screen: positive localization>=19/20, negative FP<20/20,
  invalid=0. This is not a statistical non-inferiority or deployment test.
- No threshold fitting, parser changes, constrained decoding, cloud calls or upload.

Trial A changes **only the feature query** relative to old positive-cost3. After
observing all-present behavior, trial B changes **only positive cost3 to cost1**
with the canonical query retained. Both configurations were recorded before their
respective execution. B is development-guided, not blind testing:

- `configs/existence-head-canonical-cost3-trial.json`
- `configs/existence-head-canonical-cost1-trial.json`

## Full development curves

Counts are question-level. The forty questions cover ten images and twenty
image/phrase units; paired point/box questions are not independent observations.

| Feature / cost | Step | Positive localized /20 | Positive falsely rejected /20 | Negative correctly rejected /20 | Negative FP /20 | Invalid /40 | Screen |
|---|---:|---:|---:|---:|---:|---:|---|
| No gate | — | 19 | 0 | 0 | 20 | 0 | Fail |
| Old task-specific / cost3 | 32 | 19 | 0 | 5 | 15 | 0 | Pass |
| Canonical / cost3 | 32 | 19 | 0 | 0 | 20 | 0 | Fail |
| Canonical / cost3 | 64 | 19 | 0 | 0 | 20 | 0 | Fail |
| Canonical / cost3 | 128 | 19 | 0 | 0 | 20 | 0 | Fail |
| Canonical / cost1 | 32 | **0** | **20** | 20 | 0 | 0 | Fail |
| Canonical / cost1 | **64** | **19** | **0** | **20** | **0** | **0** | **Pass** |
| Canonical / cost1 | 128 | 19 | 0 | 20 | 0 | 0 | Pass |

Canonical cost3 predicts present for all development examples. Canonical cost1 at
step32 predicts absent for all examples, including every true target: a retained
failure, not a good rejection checkpoint. At steps64 and128 every binary existence
decision is correct on this development split; one pre-existing positive box
localization still fails. The gate does not repair coordinates.

Step64 is the earliest passing of three prespecified checkpoints. Highlighting it
is development-set selection; no claim is made about unobserved intermediate steps
or optimal hyperparameters.

### Subgroups at cost1 / step64 (same counts at step128)

| Task | Positive localized /10 | Correct negative rejection /10 | Negative FP /10 |
|---|---:|---:|---:|
| Point | 10 | 10 | 0 |
| Box | 9 | 10 | 0 |

By absence type: missing color10/10, shape6/6, attribute combination4/4 correctly
rejected. These very small, paired subgroups do not establish robust attribute
binding or natural-image performance.

## Execution, input isolation and independent reload

| Measurement | Canonical cost3 | Canonical cost1 |
|---|---:|---:|
| Full run seconds | 39.64 | 39.91 |
| Peak MLX allocation, not process RSS | 3,702,521,444 bytes | 3,702,521,444 bytes |
| Training objective before -> after | 0.693147 -> 0.519065 | 0.693147 -> 0.621041 |
| Final train TP / TN / FP / FN | 16 / 0 / 16 / 0 | 16 / 16 / 0 / 0 |

Objectives have different class costs, so their numerical sizes are not comparable
quality scores. Timings include load/features/baseline/training/evaluation, exclude
six additional canonical reload runs and the legacy compatibility run, and are not
a statistical speed benchmark. All original localization outputs are generated
before routing; no skipped-inference latency saving is claimed.

Checks actually completed:

1. Training and development feature tensors are elementwise identical between
   canonical cost1 and cost3. All forty baseline raw outputs/finish reasons are
   byte-identical across these runs and the old task-specific cost3 run.
2. Sixteen training and twenty development point/box feature pairs have maximum
   absolute difference zero. All twenty development pairs have identical gate
   decisions at each checkpoint.
3. All six canonical checkpoints were loaded in separate fresh processes, with
   image/query features recomputed and baseline generation rerun. Every raw routed
   output, gate decision/logit row, baseline output and metric matches its training
   process evaluation exactly. See each run's `reload-verification.json`.
4. A further fresh-process load of the **old v1 cost3/step32 head**, with the unchanged
   CLI default, reproduces all original baseline, gate/logit and routed rows.
5. The 72 feature-input audit rows per canonical run match the fixed template and
   original target phrase. Input phrases agree with the original query strings.
   Training/development scene IDs and image hashes are disjoint. Audit IDs are not
   model input; labels are used for head training and evaluation only.
6. Among twenty unique development image/phrase units there are thirteen overlapping
   opposite-label pairs with the same target phrase. Thus a target string can appear
   under both labels. This is descriptive, not thirteen extra independent tests,
   a new selection criterion or proof of robust visual reasoning.
7. The backbone has an empty trainable tree; sampled frozen arrays remain unchanged.
   As before, this is not a new hash of all model tensors. Loaders validate model
   metadata fingerprint, baseline commit, feature version, dimensions, budget,
   class order, parameter shape/dtype and finite values.

Both `acceptance.json` files retain `deployment_accepted=false`. Cost1 separately
records development-screen passes at64/128. The old outputs and baseline checkout
were not overwritten. No dataset/head is uploaded or published.

## Compatibility and reproduction

The CLI default remains `--feature-query task-specific` with legacy feature version
`frozen-final-prompt-hidden-l2-v1`. Canonical mode uses a distinct version,
`frozen-final-canonical-existence-hidden-l2-v2`; cross-version loads are rejected.
A canonical head must be evaluated with `--feature-query canonical` explicitly.

Use new output directories, the existing trusted Mac environment and existing data:

```sh
BASELINE_REPO=/path/to/mlx-grounding-pi
PYTHON="$BASELINE_REPO/.venv/bin/python"
"$PYTHON" -m rejection_lab.train_existence \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-4bit" \
  --dataset data/multiscene-dev-v1 --train-scenes 8 --steps 128 \
  --seed 42 --image-tokens 64 --learning-rate 0.05 --regularization 0.01 \
  --positive-weight 1 --feature-query canonical --max-seconds 240 \
  --output outputs/canonical-cost1-repro
"$PYTHON" -m rejection_lab.train_existence \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-4bit" \
  --dataset data/multiscene-dev-v1 --image-tokens 64 \
  --feature-query canonical --max-seconds 240 \
  --head outputs/canonical-cost1-repro/checkpoints/step-0064 \
  --output outputs/canonical-cost1-reloaded-repro
```

For trial A change `--positive-weight 3` and use another output directory. Outputs
and checkpoints are local under `outputs/mac-existence-canonical-cost{1,3}-001/`.
The old compatibility evaluation is `outputs/mac-existence-legacy-compat-001/`.
At this training stage, the local suite had **61 passing tests**, including canonical-query validation,
both feature-version roundtrips and rejection of incompatible/unknown versions;
Ruff check and format pass. These implementation tests are not generalization tests.

## Follow-up status and remaining questions

- A fixed64-step candidate was subsequently evaluated on a predeclared fresh
  same-generator holdout; see the [separate results and limits](independent-synthetic-evaluation.md).
  It preserves98/100 positive localization and rejects100/100 negatives without
  further training. Do not treat either finite-set score as a population guarantee.
- Expand scene/object/attribute coverage and add rights-cleared, independently
  labeled real photographs. Point/box pairing consistency alone proves no broad
  existence reasoning ability.
- Check sensitivity to prompts, image budgets and new distributions without
  silently reusing a head trained under another feature version.
- Define application-specific error budgets and deployment checks. This experiment
  does not authorize deployment, additional paid cloud work or final-test selection.
