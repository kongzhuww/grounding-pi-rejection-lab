# Positive-weight ablation: no acceptable checkpoint

**Development screen: all checkpoints failed. No adapter is promoted.**

This trial changes the positive-example loss multiplier from 1 to **3**, keeping
negative weight 1. It also saves adapter-only checkpoints every 16 steps for offline
evaluation. Saving checkpoints does not change optimizer updates or sample order.

## Fixed comparison

Same original 4-bit base, last-two-layer q/v rank-8 LoRA, scale 2, dropout 0, Adam
`1e-4`, batch 1, seed 42, image budget 64. Same dataset seed 2025 and the same eight
training scenes / 32 balanced questions; 64 shuffled updates, each example twice.
Same complete development split: ten other scenes / 40 questions. Test is unused.

All 40 untrained baseline raw outputs and all 64 sampled example IDs exactly match
the preceding unweighted trial. The initialization's unweighted training loss is
also identical. This is not a comparison against the earlier 8-bit baseline.

The development screen was specified before the run:

- Positive localization at least **19/20**, preserving the observed baseline count.
- Negative false-positive count less than **20/20**.
- **Zero invalid outputs**, preventing malformed responses from looking like rejection.

This small, repeatedly used development set is not a final acceptance benchmark.

## Full checkpoint curve

| Checkpoint | Positive localized /20 | Positive wrongly `None` /20 | Negative correctly `None` /20 | Negative false positives /20 | Invalid /40 |
|---|---:|---:|---:|---:|---:|
| Untrained | 19 | 0 | 0 | 20 | 0 |
| 16 steps | 19 | 0 | 0 | 11 | 9 |
| 32 steps | 17 | 2 | 7 | 9 | 4 |
| 48 steps | 13 | 6 | 13 | 1 | 6 |
| 64 steps | 5 | 13 | 19 | 0 | 3 |

At 16 steps, the nine apparent reductions in false positives are **all malformed
negative responses**, not correct rejection. Negative failure including invalids is
still 20/20. At 32 steps negative failure is 13/20, at 48 it is 7/20, and at 64 it
is 1/20, but positive localization degrades throughout.

Invalid output was reparsed, not relabeled as success:

- Step 16: nine absent point answers contain a curly quote `”` instead of a valid
  payload inside the spatial block.
- Step 32: missing box-start markers or duplicated `NoneNone` payloads.
- Step 48: the same missing-marker / duplicated-payload failure families.
- Step 64: two present boxes contain coordinate tokens followed by `NoneNone`;
  one absent box contains `NoneNone`.

These are actual protocol failures, not merely a strict label-normalization mismatch.
No parser rule or output repair was added to make them pass.

## Final comparison with the unweighted trial

| At 64 steps | Weight 1 | Positive weight 3 |
|---|---:|---:|
| Positive localized | 3/20 | 5/20 |
| Positive false rejection | 17/20 | 13/20 |
| Negative correct rejection | 20/20 | 19/20 |
| Invalid outputs | 0/40 | 3/40 |

Positive weighting slightly reduces final over-refusal in these counts, but does not
preserve localization and introduces protocol instability. It is not an acceptable
joint improvement. No looser threshold was selected after seeing the results.

## Execution and reload

- Full training run: **92.75 seconds**, including load/checks/before-and-after evaluation.
  The four offline checkpoint evaluations are additional and are not included in that time.
- Peak MLX allocation: **3,700,120,636 bytes** (3.70 GB; not process RSS).
- Unweighted mean train loss on the same 32 questions: **2.683087 → 0.636311**.
- Four checkpoints: 16/32/48/64, plus the final adapter; only LoRA weights are saved.
- Each checkpoint was evaluated in its own fresh process. The 64-step checkpoint's
  40 raw outputs exactly match the training process's final outputs.
- Original baseline and previous adapters remain unchanged. No cloud call, model
  upload or final-test inference was made.

## Implementation details and limits

`weighted_answer_loss` multiplies each example's answer/EOS-only loss. Unweighted
mean losses remain separately reported for comparison; per-step logs include both
weighted and unweighted values. With batch-1 Adam, multiplying a gradient by three
is **not** a guarantee of tripling that example's parameter-update influence, because
Adam normalizes using its running moments. No gradient-accumulation change was mixed
into this ablation.

New flags default to the old behavior: `--positive-weight 1`, `--checkpoint-every 0`.
Weight must be finite/positive and <=10; checkpoints contain adapters, not resumable
optimizer state. Tests verify loss/gradient scaling and default-loss equivalence.

This trial suggests simple example weighting is insufficient here. Exact causes
are not proven. Next hypotheses are teacher-preservation for known-present examples
and protocol positions, or separating existence decisions from coordinate generation.
Subsequent [distillation ablations](distillation-development.md) now test positive/
protocol teacher-preservation and positive-only KL; no evaluated checkpoint passes
the joint screen. A later [independent existence-head trial](existence-head-development.md)
now tests the separation, with a limited development-screen pass rather than final
acceptance.
Do not blindly add training steps or silently constrain/repair decoding and call that
a fine-tuning gain.

## Reproduce locally

```sh
BASELINE_REPO=/path/to/mlx-grounding-pi
PYTHON="$BASELINE_REPO/.venv/bin/python"
# Prepare a new data directory using the data tool if needed (100 scenes, seed 2025).
"$PYTHON" -m rejection_lab.train_mac \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-4bit" \
  --dataset data/multiscene-dev-v1 --train-scenes 8 --shuffle --seed 42 \
  --steps 64 --max-seconds 240 --image-tokens 64 \
  --positive-weight 3 --checkpoint-every 16 --output outputs/positive3-repro
# Example offline checkpoint evaluation; repeat for 0016, 0032, 0048, 0064.
"$PYTHON" -m rejection_lab.baseline \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-4bit" \
  --dataset data/multiscene-dev-v1 --split validation \
  --adapter outputs/positive3-repro/checkpoints/step-0032 \
  --output outputs/positive3-repro-step0032-eval
```

Original local run: `outputs/mac-lora-positive3-001/`, with its checkpoint comparison
and per-checkpoint evaluation directories. Data, raw reports and adapters remain ignored
and unuploaded. This phase has 32 passing local tests (ten optional MLX training cases).
