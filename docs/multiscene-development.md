# Multi-scene development trial: rejected for over-refusal

**Acceptance: failed. Do not promote or deploy this adapter.**

The run executed successfully, but lowering false positives came with a catastrophic
loss of positive localization. Execution completion and training-loss reduction are
not model-quality acceptance.

## Fixed setup before execution

- Same trusted baseline commit recorded in `baseline.json`; start from the original
  **4-bit** language / BF16 vision model, not the earlier eight-step adapter.
- New synthetic dataset: `data/multiscene-dev-v1`, generator seed **2025**, 100 scenes.
  Exact image hashes do not overlap the earlier 10-scene smoke dataset.
- Select the first **8 training scenes / 32 questions**, 16 positive and 16 negative.
  The generated train split has 80 scenes, but only these eight were actually used.
- Use the **complete validation split: 10 different scenes / 40 questions**, 20
  positive and 20 negative. No question filtering based on model output.
- Test split: no model inference; not used for training or selection.
- Fixed seed 42; shuffle every epoch; **64 updates**, each training question visited
  exactly twice. Last two layers, q/v rank-8 LoRA, scale 2, dropout 0, Adam `1e-4`.
- Batch 1, maximum 64 image tokens, same budget/decoding before and after training.
- 240-second soft wall budget checked between operations; MLX memory guideline and
  active-allocation checks remain, not OS hard limits.

Training source hashes, selected scene/question IDs, dataset manifest, per-step loss
and gradient norms are recorded in the local report. No learning-rate/step sweep or
post-hoc cherry-picking was performed.

## Observed results

| Development-set measure | Before | After 64 updates |
|---|---:|---:|
| Absent questions with false-positive coordinates | 20 / 20 | 0 / 20 |
| Absent questions correctly answered `None` | 0 / 20 | 20 / 20 |
| Present questions with successful localization | 19 / 20 | **3 / 20** |
| Present questions incorrectly answered `None` | 0 / 20 | **17 / 20** |
| Invalid responses | 0 / 40 | 0 / 40 |

| Present-object breakdown | Before | After |
|---|---:|---:|
| Point localization passing 15-pixel threshold | 10 / 10 | 3 / 10 |
| Boxes passing 0.70 IoU threshold | 9 / 10 | 0 / 10 |

After training **37/40 answers are `None`**, including all negatives and most positives.
This is near-collapse toward a rejection shortcut, not reliable presence reasoning.
Balanced training-example counts alone did not prevent that shortcut.

Other measurements:

- Mean loss on the same 32 training questions: **2.683087 → 0.653915**.
- Full run, including load/checks/before-and-after validation: **91.69 seconds**.
- Peak MLX allocation: **3,700,120,148 bytes** (3.70 GB), not process RSS.
- Trainable parameters: **163,840**; adapter weights: **656,350 bytes**.
- Positive and negative gradient checks finite/nonzero; sampled frozen arrays unchanged.
- In a fresh process, loading the saved adapter reproduces **all 40** validation
  raw outputs exactly. The bad behavior is not an adapter-reload discrepancy.
- Original baseline checkout remains clean; previous adapter was not overwritten.

Counts are question-level. Each scene contributes related point/box questions, so
40 questions are not 40 independent images. These drawings and ten validation scenes
are a development diagnostic, not evidence of natural-image performance. Reusing this
set to design the next change makes it development data, not a blind final benchmark.

## Interpretation and next controlled changes

The **observed failure mode** is over-refusal: the model can produce valid `None`
answers but does so for present objects too. This does not identify one proven internal
cause. A small, narrow dataset, repeated shared `None` supervision, update strength and
limited adapter locations are hypotheses to test, not established explanations.

Do not blindly add steps or select by FPR/loss alone. Next candidates, one controlled
change at a time, are:

1. Preserve positive behavior with stronger positive replay/loss weighting, or a
   separately validated positive-only teacher-preservation objective.
2. Save intermediate checkpoints and compare the **joint** rejection/localization
   curve on this fixed development set; select no checkpoint on FPR alone.
3. Broaden scene/category/attribute diversity and add rights-cleared, independently
   annotated real images before making broader quality claims.
4. Pre-register positive non-inferiority and negative-improvement criteria for a
   sufficiently large new independent final holdout. Do not label a tuned development
   score as final test performance.

A follow-up [positive-weight/checkpoint trial](positive-weight-development.md) now tests
example weighting and intermediate snapshots; no checkpoint passes its development
screen. Subsequent [teacher-preservation trials](distillation-development.md) also
find no acceptable checkpoint, despite some intermediate positive preservation.
Broader-data corrections remain untested. No cloud hardware or paid training was used.

## Reproduce in the existing Mac environment

Run from this experiment repository, use new output directories, and keep adapters local:

```sh
BASELINE_REPO=/path/to/mlx-grounding-pi
PYTHON="$BASELINE_REPO/.venv/bin/python"
"$PYTHON" -m rejection_lab.data --output data/multiscene-repro --scenes 100 --seed 2025
"$PYTHON" -m rejection_lab.train_mac \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-4bit" \
  --dataset data/multiscene-repro --train-scenes 8 --shuffle --seed 42 \
  --steps 64 --max-seconds 240 --image-tokens 64 \
  --output outputs/multiscene-repro
"$PYTHON" -m rejection_lab.baseline \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-4bit" \
  --dataset data/multiscene-repro --split validation \
  --adapter outputs/multiscene-repro/adapter --output outputs/multiscene-reloaded-repro
```

Original local artifacts: `outputs/mac-lora-multiscene-001/report.json`,
`validation-{before,after}.jsonl`, `adapter/`, `reload-verification.json`, and
`outputs/mac-lora-multiscene-reloaded-001/`. They remain ignored and unuploaded.
