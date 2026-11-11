# Experimental Mac-native multimodal LoRA

A real local training smoke is now implemented and tested on Apple M4 (16 GiB),
MLX 0.32.4 / mlx-lm 0.32.2. It does not call CUDA, Torch, Modal or a hosted model.
This is a bounded compatibility prototype, not a general trainer or a demonstrated
fix for missing-target hallucinations.

## Implemented path

- Start from the existing **4-bit language / BF16 vision** MLX checkpoint, with its
  original weights frozen. No full-size dequantization or full-model training.
- Import `mlx_lm.tuner.lora.LoRALinear` and wrap q/v projections in the last **two**
  Qwen3 layers: rank 8, dropout 0, direct MLX scale **2.0** (equivalent to alpha/r
  with alpha 16 and rank 8; do not pass 16 as the direct scale).
- Only **163,840** FP32 LoRA parameters are trainable. Adam learning rate `1e-4`,
  batch 1. Defaults preserve the initial single-scene sequential smoke; use
  `--train-scenes` and `--shuffle` for bounded multi-scene development.
- Precompute frozen vision/projector features, using 64 visual tokens maximum.
  Cache by image hash within this fixed-budget run and explicitly stop their gradients.
- Reuse the original expanded image placeholders and assistant prompt. Verify that
  concatenated answer tokenization preserves the prompt's token prefix.
- Teacher forcing predicts **answer tokens plus EOS**, starting at the last prompt
  position. User/image/padding tokens are not supervised. There is no padded batching.
- Project vocabulary logits only at supervised positions rather than allocating
  full prompt-length vocabulary logits. Tests compare to a full masked-loss reference.
- Save only LoRA arrays; original checkpoint files are never rewritten.

## Local commands

Use the trusted baseline environment from the experiment repository's working directory.
No additional trainer installation is needed beyond that baseline's MLX dependencies.
Original weights must already be downloaded and converted; this CLI never downloads them.

```sh
BASELINE_REPO=/path/to/mlx-grounding-pi
PYTHON="$BASELINE_REPO/.venv/bin/python"

# Use a new data directory; generation refuses to overwrite.
"$PYTHON" -m rejection_lab.data --output data/mac-training-smoke --scenes 10 --seed 42

"$PYTHON" -m rejection_lab.train_mac \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-4bit" \
  --dataset data/mac-training-smoke \
  --steps 8 --max-seconds 180 --image-tokens 64 \
  --output outputs/mac-training-new

# Load the saved adapter in a NEW process and evaluate validation again.
"$PYTHON" -m rejection_lab.baseline \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-4bit" \
  --dataset data/mac-training-smoke --split validation \
  --adapter outputs/mac-training-new/adapter \
  --output outputs/mac-training-reloaded-new
```

Do not use the earlier 8-bit/default-image-budget baseline as the before/after control.
This trainer evaluates the same 4-bit model with the same image budget both before
and after optimization. Adapter evaluation defaults to its saved image budget and
rejects a conflicting override or different baseline code commit.

## Observed eight-step run

Local files: `outputs/mac-lora-smoke-001/` and `outputs/mac-lora-reloaded-001/`.
These files are ignored and are not part of the public repository.

| Measurement | Result |
|---|---:|
| Optimizer updates | 8 |
| Training questions / distinct training scenes | 4 / 1 |
| Train mean teacher-forcing loss before | 2.726448 |
| Train mean teacher-forcing loss after | 2.214356 |
| Entire smoke elapsed, including load/checks/before-and-after validation | 13.27 s |
| Peak MLX allocation | 3,697,812,976 bytes (3.70 GB; not process RSS) |
| Adapter weight file | 656,350 bytes |
| Fresh-process validation output equality after reloading | 4 / 4 exact |

Both positive and negative gradient checks were finite and nonzero. Initially some
LoRA A gradients may be zero because B starts at zero; the check correctly requires
nonzero aggregate gradients rather than every individual array being nonzero.
Tiny-model tests check all frozen arrays remain identical; the real run checks sampled
frozen vision and language arrays and an exclusive LoRA trainable-parameter list.
The original baseline checkout remains clean.

Validation uses **one separate scene with four related questions**, not four independent
images. Positive localization is 2/2 before and after, negative false positives are
2/2 before and after, with no invalid outputs. **Rejection did not improve in this run.**
A falling training loss only confirms optimization is working; it is not a generalization
or localization guarantee. Test-set model inference was not run.

## Limits and safety

- The CLI defaults to the first training scene, with optional `--train-scenes 1..16`,
  `--shuffle`, and explicit `--seed` (default 42). It permits 1..128 steps and complete
  validation splits of at most 64 questions. All questions of a selected scene stay
  together, and shuffled epochs visit every selected question once before repeating.
  This remains a bounded prototype rather than a full-dataset trainer.
- The wall budget is checked **between operations**, not a preemptive hard timeout.
- MLX's memory limit is a guideline, not OS isolation. The CLI also checks active
  allocation between operations, limits the cache and reserves weight headroom.
  Close other large applications if memory pressure occurs; no system settings change.
- Adapter/base fingerprint covers config, conversion metadata and weight index,
  **not a fresh checksum of every converted tensor file**. Use a trusted, unmodified
  local model directory; names/shapes and finite adapter values are also checked.
- Interrupted runs are not automatically resumed; inspect the report and use a new
  output directory. Adapter merge, resumed optimizer state, all-layer LoRA, larger
  datasets and cross-machine results are not validated by this smoke.
- No trained adapter is publicly uploaded. Model/data licensing still applies to
  any future artifact release.

Run local tests using the baseline environment. There are 62 tests on the prepared
Mac; the MLX training/feature cases skip on hosts without MLX/the baseline package.

Optional `--positive-weight` (default 1, range >0 and <=10) applies example-level
loss weighting; `--checkpoint-every` (default 0) saves adapter-only snapshots for
offline evaluation, not optimizer-resume checkpoints. The
[weight-3 checkpoint trial](positive-weight-development.md) found no acceptable model.
Batch-1 Adam normalization means a 3x loss multiplier is not a literal 3x update.

Optional `--distillation-strength` (default 0, finite range 0..10) adds full-vocabulary
FP32 teacher-to-student KL at temperature1. `--distillation-scope` selects
`positive-and-protocol` (default: all positive targets, negative nonpayload targets)
or `positive-only` (no negative KL/cache). The teacher is cached at zero-delta LoRA
before updates and stopped from gradient flow. Reported mean train losses remain
unweighted CE; step logs separate CE, weighted CE, KL and total objective.
[Both distillation trials](distillation-development.md) failed the unchanged joint
screen. In particular, preserving a teacher's negative suffix can preserve its errors.
This does not install a second model or save the transient teacher distributions.

A different CLI, `rejection_lab.train_existence`, trains an independent binary head
on frozen image/query prompt features, without LoRA or backbone updates. Its
[class-cost1/cost3 feasibility trials](existence-head-development.md) obtain a limited
development-screen pass, not deployment acceptance. Do not attribute its constructed
`None` answers to the original localizer or confuse its weighted-full-batch objective
with this LoRA trainer's example-wise multiplier. Its subsequent
[canonical-query follow-up](canonical-existence-development.md) reaches positive
localization19/20 and negative rejection20/20 with cost1 at64/128 on reused synthetic
development, but retains failed cost3/all-absent checkpoints and is not deployment
acceptance. New canonical heads require `--feature-query canonical`; the old default
and v1 head compatibility remain unchanged. A further
[fixed-candidate fresh same-generator evaluation](independent-synthetic-evaluation.md)
preserves98/100 positive localization and rejects100/100 negatives on50 new images;
no training occurs in that evaluation. Broader-distribution/real-photo validation
and deployment acceptance remain unclaimed.

A later [multi-scene development trial](multiscene-development.md) used 64 updates on
8 scenes and validated on 10 other scenes. It was **rejected for severe over-refusal**;
do not interpret the successful training/reload path as an accepted model improvement.
