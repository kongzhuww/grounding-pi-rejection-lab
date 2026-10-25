# Missing-target rejection: experiment protocol

## Hypothesis, not a promised result

Supervised positive/negative examples may improve explicit `None` generation.
The failure reproduced in official CUDA BF16 is not a reason to declare the whole
model defective. It is a concrete starting point with measurable trade-offs.

## Separate implementation fidelity from behavior improvement

- Preserve `mlx-grounding-pi` commit recorded in `baseline.json` as the initial reference.
- Use the same prompt templates, tokenizer, image budget, decoding, quantization and
  evaluation thresholds before/after an adapter. Record any deliberate difference.
- 8-bit inference is a convenient local baseline, not automatic support for training
  its affine quantized matrices with a generic text-only LoRA trainer.
- A merged Torch adapter followed by a fresh MLX conversion is one candidate path;
  adapter-to-MLX compatibility and inference equivalence must be tested first.

## Dataset v0: smoke only

Generate original colored shapes with balanced present/absent questions, point/box
answers and explicit `None` supervision. Do not use these drawings as a claim of
broad object recognition. Group by source image before splitting; keep related crops,
near-duplicates, edited versions and all questions of a real scene in one split too.
The initial validator checks exact hashes and scene IDs, not perceptual near-duplicates.

Before meaningful training, add rights-cleared natural images and independent labels:

- Clearly present targets, small/occluded targets, and genuinely absent targets.
- Confusable categories and color/attribute combinations.
- More than one matching object (a later schema revision; current v0 has one match).
- Alternate valid phrases; do not tune against a single held-out prompt.
- Record annotator uncertainty instead of silently forcing ambiguous negatives.

Do not put private photographs or licensed benchmark archives in Git. A future data
manifest must track provenance, permitted use and scene grouping. Test is locked;
validation is for model selection. If test is used to design a fix, replace it with
a new independent final holdout and label the old set as development data.

## Training compatibility gate

No training code is shipped yet. Before building a real recipe, verify:

1. GroundingPI's original multimodal processor/template creates matching image slots.
2. Labels supervise only assistant answer tokens; image/user/padding positions use
   the loss ignore index. Confirm whether EOS is supervised and document it.
3. Vision and projector are frozen; LoRA targets actual language q/v projections.
4. Loss/gradients are finite, only intended parameters receive gradients, and visual
   features are used rather than accidentally training on text without the image.
5. A tiny training subset can be overfit; weights/adapters reload with matching output.
6. Conversion/quantization preserves the trained model sufficiently for the intended
   deployment. Keep original BF16 comparison if evaluating a quantized deployment.

A rank-8 LoRA with a frozen vision path is an initial candidate, not a validated
hyperparameter choice. Do not train only negatives: an always-`None` shortcut is a
failure. Start with balanced pipeline data, then select mixtures on validation.

## Acceptance criteria

Report counts and rates jointly: negative false positives, correct rejection,
negative failures including invalids, positive false rejection, positive localization,
and invalid output. Compare by point/box and difficulty strata on a sufficiently
large sample, with uncertainty estimates. The current scorer reports pooled rates;
stratified confidence intervals remain future work.

Require lower negative failure without unacceptable loss of positive localization.
Pre-register a non-inferiority margin on a meaningful held-out set before a real
training comparison; no margin is justified from a four-query validation smoke.
Do not select a model on FPR alone, self-reported certainty or raw token probability.

## Resource and publication gate

Data tools and local inference do not provision cloud hardware. Before paid training,
confirm provider, GPU, wall-time/resource ceilings, retry policy, uploaded data and
expected budget with the owner. Approval for the earlier official-reference run does
not authorize a new training run. No credentials are required in a dataset/config.

Publish adapters only after inspecting data rights, upstream model/Kimi K3 conditions,
license notices and training/data documentation. Do not claim the original model's
license becomes Apache-2.0 because the experiment scripts are Apache-2.0.
