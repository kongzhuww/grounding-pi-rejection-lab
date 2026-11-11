# Fixed-candidate fresh synthetic evaluation

**2026-10-28: a fixed64-step canonical-query head passes the preregistered research
screen on50 fresh same-generator images /200 paired questions. It preserves the
original localizer's98/100 positive localizations and correctly rejects100/100
negative questions, with no false rejection or invalid output. Deployment remains
unaccepted. This is not natural-image or out-of-distribution validation.**

## What was fixed before seeing new results

Candidate: `outputs/mac-existence-canonical-cost1-001/checkpoints/step-0064`, selected
in the preceding [canonical development experiment](canonical-existence-development.md),
not selected using this new holdout. The128-step head was not tried on it.

`configs/canonical-independent-v1.json` was written **before generating new data**.
It fixes the candidate weights/config SHA256, feature/gate/metric/generator source
hashes, five generator seeds, sample count, comparisons, thresholds and recheck.
A byte copy and its digest are in `outputs/canonical-independent-v1/plan.json` and
`preparation.json`.

- Seeds:2026102801,2026102802,2026102803,2026102804,2026102805. No reseeding or
  result-dependent question filtering.
- Each shard generates100 scenes with the unchanged generator; only its new
  validation partition is evaluated:10 images /40 questions. The generated
  train/test partitions are unused. Total:50 evaluated images,100 positive and100
  negative questions. No additional head or backbone training.
- Same64-image-token budget, canonical feature versionv2, equal-cost saved head,
  fixed argmax, original localization prompts and greedy decoding. No threshold,
  query, checkpoint or parser changes after seeing scores.
- The runner adds only an explicit `fresh-synthetic-holdout` provenance role and
  rejects that role without `--head`. Features, gate and metric source hashes match
  the candidate's training metadata. This is a labeling/safety change, not a new
  inference or training algorithm.

### Predeclared research screen

1. No positive localization success lost by question ID.
2. Zero positive false rejection and zero invalid outputs.
3. At least90/100 negatives correctly rejected.
4. All three negative kinds represented, each with correct rejection rate>=80%.

These are a research screen, not formal statistical non-inferiority, a product
error budget or authorization to deploy. All checks pass; this does not remove the
sample-size and distribution limitations below.

## Isolation and original locked test

All500 generated image hashes were checked against all100 original
`multiscene-dev-v1` images (every split) and between new shards: no overlap.
The evaluated50 are therefore new image draws, not copied or renamed old examples.

Generator-local scene/question IDs repeat across seeds. Combined files explicitly
namespace them by shard; shared local IDs are not mistaken for shared images or
silently dropped. Source dataset/ID fields preserve traceability.

Original train/validation/test JSONL hashes were recorded before preparation and
checked after evaluation. They are unchanged. Neither the original locked test
nor any newly generated train/test partition was passed to model inference. The
new validation-named partitions serve as this separate, fixed-candidate holdout;
calling them holdout does not imply they came from the old locked test.

## Results on the same new inputs

| Question-level measure | Frozen localizer alone | Fixed external gate |
|---|---:|---:|
| Positive localization passing threshold | 98/100 | 98/100 |
| Positive falsely rejected | 0/100 | 0/100 |
| Negative correctly rejected | 0/100 | 100/100 |
| Negative false positive | 100/100 | 0/100 |
| Invalid responses | 0/200 | 0/200 |

All positive raw outputs and finish reasons remain unchanged. The two original
localization failures are boxes; the gate neither repairs nor suppresses them.
Point threshold remains15 pixels and minimum box IoU0.70, not a newly relaxed rule.

### By task

| Task | Positive localization | Correct negative rejection | Negative FP |
|---|---:|---:|---:|
| Point | 50/50 | 50/50 | 0/50 |
| Box | 48/50 | 50/50 | 0/50 |

### By absence kind

| Kind | Correct negative rejection, questions | Distinct negative image/phrase units |
|---|---:|---:|
| Missing color | 60/60 | 30 |
| Missing shape | 36/36 | 18 |
| Missing attribute combination | 4/4 | **2** |

Attribute-combination coverage remains particularly thin. Four successful paired
questions do not establish robust attribute binding. There are only two such
scenes, despite the larger overall evaluation.

### All five shards, not a selected best subset

| Shard | Positive localization | Correct negative rejection | False rejection / invalid |
|---|---:|---:|---:|
| 01 | 20/20 | 20/20 | 0 / 0 |
| 02 | 19/20 | 20/20 | 0 / 0 |
| 03 | 20/20 | 20/20 | 0 / 0 |
| 04 | 20/20 | 20/20 | 0 / 0 |
| 05 | 19/20 | 20/20 | 0 / 0 |

## Sample units and uncertainty

The200 questions are not200 independent images. Each of50 images contributes a
present and an absent target, each repeated as point and box queries. There are100
image/phrase units, with present/absent units correlated within image.

All100 point/box pairs have identical gate decisions. Collapsing these duplicates
leaves50 positive decisions and50 negative decisions. For each rate separately,
50/50 observed successes give a descriptive Wilson95 interval of approximately
**92.87%–100%**, assuming independent same-generator image draws. This is not a
population guarantee, a joint confidence statement for all metrics, or a confidence
interval on200 supposedly independent answers.

The images still use the same three colors, three shapes, two-object layout family
and rendering conventions as training. No occlusion, natural photos, new categories
or distribution-shift performance is established.

## Recheck and integrity

The predesignated first shard was rerun once in a separate process with the same
candidate. All40 baseline, gate/logit, routed and feature-input rows match byte for
byte, and metrics match. It was not chosen because it scored best.

Candidate weights/config and frozen computation hashes remain unchanged. Per-shard
reports have zero requested training updates, no train IDs/optimizer steps and the
explicit fresh-holdout role. The original baseline checkout remains clean.

The five first runs total about169.76 seconds, including model loading, feature
extraction, original localization and routing evaluation. The recheck adds33.17
seconds; data preparation and software tests are additional. Original localization
was run for every question before routing, so this is not an inference-skipping
latency benchmark. No cloud resources or uploads were used.

The prepared Mac suite now has62 passing tests, including rejection of fresh-holdout
training before data/model loading; Ruff check/format pass. These tests alone do
not prove model generalization.

## Artifacts and reproduction

- Fixed plan: `configs/canonical-independent-v1.json`.
- New data: `data/independent-canonical-v1-shard01/` through`shard05/`.
- Per-shard results: `outputs/independent-canonical-v1-shard01/` through`shard05/`.
- Combined records/report/acceptance: `outputs/canonical-independent-v1/`.
- Predesignated recheck: `outputs/independent-canonical-v1-shard01-recheck/` and
  combined `recheck-verification.json`.

All model/data artifacts remain local and ignored. `acceptance.json` separately
records the research-screen pass and **`deployment_accepted=false`**. Combined
truth/prediction files are namespaced metric records with source paths, not a new
training dataset.

Evaluation command pattern, using a new output directory and an existing prepared
shard (no training or checkpoint selection):

```sh
BASELINE_REPO=/path/to/mlx-grounding-pi
"$BASELINE_REPO/.venv/bin/python" -m rejection_lab.train_existence \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-4bit" \
  --dataset data/independent-canonical-v1-shard01 \
  --head outputs/mac-existence-canonical-cost1-001/checkpoints/step-0064 \
  --feature-query canonical --evaluation-role fresh-synthetic-holdout \
  --image-tokens 64 --max-seconds 240 \
  --output outputs/independent-shard01-reproduced
```

## Next boundary

This supports transfer to new samples **within this generator distribution**, not
merely repeatability on the old development set. It does not justify more training
steps, broad absence reasoning claims or deployment. A fixed-candidate stress test
with changed layouts/categories/backgrounds, followed by rights-cleared independent
natural-image annotations, would test stronger claims. Those tests have not run.
If future design choices use this holdout's results, it becomes observed development
data and cannot keep being described as a fresh blind final test.
