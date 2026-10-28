# GroundingPI Rejection Lab

<p align="left">
  <a href="https://github.com/kongzhuww/grounding-pi-rejection-lab/actions/workflows/ci.yml"><img src="https://github.com/kongzhuww/grounding-pi-rejection-lab/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>
</p>

**English** | [简体中文](README_zh.md)

**Independent, unofficial experiment repository. Preparation stage; no trained model
or validated fine-tuning backend is provided yet.**

Goal: improve **missing-target rejection** without sacrificing localization when a
target really exists. Keep the runnable [MLX baseline](https://github.com/kongzhuww/mlx-grounding-pi)
separate and reproducible. Its initial commit is pinned in `baseline.json`.

## Why a separate lab?

In the baseline's development scene, the original CUDA BF16 model and Mac 8-bit
both return coordinates for an absent yellow triangle rather than `None`. This is
one reproduced false positive, not evidence that the model is generally poor.
The lab investigates this behavior; it does not silently alter the baseline.

## What works now

- Deterministic original synthetic scene generation, with paired present/absent
  queries for both point and box tasks and native GAM answer strings.
- 80/10/10 splits by **scene**, keeping all queries for an image together.
- Image hashes and checks against cross-split scene/image leakage.
- Metrics for false positives, false rejection, positive localization, invalid
  output and negative failures including invalid responses.
- Optional local MLX evaluation through a trusted baseline checkout, loading a
  local model once; no remote inference or cloud provisioning.
- 16 passing unit tests for the data protocol and degenerate predictors.
- A completed [untrained local validation smoke](docs/initial-smoke.md): two positive
  questions pass and two negative questions fail, all concerning one scene. This is
  pipeline evidence, not a fine-tuning gain or an accuracy estimate.

**Not implemented:** optimizer/training loop, LoRA attachment, gradient checks,
adapter export/merge or fine-tuning improvements. `configs/lora-plan.json` is a
proposal, not an executable training configuration. A text-only `mlx_lm` training
command is not assumed to support this custom multimodal architecture.

## Local data preparation

Python 3.11+. Install the small data-tool dependencies in your chosen environment:

```sh
python -m pip install -e '.[dev]'
python -m rejection_lab.data --output data/synthetic-v0 --scenes 100 --seed 42
python -m rejection_lab.data --output data/synthetic-v0 --validate-only
python -m pytest -q
```

Each scene has two colored shapes and four queries: present point/box, absent
point/box. Absences include missing colors, missing shapes and absent combinations
of otherwise present attributes. Point anchors are centers for circles/squares and
centroids for triangles. Coordinates are quantized to 0–999 using original-image
pixel extents. The dataset is a **pipeline smoke test**, not evidence of natural-image
generalization; add independently labeled natural images before making quality claims.
Output directories are never overwritten. If generation is interrupted, use a new
output directory; validation rejects incomplete splits. Generated data stays ignored.

For the first tiny end-to-end smoke, `--scenes 10` produces 40 rows: 32 train,
4 validation, 4 test. Do not train on validation or repeatedly tune on test.

## Local baseline evaluation (optional)

Use an existing Apple Silicon environment containing the baseline's MLX dependencies.
This command neither installs a model nor contacts a hosted inference service:

```sh
# Run from this repository, using your trusted baseline checkout's Python.
BASELINE_REPO=/path/to/mlx-grounding-pi
"$BASELINE_REPO/.venv/bin/python" -m rejection_lab.baseline \
  --baseline-repo "$BASELINE_REPO" \
  --model "$BASELINE_REPO/models/groundingpi-mlx-8bit" \
  --dataset data/synthetic-v0 --split validation --output outputs/baseline-validation
```

The baseline checkout must be clean. Its actual commit and model conversion metadata
are recorded in the local report; compare the commit with `baseline.json`. The model
is used with its normal image budget and greedy generation. A strict normalization
adapter requires one matching label and at most one coordinate tuple; wrong-label,
multiple-target, malformed or truncated output is counted invalid, not as rejection.

Metrics consume one row per prediction:

```json
{"id":"scene-000008-absent-point","valid":true,"coordinates":null}
```

`null` means a valid explicit `None` answer. Invalid output must have `valid: false`;
it must never be converted into a correct rejection. Coordinates are 0–999 integers,
with two values for point or four ordered values for bbox.

```sh
python -m rejection_lab.metrics \
  --truth data/synthetic-v0/validation.jsonl \
  --predictions outputs/baseline-validation/predictions.jsonl \
  --output outputs/rescored-validation.json
```

Missing/duplicate/extra prediction IDs fail rather than disappear from denominators.
Defaults: 15-pixel point tolerance, box IoU >= 0.70, at the generated 320×256 resolution.
These thresholds are development choices, not official benchmark settings.
An always-`None` predictor gets perfect rejection but zero positive localization;
invalid-only output cannot pass negative-failure evaluation.

## Next steps before training

1. Establish a baseline on scene-disjoint positive/negative samples.
2. Add human-checked real photographs with rights/provenance and harder counterexamples.
3. Validate the original multimodal training path and frozen vision/projector LoRA
   gradient flow; inspect token loss masks and actual trainable parameters.
4. Run a tiny overfit/reload check, then a controlled validation comparison.
5. Only after compatibility and budget approval, scale training and evaluate a locked test set.

See [experiment protocol](docs/experiment-plan.md). No paid training has been started.
No credentials, model weights, personal photos or generated datasets are published here.

## License

Original lab code is Apache-2.0. GroundingPI/Kimi-derived code, original weights and
future adapters retain applicable upstream conditions; this repo does not relicense
them. See [third-party notices](THIRD_PARTY_NOTICES.md) and preserved license texts.
