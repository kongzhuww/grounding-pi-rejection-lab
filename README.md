# GroundingPI Rejection Lab

<p align="left">
  <a href="https://github.com/kongzhuww/grounding-pi-rejection-lab/actions/workflows/ci.yml"><img src="https://github.com/kongzhuww/grounding-pi-rejection-lab/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-blue.svg" alt="License"></a>
</p>

**English** | [简体中文](README_zh.md)

**Independent, unofficial experiment repository. Bounded Mac-native LoRA and a separate
existence-head experiment run locally, but no natural-image or production-ready
rejection fix is claimed. A fixed canonical-query gate now passes a preregistered
fresh same-generator synthetic screen: 50 new images / 200 paired questions, preserving
98/100 positive localizations and rejecting 100/100 negatives. This is not distribution-
shift validation or deployment acceptance. Trained adapters/heads stay local and are
not distributed here.**

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
- 62 passing local tests, including evaluation-only fresh-holdout role enforcement,
  canonical task-independent queries, feature-version
  isolation and legacy loading, frozen image/query features, independent head
  training/reloading and explicit routing, masked-KL/teacher detachment, exact zero-KL
  objective/gradient equivalence, positive-loss/gradient scaling, default weighting,
  scene selection/epoch coverage, grouped metrics,
  and optional MLX quantized multimodal autodiff,
  frozen weights, answer-only loss, image conditioning and adapter roundtrip.
- A completed [untrained local validation smoke](docs/initial-smoke.md): two positive
  questions pass and two negative questions fail, all concerning one scene. This is
  pipeline evidence, not a fine-tuning gain or an accuracy estimate.

- **Real local training:** last-two-layer q/v LoRA on a frozen 4-bit base,
  answer/EOS-only loss, multi-scene selection, seeded epoch shuffling and adapter-only
  saving. Fresh-process reload reproduces validation outputs. See
  [Mac training](docs/mac-training.md).
- **Documented unsuccessful multi-scene trial:** 8 training scenes, 64 updates,
  10 validation scenes. Negative false positives fell from 20/20 to 0/20, but positive
  localization fell from 19/20 to 3/20. See [the retained failure](docs/multiscene-development.md).
- **Positive-weight ablation:** weight 3 with 16/32/48/64-step checkpoints. No checkpoint
  passes the fixed development screen; early checkpoints produce malformed negative
  outputs and later ones over-refuse. See [the complete curve](docs/positive-weight-development.md).
- **Two distillation ablations:** intermediate positive preservation improves, but no
  checkpoint passes the unchanged joint screen. Positive-only KL at step32 localizes
  20/20 positives and rejects 8/20 negatives, with 9 false positives and 3 invalids.
  See [both full curves and the teacher-conflict diagnostic](docs/distillation-development.md).

- **Initial task-specific frozen-localizer existence gate:** 5,122-parameter binary head with positive cost3,
  step32: retains original positive localization 19/20, correctly rejects 5/20 negatives,
  no added false rejection or invalid output. Still **15/20 false positives**, with
  **0/10 point negatives rejected**. This passes only the permissive development screen,
  not deployment acceptance. See [the complete cost1/cost3 experiment](docs/existence-head-development.md).
- **Task-independent existence query:** canonical input with equal class costs at
  steps64/128 preserves positive localization19/20, rejects20/20 negatives, and has
  zero false rejection/invalids. Point negatives10/10 and attribute-combination4/4
  pass on this same small reused development split. Cost3 rejects nothing; cost1/step32
  wrongly rejects every real target. All six checkpoints were independently reloaded;
  see [the full follow-up and its limits](docs/canonical-existence-development.md).

- **Fixed-candidate fresh synthetic holdout:** five predeclared seeds,50 new images,
  200 paired questions. Original localization98/100 preserved, negatives100/100
  correctly rejected, zero false rejection/invalids. Candidate and thresholds were
  fixed before data generation; the original locked test remains uninferred. Only
  two scenes cover missing attribute combinations; this is same-generator evidence,
  not natural-photo or OOD acceptance. See [plan, complete results and recheck](docs/independent-synthetic-evaluation.md).

**Not yet validated:** natural-image/OOD rejection improvements, broad-data training, resumed optimizer
state, adapter merging or full-model deployment after merging. `configs/lora-plan.json`
remains a broader proposal, not an executable training configuration. The implemented
Mac smoke has its own bounded CLI; it does not reuse a text-only trainer unchanged.

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

## Mac training and next steps

The first real local run completed: mean training loss **2.726 → 2.214** in eight
updates, about **13.27 seconds** including setup/checks/evaluation, with **3.70 GB**
peak MLX allocation (not process RSS). Two negative validation questions still fail;
this is compatibility evidence, **not a successful rejection fix**. The four validation
questions share one scene, and the four training questions share a different scene.

Follow the [Mac training commands and limitations](docs/mac-training.md) to reproduce.
The subsequent [multi-scene development trial](docs/multiscene-development.md) confirms
that training loss and negative FPR alone are unsafe objectives: the adapter over-refused
present objects and was not accepted. No cloud or paid training has been started.
A subsequent [positive-weight/checkpoint trial](docs/positive-weight-development.md)
found no acceptable checkpoint either. Subsequent
[distillation ablations](docs/distillation-development.md) show partial intermediate
positive preservation, but protocol errors and late over-refusal still prevent
acceptance. Negative teacher suffixes are themselves unreliable; removing their KL
improves protocol counts without solving the task. The subsequent
[separate existence-head experiment](docs/existence-head-development.md) preserves the
original localization weights and passes the weak development screen at cost3, but
rejection coverage is small and point/attribute-combination failures remain in that
initial task-specific trial. The [canonical-query follow-up](docs/canonical-existence-development.md)
now passes those subgroups on reused synthetic development with cost1, while retaining
its failed cost3 run and all-absent early checkpoint. Broader training data and new
independent evaluation remain necessary, not another blind step increase. A later
[fixed-candidate same-generator holdout](docs/independent-synthetic-evaluation.md)
now preserves98/100 positive localization and rejects100/100 negatives on50 new images.
It leaves the old locked test untouched; broader distribution-shift and natural-image
validation and application-specific acceptance are still required.

See [experiment protocol](docs/experiment-plan.md).
No credentials, model weights, personal photos or generated datasets are published here.

## License

Original lab code is Apache-2.0. GroundingPI/Kimi-derived code, original weights and
future adapters retain applicable upstream conditions; this repo does not relicense
them. See [third-party notices](THIRD_PARTY_NOTICES.md) and preserved license texts.
