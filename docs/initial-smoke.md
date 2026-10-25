# Initial local validation smoke (untrained)

This is an **untrained** 8-bit baseline, not a fine-tuning result or a benchmark.
The code tests also passed: 16 unit tests. No cloud task or training was started.

## Reproduction

Generate with `--scenes 10 --seed 42`, then run `rejection_lab.baseline` on the
**validation** split, with the baseline commit pinned in `baseline.json` and its
language-only 8-bit / BF16-vision checkpoint. Keep the normal image budget and
128-token output cap used by the local evaluation entry point.

- 10 independently rendered scenes; 40 total question/answer rows.
- Train: 8 scenes / 32 questions.
- Validation: 1 scene / 4 questions.
- Test: 1 scene / 4 questions; no model inference was run on test.

The validation's four questions concern **the same image**: point and box for a
present yellow circle, and point and box for an absent yellow square. They are not
four independent images or two independent negative scenes.

| Query | Target (0–999) | Untrained baseline prediction |
|---|---|---|
| Present yellow circle, point | `(745,474)` | `(732,472)` |
| Present yellow circle, box | `(633,333,858,615)` | `(617,330,840,612)` |
| Absent yellow square, point | `None` | `(728,472)` |
| Absent yellow square, box | `None` | `(158,413,317,611)` |

At the declared 15-pixel point / 0.70 box-IoU thresholds: 2/2 positive questions
pass, both negative questions are false positives, and no responses are invalid.
These tiny counts cannot estimate general accuracy or false-positive rates. They
only establish that generation, image-conditioned baseline inference, strict
normalization and metric reporting work end to end, with failures retained.

Local raw data and per-question output stay under ignored `data/` and `outputs/`.
Any later adapter must be evaluated independently without training on this validation
image; natural-image data and a larger locked holdout are still required.
