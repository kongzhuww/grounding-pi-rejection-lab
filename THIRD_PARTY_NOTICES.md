# Third-party notices

Original lab scripts and documentation are offered under Apache-2.0 (`LICENSE`).
The synthetic shapes are generated locally by this project's own code; no third-party
photo dataset or model weights are included in the repository.

## GroundingPI and baseline

- Model: https://huggingface.co/GroundingPI/GroundingPI
- Model revision: `08d397c8769f7f10b96ff6d56ee6f56acb04181f`
- Original project: https://github.com/groundingpi/GroundingPI
- Independent MLX baseline: https://github.com/kongzhuww/mlx-grounding-pi
- Baseline commit: see `baseline.json`.

The original model package distinguishes its authors' Apache-2.0 contributions from
Kimi-derived material subject to Kimi K3 conditions. Copies of its scope statement
and Kimi K3 text are retained unchanged under `licenses/`. The permission/copyright
notices and conditional commercial/attribution terms remain applicable where relevant.
This lab's license does not relicense model weights, derived code or future adapters.
Review upstream terms and training-data rights before distributing a trained artifact.

## Installed dependencies

Pillow and optional development tools retain their own licenses. The optional local
inference entry point imports an explicitly selected baseline checkout; that project
retains its MLX/mlx-lm and upstream notices. No third-party implementation is vendored
into this lab, apart from preserving license texts for clarity.
