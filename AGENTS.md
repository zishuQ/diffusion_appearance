# Repository Guidelines

## Project Structure & Module Organization

This repository implements APUDiff, a feature-level appearance model for MOT tracking. The current exploration branch is a raw 2048-d direct-delta diffusion predictor. Core package code lives under `apu_diff/`:

- `apu_diff/models/`: diffusion predictor and APUDiff wrapper.
- `apu_diff/datasets/`: feature tracklet dataset and dataloader construction.
- `apu_diff/utils/`: feature I/O, losses, metrics, diffusion helpers, and training utilities.
- `configs/`: shared defaults in `apu_diff_base.yaml` plus dataset and experiment YAML files that inherit with `base:`.
- Top-level scripts handle workflows: `export_gt_features.py`, `train_stage1_predictor.py`, and `eval_feature_level.py`.

Large generated assets are expected under `datasets/`, `checkpoints/`, `logs/`, and `tracktrack_outputs/`; do not commit bulky generated files unless explicitly required.

## Build, Test, and Development Commands

Use the existing virtual environment. If packages must be installed, use `uv pip`, not plain `pip`.

```bash
.venv/bin/python -m compileall -q apu_diff train_stage1_predictor.py eval_feature_level.py eval_stage1_by_sequence.py
```

Runs a fast syntax/import sanity check.

```bash
.venv/bin/python train_stage1_predictor.py --config configs/apu_diff_mot20.yaml --batch-size 32 --num-workers 0
```

Runs Stage 1 predictor training. Keep long jobs serial to avoid exhausting GPU/system resources.

```bash
.venv/bin/python eval_feature_level.py --config configs/apu_diff_mot20.yaml --checkpoint checkpoints/mot20/apu_diff_full.pth
```

Runs feature-level evaluation.

## Coding Style & Naming Conventions

Python code uses 4-space indentation, type hints where useful, and concise module-level organization. Prefer clear tensor names already used in the project: `history_feats`, `history_mask`, `local_queue`, `target_z`, and `pred_feat`. Keep APIs compatible with batched tensors and avoid hard-coding feature dimensions; the current predictor uses 2048-D normalized raw ReID features.

## Testing Guidelines

There is no formal test suite yet. Before submitting changes, run `compileall` and a small shape smoke test for APUDiff with `B=2`, `K=5`, `reid_dim=2048`, and `latent_dim=2048`. Verify prediction and appearance-cost outputs have expected shapes and prediction features are L2-normalized.

## Commit & Pull Request Guidelines

The repository history uses conventional-style commits, e.g. `feat: ...`. Use short imperative messages such as `fix: handle MOT20 sequence split` or `feat: add detection observation loader`.

Pull requests should describe the experiment or code path changed, list commands run, mention dataset/checkpoint assumptions, and include key metrics or smoke-test output when applicable.

## Agent-Specific Instructions

Do not introduce EMA or original ReID feature blending into APUDiff evaluation unless explicitly requested. Keep TrackTrack baseline parameters unchanged when testing APUDiff; only APUDiff-specific components should vary. Stage 2 update/gate code has been removed from this branch; rebuild it deliberately if that direction is resumed.
