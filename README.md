# APUDiff

APUDiff is currently focused on a raw 2048-d direct delta diffusion predictor for MOT appearance modeling.

Main experiment path:

```text
normalized raw FastReID history features [B, K, 2048]
-> DiffusionPredictor
-> raw delta_feat [B, 2048]
-> pred_feat = normalize(last_feat + delta_feat)
```

The active Stage 1 target is to beat last-feature and EMA baselines at feature level:

```text
loss_pred < loss_ema
rank_acc_pred > rank_acc_ema
```

The codebase is intentionally predictor-only in this exploration branch. `ProjectionHead`, `UpdateBlock`, `CrossAttentionGate`, Stage 2 training, and match-level gate evaluation have been removed instead of preserved as compatibility paths.

Dataset configs inherit shared defaults through `base:`. Common raw-delta predictor and training defaults live in `configs/apu_diff_base.yaml`; dataset configs only override paths, sequence splits, and experiment-specific values.

All training, validation, export, evaluation, and tracker experiments must be run serially. Do not start parallel jobs or background experiments on this machine.
