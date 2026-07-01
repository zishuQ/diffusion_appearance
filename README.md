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

`ProjectionHead`, `UpdateBlock`, and `CrossAttentionGate` remain in the codebase for compatibility, but they are disabled in the main Stage 1 training and `eval_feature_level.py` path. Stage 2 and TrackTrack integration are intentionally not part of this refactor.

All training, validation, export, evaluation, and tracker experiments must be run serially. Do not start parallel jobs or background experiments on this machine.
