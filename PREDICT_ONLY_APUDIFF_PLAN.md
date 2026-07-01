# Predict-Only APUDiff Plan

## Goal

Refactor the main APUDiff tracking path to use the diffusion predictor as a
single-stage appearance transition model:

```text
matched raw ReID history [K, 2048]
-> APUDiff.predict()
-> pred_feat for current-frame association cost
```

The Stage 2 update/gate path is removed from the main codebase so metric
changes can be attributed to the predictor itself.

## Main Design

Use APUDiff only for prediction and association. Learned Stage 2 updates are
not part of this branch.

At frame `t`:

```text
track history before frame t
track identity state before frame t
-> pred_feat_t = APUDiff.predict(history, identity_state)
-> app_cost(track, det) = 1 - cos(pred_feat_t, det_feat_t)
```

After association:

```text
if matched:
    append normalized matched detection feature det_z to the track history
    update learned identity state from det_z
else:
    do not update appearance history or identity state
```

This keeps the tracker anchored to real observations after confirmed matches,
but the current-frame association cost is produced only by the predictor. There
is no learned update block, no gate, no EMA/original-ReID blending, and no
weighted mixture of predicted and observed features.

## Why Not Pure Closed-Loop First

Pure closed-loop would append `pred_feat_t` instead of the matched detection:

```text
if matched:
    append pred_feat_t
```

This should only be an ablation, not the first main path. A one-step predictor
trained on real history can drift if it is forced to consume its own predictions
for many frames. If we want this later, it needs explicit multi-step rollout
training and horizon metrics.

## Training Path

Keep Stage1 as the main training path:

```text
history real normalized ReID features [B, K, 2048]
identity warmup observations [B, M, 2048]
target real normalized ReID feature [B, 2048]
identity_state = learned online updates over identity warmup observations
delta_target = target - last
pred_feat = normalize(last + delta_hat), conditioned on identity_state
```

Losses remain predictor-focused:

```text
loss_pred
loss_diff
loss_improve_vs_ema
```

Do not train gate/update/projection-warmup components in this branch. Those
modules and scripts have been deleted.

## Tracker Integration

Add an explicit APUDiff update mode for tracker integration:

```text
--apu-history-update-mode observed
--apu-history-update-mode predicted
```

Default:

```text
observed
```

Mode semantics:

```text
observed:
    matched track appends matched det_z after association

predicted:
    matched track appends pred_feat after association
    ablation only, not the main result
```

There are no `model.update()` or `model.match_gate_value()` calls in the
predictor-only tracker path.

Keep TrackTrack baseline parameters and matching weights unchanged. Only the
APUDiff-specific feature source/update behavior should change.

## Evaluation Order

1. Let the current no-identity fixed-window tuning run finish.
2. Select the best no-identity predictor checkpoint from feature-level metrics.
3. Update TrackTrack APUDiff integration to use predictor-only observed-history
   updates.
4. Compile-check APUDiff and TrackTrack touched files.
5. Run a small TrackTrack MOT20-05 test with `--apu-history-update-mode observed`.

6. Only after observed-history predictor-only is understood, run:

```text
--apu-history-update-mode predicted
```

as a closed-loop ablation.

## Acceptance Criteria

The main path is valid only if:

1. Current-frame APUDiff app cost uses `pred_feat` only.
2. Matched detection feature is appended only after association.
3. No learned Stage2 update/gate is called.
4. No EMA/original-ReID blending is introduced.
5. Missed/lost tracks do not update appearance history.
6. Feature-level eval still reports:

```text
loss_pred
loss_ema
loss_last
cos_pred
cos_ema
cos_last
rank_acc_pred
rank_acc_ema
rank_acc_last
```

7. TrackTrack evaluation can be run with deterministic one-step APUDiff
   prediction.

## Identity Memory Design

The predictor uses true long-term identity memory as a predictor condition:

```text
recent local history tokens [B, K, 2048]
+ one learned identity state token [B, 1, 2048]
-> DDM context
-> delta prediction
```

The identity state is not `last_feat` and not a bag of stored frame tokens. It
is initialized from the first matched observation and updated by a learned
online update rule using matched observations before the prediction target. It
conditions prediction only and is not mixed into the output feature.
