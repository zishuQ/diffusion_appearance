#!/usr/bin/env bash
set -euo pipefail

cd /home/shang/workspace/diffusion_appearance

BASE_CONFIG="${BASE_CONFIG:-configs/apu_diff_mot20.yaml}"
DEVICE="${DEVICE:-cuda}"
SESSION_NAME="${TMUX_SESSION:-apu_tune_raw_delta}"
SHORT_EPOCHS="${SHORT_EPOCHS:-5}"
LONG_EPOCHS="${LONG_EPOCHS:-15}"
LOG_ROOT="logs/tuning_raw_delta"
CONFIG_ROOT="configs/tuning_raw_delta"
CKPT_ROOT="checkpoints/tuning_raw_delta"
METRICS_FILE="${LOG_ROOT}/metrics.tsv"
RUNNING_FILE="${LOG_ROOT}/RUNNING.md"
SUMMARY_FILE="${LOG_ROOT}/SUMMARY.md"

mkdir -p "${LOG_ROOT}" "${CONFIG_ROOT}" "${CKPT_ROOT}"

cat > "${RUNNING_FILE}" <<EOF
# Raw Delta Tuning Run

- status: running
- started_at: $(date -Is)
- tmux_session: ${SESSION_NAME}
- script_pid: $$
- base_config: ${BASE_CONFIG}
- device: ${DEVICE}
- main_log_dir: ${LOG_ROOT}
- config_dir: ${CONFIG_ROOT}
- checkpoint_dir: ${CKPT_ROOT}
- metrics_file: ${METRICS_FILE}

All trials run serially in this one script.
EOF

printf "trial\tkind\tbatch_size\tlr_stage1\tstage1_diff_weight\tstage1_improve_weight\tema_alpha\tepochs\tcheckpoint\ttrain_log\teval_log\tloss_pred\tloss_ema\trank_acc_pred\trank_acc_ema\tcos_pred_minus_ema\tcos_pred_minus_last\tstatus\n" > "${METRICS_FILE}"

write_config() {
  local name="$1"
  local epochs="$2"
  local batch_size="$3"
  local lr_stage1="$4"
  local diff_weight="$5"
  local improve_weight="$6"
  local ema_alpha="$7"
  local cfg_path="${CONFIG_ROOT}/${name}.yaml"
  local ckpt_dir="${CKPT_ROOT}/${name}"
  local log_dir="${LOG_ROOT}/${name}_script_logs"
  mkdir -p "${ckpt_dir}" "${log_dir}"
  .venv/bin/python - "${BASE_CONFIG}" "${cfg_path}" "${ckpt_dir}" "${log_dir}" "${epochs}" "${batch_size}" "${lr_stage1}" "${diff_weight}" "${improve_weight}" "${ema_alpha}" <<'PY'
import sys
from pathlib import Path
import yaml

base_path, out_path, ckpt_dir, log_dir, epochs, batch_size, lr_stage1, diff_weight, improve_weight, ema_alpha = sys.argv[1:]
with Path(base_path).open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f) or {}

data["reid_dim"] = 2048
data["latent_dim"] = 2048
data.setdefault("predictor", {})
data["predictor"]["num_diffusion_steps"] = 1

training = data.setdefault("training", {})
training["batch_size"] = int(batch_size)
training["stage1_epochs"] = int(epochs)
training["lr_stage1"] = float(lr_stage1)
training["stage1_diff_weight"] = float(diff_weight)
training["stage1_improve_weight"] = float(improve_weight)
training["stage1_projection_warmup_epochs"] = 0
training["freeze_projection_after_warmup"] = False
training["ema_alpha"] = float(ema_alpha)

data["checkpoint_dir"] = ckpt_dir
data["log_dir"] = log_dir

with Path(out_path).open("w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)
PY
  printf "%s\n" "${cfg_path}"
}

select_checkpoint() {
  local ckpt_dir="$1"
  if [[ -f "${ckpt_dir}/stage1_predictor_best_improve.pth" ]]; then
    printf "%s\n" "${ckpt_dir}/stage1_predictor_best_improve.pth"
  elif [[ -f "${ckpt_dir}/stage1_predictor.pth" ]]; then
    printf "%s\n" "${ckpt_dir}/stage1_predictor.pth"
  else
    printf "%s\n" "${ckpt_dir}/APUDiff_stage1_last.pth"
  fi
}

record_metrics() {
  local trial="$1"
  local kind="$2"
  local batch_size="$3"
  local lr_stage1="$4"
  local diff_weight="$5"
  local improve_weight="$6"
  local ema_alpha="$7"
  local epochs="$8"
  local checkpoint="$9"
  local train_log="${10}"
  local eval_log="${11}"
  local status="${12}"
  .venv/bin/python - "${METRICS_FILE}" "${trial}" "${kind}" "${batch_size}" "${lr_stage1}" "${diff_weight}" "${improve_weight}" "${ema_alpha}" "${epochs}" "${checkpoint}" "${train_log}" "${eval_log}" "${status}" <<'PY'
import re
import sys
from pathlib import Path

metrics_file, trial, kind, batch_size, lr_stage1, diff_weight, improve_weight, ema_alpha, epochs, checkpoint, train_log, eval_log, status = sys.argv[1:]
metrics = {}
eval_path = Path(eval_log)
if eval_path.exists():
    for line in eval_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"^([A-Za-z0-9_]+):\s*([-+0-9.eE]+)\s*$", line.strip())
        if match:
            metrics[match.group(1)] = match.group(2)

fields = [
    trial,
    kind,
    batch_size,
    lr_stage1,
    diff_weight,
    improve_weight,
    ema_alpha,
    epochs,
    checkpoint,
    train_log,
    eval_log,
    metrics.get("loss_pred", "nan"),
    metrics.get("loss_ema", "nan"),
    metrics.get("rank_acc_pred", "nan"),
    metrics.get("rank_acc_ema", "nan"),
    metrics.get("cos_pred_minus_ema", "nan"),
    metrics.get("cos_pred_minus_last", "nan"),
    status,
]
with Path(metrics_file).open("a", encoding="utf-8") as f:
    f.write("\t".join(fields) + "\n")
PY
}

meets_target() {
  local eval_log="$1"
  .venv/bin/python - "${eval_log}" <<'PY'
import re
import sys
from pathlib import Path

metrics = {}
for line in Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace").splitlines():
    match = re.match(r"^([A-Za-z0-9_]+):\s*([-+0-9.eE]+)\s*$", line.strip())
    if match:
        metrics[match.group(1)] = float(match.group(2))

ok = (
    metrics.get("loss_pred", float("inf")) < metrics.get("loss_ema", float("-inf"))
    and metrics.get("rank_acc_pred", float("-inf")) > metrics.get("rank_acc_ema", float("inf"))
    and metrics.get("cos_pred_minus_ema", float("-inf")) > 0.0
)
raise SystemExit(0 if ok else 1)
PY
}

run_trial() {
  local trial="$1"
  local kind="$2"
  local epochs="$3"
  local batch_size="$4"
  local lr_stage1="$5"
  local diff_weight="$6"
  local improve_weight="$7"
  local ema_alpha="$8"
  local cfg_path
  cfg_path="$(write_config "${trial}" "${epochs}" "${batch_size}" "${lr_stage1}" "${diff_weight}" "${improve_weight}" "${ema_alpha}")"
  local ckpt_dir="${CKPT_ROOT}/${trial}"
  local train_log="${LOG_ROOT}/${trial}.log"
  local eval_log="${LOG_ROOT}/${trial}_eval.log"

  echo "[$(date -Is)] START ${trial}" | tee -a "${LOG_ROOT}/run_raw_delta_tuning.log"
  if ! .venv/bin/python train_stage1_predictor.py \
      --config "${cfg_path}" \
      --device "${DEVICE}" \
      --checkpoint-dir "${ckpt_dir}" \
      --log-dir "${LOG_ROOT}/${trial}_script_logs" \
      --log-name "${trial}" \
      > "${train_log}" 2>&1; then
    if rg -qi "out of memory|CUDA.*OOM|CUDA out of memory" "${train_log}" && [[ "${batch_size}" -gt 16 ]]; then
      echo "[$(date -Is)] ${trial} hit OOM, retrying with batch_size=16" | tee -a "${LOG_ROOT}/run_raw_delta_tuning.log"
      batch_size=16
      cfg_path="$(write_config "${trial}_bs16" "${epochs}" "${batch_size}" "${lr_stage1}" "${diff_weight}" "${improve_weight}" "${ema_alpha}")"
      ckpt_dir="${CKPT_ROOT}/${trial}_bs16"
      train_log="${LOG_ROOT}/${trial}_bs16.log"
      eval_log="${LOG_ROOT}/${trial}_bs16_eval.log"
      trial="${trial}_bs16"
      .venv/bin/python train_stage1_predictor.py \
        --config "${cfg_path}" \
        --device "${DEVICE}" \
        --checkpoint-dir "${ckpt_dir}" \
        --log-dir "${LOG_ROOT}/${trial}_script_logs" \
        --log-name "${trial}" \
        > "${train_log}" 2>&1
    else
      record_metrics "${trial}" "${kind}" "${batch_size}" "${lr_stage1}" "${diff_weight}" "${improve_weight}" "${ema_alpha}" "${epochs}" "" "${train_log}" "${eval_log}" "train_failed"
      return 2
    fi
  fi

  local checkpoint
  checkpoint="$(select_checkpoint "${ckpt_dir}")"
  if [[ ! -f "${checkpoint}" ]]; then
    record_metrics "${trial}" "${kind}" "${batch_size}" "${lr_stage1}" "${diff_weight}" "${improve_weight}" "${ema_alpha}" "${epochs}" "${checkpoint}" "${train_log}" "${eval_log}" "missing_checkpoint"
    return 2
  fi

  .venv/bin/python eval_feature_level.py \
    --config "${cfg_path}" \
    --checkpoint "${checkpoint}" \
    --device "${DEVICE}" \
    --log-dir "${LOG_ROOT}/${trial}_script_logs" \
    --log-name "${trial}_eval" \
    > "${eval_log}" 2>&1

  record_metrics "${trial}" "${kind}" "${batch_size}" "${lr_stage1}" "${diff_weight}" "${improve_weight}" "${ema_alpha}" "${epochs}" "${checkpoint}" "${train_log}" "${eval_log}" "ok"
  echo "[$(date -Is)] DONE ${trial}" | tee -a "${LOG_ROOT}/run_raw_delta_tuning.log"
  if meets_target "${eval_log}"; then
    echo "${trial}" > "${LOG_ROOT}/target_met_trial.txt"
    return 10
  fi
  return 0
}

best_trial_args() {
  .venv/bin/python - "${METRICS_FILE}" <<'PY'
import csv
import math
import sys
from pathlib import Path

rows = list(csv.DictReader(Path(sys.argv[1]).open("r", encoding="utf-8"), delimiter="\t"))
valid = [r for r in rows if r.get("kind") == "short" and r.get("status") == "ok"]
if not valid:
    raise SystemExit(1)

def f(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")

def score(row):
    c = f(row, "cos_pred_minus_ema")
    rp = f(row, "rank_acc_pred")
    re = f(row, "rank_acc_ema")
    if math.isnan(c) or math.isnan(rp) or math.isnan(re):
        return -1e9
    return c + (rp - re)

best = max(valid, key=score)
print(best["batch_size"], best["lr_stage1"], best["stage1_diff_weight"], best["stage1_improve_weight"], best["ema_alpha"], best["trial"])
PY
}

write_summary() {
  local still_running="$1"
  .venv/bin/python - "${METRICS_FILE}" "${SUMMARY_FILE}" "${RUNNING_FILE}" "${still_running}" <<'PY'
import csv
import sys
from pathlib import Path

metrics_file, summary_file, running_file, still_running = sys.argv[1:]
rows = list(csv.DictReader(Path(metrics_file).open("r", encoding="utf-8"), delimiter="\t"))

def f(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")

ok_rows = [r for r in rows if r.get("status") == "ok"]
best = None
if ok_rows:
    best = max(ok_rows, key=lambda r: f(r, "cos_pred_minus_ema") + (f(r, "rank_acc_pred") - f(r, "rank_acc_ema")))

lines = ["# Raw Delta Tuning Summary", ""]
lines.append(f"- metrics_file: {metrics_file}")
lines.append(f"- background_task_still_running: {still_running}")
lines.append("")
lines.append("## Experiments")
lines.append("")
lines.append("| trial | kind | bs | lr | diff_w | improve_w | ema_alpha | epochs | loss_pred | loss_ema | rank_pred | rank_ema | cos_pred_minus_ema | status |")
lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
for r in rows:
    lines.append(
        f"| {r['trial']} | {r['kind']} | {r['batch_size']} | {r['lr_stage1']} | {r['stage1_diff_weight']} | "
        f"{r['stage1_improve_weight']} | {r['ema_alpha']} | {r['epochs']} | {r['loss_pred']} | {r['loss_ema']} | "
        f"{r['rank_acc_pred']} | {r['rank_acc_ema']} | {r['cos_pred_minus_ema']} | {r['status']} |"
    )
lines.append("")
lines.append("## Best Trial")
lines.append("")
if best:
    beat_ema = (
        f(best, "loss_pred") < f(best, "loss_ema")
        and f(best, "rank_acc_pred") > f(best, "rank_acc_ema")
        and f(best, "cos_pred_minus_ema") > 0
    )
    lines.extend([
        f"- best_trial: {best['trial']}",
        f"- beat_ema: {beat_ema}",
        f"- checkpoint: {best['checkpoint']}",
        f"- train_log: {best['train_log']}",
        f"- eval_log: {best['eval_log']}",
        f"- loss_pred: {best['loss_pred']}",
        f"- loss_ema: {best['loss_ema']}",
        f"- rank_acc_pred: {best['rank_acc_pred']}",
        f"- rank_acc_ema: {best['rank_acc_ema']}",
        f"- cos_pred_minus_ema: {best['cos_pred_minus_ema']}",
    ])
else:
    lines.append("- best_trial: none")
lines.append("")
lines.append("## Notes")
lines.append("")
lines.append("- Stage2, TrackTrack, Gate, UpdateBlock, ProjectionHead forward, and base/residual predictor paths were not used.")
lines.append("- Continue by inspecting RUNNING.md while the tmux job is active, or SUMMARY.md after it finishes.")
Path(summary_file).write_text("\n".join(lines) + "\n", encoding="utf-8")

if still_running == "false":
    text = Path(running_file).read_text(encoding="utf-8", errors="replace")
    text = text.replace("- status: running", "- status: completed")
    text += f"\n- completed_at: __COMPLETED_AT__\n- summary_file: {summary_file}\n"
    Path(running_file).write_text(text, encoding="utf-8")
PY
  if [[ "${still_running}" == "false" ]]; then
    .venv/bin/python - "${RUNNING_FILE}" <<'PY'
import sys
from datetime import datetime
from pathlib import Path

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
text = text.replace("__COMPLETED_AT__", datetime.now().astimezone().isoformat())
path.write_text(text, encoding="utf-8")
PY
  fi
}

trap 'write_summary true' EXIT

echo "[$(date -Is)] Health compile" | tee -a "${LOG_ROOT}/run_raw_delta_tuning.log"
.venv/bin/python -m compileall -q apu_diff train_stage1_predictor.py eval_feature_level.py

run_trial "trial1_baseline_stable" "short" "${SHORT_EPOCHS}" 32 1e-4 0.2 0.5 0.9 || status=$?
status="${status:-0}"
if [[ "${status}" == "10" ]]; then
  write_summary false
  trap - EXIT
  exit 0
elif [[ "${status}" != "0" ]]; then
  write_summary false
  trap - EXIT
  exit 1
fi
unset status

run_trial "trial2_weaker_diff" "short" "${SHORT_EPOCHS}" 32 1e-4 0.1 0.5 0.9 || status=$?
status="${status:-0}"
if [[ "${status}" == "10" ]]; then
  write_summary false
  trap - EXIT
  exit 0
elif [[ "${status}" != "0" ]]; then
  write_summary false
  trap - EXIT
  exit 1
fi
unset status

run_trial "trial3_stronger_improve" "short" "${SHORT_EPOCHS}" 32 1e-4 0.1 1.0 0.9 || status=$?
status="${status:-0}"
if [[ "${status}" == "10" ]]; then
  write_summary false
  trap - EXIT
  exit 0
elif [[ "${status}" != "0" ]]; then
  write_summary false
  trap - EXIT
  exit 1
fi
unset status

read -r best_bs best_lr best_diff best_improve best_alpha best_short_trial < <(best_trial_args)
echo "[$(date -Is)] Best short trial: ${best_short_trial}; starting long run" | tee -a "${LOG_ROOT}/run_raw_delta_tuning.log"
run_trial "long_from_${best_short_trial}" "long" "${LONG_EPOCHS}" "${best_bs}" "${best_lr}" "${best_diff}" "${best_improve}" "${best_alpha}" || status=$?
status="${status:-0}"
if [[ "${status}" != "0" && "${status}" != "10" ]]; then
  write_summary false
  trap - EXIT
  exit 1
fi

write_summary false
trap - EXIT
