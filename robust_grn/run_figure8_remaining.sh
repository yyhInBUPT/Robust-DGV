#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate dgv
else
  echo "Cannot find conda init script: $HOME/miniconda3/etc/profile.d/conda.sh" >&2
  exit 1
fi

OUT_DIR="result/figure8_degree_maxq"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

MIN_FREE_GPU_MB="${MIN_FREE_GPU_MB:-12000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-300}"

COMBOS=(
  "citeseer:gcn"
  "amazon_photo:gcnii"
  "amazon_cs:gcnii"
  "actor:gcnii"
)

csv_path_for_combo() {
  case "$1" in
    "citeseer:gcn") echo "$OUT_DIR/csv/citeseer_gcn_degree_maxq.csv" ;;
    "amazon_photo:gcnii") echo "$OUT_DIR/csv/amazon_photo_gcnii_degree_maxq.csv" ;;
    "amazon_cs:gcnii") echo "$OUT_DIR/csv/amazon_cs_gcnii_degree_maxq.csv" ;;
    "actor:gcnii") echo "$OUT_DIR/csv/actor_gcnii_degree_maxq.csv" ;;
    *) echo "" ;;
  esac
}

echo "Figure 8 remaining experiments started at $(date)"
echo "Working directory: $SCRIPT_DIR"
echo "Output directory: $OUT_DIR"
echo "Minimum free GPU memory before each run: ${MIN_FREE_GPU_MB} MiB"
echo

wait_for_gpu_memory() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; skip GPU memory wait."
    return
  fi

  while true; do
    free_mb="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [ -z "$free_mb" ]; then
      echo "Could not read free GPU memory; retry in ${GPU_WAIT_SECONDS}s."
      sleep "$GPU_WAIT_SECONDS"
      continue
    fi
    if [ "$free_mb" -ge "$MIN_FREE_GPU_MB" ]; then
      echo "[$(date)] GPU free memory ${free_mb} MiB >= ${MIN_FREE_GPU_MB} MiB; continue."
      return
    fi
    echo "[$(date)] GPU free memory ${free_mb} MiB < ${MIN_FREE_GPU_MB} MiB; wait ${GPU_WAIT_SECONDS}s."
    sleep "$GPU_WAIT_SECONDS"
  done
}

for combo in "${COMBOS[@]}"; do
  csv_path="$(csv_path_for_combo "$combo")"
  log_name="${combo/:/_}.log"
  log_path="$LOG_DIR/$log_name"

  if [ -s "$csv_path" ]; then
    echo "[$(date)] Skip $combo because CSV already exists: $csv_path"
    continue
  fi

  wait_for_gpu_memory

  echo "[$(date)] Start $combo"
  echo "Log: $log_path"
  PYTHONUNBUFFERED=1 python figure8_degree_maxq.py \
    --only "$combo" \
    --output-dir "$OUT_DIR" \
    --cert-batch-size 8 \
    2>&1 | tee "$log_path"
  echo "[$(date)] Finished $combo"
  echo
done

required_csvs=(
  "$OUT_DIR/csv/citeseer_gcn_degree_maxq.csv"
  "$OUT_DIR/csv/citeseer_gcnii_degree_maxq.csv"
  "$OUT_DIR/csv/amazon_photo_gcnii_degree_maxq.csv"
  "$OUT_DIR/csv/amazon_cs_gcnii_degree_maxq.csv"
  "$OUT_DIR/csv/actor_gcnii_degree_maxq.csv"
)

missing=0
for csv in "${required_csvs[@]}"; do
  if [ ! -s "$csv" ]; then
    echo "Missing CSV, skip final plot-only step for now: $csv" >&2
    missing=1
  fi
done

if [ "$missing" -eq 0 ]; then
  echo "[$(date)] All CSVs exist. Rebuilding summary and figures."
  PYTHONUNBUFFERED=1 python figure8_degree_maxq.py \
    --plot-only \
    --output-dir "$OUT_DIR" \
    2>&1 | tee "$LOG_DIR/plot_only.log"
fi

echo "Figure 8 remaining experiments ended at $(date)"
