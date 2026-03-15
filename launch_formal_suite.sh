#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/cxj/experiments/vit_quant_5ideas"
PY="/home/cxj/miniconda3/envs/cdfquant/bin/python"
LOG_DIR="$ROOT/results/logs"

mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  echo "[$(date '+%F %T')] START $name" | tee -a "$LOG_DIR/suite_progress.log"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
  echo "[$(date '+%F %T')] END $name" | tee -a "$LOG_DIR/suite_progress.log"
}

run_step "idea2_main_full" \
  "$PY" "$ROOT/run_idea2_full.py" \
  --run-name main_full \
  --models deit_small vit_base \
  --bits 3 4 \
  --modes direct_qkv rotated_qk rotated_qkv \
  --val-images 50000 \
  --calib-sizes 128 \
  --num-workers 8

run_step "idea4_main_full" \
  "$PY" "$ROOT/run_idea4_full.py" \
  --run-name main_full \
  --models deit_small vit_base \
  --val-images 50000 \
  --calib-sizes 64 \
  --target-avgs 3.25 3.5 3.75 \
  --random-seeds 0 1 2 \
  --num-workers 8

run_step "idea2_ablations_5k" \
  "$PY" "$ROOT/run_idea2_full.py" \
  --run-name ablations_5k \
  --models deit_small vit_base \
  --bits 3 4 \
  --modes direct_qkv rotated_q rotated_k rotated_qk rotated_qkv \
  --val-images 5000 \
  --calib-sizes 32 128 \
  --num-workers 8

run_step "idea4_ablations_5k" \
  "$PY" "$ROOT/run_idea4_full.py" \
  --run-name ablations_5k \
  --models deit_small vit_base \
  --val-images 5000 \
  --calib-sizes 16 64 \
  --target-avgs 3.25 3.5 3.75 \
  --random-seeds 0 1 2 \
  --num-workers 8
