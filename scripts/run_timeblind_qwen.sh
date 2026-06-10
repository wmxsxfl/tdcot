#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:-/data/datasets/TimeBlind/data.jsonl}
DATASET_ROOT=${DATASET_ROOT:-/data/datasets}
AUX_MODEL=${AUX_MODEL:-/data/models/Qwen2.5-7B-Instruct}
VL_MODEL=${VL_MODEL:-${QWEN_VL_MODEL:-/data/models/Qwen2.5-VL-7B-Instruct}}
OUT_DIR=${OUT_DIR:-outputs}
TIMEBLIND_LIMIT=${TIMEBLIND_LIMIT:-2400}
TIMEBLIND_LIMIT_PAIRS=${TIMEBLIND_LIMIT_PAIRS:-1200}

mkdir -p "$OUT_DIR"

python src/run_timeblind_direct_qwen.py \
  --data-path "$DATA_PATH" \
  --dataset-root "$DATASET_ROOT" \
  --model-path "$VL_MODEL" \
  --limit "$TIMEBLIND_LIMIT" \
  --output "$OUT_DIR/timeblind_qwen_base.jsonl"

python src/run_timeblind_tdcot_pair_qwen.py \
  --data-path "$DATA_PATH" \
  --dataset-root "$DATASET_ROOT" \
  --aux-model "$AUX_MODEL" \
  --vl-model "$VL_MODEL" \
  --limit-pairs "$TIMEBLIND_LIMIT_PAIRS" \
  --output "$OUT_DIR/timeblind_qwen_tdcot.jsonl"
