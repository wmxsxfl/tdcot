#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:-/data/datasets/TimeBlind/data.jsonl}
DATASET_ROOT=${DATASET_ROOT:-/data/datasets}
AUX_MODEL=${AUX_MODEL:-/data/models/Qwen2.5-7B-Instruct}
VL_MODEL=${VL_MODEL:-${QWEN_VL_MODEL:-/data/models/Qwen2.5-VL-7B-Instruct}}
TIMEBLIND_LIMIT=${TIMEBLIND_LIMIT:-2400}
OUT_DIR=${OUT_DIR:-outputs}

mkdir -p "$OUT_DIR"

python src/run_timeblind_generic_cot_qwen.py \
  --data-path "$DATA_PATH" \
  --data-root "$DATASET_ROOT" \
  --model-path "$VL_MODEL" \
  --limit "$TIMEBLIND_LIMIT" \
  --output "$OUT_DIR/timeblind_qwen_generic_cot.jsonl"

python src/run_timeblind_aux_cot_qwen.py \
  --data-path "$DATA_PATH" \
  --data-root "$DATASET_ROOT" \
  --aux-model-path "$AUX_MODEL" \
  --vl-model-path "$VL_MODEL" \
  --limit "$TIMEBLIND_LIMIT" \
  --output "$OUT_DIR/timeblind_qwen_auxiliary_cot.jsonl"

python src/run_timeblind_localize_answer_qwen.py \
  --data-path "$DATA_PATH" \
  --data-root "$DATASET_ROOT" \
  --aux-model-path "$AUX_MODEL" \
  --vl-model-path "$VL_MODEL" \
  --limit "$TIMEBLIND_LIMIT" \
  --output "$OUT_DIR/timeblind_qwen_localize_answer.jsonl"

python src/run_timeblind_tsp_qwen.py \
  --data-path "$DATA_PATH" \
  --data-root "$DATASET_ROOT" \
  --model-path "$VL_MODEL" \
  --limit "$TIMEBLIND_LIMIT" \
  --output "$OUT_DIR/timeblind_qwen_tsp.jsonl"
