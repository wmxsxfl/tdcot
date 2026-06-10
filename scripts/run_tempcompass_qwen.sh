#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:-/data/datasets/TempCompass/tempcompass_directionality_items.parquet}
VIDEO_ROOT=${VIDEO_ROOT:-/data/datasets/TempCompass/videos/videos}
AUX_MODEL=${AUX_MODEL:-/data/models/Qwen2.5-7B-Instruct}
VL_MODEL=${VL_MODEL:-${QWEN_VL_MODEL:-/data/models/Qwen2.5-VL-7B-Instruct}}
OUT_DIR=${OUT_DIR:-outputs}
TEMPCOMPASS_LIMIT=${TEMPCOMPASS_LIMIT:-0}

mkdir -p "$OUT_DIR"

python src/run_directionality_qwen.py \
  --dataset tempcompass \
  --method direct \
  --data-path "$DATA_PATH" \
  --video-root "$VIDEO_ROOT" \
  --vl-model "$VL_MODEL" \
  --limit "$TEMPCOMPASS_LIMIT" \
  --output "$OUT_DIR/tempcompass_directionality_qwen_base.jsonl"

python src/run_directionality_qwen.py \
  --dataset tempcompass \
  --method tdcot \
  --data-path "$DATA_PATH" \
  --video-root "$VIDEO_ROOT" \
  --aux-model "$AUX_MODEL" \
  --vl-model "$VL_MODEL" \
  --limit "$TEMPCOMPASS_LIMIT" \
  --output "$OUT_DIR/tempcompass_directionality_qwen_tdcot.jsonl"
