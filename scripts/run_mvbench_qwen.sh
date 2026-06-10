#!/usr/bin/env bash
set -euo pipefail

DATA_PATH=${DATA_PATH:-/data/datasets/MVBench/mvbench_directionality_setting.jsonl}
AUX_MODEL=${AUX_MODEL:-/data/models/Qwen2.5-7B-Instruct}
VL_MODEL=${VL_MODEL:-${QWEN_VL_MODEL:-/data/models/Qwen2.5-VL-7B-Instruct}}
OUT_DIR=${OUT_DIR:-outputs}
MVBENCH_LIMIT=${MVBENCH_LIMIT:-0}

mkdir -p "$OUT_DIR"

python src/run_directionality_qwen.py \
  --dataset mvbench \
  --method direct \
  --data-path "$DATA_PATH" \
  --vl-model "$VL_MODEL" \
  --limit "$MVBENCH_LIMIT" \
  --output "$OUT_DIR/mvbench_directionality_qwen_base.jsonl"

python src/run_directionality_qwen.py \
  --dataset mvbench \
  --method tdcot \
  --data-path "$DATA_PATH" \
  --aux-model "$AUX_MODEL" \
  --vl-model "$VL_MODEL" \
  --limit "$MVBENCH_LIMIT" \
  --output "$OUT_DIR/mvbench_directionality_qwen_tdcot.jsonl"
