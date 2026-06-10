#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${DATASET_ROOT:-/data/datasets}
TIMEBLIND_DATA=${TIMEBLIND_DATA:-$DATASET_ROOT/TimeBlind/data.jsonl}
AUX_MODEL=${AUX_MODEL:-/data/models/Qwen2.5-7B-Instruct}
OUT_DIR=${OUT_DIR:-outputs}

mkdir -p "$OUT_DIR"

python src/build_router_eval_set.py \
  --timeblind-data "$TIMEBLIND_DATA" \
  --output "$OUT_DIR/router_eval_set.jsonl"

python src/router_qwen.py \
  --model-path "$AUX_MODEL" \
  --input "$OUT_DIR/router_eval_set.jsonl" \
  --output "$OUT_DIR/router_eval_qwen.jsonl" \
  --threshold 0.8

python src/analyze_router_thresholds.py > "$OUT_DIR/router_threshold_sweep.md"
python src/eval_timeblind_routed_pipeline.py > "$OUT_DIR/timeblind_routed_pipeline.md"

python src/bootstrap_timeblind.py \
  --direct "$OUT_DIR/timeblind_qwen_base.jsonl" \
  --tdcot "$OUT_DIR/timeblind_qwen_tdcot.jsonl" \
  --bootstrap 1000 \
  --permutation 1000 \
  --seed 42 > "$OUT_DIR/timeblind_bootstrap.md"
