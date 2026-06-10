#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${DATASET_ROOT:-/data/datasets}
TIMEBLIND_DATA=${TIMEBLIND_DATA:-$DATASET_ROOT/TimeBlind/data.jsonl}
TEMPCOMPASS_SETTING=${TEMPCOMPASS_SETTING:-$DATASET_ROOT/TempCompass/tempcompass_directionality_items.parquet}
TEMPCOMPASS_VIDEO_ROOT=${TEMPCOMPASS_VIDEO_ROOT:-$DATASET_ROOT/TempCompass/videos/videos}
MVBENCH_SETTING=${MVBENCH_SETTING:-$DATASET_ROOT/MVBench/mvbench_directionality_setting.jsonl}

AUX_MODEL=${AUX_MODEL:-/data/models/Qwen2.5-7B-Instruct}
QWEN_VL_MODEL=${QWEN_VL_MODEL:-/data/models/Qwen2.5-VL-7B-Instruct}
INTERNVL_MODEL=${INTERNVL_MODEL:-/data/models/InternVL2-8B}
OUT_DIR=${OUT_DIR:-outputs}
TIMEBLIND_LIMIT=${TIMEBLIND_LIMIT:-2400}
TIMEBLIND_LIMIT_PAIRS=${TIMEBLIND_LIMIT_PAIRS:-1200}
TEMPCOMPASS_LIMIT=${TEMPCOMPASS_LIMIT:-0}
MVBENCH_LIMIT=${MVBENCH_LIMIT:-0}

mkdir -p "$OUT_DIR"

DATA_PATH="$TIMEBLIND_DATA" DATASET_ROOT="$DATASET_ROOT" AUX_MODEL="$AUX_MODEL" VL_MODEL="$QWEN_VL_MODEL" OUT_DIR="$OUT_DIR" \
  bash scripts/run_timeblind_qwen.sh

DATA_PATH="$TEMPCOMPASS_SETTING" VIDEO_ROOT="$TEMPCOMPASS_VIDEO_ROOT" AUX_MODEL="$AUX_MODEL" VL_MODEL="$QWEN_VL_MODEL" OUT_DIR="$OUT_DIR" \
  bash scripts/run_tempcompass_qwen.sh

DATA_PATH="$MVBENCH_SETTING" AUX_MODEL="$AUX_MODEL" VL_MODEL="$QWEN_VL_MODEL" OUT_DIR="$OUT_DIR" \
  bash scripts/run_mvbench_qwen.sh

python src/run_timeblind_direct_internvl.py \
  --data-path "$TIMEBLIND_DATA" \
  --dataset-root "$DATASET_ROOT" \
  --model-path "$INTERNVL_MODEL" \
  --limit "$TIMEBLIND_LIMIT" \
  --output "$OUT_DIR/timeblind_internvl_base.jsonl"

python src/run_timeblind_tdcot_pair_internvl.py \
  --data-path "$TIMEBLIND_DATA" \
  --dataset-root "$DATASET_ROOT" \
  --aux-model "$AUX_MODEL" \
  --vl-model "$INTERNVL_MODEL" \
  --limit-pairs "$TIMEBLIND_LIMIT_PAIRS" \
  --output "$OUT_DIR/timeblind_internvl_tdcot.jsonl"

python src/run_directionality_internvl.py \
  --dataset tempcompass \
  --method direct \
  --data-path "$TEMPCOMPASS_SETTING" \
  --video-root "$TEMPCOMPASS_VIDEO_ROOT" \
  --vl-model "$INTERNVL_MODEL" \
  --limit "$TEMPCOMPASS_LIMIT" \
  --output "$OUT_DIR/tempcompass_directionality_internvl_base.jsonl"

python src/run_directionality_internvl.py \
  --dataset tempcompass \
  --method tdcot \
  --data-path "$TEMPCOMPASS_SETTING" \
  --video-root "$TEMPCOMPASS_VIDEO_ROOT" \
  --aux-model "$AUX_MODEL" \
  --vl-model "$INTERNVL_MODEL" \
  --limit "$TEMPCOMPASS_LIMIT" \
  --output "$OUT_DIR/tempcompass_directionality_internvl_tdcot.jsonl"

python src/run_directionality_internvl.py \
  --dataset mvbench \
  --method direct \
  --data-path "$MVBENCH_SETTING" \
  --vl-model "$INTERNVL_MODEL" \
  --limit "$MVBENCH_LIMIT" \
  --output "$OUT_DIR/mvbench_directionality_internvl_base.jsonl"

python src/run_directionality_internvl.py \
  --dataset mvbench \
  --method tdcot \
  --data-path "$MVBENCH_SETTING" \
  --aux-model "$AUX_MODEL" \
  --vl-model "$INTERNVL_MODEL" \
  --limit "$MVBENCH_LIMIT" \
  --output "$OUT_DIR/mvbench_directionality_internvl_tdcot.jsonl"
