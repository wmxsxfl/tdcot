#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${DATASET_ROOT:-/data/datasets}
TIMEBLIND_DATA=${TIMEBLIND_DATA:-$DATASET_ROOT/TimeBlind/data.jsonl}

TEMPCOMPASS_PARQUET=${TEMPCOMPASS_PARQUET:-$DATASET_ROOT/TempCompass/tempcompass_combined.parquet}
TEMPCOMPASS_VIDEO_ROOT=${TEMPCOMPASS_VIDEO_ROOT:-$DATASET_ROOT/TempCompass/videos/videos}
TEMPCOMPASS_SETTING=${TEMPCOMPASS_SETTING:-$DATASET_ROOT/TempCompass/tempcompass_directionality_items.parquet}
TEMPCOMPASS_JSONL=${TEMPCOMPASS_JSONL:-$DATASET_ROOT/TempCompass/tempcompass_directionality_pairs.jsonl}

MVBENCH_DIRECTIONAL=${MVBENCH_DIRECTIONAL:-$DATASET_ROOT/MVBench/mvbench_directional.jsonl}
MVBENCH_SETTING=${MVBENCH_SETTING:-$DATASET_ROOT/MVBench/mvbench_directionality_setting.jsonl}

OUT_DIR=${OUT_DIR:-outputs}
mkdir -p "$OUT_DIR"

python src/build_tempcompass_reverse_pairs.py \
  --input "$TEMPCOMPASS_PARQUET" \
  --video-root "$TEMPCOMPASS_VIDEO_ROOT" \
  --output-parquet "$TEMPCOMPASS_SETTING" \
  --output-jsonl "$TEMPCOMPASS_JSONL"

python src/build_mvbench_directionality_setting.py \
  --input "$MVBENCH_DIRECTIONAL" \
  --output "$MVBENCH_SETTING"

python src/build_router_eval_set.py \
  --timeblind-data "$TIMEBLIND_DATA" \
  --output "$OUT_DIR/router_eval_set.jsonl"
