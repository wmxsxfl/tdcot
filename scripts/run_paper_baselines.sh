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
TEMPCOMPASS_LIMIT=${TEMPCOMPASS_LIMIT:-0}
MVBENCH_LIMIT=${MVBENCH_LIMIT:-0}

mkdir -p "$OUT_DIR"

for METHOD in generic_cot auxiliary_cot; do
  python src/run_directionality_qwen.py \
    --dataset tempcompass \
    --method "$METHOD" \
    --data-path "$TEMPCOMPASS_SETTING" \
    --video-root "$TEMPCOMPASS_VIDEO_ROOT" \
    --aux-model "$AUX_MODEL" \
    --vl-model "$QWEN_VL_MODEL" \
    --limit "$TEMPCOMPASS_LIMIT" \
    --output "$OUT_DIR/tempcompass_directionality_qwen_${METHOD}.jsonl"

  python src/run_directionality_qwen.py \
    --dataset mvbench \
    --method "$METHOD" \
    --data-path "$MVBENCH_SETTING" \
    --aux-model "$AUX_MODEL" \
    --vl-model "$QWEN_VL_MODEL" \
    --limit "$MVBENCH_LIMIT" \
    --output "$OUT_DIR/mvbench_directionality_qwen_${METHOD}.jsonl"
done

for METHOD in generic_cot auxiliary_cot localize_answer tsp; do
  python src/run_timeblind_prompted_internvl.py \
    --method "$METHOD" \
    --data-path "$TIMEBLIND_DATA" \
    --dataset-root "$DATASET_ROOT" \
    --model-path "$INTERNVL_MODEL" \
    --aux-model "$AUX_MODEL" \
    --limit "$TIMEBLIND_LIMIT" \
    --output "$OUT_DIR/timeblind_internvl_${METHOD}.jsonl"

  python src/run_directionality_internvl.py \
    --dataset tempcompass \
    --method "$METHOD" \
    --data-path "$TEMPCOMPASS_SETTING" \
    --video-root "$TEMPCOMPASS_VIDEO_ROOT" \
    --aux-model "$AUX_MODEL" \
    --vl-model "$INTERNVL_MODEL" \
    --limit "$TEMPCOMPASS_LIMIT" \
    --output "$OUT_DIR/tempcompass_directionality_internvl_${METHOD}.jsonl"

  python src/run_directionality_internvl.py \
    --dataset mvbench \
    --method "$METHOD" \
    --data-path "$MVBENCH_SETTING" \
    --aux-model "$AUX_MODEL" \
    --vl-model "$INTERNVL_MODEL" \
    --limit "$MVBENCH_LIMIT" \
    --output "$OUT_DIR/mvbench_directionality_internvl_${METHOD}.jsonl"
done
