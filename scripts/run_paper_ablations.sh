#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${DATASET_ROOT:-/data/datasets}
TIMEBLIND_DATA=${TIMEBLIND_DATA:-$DATASET_ROOT/TimeBlind/data.jsonl}
TEMPCOMPASS_SETTING=${TEMPCOMPASS_SETTING:-$DATASET_ROOT/TempCompass/tempcompass_directionality_items.parquet}
TEMPCOMPASS_VIDEO_ROOT=${TEMPCOMPASS_VIDEO_ROOT:-$DATASET_ROOT/TempCompass/videos/videos}
MVBENCH_SETTING=${MVBENCH_SETTING:-$DATASET_ROOT/MVBench/mvbench_directionality_setting.jsonl}

AUX_MODEL=${AUX_MODEL:-/data/models/Qwen2.5-7B-Instruct}
AUX_MODEL_3B=${AUX_MODEL_3B:-/data/models/Qwen2.5-3B-Instruct}
INTERNVL_MODEL=${INTERNVL_MODEL:-/data/models/InternVL2-8B}
OUT_DIR=${OUT_DIR:-outputs}
TIMEBLIND_LIMIT=${TIMEBLIND_LIMIT:-2400}
TIMEBLIND_LIMIT_PAIRS=${TIMEBLIND_LIMIT_PAIRS:-1200}
TEMPCOMPASS_LIMIT=${TEMPCOMPASS_LIMIT:-0}
MVBENCH_LIMIT=${MVBENCH_LIMIT:-0}

mkdir -p "$OUT_DIR"

run_directionality_variant() {
  local dataset="$1"
  local data_path="$2"
  local video_root="$3"
  local method="$4"
  local out_name="$5"
  local aux_model="$6"

  local video_root_args=()
  if [[ -n "$video_root" ]]; then
    video_root_args=(--video-root "$video_root")
  fi
  local limit="$TEMPCOMPASS_LIMIT"
  if [[ "$dataset" == "mvbench" ]]; then
    limit="$MVBENCH_LIMIT"
  fi

  python src/run_directionality_internvl.py \
    --dataset "$dataset" \
    --method "$method" \
    --data-path "$data_path" \
    "${video_root_args[@]}" \
    --aux-model "$aux_model" \
    --vl-model "$INTERNVL_MODEL" \
    --limit "$limit" \
    --output "$OUT_DIR/${out_name}.jsonl"
}

run_timeblind_variant() {
  local variant="$1"
  local out_name="$2"
  local aux_model="$3"
  python src/run_timeblind_tdcot_pair_internvl.py \
    --data-path "$TIMEBLIND_DATA" \
    --dataset-root "$DATASET_ROOT" \
    --aux-model "$aux_model" \
    --vl-model "$INTERNVL_MODEL" \
    --limit-pairs "$TIMEBLIND_LIMIT_PAIRS" \
    --variant "$variant" \
    --output "$OUT_DIR/${out_name}.jsonl"
}

for METHOD in tdcot no_seg state2 state5 no_routing joint no_cond no_early_exit; do
  run_directionality_variant mvbench "$MVBENCH_SETTING" "" "$METHOD" "mvbench_directionality_internvl_${METHOD}" "$AUX_MODEL"
  run_directionality_variant tempcompass "$TEMPCOMPASS_SETTING" "$TEMPCOMPASS_VIDEO_ROOT" "$METHOD" "tempcompass_directionality_internvl_${METHOD}" "$AUX_MODEL"
  run_timeblind_variant "$METHOD" "timeblind_internvl_${METHOD}" "$AUX_MODEL"
done

run_directionality_variant mvbench "$MVBENCH_SETTING" "" freeform_cot_matched "mvbench_directionality_internvl_freeform_cot_matched" "$AUX_MODEL"
run_directionality_variant tempcompass "$TEMPCOMPASS_SETTING" "$TEMPCOMPASS_VIDEO_ROOT" freeform_cot_matched "tempcompass_directionality_internvl_freeform_cot_matched" "$AUX_MODEL"
python src/run_timeblind_prompted_internvl.py \
  --method freeform_cot_matched \
  --data-path "$TIMEBLIND_DATA" \
  --dataset-root "$DATASET_ROOT" \
  --model-path "$INTERNVL_MODEL" \
  --aux-model "$AUX_MODEL" \
  --limit "$TIMEBLIND_LIMIT" \
  --output "$OUT_DIR/timeblind_internvl_freeform_cot_matched.jsonl"

run_directionality_variant mvbench "$MVBENCH_SETTING" "" tdcot "mvbench_directionality_internvl_aux3b" "$AUX_MODEL_3B"
run_directionality_variant tempcompass "$TEMPCOMPASS_SETTING" "$TEMPCOMPASS_VIDEO_ROOT" tdcot "tempcompass_directionality_internvl_aux3b" "$AUX_MODEL_3B"
run_timeblind_variant tdcot "timeblind_internvl_aux3b" "$AUX_MODEL_3B"
