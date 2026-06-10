#!/usr/bin/env bash
set -euo pipefail

QWEN_VL_MODEL=${QWEN_VL_MODEL:-/data/models/Qwen2.5-VL-7B-Instruct}
INTERNVL_MODEL=${INTERNVL_MODEL:-/data/models/InternVL2-8B}
AUX_MODEL=${AUX_MODEL:-/data/models/Qwen2.5-7B-Instruct}
ROUTER_THRESHOLD=${ROUTER_THRESHOLD:-0.8}
EARLY_EXIT=${EARLY_EXIT:-0.3}

VIDEOMME_SHORT_TASK=${VIDEOMME_SHORT_TASK:-videomme_short_wo_subtitle}
VIDEOMME_LONG_TASK=${VIDEOMME_LONG_TASK:-videomme_long_wo_subtitle}
MSRVTT_TASK=${MSRVTT_TASK:-msrvtt_qa}
TASKS="$VIDEOMME_SHORT_TASK,$VIDEOMME_LONG_TASK,$MSRVTT_TASK"

OUT_DIR=${OUT_DIR:-outputs/general_videoqa}
mkdir -p "$OUT_DIR"

run_lmms() {
  local model_name="$1"
  local model_path="$2"
  local method="$3"
  local output_name="$4"

  local method_args=()
  if [[ -n "$method" ]]; then
    method_args=(
      --method "$method"
      --aux-model "$AUX_MODEL"
      --router-threshold "$ROUTER_THRESHOLD"
      --early-exit "$EARLY_EXIT"
    )
  fi

  python src/run_lmms_eval_suite.py \
    --model-name "$model_name" \
    --model-path "$model_path" \
    --tasks "$TASKS" \
    --output-dir "$OUT_DIR/$output_name" \
    "${method_args[@]}"
}

run_lmms qwen2_5_vl "$QWEN_VL_MODEL" "" qwen2_5_vl
run_lmms qwen2_5_vl "$QWEN_VL_MODEL" generic_cot qwen2_5_vl_generic_cot
run_lmms qwen2_5_vl "$QWEN_VL_MODEL" auxiliary_cot qwen2_5_vl_auxiliary_cot
run_lmms qwen2_5_vl "$QWEN_VL_MODEL" tdcot qwen2_5_vl_tdcot

run_lmms internvl2 "$INTERNVL_MODEL" "" internvl2
run_lmms internvl2 "$INTERNVL_MODEL" generic_cot internvl2_generic_cot
run_lmms internvl2 "$INTERNVL_MODEL" auxiliary_cot internvl2_auxiliary_cot
run_lmms internvl2 "$INTERNVL_MODEL" localize_answer internvl2_localize_answer
run_lmms internvl2 "$INTERNVL_MODEL" tsp internvl2_tsp
run_lmms internvl2 "$INTERNVL_MODEL" tdcot internvl2_tdcot
