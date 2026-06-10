#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${DATASET_ROOT:-/data/datasets}
TIMEBLIND_DATA=${TIMEBLIND_DATA:-$DATASET_ROOT/TimeBlind/data.jsonl}
AUX_MODEL=${AUX_MODEL:-/data/models/Qwen2.5-7B-Instruct}
QWEN_VL_MODEL=${QWEN_VL_MODEL:-/data/models/Qwen2.5-VL-7B-Instruct}
INTERNVL_MODEL=${INTERNVL_MODEL:-/data/models/InternVL2-8B}
OUT_DIR=${OUT_DIR:-outputs}
LATENCY_LIMIT=${LATENCY_LIMIT:-100}
LATENCY_LIMIT_PAIRS=${LATENCY_LIMIT_PAIRS:-50}

mkdir -p "$OUT_DIR/latency"
LATENCY_CSV="$OUT_DIR/latency/latency.csv"
printf "method,queries,seconds,ms_per_query\n" > "$LATENCY_CSV"

measure() {
  local method="$1"
  local queries="$2"
  shift 2
  local start
  local end
  start=$(python -c "import time; print(time.time())")
  "$@"
  end=$(python -c "import time; print(time.time())")
  python - "$LATENCY_CSV" "$method" "$queries" "$start" "$end" <<'PY'
import sys

path, method, queries, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
seconds = end - start
ms = 1000.0 * seconds / max(queries, 1)
with open(path, "a", encoding="utf-8") as f:
    f.write(f"{method},{queries},{seconds:.6f},{ms:.3f}\n")
PY
}

measure qwen_base "$LATENCY_LIMIT" \
  python src/run_timeblind_direct_qwen.py \
    --data-path "$TIMEBLIND_DATA" \
    --dataset-root "$DATASET_ROOT" \
    --model-path "$QWEN_VL_MODEL" \
    --limit "$LATENCY_LIMIT" \
    --output "$OUT_DIR/latency/timeblind_qwen_base_latency.jsonl"

measure internvl_base "$LATENCY_LIMIT" \
  python src/run_timeblind_direct_internvl.py \
    --data-path "$TIMEBLIND_DATA" \
    --dataset-root "$DATASET_ROOT" \
    --model-path "$INTERNVL_MODEL" \
    --limit "$LATENCY_LIMIT" \
    --output "$OUT_DIR/latency/timeblind_internvl_base_latency.jsonl"

measure qwen_generic_cot "$LATENCY_LIMIT" \
  python src/run_timeblind_generic_cot_qwen.py \
    --data-path "$TIMEBLIND_DATA" \
    --data-root "$DATASET_ROOT" \
    --model-path "$QWEN_VL_MODEL" \
    --limit "$LATENCY_LIMIT" \
    --output "$OUT_DIR/latency/timeblind_qwen_generic_cot_latency.jsonl"

measure qwen_auxiliary_cot "$LATENCY_LIMIT" \
  python src/run_timeblind_aux_cot_qwen.py \
    --data-path "$TIMEBLIND_DATA" \
    --data-root "$DATASET_ROOT" \
    --aux-model-path "$AUX_MODEL" \
    --vl-model-path "$QWEN_VL_MODEL" \
    --limit "$LATENCY_LIMIT" \
    --output "$OUT_DIR/latency/timeblind_qwen_auxiliary_cot_latency.jsonl"

for METHOD in generic_cot auxiliary_cot localize_answer tsp; do
  measure "internvl_${METHOD}" "$LATENCY_LIMIT" \
    python src/run_timeblind_prompted_internvl.py \
      --method "$METHOD" \
      --data-path "$TIMEBLIND_DATA" \
      --dataset-root "$DATASET_ROOT" \
      --model-path "$INTERNVL_MODEL" \
      --aux-model "$AUX_MODEL" \
      --limit "$LATENCY_LIMIT" \
      --output "$OUT_DIR/latency/timeblind_internvl_${METHOD}_latency.jsonl"
done

measure qwen_tdcot "$((LATENCY_LIMIT_PAIRS * 2))" \
  python src/run_timeblind_tdcot_pair_qwen.py \
    --data-path "$TIMEBLIND_DATA" \
    --dataset-root "$DATASET_ROOT" \
    --aux-model "$AUX_MODEL" \
    --vl-model "$QWEN_VL_MODEL" \
    --limit-pairs "$LATENCY_LIMIT_PAIRS" \
    --output "$OUT_DIR/latency/timeblind_qwen_tdcot_latency.jsonl"

measure internvl_tdcot "$((LATENCY_LIMIT_PAIRS * 2))" \
  python src/run_timeblind_tdcot_pair_internvl.py \
    --data-path "$TIMEBLIND_DATA" \
    --dataset-root "$DATASET_ROOT" \
    --aux-model "$AUX_MODEL" \
    --vl-model "$INTERNVL_MODEL" \
    --limit-pairs "$LATENCY_LIMIT_PAIRS" \
    --output "$OUT_DIR/latency/timeblind_internvl_tdcot_latency.jsonl"

echo "saved to: $LATENCY_CSV"
