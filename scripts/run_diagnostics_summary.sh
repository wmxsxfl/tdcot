#!/usr/bin/env bash
set -euo pipefail

DIAGNOSTIC_DIR=${DIAGNOSTIC_DIR:-data/diagnostics}
OUT_DIR=${OUT_DIR:-outputs/diagnostics}

mkdir -p "$OUT_DIR"

python src/summarize_diagnostics.py \
  --decomposition "$DIAGNOSTIC_DIR/decomposition_quality.jsonl" \
  --holistic-atomic "$DIAGNOSTIC_DIR/holistic_atomic_gap.jsonl" \
  --failures "$DIAGNOSTIC_DIR/failure_taxonomy.jsonl" \
  --output "$OUT_DIR/diagnostic_summary.md"
