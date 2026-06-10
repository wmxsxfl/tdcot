#!/usr/bin/env bash
set -euo pipefail

bash scripts/build_paper_settings.sh
bash scripts/run_paper_directionality.sh
bash scripts/run_paper_baselines.sh
python src/eval_tempcompass_directionality.py > outputs/tempcompass_directionality_eval.md
bash scripts/run_timeblind_baselines_qwen.sh
bash scripts/run_paper_ablations.sh
bash scripts/run_paper_general_videoqa.sh
bash scripts/run_router_and_statistics.sh
bash scripts/run_latency_measurement.sh

python src/summarize_paper_tables.py
python src/audit_experiment_status.py
