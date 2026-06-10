# TD-CoT

Official code for **TD-CoT: Bridging the Holistic-Atomic Gap for Training-Free Temporal Reversal Detection in Video LVLMs**.

TD-CoT is a training-free inference framework for directionality-sensitive video reasoning. It routes temporal questions to a structured path, decomposes each routed query into ordered visual checkpoints, verifies the checkpoints on aligned temporal frame segments, and scores candidate answers with Yes/No first-token logits.

This codebase is intended to support reproduction of the TD-CoT inference pipeline. It includes the core routing, decomposition, segmented verification, scoring, baseline, ablation, and table aggregation scripts.

Due to dataset licenses and file sizes, the repository does not redistribute benchmark videos, pretrained model weights, or generated output files. Users should obtain TimeBlind, TempCompass, MVBench, VideoMME, MSRVTT-QA, Qwen2.5, Qwen2.5-VL, and InternVL2 from their official sources.

## Components

- Text-only temporal/general router
- Temporal checkpoint decomposition
- Sequential segmented visual verification
- Yes/No logit scoring with product aggregation
- Early-exit verification with the paper threshold `theta_e = 0.3`
- Direct, Generic CoT, Auxiliary-CoT, Localize-then-Answer, and TSP baselines
- TimeBlind, TempCompass, MVBench, VideoMME, and MSRVTT experiment entry points
- Table aggregation, routing analysis, significance utilities, and status auditing

## Repository Structure

```text
tdcot/
  src/        Dataset builders, model runners, scoring, aggregation, and audits
  scripts/    End-to-end experiment launchers
  configs/    Experiment manifest and path examples
  docs/       Data, experiment, diagnostics, FlashAttention2, and troubleshooting notes
```

## Setup

```bash
conda create -n tdcot python=3.10
conda activate tdcot
pip install -r requirements.txt
```

The paper experiments were run with PyTorch 2.5, Transformers 4.51, CUDA 12.x, and a single NVIDIA A100 80GB GPU. FlashAttention2 is optional for memory-heavy full-resolution verification; see `docs/FLASH_ATTN.md`.

## Models

Set local model paths before running experiments:

```bash
export AUX_MODEL=/data/models/Qwen2.5-7B-Instruct
export AUX_MODEL_3B=/data/models/Qwen2.5-3B-Instruct
export QWEN_VL_MODEL=/data/models/Qwen2.5-VL-7B-Instruct
export INTERNVL_MODEL=/data/models/InternVL2-8B
```

## Data

Download datasets from their official sources and prepare the paper settings with:

```bash
bash scripts/build_paper_settings.sh
```

The launchers expect:

- TimeBlind minimal-pair JSONL and videos
- TempCompass combined parquet and forward/reverse videos
- MVBench directionality source JSONL and videos
- VideoMME short/long splits without subtitles
- MSRVTT-QA

See `docs/DATA.md` for schemas and path variables.

## Running Experiments

Run all configured experiment stages:

```bash
bash scripts/run_all_experiments.sh
```

Or run stages individually:

```bash
bash scripts/run_paper_directionality.sh
bash scripts/run_paper_baselines.sh
bash scripts/run_timeblind_baselines_qwen.sh
bash scripts/run_paper_ablations.sh
bash scripts/run_paper_general_videoqa.sh
bash scripts/run_router_and_statistics.sh
bash scripts/run_latency_measurement.sh
```

Generate computed tables from result files:

```bash
python src/summarize_paper_tables.py
python src/audit_experiment_status.py
```

Print the paper-reported reference values:

```bash
python src/summarize_paper_tables.py --table reference
```

`summarize_paper_tables.py` computes values from `outputs/`. Use `--strict` to require all expected output files before table generation.

## Outputs

Prediction files, frame caches, datasets, model weights, and downloaded wheels are ignored by Git. Standard output groups are:

- `outputs/*_qwen_*.jsonl`
- `outputs/*_internvl_*.jsonl`
- `outputs/general_videoqa/`
- `outputs/router_*.jsonl`
- `outputs/*_eval.md`

## Citation

If you use this project, please cite the paper. See `CITATION.cff`.
