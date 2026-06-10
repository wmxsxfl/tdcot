# Data Preparation

Place datasets and model weights on local storage, then point the scripts to them with environment variables or command-line arguments.

## Models

Example layout:

```text
/data/models/
  Qwen2.5-7B-Instruct/
  Qwen2.5-3B-Instruct/
  Qwen2.5-VL-7B-Instruct/
  InternVL2-8B/
```

Environment variables:

```bash
export AUX_MODEL=/data/models/Qwen2.5-7B-Instruct
export AUX_MODEL_3B=/data/models/Qwen2.5-3B-Instruct
export QWEN_VL_MODEL=/data/models/Qwen2.5-VL-7B-Instruct
export INTERNVL_MODEL=/data/models/InternVL2-8B
```

## TimeBlind

Expected layout:

```text
/data/datasets/
  TimeBlind/
    data.jsonl
    videos/
      vid_00000_0.mp4
      vid_00000_1.mp4
```

Expected JSONL fields:

- `index`
- `video_path`
- `question`
- `answer`

`video_path` may be relative to `DATASET_ROOT`. The paper setting uses 1,200 pairwise directionality judgments, represented as 2,400 item rows.

## TempCompass

Expected layout:

```text
/data/datasets/
  TempCompass/
    tempcompass_combined.parquet
    videos/
      videos/
        1034419625.mp4
        1034419625_reverse.mp4
```

The combined parquet must contain:

- `uuid`
- `video_id`
- `question`
- `options`
- `answer`

Build the forward/reverse directionality setting:

```bash
python src/build_tempcompass_reverse_pairs.py \
  --input /data/datasets/TempCompass/tempcompass_combined.parquet \
  --video-root /data/datasets/TempCompass/videos/videos \
  --output-parquet /data/datasets/TempCompass/tempcompass_directionality_items.parquet \
  --output-jsonl /data/datasets/TempCompass/tempcompass_directionality_pairs.jsonl
```

The paper setting contains 1,500 paired comparisons. To enforce that count:

```bash
python src/build_tempcompass_reverse_pairs.py \
  --input /data/datasets/TempCompass/tempcompass_combined.parquet \
  --video-root /data/datasets/TempCompass/videos/videos \
  --output-parquet /data/datasets/TempCompass/tempcompass_directionality_items.parquet \
  --output-jsonl /data/datasets/TempCompass/tempcompass_directionality_pairs.jsonl \
  --strict-count
```

## MVBench

The MVBench directionality source JSONL should contain:

- `id`
- `task`
- `video`
- `video_path`
- `question`
- `candidates`
- `answer`

Build the paper directionality setting:

```bash
python src/build_mvbench_directionality_setting.py \
  --input /data/datasets/MVBench/mvbench_directional.jsonl \
  --output /data/datasets/MVBench/mvbench_directionality_setting.jsonl
```

## General Video QA

The paper reports general video QA checks on:

- VideoMME short split without subtitles
- VideoMME long split without subtitles
- MSRVTT-QA

The general QA launch script delegates to LMMs-Eval:

```bash
export VIDEOMME_SHORT_TASK=videomme_short_wo_subtitle
export VIDEOMME_LONG_TASK=videomme_long_wo_subtitle
export MSRVTT_TASK=msrvtt_qa
bash scripts/run_paper_general_videoqa.sh
```

## Path Configuration

`configs/paths.example.yaml` lists the path variables used by the launch scripts.
