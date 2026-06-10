import argparse
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from decord import VideoReader, cpu


def normalize_yesno(text: str):
    t = text.strip().lower()
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return "unknown"


def load_items(path, limit=None):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    if limit is not None:
        items = items[:limit]
    return items



def safe_nframes(video_path, requested):
    try:
        total = len(VideoReader(str(video_path), ctx=cpu(0), num_threads=1))
        return max(2, min(requested, total))
    except Exception:
        return requested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/data/models/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--data-path", default="/data/datasets/TimeBlind/data.jsonl")
    parser.add_argument("--dataset-root", default="/data/datasets")
    parser.add_argument("--output", default="outputs/timeblind_qwen_base.jsonl")
    parser.add_argument("--limit", type=int, default=2400)
    parser.add_argument("--nframes", type=int, default=16)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)

    items = load_items(args.data_path, args.limit)

    correct = 0
    total = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for item in tqdm(items, desc="TimeBlind direct"):
            video_path = Path(args.dataset_root) / item["video_path"]

            prompt = (
                f"{item['question']}\n"
                "Answer exactly one word: yes or no."
            )

            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path), "nframes": safe_nframes(video_path, args.nframes)},
                    {"type": "text", "text": prompt},
                ],
            }]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=16,
                    do_sample=False,
                )

            generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            pred = normalize_yesno(raw)
            gold = item["answer"].strip().lower()
            ok = pred == gold

            total += 1
            correct += int(ok)

            record = {
                "index": item["index"],
                "video_path": item["video_path"],
                "question": item["question"],
                "answer": gold,
                "raw_prediction": raw,
                "prediction": pred,
                "correct": ok,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"accuracy: {correct}/{total} = {correct / total:.4f}")
    print(f"saved to: {args.output}")


if __name__ == "__main__":
    main()
