import argparse
import json
import re
from pathlib import Path

import torch
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def safe_nframes(video_path, requested):
    try:
        total = len(VideoReader(str(video_path), ctx=cpu(0), num_threads=1))
        return max(2, min(requested, total))
    except Exception:
        return requested


def normalize_yesno(text):
    hits = re.findall(r"\b(yes|no)\b", str(text).lower())
    return hits[-1] if hits else "unknown"


def resolve_video_path(data_root, video_path):
    p = Path(video_path)
    if p.is_absolute():
        return p
    return Path(data_root) / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/data/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data-path", default="/data/datasets/TimeBlind/data.jsonl")
    ap.add_argument("--data-root", default="/data/datasets")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--nframes", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.data_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit is not None:
        rows = rows[:args.limit]

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for x in tqdm(rows, desc="TimeBlind generic CoT"):
            video_path = resolve_video_path(args.data_root, x["video_path"])

            prompt = f"""You are solving a temporal video question.
First write one short sentence explaining the temporal evidence you used.
Then write the final answer.

Question: {x["question"]}

Output format:
Reasoning: <one short sentence>
Answer: yes or no"""

            if not video_path.exists():
                raw = "MISSING_VIDEO"
                pred = "unknown"
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": str(video_path),
                                "nframes": safe_nframes(video_path, args.nframes),
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]

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
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )

                generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
                raw = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0].strip()
                pred = normalize_yesno(raw)

            gold = str(x["answer"]).lower()
            ok = pred == gold
            total += 1
            correct += int(ok)

            fout.write(json.dumps({
                "index": x["index"],
                "video_path": x["video_path"],
                "question": x["question"],
                "answer": gold,
                "raw_prediction": raw,
                "prediction": pred,
                "correct": ok,
            }, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"accuracy: {correct}/{total} = {correct/total if total else 0:.4f}")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
