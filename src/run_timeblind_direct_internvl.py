import argparse, json, re
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def normalize_yesno(text):
    t = text.strip().lower()
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return "unknown"

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def load_video(video_path, nframes=16, input_size=448):
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    n = max(2, min(nframes, len(vr)))
    idx = np.linspace(0, len(vr) - 1, n).astype(int)
    transform = build_transform(input_size)

    pixels = []
    for i in idx:
        img = Image.fromarray(vr[i].asnumpy()).convert("RGB")
        pixels.append(transform(img).unsqueeze(0))
    return torch.cat(pixels, dim=0), [1] * len(pixels)

def load_items(path, limit=None):
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items[:limit] if limit else items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/data/models/InternVL2-8B")
    ap.add_argument("--data-path", default="/data/datasets/TimeBlind/data.jsonl")
    ap.add_argument("--dataset-root", default="/data/datasets")
    ap.add_argument("--output", default="outputs/timeblind_internvl_base.jsonl")
    ap.add_argument("--limit", type=int, default=2400)
    ap.add_argument("--nframes", type=int, default=16)
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    items = load_items(args.data_path, args.limit)
    correct = total = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for item in tqdm(items, desc="InternVL2 direct"):
            video_path = Path(args.dataset_root) / item["video_path"]
            pixel_values, num_patches_list = load_video(video_path, args.nframes)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
            question = prefix + item["question"] + "\nAnswer exactly one word: yes or no."

            raw = model.chat(
                tokenizer,
                pixel_values,
                question,
                dict(max_new_tokens=8, do_sample=False),
                num_patches_list=num_patches_list,
                history=None,
                return_history=False,
            )

            pred = normalize_yesno(raw)
            gold = item["answer"].strip().lower()
            ok = pred == gold
            correct += int(ok)
            total += 1

            fout.write(json.dumps({
                "index": item["index"],
                "video_path": item["video_path"],
                "question": item["question"],
                "answer": gold,
                "raw_prediction": raw,
                "prediction": pred,
                "correct": ok,
            }, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"accuracy: {correct}/{total} = {correct/total:.4f}")
    print("saved to:", args.output)

if __name__ == "__main__":
    main()
