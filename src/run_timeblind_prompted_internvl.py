import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

METHODS = [
    "direct",
    "generic_cot",
    "auxiliary_cot",
    "localize_answer",
    "tsp",
    "freeform_cot_matched",
]


def normalize_yesno(text):
    hits = re.findall(r"\b(yes|no)\b", str(text).lower())
    return hits[-1] if hits else "unknown"


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


def generate_text_hint(question, tok, model, method, max_new_tokens=96):
    if method == "auxiliary_cot":
        instruction = "Write a concise temporal reasoning plan for answering this yes/no video question. Do not answer yes or no."
    elif method == "localize_answer":
        instruction = "Write a concise temporal localization hint describing where to inspect the video. Do not answer yes or no."
    else:
        instruction = "Write a concise temporal sensitivity hint about order, direction, speed, or state change. Do not answer yes or no."
    messages = [
        {"role": "system", "content": "You provide concise temporal guidance for video question answering."},
        {"role": "user", "content": f"{instruction}\n\nQuestion: {question}\n\nHint:"},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    hint = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return re.sub(r"\s+", " ", hint)


def answer_question(model, tokenizer, pixel_values, num_patches_list, question, method, hint=None):
    prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
    if method == "direct":
        prompt = prefix + question + "\nAnswer exactly one word: yes or no."
        max_new_tokens = 8
    elif method == "generic_cot":
        prompt = prefix + f"""Question:
{question}

Reason about the temporal evidence, then give the final answer as exactly one word: yes or no."""
        max_new_tokens = 96
    elif method == "freeform_cot_matched":
        prompt = prefix + f"""Question:
{question}

Use a concise free-form temporal chain of thought with comparable reasoning budget, then answer exactly one word: yes or no."""
        max_new_tokens = 140
    else:
        prompt = prefix + f"""Question:
{question}

Temporal hint:
{hint}

Answer exactly one word: yes or no."""
        max_new_tokens = 32

    return model.chat(
        tokenizer,
        pixel_values,
        prompt,
        dict(max_new_tokens=max_new_tokens, do_sample=False),
        num_patches_list=num_patches_list,
        history=None,
        return_history=False,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--model-path", default="/data/models/InternVL2-8B")
    ap.add_argument("--aux-model", default="/data/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--data-path", default="/data/datasets/TimeBlind/data.jsonl")
    ap.add_argument("--dataset-root", default="/data/datasets")
    ap.add_argument("--output", required=True)
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)

    needs_aux = args.method in {"auxiliary_cot", "localize_answer", "tsp"}
    aux_tok = aux_model = None
    if needs_aux:
        aux_tok = AutoTokenizer.from_pretrained(args.aux_model, trust_remote_code=True)
        aux_model = AutoModelForCausalLM.from_pretrained(
            args.aux_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    correct = total = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for item in tqdm(load_items(args.data_path, args.limit), desc=f"TimeBlind InternVL2 {args.method}"):
            video_path = Path(args.dataset_root) / item["video_path"]
            pixel_values, num_patches_list = load_video(video_path, args.nframes)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            hint = None
            if needs_aux:
                hint = generate_text_hint(item["question"], aux_tok, aux_model, args.method)
            raw = answer_question(model, tokenizer, pixel_values, num_patches_list, item["question"], args.method, hint)
            pred = normalize_yesno(raw)
            gold = item["answer"].strip().lower()
            ok = pred == gold
            correct += int(ok)
            total += 1

            record = {
                "index": item["index"],
                "video_path": item["video_path"],
                "question": item["question"],
                "answer": gold,
                "raw_prediction": raw,
                "prediction": pred,
                "correct": ok,
                "method": args.method,
            }
            if hint is not None:
                record["hint"] = hint
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"accuracy: {correct}/{total} = {correct / total if total else 0:.4f}")
    print("saved to:", args.output)


if __name__ == "__main__":
    main()
