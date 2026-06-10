import argparse
import json
import re
from pathlib import Path

import torch
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
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


def load_rows(path, limit):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:limit] if limit is not None else rows


def generate_text_hint(question, tok, model, max_new_tokens=96):
    messages = [
        {
            "role": "system",
            "content": "You generate concise temporal localization hints for video QA. You do not see the video.",
        },
        {
            "role": "user",
            "content": (
                "Given only the question, write one concise localization hint describing where in the video "
                "a video model should look. Mention the relevant beginning, middle, or end portion if useful, "
                "and mention the key event boundary or temporal order to inspect. Do not answer yes or no.\n\n"
                f"Question: {question}\n\n"
                "Temporal localization hint:"
            ),
        },
    ]

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)

    bad_words_ids = []
    try:
        bad_words_ids = [tok.encode("!", add_special_tokens=False)]
    except Exception:
        pass

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            bad_words_ids=bad_words_ids if bad_words_ids and bad_words_ids[0] else None,
        )

    hint = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    hint = re.sub(r"\s+", " ", hint)
    return hint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aux-model-path", default="/data/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--vl-model-path", default="/data/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data-path", default="/data/datasets/TimeBlind/data.jsonl")
    ap.add_argument("--data-root", default="/data/datasets")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--nframes", type=int, default=16)
    ap.add_argument("--hint-max-new-tokens", type=int, default=96)
    ap.add_argument("--answer-max-new-tokens", type=int, default=32)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    rows = load_rows(args.data_path, args.limit)

    aux_tok = AutoTokenizer.from_pretrained(args.aux_model_path, trust_remote_code=True)
    aux_model = AutoModelForCausalLM.from_pretrained(
        args.aux_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vl_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(args.vl_model_path, trust_remote_code=True)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for x in tqdm(rows, desc="TimeBlind Localize-then-Answer"):
            video_path = resolve_video_path(args.data_root, x["video_path"])
            hint = generate_text_hint(
                x["question"],
                aux_tok,
                aux_model,
                max_new_tokens=args.hint_max_new_tokens,
            )

            prompt = f"""You are given a video and a temporal question.

Auxiliary reasoning hint generated from the question:
{hint}

Use the hint only as guidance. Judge the actual video carefully.

Question: {x["question"]}

Return exactly one word: yes or no."""

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
                ).to(vl_model.device)

                with torch.no_grad():
                    generated_ids = vl_model.generate(
                        **inputs,
                        max_new_tokens=args.answer_max_new_tokens,
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
                "hint": hint,
                "raw_prediction": raw,
                "prediction": pred,
                "correct": ok,
            }, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"accuracy: {correct}/{total} = {correct/total if total else 0:.4f}")
    print(f"saved to: {out_path}")


if __name__ == "__main__":
    main()
