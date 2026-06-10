import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


TD_METHODS = {
    "tdcot",
    "no_seg",
    "joint",
    "no_cond",
    "state2",
    "state5",
    "no_early_exit",
    "no_routing",
}

PROMPT_METHODS = {
    "direct",
    "generic_cot",
    "auxiliary_cot",
    "localize_answer",
    "tsp",
    "freeform_cot_matched",
}


def to_jsonable(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    return x


def option_letter(opt, idx):
    m = re.match(r"\s*([A-E])\s*[\.\)]", str(opt), flags=re.I)
    return m.group(1).upper() if m else chr(ord("A") + idx)


def option_text(opt):
    return re.sub(r"^\s*[A-E]\s*[\.\)]\s*", "", str(opt), flags=re.I).strip()


def parse_letter(text, n_options):
    m = re.search(r"\b([A-E])\b", str(text), flags=re.I)
    if not m:
        m = re.search(r"answer\s*[:\-]?\s*([A-E])", str(text), flags=re.I)
    if m:
        letter = m.group(1).upper()
        if 0 <= ord(letter) - ord("A") < n_options:
            return letter
    return "unknown"


def parse_choice(text, candidates):
    letter = parse_letter(text, len(candidates))
    if letter != "unknown":
        return letter
    low = str(text).strip().lower()
    for c in candidates:
        cand = c["text"].lower()
        if low == cand or cand in low:
            return c["letter"]
    return "unknown"


def normalize_gold(answer, candidates, answer_type):
    if answer_type == "letter":
        m = re.search(r"([A-E])", str(answer), flags=re.I)
        return m.group(1).upper() if m else str(answer).strip().upper()
    answer = str(answer).strip()
    for c in candidates:
        if answer == c["text"] or answer.lower() == c["text"].lower():
            return c["letter"]
    m = re.search(r"([A-E])", answer, flags=re.I)
    return m.group(1).upper() if m else answer


def build_video_map(video_root):
    return {
        p.stem: p
        for p in Path(video_root).rglob("*")
        if p.suffix.lower() in [".mp4", ".avi", ".webm", ".mkv"]
    }


def load_rows(args):
    rows = []
    if args.dataset == "tempcompass":
        df = pd.read_parquet(args.data_path)
        if args.limit:
            df = df.head(args.limit)
        video_map = build_video_map(args.video_root)
        for _, row in df.iterrows():
            options = to_jsonable(row["options"])
            candidates = [
                {"letter": option_letter(opt, i), "text": option_text(opt), "raw": str(opt)}
                for i, opt in enumerate(options)
            ]
            video_path = video_map.get(str(row["video_id"]))
            rows.append({
                "uid": str(row["uuid"]),
                "video_id": str(row["video_id"]),
                "video_path": str(video_path) if video_path else "",
                "question": str(row["question"]),
                "candidates": candidates,
                "answer": normalize_gold(row["answer"], candidates, "letter"),
                "answer_type": "letter",
                "raw": row.to_dict(),
            })
    else:
        with open(args.data_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                options = to_jsonable(row["candidates"])
                candidates = [
                    {"letter": option_letter(opt, i), "text": option_text(opt), "raw": str(opt)}
                    for i, opt in enumerate(options)
                ]
                rows.append({
                    "uid": str(row.get("id", row.get("uuid", len(rows)))),
                    "video_id": str(row.get("video", row.get("id", len(rows)))),
                    "video_path": str(row["video_path"]),
                    "question": str(row["question"]),
                    "candidates": candidates,
                    "answer": normalize_gold(row["answer"], candidates, "text"),
                    "answer_type": "text",
                    "task": row.get("task"),
                    "raw": row,
                })
        if args.limit:
            rows = rows[:args.limit]
    return rows


def cache_frames(video_path, cache_dir, nframes=16, max_side=512):
    cache_dir = Path(cache_dir)
    key = hashlib.md5(str(video_path).encode()).hexdigest()[:12]
    out_dir = cache_dir / key
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [out_dir / f"f{i:02d}.jpg" for i in range(nframes)]
    if all(p.exists() for p in paths):
        return [str(p) for p in paths]

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    n = max(2, min(nframes, len(vr)))
    idx = np.linspace(0, len(vr) - 1, n).astype(int)
    frames = vr.get_batch(idx).asnumpy()

    actual = []
    for i, arr in enumerate(frames):
        p = out_dir / f"f{i:02d}.jpg"
        img = Image.fromarray(arr).convert("RGB")
        img.thumbnail((max_side, max_side))
        img.save(p, quality=95)
        actual.append(str(p))
    return actual


def format_options(candidates):
    return "\n".join(f"{c['letter']}. {c['text']}" for c in candidates)


def apply_chat(processor, model, frame_paths, prompt, max_new_tokens=16):
    content = [{"type": "image", "image": p} for p in frame_paths]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    out = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]


def generate_text_hint(question, candidates, tok, model, kind):
    if kind == "auxiliary_cot":
        instruction = "Write a concise temporal reasoning plan for selecting the correct option. Do not answer."
    elif kind == "localize_answer":
        instruction = "Write a concise temporal localization hint describing where to inspect the video. Do not answer."
    else:
        instruction = "Write a concise temporal sensitivity hint about order, direction, speed, or state change. Do not answer."
    prompt = f"""{instruction}

Question:
{question}

Options:
{format_options(candidates)}

Hint:"""
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=96, do_sample=False)
    hint = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return re.sub(r"\s+", " ", hint)


def prompted_answer(row, frame_paths, processor, model, aux_tok, aux_model, method):
    base = f"""Question:
{row['question']}

Options:
{format_options(row['candidates'])}
"""
    if method == "direct":
        prompt = base + "\nAnswer with exactly one option letter."
        raw = apply_chat(processor, model, frame_paths, prompt, max_new_tokens=16)
        return raw, None
    if method == "generic_cot":
        prompt = base + "\nReason about the temporal evidence, then give the final answer as exactly one option letter."
        raw = apply_chat(processor, model, frame_paths, prompt, max_new_tokens=96)
        return raw, None
    if method == "freeform_cot_matched":
        prompt = base + "\nUse a concise free-form temporal chain of thought with comparable reasoning budget, then answer exactly one option letter."
        raw = apply_chat(processor, model, frame_paths, prompt, max_new_tokens=140)
        return raw, None

    hint = generate_text_hint(row["question"], row["candidates"], aux_tok, aux_model, method)
    prompt = base + f"\nTemporal hint:\n{hint}\n\nAnswer with exactly one option letter."
    raw = apply_chat(processor, model, frame_paths, prompt, max_new_tokens=32)
    return raw, hint


def parse_triplet(text):
    text = text.replace("```json", "").replace("```", "").strip()
    m = re.search(r"\{[\s\S]*?\}", text, flags=re.S)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    cleaned = {}
    for k, v in obj.items():
        key = str(k).strip().lower().rstrip("!")
        cleaned[key] = str(v).replace("!", " ").strip()
    if not all(cleaned.get(k) for k in ["initial", "transition", "end"]):
        return None
    return {k: cleaned[k] for k in ["initial", "transition", "end"]}


def decompose_hypothesis(question, hypothesis, tok, model):
    prompt = f"""You decompose a candidate video answer into three visible checkpoints.

Return JSON only. No markdown. Do not use exclamation marks.

Rules:
- Treat the candidate answer as the hypothesis to verify.
- Preserve directions, before/after, left/right, faster/slower, color changes, and object states exactly.
- Each checkpoint should be visually checkable from video frames.
- Use early segment, middle segment, late segment.

Schema:
{{"initial":"...","transition":"...","end":"..."}}

Question: {question}
Candidate answer: {hypothesis}
"""
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=140,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    triplet = parse_triplet(raw)
    if triplet is None:
        triplet = {
            "initial": f"the early visible evidence for: {hypothesis}",
            "transition": f"the middle visible evidence for: {hypothesis}",
            "end": f"the late visible evidence for: {hypothesis}",
        }
    return triplet, raw


def candidate_ids(tokenizer, words):
    ids = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(toks[0])
    return sorted(set(ids))


def yes_probability(question, hypothesis, state, frame_paths, processor, model, yes_ids, no_ids, previous=""):
    prompt = f"""The attached images are consecutive video frames from one temporal segment.
They are ordered from earliest to latest.

Question:
{question}

Candidate answer being checked:
{hypothesis}

Previously checked observations:
{previous if previous else "none"}

Current checkpoint:
"{state}"

Does this segment visually support the current checkpoint for the candidate answer?
Answer exactly one word: Yes or No.
"""
    content = [{"type": "image", "image": p} for p in frame_paths]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[:, -1, :].float()[0]
    yes_score = torch.logsumexp(logits[yes_ids], dim=0)
    no_score = torch.logsumexp(logits[no_ids], dim=0)
    return torch.softmax(torch.stack([yes_score, no_score]), dim=0)[0].item()


def temporal_hypothesis(question, candidate_text):
    q = str(question).strip()
    c = str(candidate_text).strip().rstrip(".")
    m = re.match(r"(?i)^what happened after (.+?)\??$", q)
    if m:
        return f"First, {m.group(1).strip()}. After that, {c}."
    m = re.match(r"(?i)^what happened before (.+?)\??$", q)
    if m:
        return f"First, {c}. After that, {m.group(1).strip()}."
    return c


def state_sequence(triplet, method):
    if method == "state2":
        return [("initial", triplet["initial"]), ("end", triplet["end"])]
    if method == "state5":
        return [
            ("initial", triplet["initial"]),
            ("early_transition", f"{triplet['initial']} begins changing toward {triplet['transition']}"),
            ("transition", triplet["transition"]),
            ("late_transition", f"{triplet['transition']} leads toward {triplet['end']}"),
            ("end", triplet["end"]),
        ]
    return [("initial", triplet["initial"]), ("transition", triplet["transition"]), ("end", triplet["end"])]


def segment_frames(frame_paths, names, method):
    if method == "no_seg":
        return {name: frame_paths for name in names}
    n = len(frame_paths)
    if len(names) == 1:
        return {names[0]: frame_paths}
    spans = np.linspace(0, n, len(names) + 1).astype(int)
    out = {}
    for i, name in enumerate(names):
        seg = frame_paths[spans[i]:spans[i + 1]]
        out[name] = seg if len(seg) >= 2 else frame_paths[max(0, spans[i] - 1): min(n, spans[i + 1] + 1)]
    return out


def score_option(row, candidate, frame_paths, aux_tok, aux_model, processor, model, yes_ids, no_ids, method, early_exit):
    hyp = temporal_hypothesis(row["question"], candidate["text"])
    triplet, raw_triplet = decompose_hypothesis(row["question"], hyp, aux_tok, aux_model)

    if method == "joint":
        joint_state = (
            f"The video supports this full temporal sequence: initially, {triplet['initial']}; "
            f"then, {triplet['transition']}; finally, {triplet['end']}."
        )
        p = yes_probability(row["question"], hyp, joint_state, frame_paths, processor, model, yes_ids, no_ids)
        return p, [{"name": "joint", "state": joint_state, "p_yes": p}], triplet, raw_triplet, p < early_exit

    states = state_sequence(triplet, method)
    segments = segment_frames(frame_paths, [name for name, _ in states], method)
    probs = []
    previous = []
    score = 1.0
    stopped = False
    for name, state in states:
        prev = "" if method == "no_cond" else "; ".join(previous)
        p = yes_probability(row["question"], hyp, state, segments[name], processor, model, yes_ids, no_ids, previous=prev)
        probs.append({"name": name, "state": state, "p_yes": p})
        score *= p
        if early_exit >= 0 and p < early_exit:
            score = 0.0
            stopped = True
            break
        previous.append(state)
    return score, probs, triplet, raw_triplet, stopped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mvbench", "tempcompass"], required=True)
    ap.add_argument("--method", choices=sorted(TD_METHODS | PROMPT_METHODS), required=True)
    ap.add_argument("--data-path", required=True)
    ap.add_argument("--video-root", default="")
    ap.add_argument("--aux-model", default="/data/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--vl-model", default="/data/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache-dir", default="outputs/frame_cache")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nframes", type=int, default=16)
    ap.add_argument("--early-exit", type=float, default=0.3)
    args = ap.parse_args()

    rows = load_rows(args)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.vl_model, use_fast=False)
    vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vl_model,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    needs_aux = args.method not in ["direct", "generic_cot", "freeform_cot_matched"]
    aux_tok = aux_model = None
    if needs_aux:
        aux_tok = AutoTokenizer.from_pretrained(args.aux_model, trust_remote_code=True)
        aux_model = AutoModelForCausalLM.from_pretrained(
            args.aux_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        ).eval()

    yes_ids = no_ids = None
    if args.method in TD_METHODS:
        yes_ids = candidate_ids(processor.tokenizer, ["Yes", " yes", "YES", "yes"])
        no_ids = candidate_ids(processor.tokenizer, ["No", " no", "NO", "no"])

    total = correct = missing = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for row in tqdm(rows, desc=f"{args.dataset} {args.method} qwen"):
            if not row["video_path"] or not Path(row["video_path"]).exists():
                missing += 1
                continue
            frames = cache_frames(row["video_path"], args.cache_dir, nframes=args.nframes)

            if args.method in PROMPT_METHODS:
                raw, hint = prompted_answer(row, frames, processor, vl_model, aux_tok, aux_model, args.method)
                pred = parse_choice(raw, row["candidates"])
                option_scores = []
                extra = {"raw_prediction": raw}
                if hint is not None:
                    extra["hint"] = hint
            else:
                option_scores = []
                threshold = -1.0 if args.method == "no_early_exit" else args.early_exit
                for candidate in row["candidates"]:
                    score, probs, triplet, raw_triplet, stopped = score_option(
                        row,
                        candidate,
                        frames,
                        aux_tok,
                        aux_model,
                        processor,
                        vl_model,
                        yes_ids,
                        no_ids,
                        args.method,
                        threshold,
                    )
                    option_scores.append({
                        "letter": candidate["letter"],
                        "candidate": candidate["text"],
                        "score": score,
                        "probs": probs,
                        "triplet": triplet,
                        "raw_triplet": raw_triplet,
                        "early_exit": stopped,
                    })
                pred = max(option_scores, key=lambda x: x["score"])["letter"]
                extra = {}

            ok = pred == row["answer"]
            total += 1
            correct += int(ok)
            rec = {
                "uid": row["uid"],
                "video_id": row["video_id"],
                "video_path": row["video_path"],
                "question": row["question"],
                "options": [c["raw"] for c in row["candidates"]],
                "answer": row["answer"],
                "prediction": pred,
                "correct": ok,
                "method": args.method,
                "option_scores": option_scores,
            }
            rec.update(extra)
            if "task" in row:
                rec["task"] = row["task"]
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"accuracy: {correct}/{total} = {correct / total if total else 0:.4f}")
    print("missing videos:", missing)
    print("saved to:", args.output)


if __name__ == "__main__":
    main()
