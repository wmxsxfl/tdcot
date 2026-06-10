import argparse, json, re, hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


def base_key(video_path):
    m = re.match(r"(.*/(?:mcq_)?vid_\d+)_([01])\.mp4$", video_path)
    return m.group(1) if m else video_path


def load_pairs(data_path, limit_pairs=None):
    groups = defaultdict(list)
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            x = json.loads(line)
            groups[(base_key(x["video_path"]), x["question"])].append(x)

    pairs = []
    for _, g in groups.items():
        if len(g) == 2 and sorted([x["answer"].lower() for x in g]) == ["no", "yes"]:
            pairs.append(sorted(g, key=lambda x: x["index"]))
    return pairs[:limit_pairs] if limit_pairs else pairs


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

    required = ["initial", "transition", "end"]
    if not all(cleaned.get(k) for k in required):
        return None
    return {k: cleaned[k] for k in required}

def decompose_question(question, tok, model):
    prompt = f"""You decompose a temporal video yes/no question into three visible checkpoints for the YES answer.

Return JSON only. No markdown. Do not use exclamation marks.

Very important:
- Preserve the exact meaning of the question.
- Do not reverse comparisons such as first vs second, faster vs slower, before vs after.
- Each checkpoint should be visually checkable from video frames.
- The checkpoints should follow early segment, middle segment, late segment.

Schema:
{{"initial":"...","transition":"...","end":"..."}}

Example:
Question: Does the boy start eating shortly after the girl puts her cup down?
{{"initial":"the girl is holding or lowering her cup while the boy is not yet eating","transition":"the girl puts the cup down before the boy begins eating","end":"shortly after the cup is down, the boy starts eating"}}

Example:
Question: Does the person swing the badminton racket faster the first time than the second?
{{"initial":"the person performs the first badminton racket swing","transition":"the person performs the second badminton racket swing after the first","end":"the first swing is visibly faster than the second swing"}}

Example:
Question: Does the man hit the shuttlecock without moving his feet?
{{"initial":"the man stands with his feet planted before hitting","transition":"the man hits the shuttlecock while his feet remain planted","end":"the shuttlecock moves away while the man has not moved his feet"}}

Question: {question}
"""
    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok([text], return_tensors="pt").to(model.device)

    bang = tok.encode("!", add_special_tokens=False)
    bad_words_ids = [bang] if len(bang) == 1 else None

    gen_kwargs = dict(
        max_new_tokens=140,
        do_sample=False,
        repetition_penalty=1.05,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
    )
    if bad_words_ids:
        gen_kwargs["bad_words_ids"] = bad_words_ids

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    raw = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    triplet = parse_triplet(raw)
    if triplet is None:
        triplet = {
            "initial": "the early visible state needed for the yes answer",
            "transition": "the middle visible change needed for the yes answer",
            "end": "the late visible state needed for the yes answer",
        }
    return triplet, raw

def cache_16_frames(video_path, cache_dir, nframes=16):
    cache_dir = Path(cache_dir)
    key = hashlib.md5(str(video_path).encode()).hexdigest()[:12]
    out_dir = cache_dir / key
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [out_dir / f"f{i:02d}.jpg" for i in range(nframes)]
    if all(p.exists() for p in paths):
        return [str(p) for p in paths]

    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    idx = np.linspace(0, len(vr) - 1, nframes).astype(int)
    frames = vr.get_batch(idx).asnumpy()

    for p, arr in zip(paths, frames):
        Image.fromarray(arr).save(p, quality=95)
    return [str(p) for p in paths]


def candidate_ids(tokenizer, words):
    ids = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(toks[0])
    return sorted(set(ids))


def yes_probability(question, state, image_paths, processor, model, yes_ids, no_ids, previous=""):
    prompt = f"""The attached images are consecutive video frames from one temporal segment.
They are ordered from earliest to latest.

Original question: {question}
We are checking the YES answer.

Previously checked observations: {previous if previous else "none"}

Current checkpoint:
"{state}"

Does this temporal segment visually support the current checkpoint?
Answer exactly one word: Yes or No.
"""
    content = [{"type": "image", "image": p} for p in image_paths]
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
    p_yes = torch.softmax(torch.stack([yes_score, no_score]), dim=0)[0].item()
    return p_yes


def score_video(question, triplet, frame_paths, processor, model, yes_ids, no_ids, early_exit=0.3):
    # 5-state decomposition ablation.
    # Reuse the original initial / transition / end decomposition, then add two
    # interpolated temporal checkpoints for finer-grained verification.
    state_specs = [
        ("initial", triplet["initial"]),
        (
            "initial_to_transition",
            "the video begins changing from this early state: "
            + triplet["initial"]
            + " toward this middle event: "
            + triplet["transition"],
        ),
        ("transition", triplet["transition"]),
        (
            "transition_to_end",
            "after the middle event, the video progresses toward this late state: "
            + triplet["end"],
        ),
        ("end", triplet["end"]),
    ]

    cuts = np.linspace(0, len(frame_paths), len(state_specs) + 1).astype(int)
    segments = {}
    for i, (name, _) in enumerate(state_specs):
        seg = frame_paths[cuts[i]:cuts[i + 1]]
        if not seg:
            seg = [frame_paths[min(cuts[i], len(frame_paths) - 1)]]
        segments[name] = seg

    probs = []
    previous = []
    score = 1.0
    early = False

    for name, state in state_specs:
        p = yes_probability(
            question, state, segments[name], processor, model, yes_ids, no_ids,
            previous="\n".join(previous),
        )
        probs.append({"name": name, "state": state, "p_yes": p})
        score *= p
        previous.append(state)

        if early_exit >= 0 and score < early_exit:
            early = True
            break

    return score, probs, early

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="/data/datasets/TimeBlind/data.jsonl")
    ap.add_argument("--dataset-root", default="/data/datasets")
    ap.add_argument("--aux-model", default="/data/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--vl-model", default="/data/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--output", default="outputs/timeblind_tdcot_pair_qwen5.jsonl")
    ap.add_argument("--cache-dir", default="outputs/frame_cache")
    ap.add_argument("--limit-pairs", type=int, default=100)
    ap.add_argument(
        "--early-exit",
        type=float,
        default=0.3,
        help="Stop verifying an item once the running product score falls below this threshold. Use -1 to disable.",
    )
    args = ap.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    aux_tok = AutoTokenizer.from_pretrained(args.aux_model, trust_remote_code=True)
    aux_model = AutoModelForCausalLM.from_pretrained(
        args.aux_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    vl_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vl_model,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.vl_model, use_fast=False)

    tokenizer = processor.tokenizer
    yes_ids = candidate_ids(tokenizer, ["Yes", " yes", "YES", "yes"])
    no_ids = candidate_ids(tokenizer, ["No", " no", "NO", "no"])
    print("yes_ids:", yes_ids, "no_ids:", no_ids)

    pairs = load_pairs(args.data_path, args.limit_pairs)

    item_correct = 0
    item_total = 0
    pair_correct = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for pair in tqdm(pairs, desc="TD-CoT pairs"):
            question = pair[0]["question"]
            triplet, raw_triplet = decompose_question(question, aux_tok, aux_model)

            scored = []
            for item in pair:
                vp = Path(args.dataset_root) / item["video_path"]
                frames = cache_16_frames(vp, args.cache_dir, nframes=16)
                score, probs, early = score_video(
                    question, triplet, frames, processor, vl_model, yes_ids, no_ids,
                    early_exit=args.early_exit,
                )
                scored.append({
                    "index": item["index"],
                    "video_path": item["video_path"],
                    "answer": item["answer"].lower(),
                    "score": score,
                    "probs": probs,
                    "early_exit": early,
                })

            yes_pos = 0 if scored[0]["score"] >= scored[1]["score"] else 1
            for i, s in enumerate(scored):
                pred = "yes" if i == yes_pos else "no"
                s["prediction"] = pred
                s["correct"] = pred == s["answer"]
                item_correct += int(s["correct"])
                item_total += 1

            pair_ok = all(s["correct"] for s in scored)
            pair_correct += int(pair_ok)

            rec = {
                "question": question,
                "triplet": triplet,
                "raw_triplet": raw_triplet,
                "videos": scored,
                "pair_correct": pair_ok,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"item accuracy: {item_correct}/{item_total} = {item_correct / item_total:.4f}")
    print(f"pair accuracy: {pair_correct}/{len(pairs)} = {pair_correct / len(pairs):.4f}")
    print("saved to:", args.output)


if __name__ == "__main__":
    main()
