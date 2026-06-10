import argparse, json, re, hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from tqdm import tqdm
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def base_key(video_path):
    m = re.match(r"(.*/(?:mcq_)?vid_\d+)_([01])\.mp4$", video_path)
    return m.group(1) if m else video_path

def load_pairs(data_path, limit_pairs=None):
    groups = defaultdict(list)
    with open(data_path, encoding="utf-8") as f:
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
    m = re.search(r"\{[\s\S]*?\}", text)
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
{{"initial":"the man stands with his feet planted before hitting the shuttlecock","transition":"the man hits the shuttlecock while his feet remain stationary","end":"the shuttlecock flies away while the man's feet have not moved"}}

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

def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def sample_segment_pixels(
    video_path,
    segment_index=0,
    total_segments=3,
    full_video=False,
    nframes=16,
    input_size=448,
):
    vr = VideoReader(str(video_path), ctx=cpu(0), num_threads=1)
    total = len(vr)
    n = max(2, min(nframes, total))
    full_idx = np.linspace(0, total - 1, n).astype(int)

    if full_video or total_segments <= 1:
        idx = full_idx
    else:
        spans = np.linspace(0, n, total_segments + 1).astype(int)
        left = spans[segment_index]
        right = spans[segment_index + 1]
        idx = full_idx[left:right]
        if len(idx) < 2:
            idx = full_idx[max(0, left - 1): min(n, right + 1)]

    transform = build_transform(input_size)
    pixels = []
    for i in idx:
        img = Image.fromarray(vr[i].asnumpy()).convert("RGB")
        pixels.append(transform(img).unsqueeze(0))
    pixel_values = torch.cat(pixels, dim=0)
    return pixel_values, [1] * len(pixels)

def normalize_yesno(text):
    t = text.strip().lower()
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return "unknown"

def candidate_ids(tokenizer, words):
    ids = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(toks[0])
    return sorted(set(ids))

def chat_first_token_logits(model, tokenizer, pixel_values, prompt, num_patches_list):
    captured = {}
    orig_generate = model.generate

    def wrapped_generate(*args, **kwargs):
        kwargs["return_dict_in_generate"] = True
        kwargs["output_scores"] = True
        out = orig_generate(*args, **kwargs)
        captured["scores"] = out.scores
        return out.sequences

    model.generate = wrapped_generate
    try:
        raw = model.chat(
            tokenizer,
            pixel_values,
            prompt,
            dict(max_new_tokens=1, do_sample=False),
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )
    finally:
        model.generate = orig_generate

    logits = captured["scores"][0][0].float()
    return raw, logits


def verify_segment(
    question,
    state,
    video_path,
    segment_name,
    segment_index,
    total_segments,
    tokenizer,
    model,
    yes_ids,
    no_ids,
    previous="",
    full_video=False,
    nframes=16,
):
    pixel_values, num_patches_list = sample_segment_pixels(
        video_path,
        segment_index=segment_index,
        total_segments=total_segments,
        full_video=full_video,
        nframes=nframes,
    )
    pixel_values = pixel_values.to(torch.bfloat16).cuda()

    prefix = "".join([f"Frame{i+1}: <image>\n" for i in range(len(num_patches_list))])
    prompt = prefix + f"""The frames are consecutive and ordered from earliest to latest.
They show the {segment_name} temporal evidence for one video.

You must verify one checkpoint for the YES answer.

Original question:
{question}

Previously verified checkpoints:
{previous if previous else "none"}

Checkpoint to verify:
"{state}"

Strict rules:
- Answer Yes only if this checkpoint is clearly and directly visible in these frames.
- Answer No if the checkpoint is absent, ambiguous, only partly visible, or only guessed from common sense.
- For before/after, faster/slower, same-time, duration, or color-change claims, answer Yes only if the relation is visually supported by this segment.
- Do not assume the video is the correct one. It may be the reversed or negative example.

Answer exactly one word: Yes or No.
"""
    raw, logits = chat_first_token_logits(model, tokenizer, pixel_values, prompt, num_patches_list)
    yes_score = torch.logsumexp(logits[yes_ids], dim=0)
    no_score = torch.logsumexp(logits[no_ids], dim=0)
    p_yes = torch.softmax(torch.stack([yes_score, no_score]), dim=0)[0].item()
    pred = "yes" if p_yes >= 0.5 else "no"
    return p_yes, raw, pred


def state_sequence(triplet, variant):
    if variant == "state2":
        return [("initial", triplet["initial"]), ("end", triplet["end"])]
    if variant == "state5":
        return [
            ("initial", triplet["initial"]),
            ("early_transition", f"{triplet['initial']} begins changing toward {triplet['transition']}"),
            ("transition", triplet["transition"]),
            ("late_transition", f"{triplet['transition']} leads toward {triplet['end']}"),
            ("end", triplet["end"]),
        ]
    return [("initial", triplet["initial"]), ("transition", triplet["transition"]), ("end", triplet["end"])]


def score_video(question, triplet, video_path, tokenizer, model, yes_ids, no_ids, variant="tdcot", early_exit=0.3, nframes=16):
    if variant == "joint":
        state = (
            f"The video supports this full temporal sequence: initially, {triplet['initial']}; "
            f"then, {triplet['transition']}; finally, {triplet['end']}."
        )
        p, raw, pred = verify_segment(
            question,
            state,
            video_path,
            "joint",
            0,
            1,
            tokenizer,
            model,
            yes_ids,
            no_ids,
            full_video=True,
            nframes=nframes,
        )
        return p, [{"name": "joint", "state": state, "p_yes": p, "raw": raw, "prediction": pred}], False

    states = state_sequence(triplet, variant)
    probs = []
    previous = []
    threshold = -1.0 if variant == "no_early_exit" else early_exit
    full_video = variant == "no_seg"
    for i, (name, state) in enumerate(states):
        prior = "" if variant == "no_cond" else "; ".join(previous)
        p, raw, pred = verify_segment(
            question,
            state,
            video_path,
            name,
            i,
            len(states),
            tokenizer,
            model,
            yes_ids,
            no_ids,
            previous=prior,
            full_video=full_video,
            nframes=nframes,
        )
        probs.append({"name": name, "state": state, "p_yes": p, "raw": raw, "prediction": pred})
        if threshold >= 0 and p < threshold:
            return 0.0, probs, True
        previous.append(state)
    score = 1.0
    for p in probs:
        score *= p["p_yes"]
    return score, probs, False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="/data/datasets/TimeBlind/data.jsonl")
    ap.add_argument("--dataset-root", default="/data/datasets")
    ap.add_argument("--aux-model", default="/data/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--vl-model", default="/data/models/InternVL2-8B")
    ap.add_argument("--output", default="outputs/timeblind_tdcot_pair_internvl100.jsonl")
    ap.add_argument("--limit-pairs", type=int, default=100)
    ap.add_argument("--nframes", type=int, default=16)
    ap.add_argument(
        "--variant",
        choices=["tdcot", "no_seg", "joint", "no_cond", "state2", "state5", "no_early_exit", "no_routing"],
        default="tdcot",
    )
    ap.add_argument(
        "--early-exit",
        type=float,
        default=0.3,
        help="Stop verifying an item once a checkpoint probability falls below this threshold. Use -1 to disable.",
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

    vl_model = AutoModel.from_pretrained(
        args.vl_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=False,
    ).eval().cuda()
    vl_tok = AutoTokenizer.from_pretrained(args.vl_model, trust_remote_code=True, use_fast=False)

    yes_ids = candidate_ids(vl_tok, ["Yes", " yes", "YES", "yes"])
    no_ids = candidate_ids(vl_tok, ["No", " no", "NO", "no"])
    print("yes_ids:", yes_ids, "no_ids:", no_ids)

    pairs = load_pairs(args.data_path, args.limit_pairs)

    item_correct = item_total = pair_correct = 0

    with open(args.output, "w", encoding="utf-8") as fout:
        for pair in tqdm(pairs, desc="TD-CoT InternVL2 pairs"):
            question = pair[0]["question"]
            triplet, raw_triplet = decompose_question(question, aux_tok, aux_model)

            scored = []
            for item in pair:
                vp = Path(args.dataset_root) / item["video_path"]
                score, probs, early = score_video(
                    question, triplet, vp, vl_tok, vl_model, yes_ids, no_ids,
                    variant=args.variant,
                    early_exit=args.early_exit,
                    nframes=args.nframes,
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

            fout.write(json.dumps({
                "question": question,
                "triplet": triplet,
                "raw_triplet": raw_triplet,
                "videos": scored,
                "pair_correct": pair_ok,
            }, ensure_ascii=False) + "\n")
            fout.flush()

    print(f"item accuracy: {item_correct}/{item_total} = {item_correct/item_total:.4f}")
    print(f"pair accuracy: {pair_correct}/{len(pairs)} = {pair_correct/len(pairs):.4f}")
    print("saved to:", args.output)

if __name__ == "__main__":
    main()
