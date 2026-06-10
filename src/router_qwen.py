import argparse
import json
import math
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


TEMPORAL_LABELS = ["temporal", "Temporal", " temporal", " Temporal"]
GENERAL_LABELS = ["general", "General", " general", " General"]


def label_ids(tokenizer, words):
    ids = []
    for w in words:
        toks = tokenizer.encode(w, add_special_tokens=False)
        if len(toks) == 1:
            ids.append(toks[0])
    return sorted(set(ids))


def build_prompt(question):
    return f"""Classify whether the video question requires temporal order or direction reasoning.

Return exactly one word:
temporal
general

Examples:
Question: Does the person open the door before sitting down?
Answer: temporal

Question: Is the person moving faster in the second half than the first half?
Answer: temporal

Question: Does the object change from red to blue?
Answer: temporal

Question: What color is the car?
Answer: general

Question: How many people are in the room?
Answer: general

Question: What object is on the table?
Answer: general

Question: {question}
Answer:"""


def classify(question, tokenizer, model, temporal_ids, general_ids):
    prompt = build_prompt(question)
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model(**inputs)
        logits = out.logits[:, -1, :].float()[0]

    temporal_logit = torch.logsumexp(logits[temporal_ids], dim=0)
    general_logit = torch.logsumexp(logits[general_ids], dim=0)
    probs = torch.softmax(torch.stack([temporal_logit, general_logit]), dim=0)
    p_temporal = probs[0].item()
    p_general = probs[1].item()

    pred = "temporal" if p_temporal >= p_general else "general"
    conf = max(p_temporal, p_general)

    return {
        "prediction": pred,
        "confidence": conf,
        "p_temporal": p_temporal,
        "p_general": p_general,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="/data/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--input", required=True, help="jsonl with question and optional label")
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.8)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    ).eval()

    temporal_ids = label_ids(tok, TEMPORAL_LABELS)
    general_ids = label_ids(tok, GENERAL_LABELS)

    print("temporal_ids:", temporal_ids)
    print("general_ids:", general_ids)

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = correct = 0
    temporal_total = temporal_correct = 0
    general_total = general_correct = 0
    routed_tdcot = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for x in rows:
            q = x["question"]
            res = classify(q, tok, model, temporal_ids, general_ids)
            route = "tdcot" if (res["prediction"] == "temporal" and res["confidence"] >= args.threshold) else "direct"

            y = dict(x)
            y.update(res)
            y["route"] = route
            fout.write(json.dumps(y, ensure_ascii=False) + "\n")

            if "label" in x:
                total += 1
                ok = res["prediction"] == x["label"]
                correct += int(ok)
                if x["label"] == "temporal":
                    temporal_total += 1
                    temporal_correct += int(ok)
                elif x["label"] == "general":
                    general_total += 1
                    general_correct += int(ok)

            routed_tdcot += int(route == "tdcot")

    print("saved to:", out_path)
    print("total:", total)
    if total:
        print("accuracy:", correct / total)
    if temporal_total:
        print("temporal recall:", temporal_correct / temporal_total, f"({temporal_correct}/{temporal_total})")
    if general_total:
        print("general accuracy:", general_correct / general_total, f"({general_correct}/{general_total})")
    print("tdcot route rate:", routed_tdcot / len(rows), f"({routed_tdcot}/{len(rows)})")


if __name__ == "__main__":
    main()
