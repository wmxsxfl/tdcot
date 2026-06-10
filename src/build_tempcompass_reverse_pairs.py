import argparse
import json
import re
from pathlib import Path
from collections import Counter

import pandas as pd

temporal_keywords = [
    "left", "right", "towards", "toward", "away", "direction", "moving", "camera",
    "before", "after", "first", "then", "followed", "order", "sequence",
    "open", "closed", "closing", "opening", "from", "to", "changing", "turning",
    "construct", "deconstruct", "transform", "become",
    "slow", "fast", "speed", "tempo", "time-lapse", "reverse",
    "clockwise", "counterclockwise", "up", "down",
]

def norm_text(x):
    return re.sub(r"\s+", " ", str(x).strip()).lower()

def norm_question(q):
    return norm_text(q)

def parse_opts(opts):
    if isinstance(opts, str):
        try:
            opts = json.loads(opts)
        except Exception:
            pass
    if hasattr(opts, "tolist"):
        opts = opts.tolist()
    if not isinstance(opts, (list, tuple)) or len(opts) != 2:
        return None
    parsed = []
    for i, o in enumerate(opts):
        o = re.sub(r"\s+", " ", str(o).strip())
        m = re.match(r"^([A-Z])\.\s*(.*)$", o)
        letter = m.group(1) if m else chr(ord("A") + i)
        text = m.group(2).strip() if m else o
        parsed.append((letter, text, f"{letter}. {text}"))
    return parsed

def option_set_key(parsed):
    return tuple(sorted(norm_text(t) for _, t, _ in parsed))

def is_yesno(parsed):
    return option_set_key(parsed) == ("no", "yes")

def answer_text(answer, parsed):
    ans = str(answer).strip()
    for letter, text, full in parsed:
        if ans.upper() == letter:
            return text
        if norm_text(ans) == norm_text(full):
            return text
        if norm_text(ans) == norm_text(text):
            return text
    return ans

def temporal_score(question, parsed):
    text = norm_text(question) + " " + " ".join(norm_text(t) for _, t, _ in parsed)
    return sum(1 for k in temporal_keywords if k in text)

def build_records(df, video_root):
    df = df.copy()
    df["video_id"] = df["video_id"].astype(str)

    ids = set(df["video_id"])
    reverse_pairs = []
    for vid in sorted(ids):
        if vid.endswith("_reverse"):
            base = vid[:-8]
            if base in ids:
                reverse_pairs.append((base, vid))

    records = []
    pair_id = 0
    video_root = Path(video_root)

    for base, rev in reverse_pairs:
        fwd = df[df["video_id"] == base]
        bwd = df[df["video_id"] == rev]

        bwd_by_key = {}
        for _, r in bwd.iterrows():
            parsed = parse_opts(r["options"])
            if not parsed:
                continue
            score = temporal_score(r["question"], parsed)
            if score <= 0:
                continue

            # Pair matching uses the normalized question and unordered option text set.
            key = (norm_question(r["question"]), option_set_key(parsed))
            bwd_by_key.setdefault(key, []).append((r, parsed, score))

        used_reverse_uuids = set()

        for _, rf in fwd.iterrows():
            fparsed = parse_opts(rf["options"])
            if not fparsed:
                continue
            fscore = temporal_score(rf["question"], fparsed)
            if fscore <= 0:
                continue

            key = (norm_question(rf["question"]), option_set_key(fparsed))
            candidates = bwd_by_key.get(key, [])

            for rr, rparsed, rscore in candidates:
                if str(rr["uuid"]) in used_reverse_uuids:
                    continue

                f_ans_text = norm_text(answer_text(rf["answer"], fparsed))
                r_ans_text = norm_text(answer_text(rr["answer"], rparsed))

                # Reversal pair should flip the correct textual answer.
                if f_ans_text == r_ans_text:
                    continue

                used_reverse_uuids.add(str(rr["uuid"]))
                pid = f"{base}__{pair_id}"
                pair_id += 1
                score = max(fscore, rscore)

                for side, r, parsed in [("forward", rf, fparsed), ("reverse", rr, rparsed)]:
                    records.append({
                        "pair_id": pid,
                        "side": side,
                        "uuid": str(r["uuid"]),
                        "video_id": str(r["video_id"]),
                        "video_path": str(video_root / f"{r['video_id']}.mp4"),
                        "question": str(r["question"]).strip(),
                        "options": [full for _, _, full in parsed],
                        "answer": str(r["answer"]).strip(),
                        "answer_text": answer_text(r["answer"], parsed),
                        "temporal_score": score,
                        "is_yesno": is_yesno(parsed),
                    })
                break

    return reverse_pairs, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/datasets/TempCompass/tempcompass_combined.parquet")
    ap.add_argument("--video-root", default="/data/datasets/TempCompass/videos/videos")
    ap.add_argument("--output-jsonl", default="/data/datasets/TempCompass/tempcompass_directionality_pairs.jsonl")
    ap.add_argument("--output-parquet", default="/data/datasets/TempCompass/tempcompass_directionality_items.parquet")
    ap.add_argument("--expected-pairs", type=int, default=1500)
    ap.add_argument("--strict-count", action="store_true")
    args = ap.parse_args()

    df = pd.read_parquet(args.input)
    reverse_pairs, records = build_records(df, args.video_root)

    print("reverse video pairs:", len(reverse_pairs))
    print("paired item rows:", len(records))
    print("paired question pairs:", len(records) // 2)
    if args.expected_pairs and len(records) // 2 != args.expected_pairs:
        msg = (
            f"expected {args.expected_pairs} paired comparisons for the paper setting, "
            f"but built {len(records) // 2}. Check dataset version and filtering rules."
        )
        if args.strict_count:
            raise SystemExit(msg)
        print("WARNING:", msg)
    print("by side:", Counter(r["side"] for r in records))
    print("answer counts:", Counter(r["answer"] for r in records))
    print("yes/no rows:", sum(r["is_yesno"] for r in records))
    print("temporal score counts:", Counter(r["temporal_score"] for r in records))

    out_jsonl = Path(args.output_jsonl)
    out_parquet = Path(args.output_parquet)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    pd.DataFrame(records).to_parquet(out_parquet, index=False)

    print("saved jsonl:", out_jsonl)
    print("saved parquet:", out_parquet)

    print("\nexamples:")
    for r in records[:30]:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
