import argparse
import json, math
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="/data/datasets/TempCompass/tempcompass_directionality_items.parquet")
    ap.add_argument("--direct", default="outputs/tempcompass_directionality_qwen_base.jsonl")
    ap.add_argument("--tdcot", default="outputs/tempcompass_directionality_qwen_tdcot.jsonl")
    ap.add_argument("--title", default="TempCompass Directionality Evaluation")
    return ap.parse_args()

def attach_meta(rows, meta_by_uuid, meta):
    out = []
    for i, x in enumerate(rows):
        uid = str(x.get("uuid", ""))
        if uid in meta_by_uuid:
            m = meta_by_uuid[uid]
        else:
            m = meta.iloc[i]
        y = dict(x)
        y["pair_id"] = str(m["pair_id"])
        y["side"] = str(m["side"])
        y["is_yesno"] = bool(m.get("is_yesno", False))
        out.append(y)
    return out

def load_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

def pair_from_correct(rows, correct_key="correct"):
    by = defaultdict(list)
    for x in rows:
        by[x["pair_id"]].append(x)

    it=ic=pt=pc=0
    for pid, rs in by.items():
        if len(rs) != 2:
            continue
        pt += 1
        ok = True
        for x in rs:
            it += 1
            good = bool(x[correct_key])
            ic += int(good)
            ok = ok and good
        pc += int(ok)
    return ic,it,ic/it if it else 0,pc,pt,pc/pt if pt else 0

def clamp(x):
    return min(max(float(x), 1e-6), 1 - 1e-6)

def logit(x):
    x = clamp(x)
    return math.log(x / (1 - x))

def score(ps, mode):
    if mode == "prod_all":
        return ps[0] * ps[1] * ps[2]
    if mode == "sum_all":
        return sum(ps)
    if mode == "sum_logit_all":
        return sum(logit(v) for v in ps)
    if mode == "transition_only":
        return ps[1]
    if mode == "end_only":
        return ps[2]
    if mode == "sum_trans_end":
        return ps[1] + ps[2]
    if mode == "prod_trans_end":
        return ps[1] * ps[2]
    raise ValueError(mode)

def tdcot_mode(rows, mode):
    out = []
    cnt = Counter()

    for x in rows:
        best = None
        best_score = None
        for o in x["option_scores"]:
            ps = [q["p_yes"] for q in o["probs"]]
            sc = score(ps, mode)
            if best_score is None or sc > best_score:
                best_score = sc
                best = o["letter"]

        y = dict(x)
        y["prediction_mode"] = best
        y["correct_mode"] = best == x["answer"]
        cnt[best] += 1
        out.append(y)

    return (*pair_from_correct(out, "correct_mode"), cnt)

def main():
    args = parse_args()
    meta = pd.read_parquet(args.meta).reset_index(drop=True)
    meta_by_uuid = {str(r["uuid"]): r for _, r in meta.iterrows()}

    direct_rows = attach_meta(load_jsonl(args.direct), meta_by_uuid, meta)
    tdcot_rows = attach_meta(load_jsonl(args.tdcot), meta_by_uuid, meta)

    print(f"# {args.title}\n")

    ic,it,ia,pc,pt,pa = pair_from_correct(direct_rows)
    print(f"Direct item={ic}/{it}={ia:.3f} pair={pc}/{pt}={pa:.3f}")

    ic,it,ia,pc,pt,pa = pair_from_correct(tdcot_rows)
    print(f"TD-CoT stored item={ic}/{it}={ia:.3f} pair={pc}/{pt}={pa:.3f}")

    print("\n## TD-CoT Score Modes")
    for mode in [
        "prod_all",
        "sum_all",
        "sum_logit_all",
        "transition_only",
        "end_only",
        "sum_trans_end",
        "prod_trans_end",
    ]:
        ic,it,ia,pc,pt,pa,cnt = tdcot_mode(tdcot_rows, mode)
        print(f"{mode:16s} item={ic}/{it}={ia:.3f} pair={pc}/{pt}={pa:.3f} pred_counts={dict(cnt)}")


if __name__ == "__main__":
    main()
