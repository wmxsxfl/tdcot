import argparse
import json
import math
import random
from pathlib import Path


def load_jsonl(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def clamp(x):
    return min(max(float(x), 1e-6), 1 - 1e-6)


def logit(x):
    x = clamp(x)
    return math.log(x / (1 - x))


def score_probs(ps, mode):
    if mode == "prod_all":
        return ps[0] * ps[1] * ps[2]
    if mode == "sum_all":
        return sum(ps)
    if mode == "sum_logit_all":
        return sum(logit(x) for x in ps)
    if mode == "min_all":
        return min(ps)
    if mode == "initial_only":
        return ps[0]
    if mode == "transition_only":
        return ps[1]
    if mode == "end_only":
        return ps[2]
    if mode == "sum_trans_end":
        return ps[1] + ps[2]
    if mode == "prod_trans_end":
        return ps[1] * ps[2]
    raise ValueError(mode)


def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    k = (len(xs) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - k) + xs[hi] * (k - lo)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def direct_pair_metrics(pair, direct_by_index):
    item_values = []
    preds = []
    answers = []

    for v in pair["videos"]:
        d = direct_by_index[v["index"]]
        item_values.append(1.0 if d["correct"] else 0.0)
        preds.append(str(d.get("prediction", "")).lower())
        answers.append(str(v["answer"]).lower())

    item_acc = mean(item_values)
    strict_pair = 1.0 if all(x == 1.0 for x in item_values) else 0.0

    if preds.count("yes") == 1:
        yes_i = preds.index("yes")
        choice_pair = 1.0 if answers[yes_i] == "yes" else 0.0
    else:
        choice_pair = 0.0

    return item_acc, strict_pair, choice_pair


def tdcot_pair_metrics(pair, mode):
    vids = pair["videos"]

    if mode == "stored":
        item_acc = mean([1.0 if v["correct"] else 0.0 for v in vids])
        pair_acc = 1.0 if pair.get("pair_correct", False) else 0.0
        return item_acc, pair_acc

    scores = []
    for v in vids:
        ps = [q["p_yes"] for q in v["probs"]]
        scores.append(score_probs(ps, mode))

    yes_i = max(range(len(scores)), key=lambda i: scores[i])

    item_values = []
    for i, v in enumerate(vids):
        pred = "yes" if i == yes_i else "no"
        item_values.append(1.0 if pred == v["answer"] else 0.0)

    item_acc = mean(item_values)
    pair_acc = 1.0 if vids[yes_i]["answer"] == "yes" else 0.0
    return item_acc, pair_acc


def bootstrap_ci(values, rng, n_boot):
    boots = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(sample))
    return mean(values), percentile(boots, 0.025), percentile(boots, 0.975)


def bootstrap_diff_ci(a, b, rng, n_boot):
    boots = []
    n = len(a)
    diffs = [b[i] - a[i] for i in range(n)]
    obs = mean(diffs)

    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        boots.append(mean(sample))

    return obs, percentile(boots, 0.025), percentile(boots, 0.975)


def randomization_pvalue(a, b, rng, n_perm):
    diffs = [b[i] - a[i] for i in range(len(a))]
    obs = abs(mean(diffs))
    count = 0

    for _ in range(n_perm):
        signed = []
        for d in diffs:
            signed.append(d if rng.random() < 0.5 else -d)
        if abs(mean(signed)) >= obs:
            count += 1

    return (count + 1) / (n_perm + 1)


def fmt_ci(x, lo, hi):
    return f"{x:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direct", default="outputs/timeblind_qwen_base.jsonl")
    ap.add_argument("--tdcot", default="outputs/timeblind_qwen_tdcot.jsonl")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--permutation", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    direct_rows = load_jsonl(args.direct)
    direct_by_index = {x["index"]: x for x in direct_rows}
    pairs = load_jsonl(args.tdcot)

    pairs = [
        p for p in pairs
        if all(v["index"] in direct_by_index for v in p["videos"])
    ]

    direct_item = []
    direct_pair_strict = []
    direct_pair_choice = []

    for p in pairs:
        item, strict, choice = direct_pair_metrics(p, direct_by_index)
        direct_item.append(item)
        direct_pair_strict.append(strict)
        direct_pair_choice.append(choice)

    modes = [
        "stored",
        "prod_all",
        "sum_all",
        "sum_logit_all",
        "min_all",
        "initial_only",
        "transition_only",
        "end_only",
        "sum_trans_end",
        "prod_trans_end",
    ]

    print("# TimeBlind Bootstrap Evaluation")
    print()
    print(f"direct: `{args.direct}`")
    print(f"tdcot: `{args.tdcot}`")
    print(f"pairs: {len(pairs)}")
    print(f"items: {len(pairs) * 2}")
    print()

    print("## Direct Baseline")
    x, lo, hi = bootstrap_ci(direct_item, rng, args.bootstrap)
    print(f"- Direct item accuracy: {fmt_ci(x, lo, hi)}")

    x, lo, hi = bootstrap_ci(direct_pair_strict, rng, args.bootstrap)
    print(f"- Direct strict pair accuracy: {fmt_ci(x, lo, hi)}")

    x, lo, hi = bootstrap_ci(direct_pair_choice, rng, args.bootstrap)
    print(f"- Direct choice-like pair accuracy: {fmt_ci(x, lo, hi)}")
    print()

    print("## TD-CoT Score Modes")
    print("| Mode | Item Acc 95% CI | Pair Acc 95% CI | Diff vs Direct Item 95% CI | p-value |")
    print("|---|---:|---:|---:|---:|")

    best = None

    for mode in modes:
        t_item = []
        t_pair = []
        for p in pairs:
            item, pair_acc = tdcot_pair_metrics(p, mode)
            t_item.append(item)
            t_pair.append(pair_acc)

        item_x, item_lo, item_hi = bootstrap_ci(t_item, rng, args.bootstrap)
        pair_x, pair_lo, pair_hi = bootstrap_ci(t_pair, rng, args.bootstrap)
        diff_x, diff_lo, diff_hi = bootstrap_diff_ci(direct_item, t_item, rng, args.bootstrap)
        pvalue = randomization_pvalue(direct_item, t_item, rng, args.permutation)

        print(
            f"| {mode} | {fmt_ci(item_x, item_lo, item_hi)} "
            f"| {fmt_ci(pair_x, pair_lo, pair_hi)} "
            f"| {fmt_ci(diff_x, diff_lo, diff_hi)} | {pvalue:.4f} |"
        )

        if best is None or item_x > best[1]:
            best = (mode, item_x)

    print()
    print(f"Best item mode: `{best[0]}` = {best[1]:.3f}")
    print()

    print("## Direct vs TD-CoT Stored Overlap")
    both = direct_only = tdcot_only = both_wrong = 0
    for p in pairs:
        for v in p["videos"]:
            d_ok = bool(direct_by_index[v["index"]]["correct"])
            t_ok = bool(v["correct"])
            if d_ok and t_ok:
                both += 1
            elif d_ok and not t_ok:
                direct_only += 1
            elif not d_ok and t_ok:
                tdcot_only += 1
            else:
                both_wrong += 1

    print(f"- both_correct: {both}")
    print(f"- direct_only: {direct_only}")
    print(f"- tdcot_only: {tdcot_only}")
    print(f"- both_wrong: {both_wrong}")


if __name__ == "__main__":
    main()
