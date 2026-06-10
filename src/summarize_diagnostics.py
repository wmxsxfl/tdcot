import argparse
import json
from collections import Counter
from pathlib import Path


def load_jsonl(path):
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"missing diagnostic file: {p}")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def bool_value(row, keys):
    for key in keys:
        if key in row:
            return bool(row[key])
    return False


def cohen_kappa(a, b):
    if not a or len(a) != len(b):
        return 0.0
    n = len(a)
    agree = sum(int(x == y) for x, y in zip(a, b)) / n
    pa = sum(a) / n
    pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return 1.0
    return (agree - pe) / (1 - pe)


def summarize_decomposition(rows):
    if not rows:
        return []
    valid = [bool_value(r, ["valid", "accepted", "decomposition_valid"]) for r in rows]
    out = [
        f"- items: {len(rows)}",
        f"- valid decompositions: {sum(valid)}/{len(valid)} = {mean(valid):.3f}",
    ]

    a_keys = ["annotator_a_valid", "ann_a_valid", "a_valid"]
    b_keys = ["annotator_b_valid", "ann_b_valid", "b_valid"]
    if any(any(k in r for k in a_keys) for r in rows) and any(any(k in r for k in b_keys) for r in rows):
        a = [bool_value(r, a_keys) for r in rows]
        b = [bool_value(r, b_keys) for r in rows]
        out.append(f"- Cohen kappa: {cohen_kappa(a, b):.3f}")
    return out


def summarize_holistic_atomic(rows):
    if not rows:
        return []
    holistic = [bool_value(r, ["holistic_correct", "direct_correct"]) for r in rows]
    atomic = [bool_value(r, ["atomic_correct", "tdcot_correct"]) for r in rows]
    return [
        f"- pairwise trials: {len(rows)}",
        f"- holistic accuracy: {sum(holistic)}/{len(holistic)} = {mean(holistic):.3f}",
        f"- atomic verification accuracy: {sum(atomic)}/{len(atomic)} = {mean(atomic):.3f}",
        f"- gap: {mean(atomic) - mean(holistic):+.3f}",
    ]


def summarize_failures(rows):
    if not rows:
        return []
    counts = Counter(str(r.get("category", "unknown")) for r in rows)
    total = sum(counts.values())
    out = [f"- categorized failures: {total}"]
    for category, count in counts.most_common():
        out.append(f"- {category}: {count}/{total} = {count / total:.3f}")

    a = [str(r.get("annotator_a_category", "")) for r in rows]
    b = [str(r.get("annotator_b_category", "")) for r in rows]
    if any(a) and any(b):
        labels = sorted(set(a) | set(b))
        kappas = []
        for label in labels:
            aa = [x == label for x in a]
            bb = [x == label for x in b]
            kappas.append(cohen_kappa(aa, bb))
        out.append(f"- mean one-vs-rest Cohen kappa: {mean(kappas):.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomposition", default=None, help="JSONL with decomposition annotation fields.")
    ap.add_argument("--holistic-atomic", default=None, help="JSONL with holistic_correct and atomic_correct fields.")
    ap.add_argument("--failures", default=None, help="JSONL with failure taxonomy category fields.")
    ap.add_argument("--output", default="outputs/diagnostics/diagnostic_summary.md")
    args = ap.parse_args()

    sections = []

    decomp = summarize_decomposition(load_jsonl(args.decomposition))
    if decomp:
        sections.append(("Decomposition Quality", decomp))

    gap = summarize_holistic_atomic(load_jsonl(args.holistic_atomic))
    if gap:
        sections.append(("Holistic-Atomic Gap", gap))

    failures = summarize_failures(load_jsonl(args.failures))
    if failures:
        sections.append(("Failure Taxonomy", failures))

    if not sections:
        raise SystemExit("no diagnostic inputs were provided")

    lines = ["# Diagnostic Summary", ""]
    for title, body in sections:
        lines.extend([f"## {title}", "", *body, ""])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print("saved to:", out)


if __name__ == "__main__":
    main()
