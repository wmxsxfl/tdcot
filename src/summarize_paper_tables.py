import argparse
import json
import math
import sys
from pathlib import Path

from paper_experiments import (
    ABLATION_RUNS,
    MAIN_TABLE_RUNS,
    PAPER_BENCHMARKS,
    PAPER_REFERENCE_ABLATION_TABLE,
    PAPER_REFERENCE_MAIN_TABLE,
)


GENERAL_TASK_ALIASES = {
    "VideoMME-S": [
        "videomme_short_wo_subtitle",
        "videomme_short",
        "videomme_s",
        "VideoMME-S",
    ],
    "VideoMME-L": [
        "videomme_long_wo_subtitle",
        "videomme_long",
        "videomme_l",
        "VideoMME-L",
    ],
    "MSRVTT": [
        "msrvtt_qa",
        "msrvtt",
        "MSRVTT",
    ],
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def as_accuracy(value):
    value = float(value)
    if value <= 1.0:
        return value * 100.0
    return value


def jsonl_accuracy(path):
    rows = load_jsonl(path)
    if not rows:
        return None

    first = rows[0]
    if "videos" in first:
        correct = 0
        total = 0
        for row in rows:
            for video in row.get("videos", []):
                total += 1
                correct += int(bool(video.get("correct", False)))
        return 100.0 * correct / total if total else None

    if "correct" in first:
        total = len(rows)
        correct = sum(int(bool(row.get("correct", False))) for row in rows)
        return 100.0 * correct / total if total else None

    if "correct_mode" in first:
        total = len(rows)
        correct = sum(int(bool(row.get("correct_mode", False))) for row in rows)
        return 100.0 * correct / total if total else None

    return None


def find_metric(obj):
    if isinstance(obj, dict):
        for key in ["acc", "accuracy", "exact_match", "score", "acc,none"]:
            if key in obj and isinstance(obj[key], (int, float)):
                return as_accuracy(obj[key])
    if isinstance(obj, (int, float)):
        return as_accuracy(obj)
    return None


def general_accuracy(path, benchmark):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    results = data.get("results", data)
    aliases = GENERAL_TASK_ALIASES.get(benchmark, [benchmark])

    for alias in aliases:
        if alias in results:
            metric = find_metric(results[alias])
            if metric is not None:
                return metric

    for alias in aliases:
        alias_low = alias.lower()
        for key, value in results.items():
            if alias_low == str(key).lower() or alias_low in str(key).lower():
                metric = find_metric(value)
                if metric is not None:
                    return metric

    return None


def compute_cell(path, benchmark):
    p = Path(path)
    if not p.exists():
        return None, "MISSING"
    if p.suffix == ".jsonl":
        value = jsonl_accuracy(p)
    elif p.suffix == ".json":
        value = general_accuracy(p, benchmark)
    else:
        value = None

    if value is None or math.isnan(value):
        return None, "UNREADABLE"
    return value, "OK"


def fmt_score(value, status):
    if status != "OK":
        return status
    return f"{value:.1f}"


def fmt_delta(value):
    if value is None:
        return "--"
    return f"{value:+.1f}"


def average(values):
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def print_main_table(strict=False):
    print("# Main Results\n")
    headers = ["Method", *PAPER_BENCHMARKS, "Avg. Delta"]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|")

    missing = []
    base_by_family = {}
    computed = []

    for row in MAIN_TABLE_RUNS:
        method = row["method"]
        values = {}
        cells = []
        for benchmark in PAPER_BENCHMARKS:
            value, status = compute_cell(row["outputs"][benchmark], benchmark)
            if status != "OK":
                missing.append((method, benchmark, row["outputs"][benchmark], status))
            values[benchmark] = value
            cells.append(fmt_score(value, status))

        method_low = method.lower()
        if "qwen2.5-vl-7b (base)" in method_low:
            base_by_family["qwen"] = average([values[b] for b in ["MVB", "TC", "TB"]])
        if "internvl2-8b (base)" in method_low:
            base_by_family["internvl"] = average([values[b] for b in ["MVB", "TC", "TB"]])

        family = "qwen" if "qwen" in method_low else "internvl" if "internvl" in method_low else None
        baseline = base_by_family.get(family)
        current = average([values[b] for b in ["MVB", "TC", "TB"]])
        delta = current - baseline if baseline is not None and current is not None and "base" not in method_low else None
        computed.append((method, cells, delta))

    for method, cells, delta in computed:
        print("| " + " | ".join([method, *cells, fmt_delta(delta)]) + " |")

    if missing:
        print("\n## Missing or Unreadable Main Outputs\n")
        print("| Method | Benchmark | Status | Path |")
        print("|---|---|---|---|")
        for method, benchmark, path, status in missing:
            print(f"| {method} | {benchmark} | {status} | `{path}` |")
        if strict:
            return 1
    return 0


def print_ablation_table(strict=False):
    print("\n# Ablation Results\n")
    print("| Variant | MVB | TC | TB | Avg. Delta vs. Full |")
    print("|---|---:|---:|---:|---:|")

    missing = []
    rows = []
    full_avg = None
    for row in ABLATION_RUNS:
        values = {}
        cells = []
        for benchmark in ["MVB", "TC", "TB"]:
            value, status = compute_cell(row["outputs"][benchmark], benchmark)
            if status != "OK":
                missing.append((row["variant"], benchmark, row["outputs"][benchmark], status))
            values[benchmark] = value
            cells.append(fmt_score(value, status))
        avg = average(values.values())
        if row["variant"] == "Full TD-CoT":
            full_avg = avg
        delta = avg - full_avg if avg is not None and full_avg is not None and row["variant"] != "Full TD-CoT" else None
        rows.append((row["variant"], cells, delta))

    for variant, cells, delta in rows:
        print("| " + " | ".join([variant, *cells, fmt_delta(delta)]) + " |")

    if missing:
        print("\n## Missing or Unreadable Ablation Outputs\n")
        print("| Variant | Benchmark | Status | Path |")
        print("|---|---|---|---|")
        for variant, benchmark, path, status in missing:
            print(f"| {variant} | {benchmark} | {status} | `{path}` |")
        if strict:
            return 1
    return 0


def print_reference_tables():
    print("# Paper Reference Values\n")
    print("These are the values reported in the paper. They are not used to compute experiment tables.\n")
    print("## Main Table\n")
    print("| Method | " + " | ".join(PAPER_BENCHMARKS) + " | Lat. (ms) | Avg. Delta |")
    print("|---|" + "|".join(["---:"] * (len(PAPER_BENCHMARKS) + 2)) + "|")
    for row in PAPER_REFERENCE_MAIN_TABLE:
        cells = [f"{row['scores'][b]:.1f}" for b in PAPER_BENCHMARKS]
        delta = "--" if row["avg_delta"] is None else f"+{row['avg_delta']:.1f}"
        print("| " + " | ".join([row["method"], *cells, str(row["latency_ms"]), delta]) + " |")

    print("\n## Ablation Table\n")
    print("| Variant | MVB | TC | TB | Avg. Delta vs. Full |")
    print("|---|---:|---:|---:|---:|")
    for row in PAPER_REFERENCE_ABLATION_TABLE:
        print(
            "| "
            + " | ".join([
                row["variant"],
                f"{row['scores']['MVB']:.1f}",
                f"{row['scores']['TC']:.1f}",
                f"{row['scores']['TB']:.1f}",
                row["avg_delta"],
            ])
            + " |"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=["all", "main", "ablation", "reference"], default="all")
    ap.add_argument("--strict", action="store_true", help="Exit non-zero if a required output is missing or unreadable.")
    args = ap.parse_args()

    code = 0
    if args.table in ["all", "main"]:
        code |= print_main_table(strict=args.strict)
    if args.table in ["all", "ablation"]:
        code |= print_ablation_table(strict=args.strict)
    if args.table == "reference":
        print_reference_tables()

    if code:
        sys.exit(code)


if __name__ == "__main__":
    main()
