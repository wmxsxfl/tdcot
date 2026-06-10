from pathlib import Path

from paper_experiments import iter_required_outputs


def file_detail(path):
    p = Path(path)
    if not p.exists():
        return "MISSING", "-"
    if p.suffix == ".jsonl":
        n = sum(1 for _ in open(p, encoding="utf-8"))
        return "OK", f"{n} lines"
    return "OK", f"{p.stat().st_size / 1024:.1f} KB"


def main():
    missing = 0
    print("# Experiment Status\n")
    print("| Table | Item | Status | Lines/Size | Path |")
    print("|---|---|---|---:|---|")
    for row in iter_required_outputs():
        status, detail = file_detail(row["path"])
        missing += int(status != "OK")
        print(f"| {row['table']} | {row['name']} | {status} | {detail} | `{row['path']}` |")

    print()
    if missing:
        print(f"Missing or unreadable outputs: {missing}")
    else:
        print("All expected outputs are present.")


if __name__ == "__main__":
    main()
