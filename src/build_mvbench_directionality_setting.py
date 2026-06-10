import argparse
import json
from pathlib import Path

keywords = [
    "after", "before", "first", "then", "next",
    "beginning", "middle", "end",
    "from", "to",
    "open", "close", "closing", "opening",
    "change", "changing", "turn", "become",
    "write first", "letter did",
    "scene", "transition",
]

exclude = [
    "temperature",
    "hot",
    "cold",
    "stationary",
    "moving direction",
    "what direction",
]

def text_of(x):
    return (x["question"] + " " + " ".join(map(str, x["candidates"]))).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/data/datasets/MVBench/mvbench_directional.jsonl")
    ap.add_argument("--output", default="/data/datasets/MVBench/mvbench_directionality_setting.jsonl")
    args = ap.parse_args()

    src = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    by = {}
    with open(out, "w", encoding="utf-8") as f:
        for line in open(src, encoding="utf-8"):
            x = json.loads(line)
            t = text_of(x)
            if any(k in t for k in keywords) and not any(k in t for k in exclude):
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
                n += 1
                by[x["task"]] = by.get(x["task"], 0) + 1

    print("saved", out)
    print("items", n)
    print("by task", by)

    print("\nexamples:")
    for i,line in zip(range(20), open(out, encoding="utf-8")):
        x=json.loads(line)
        print("\n", x["id"], x["task"])
        print(x["question"])
        print(x["candidates"])
        print("answer:", x["answer"])


if __name__ == "__main__":
    main()
