import argparse
import json
import random
from pathlib import Path


GENERAL_QUESTIONS = [
    "What color is the car in the video?",
    "How many people are visible in the scene?",
    "What object is on the table?",
    "Is there a dog in the video?",
    "What is the person wearing?",
    "What room is shown in the video?",
    "Is the sky visible?",
    "What sport is being played?",
    "How many vehicles are present?",
    "What kind of food is shown?",
    "Is the person indoors or outdoors?",
    "What color is the shirt?",
    "What object is the person holding?",
    "Is there water in the scene?",
    "What animal appears in the video?",
    "How many chairs can be seen?",
    "What is written on the sign?",
    "Is the video filmed in a kitchen?",
    "What is the main object in the frame?",
    "What color is the ball?",
    "Is a bicycle visible?",
    "What type of building is shown?",
    "Is the person wearing glasses?",
    "How many cups are on the table?",
    "What instrument is visible?",
    "Is there snow in the video?",
    "What is the person sitting on?",
    "What color is the wall?",
    "Is a phone visible?",
    "What is the background location?",
    "How many children are shown?",
    "Is the scene at night?",
    "What type of vehicle is shown?",
    "Is a computer visible?",
    "What color is the door?",
    "What object is next to the person?",
    "Is the person alone?",
    "What kind of bag is visible?",
    "Is there a tree in the scene?",
    "What is the floor made of?",
    "What color are the shoes?",
    "Is the person in a bedroom?",
    "What object is hanging on the wall?",
    "How many windows are visible?",
    "Is a laptop on the desk?",
    "What color is the cup?",
    "What type of road is shown?",
    "Is there a mirror in the scene?",
    "What furniture is visible?",
    "Is the person holding a book?",
]


def load_timeblind_temporal(path, limit):
    rows = []
    seen = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            x = json.loads(line)
            question = str(x["question"]).strip()
            if question in seen:
                continue
            seen.add(question)
            rows.append({
                "source": "timeblind",
                "question": question,
                "label": "temporal",
            })
            if len(rows) >= limit:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeblind-data", default="/data/datasets/TimeBlind/data.jsonl")
    ap.add_argument("--output", default="outputs/router_eval_set.jsonl")
    ap.add_argument("--temporal-limit", type=int, default=100)
    ap.add_argument("--general-limit", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    temporal = load_timeblind_temporal(args.timeblind_data, args.temporal_limit)
    if len(temporal) < args.temporal_limit:
        raise RuntimeError(
            f"Only found {len(temporal)} unique TimeBlind questions, "
            f"but --temporal-limit={args.temporal_limit}."
        )

    general_questions = list(GENERAL_QUESTIONS)
    if args.general_limit > len(general_questions):
        repeats = (args.general_limit + len(general_questions) - 1) // len(general_questions)
        general_questions = (general_questions * repeats)[:args.general_limit]
    else:
        general_questions = general_questions[:args.general_limit]

    general = [
        {"source": "heldout_general", "question": q, "label": "general"}
        for q in general_questions
    ]

    rows = temporal + general
    random.Random(args.seed).shuffle(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("saved:", out)
    print("rows:", len(rows))
    print("temporal:", len(temporal))
    print("general:", len(general))


if __name__ == "__main__":
    main()
