import json
from collections import defaultdict

direct_path = "outputs/timeblind_qwen_base.jsonl"
tdcot_path = "outputs/timeblind_qwen_tdcot.jsonl"
router_path = "outputs/router_eval_qwen.jsonl"

# Direct item predictions by question/video_path/index fallback.
direct_rows = [json.loads(l) for l in open(direct_path, encoding="utf-8")]
direct_by_index = {x["index"]: x for x in direct_rows}

# TD-CoT item predictions from pair file.
tdcot_by_index = {}
for line in open(tdcot_path, encoding="utf-8"):
    x = json.loads(line)
    for v in x["videos"]:
        tdcot_by_index[v["index"]] = {
            "prediction": v["prediction"],
            "correct": v["correct"],
            "answer": v["answer"],
            "question": x["question"],
        }

# Router decisions by question.
router_by_question = {}
for line in open(router_path, encoding="utf-8"):
    x = json.loads(line)
    router_by_question[x["question"]] = x

def eval_tau(tau):
    total = correct = 0
    routed_tdcot = routed_direct = 0
    missed_temporal = 0

    for idx, d in direct_by_index.items():
        if idx not in tdcot_by_index:
            continue
        q = d["question"]
        r = router_by_question.get(q)

        use_tdcot = False
        if r is not None and r["prediction"] == "temporal" and r["confidence"] >= tau:
            use_tdcot = True

        if use_tdcot:
            pred_ok = bool(tdcot_by_index[idx]["correct"])
            routed_tdcot += 1
        else:
            pred_ok = bool(d["correct"])
            routed_direct += 1
            if r is None or r.get("label") == "temporal":
                missed_temporal += 1

        total += 1
        correct += int(pred_ok)

    return correct, total, correct / total, routed_tdcot, routed_direct, missed_temporal

print("# TimeBlind Routed Pipeline Evaluation\n")
print("| tau | Accuracy | TD-CoT Routed | Direct Routed | Missed Temporal Items |")
print("|---:|---:|---:|---:|---:|")
for tau in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    c,t,a,rt,rd,miss = eval_tau(tau)
    print(f"| {tau:.2f} | {c}/{t} = {a:.3f} | {rt} | {rd} | {miss} |")
