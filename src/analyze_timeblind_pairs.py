import json, math
from collections import Counter, defaultdict

direct_p = "outputs/timeblind_qwen_base.jsonl"
tdcot_p = "outputs/timeblind_qwen_tdcot.jsonl"

direct = {}
for line in open(direct_p, encoding="utf-8"):
    x = json.loads(line)
    direct[x["index"]] = x

pairs = [json.loads(l) for l in open(tdcot_p, encoding="utf-8")]

def clamp(x):
    return min(max(float(x), 1e-6), 1 - 1e-6)

def logit(x):
    x = clamp(x)
    return math.log(x / (1 - x))

def score(ps, mode):
    if mode == "prod_all": return ps[0] * ps[1] * ps[2]
    if mode == "sum_all": return sum(ps)
    if mode == "sum_logit_all": return sum(logit(x) for x in ps)
    if mode == "min_all": return min(ps)
    if mode == "initial_only": return ps[0]
    if mode == "transition_only": return ps[1]
    if mode == "end_only": return ps[2]
    if mode == "sum_trans_end": return ps[1] + ps[2]
    if mode == "prod_trans_end": return ps[1] * ps[2]
    raise ValueError(mode)

def eval_mode(mode):
    pt=po=it=io=0
    for x in pairs:
        vids = x["videos"]
        scores = []
        for v in vids:
            ps = [q["p_yes"] for q in v["probs"]]
            scores.append(score(ps, mode))
        yes_i = max(range(len(scores)), key=lambda i: scores[i])
        pt += 1
        po += int(vids[yes_i]["answer"] == "yes")
        for i,v in enumerate(vids):
            pred = "yes" if i == yes_i else "no"
            it += 1
            io += int(pred == v["answer"])
    return io,it,io/it,po,pt,po/pt

def eval_early(th):
    pt=po=it=io=steps=0
    for x in pairs:
        vids=x["videos"]
        scores=[]
        for v in vids:
            prod=1.0
            used=0
            for q in v["probs"]:
                used += 1
                p = q["p_yes"]
                if p < th:
                    prod = 0.0
                    break
                prod *= p
            steps += used
            scores.append(prod)
        yes_i=max(range(len(scores)), key=lambda i:scores[i])
        pt += 1
        po += int(vids[yes_i]["answer"] == "yes")
        for i,v in enumerate(vids):
            pred = "yes" if i == yes_i else "no"
            it += 1
            io += int(pred == v["answer"])
    return io,it,io/it,po,pt,po/pt,steps/it

print("# TimeBlind Pair Analysis\n")

dc=dt=0
for i,x in direct.items():
    if i < 200:
        dt += 1
        dc += int(x["correct"])
print(f"Direct Qwen2.5-VL matched items: {dc}/{dt} = {dc/dt:.3f}\n")

print("## Score Modes")
for m in ["prod_all","sum_all","sum_logit_all","min_all","initial_only","transition_only","end_only","sum_trans_end","prod_trans_end"]:
    io,it,ia,po,pt,pa = eval_mode(m)
    print(f"{m:16s} item={io}/{it}={ia:.3f} pair={po}/{pt}={pa:.3f}")

print("\n## Post-hoc Early Exit")
for th in [0.05,0.10,0.15,0.20,0.25,0.30]:
    io,it,ia,po,pt,pa,avg = eval_early(th)
    print(f"threshold={th:.2f} item={io}/{it}={ia:.3f} pair={po}/{pt}={pa:.3f} avg_steps={avg:.2f}")

print("\n## Direct vs TD-CoT Overlap")
both=direct_only=tdcot_only=both_wrong=0
for x in pairs:
    for v in x["videos"]:
        idx=v["index"]
        d_ok = direct[idx]["correct"]
        t_ok = v["correct"]
        if d_ok and t_ok: both += 1
        elif d_ok and not t_ok: direct_only += 1
        elif not d_ok and t_ok: tdcot_only += 1
        else: both_wrong += 1
print("both_correct:", both)
print("direct_only:", direct_only)
print("tdcot_only:", tdcot_only)
print("both_wrong:", both_wrong)

print("\n## Pair Error Examples")
shown=0
for x in pairs:
    if x["pair_correct"]:
        continue
    print("\nQ:", x["question"])
    print("triplet:", x["triplet"])
    for v in x["videos"]:
        print(" ", v["index"], "gold=", v["answer"], "tdcot=", v["prediction"], "score=", round(v["score"],6),
              "direct=", direct[v["index"]]["prediction"])
        print("   probs:", [(q["name"], round(q["p_yes"],3)) for q in v["probs"]])
    shown += 1
    if shown >= 5:
        break
