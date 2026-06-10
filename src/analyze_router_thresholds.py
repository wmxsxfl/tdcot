import json

p = "outputs/router_eval_qwen.jsonl"
rows = [json.loads(l) for l in open(p, encoding="utf-8")]

print("| tau | Temporal Recall | Temporal Precision among TD-CoT routes | General Bypass Rate | TD-CoT Route Rate |")
print("|---:|---:|---:|---:|---:|")

for tau in [0.50,0.60,0.70,0.75,0.80,0.85,0.90,0.95]:
    temporal_total = sum(x["label"] == "temporal" for x in rows)
    general_total = sum(x["label"] == "general" for x in rows)

    routed = []
    for x in rows:
        route = "tdcot" if (x["prediction"] == "temporal" and x["confidence"] >= tau) else "direct"
        routed.append((x, route))

    temporal_routed = sum(x["label"] == "temporal" and route == "tdcot" for x,route in routed)
    total_routed = sum(route == "tdcot" for x,route in routed)
    general_bypass = sum(x["label"] == "general" and route == "direct" for x,route in routed)

    temporal_recall = temporal_routed / temporal_total
    precision = temporal_routed / total_routed if total_routed else 0
    general_bypass_rate = general_bypass / general_total
    route_rate = total_routed / len(rows)

    print(f"| {tau:.2f} | {temporal_recall:.3f} | {precision:.3f} | {general_bypass_rate:.3f} | {route_rate:.3f} |")
