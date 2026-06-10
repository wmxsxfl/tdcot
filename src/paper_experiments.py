PAPER_BENCHMARKS = ["MVB", "TC", "TB", "VideoMME-S", "VideoMME-L", "MSRVTT"]


PAPER_REFERENCE_MAIN_TABLE = [
    {
        "method": "Qwen2.5-VL-7B (Base)",
        "scores": {"MVB": 51.2, "TC": 55.8, "TB": 49.7, "VideoMME-S": 74.8, "VideoMME-L": 56.2, "MSRVTT": 72.1},
        "latency_ms": 350,
        "avg_delta": None,
    },
    {
        "method": "InternVL2-8B (Base)",
        "scores": {"MVB": 62.5, "TC": 64.7, "TB": 58.4, "VideoMME-S": 64.5, "VideoMME-L": 46.8, "MSRVTT": 74.8},
        "latency_ms": 420,
        "avg_delta": None,
    },
    {
        "method": "Generic CoT (InternVL2)",
        "scores": {"MVB": 63.8, "TC": 66.5, "TB": 62.3, "VideoMME-S": 63.4, "VideoMME-L": 45.9, "MSRVTT": 73.0},
        "latency_ms": 730,
        "avg_delta": 2.3,
    },
    {
        "method": "Auxiliary-CoT (InternVL2)",
        "scores": {"MVB": 64.9, "TC": 67.5, "TB": 63.5, "VideoMME-S": 64.0, "VideoMME-L": 46.3, "MSRVTT": 74.0},
        "latency_ms": 785,
        "avg_delta": 3.4,
    },
    {
        "method": "Localize-then-Answer (InternVL2)",
        "scores": {"MVB": 64.4, "TC": 67.9, "TB": 62.6, "VideoMME-S": 62.8, "VideoMME-L": 44.8, "MSRVTT": 71.5},
        "latency_ms": 690,
        "avg_delta": 3.1,
    },
    {
        "method": "TSP (InternVL2)",
        "scores": {"MVB": 64.9, "TC": 69.8, "TB": 65.7, "VideoMME-S": 63.7, "VideoMME-L": 45.8, "MSRVTT": 73.8},
        "latency_ms": 530,
        "avg_delta": 4.9,
    },
    {
        "method": "Generic CoT (Qwen2.5-VL)",
        "scores": {"MVB": 52.5, "TC": 57.6, "TB": 52.3, "VideoMME-S": 74.4, "VideoMME-L": 55.8, "MSRVTT": 71.8},
        "latency_ms": 690,
        "avg_delta": 1.9,
    },
    {
        "method": "Auxiliary-CoT (Qwen2.5-VL)",
        "scores": {"MVB": 53.5, "TC": 58.6, "TB": 53.6, "VideoMME-S": 74.6, "VideoMME-L": 56.0, "MSRVTT": 72.0},
        "latency_ms": 750,
        "avg_delta": 3.0,
    },
    {
        "method": "TD-CoT + Qwen2.5-VL-7B",
        "scores": {"MVB": 65.5, "TC": 69.2, "TB": 66.6, "VideoMME-S": 74.7, "VideoMME-L": 56.1, "MSRVTT": 71.9},
        "latency_ms": 760,
        "avg_delta": 14.9,
    },
    {
        "method": "TD-CoT + InternVL2-8B",
        "scores": {"MVB": 70.2, "TC": 76.6, "TB": 73.8, "VideoMME-S": 64.2, "VideoMME-L": 46.5, "MSRVTT": 74.5},
        "latency_ms": 810,
        "avg_delta": 11.7,
    },
]


PAPER_REFERENCE_ABLATION_TABLE = [
    {"variant": "Full TD-CoT", "scores": {"MVB": 70.2, "TC": 76.6, "TB": 73.8}, "avg_delta": "--"},
    {"variant": "w/o Temporal Segmentation", "scores": {"MVB": 63.2, "TC": 68.5, "TB": 65.1}, "avg_delta": "-7.9"},
    {"variant": "2-state Decomposition", "scores": {"MVB": 67.5, "TC": 73.5, "TB": 70.7}, "avg_delta": "-3.0"},
    {"variant": "5-state Decomposition (+1.7x lat.)", "scores": {"MVB": 70.3, "TC": 76.7, "TB": 74.1}, "avg_delta": "+0.2"},
    {"variant": "w/o Routing (always slow)", "scores": {"MVB": 69.7, "TC": 75.9, "TB": 73.2}, "avg_delta": "-0.6"},
    {"variant": "Free-form CoT (matched budget)", "scores": {"MVB": 65.7, "TC": 71.0, "TB": 68.2}, "avg_delta": "-5.2"},
    {"variant": "Seq. to Joint Verification", "scores": {"MVB": 67.6, "TC": 73.2, "TB": 70.1}, "avg_delta": "-3.2"},
    {"variant": "w/o Cond. Prompting", "scores": {"MVB": 68.1, "TC": 74.4, "TB": 71.8}, "avg_delta": "-2.1"},
    {"variant": "w/o Early Exit", "scores": {"MVB": 69.4, "TC": 75.6, "TB": 72.8}, "avg_delta": "-0.9"},
    {"variant": "Weaker LLM (Qwen2.5-3B)", "scores": {"MVB": 67.9, "TC": 73.5, "TB": 70.5}, "avg_delta": "-2.9"},
]


MAIN_TABLE_RUNS = [
    {
        "method": "Qwen2.5-VL-7B (Base)",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_qwen_base.jsonl",
            "TC": "outputs/tempcompass_directionality_qwen_base.jsonl",
            "TB": "outputs/timeblind_qwen_base.jsonl",
            "VideoMME-S": "outputs/general_videoqa/qwen2_5_vl/results.json",
            "VideoMME-L": "outputs/general_videoqa/qwen2_5_vl/results.json",
            "MSRVTT": "outputs/general_videoqa/qwen2_5_vl/results.json",
        },
    },
    {
        "method": "InternVL2-8B (Base)",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_base.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_base.jsonl",
            "TB": "outputs/timeblind_internvl_base.jsonl",
            "VideoMME-S": "outputs/general_videoqa/internvl2/results.json",
            "VideoMME-L": "outputs/general_videoqa/internvl2/results.json",
            "MSRVTT": "outputs/general_videoqa/internvl2/results.json",
        },
    },
    {
        "method": "Generic CoT (InternVL2)",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_generic_cot.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_generic_cot.jsonl",
            "TB": "outputs/timeblind_internvl_generic_cot.jsonl",
            "VideoMME-S": "outputs/general_videoqa/internvl2_generic_cot/results.json",
            "VideoMME-L": "outputs/general_videoqa/internvl2_generic_cot/results.json",
            "MSRVTT": "outputs/general_videoqa/internvl2_generic_cot/results.json",
        },
    },
    {
        "method": "Auxiliary-CoT (InternVL2)",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_auxiliary_cot.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_auxiliary_cot.jsonl",
            "TB": "outputs/timeblind_internvl_auxiliary_cot.jsonl",
            "VideoMME-S": "outputs/general_videoqa/internvl2_auxiliary_cot/results.json",
            "VideoMME-L": "outputs/general_videoqa/internvl2_auxiliary_cot/results.json",
            "MSRVTT": "outputs/general_videoqa/internvl2_auxiliary_cot/results.json",
        },
    },
    {
        "method": "Localize-then-Answer (InternVL2)",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_localize_answer.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_localize_answer.jsonl",
            "TB": "outputs/timeblind_internvl_localize_answer.jsonl",
            "VideoMME-S": "outputs/general_videoqa/internvl2_localize_answer/results.json",
            "VideoMME-L": "outputs/general_videoqa/internvl2_localize_answer/results.json",
            "MSRVTT": "outputs/general_videoqa/internvl2_localize_answer/results.json",
        },
    },
    {
        "method": "TSP (InternVL2)",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_tsp.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_tsp.jsonl",
            "TB": "outputs/timeblind_internvl_tsp.jsonl",
            "VideoMME-S": "outputs/general_videoqa/internvl2_tsp/results.json",
            "VideoMME-L": "outputs/general_videoqa/internvl2_tsp/results.json",
            "MSRVTT": "outputs/general_videoqa/internvl2_tsp/results.json",
        },
    },
    {
        "method": "Generic CoT (Qwen2.5-VL)",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_qwen_generic_cot.jsonl",
            "TC": "outputs/tempcompass_directionality_qwen_generic_cot.jsonl",
            "TB": "outputs/timeblind_qwen_generic_cot.jsonl",
            "VideoMME-S": "outputs/general_videoqa/qwen2_5_vl_generic_cot/results.json",
            "VideoMME-L": "outputs/general_videoqa/qwen2_5_vl_generic_cot/results.json",
            "MSRVTT": "outputs/general_videoqa/qwen2_5_vl_generic_cot/results.json",
        },
    },
    {
        "method": "Auxiliary-CoT (Qwen2.5-VL)",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_qwen_auxiliary_cot.jsonl",
            "TC": "outputs/tempcompass_directionality_qwen_auxiliary_cot.jsonl",
            "TB": "outputs/timeblind_qwen_auxiliary_cot.jsonl",
            "VideoMME-S": "outputs/general_videoqa/qwen2_5_vl_auxiliary_cot/results.json",
            "VideoMME-L": "outputs/general_videoqa/qwen2_5_vl_auxiliary_cot/results.json",
            "MSRVTT": "outputs/general_videoqa/qwen2_5_vl_auxiliary_cot/results.json",
        },
    },
    {
        "method": "TD-CoT + Qwen2.5-VL-7B",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_qwen_tdcot.jsonl",
            "TC": "outputs/tempcompass_directionality_qwen_tdcot.jsonl",
            "TB": "outputs/timeblind_qwen_tdcot.jsonl",
            "VideoMME-S": "outputs/general_videoqa/qwen2_5_vl_tdcot/results.json",
            "VideoMME-L": "outputs/general_videoqa/qwen2_5_vl_tdcot/results.json",
            "MSRVTT": "outputs/general_videoqa/qwen2_5_vl_tdcot/results.json",
        },
    },
    {
        "method": "TD-CoT + InternVL2-8B",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_tdcot.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_tdcot.jsonl",
            "TB": "outputs/timeblind_internvl_tdcot.jsonl",
            "VideoMME-S": "outputs/general_videoqa/internvl2_tdcot/results.json",
            "VideoMME-L": "outputs/general_videoqa/internvl2_tdcot/results.json",
            "MSRVTT": "outputs/general_videoqa/internvl2_tdcot/results.json",
        },
    },
]


ABLATION_RUNS = [
    {
        "variant": "Full TD-CoT",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_tdcot.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_tdcot.jsonl",
            "TB": "outputs/timeblind_internvl_tdcot.jsonl",
        },
    },
    {
        "variant": "w/o Temporal Segmentation",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_no_seg.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_no_seg.jsonl",
            "TB": "outputs/timeblind_internvl_no_seg.jsonl",
        },
    },
    {
        "variant": "2-state Decomposition",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_state2.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_state2.jsonl",
            "TB": "outputs/timeblind_internvl_state2.jsonl",
        },
    },
    {
        "variant": "5-state Decomposition",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_state5.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_state5.jsonl",
            "TB": "outputs/timeblind_internvl_state5.jsonl",
        },
    },
    {
        "variant": "w/o Routing",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_no_routing.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_no_routing.jsonl",
            "TB": "outputs/timeblind_internvl_no_routing.jsonl",
        },
    },
    {
        "variant": "Free-form CoT (matched budget)",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_freeform_cot_matched.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_freeform_cot_matched.jsonl",
            "TB": "outputs/timeblind_internvl_freeform_cot_matched.jsonl",
        },
    },
    {
        "variant": "Seq. to Joint Verification",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_joint.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_joint.jsonl",
            "TB": "outputs/timeblind_internvl_joint.jsonl",
        },
    },
    {
        "variant": "w/o Cond. Prompting",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_no_cond.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_no_cond.jsonl",
            "TB": "outputs/timeblind_internvl_no_cond.jsonl",
        },
    },
    {
        "variant": "w/o Early Exit",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_no_early_exit.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_no_early_exit.jsonl",
            "TB": "outputs/timeblind_internvl_no_early_exit.jsonl",
        },
    },
    {
        "variant": "Weaker LLM (Qwen2.5-3B)",
        "outputs": {
            "MVB": "outputs/mvbench_directionality_internvl_aux3b.jsonl",
            "TC": "outputs/tempcompass_directionality_internvl_aux3b.jsonl",
            "TB": "outputs/timeblind_internvl_aux3b.jsonl",
        },
    },
]


SUPPORTING_OUTPUTS = [
    {"name": "Router Predictions", "path": "outputs/router_eval_qwen.jsonl"},
    {"name": "Router Threshold Sweep", "path": "outputs/router_threshold_sweep.md"},
    {"name": "TimeBlind Routed Pipeline", "path": "outputs/timeblind_routed_pipeline.md"},
    {"name": "TimeBlind Bootstrap", "path": "outputs/timeblind_bootstrap.md"},
    {"name": "Latency Measurements", "path": "outputs/latency/latency.csv"},
    {"name": "TempCompass Directionality Evaluation", "path": "outputs/tempcompass_directionality_eval.md"},
    {"name": "Diagnostic Summary", "path": "outputs/diagnostics/diagnostic_summary.md"},
]


def iter_required_outputs():
    seen = set()
    for row in MAIN_TABLE_RUNS:
        for benchmark, path in row["outputs"].items():
            key = (path, row["method"], benchmark)
            if key not in seen:
                seen.add(key)
                yield {
                    "name": f"{row['method']} / {benchmark}",
                    "path": path,
                    "table": "main",
                    "expected": None,
                }
    for row in ABLATION_RUNS:
        for benchmark, path in row["outputs"].items():
            key = (path, row["variant"], benchmark)
            if key not in seen:
                seen.add(key)
                yield {
                    "name": f"{row['variant']} / {benchmark}",
                    "path": path,
                    "table": "ablation",
                    "expected": None,
                }
    for row in SUPPORTING_OUTPUTS:
        yield {
            "name": row["name"],
            "path": row["path"],
            "table": "supporting",
            "expected": None,
        }
