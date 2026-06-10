import argparse
import json
import shutil
import subprocess
from pathlib import Path


def build_model_args(model_path, extra, method, aux_model, router_threshold, early_exit):
    parts = [f"pretrained={model_path}"]
    if method:
        parts.append(f"tdcot_method={method}")
    if aux_model:
        parts.append(f"aux_pretrained={aux_model}")
    if method and router_threshold is not None:
        parts.append(f"router_threshold={router_threshold}")
    if method and early_exit is not None:
        parts.append(f"early_exit={early_exit}")
    if extra:
        parts.append(extra)
    return ",".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tasks", required=True, help="Comma-separated LMMs-Eval task names.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-args", default="")
    ap.add_argument("--method", default="", help="Inference method passed to the LMMs-Eval model adapter.")
    ap.add_argument("--aux-model", default="", help="Auxiliary text model path for TD-CoT and auxiliary baselines.")
    ap.add_argument("--router-threshold", type=float, default=0.8)
    ap.add_argument("--early-exit", type=float, default=0.3)
    ap.add_argument("--batch-size", default="1")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    executable = shutil.which("lmms-eval")
    if executable is None and not args.dry_run:
        raise SystemExit(
            "lmms-eval was not found on PATH. Install LMMs-Eval or run with --dry-run to record the command only."
        )

    cmd = [
        executable or "lmms-eval",
        "--model",
        args.model_name,
        "--model_args",
        build_model_args(
            args.model_path,
            args.model_args,
            args.method,
            args.aux_model,
            args.router_threshold,
            args.early_exit,
        ),
        "--tasks",
        args.tasks,
        "--batch_size",
        args.batch_size,
        "--output_path",
        str(output_dir),
    ]

    manifest = {
        "model_name": args.model_name,
        "model_path": args.model_path,
        "method": args.method or "base",
        "aux_model": args.aux_model,
        "router_threshold": args.router_threshold,
        "early_exit": args.early_exit,
        "tasks": args.tasks.split(","),
        "output_dir": str(output_dir),
        "command": cmd,
    }
    (output_dir / "command.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(" ".join(cmd))

    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
