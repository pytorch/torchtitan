"""Run ATE-Bench Operate-and-Profile tasks (OP1-OP4) with the fixed agent.

Unlike Q&A, these tasks run/instrument/profile the framework, so the agent gets
the full toolset (edits auto-accepted) and we measure *active GPU time* alongside
the effort metrics. After each attempt the task's programmatic artifact check runs
(paper B.2.2: verify the artifact, not the path taken).

These tasks are GPU-gated: they need 8 GPUs, an MoE checkpoint, vLLM +
lm-eval-harness (OP2), and Nsight Systems (OP4). Without that substrate the agent
still runs and metrics are logged, but the artifact checks will fail/skip.

Example:
    python agent_tooling/ate_bench/runner/run_operate.py \
        --tasks op3 --runs 1 --label titan \
        --out agent_tooling/ate_bench/results/operate_titan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_config as rc  # noqa: E402
from agent_session import run_session  # noqa: E402
from checks import op1_getting_started, op2_train_evaluate  # noqa: E402
from checks import op3_routing_trace, op4_heavy_kernels  # noqa: E402

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks" / "operate_profile"
PREAMBLE = TASKS_DIR / "preamble.md"

# Operate tasks edit/run the framework: full toolset, edits auto-accepted.
ALLOWED_TOOLS = ["Read", "Grep", "Glob", "Bash", "Edit", "Write"]


def run_check(task_id: str, cfg: rc.ReproConfig) -> dict:
    """Run the task's programmatic artifact check; never raises."""
    try:
        if task_id == "op1":
            log = op1_getting_started.rerun(steps=5)
            return op1_getting_started.check(log, target_step=5).to_dict()
        if task_id == "op2":
            hs = cfg.workspace_dir() / "eval" / "hellaswag.json"
            return op2_train_evaluate.check(hs).to_dict()
        if task_id == "op3":
            return op3_routing_trace.check(cfg.routing_traces_dir()).to_dict()
        if task_id == "op4":
            return op4_heavy_kernels.check(cfg.heavy_kernels_dir()).to_dict()
    except Exception as exc:  # noqa: BLE001
        return {"passed": False, "detail": f"check errored: {exc}", "extra": {}}
    return {"passed": False, "detail": "no check for task", "extra": {}}


def discover_tasks() -> dict[str, Path]:
    return {
        p.name.split("_", 1)[0]: p
        for p in sorted(TASKS_DIR.glob("op[0-9]_*.md"))
    }


def build_prompt(task_path: Path, cfg: rc.ReproConfig) -> str:
    task_id = task_path.name.split("_", 1)[0]
    preamble = cfg.render(PREAMBLE.read_text(encoding="utf-8"), task_id).strip()
    body = task_path.read_text(encoding="utf-8").strip()
    lines = body.splitlines()
    if lines and lines[0].startswith("#"):
        body = "\n".join(lines[1:]).strip()
    body = cfg.render(body, task_id)
    return f"{preamble}\n\n{body}\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="all", help="comma list (op1,op3) or 'all'")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="titan", help="framework label (workspace/<label>/...)")
    ap.add_argument("--module", default=rc.DEFAULT.module)
    ap.add_argument("--config", default=rc.DEFAULT.config)
    ap.add_argument("--model", default=None, help="agent model (paper: Opus 4.7 @ xhigh)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--no-gpu-monitor", action="store_true")
    args = ap.parse_args(argv)

    cfg = rc.ReproConfig(module=args.module, config=args.config, label=args.label)
    all_tasks = discover_tasks()
    selected = list(all_tasks) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    unknown = [t for t in selected if t not in all_tasks]
    if unknown:
        ap.error(f"unknown tasks {unknown}; available {list(all_tasks)}")

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg.workspace_dir().mkdir(parents=True, exist_ok=True)

    print(f"ATE-Bench Operate&Profile | module={cfg.module} config={cfg.config} "
          f"label={cfg.label} model={args.model or 'default'}")
    print(f"tasks={selected} runs={args.runs} out={out}\n")

    manifest = {"config": cfg.to_dict(), "model": args.model, "results": {}}
    for task_id in selected:
        prompt = build_prompt(all_tasks[task_id], cfg)
        print(f"=== {task_id} ===")
        per_run = []
        for run_idx in range(1, args.runs + 1):
            tdir = out / task_id
            transcript = tdir / f"run{run_idx}.jsonl"
            m, code, _stderr, gpu = run_session(
                prompt, rc.REPO_ROOT, transcript,
                allowed_tools=ALLOWED_TOOLS,
                permission_mode="acceptEdits",
                model=args.model, timeout=args.timeout,
                monitor_gpu=not args.no_gpu_monitor,
            )
            check_res = run_check(task_id, cfg)
            row = (m.to_dict() if m else {"error": True, "returncode": code})
            row.update({"run": run_idx, "gpu": gpu, "correctness": check_res})
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / f"run{run_idx}.metrics.json").write_text(json.dumps(row, indent=2))
            per_run.append(row)
            turns = (m.agent_turns if m else None)
            agt = gpu.get("active_gpu_time_s")
            print(f"  run {run_idx}: turns={turns} "
                  f"gpu_time={'%.0fs' % agt if agt is not None else 'n/a'} "
                  f"correct={check_res.get('passed')} ({check_res.get('detail')})")
        manifest["results"][task_id] = per_run
        print()

    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
