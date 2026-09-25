"""Run ATE-Bench New-Feature tasks (NF1-NF4) with the fixed agent.

Each attempt integrates a published architecture (Diff attention / DynMoE / MoBA /
MoE++) into the base MoE model end-to-end. To match the paper (B.3.2) each attempt
runs in an isolated git worktree off ``main`` so the change can be diffed against
``main`` and attempts don't collide. Correctness has two axes:

  1. loss axis  — 64-step CE loss decreases and stays finite (checks/nf_loss_curve)
  2. rule axis  — an independent LLM judge scores the diff against 3 fixed rules
                  (runner/judge.py; paper used claude-opus-4-7 @ xhigh)

An attempt is correct iff both axes pass. We also log active GPU time.

GPU-gated: needs 8 GPUs + C4 data to actually train. Without it the agent still
runs, the diff + judge still work, but the loss check will not pass.

Example:
    python agent_tooling/ate_bench/runner/run_feature.py \
        --tasks nf1 --runs 1 --label titan \
        --judge-model claude-opus-4-7 \
        --out agent_tooling/ate_bench/results/feature_titan
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import repro_config as rc  # noqa: E402
import judge as judge_mod  # noqa: E402
import worktrees  # noqa: E402
from agent_session import run_session  # noqa: E402
from checks import nf_loss_curve  # noqa: E402

TASKS_DIR = Path(__file__).resolve().parent.parent / "tasks" / "new_feature"
PREAMBLE = TASKS_DIR / "preamble.md"
REFERENCES = TASKS_DIR / "references.md"
RULES_DIR = TASKS_DIR / "rules"

ALLOWED_TOOLS = ["Read", "Grep", "Glob", "Bash", "Edit", "Write"]


def discover_tasks() -> dict[str, Path]:
    return {p.name.split("_", 1)[0]: p for p in sorted(TASKS_DIR.glob("nf[0-9]_*.md"))}


def build_prompt(task_path: Path, cfg: rc.ReproConfig) -> str:
    task_id = task_path.name.split("_", 1)[0]
    ws_root = cfg.workspace.resolve()
    preamble = cfg.render(PREAMBLE.read_text(), task_id, workspace_root=ws_root).strip()
    body = task_path.read_text().strip()
    refs = REFERENCES.read_text().strip()
    return f"{preamble}\n\n{body}\n\n---\n## Reference materials\n{refs}\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="all", help="comma list (nf1,nf3) or 'all'")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--label", default="titan")
    ap.add_argument("--module", default=rc.DEFAULT.module)
    ap.add_argument("--config", default=rc.DEFAULT.config)
    ap.add_argument("--dataset", default=rc.DEFAULT.dataset, help="training dataset (c4 / c4_test)")
    ap.add_argument(
        "--global-batch-size", type=int, default=rc.DEFAULT.global_batch_size,
        help="paper uses 1024 (for 30B models); use ~64 for the debug model so 64 "
        "steps show a decreasing loss without grad-accum blowup",
    )
    ap.add_argument("--model", default=None, help="implementing agent model")
    ap.add_argument("--judge-model", default=None, help="judge model (paper: claude-opus-4-7 @ xhigh)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=int, default=10800)
    ap.add_argument("--no-gpu-monitor", action="store_true")
    ap.add_argument("--keep-worktrees", action="store_true", help="don't remove worktrees after")
    args = ap.parse_args(argv)

    cfg = rc.ReproConfig(
        module=args.module, config=args.config, label=args.label,
        dataset=args.dataset, global_batch_size=args.global_batch_size,
    )
    all_tasks = discover_tasks()
    selected = list(all_tasks) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    unknown = [t for t in selected if t not in all_tasks]
    if unknown:
        ap.error(f"unknown tasks {unknown}; available {list(all_tasks)}")

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    wt_root = rc.ATE_ROOT / "worktrees"

    # Base worktrees on the CURRENT branch tip (it carries the ate_bench tooling,
    # which is not on main), and diff the agent's changes against that same point so
    # the judge sees only the agent's model-code edits, not the tooling.
    import subprocess as _sp
    base_ref = _sp.run(
        ["git", "-C", str(rc.REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "HEAD"

    print(f"ATE-Bench New-Feature | module={cfg.module} config={cfg.config} "
          f"label={cfg.label} model={args.model or 'default'} judge={args.judge_model or 'default'}")
    print(f"tasks={selected} runs={args.runs} out={out}\n")

    manifest = {"config": cfg.to_dict(), "model": args.model, "judge_model": args.judge_model, "results": {}}
    for task_id in selected:
        prompt = build_prompt(all_tasks[task_id], cfg)
        rules_text = (RULES_DIR / f"{task_id}.rules.md").read_text()
        print(f"=== {task_id} ===")
        per_run = []
        for run_idx in range(1, args.runs + 1):
            tdir = out / task_id
            tdir.mkdir(parents=True, exist_ok=True)
            wt = wt_root / f"{task_id}-run{run_idx}"
            row: dict = {"run": run_idx}
            try:
                worktrees.create_worktree(rc.REPO_ROOT, wt, base=base_ref)
            except Exception as exc:  # noqa: BLE001
                print(f"  run {run_idx}: worktree create failed: {exc}")
                row.update({"error": True, "detail": str(exc)})
                per_run.append(row)
                continue
            try:
                transcript = tdir / f"run{run_idx}.jsonl"
                m, code, _stderr, gpu = run_session(
                    prompt, wt, transcript,
                    allowed_tools=ALLOWED_TOOLS, permission_mode="acceptEdits",
                    model=args.model, timeout=args.timeout,
                    monitor_gpu=not args.no_gpu_monitor,
                )
                diff = worktrees.diff_vs_base(wt, base_ref)
                (tdir / f"run{run_idx}.diff").write_text(diff)

                # Axis 1: loss curve (log saved by the agent under the shared workspace).
                log_path = cfg.workspace_dir() / task_id / "train.log"
                if log_path.exists():
                    loss_res = nf_loss_curve.check(log_path.read_text(errors="ignore")).to_dict()
                else:
                    loss_res = {"passed": False, "detail": f"no train.log at {log_path}"}

                # Axis 2: independent rule judge over the diff.
                verdict = judge_mod.judge_feature(rules_text, diff, model=args.judge_model)

                correct = bool(loss_res.get("passed")) and verdict.get("overall") == "PASS"
                row.update({
                    **(m.to_dict() if m else {"error": True, "returncode": code}),
                    "gpu": gpu,
                    "correctness": {
                        "correct": correct,
                        "loss_axis": loss_res,
                        "rule_axis": verdict,
                    },
                })
                (tdir / f"run{run_idx}.metrics.json").write_text(json.dumps(row, indent=2))
                turns = (m.agent_turns if m else None)
                agt = gpu.get("active_gpu_time_s")
                print(f"  run {run_idx}: turns={turns} "
                      f"gpu_time={'%.0fs' % agt if agt is not None else 'n/a'} "
                      f"loss={loss_res.get('passed')} judge={verdict.get('overall')} "
                      f"correct={correct}")
            finally:
                if not args.keep_worktrees:
                    worktrees.remove_worktree(rc.REPO_ROOT, wt)
            per_run.append(row)
        manifest["results"][task_id] = per_run
        print()

    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {out / 'run_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
