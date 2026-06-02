"""Run ATE-Bench Q&A tasks against a target framework with a fixed agent.

Reproduces the Q&A category of ATE-Bench (PithTrain, arXiv:2605.31463): a *fixed*
coding agent is pointed at a framework codebase and asked code-property questions
with read-only tools; we log the effort metrics. The paper's fixed agent was
Claude Code (Opus 4.7, xhigh effort), each task run 3 times with medians reported.

The agent is invoked headless:
    claude -p --output-format stream-json --verbose
      --allowedTools Read Grep Glob Bash
      --disallowedTools Edit Write NotebookEdit
run with the target repo as the working directory so the agent explores *that*
codebase. The universal instruction (tasks/qa/universal_instruction.txt) is
prepended verbatim to each question.

Example:
    python agent_tooling/ate_bench/runner/run_qa.py \
        --repo /home/bahuang/local/torchtitan_autodev \
        --tasks all --runs 3 --out agent_tooling/ate_bench/results/titan

Then summarize with:
    python agent_tooling/ate_bench/runner/aggregate.py agent_tooling/ate_bench/results/titan
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

# Allow running as a plain script (python .../run_qa.py) by importing the sibling.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as metrics_mod  # noqa: E402

QA_DIR = Path(__file__).resolve().parent.parent / "tasks" / "qa"
UNIVERSAL = QA_DIR / "universal_instruction.txt"

# Faithful to the paper: read-only exploration tools, mutating tools disabled.
ALLOWED_TOOLS = ["Read", "Grep", "Glob", "Bash"]
DISALLOWED_TOOLS = ["Edit", "Write", "NotebookEdit"]


def discover_tasks() -> dict[str, Path]:
    """Return {task_id: prompt_path} for q01..q12, sorted by id."""
    tasks: dict[str, Path] = {}
    for p in sorted(QA_DIR.glob("q[0-9][0-9]_*.md")):
        task_id = p.name.split("_", 1)[0]  # e.g. "q01"
        tasks[task_id] = p
    return tasks


def build_prompt(task_path: Path) -> str:
    universal = UNIVERSAL.read_text(encoding="utf-8").strip()
    body = task_path.read_text(encoding="utf-8").strip()
    # Drop the leading "# Qn: Title" heading; keep the question text only, as the
    # paper prepends the universal instruction followed by the query text.
    lines = body.splitlines()
    if lines and lines[0].startswith("#"):
        body = "\n".join(lines[1:]).strip()
    return f"{universal}\n\n{body}\n"


def git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def run_once(
    prompt: str,
    repo: Path,
    transcript_path: Path,
    model: str | None,
    timeout: int,
) -> tuple[metrics_mod.TaskMetrics | None, int, str]:
    """Run the fixed agent once; write transcript; return (metrics, returncode, stderr)."""
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "stream-json",
        "--verbose",
        "--allowedTools",
        *ALLOWED_TOOLS,
        "--disallowedTools",
        *DISALLOWED_TOOLS,
    ]
    if model:
        cmd += ["--model", model]

    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        cwd=str(repo),
        timeout=timeout,
    )
    transcript_path.write_text(proc.stdout, encoding="utf-8")
    if proc.stderr:
        transcript_path.with_suffix(".stderr.log").write_text(
            proc.stderr, encoding="utf-8"
        )
    m = None
    if proc.stdout.strip():
        try:
            m = metrics_mod.parse_transcript(transcript_path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"    ! failed to parse transcript: {exc}", file=sys.stderr)
    return m, proc.returncode, proc.stderr


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo",
        required=True,
        type=Path,
        help="path to the framework checkout the agent should explore",
    )
    ap.add_argument(
        "--tasks",
        default="all",
        help="comma-separated task ids (e.g. q01,q05) or 'all'",
    )
    ap.add_argument("--runs", type=int, default=3, help="attempts per task (paper: 3)")
    ap.add_argument(
        "--model",
        default=None,
        help="agent model override (default: CLI default). Paper used Opus 4.7 "
        "at xhigh effort; set this to match the comparison you want.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output directory for transcripts + metrics",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="per-attempt timeout in seconds (default 1800)",
    )
    args = ap.parse_args(argv)

    repo = args.repo.expanduser().resolve()
    if not repo.is_dir():
        ap.error(f"--repo {repo} is not a directory")
    if not UNIVERSAL.exists():
        ap.error(f"missing universal instruction: {UNIVERSAL}")

    all_tasks = discover_tasks()
    if args.tasks.strip().lower() == "all":
        selected = list(all_tasks)
    else:
        selected = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in selected if t not in all_tasks]
        if unknown:
            ap.error(f"unknown task ids {unknown}; available: {list(all_tasks)}")

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    commit = git_commit(repo)

    run_manifest = {
        "repo": str(repo),
        "repo_commit": commit,
        "model": args.model,
        "runs_per_task": args.runs,
        "tasks": selected,
        "allowed_tools": ALLOWED_TOOLS,
        "disallowed_tools": DISALLOWED_TOOLS,
        "results": {},
    }

    print(f"ATE-Bench Q&A  |  repo={repo}  commit={commit}  model={args.model or 'default'}")
    print(f"tasks={selected}  runs={args.runs}  out={out}\n")

    for task_id in selected:
        task_path = all_tasks[task_id]
        prompt = build_prompt(task_path)
        title = task_path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip()
        print(f"=== {task_id}: {title} ===")
        per_run = []
        for run_idx in range(1, args.runs + 1):
            tdir = out / task_id
            tdir.mkdir(parents=True, exist_ok=True)
            transcript_path = tdir / f"run{run_idx}.jsonl"
            t0 = time.time()
            m, rc, stderr = run_once(
                prompt, repo, transcript_path, args.model, args.timeout
            )
            wall = time.time() - t0
            if m is None:
                tail = (stderr or "").strip().splitlines()[-1:] or ["(no output)"]
                print(f"  run {run_idx}: FAILED rc={rc} wall={wall:.0f}s  {tail[0]}")
                per_run.append({"run": run_idx, "error": True, "returncode": rc})
                continue
            md = m.to_dict()
            md["run"] = run_idx
            md["wall_time_s"] = wall
            (tdir / f"run{run_idx}.metrics.json").write_text(
                json.dumps(md, indent=2), encoding="utf-8"
            )
            per_run.append(md)
            print(
                f"  run {run_idx}: turns={m.agent_turns}  "
                f"ctx={_k(m.per_turn_context)}  out={_k(m.output_tokens)}  "
                f"dur={m.session_duration_s:.0f}s  err={m.is_error}"
            )
        run_manifest["results"][task_id] = per_run
        print()

    (out / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote run manifest -> {out / 'run_manifest.json'}")
    print(f"Summarize with: python {Path(__file__).with_name('aggregate.py')} {out}")
    return 0


def _k(x: float | None) -> str:
    return "-" if x is None else f"{x / 1000:.1f}K"


if __name__ == "__main__":
    raise SystemExit(main())
