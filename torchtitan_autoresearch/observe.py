# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Observer: the single human-facing control + follow-along agent.

This is the ONLY way to start autoresearch. It is a *different* actor from the
creating agent (ARCHITECTURE.md section 6): it launches the creating loop as a
separate background process and then only *reads* the artifacts that loop writes
(ledger, state, report, loop stdout) to broadcast status and answer questions. It
never proposes candidates and never touches the gate, the ledger, or the branch.

    python -m torchtitan_autoresearch.observe start --tag may30-qwen3
    python -m torchtitan_autoresearch.observe start --tag may30-qwen3 --eval-dataset c4_validation
    python -m torchtitan_autoresearch.observe watch  --tag may30-qwen3
    python -m torchtitan_autoresearch.observe status --tag may30-qwen3
    python -m torchtitan_autoresearch.observe ask    --tag may30-qwen3 "any free-form question"
    python -m torchtitan_autoresearch.observe stop   --tag may30-qwen3

Free-form questions are answered by an LLM (the local `claude` CLI / AI gateway,
no API key) grounded in the run's read-only state; offline it falls back to a
keyword matcher. The LLM only reads state -- it still cannot touch the gate.

`start` launches the creating loop and OWNS its lifetime: exiting the start
observer for any reason (Ctrl-C, normal end, SIGTERM/SIGHUP, terminal close)
tears down the loop and its GPU children. `watch` is a passive re-attach viewer
that does not own the experiment; `stop` ends it explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from torchtitan_autoresearch.ledger import Ledger


def _run_dir(tag: str | None, run_dir: str | None) -> str:
    if run_dir:
        return run_dir
    if tag:
        return f"/tmp/ar_{tag}"
    raise ValueError("provide --tag or --run-dir")


class Observer:
    """Read-only view over a run directory: ledger, state, report, loop stdout."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.ledger = Ledger(os.path.join(run_dir, "results.tsv"))
        self.statefile = os.path.join(run_dir, "state.json")
        self.report_path = os.path.join(run_dir, "report.json")
        self.loop_log = os.path.join(run_dir, "loop.out")
        self.pidfile = os.path.join(run_dir, "loop.pid")
        self._proc: subprocess.Popen | None = None  # set when this observer started it

    # --- start / stop the creating loop (control, not creation) ---
    def start(self, *, tag: str, dataset: str, max_iters: int) -> int:
        os.makedirs(self.run_dir, exist_ok=True)
        if self.is_running():
            print(f"a loop is already running for {self.run_dir} (pid {self._pid()})")
            return 1
        argv = [
            sys.executable,
            "-m",
            "torchtitan_autoresearch.run",
            "--tag",
            tag,
            "--dataset",
            dataset,
            "--max-iters",
            str(max_iters),
            "--run-dir",
            self.run_dir,
        ]
        with open(self.loop_log, "w") as f:
            p = subprocess.Popen(
                argv,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=os.getcwd(),
                # Pin PYTHONUNBUFFERED so the loop's progress lines flush immediately
                # to loop.out -- it's a file, not a TTY, so a shell that doesn't
                # already export this would block-buffer stdout and the observer's
                # watch/console would only see progress in delayed chunks.
                env={
                    **os.environ,
                    "PYTHONPATH": os.getcwd(),
                    "AR_RUN_FROM_OBSERVER": "1",
                    "PYTHONUNBUFFERED": "1",
                },
            )
        self._proc = p
        with open(self.pidfile, "w") as f:
            f.write(str(p.pid))
        print(f"started autoresearch loop (pid {p.pid}); run dir {self.run_dir}")
        return p.pid

    def _pid(self) -> int | None:
        if not os.path.exists(self.pidfile):
            return None
        try:
            return int(open(self.pidfile).read().strip())
        except ValueError:
            return None

    def is_running(self) -> bool:
        # If we started it, poll() reaps the child and reports liveness reliably.
        if self._proc is not None:
            return self._proc.poll() is None
        pid = self._pid()
        if pid is None:
            return False
        try:  # a finished-but-unreaped child is a zombie -> treat as not running
            with open(f"/proc/{pid}/stat") as f:
                state = f.read().rsplit(")", 1)[-1].split()[0]
            return state != "Z"
        except (OSError, IndexError):
            return False

    def stop(self, grace: float = 8.0) -> None:
        """Tear down the loop and its GPU children (SIGTERM, then SIGKILL)."""
        pid = self._pid()
        if not (pid and self.is_running()):
            return
        try:
            pgid = os.getpgid(pid)
            os.killpg(pgid, signal.SIGTERM)
            deadline = time.time() + grace
            while time.time() < deadline and self.is_running():
                time.sleep(0.3)
            if self.is_running():
                os.killpg(pgid, signal.SIGKILL)
            print(f"[observer] stopped experiment loop (pgid {pgid})")
        except OSError:
            pass

    # --- read-only state ---
    def _state(self) -> dict:
        if not os.path.exists(self.statefile):
            return {}
        with open(self.statefile) as f:
            return json.load(f)

    def _report(self) -> dict:
        if not os.path.exists(self.report_path):
            return {}
        with open(self.report_path) as f:
            return json.load(f)

    def status(self) -> str:
        rows = self.ledger.read()
        st = self._state()
        rep = self._report()
        by_status: dict[str, int] = {}
        for r in rows:
            by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        champ_tps = st.get("champion_tps") or []
        champ = f"{int(champ_tps[-1])} tps" if champ_tps else "none yet"
        running = "running" if self.is_running() else "stopped/idle"
        lines = [
            f"experiment @ {self.run_dir}  [{running}]",
            f"  champion: {champ}  (commit {st.get('champion_commit', '-')})",
            f"  golden:   loss_band={st.get('loss_band')} grad_band={st.get('grad_band')} (faithfulness)",
            f"  candidates: {len(rows)}  "
            + " ".join(f"{k}={v}" for k, v in sorted(by_status.items())),
        ]
        if st.get("family_deferred"):
            lines.append(f"  deferred families: {st['family_deferred']}")
        for r in rows[-3:]:
            lines.append(
                f"    {r['status']:8s} {r['verdict']:8s} {r['tps_mean']:>7s} tps  "
                f"verify={r['verify']:9s} {r['label']}"
            )
        if rep.get("plan"):
            lines.append(f"  agent plan: {rep['plan']}")
        return "\n".join(lines)

    def answer(self, q: str) -> str:
        ql = q.lower()
        rows = self.ledger.read()
        st = self._state()
        if any(
            w in ql
            for w in (
                "status",
                "current",
                "how is",
                "how's",
                "going",
                "overall",
                "summary",
            )
        ):
            return self.status()
        if any(w in ql for w in ("best", "champion", "fastest")):
            tps = st.get("champion_tps") or []
            return (
                (
                    f"Champion is {int(tps[-1])} tps (commit {st.get('champion_commit')}); "
                    f"kept because it is faster than the prior best and faithful to the golden "
                    f"(within loss_band {st.get('loss_band')}, grad_band {st.get('grad_band')})."
                )
                if tps
                else "No champion yet."
            )
        if any(w in ql for w in ("how many", "count", "progress")):
            return f"{len(rows)} candidates run so far; loop is {'running' if self.is_running() else 'not running'}."
        if "reject" in ql or "discard" in ql or "why" in ql:
            bad = [
                r for r in rows if r["status"] in ("discard", "crash", "oom", "invalid")
            ]
            return (
                (
                    "Rejected:\n"
                    + "\n".join(
                        f"  {r['label']}: {r['status']}/{r['verdict']} (crash={r['crash_class']})"
                        for r in bad[-5:]
                    )
                )
                if bad
                else "Nothing rejected yet."
            )
        if "defer" in ql or "stuck" in ql:
            return f"Deferred families: {st.get('family_deferred') or 'none'}."
        if "quality" in ql or "eval" in ql:
            kept = [r for r in rows if r["status"] == "keep"]
            return (
                (
                    "Quality (kept):\n"
                    + "\n".join(
                        f"  {r['label']}: verify={r['verify']} margin={r['quality_margin']}"
                        for r in kept
                    )
                )
                if kept
                else "none kept yet"
            )
        if "recent" in ql or "last" in ql or "latest" in ql:
            return (
                "\n".join(
                    f"{r['label']}: {r['status']} ({r['tps_mean']} tps)"
                    for r in rows[-5:]
                )
                or "nothing yet"
            )
        # Unrecognized phrasing: this is a keyword router, not an LLM, so fall back
        # to the live status (grounded) rather than a help string.
        return (
            self.status()
            + "\n(note: free-form Q&A needs an LLM backend; I matched on keywords. "
            "Try: best, progress, rejected/why, deferred, quality, recent.)"
        )

    # --- full free-form Q&A via an LLM (falls back to keyword answer offline) ---
    def context(self) -> str:
        """A grounded, read-only snapshot the LLM answers from (never invents)."""
        parts = [self.status()]
        rows = self.ledger.read()
        if rows:
            parts.append("\nFull ledger:")
            for r in rows:
                parts.append(
                    f"  {r['label']}: status={r['status']} verdict={r['verdict']} "
                    f"tps={r['tps_mean']} cv={r['tps_cv']} verify={r['verify']} "
                    f"quality_margin={r['quality_margin']} crash={r['crash_class']} "
                    f"rationale={r.get('rationale', '')}"
                )
        rep = self._report()
        if rep:
            parts.append("\nAgent report (its own learnings): " + json.dumps(rep))
        return "\n".join(parts)

    def ask_llm(self, question: str, timeout: float = 180) -> str:
        """Answer any free-form question, grounded in the run state, via `claude -p`.

        Uses the local Claude Code CLI (the AI gateway / existing auth, no API key).
        Falls back to the offline keyword answer when the CLI isn't available.
        """
        if not shutil.which("claude"):
            return self.answer(question)
        prompt = (
            "You are a read-only observer of a TorchTitan autoresearch experiment. "
            "Answer the user's question concisely and ONLY from the state below; if "
            "the state does not contain the answer, say so plainly.\n\n"
            f"STATE:\n{self.context()}\n\nQUESTION: {question}"
        )
        try:
            out = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return out.stdout.strip() or self.answer(question)
        except (subprocess.TimeoutExpired, OSError):
            return self.answer(question)

    def console(self, interval: float = 8.0) -> None:
        """Interactive follow-along: broadcast updates in the background while you
        type questions at a prompt. 'status' for a snapshot, 'quit' to end (quit
        stops the experiment, since this observer owns it)."""
        import threading

        stop_flag = threading.Event()
        lock = threading.Lock()

        def broadcaster() -> None:
            seen, champ, offset, done = -1, None, 0, False
            while not stop_flag.is_set():
                chunk = ""
                if os.path.exists(self.loop_log):
                    with open(self.loop_log) as f:
                        f.seek(offset)
                        chunk = f.read()
                        offset = f.tell()
                rows = self.ledger.read()
                cur = (self._state().get("champion_tps") or [None])[-1]
                changed = len(rows) != seen or cur != champ
                running = self.is_running()
                if chunk.strip() or changed or (not running and rows and not done):
                    with lock:
                        if chunk.strip():
                            print("\n" + chunk.strip())
                        if changed:
                            print("\n" + self.status())
                        if not running and rows and not done:
                            print(
                                "\n[observer] the loop has finished. Ask questions, or 'quit'."
                            )
                            done = True
                        print("> ", end="", flush=True)
                    seen, champ = len(rows), cur
                stop_flag.wait(interval)

        t = threading.Thread(target=broadcaster, daemon=True)
        t.start()
        llm = "(LLM-backed)" if shutil.which("claude") else "(offline keyword mode)"
        print(
            f"Interactive observer {llm}. Ask anything, 'status', or 'quit' "
            "('quit'/Ctrl-C stops the experiment)."
        )
        print("> ", end="", flush=True)
        try:
            for line in sys.stdin:
                q = line.strip()
                if q.lower() in ("quit", "exit", "q", "stop"):
                    break
                if q.lower() == "status":
                    resp = self.status()
                elif q:
                    with lock:
                        print("[thinking...]", flush=True)
                    resp = self.ask_llm(q)  # slow LLM call OUTSIDE the lock
                else:
                    resp = None
                with lock:
                    if resp is not None:
                        print(resp)
                    print("> ", end="", flush=True)
        except KeyboardInterrupt:
            pass
        stop_flag.set()

    def watch(self, interval: float = 10.0) -> None:
        """Follow along: stream the loop's stdout and broadcast status on change."""
        seen_rows, last_champ, offset = -1, None, 0
        while True:
            if os.path.exists(self.loop_log):  # stream new creating-loop output
                with open(self.loop_log) as f:
                    f.seek(offset)
                    new = f.read()
                    offset = f.tell()
                if new.strip():
                    print(new, end="", flush=True)
            rows = self.ledger.read()
            champ = (self._state().get("champion_tps") or [None])[-1]
            if len(rows) != seen_rows or champ != last_champ:
                print("\n" + self.status() + "\n", flush=True)
                seen_rows, last_champ = len(rows), champ
            if not self.is_running() and rows:  # loop finished
                print("\n[observer] loop has finished.", flush=True)
                return
            time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="torchtitan_autoresearch.observe",
        description="Start and follow autoresearch (the only entry point).",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("start", "watch", "status", "ask", "stop"):
        p = sub.add_parser(name)
        p.add_argument("--tag", default=None)
        p.add_argument("--run-dir", default=None)
        if name == "start":
            p.add_argument(
                "--dataset",
                default="c4",
                help="dataset for training and its held-out eval split "
                "(c4 real/streamed; c4_test local/offline)",
            )
            p.add_argument("--max-iters", type=int, default=8)
        if name == "ask":
            p.add_argument("question")
    args = ap.parse_args(argv)
    if args.cmd == "start" and not args.tag:
        import datetime

        args.tag = "qwen3-" + datetime.datetime.now().strftime("%m%d-%H%M%S")
        print(f"[observer] no --tag given; using auto tag: {args.tag}")
    obs = Observer(_run_dir(args.tag, args.run_dir))

    if args.cmd == "start":
        # The start observer OWNS the experiment: when it exits for any reason
        # (Ctrl-C, normal end, SIGTERM/SIGHUP, terminal close), the loop and its
        # GPU children are torn down. There is no detached mode.
        obs.start(tag=args.tag, dataset=args.dataset, max_iters=args.max_iters)
        import atexit

        atexit.register(obs.stop)
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, lambda *_: sys.exit(0))  # -> atexit -> obs.stop()
        try:
            obs.console()  # interactive: broadcasts updates AND answers questions
        finally:
            print("\n[observer] exiting -> stopping the experiment...")
            obs.stop()
    elif args.cmd == "watch":
        # A passive re-attach viewer; it does NOT own the experiment, so exiting
        # it leaves the experiment running (use `stop` or the owning observer).
        try:
            obs.watch()
        except KeyboardInterrupt:
            print(
                "\n[observer] detached (experiment still running; use `stop` to end it)"
            )
    elif args.cmd == "status":
        print(obs.status())
    elif args.cmd == "ask":
        print(obs.ask_llm(args.question))
    elif args.cmd == "stop":
        obs.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
