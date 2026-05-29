"""Observer: a read-only follow-along agent for a running autoresearch.

This is a *different* actor from the creating agent (ARCHITECTURE.md section 6):
it never proposes candidates and cannot touch the gate, the ledger, or the
branch. It only reads the artifacts the harness writes -- the ledger, the state,
and the agent's `report()` snapshot -- to broadcast status and answer questions.
Because it is purely read-only over files, it can run in a separate shell (or be
driven by a conversational/broadcast agent) while the experiment runs.

    python -m torchtitan_autoresearch.observe --run-dir /tmp/ar_<tag>            # status
    python -m torchtitan_autoresearch.observe --run-dir /tmp/ar_<tag> --watch    # follow along
    python -m torchtitan_autoresearch.observe --run-dir /tmp/ar_<tag> --ask "best so far?"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from torchtitan_autoresearch.ledger import Ledger


class Observer:
    """Read-only view over a run directory's ledger / state / report."""

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        self.ledger = Ledger(os.path.join(run_dir, "results.tsv"))
        self.statefile = os.path.join(run_dir, "state.json")
        self.report_path = os.path.join(run_dir, "report.json")

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
        lines = [
            f"experiment @ {self.run_dir}",
            f"  champion: {champ}  (commit {st.get('champion_commit','-')})",
            f"  golden:   eval_loss={st.get('golden_eval_loss')}",
            f"  candidates: {len(rows)}  " + " ".join(f"{k}={v}" for k, v in sorted(by_status.items())),
        ]
        if st.get("family_deferred"):
            lines.append(f"  deferred families: {st['family_deferred']}")
        if rows:
            lines.append("  recent:")
            for r in rows[-3:]:
                lines.append(f"    {r['status']:8s} {r['verdict']:8s} {r['tps_mean']:>7s} tps  {r['label']}")
        if rep.get("plan"):
            lines.append(f"  agent plan: {rep['plan']}")
        return "\n".join(lines)

    def answer(self, q: str) -> str:
        """Structured answers to common follow-along questions (keyword-routed).

        A conversational agent can wrap this for free-form Q&A; the data it
        returns is grounded in the ledger/state, never invented.
        """
        ql = q.lower()
        rows = self.ledger.read()
        st = self._state()
        if any(w in ql for w in ("best", "champion", "fastest")):
            tps = st.get("champion_tps") or []
            return (f"Champion is {int(tps[-1])} tps (commit {st.get('champion_commit')}), "
                    f"golden eval_loss {st.get('golden_eval_loss')}.") if tps else "No champion yet."
        if any(w in ql for w in ("how many", "count", "progress")):
            return f"{len(rows)} candidates run so far."
        if "reject" in ql or "discard" in ql or "why" in ql:
            bad = [r for r in rows if r["status"] in ("discard", "crash", "oom", "invalid")]
            if not bad:
                return "Nothing rejected yet."
            return "Rejected:\n" + "\n".join(
                f"  {r['label']}: {r['status']}/{r['verdict']} (crash={r['crash_class']})" for r in bad[-5:])
        if "defer" in ql or "stuck" in ql:
            return f"Deferred families: {st.get('family_deferred') or 'none'}."
        if "quality" in ql or "eval" in ql:
            kept = [r for r in rows if r["status"] == "keep"]
            return "Quality margins (kept):\n" + "\n".join(
                f"  {r['label']}: verify={r['verify']} margin={r['quality_margin']}" for r in kept) or "none"
        if "recent" in ql or "last" in ql or "latest" in ql:
            return "\n".join(f"{r['label']}: {r['status']} ({r['tps_mean']} tps)" for r in rows[-5:]) or "nothing yet"
        return ("I can answer: best/champion, progress/count, rejected/why, deferred, "
                "quality/eval, recent/latest. Ask one of those.")

    def watch(self, interval: float = 15.0) -> None:
        """Broadcast status whenever the ledger grows or the champion changes."""
        seen = -1
        last_champ = None
        while True:
            rows = self.ledger.read()
            champ = (self._state().get("champion_tps") or [None])[-1]
            if len(rows) != seen or champ != last_champ:
                print("\n" + self.status(), flush=True)
                seen, last_champ = len(rows), champ
            time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="torchtitan_autoresearch.observe")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--watch", action="store_true", help="follow along, broadcasting changes")
    ap.add_argument("--ask", default="", help="ask a follow-along question")
    args = ap.parse_args(argv)
    obs = Observer(args.run_dir)
    if args.ask:
        print(obs.answer(args.ask))
    elif args.watch:
        try:
            obs.watch()
        except KeyboardInterrupt:
            pass
    else:
        print(obs.status())
    return 0


if __name__ == "__main__":
    sys.exit(main())
