# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The unattended hill-climbing loop (single box, serial).

Drives any Agent against the Harness: observe -> propose -> submit, pulling the
agent's report() snapshot periodically so the human can inspect learnings without
touching agent-private memory. This is the harness-side orchestration; the agent
behind it is fully pluggable.
"""

from __future__ import annotations

from torchtitan_autoresearch.agent import Agent
from torchtitan_autoresearch.api import Harness


def run_loop(
    harness: Harness, agent: Agent, *, max_iters: int = 1000, report_every: int = 10
) -> dict:
    """Run until the agent stops proposing or max_iters is hit. Returns a summary.

    Narrates each iteration to stdout (-> loop.out) so the agent's activity is
    visible live: when it is proposing, what it proposed (label / command /
    file-edits / rationale), and the gate's verdict.
    """
    # Lifecycle: let the agent set up persistent resources (e.g. an LLM session
    # that outlives turns) before the loop and tear them down after. Whether an
    # agent is created per turn or persists is the agent's own concern.
    start = getattr(agent, "start", None)
    if callable(start):
        start()
    try:
        return _drive(harness, agent, max_iters, report_every)
    finally:
        stop = getattr(agent, "stop", None)
        if callable(stop):
            stop()


def _drive(harness: Harness, agent: Agent, max_iters: int, report_every: int) -> dict:
    submitted = 0
    promotions = 0
    for n in range(1, max_iters + 1):
        obs = harness.observe()
        print(f"[iter {n}] {type(agent).__name__} proposing...", flush=True)
        cand = agent.propose(obs)
        if cand is None:
            print(f"[iter {n}] agent has no further candidate; stopping.", flush=True)
            break
        submitted += 1
        edits = list(cand.file_edits) if cand.file_edits else []
        print(
            f"[iter {n}] proposed: {cand.label}  ({cand.family})\n"
            f"          command:    {' '.join(cand.command) or '(none)'}\n"
            f"          file_edits: {edits or '(none)'}\n"
            f"          rationale:  {cand.rationale[:400]}",
            flush=True,
        )
        v = harness.submit(cand)
        print(
            f"[iter {n}] verdict: {v.verdict}/{v.status}  verify={v.verify}  "
            f"tps={v.throughput_mean:.0f}  :: {v.detail}\n"
            f"          profile: {v.profile_trace or '(none)'}",
            flush=True,
        )
        if v.status == "keep":
            promotions += 1
            print(
                f"[iter {n}] >>> PROMOTED new champion @ {v.throughput_mean:.0f} tps",
                flush=True,
            )
        if report_every and n % report_every == 0:
            harness.pull_report(agent)
    harness.pull_report(agent)  # final snapshot
    return {
        "iterations": submitted,
        "promotions": promotions,
        "champion": harness.state.champion_commit,
        "champion_tps": harness.state.champion_tps,
        "deferred_families": list(harness.state.family_deferred),
    }
