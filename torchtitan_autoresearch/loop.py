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
    """Run until the agent stops proposing or max_iters is hit. Returns a summary."""
    n = 0
    promotions = 0
    for n in range(1, max_iters + 1):
        obs = harness.observe()
        cand = agent.propose(obs)
        if cand is None:
            break
        v = harness.submit(cand)
        if v.status == "keep":
            promotions += 1
        if report_every and n % report_every == 0:
            harness.pull_report(agent)
    harness.pull_report(agent)  # final snapshot
    return {
        "iterations": n,
        "promotions": promotions,
        "champion": harness.state.champion_commit,
        "champion_tps": harness.state.champion_tps,
        "deferred_families": list(harness.state.family_deferred),
    }
