# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Harness: enforces the constitution and exposes the Agent API.

This is the only surface the agent touches (ARCHITECTURE.md section 5.6). It
serves a read-only `observe()`, judges a `submit(candidate)` through the gate,
and pulls/serves the agent's `report()` projection. The human's two channels
(binding `amend_constitution`, advisory `post_idea`) also live here.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

from torchtitan_autoresearch import gate as gate_mod
from torchtitan_autoresearch.constitution import load_constitution
from torchtitan_autoresearch.executor import Executor
from torchtitan_autoresearch.ideas import load_ideas
from torchtitan_autoresearch.ledger import Ledger
from torchtitan_autoresearch.state import HarnessState
from torchtitan_autoresearch.types import Candidate, Observation, Verdict


class Harness:
    def __init__(
        self,
        *,
        constitution_path: str,
        ideas_path: str,
        ledger_path: str,
        statefile: str,
        executor: Executor,
        session=None,
        report_path: str | None = None,
    ):
        self.constitution_path = constitution_path
        self.ideas_path = ideas_path
        self.executor = executor
        self.session = session
        self.statefile = statefile
        self.report_path = report_path or os.path.join(
            os.path.dirname(statefile) or ".", "report.json"
        )
        self.rules = load_constitution(constitution_path)
        self.ideas = load_ideas(ideas_path)
        self.ledger = Ledger(ledger_path)
        self.state = HarnessState.load(statefile)
        # Roofline/bound summary of the golden, set by the driver after calibration;
        # surfaced to the agent so it optimizes the measured bottleneck, not a guess.
        self.profile_summary = ""

    # --- Harness -> Agent ---
    def observe(self) -> Observation:
        champ = None
        if self.state.champion_commit:
            champ = {
                "commit": self.state.champion_commit,
                "tps_samples": self.state.champion_tps,
                "tps_mean": (
                    sum(self.state.champion_tps) / len(self.state.champion_tps)
                )
                if self.state.champion_tps
                else 0.0,
            }
        golden = None
        if self.state.golden_commit:
            golden = {
                "commit": self.state.golden_commit,
                "loss_band": self.state.loss_band,
                "grad_band": self.state.grad_band,
            }
        return Observation(
            rules=self.rules.raw,
            ledger=self.ledger.read(),
            champion=champ,
            golden=golden,
            deferred_families=list(self.state.family_deferred),
            ideas=[asdict(i) for i in self.ideas],
            profile=self.profile_summary,
        )

    def get_traces(self, commit: str) -> dict:
        # Real executors persist traces per run; the fake has none.
        getter = getattr(self.executor, "get_traces", None)
        return getter(commit) if getter else {}

    # --- Agent -> Harness (the only write) ---
    def submit(self, c: Candidate, mode: str = "screen") -> Verdict:
        return gate_mod.gate(
            c,
            rules=self.rules,
            state=self.state,
            ledger=self.ledger,
            executor=self.executor,
            statefile=self.statefile,
            session=self.session,
            mode=mode,
        )

    # --- report projection (Harness-pulled, persisted, served) ---
    def pull_report(self, agent) -> dict:
        """Pull the agent's report() projection and persist it verbatim."""
        rep = asdict(agent.report())
        with open(self.report_path, "w") as f:
            json.dump(rep, f, indent=2)
        return rep

    def read_report(self) -> dict | None:
        if not os.path.exists(self.report_path):
            return None
        with open(self.report_path) as f:
            return json.load(f)

    # --- Human -> Harness ---
    def amend_constitution(self) -> None:
        """Reload the constitution after the human edits Constitution.md."""
        self.rules = load_constitution(self.constitution_path)

    def post_idea(self, item: dict) -> None:
        """Append one advisory item to ideas.md (human channel)."""
        with open(self.ideas_path, "a") as f:
            f.write(
                f"\n- id: {item['id']}\n  kind: {item.get('kind', 'hint')}\n"
                f"  target: {item.get('target', '')}\n  weight: {item.get('weight', 1.0)}\n"
                f"  text: {item['text']}\n"
            )
        self.ideas = load_ideas(self.ideas_path)
