"""The Agent contract and reference implementations.

The agent is a pluggable search policy (ARCHITECTURE.md section 6). The contract
is exactly two methods: `propose` (observation -> candidate) and `report`
(project grounded learnings). Everything else is private. Two reference agents
are provided: a `ScriptedAgent` (for tests / replay) and a `PlaybookAgent` (a
trivial non-LLM policy that turns the human's front-loaded ideas into candidates).
A real LLM agent, an ensemble, or a classical optimizer plug in the same way.
"""

from __future__ import annotations

from typing import Protocol

from torchtitan_autoresearch.types import Candidate, Observation, Report


class Agent(Protocol):
    def propose(self, obs: Observation) -> Candidate | None: ...
    def report(self) -> Report: ...


class ScriptedAgent:
    """Replays a fixed list of candidates; trivial report. For tests and replay."""

    def __init__(self, candidates: list[Candidate]):
        self._queue = list(candidates)
        self._submitted: list[str] = []

    def propose(self, obs: Observation) -> Candidate | None:
        # Skip candidates whose family the harness has deferred.
        while self._queue:
            c = self._queue.pop(0)
            fam = c.family or (c.addresses[0] if c.addresses else "")
            if fam and fam in obs.deferred_families:
                continue
            self._submitted.append(c.commit)
            return c
        return None

    def report(self) -> Report:
        return Report(
            beliefs=[],
            conclusions=[],
            plan="scripted replay; see ledger",
            ideas_usage={},
        )


class KnobAgent:
    """Built-in, non-LLM proposing agent over the config/command search space.

    Emits a fixed exploration plan of command-only candidates (no code edits), so
    the packaged single-command entrypoint runs genuine autoresearch end to end
    without an LLM. Each candidate is a real recipe knob the baseline respects
    (batch size, activation-checkpoint mode, FSDP reshard policy). The LLM
    code-writing agent is a separate, future plug behind the same contract.

    Families let the harness time-box knobs that keep crashing; quality-affecting
    knobs (batch) get the eval, quality-neutral ones (AC/reshard) route faithful.
    """

    DEFAULT_KNOBS = [
        ("local_batch_size=8", "batch", ["--training.local_batch_size=8"]),
        ("selective_activation_checkpoint", "ac", ["--activation_checkpoint.mode=selective"]),
        ("local_batch_size=12", "batch", ["--training.local_batch_size=12"]),
        ("local_batch_size=16", "batch", ["--training.local_batch_size=16"]),
    ]

    def __init__(self, knobs: list[tuple[str, str, list[str]]] | None = None):
        self._knobs = list(knobs or self.DEFAULT_KNOBS)
        self._i = 0
        self._notes: dict[str, str] = {}

    def propose(self, obs: Observation) -> Candidate | None:
        while self._i < len(self._knobs):
            label, family, command = self._knobs[self._i]
            self._i += 1
            if family in obs.deferred_families:
                self._notes[label] = "skipped: family deferred"
                continue
            self._notes[label] = "tried"
            return Candidate(label=label, family=family, command=command,
                             rationale=f"config-space knob ({family})")
        return None

    def report(self) -> Report:
        return Report(
            beliefs=["exploring config-space throughput knobs (batch, AC, reshard)"],
            conclusions=[],
            plan="exhaust the knob plan, skipping deferred families",
            ideas_usage=dict(self._notes),
        )


class PlaybookAgent:
    """Non-LLM policy: emit one candidate per advisory idea, highest weight first.

    A deliberately simple baseline so the loop is exercisable end to end. It
    grounds its report in the ledger (which ideas led to which verdicts) and
    skips ideas the harness has deferred or that the human marked do-not-pursue.
    """

    def __init__(self, candidate_for):
        # candidate_for(idea_dict, index) -> Candidate, supplied by the caller so
        # the actual recipe edit / commit is produced outside this policy.
        self._candidate_for = candidate_for
        self._i = 0
        self._order: list[dict] | None = None
        self._used: dict[str, str] = {}

    def propose(self, obs: Observation) -> Candidate | None:
        if self._order is None:
            actionable = [i for i in obs.ideas if i["kind"] in ("hint", "prior")]
            self._order = sorted(actionable, key=lambda i: -float(i.get("weight", 1.0)))
        while self._i < len(self._order):
            idea = self._order[self._i]
            self._i += 1
            if idea["id"] in obs.deferred_families:
                self._used[idea["id"]] = "skipped: family deferred"
                continue
            cand = self._candidate_for(idea, self._i)
            if cand is None:
                self._used[idea["id"]] = "skipped: no concrete candidate"
                continue
            self._used[idea["id"]] = "tried"
            return cand
        return None

    def report(self) -> Report:
        return Report(
            beliefs=["front-loading high-weight ideas from ideas.md"],
            conclusions=[],
            plan="exhaust prior/hint ideas by descending weight, then stop",
            ideas_usage=dict(self._used),
        )
