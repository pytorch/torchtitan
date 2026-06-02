# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Agent contract and reference implementations.

The agent is a pluggable search policy (ARCHITECTURE.md section 6). The contract
is exactly two methods: `propose` (observation -> candidate) and `report`
(project grounded learnings). Everything else is private. Implementations:
`LLMAgent` is the real autoresearcher (an LLM via `claude -p`, proposing knobs
and file edits grounded in the ledger); `KnobAgent` is a deterministic knob-sweep
fallback for offline/CI; `ScriptedAgent` replays a fixed list (tests); and
`PlaybookAgent` turns the human's front-loaded ideas into candidates. An ensemble
or a classical optimizer plug in the same way.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from typing import Protocol

from torchtitan_autoresearch.types import Candidate, Observation, Report


class Agent(Protocol):
    def start(self) -> None:  # lifecycle: create persistent resources (optional)
        ...

    def propose(self, obs: Observation) -> Candidate | None:
        ...

    def report(self) -> Report:
        ...

    def stop(self) -> None:  # lifecycle: destroy them (optional)
        ...


def _claude_ask(prompt: str, timeout: float = 420) -> str:
    """Query the local Claude Code CLI (keyless AI gateway), same as the observer."""
    out = subprocess.run(
        ["claude", "-p", prompt], capture_output=True, text=True, timeout=timeout
    )
    return out.stdout.strip()


def summarize_trace(path: str, ask=None, timeout: float = 600) -> str:
    """One-off LLM aggregation of a chrome trace into a tiny text summary.

    No programmatic parsing in the harness: a ``claude -p`` agent reads/aggregates
    the large trace with its OWN tools and returns a short comm-vs-compute +
    top-kernels table. This isolates the slow 30MB read into a single dedicated
    call (its own timeout) so the proposing agent only ever sees the small summary
    and never blows its reasoning budget reading the raw trace.
    """
    if not path or not os.path.isfile(path):
        return ""
    prompt = (
        f"Read the PyTorch profiler chrome trace at {path} (a large JSON; use your "
        "tools, e.g. python, to aggregate it -- do not try to read it whole). Output "
        "a SHORT summary ONLY: total GPU-kernel time; the percent of GPU-kernel time "
        "in communication (NCCL all-gather / reduce-scatter) vs compute (GEMM / "
        "matmul / attention) vs other; and the top ~8 kernels by total time (name, "
        "%, call count). A small table, no prose, no recommendations."
    )
    fn = ask or (lambda p: _claude_ask(p, timeout=timeout))
    try:
        return fn(prompt)
    except (subprocess.TimeoutExpired, OSError):
        return ""


def _extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply (fenced or bare)."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:  # fall back to the first balanced {...}
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    blob = text[start : i + 1]
                    break
    if not blob:
        return None
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return None


class _PerTurnSession:
    """A fresh, stateless ``claude -p`` call each turn -- no memory between turns.
    Continuity comes from re-injecting the ledger/state into every prompt."""

    def __init__(self, ask=None, timeout: float = 420):
        self._ask = ask or (lambda p: _claude_ask(p, timeout))

    def ask(self, prompt: str) -> str:
        return self._ask(prompt)

    def close(self) -> None:
        pass


class _PersistentSession:
    """ONE Claude conversation that OUTLIVES turns: created with an explicit
    ``--session-id`` and continued with ``--resume``, so the agent keeps native
    memory across propose() calls. The explicit id isolates it from the
    summarizer's separate one-off ``claude`` calls. An injected ``ask`` is used
    instead when provided (tests)."""

    def __init__(self, ask=None, timeout: float = 420):
        import uuid

        self._sid = str(uuid.uuid4())
        self._ask = ask
        self._timeout = timeout
        self._resuming = False

    def ask(self, prompt: str) -> str:
        resuming = self._resuming
        self._resuming = True
        if self._ask is not None:
            return self._ask(prompt)
        flag = ["--resume", self._sid] if resuming else ["--session-id", self._sid]
        out = subprocess.run(
            ["claude", "-p", *flag, prompt],
            capture_output=True,
            text=True,
            timeout=self._timeout,
        )
        return out.stdout.strip()

    def close(self) -> None:
        self._resuming = False


class ScriptedAgent:
    """Replays a fixed list of candidates; trivial report. For tests and replay."""

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

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
    without an LLM. The LLM code-writing agent is a separate, future plug behind
    the same contract.

    v1 is faithfulness-only (affecting changes are rejected), so the default plan
    targets the math-preserving throughput knobs the verify path can promote for
    free: activation-checkpoint mode, torch.compile, FSDP reshard policy, and TP
    degree (a boundary case the faithfulness check adjudicates). Families let the
    harness time-box knobs that keep crashing.
    """

    DEFAULT_KNOBS = [
        ("activation_checkpoint=none", "ac", ["--activation_checkpoint.mode=none"]),
        ("activation_checkpoint=full", "ac", ["--activation_checkpoint.mode=full"]),
        ("compile", "compile", ["--compile.enable"]),
        (
            "fsdp_reshard=never",
            "fsdp",
            ["--parallelism.fsdp_reshard_after_forward=never"],
        ),
        ("tensor_parallel=2", "tp", ["--parallelism.tensor_parallel_degree=2"]),
    ]

    def __init__(self, knobs: list[tuple[str, str, list[str]]] | None = None):
        self._knobs = list(knobs or self.DEFAULT_KNOBS)
        self._i = 0
        self._notes: dict[str, str] = {}

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def propose(self, obs: Observation) -> Candidate | None:
        while self._i < len(self._knobs):
            label, family, command = self._knobs[self._i]
            self._i += 1
            if family in obs.deferred_families:
                self._notes[label] = "skipped: family deferred"
                continue
            self._notes[label] = "tried"
            return Candidate(
                label=label,
                family=family,
                command=command,
                rationale=f"config-space knob ({family})",
            )
        return None

    def report(self) -> Report:
        return Report(
            beliefs=["exploring faithful throughput knobs (AC, compile, reshard, TP)"],
            conclusions=[],
            plan="exhaust the knob plan, skipping deferred families",
            ideas_usage=dict(self._notes),
        )


class LLMAgent:
    """LLM-backed proposing agent -- the real autoresearcher.

    Implements the Agent contract by asking an LLM (the keyless ``claude -p``
    gateway, same as the observer) for the NEXT candidate, grounded in the live
    Observation: the constitution rules, the ledger of every past candidate +
    verdict, the current champion, advisory ideas, and deferred families. It
    proposes CLI knobs and/or edits to the editable model files, learning from
    what was faithful / affecting / faster / crashed instead of replaying a fixed
    list. The harness gate still bounds it -- it can only propose, never promote,
    and can never edit outside the constitution's editable scope.

    ``ask`` is injectable (prompt -> reply text) so the agent is testable on CPU
    without the real CLI; by default it shells out to ``claude -p``.
    """

    def __init__(
        self,
        *,
        repo_root: str = ".",
        ask=None,
        max_ledger: int = 30,
        log_path=None,
        profiler=None,
        logs_dir: str = "",
        persistent: bool = False,
        timeout: float = 420,
    ):
        self.repo_root = repo_root
        # ``ask`` (prompt -> reply) is an injectable transport for CPU tests; the
        # real transport is a session created in start(). ``persistent`` decides
        # whether that session is one Claude conversation that OUTLIVES turns
        # (native memory via --resume) or a fresh stateless call per turn. Either
        # way, whether an "agent" is created per turn is an LLMAgent detail, not
        # something the loop hardcodes -- the loop only calls start()/stop().
        self._ask_override = ask
        self._persistent = persistent
        self._timeout = timeout
        self._session = None  # set in start(); the LLM transport
        self.max_ledger = max_ledger
        self.log_path = log_path  # full prompt+reply transcript, for deep inspection
        self._profiler = profiler  # callable(command=None) -> baseline profile summary
        self.logs_dir = logs_dir  # dir of raw per-run .log files the agent may read
        self._baseline_profile = None  # cached baseline profile summary (fetched once)
        self._proposed: set[tuple] = set()  # de-dupe (command, edited-paths)
        self._notes: list[str] = []
        self._plan = "LLM-driven search over faithful throughput knobs"
        self._n = 0

    def start(self) -> None:
        """Create the LLM session (persistent or per-turn). Idempotent."""
        if self._session is not None:
            return
        if self._persistent:
            self._session = _PersistentSession(
                ask=self._ask_override, timeout=self._timeout
            )
            print("[llm] started PERSISTENT session (outlives turns)", flush=True)
        else:
            self._session = _PerTurnSession(
                ask=self._ask_override, timeout=self._timeout
            )

    def stop(self) -> None:
        """Tear down the LLM session."""
        if self._session is not None:
            self._session.close()
            self._session = None

    def _ensure_profile(self) -> None:
        """Ask the harness for the baseline profile summary once (the agent's call)."""
        if self._profiler is None or self._baseline_profile is not None:
            return
        print(
            "[llm] requesting a profile of the baseline (one-off summary)...",
            flush=True,
        )
        try:
            self._baseline_profile = self._profiler() or ""
        except Exception as e:  # profiling is best-effort; proceed without it
            print(f"[llm] profile request failed ({e})", flush=True)
            self._baseline_profile = ""
        print(
            f"[llm] baseline profile: {self._baseline_profile[:200] or '(unavailable)'}",
            flush=True,
        )

    def _transcript(self, prompt: str, reply: str) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path, "a") as f:
                f.write(
                    f"\n{'=' * 70}\n===== proposal #{self._n} =====\n"
                    f"--- PROMPT ---\n{prompt}\n--- REPLY ---\n{reply}\n"
                )
        except OSError:
            pass

    def _editable_sources(self, obs: Observation) -> dict[str, str]:
        out: dict[str, str] = {}
        for path in obs.rules.get("editable", {}).get("files", []):
            try:
                with open(os.path.join(self.repo_root, path)) as f:
                    out[path] = f.read()
            except OSError:
                pass
        return out

    def _prompt(self, obs: Observation) -> str:
        rules = obs.rules

        def _row(r: dict) -> str:
            head = (
                f"- {r.get('label', '?')} | status={r.get('status')} "
                f"verdict={r.get('verdict')} verify={r.get('verify')} "
                f"tps={r.get('tps_mean')} :: {r.get('rationale', '')}"
            )
            summ = (r.get("profile_summary") or "").strip()
            if summ:
                head += "\n    profile-summary: " + summ.replace("\n", "\n    ")
            elif r.get("profile_trace"):
                head += (
                    f"\n    profile-trace (raw, read if needed): {r['profile_trace']}"
                )
            return head

        ledger = (
            "\n".join(_row(r) for r in obs.ledger[-self.max_ledger :])
            or "(no candidates run yet)"
        )
        sources = "\n\n".join(
            f"### {p}\n```python\n{c}\n```"
            for p, c in self._editable_sources(obs).items()
        )
        # Full advisory ideas (human channel; prior-experiment learnings, weighted).
        # 'soft_constraint' / 'na-*' items are warnings or do-NOT-do bans -- honor them.
        ideas = (
            "\n".join(
                f"- [{i.get('kind')}] {i.get('id')} (w={i.get('weight')}): "
                f"{i.get('text', '')}"
                for i in obs.ideas
            )
            or "(none)"
        )
        return (
            "You are the proposing agent in an autonomous TorchTitan autoresearch "
            "loop. Propose the SINGLE next experiment candidate that maximizes "
            "training throughput (tokens/sec) for the locked workload WITHOUT "
            "changing the math.\n\n"
            "POLICY (v1 = faithfulness-only): a candidate is KEPT only if it is "
            "faster than the champion AND numerically FAITHFUL to the golden (its "
            "short deterministic loss+grad_norm trajectory stays within the "
            "golden's own rounding noise). Changes that move the math (fp8/"
            "precision, batch size, LR, optimizer) are REJECTED. So propose "
            "math-preserving throughput knobs: activation checkpointing, "
            "torch.compile, FSDP reshard policy, tensor-parallel degree, attention "
            "backend, comm/memory layout.\n\n"
            "BE PROFILER-GUIDED: optimize the MEASURED bottleneck, not a guess. A "
            "profile summary of the baseline is below (comm vs compute vs other). "
            "Decide what the run is bound by and target THAT; do not blind-sweep "
            "knobs that are off the critical path. Each ledger candidate also has a "
            "profile-summary. For a deeper look you MAY read a raw trace path, but "
            "you do not need to -- the summaries should suffice.\n"
            f"BASELINE PROFILE (LLM-summarized):\n{self._baseline_profile or '(none)'}\n"
            f"RAW RUN LOGS dir (one .log per past run; read to diagnose crashes or "
            f"inspect step times if needed): {self.logs_dir or '(none)'}\n\n"
            f"OBJECTIVE: {json.dumps(rules.get('objective', {}))}\n"
            f"QUALITY POLICY: {json.dumps(rules.get('quality', {}))}\n"
            f"WORKLOAD (locked, do NOT change): {json.dumps(rules.get('workload', {}))}\n"
            f"BANNED fields: {rules.get('banned_workload_fields')}; "
            f"FIXED fields: {rules.get('fixed_fields')}\n"
            f"EDITABLE files (you MAY rewrite these): "
            f"{rules.get('editable', {}).get('files')}\n\n"
            f"CURRENT CHAMPION: {json.dumps(obs.champion)}\n"
            f"DEFERRED FAMILIES (skip): {obs.deferred_families}\n"
            f"ADVISORY IDEAS (human; weighted prior-experiment learnings; "
            f"soft_constraint/na-* are warnings or bans -- honor them):\n{ideas}\n\n"
            f"LEDGER (past candidates + verdicts; each has a single-step profile "
            f"trace you can READ to see why it was faster/slower/affecting -- learn "
            f"from these, do NOT repeat):\n{ledger}\n\n"
            f"EDITABLE FILE CONTENTS (for file_edits):\n{sources}\n\n"
            "Reply with ONE JSON object and nothing else:\n"
            '{"label": "<short unique label>", '
            '"family": "<ac|compile|fsdp|tp|attn|...>", '
            '"command": ["--flag=value", ...], '
            '"file_edits": {"path": "<full new file content>"}, '
            '"rationale": "<why it is faster AND faithful>", '
            '"done": false}\n'
            "Use command for CLI/config knobs and/or file_edits to rewrite an "
            "editable file (full content; editable files only). Propose something "
            "NOT already in the ledger. Set done=true only if no promising "
            "candidate remains."
        )

    def propose(self, obs: Observation) -> Candidate | None:
        if self._session is None:
            self.start()  # lazy-start so direct propose() calls still work (tests)
        self._n += 1
        self._ensure_profile()  # the agent asks the harness for a baseline trace
        prompt = self._prompt(obs)
        for _ in range(2):  # one retry on a parse failure or a repeat
            print("[llm] querying claude for the next candidate...", flush=True)
            try:
                reply = self._session.ask(prompt)
            except subprocess.TimeoutExpired:
                print("[llm] query timed out; retrying once", flush=True)
                continue
            except OSError as e:
                print(f"[llm] query failed ({e}); no candidate", flush=True)
                return None
            self._transcript(prompt, reply)
            data = _extract_json(reply or "")
            if data is None:
                print("[llm] reply was not parseable JSON; retrying", flush=True)
                continue
            if data.get("done"):
                self._plan = str(data.get("rationale", self._plan))
                print(f"[llm] signaled done: {self._plan[:200]}", flush=True)
                return None
            command = [str(x) for x in (data.get("command") or [])]
            file_edits = {
                str(k): str(v) for k, v in (data.get("file_edits") or {}).items()
            }
            if not command and not file_edits:
                print("[llm] proposal had no command/file_edits; retrying", flush=True)
                continue
            # De-dupe on command + file-edit CONTENT (not just paths): two distinct
            # rewrites of the same file (e.g. parallelize.py) are different
            # candidates and must not collide.
            edits_sig = tuple(
                sorted(
                    (p, hashlib.md5(c.encode()).hexdigest())
                    for p, c in file_edits.items()
                )
            )
            key = (tuple(command), edits_sig)
            if key in self._proposed:
                print(
                    f"[llm] '{data.get('label')}' already tried; asking for a different one",
                    flush=True,
                )
                prompt += (
                    "\n\nThat candidate was already tried; propose a DIFFERENT one."
                )
                continue
            self._proposed.add(key)
            family = str(data.get("family") or "misc")
            label = str(data.get("label") or f"llm-{self._n}")
            rationale = str(data.get("rationale", ""))
            self._notes.append(f"{label}: {rationale}")
            return Candidate(
                label=label,
                family=family,
                command=command,
                file_edits=file_edits,
                rationale=rationale,
            )
        print("[llm] no valid candidate after retries; stopping", flush=True)
        return None

    def report(self) -> Report:
        return Report(
            beliefs=["LLM-driven search over faithful throughput knobs (claude -p)"],
            conclusions=[],
            plan=self._plan,
            open_questions=[],
            ideas_usage={f"prop{i + 1}": n for i, n in enumerate(self._notes)},
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

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

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
