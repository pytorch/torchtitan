# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared dataclasses for the harness <-> agent contract (ARCHITECTURE.md s. 5.6).

Kept dependency-free so every other module can import these without cycles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    """What the agent produces. The only thing it can write into the system.

    ``command`` is the CLI knob list for the launcher; ``changed_files`` is the
    set of repo files the commit touched (used to enforce editable scope).
    ``addresses`` lists ids from ideas.md the candidate responds to.
    """

    label: str
    command: list[str] = field(default_factory=list)
    file_edits: dict[str, str] = field(default_factory=dict)  # path -> full new content
    commit: str = (
        ""  # harness-assigned (the sha the harness creates); agent never sets it
    )
    rationale: str = ""
    addresses: list[str] = field(default_factory=list)
    family: str = ""  # idea family for substrate time-boxing; derived if empty

    @property
    def changed_files(self) -> list[str]:
        return list(self.file_edits.keys())


@dataclass
class Quality:
    checked: bool  # was the held-out eval run (quality-affecting change)?
    passed: bool  # quality >= golden - epsilon (one-sided)
    margin: float  # relative margin; >= 0 passes (0 if not checked / preserved)


@dataclass
class Verdict:
    admitted: bool
    throughput_mean: float
    throughput_cv: float
    quality: Quality
    verdict: str  # promote | reject | rerun | invalid
    status: str  # keep | discard | crash | oom | invalid
    crash_class: str
    detail: str
    verify: str = "n/a"  # faithful | affecting | fail | n/a


@dataclass
class Observation:
    """The read-only view the harness serves to the agent."""

    rules: dict[str, Any]
    ledger: list[dict]
    champion: dict | None
    golden: dict | None
    deferred_families: list[str]
    ideas: list[dict]


@dataclass
class Report:
    """The agent's public, grounded projection of its private learnings."""

    beliefs: list[str] = field(default_factory=list)
    conclusions: list[str] = field(
        default_factory=list
    )  # each should cite ledger/trace ids
    plan: str = ""
    open_questions: list[str] = field(default_factory=list)
    ideas_usage: dict[str, str] = field(
        default_factory=dict
    )  # idea id -> used/discarded/why
