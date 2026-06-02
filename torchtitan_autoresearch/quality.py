# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verify-routes-quality: the one quality decision (v1 = faithfulness only).

Verify decides by MEASURED faithfulness, not by an agent declaration. A candidate
whose short deterministic loss+grad_norm trajectory stays within the golden's own
rounding noise (no magnitude excursion, no directional bias) is faithful => the
training trajectory is the golden's => quality preserved by construction, no eval.

v1 policy: a candidate that is NOT faithful is "affecting" and is REJECTED. There
is no held-out eval and no quality floor in v1 -- the only way to be kept is to be
faithful, so every champion is numerically indistinguishable from the golden.
"""

from __future__ import annotations

from dataclasses import dataclass

from torchtitan_autoresearch.constitution import Rules
from torchtitan_autoresearch.executor import Executor
from torchtitan_autoresearch.state import HarnessState
from torchtitan_autoresearch.types import Candidate, Quality


@dataclass
class QualityOutcome:
    quality: Quality
    verify: str  # faithful | affecting
    detail: str = ""


def verify_routes_quality(
    c: Candidate, executor: Executor, state: HarnessState, rules: Rules
) -> QualityOutcome:
    vr = executor.run_verify(c)
    if vr.faithful:
        # Numerically faithful to the golden => quality preserved by construction.
        return QualityOutcome(
            Quality(checked=False, passed=True, margin=0.0), "faithful", vr.detail
        )
    # v1: affecting changes (math moved beyond the golden's noise) are rejected
    # outright -- no held-out eval, no from-scratch validation.
    return QualityOutcome(
        Quality(checked=False, passed=False, margin=0.0), "affecting", vr.detail
    )
