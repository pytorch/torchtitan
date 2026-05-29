"""Verify-routes-quality: the one quality decision, anchored to the golden.

Step 3 of the pipeline (ARCHITECTURE.md sections 5.2-5.3). Verify decides the
path by measured faithfulness, not by an agent declaration: faithful-to-golden
implies quality preserved (no eval); not-faithful means quality-affecting, so the
held-out eval must clear the absolute one-sided floor `eval_loss <= golden *
(1 + epsilon)`. Improvement is always allowed; only degradation past the floor
fails.
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
    verify: str  # faithful | affecting | fail
    eval_crash_text: str | None = None  # set when the eval run itself crashed


def verify_routes_quality(
    c: Candidate, executor: Executor, state: HarnessState, rules: Rules
) -> QualityOutcome:
    vr = executor.run_verify(c)
    if vr.faithful:
        # Numerically faithful to the golden => quality preserved by construction.
        return QualityOutcome(Quality(checked=False, passed=True, margin=0.0), "faithful")

    er = executor.run_eval(c)
    if not er.ok:
        return QualityOutcome(
            Quality(checked=True, passed=False, margin=0.0), "fail", er.crash_text
        )

    golden = state.golden_eval_loss
    if golden is None:
        # No quality bar yet: the first quality-affecting run cannot be floored;
        # treat as not-passed so the bootstrap golden is a high-precision recipe.
        return QualityOutcome(Quality(checked=True, passed=False, margin=0.0), "affecting")

    floor = golden * (1.0 + rules.epsilon_rel)  # max tolerated eval loss
    passed = er.eval_loss <= floor
    margin = (floor - er.eval_loss) / golden  # relative; >= 0 passes
    return QualityOutcome(Quality(checked=True, passed=passed, margin=margin), "affecting")
