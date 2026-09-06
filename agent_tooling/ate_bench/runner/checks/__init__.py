"""Programmatic correctness checks for ATE-Bench operate/profile + new-feature tasks.

Each check exposes a ``check(...) -> CheckResult`` function and a small CLI. The
checks verify the *artifact* the agent produced (or the training log), matching
the paper's "verify the artifact, not the path taken" protocol (Appendix B.2.2,
B.3.2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_ANSI = re.compile(r"\x1b\[[0-9;]*m")

# TorchTitan logs: "step: {N:2}  loss: {avg:8.5f} ...". (metrics.py:527-528)
_STEP_LOSS = re.compile(
    r"step:\s*(\d+).*?loss:\s*([0-9]*\.?[0-9]+(?:[eE][+-]?\d+)?)"
)


def strip_ansi(text: str) -> str:
    return _ANSI.sub("", text)


def parse_loss_log(text: str) -> list[tuple[int, float]]:
    """Extract [(step, loss), ...] from a TorchTitan training log.

    Only training lines are returned ("validate step:" lines are skipped).
    """
    out: list[tuple[int, float]] = []
    for line in strip_ansi(text).splitlines():
        if "validate" in line.lower():
            continue
        m = _STEP_LOSS.search(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


@dataclass
class CheckResult:
    passed: bool
    detail: str
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "detail": self.detail, "extra": self.extra}
