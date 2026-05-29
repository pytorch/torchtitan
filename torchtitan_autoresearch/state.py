"""Harness reference state: golden, champion, noise models, family budget.

This is harness-owned facts/state (ARCHITECTURE.md section 5.5), persisted as
JSON so decisions are consistent across invocations. The golden's eval loss is
the absolute quality bar; the champion is only the throughput target plus the
verify increment audit.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

from torchtitan_autoresearch import crash_classify as cc
from torchtitan_autoresearch import significance as sig


@dataclass
class HarnessState:
    golden_commit: str = ""
    golden_eval_loss: float | None = None  # the quality bar (lower is better)
    champion_commit: str = ""
    champion_tps: list[float] = field(default_factory=list)
    tps_cv: float = 0.02
    tps_tail_pct: float = 4.5
    eval_noise_rel: float = 0.0  # relative std of eval loss (sets margin meaningfulness)
    family_streaks: dict[str, int] = field(default_factory=dict)
    family_deferred: list[str] = field(default_factory=list)

    # --- persistence ---
    @classmethod
    def load(cls, path: str) -> "HarnessState":
        if not os.path.exists(path):
            return cls()
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    # --- derived helpers ---
    def tps_noise(self) -> sig.NoiseModel:
        return sig.NoiseModel(cv=self.tps_cv, tail_pct=self.tps_tail_pct)

    def family_budget(self, defer_substrate: int, defer_other: int) -> cc.FamilyBudget:
        b = cc.FamilyBudget(max_substrate=defer_substrate, max_other=defer_other)
        b.deferred = set(self.family_deferred)
        for k, v in self.family_streaks.items():
            fam, cls_ = k.split("|", 1)
            b._streak[(fam, cls_)] = v
        return b

    def absorb_budget(self, b: cc.FamilyBudget) -> None:
        self.family_deferred = sorted(b.deferred)
        self.family_streaks = {f"{fam}|{cls_}": n for (fam, cls_), n in b._streak.items()}
