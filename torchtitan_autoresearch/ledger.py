"""The ledger: authoritative, append-only facts authored by the Harness.

The Harness authors the measured columns and *stores* the agent's verbatim
label/rationale (it never authors interpretation). The agent's deeper narrative
lives in its private memory and is surfaced only via `report()` (see
ARCHITECTURE.md section 5.5).
"""

from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass, field

HEADER = [
    "commit",
    "tps_mean",
    "tps_cv",
    "quality_margin",
    "verify",
    "verdict",
    "crash_class",
    "status",
    "label",
    "addresses",
    "rationale",
]


@dataclass
class Record:
    """One candidate's outcome. Facts plus the agent's verbatim words."""

    commit: str
    tps_mean: float = 0.0
    tps_cv: float = 0.0
    quality_margin: float = 0.0  # eval quality minus (golden - epsilon); >=0 passes
    verify: str = "n/a"  # faithful | affecting | fail | n/a
    verdict: str = "-"  # promote | reject | rerun | invalid | -
    crash_class: str = "-"
    status: str = "invalid"  # keep | discard | crash | oom | invalid
    label: str = ""
    addresses: list[str] = field(default_factory=list)  # ideas ids acted on
    rationale: str = ""

    def to_row(self) -> list[str]:
        d = asdict(self)
        d["addresses"] = ",".join(self.addresses)
        d["tps_mean"] = f"{int(self.tps_mean)}"
        d["tps_cv"] = f"{self.tps_cv:.3f}"
        d["quality_margin"] = f"{self.quality_margin:.4f}"
        return [str(d[c]) for c in HEADER]


class Ledger:
    """Append-only TSV of Records."""

    def __init__(self, path: str):
        self.path = path

    def append(self, rec: Record) -> None:
        new = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            if new:
                w.writerow(HEADER)
            w.writerow(rec.to_row())

    def read(self) -> list[dict]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, newline="") as f:
            return list(csv.DictReader(f, delimiter="\t"))
