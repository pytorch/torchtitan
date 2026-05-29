"""Load the binding rules from `Constitution.md` (the human's binding channel).

The constitution is the enforced source of truth (ARCHITECTURE.md section 4.1).
The human authors it as prose plus one fenced ```json``` block; the harness reads
the JSON block. Keeping the machine-readable rules in a JSON fence (stdlib only,
no YAML dependency) makes parsing robust while the surrounding prose stays
human-facing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_JSON_FENCE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class Rules:
    """Typed view over the constitution. Unknown keys stay accessible via raw."""

    raw: dict[str, Any]

    # --- objective ---
    @property
    def maximize(self) -> str:
        return self.raw["objective"]["maximize"]

    # --- workload (locked) ---
    @property
    def model_flavor(self) -> str:
        return self.raw["workload"]["model_flavor"]

    @property
    def dataset(self) -> str:
        return self.raw["workload"]["dataset"]

    @property
    def seq_len(self) -> int:
        return int(self.raw["workload"]["seq_len"])

    @property
    def ngpu(self) -> int | str:
        """Raw world-size setting; ``"auto"`` means detect at run start."""
        return self.raw["workload"]["ngpu"]

    def resolve_ngpu(self) -> int:
        """Resolve the world size, detecting the GPU count when set to ``auto``.

        Hardware is not hardcoded: ``auto`` uses the visible single-node GPU
        count, which the harness then fixes for the whole run so throughput stays
        comparable across candidates.
        """
        val = self.raw["workload"]["ngpu"]
        if val != "auto":
            return int(val)
        import torch

        return torch.cuda.device_count()

    # --- quality (locked) ---
    @property
    def epsilon_rel(self) -> float:
        return float(self.raw["quality"]["epsilon_rel"])

    @property
    def quality_affecting(self) -> set[str]:
        return set(self.raw["quality"]["quality_affecting"])

    @property
    def coupled(self) -> list[list[str]]:
        return self.raw["quality"].get("coupled", [])

    # --- scope ---
    @property
    def editable_files(self) -> list[str]:
        return self.raw["editable"]["files"]

    @property
    def fixed_fields(self) -> list[str]:
        return self.raw["fixed_fields"]

    @property
    def locked_paths(self) -> list[str]:
        return self.raw["locked_paths"]

    @property
    def banned_workload_fields(self) -> list[str]:
        return self.raw["banned_workload_fields"]

    # --- provenance (branch lifecycle) ---
    @property
    def base_commit(self) -> str:
        return self.raw.get("provenance", {}).get("base_commit", "HEAD")

    @property
    def branch_pattern(self) -> str:
        return self.raw.get("provenance", {}).get("branch_pattern", "autoresearch/{tag}")

    @property
    def allow_resume(self) -> bool:
        return bool(self.raw.get("provenance", {}).get("allow_resume", False))

    # --- measurement ---
    @property
    def log_freq(self) -> int:
        return int(self.raw["measurement"]["log_freq"])

    def window(self, mode: str) -> tuple[int, int]:
        w = self.raw["measurement"][mode]["window"]
        return (int(w[0]), int(w[1]))

    def steps(self, mode: str) -> int:
        return int(self.raw["measurement"][mode]["steps"])

    # --- significance ---
    @property
    def significance(self) -> dict[str, Any]:
        return self.raw["significance"]

    # --- substrate ---
    @property
    def family_defer_substrate(self) -> int:
        return int(self.raw["substrate"]["family_defer_substrate"])

    @property
    def family_defer_other(self) -> int:
        return int(self.raw["substrate"]["family_defer_other"])


def load_constitution(path: str) -> Rules:
    """Parse the JSON fence out of a Constitution.md and return typed Rules."""
    with open(path) as f:
        text = f.read()
    m = _JSON_FENCE.search(text)
    if not m:
        raise ValueError(f"no ```json``` rules block found in {path}")
    return Rules(raw=json.loads(m.group(1)))
