# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

    # --- quality eval horizon + noise calibration ---
    def _eval(self) -> dict[str, Any]:
        return self.raw["quality"].get("eval", {})

    @property
    def eval_tokens(self) -> int:
        """Token budget for the held-out eval (equal-compute across recipes)."""
        return int(self._eval().get("tokens", 0))

    @property
    def eval_fallback_steps(self) -> int:
        """Step count used only if the token budget can't be converted (no batch)."""
        return int(self._eval().get("fallback_steps", 50))

    @property
    def eval_val_steps(self) -> int:
        """Number of held-out validation batches averaged into the eval loss."""
        return int(self._eval().get("val_steps", 16))

    @property
    def eval_calibration_repeats(self) -> int:
        """Independent golden eval runs used to measure the eval-noise band."""
        return int(self._eval().get("calibration_repeats", 3))

    @property
    def eval_warm_steps(self) -> int:
        """Steps to pre-train the golden warm checkpoint (past warmup). 0 = from scratch."""
        return int(self._eval().get("warm_steps", 0))

    @property
    def eval_lr_total_steps(self) -> int:
        """Real LR-schedule horizon for warm evals (decoupled from run length). 0 = run length."""
        return int(self._eval().get("lr_total_steps", 0))

    @property
    def eval_z(self) -> float:
        """Multiplier on the eval-noise std for the quality floor's noise band."""
        return float(self._eval().get("z", 3.0))

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
        return self.raw.get("provenance", {}).get(
            "branch_pattern", "autoresearch/{tag}"
        )

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
