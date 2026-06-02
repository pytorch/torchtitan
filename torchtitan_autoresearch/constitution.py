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
    def module(self) -> str:
        """TorchTitan ``--module`` (model family), e.g. ``llama3``/``qwen3``."""
        return self.raw["workload"]["module"]

    @property
    def config_fn(self) -> str:
        """TorchTitan ``--config`` (config function), e.g. ``llama3_8b``."""
        return self.raw["workload"].get("config_fn", self.model_flavor)

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

    # --- quality (v1: faithfulness-only) ---
    def _faithful(self) -> dict[str, Any]:
        return self.raw.get("quality", {}).get("faithful", {})

    @property
    def verify_steps(self) -> int:
        """Deterministic steps in the faithfulness probe compared to the golden."""
        return int(self._faithful().get("verify_steps", 8))

    @property
    def band_headroom(self) -> float:
        """Multiplier on the golden's own rounding jitter to set the faithful band."""
        return float(self._faithful().get("band_headroom", 3.0))

    @property
    def trend_factor(self) -> float:
        """Allowed signed-mean deviation as a fraction of the band (bias detector)."""
        return float(self._faithful().get("trend_factor", 0.5))

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
