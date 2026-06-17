# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Periodic per-rank weight-hash logging.

Used to validate that two training runs (e.g. default ProcessGroup vs
torchcomms) produce bit-identical model state. Each rank hashes its own
local parameter shards — no extra collectives — and appends a JSONL record
per logged step. Compare two runs with ``scripts/hash_compare.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from torchtitan.tools.logging import logger


@dataclass(kw_only=True, slots=True)
class WeightHashConfig:
    enable: bool = False
    """Enable periodic weight-hash logging for numerical-equivalence validation."""

    interval: int = 50
    """Hash and log every N optimizer steps (also always logs step 0)."""

    log_per_param: bool = True
    """If True, record one hash per parameter FQN in addition to the global hash.
    Lets ``hash_compare.py`` localize the first diverging parameter."""


class WeightHasher:
    """Per-rank weight hasher. One instance per Trainer.

    Writes ``{dump_folder}/weight_hashes/rank{R}.jsonl``. Hashing uses local
    shards only (no collectives), so it's safe to call on every rank
    including under pipeline parallelism.
    """

    def __init__(
        self,
        *,
        config: WeightHashConfig,
        dump_folder: str,
        rank: int,
    ) -> None:
        self.config = config
        self.rank = rank
        self._file = None
        if not config.enable:
            return
        out_dir = os.path.join(dump_folder, "weight_hashes")
        os.makedirs(out_dir, exist_ok=True)
        # Truncate at start so a re-run from step 0 doesn't append stale records.
        self._path = os.path.join(out_dir, f"rank{rank}.jsonl")
        self._file = open(self._path, "w", buffering=1)  # line-buffered
        logger.info(f"Weight-hash logging enabled: {self._path}")

    def should_hash(self, step: int) -> bool:
        if not self.config.enable:
            return False
        return step == 0 or step % self.config.interval == 0

    @torch.no_grad()
    def hash_step(
        self,
        *,
        step: int,
        model_parts: list[torch.nn.Module],
    ) -> None:
        if self._file is None:
            return
        global_hasher = hashlib.sha256()
        per_param: dict[str, str] = {}

        for model_idx, model in enumerate(model_parts):
            # Sorted FQN order ensures deterministic hashing input order
            # regardless of dict iteration order changes.
            for fqn, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
                key = f"m{model_idx}.{fqn}"
                # Hash local shard only — no collective, safe under PP.
                local = param.to_local() if isinstance(param, DTensor) else param
                # Promote to fp64 so bit-identical math produces bit-identical
                # bytes even if storage dtype has padding quirks.
                buf = (
                    local.detach()
                    .to(torch.float64)
                    .contiguous()
                    .cpu()
                    .numpy()
                    .tobytes()
                )
                global_hasher.update(key.encode("utf-8"))
                global_hasher.update(buf)
                if self.config.log_per_param:
                    per_param[key] = hashlib.sha256(buf).hexdigest()

        record: dict[str, Any] = {
            "step": step,
            "rank": self.rank,
            "global_hash": global_hasher.hexdigest(),
        }
        if self.config.log_per_param:
            record["per_param"] = per_param
        self._file.write(json.dumps(record) + "\n")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
