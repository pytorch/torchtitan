# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import pickle
from typing import Any

import torch.distributed as dist
from torch.distributed.checkpoint.stateful import Stateful
from torchtitan.tools.rng import RngSnapshot

logger = logging.getLogger(__name__)
RNG_STATE = "rng_state"


class RngState(Stateful):
    """Stage rank-local RNG state for explicit restoration after other state loads."""

    def __init__(self) -> None:
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.pending: RngSnapshot | None = None

    def state_dict(self) -> dict[str, Any]:
        # Serialize on the calling thread, before asynchronous staging starts.
        return {f"rank_{self.rank}": pickle.dumps(RngSnapshot.capture())}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.pending = pickle.loads(state_dict[f"rank_{self.rank}"])

    def can_load(self, keys: set[str]) -> bool:
        expected = {f"{RNG_STATE}.rank_{rank}" for rank in range(self.world_size)}
        actual = {key for key in keys if key.startswith(f"{RNG_STATE}.")}
        if actual != expected:
            logger.warning(
                "Checkpoint RNG rank set is absent or incompatible; restoring other "
                "training state without RNG continuity."
            )
            return False
        return True

    def restore(self) -> None:
        snapshot, self.pending = self.pending, None
        compatible = snapshot is not None and snapshot.compatible()
        if dist.is_initialized():
            # Agree on every rank before changing any training stream.
            layouts: list[Any] = [None] * self.world_size
            dist.all_gather_object(layouts, compatible)
            compatible = all(layouts)
        if not compatible:
            logger.warning(
                "Checkpoint RNG device layout differs; restoring other training "
                "state without RNG continuity on every rank."
            )
            return
        assert snapshot is not None
        snapshot.restore()
