# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The rank store of the block attention residual across pipeline stages.

Suffixes: T tokens, D model dim.
"""

import torch


class PPRankLocalCache:
    """Blocks a rank holds per micro-batch and the gradient deposits, shared by its stages."""

    def __init__(self) -> None:
        self._blocks: dict[int, dict[int, torch.Tensor]] = {}
        self._deposits: dict[tuple[int, int], torch.Tensor] = {}
        self._counts: dict[tuple[int, int], int] = {}

    def put(self, mb: int, block_idx: int, block_TD: torch.Tensor) -> None:
        self._blocks.setdefault(mb, {})[block_idx] = block_TD

    def blocks(self, mb: int) -> dict[int, torch.Tensor]:
        return self._blocks.get(mb, {})

    def release(self, mb: int) -> None:
        """Free the blocks of ``mb``; the deposits stay until collected."""
        self._blocks.pop(mb, None)

    def deposit(self, mb: int, block_idx: int, grad_TD: torch.Tensor) -> None:
        key = (mb, block_idx)
        prior = self._deposits.get(key)
        self._deposits[key] = grad_TD.clone() if prior is None else prior + grad_TD
        self._counts[key] = self._counts.get(key, 0) + 1

    def collect(self, mb: int, block_idx: int) -> tuple[torch.Tensor | None, int]:
        key = (mb, block_idx)
        return self._deposits.pop(key, None), self._counts.pop(key, 0)

    def has_deposits(self, mb: int) -> bool:
        return any(key[0] == mb for key in self._deposits)
