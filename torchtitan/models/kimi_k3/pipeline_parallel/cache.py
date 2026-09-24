# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The rank store of the block attention residual across pipeline stages.

Suffixes: N blocks, T tokens, D model dim.
"""

import torch


class PPRankLocalCache:
    """One [N, T, D] block buffer per micro-batch and the gradient deposits, shared by a rank's stages."""

    def __init__(self) -> None:
        self._rows: dict[int, torch.Tensor] = {}
        self._have: dict[int, set[int]] = {}
        self._deposits: dict[tuple[int, int], torch.Tensor] = {}
        self._counts: dict[tuple[int, int], int] = {}

    def allocate(self, mb: int, num_blocks: int, like_TD: torch.Tensor) -> None:
        if mb not in self._rows:
            self._rows[mb] = like_TD.new_empty(num_blocks, *like_TD.shape)
            self._have[mb] = set()

    def rows(self, mb: int, first: int, count: int) -> torch.Tensor:
        # Through .data, so a later write to another row does not bump a saved view's version.
        return self._rows[mb].data[first : first + count]

    def stack(self, mb: int, count: int) -> torch.Tensor:
        """Rows [0, count) as a [T, count, D] view."""
        return self.rows(mb, 0, count).transpose(0, 1)

    def put(self, mb: int, block_idx: int, block_TD: torch.Tensor) -> None:
        self._rows[mb][block_idx].copy_(block_TD)
        self._have[mb].add(block_idx)

    def mark(self, mb: int, blocks: list[int]) -> None:
        """Record blocks received in place into their rows."""
        self._have[mb].update(blocks)

    def blocks(self, mb: int) -> dict[int, torch.Tensor]:
        if mb not in self._rows:
            return {}
        return {b: self._rows[mb].data[b] for b in sorted(self._have[mb])}

    def release(self, mb: int) -> None:
        """Free the blocks of ``mb``; the deposits stay until collected."""
        self._rows.pop(mb, None)
        self._have.pop(mb, None)

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
