# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Literal

import torch
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.models.common.rope import _maybe_check_max_pos, RoPE


class MRoPE(RoPE):
    """Multi-dimensional RoPE for Qwen3-VL temporal/height/width positions."""

    @dataclass(kw_only=True, slots=True)
    class Config(RoPE.Config):
        backend: Literal["cos_sin"] = "cos_sin"
        mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])

    def __init__(self, config: Config):
        if config.backend != "cos_sin":
            raise ValueError("MRoPE only supports cos/sin RoPE.")
        if len(config.mrope_section) != 3:
            raise ValueError(
                f"mrope_section must have 3 entries, got {config.mrope_section}."
            )
        super().__init__(config)

    def _cache_for_positions(
        self, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        if positions is not None and positions.ndim == 3:
            return self._compute_mrope_cache(positions)
        return self.cache

    def _compute_mrope_cache(self, position_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        assert isinstance(cfg, MRoPE.Config)
        if position_ids.shape[0] != 3:
            raise ValueError(
                f"MRoPE position IDs must have shape (3, batch, seq), "
                f"got {tuple(position_ids.shape)}."
            )

        rope_cache = self.cache
        cache_was_dtensor = isinstance(rope_cache, DTensor)
        if cache_was_dtensor:
            rope_cache = rope_cache.to_local()
        pos_local = (
            position_ids.to_local()
            if isinstance(position_ids, DTensor)
            else position_ids
        )
        pos_local = pos_local.to(device=rope_cache.device)

        _maybe_check_max_pos(pos_local, max_valid_pos=rope_cache.shape[0] - 1)
        head_dim = rope_cache.shape[-1] // 2
        cos_cache = rope_cache[:, :head_dim]
        sin_cache = rope_cache[:, head_dim:]

        # Start from temporal positions for all dimensions, then overwrite the
        # height/width interleaved sections with their own position IDs.
        t_pos = pos_local[0].long()
        mrope_cos = cos_cache[t_pos]
        mrope_sin = sin_cache[t_pos]

        half = head_dim // 2
        for dim, offset in enumerate((1, 2), start=1):
            length = cfg.mrope_section[dim] * 3
            low = torch.arange(offset, length, 3, device=rope_cache.device)
            col_indices = torch.cat([low, low + half])
            dim_pos = pos_local[dim].long()
            mrope_cos[..., col_indices] = cos_cache[:, col_indices][dim_pos]
            mrope_sin[..., col_indices] = sin_cache[:, col_indices][dim_pos]

        mrope_cache = torch.cat([mrope_cos, mrope_sin], dim=-1).unsqueeze(2)
        if cache_was_dtensor:
            mrope_cache = distribute_tensor(
                mrope_cache,
                self.cache.device_mesh,
                list(self.cache.placements),
            )
        return mrope_cache
