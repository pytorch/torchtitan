# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
from torch.distributed.tensor import distribute_tensor, DTensor

from torchtitan.models.common.rope import _maybe_check_max_pos, CosSinRoPE


class MRoPE(CosSinRoPE):
    """Multi-dimensional RoPE for Qwen3-VL temporal/height/width positions."""

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])

    def __init__(self, config: Config):
        if len(config.mrope_section) != 3:
            raise ValueError(
                f"mrope_section must have 3 entries, got {config.mrope_section}."
            )
        super().__init__(config)

    def _reshape_cache(
        self,
        query: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build a position-specific cache for 3D MRoPE position IDs."""
        if positions is not None and positions.ndim == 3:
            return self._compute_mrope_cache(positions)
        return super()._reshape_cache(query, positions)

    def _compute_mrope_cache(self, position_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        assert isinstance(cfg, MRoPE.Config)
        if position_ids.shape[0] != 3:
            raise ValueError(
                f"MRoPE position IDs must have shape (3, batch, seq), "
                f"got {tuple(position_ids.shape)}."
            )

        rope_cache = self.cache
        cache_dtensor = rope_cache if isinstance(rope_cache, DTensor) else None
        if cache_dtensor is not None:
            rope_cache = cache_dtensor.to_local()

        position_dtensor = position_ids if isinstance(position_ids, DTensor) else None
        pos_local = (
            position_dtensor.to_local()
            if position_dtensor is not None
            else position_ids
        )
        pos_local = pos_local.to(device=rope_cache.device)

        _maybe_check_max_pos(
            pos_local,
            max_valid_pos=rope_cache.shape[0] - 1,
        )
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
        if cache_dtensor is not None:
            return distribute_tensor(
                mrope_cache,
                cache_dtensor.device_mesh,
                list(cache_dtensor.placements),
            )
        return mrope_cache
