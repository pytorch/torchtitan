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
    """Multi-dimensional RoPE for Qwen3.5 temporal/height/width positions.

    Standard per-layer RoPE: each full-attention layer owns an ``MRoPE`` and
    applies it through ``RoPE.forward`` -> ``_reshape_cache`` -> ``apply_rotary_emb``.
    The only override is ``_reshape_cache``: for 3D ``(batch, seq, 3)`` MRoPE
    positions it builds an interleaved cos/sin cache; for 2D ``(batch, seq)`` text
    positions it falls back to the plain ``CosSinRoPE`` per-token lookup.
    """

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
        """Build a query-broadcastable cos/sin cache.

        Dispatches on position rank: 3D ``(batch, seq, 3)`` MRoPE positions take
        the interleaved scatter; everything else (2D text positions or ``None``)
        falls back to the plain ``CosSinRoPE`` lookup.
        """
        if positions is not None and positions.ndim == 3:
            return self._compute_mrope_cache(positions)
        return super()._reshape_cache(query, positions)

    def _compute_mrope_cache(self, position_ids: torch.Tensor) -> torch.Tensor:
        """Build the interleaved cos/sin cache for 3D MRoPE positions.

        Args:
            position_ids: ``(batch, seq, 3)`` T/H/W positions. Plain, or a DTensor
                under TP matching the rope ``cache`` buffer's Replicate placement.

        Returns:
            ``(batch, seq, 1, dim * 2)`` cache, broadcastable to the
            ``(batch, seq, n_heads, rotary_dim)`` query/key in ``apply_rotary_emb``.

        The scatter runs on plain local tensors. Under TP the ``cache`` buffer is a
        Replicate DTensor, so it is unwrapped to local here and the result is
        re-distributed with the buffer's placements, yielding a DTensor that
        composes with the sharded query/key without any manual wrapping in the
        attention forward.
        """
        cfg = self.config
        assert isinstance(cfg, MRoPE.Config)

        rope_cache = self.cache
        cache_dtensor = rope_cache if isinstance(rope_cache, DTensor) else None
        if cache_dtensor is not None:
            rope_cache = cache_dtensor.to_local()
        pos = (
            position_ids.to_local()
            if isinstance(position_ids, DTensor)
            else position_ids
        )
        pos = pos.to(device=rope_cache.device)

        _maybe_check_max_pos(pos, max_valid_pos=rope_cache.shape[0] - 1)
        head_dim = rope_cache.shape[-1] // 2
        cos_cache = rope_cache[:, :head_dim]
        sin_cache = rope_cache[:, head_dim:]

        # Start from temporal positions for all dimensions, then overwrite the
        # height/width interleaved sections with their own position IDs.
        # ``pos`` is (batch, seq, 3); the last axis selects T/H/W.
        t_pos = pos[..., 0].long()
        mrope_cos = cos_cache[t_pos]
        mrope_sin = sin_cache[t_pos]

        half = head_dim // 2
        for dim, offset in enumerate((1, 2), start=1):
            length = cfg.mrope_section[dim] * 3
            low = torch.arange(offset, length, 3, device=rope_cache.device)
            col_indices = torch.cat([low, low + half])
            dim_pos = pos[..., dim].long()
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
