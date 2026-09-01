# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT3d vision encoder used by Kimi K3.

Shape suffixes:
- M = total merged tokens
- F = merged feature dimension
- O = projected text dimension
"""

from dataclasses import dataclass, field

import torch

from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import GELU, RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.kimi_k2_7.vision_encoder import (
    _tpool_patch_merger,
    MoonViTEncoder,
)
from torchtitan.protocols.module import Module


class KimiK3VisionProjector(Module):
    """PatchMergerMLPV2 projector from merged vision features to text width."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        linear_1: Linear.Config
        linear_2: Linear.Config
        post_norm: RMSNorm.Config
        activation: GELU.Config = field(default_factory=GELU.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.linear_1 = config.linear_1.build()
        self.linear_2 = config.linear_2.build()
        self.post_norm = config.post_norm.build()
        self.activation = config.activation.build()

    def forward(self, merged_MF: torch.Tensor) -> torch.Tensor:
        projected_MO = self.linear_2(self.activation(self.linear_1(merged_MF)))
        return self.post_norm(projected_MO)


class KimiK3VisionEncoder(MoonViTEncoder):
    @dataclass(kw_only=True, slots=True)
    class Config(MoonViTEncoder.Config):
        patch_size: int
        in_channels: int
        merge_kernel_size: tuple[int, int]  # pyrefly: ignore [bad-override]
        max_num_frames: int

        final_norm: RMSNorm.Config  # pyrefly: ignore [bad-override]
        projector: KimiK3VisionProjector.Config  # pyrefly: ignore [bad-override]

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
        part: str | None = None,
        upto_block: int | None = None,
        lo: int | None = None,
        hi: int | None = None,
        from_block: int | None = None,
    ) -> torch.Tensor:
        """The shared tower's forward, plus the DEP share entry points.

        ``part`` selects one share of a tower spanning PP stages (report sec
        5.2.3 clause 2), reached THROUGH this forward: FSDP2 hooks __call__,
        and calling forward_head directly meets still-sharded DTensor weights.
        Without ``part`` this defers to the tower unchanged.
        """
        if part is not None:
            if part == "head":
                return self.forward_head(pixel_values, grid_thw, upto_block=upto_block)
            if part == "body":
                return self.forward_body(pixel_values, grid_thw, lo=lo, hi=hi)
            if part == "tail":
                return self.forward_tail(
                    pixel_values, grid_thw, from_block=from_block or 0
                )
            raise ValueError(f"unknown tower part {part!r}")

        return super().forward(pixel_values, grid_thw=grid_thw)

    # ---- DEP: the tower split across pipeline stages (report sec 5.2.3) ----
    #
    # The report asks for vision forward and backward balanced across PP
    # stages, so the tower runs as contiguous block ranges: head / body / tail.

    def _run_blocks(self, x, *, rope_cache, block_slice=None, attention_mask=None):
        blocks = list(self.layers.values())
        if block_slice is not None:
            blocks = blocks[block_slice]
        for block in blocks:
            x = block(
                x,
                rope_cache=rope_cache,
                rope_apply=ComplexRoPE.apply_rotary_emb,
                attention_mask=attention_mask,
            )
        return x

    def block_bounds(self, num_shares: int) -> list[tuple[int, int]]:
        """Split the tower's blocks into ``num_shares`` contiguous ranges.

        Report 5.2.3 balances vision passes across PP stages, so shares are as
        even as possible. A remainder goes to the LAST shares, because share 0
        also carries ``patch_embed`` and the final share's projector is cheaper
        than that -- giving share 0 an extra block as well would make the least
        balanced stage worse.
        """
        n = len(self.layers)
        if num_shares < 1 or num_shares > n:
            raise ValueError(
                f"cannot split {n} encoder block(s) into {num_shares} share(s)"
            )
        base, extra = divmod(n, num_shares)
        bounds, lo = [], 0
        for i in range(num_shares):
            hi = lo + base + (1 if i >= num_shares - extra else 0)
            bounds.append((lo, hi))
            lo = hi
        return bounds

    # Every share recomputes position tables and the block-diagonal mask from
    # grid_thw: RoPE indices and segment bounds do not survive PP's dummy
    # metadata values, so only float activations cross a stage boundary.

    def _share_block_inputs(self, grid_thw, num_tokens, device):
        """Position tables and attention mask for a share, from the grid alone."""
        import spmd_types as spmd

        from torchtitan.models.common.vision_encoder import create_block_diagonal_mask

        grids = grid_thw.tolist()
        learned_pos, rope_cache = self.compute_position_embeddings(grids)
        with spmd.no_typecheck():
            mask = create_block_diagonal_mask(grid_thw.prod(dim=-1), num_tokens, device)
        return grids, learned_pos, rope_cache, mask

    def forward_head(self, pixel_values, grid_thw, *, upto_block: int):
        """Patch embed plus blocks ``[0, upto_block)``, without the final norm.

        The first share when the tower spans PP stages. Returns patch hidden
        states, not features -- the projector belongs to the last share.
        """
        _, learned_pos, rope_cache, mask = self._share_block_inputs(
            grid_thw, pixel_values.shape[0], pixel_values.device
        )
        x = self.patch_embed(pixel_values) + learned_pos
        return self._run_blocks(
            x,
            rope_cache=rope_cache,
            block_slice=slice(0, upto_block),
            attention_mask=mask,
        )

    def forward_body(self, x, grid_thw, *, lo: int, hi: int):
        """Blocks ``[lo, hi)`` only -- a middle share, no norm, no projector."""
        _, _, rope_cache, mask = self._share_block_inputs(
            grid_thw, x.shape[0], x.device
        )
        return self._run_blocks(
            x,
            rope_cache=rope_cache,
            block_slice=slice(lo, hi),
            attention_mask=mask,
        )

    def forward_tail(self, x, grid_thw, *, from_block: int):
        """Blocks ``[from_block, end)``, the final norm, the merge, the projector.

        The last share, and the only one that produces features.
        """
        grids, _, rope_cache, mask = self._share_block_inputs(
            grid_thw, x.shape[0], x.device
        )
        x = self._run_blocks(
            x,
            rope_cache=rope_cache,
            block_slice=slice(from_block, len(self.layers)),
            attention_mask=mask,
        )
        x = self.final_norm(x)
        return self.projector(_tpool_patch_merger(x, grids, self.merge_kernel_size))
