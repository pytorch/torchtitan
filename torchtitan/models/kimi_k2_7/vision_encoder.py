# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT3d vision encoder + multimodal projector for Kimi K2.5.

Reference (SGLang):
https://github.com/sgl-project/sglang/blob/e0c0c0a45cb1bda90392bfa2bba4184f5b0638a0/python/sglang/srt/models/kimi_k25.py

Shape suffixes:
- N = num visual items
- P = max num of patches per visual item (padded)
- D = vision dim
- M = merged tokens
- K = merged feature dim (kh*kw*D)
"""

from dataclasses import dataclass, field

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import GELU, LayerNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.common.vision_encoder import (
    compiled_create_block_mask,
    compute_2d_rope_cache,
    get_temporal_pos_embed,
    get_vision_block_mask_mod,
    VisionRotaryEmbedding2D,
    VisionTransformerBlock,
)
from torchtitan.protocols.module import Module, ModuleDict


def _compute_learned_pos_embeds(
    pos_embed: torch.Tensor,
    grids: list[list[int]],
    max_num_patch: int,
    interpolation_mode: str,
) -> torch.Tensor:
    """Interpolated learnable 2D spatial pos-emb + fixed sinusoidal temporal.

    The learnable ``(height, width, dim)`` spatial table is interpolated to each
    item's ``(h, w)`` patch grid (raster order). For video (``t > 1``) the
    spatial embedding is repeated per frame and summed with a fixed 1D sinusoidal
    temporal embedding (``get_temporal_pos_embed``; only video hits this path).

    Args:
        pos_embed: (height, width, dim) learnable spatial position table.
        grids: per-item ``[t, h, w]`` patch counts as host ints (``grid_thw``
            read to CPU once by the caller, so the per-item loop adds no syncs).
        max_num_patch: Padded sequence length.
        interpolation_mode: ``F.interpolate`` mode (e.g. ``"bicubic"``).

    Returns:
        (N, max_num_patch, dim) padded position embeddings to add to the patches.
    """
    height, width, dim = pos_embed.shape
    pos = pos_embed.new_zeros(len(grids), max_num_patch, dim)
    if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
        pos = spmd.mutate_type(pos, src=spmd.R, dst={"dp": spmd.V, "tp": spmd.I})

    # (dim, height, width) for F.interpolate; .float() for bicubic.
    grid_table = pos_embed.permute(2, 0, 1).unsqueeze(0).float()

    hw_to_indices: dict[tuple[int, int], list[int]] = {}
    for i, (_, h, w) in enumerate(grids):
        hw_to_indices.setdefault((h, w), []).append(i)

    for (h, w), indices in hw_to_indices.items():
        if (h, w) == (height, width):
            pos_hw = pos_embed.flatten(end_dim=1)
        else:
            pos_hw = (
                F.interpolate(grid_table, size=(h, w), mode=interpolation_mode)
                .squeeze(0)
                .permute(1, 2, 0)
                .reshape(h * w, dim)
                .to(pos_embed.dtype)
            )
        for i in indices:
            t = grids[i][0]
            seq_len = t * h * w
            if t == 1:
                pos[i, :seq_len] = pos_hw
            else:
                # (t, 1, dim) to broadcast the per-frame term over h*w patches.
                time_weight = get_temporal_pos_embed(
                    t, dim, device=pos.device
                ).unsqueeze(1)
                frames = pos_hw.unsqueeze(0).repeat(t, 1, 1)
                frames = frames + time_weight.to(frames.dtype)
                pos[i, :seq_len] = frames.reshape(seq_len, dim)

    return pos


def _tpool_patch_merger(
    hidden_NPD: torch.Tensor,
    grids: list[list[int]],
    merge_kernel_size: tuple[int, int],
) -> torch.Tensor:
    """Temporal mean pooling + spatial merge over the padded batch.

    For each item ``(t, h, w)``: reshape its valid patches to
    ``(t, h, w, D)``, mean over the temporal axis, then group spatial
    ``kh x kw`` neighbors and concatenate them along the feature axis, yielding
    ``(h/kh * w/kw)`` merged tokens of dim ``kh*kw*D``.

    Args:
        hidden_NPD: (N, P, D) padded patch features.
        grids: per-item ``[t, h, w]`` patch counts as host ints (``grid_thw``
            read to CPU once by the caller, so the per-item loop adds no syncs).
        merge_kernel_size: ``(kh, kw)`` spatial merge factors.

    Returns:
        merged: ``(N, max_merged, kh*kw*D)`` padded merged tokens, where
        ``max_merged = max_i (h_i/kh) * (w_i/kw)``. The valid token count per
        item is ``(h/kh) * (w/kw)`` (recomputed by the caller from ``grids``
        for the scatter).
    """
    num_vision, _, d_model = hidden_NPD.shape
    kh, kw = merge_kernel_size
    merged_dim = kh * kw * d_model

    max_merged = max((h // kh) * (w // kw) for _, h, w in grids)
    merged = hidden_NPD.new_zeros(num_vision, max_merged, merged_dim)
    if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
        merged = spmd.mutate_type(merged, src=spmd.R, dst={"dp": spmd.V, "tp": spmd.I})

    for i, (t, h, w) in enumerate(grids):
        seq = hidden_NPD[i, : t * h * w]
        new_h, new_w = h // kh, w // kw
        # (t, new_h, kh, new_w, kw, D) -> mean over t -> group spatial kernel.
        seq = seq.view(t, new_h, kh, new_w, kw, d_model)
        seq = seq.permute(0, 1, 3, 2, 4, 5).mean(dim=0)
        seq = seq.reshape(new_h * new_w, merged_dim)
        merged[i, : new_h * new_w] = seq

    return merged


class VisionProjector(Module):
    """Project merged vision features to the language-model hidden size.

    Applies a per-patch LayerNorm (on ``vt_hidden_size``), then a 2-layer MLP
    over the concatenated spatial-merge features. This is a unimodal vision
    projection head; the cross-modal fusion (scattering these features into the
    text sequence) happens later in the model, not here.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        vt_hidden_size: int
        merged_dim: int
        pre_norm: LayerNorm.Config
        linear_1: Linear.Config
        linear_2: Linear.Config
        act_fn: GELU.Config = field(default_factory=GELU.Config)

    def __init__(self, config: Config):
        super().__init__()
        self.vt_hidden_size = config.vt_hidden_size
        self.merged_dim = config.merged_dim
        self.pre_norm = config.pre_norm.build()
        self.linear_1 = config.linear_1.build()
        self.linear_2 = config.linear_2.build()
        self.act_fn = config.act_fn.build()

    def forward(self, merged_NMK: torch.Tensor) -> torch.Tensor:
        """Args: ``(N, M, kh*kw*vt_hidden_size)`` padded merged tokens.

        The pre-norm runs per-patch on ``vt_hidden_size``; the merged kernel
        features are then flattened for the projection MLP.
        """
        n, m, _ = merged_NMK.shape
        x = merged_NMK.view(n, m, -1, self.vt_hidden_size)
        x = self.pre_norm(x).view(n, m, self.merged_dim)
        x = self.linear_1(x)
        x = self.act_fn(x)
        x = self.linear_2(x)
        return x


class KimiK25VisionEncoder(Module):
    """MoonViT3d vision tower + multimodal projector for Kimi K2.5."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int = 1152
        num_layers: int = 27
        num_heads: int = 16

        patch_size: int = 14
        in_channels: int = 3
        merge_kernel_size: list[int] = field(default_factory=lambda: [2, 2])
        text_hidden_size: int = 7168

        # Learnable 2D spatial position table, shape (height, width, dim).
        init_pos_emb_height: int = 64
        init_pos_emb_width: int = 64
        interpolation_mode: str = "bicubic"

        # Sub-modules.
        patch_embed_proj: Linear.Config
        rotary_pos_emb: VisionRotaryEmbedding2D.Config
        block: VisionTransformerBlock.Config
        final_norm: LayerNorm.Config
        projector: VisionProjector.Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.merge_kernel_size = tuple(config.merge_kernel_size)
        self.interpolation_mode = config.interpolation_mode

        self.patch_embed = config.patch_embed_proj.build()
        # Learnable 2D spatial position table.
        self.pos_embed = nn.Parameter(
            torch.empty(
                config.init_pos_emb_height, config.init_pos_emb_width, config.dim
            )
        )
        self.rotary_pos_emb = config.rotary_pos_emb.build()
        self._cached_freq_table: torch.Tensor | None = None
        self.layers = ModuleDict(
            {str(idx): config.block.build() for idx in range(config.num_layers)}
        )
        self.final_norm = config.final_norm.build()
        self.projector = config.projector.build()

    def compute_position_embeddings(
        self, grids: list[list[int]], max_num_patch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both position embeddings for the padded batch.

        Delegates to two standalone helpers:
        - ``_compute_learned_pos_embeds``: interpolated learnable spatial table
          (plus a runtime sinusoidal temporal term for videos).
        - ``compute_2d_rope_cache``: 2D RoPE complex cache, gathering from the
          RoPE frequency table (cached here, regrown only when a larger side
          appears).

        Args:
            grids: per-item ``[t, h, w]`` patch counts as host ints (``grid_thw``
                read to CPU once by ``forward``).
            max_num_patch: Padded sequence length.

        Returns:
            learned_pos: ``(N, max_num_patch, dim)`` additive position embeddings.
            rope_cache: ``(N, max_num_patch, 1, head_dim/2)`` complex RoPE cache
                for ``ComplexRoPE.apply_rotary_emb``.
        """
        max_hw = max(max(h, w) for _, h, w in grids)
        if self._cached_freq_table is None or self._cached_freq_table.shape[0] < max_hw:
            self._cached_freq_table = self.rotary_pos_emb(max_hw)

        learned_pos = _compute_learned_pos_embeds(
            self.pos_embed, grids, max_num_patch, self.interpolation_mode
        )
        rope_cache = compute_2d_rope_cache(
            self._cached_freq_table,
            grids,
            max_num_patch,
            self.rotary_pos_emb.head_dim,
        )
        return learned_pos, rope_cache

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a padded batch of visual items.

        Args:
            pixel_values: ``(N, P, patch_dim)`` padded flattened patches.
            grid_thw: ``(N, 3)`` patch counts ``[t, h, w]`` per item.

        Returns:
            ``(N, max_merged, text_hidden_size)`` padded projected features.

        Each item's ``(h, w)`` must be divisible by ``merge_kernel_size`` and its
        ``t*h*w`` must fit in ``P`` (both dataloader-guaranteed; asserted below).
        Patches must arrive in raster order ``(t, h, w)`` (the dataloader's
        ``patch_order="raster"``); the 2D RoPE / position embeddings index
        ``row = p // w, col = p % w``.
        """
        num_vision, max_num_patch, _ = pixel_values.shape
        # One host sync for the whole forward: read the (N, 3) grid to CPU ints.
        grids = grid_thw.tolist()  # [[t, h, w], ...]

        # Grid contract (dataloader-guaranteed); assert so a bad batch fails
        # clearly instead of with a cryptic view/index error in the helpers.
        kh, kw = self.merge_kernel_size
        for t, h, w in grids:
            assert (
                h % kh == 0 and w % kw == 0
            ), f"grid {h}x{w} indivisible by {(kh, kw)}"
            assert t * h * w <= max_num_patch, f"t*h*w={t * h * w} > P={max_num_patch}"

        num_patch = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).to(
            torch.long
        )  # (N,)

        learned_pos, rope_cache = self.compute_position_embeddings(grids, max_num_patch)
        x = self.patch_embed(pixel_values) + learned_pos

        mask_mod = get_vision_block_mask_mod(num_patch)
        with spmd.no_typecheck():
            attention_mask = compiled_create_block_mask(
                mask_mod,
                num_vision,
                None,
                max_num_patch,
                max_num_patch,
                device=x.device,
            )

        for block in self.layers.values():
            x = block(
                x,
                rope_cache=rope_cache,
                rope_apply=ComplexRoPE.apply_rotary_emb,
                attention_mask=attention_mask,
            )

        x = self.final_norm(x)

        # Temporal pool + spatial merge, then project to the LLM hidden size.
        # pyrefly: ignore [bad-argument-type]
        merged = _tpool_patch_merger(x, grids, self.merge_kernel_size)
        return self.projector(merged)
