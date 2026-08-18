# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared model-agnostic ViT building blocks for VLM vision encoders: a
block-diagonal FlexAttention mask helper and the pre-norm transformer block
(attention + MLP) over a padded ``(N, P, D)`` batch.

RoPE differs per model, so each encoder passes it through the block to the
attention as two per-forward args: ``rope_cache`` (a tensor, so config-based
sharding can DTensor-wrap it before it meets the head-sharded q/k) and
``rope_apply`` (a pass-through callable ``(q, k, rope_cache) -> (q, k)``).

Shape suffixes:
- N = num visual items
- P = max patches per item (padded)
- D = vision dim
- H = num heads
- Dh = head dim
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import spmd_types as spmd
import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common import Linear
from torchtitan.models.common.attention import FlexAttention, local_head_split
from torchtitan.models.common.nn_modules import GELU, LayerNorm, RMSNorm
from torchtitan.models.common.rope import _maybe_wrap_positions
from torchtitan.protocols.module import Module

compiled_create_block_mask = torch.compile(create_block_mask)

# Applies rotary position embedding: (query, key, rope_cache) -> (query, key).
RopeApply = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]


def get_vision_block_mask_mod(num_patches: torch.Tensor) -> Callable:
    """Block-diagonal mask: each visual item attends only to its own patches.

    Args:
        num_patches: (N,) real (non-padding) patch count per visual item (N is
            the number of visual items, i.e. images/videos in the batch).
    """

    def mask_mod(b, h, q_idx, kv_idx):
        valid_q = q_idx < num_patches[b]
        valid_kv = kv_idx < num_patches[b]
        return valid_q & valid_kv

    return mask_mod


def get_temporal_pos_embed(
    num_frames: int,
    embed_dim: int,
    *,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Fixed 1D sinusoidal embeddings for the temporal axis (video frames).

    Returns ``(num_frames, embed_dim)`` float32; the standard 1D sincos formula
    over frame indices.

    Args:
        num_frames: Number of video frames (temporal positions).
        embed_dim: Embedding width per frame.
        base: Sinusoid base (longest wavelength); the conventional PE constant.
        device: Device for the returned tensor.
    """
    grid = torch.arange(num_frames, dtype=torch.float32, device=device)
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=device) / (
        embed_dim / 2.0
    )
    omega = 1.0 / base**omega
    out = torch.outer(grid, omega)
    return torch.cat([out.sin(), out.cos()], dim=1)


def compute_2d_rope_cache(
    freq_table: torch.Tensor,
    grids: list[list[int]],
    max_num_patch: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute the padded 2D-RoPE complex ``freqs_cis`` cache in raster order.

    For head-dim pair index ``k`` (``k`` in ``[0, head_dim/4)``), even output
    pairs are rotated by the *column* (x) position and odd pairs by the *row*
    (y) position. The per-axis angle for a position ``p`` is ``p * inv_freq[k]``;
    this looks it up by gathering row ``p`` of ``freq_table`` (built once by
    ``VisionRotaryEmbedding2D`` and cached by the encoder) rather than
    recomputing ``p * inv_freq`` each call. Frames repeat the spatial pattern.

    Returns a complex cache consumed by ``ComplexRoPE.apply_rotary_emb``; only
    the cache is 2D/per-grid, which is why it is built here rather than by the
    1D ``ComplexRoPE`` cache machinery.

    Args:
        freq_table: ``(max_hw, head_dim/4)`` position-to-frequency table, where
            ``freq_table[p, k] = p * inv_freq[k]``.
        grids: per-item ``[t, h, w]`` patch counts as host ints (``grid_thw``
            read to CPU once by the caller, so the per-item loop adds no syncs).
        max_num_patch: Padded sequence length.
        head_dim: Attention head dim (must be divisible by 4).

    Returns:
        ``(N, max_num_patch, 1, head_dim/2)`` complex64 (head axis = 1 to
        broadcast over the heads).
    """
    device = freq_table.device

    angles = freq_table.new_zeros(len(grids), max_num_patch, head_dim // 2)
    if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
        angles = spmd.mutate_type(angles, src=spmd.R, dst={"dp": spmd.V, "tp": spmd.I})

    # Group by (h, w) so the per-resolution angle grid is built once.
    hw_to_indices: dict[tuple[int, int], list[int]] = {}
    for i, (_, h, w) in enumerate(grids):
        hw_to_indices.setdefault((h, w), []).append(i)

    for (h, w), indices in hw_to_indices.items():
        # Raster order: position p -> (row = p // w, col = p % w). Gather each
        # axis's angles from the precomputed table (freq_table[pos] = pos*inv_freq).
        flat = torch.arange(h * w, device=device)
        flat = cast(torch.Tensor, _maybe_wrap_positions(flat, freq_table))
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            flat = spmd.mutate_type(flat, "tp", src=spmd.R, dst=spmd.I)
        x_ang = freq_table[flat % w]  # (h*w, head_dim/4) column
        y_ang = freq_table[flat // w]  # (h*w, head_dim/4) row
        # Interleave x/y so pair 2k uses x-position, pair 2k+1 uses y-position.
        ang = torch.stack([x_ang, y_ang], dim=-1).reshape(h * w, head_dim // 2)
        for i in indices:
            t = grids[i][0]
            seq_len = t * h * w
            angles[i, :seq_len] = ang.repeat(t, 1)

    # Complex unit-modulus cache; unsqueeze the head axis for broadcast.
    return torch.polar(torch.ones_like(angles), angles).unsqueeze(2)


class VisionRotaryEmbedding2D(Module):
    """2D rotary position embedding for the vision tower.

    Holds the per-axis frequencies ``inv_freq`` (``head_dim/4`` of them, shared
    by the row and column axes). ``forward(seqlen)`` returns the
    position-to-frequency table ``freq_table[p, k] = p * inv_freq[k]`` for
    positions up to ``seqlen``; ``compute_2d_rope_cache`` gathers per-patch
    row/col angles from it, and ``ComplexRoPE.apply_rotary_emb`` applies them.
    ``head_dim`` must be divisible by 4.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        theta: float = 10000.0

    def __init__(self, config: Config):
        super().__init__()
        if config.head_dim % 4 != 0:
            raise ValueError(
                f"2D RoPE requires head_dim divisible by 4, got {config.head_dim}."
            )
        self.head_dim = config.head_dim
        self.theta = config.theta
        self.register_buffer("inv_freq", self._compute_inv_freq(), persistent=False)

    def _compute_inv_freq(self, *, device: torch.device | None = None) -> torch.Tensor:
        # inv_freq[k] = theta**(-4k/head_dim) for k in [0, head_dim/4); the
        # step of 4 leaves room for the row/col split of the 2D rotation.
        return 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.head_dim, 4, dtype=torch.float32, device=device)
                / self.head_dim
            )
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Re-compute inv_freq on the target device after to_empty()."""
        device = buffer_device or self.inv_freq.device
        self.inv_freq = self._compute_inv_freq(device=device)

    def forward(self, seqlen: int) -> torch.Tensor:
        """Frequency table ``(seqlen, head_dim/4)`` for positions ``[0, seqlen)``."""
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        seq = cast(torch.Tensor, _maybe_wrap_positions(seq, self.inv_freq))
        if get_spmd_backend() == "spmd_types" and spmd.is_type_checking():
            seq = spmd.mutate_type(seq, "tp", src=spmd.R, dst=spmd.I)
        return torch.outer(seq, self.inv_freq)


class VisionMLP(Module):
    """Feed-forward network with GELU activation (fc1 -> act -> fc2)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        fc1: Linear.Config
        fc2: Linear.Config
        act_fn: GELU.Config = field(
            default_factory=lambda: GELU.Config(approximate="tanh")
        )

    def __init__(self, config: Config):
        super().__init__()
        self.linear_fc1 = config.fc1.build()
        self.linear_fc2 = config.fc2.build()
        self.act_fn = config.act_fn.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))


class VisionAttention(Module):
    """Multi-head self-attention with FlexAttention over a padded batch.

    Separate q/k/v projections (clean per-head ColwiseParallel under TP). RoPE is
    applied via the injected ``rope_apply`` callable so this class is reused
    across models with different rotary formulations.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        proj: Linear.Config
        inner_attention: Module.Config = field(default_factory=FlexAttention.Config)

    def __init__(self, config: Config):
        super().__init__()
        if config.dim % config.num_heads != 0:
            raise ValueError(
                f"VisionAttention dim ({config.dim}) must be divisible by "
                f"num_heads ({config.num_heads})."
            )
        self.head_dim = config.dim // config.num_heads

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.proj = config.proj.build()
        self.flex_attention = config.inner_attention.build()

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply: RopeApply,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        N, P, _ = x.shape

        # -1 infers the head count locally (= num_heads / TP under tensor
        # parallelism, where wq/wk/wv are colwise-sharded).
        q_NPHDh = local_head_split(self.wq(x), self.head_dim)
        k_NPHDh = local_head_split(self.wk(x), self.head_dim)
        v_NPHDh = local_head_split(self.wv(x), self.head_dim)

        q_NPHDh, k_NPHDh = rope_apply(q_NPHDh, k_NPHDh, rope_cache)

        out_NPHDh = self.flex_attention(
            q_NPHDh, k_NPHDh, v_NPHDh, attention_masks=attention_mask
        )
        out_NPD = out_NPHDh.reshape(N, P, -1)
        return self.proj(out_NPD)


class VisionTransformerBlock(Module):
    """Pre-norm transformer block: norm -> attn -> residual -> norm -> mlp."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm1: LayerNorm.Config | RMSNorm.Config
        norm2: LayerNorm.Config | RMSNorm.Config
        attn: VisionAttention.Config
        mlp: VisionMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = config.norm1.build()
        self.norm2 = config.norm2.build()
        self.attn = config.attn.build()
        self.mlp = config.mlp.build()

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply: RopeApply,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_cache=rope_cache,
            rope_apply=rope_apply,
            attention_mask=attention_mask,
        )
        x = x + self.mlp(self.norm2(x))
        return x
