# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""MoonViT3d Vision Encoder for Kimi K2.5.

Architecture (faithful to the SGLang Kimi-K2.5 reference, restructured into the
torchtitan config-driven ``Module`` style used by qwen3_5/qwen3_vl):

- Linear patch embedding (the collator pre-extracts and flattens patches).
- Learnable 2D spatial position embeddings (bicubic-interpolated to each image's
  resolution) summed with fixed 1D sinusoidal *temporal* embeddings for video.
- 2D rotary position embedding applied in every attention layer. The reference
  uses a complex-number formulation; here it is expressed with the numerically
  identical real cos/sin arithmetic so it is DTensor-friendly under TP.
- Pre-norm transformer blocks (LayerNorm + biased attention + GELU-tanh MLP).
- Temporal mean pooling + 2x2 spatial patch merging, then a 2-layer MLP
  projector to the language-model hidden size.

Patches flow as a padded batch ``(num_vision, max_num_patch, *)`` with a
block-diagonal FlexAttention mask (each visual item attends only to its own
patches) — the established torchtitan VLM layout, numerically equivalent to the
reference's variable-length (``cu_seqlens``) attention.

Shape suffix legend:
    N = num_vision items, P = max patches per item (padded), D = vision dim,
    H = num heads, Dh = head dim, M = merged tokens per item, Dt = text dim.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torchtitan.models.common import Linear
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.nn_modules import GELU, LayerNorm
from torchtitan.protocols.module import Module, ModuleDict

_compiled_create_block_mask = torch.compile(create_block_mask)


def get_vision_block_mask_mod(num_patch: torch.Tensor) -> Callable:
    """Block-diagonal mask: each visual item attends only to its own patches.

    Args:
        num_patch: (num_vision,) actual number of patches per visual item.
    """

    def mask_mod(b, h, q_idx, kv_idx):
        valid_q = q_idx < num_patch[b]
        valid_kv = kv_idx < num_patch[b]
        return valid_q & valid_kv

    return mask_mod


def _get_1d_sincos_pos_embed(embed_dim: int, num_positions: int) -> np.ndarray:
    """Standard 1D sinusoidal position embeddings, ``(num_positions, embed_dim)``.

    Used for the temporal axis (video frames). Matches the reference's
    ``get_1d_sincos_pos_embed`` (sin/cos concatenated, base 10000).
    """
    grid = np.arange(num_positions, dtype=np.float32)
    omega = np.arange(embed_dim // 2, dtype=np.float32) / (embed_dim / 2.0)
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", grid, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def apply_rope_2d(
    xq_NPHd: torch.Tensor,
    xk_NPHd: torch.Tensor,
    rope_cos_NPd: torch.Tensor,
    rope_sin_NPd: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply 2D rotary embedding to query and key (real cos/sin formulation).

    Equivalent to the reference's complex ``view_as_complex(x) * freqs_cis``:
    consecutive element pairs ``(x[2i], x[2i+1])`` are rotated by angle
    ``phi_i``. Using real arithmetic keeps this DTensor-safe (no
    ``view_as_complex`` on a sharded tensor).

    Args:
        xq_NPHd, xk_NPHd: (N, P, H, Dh) query/key.
        rope_cos_NPd, rope_sin_NPd: (N, P, Dh/2) per-position cos/sin, where
            entry ``i`` corresponds to pair ``(2i, 2i+1)``.

    Returns:
        Rotated query and key, same shape and dtype as inputs.
    """
    # (N, P, 1, Dh/2) so it broadcasts over the head axis.
    cos = rope_cos_NPd.unsqueeze(2)
    sin = rope_sin_NPd.unsqueeze(2)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_pairs = x.float().reshape(*x.shape[:-1], -1, 2)
        a = x_pairs[..., 0]
        b = x_pairs[..., 1]
        out_even = a * cos - b * sin
        out_odd = a * sin + b * cos
        out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
        return out.type_as(x)

    return rotate(xq_NPHd), rotate(xk_NPHd)


def _compute_2d_rope_cache(
    grid_thw: torch.Tensor,
    max_num_patch: int,
    head_dim: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute padded 2D-RoPE cos/sin caches in raster patch order.

    Mirrors the reference ``Rope2DPosEmbRepeated``: for head-dim pair index
    ``k`` (``k`` in ``[0, head_dim/4)``), even output pairs are rotated by the
    *column* (x) position and odd pairs by the *row* (y) position, both at
    frequency ``theta**(-4k/head_dim)``. Frames repeat the spatial pattern.

    Args:
        grid_thw: (N, 3) patch counts ``[t, h, w]`` per item.
        max_num_patch: Padded sequence length.
        head_dim: Attention head dim (must be divisible by 4).
        theta: RoPE base.

    Returns:
        (cos, sin), each ``(N, max_num_patch, head_dim/2)`` float32.
    """
    assert head_dim % 4 == 0, "2D RoPE requires head_dim divisible by 4"
    num_vision = grid_thw.shape[0]
    device = grid_thw.device

    freqs = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 4, device=device, dtype=torch.float32) / head_dim)
    )  # (head_dim/4,)

    angles = torch.zeros(
        num_vision, max_num_patch, head_dim // 2, device=device, dtype=torch.float32
    )

    # Group by (h, w) so the per-resolution angle grid is built once.
    hw_to_indices: dict[tuple[int, int], list[int]] = {}
    for i in range(num_vision):
        key = (int(grid_thw[i, 1].item()), int(grid_thw[i, 2].item()))
        hw_to_indices.setdefault(key, []).append(i)

    for (h, w), indices in hw_to_indices.items():
        # Raster order: position p -> (row = p // w, col = p % w).
        flat = torch.arange(h * w, device=device, dtype=torch.float32)
        x_pos = (flat % w).unsqueeze(-1)  # column
        y_pos = (flat // w).unsqueeze(-1)  # row
        x_ang = x_pos * freqs  # (h*w, head_dim/4)
        y_ang = y_pos * freqs  # (h*w, head_dim/4)
        # Interleave x/y so pair 2k uses x-position, pair 2k+1 uses y-position.
        ang = torch.stack([x_ang, y_ang], dim=-1).reshape(h * w, head_dim // 2)
        for i in indices:
            t = int(grid_thw[i, 0].item())
            seq_len = t * h * w
            angles[i, :seq_len] = ang.repeat(t, 1)

    return angles.cos(), angles.sin()


def tpool_patch_merger(
    hidden_NPD: torch.Tensor,
    grid_thw: torch.Tensor,
    merge_kernel_size: tuple[int, int],
    max_merged: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Temporal mean pooling + spatial merge over the padded batch.

    For each item ``(t, h, w)``: reshape its valid patches to
    ``(t, h, w, D)``, mean over the temporal axis, then group spatial
    ``kh x kw`` neighbors and concatenate them along the feature axis, yielding
    ``(h/kh * w/kw)`` merged tokens of dim ``kh*kw*D``.

    Args:
        hidden_NPD: (N, P, D) padded patch features.
        grid_thw: (N, 3) patch counts ``[t, h, w]`` per item.
        merge_kernel_size: ``(kh, kw)`` spatial merge factors.
        max_merged: Padded number of merged tokens per item.

    Returns:
        merged: ``(N, max_merged, kh*kw*D)`` padded merged tokens.
        num_merged: ``(N,)`` valid merged-token count per item.
    """
    num_vision, _, d_model = hidden_NPD.shape
    kh, kw = merge_kernel_size
    merged_dim = kh * kw * d_model
    device = hidden_NPD.device

    merged = hidden_NPD.new_zeros(num_vision, max_merged, merged_dim)
    num_merged = torch.zeros(num_vision, dtype=torch.long, device=device)

    for i in range(num_vision):
        t = int(grid_thw[i, 0].item())
        h = int(grid_thw[i, 1].item())
        w = int(grid_thw[i, 2].item())
        seq = hidden_NPD[i, : t * h * w]
        new_h, new_w = h // kh, w // kw
        # (t, new_h, kh, new_w, kw, D) -> mean over t -> group spatial kernel.
        seq = seq.view(t, new_h, kh, new_w, kw, d_model)
        seq = seq.permute(0, 1, 3, 2, 4, 5).contiguous().mean(dim=0)
        seq = seq.reshape(new_h * new_w, merged_dim)
        merged[i, : new_h * new_w] = seq
        num_merged[i] = new_h * new_w

    return merged, num_merged


def block_to_raster_patches(
    pixel_values: torch.Tensor,
    grid_thw: torch.Tensor,
    merge_size: int,
) -> torch.Tensor:
    """Reorder padded patches from block order to raster order, per item.

    The shared multimodal collator emits patches in *block* order — each
    ``merge_size x merge_size`` spatial group is contiguous, sequence layout
    ``(t, bh, bw, m, n)`` (to match Qwen's merger). MoonViT3d is faithful to the
    SGLang reference, which works in *raster* order ``(t, h, w)``. This converts
    the former to the latter so the shared collator can drive the raster encoder
    without changing core data code or the (verified) raster RoPE / pos-embed.

    Args:
        pixel_values: ``(N, P, patch_dim)`` padded patches (block order).
        grid_thw: ``(N, 3)`` patch counts ``[t, h, w]`` per item.
        merge_size: Spatial merge factor (``= merge_kernel_size``).

    Returns:
        ``(N, P, patch_dim)`` padded patches in raster order.
    """
    out = torch.empty_like(pixel_values)
    m = merge_size
    for i in range(grid_thw.shape[0]):
        t = int(grid_thw[i, 0].item())
        h = int(grid_thw[i, 1].item())
        w = int(grid_thw[i, 2].item())
        n = t * h * w
        bh, bw = h // m, w // m
        seq = pixel_values[i, :n]
        # (t, bh, bw, m, n) -> (t, bh, m, bw, n) -> (t, h, w) raster.
        seq = (
            seq.view(t, bh, bw, m, m, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(n, -1)
        )
        out[i, :n] = seq
        out[i, n:] = pixel_values[i, n:]
    return out


class Learnable2DInterpPosEmb(Module):
    """Learnable 2D spatial position embeddings + fixed sinusoidal temporal.

    The spatial table is a learnable ``(height, width, dim)`` parameter,
    bicubic-interpolated to each image's patch grid. For video (``t > 1``) the
    spatial embedding is repeated per frame and summed with a fixed 1D
    sinusoidal temporal embedding. Matches the reference
    ``Learnable2DInterpPosEmbDivided_fixed``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        height: int
        width: int
        num_frames: int
        dim: int
        interpolation_mode: str = "bicubic"

    def __init__(self, config: Config):
        super().__init__()
        self.height = config.height
        self.width = config.width
        self.num_frames = config.num_frames
        self.dim = config.dim
        self.interpolation_mode = config.interpolation_mode

        self.weight = nn.Parameter(torch.empty(config.height, config.width, config.dim))
        # Non-persistent: recomputed deterministically in ``_init_self_buffers``.
        self.register_buffer("time_weight", None, persistent=False)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device or self.weight.device
        time = _get_1d_sincos_pos_embed(self.dim, self.num_frames)
        self.time_weight = (
            torch.from_numpy(time).to(device=device, dtype=torch.float32).unsqueeze(1)
        )  # (num_frames, 1, dim)

    def forward(self, x_NPD: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Add interpolated 2D (+ temporal) position embeddings.

        Args:
            x_NPD: (N, P, dim) padded patch embeddings.
            grid_thw: (N, 3) patch counts ``[t, h, w]`` per item.
        """
        num_vision, max_num_patch, _ = x_NPD.shape
        pos = x_NPD.new_zeros(num_vision, max_num_patch, self.dim)

        # (dim, height, width) for F.interpolate; .float() for bicubic.
        grid_table = self.weight.permute(2, 0, 1).unsqueeze(0).float()

        hw_to_indices: dict[tuple[int, int], list[int]] = {}
        for i in range(num_vision):
            key = (int(grid_thw[i, 1].item()), int(grid_thw[i, 2].item()))
            hw_to_indices.setdefault(key, []).append(i)

        for (h, w), indices in hw_to_indices.items():
            if (h, w) == (self.height, self.width):
                pos_hw = self.weight.flatten(end_dim=1)
            else:
                pos_hw = (
                    F.interpolate(
                        grid_table,
                        size=(h, w),
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute(1, 2, 0)
                    .reshape(h * w, self.dim)
                    .to(self.weight.dtype)
                )
            for i in indices:
                t = int(grid_thw[i, 0].item())
                seq_len = t * h * w
                if t == 1:
                    pos[i, :seq_len] = pos_hw
                else:
                    frames = pos_hw.unsqueeze(0).repeat(t, 1, 1)
                    frames = frames + self.time_weight[:t].to(frames.dtype)
                    pos[i, :seq_len] = frames.reshape(seq_len, self.dim)

        return x_NPD + pos


class VisionRotaryEmbedding2D(Module):
    """2D rotary position embedding for the vision tower.

    Holds only the RoPE hyper-parameters; the per-batch cos/sin caches are built
    on the fly from the grid (no large precomputed buffer). ``head_dim`` must be
    divisible by 4.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        theta: float = 10000.0

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.theta = config.theta

    def forward(
        self, grid_thw: torch.Tensor, max_num_patch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _compute_2d_rope_cache(
            grid_thw, max_num_patch, self.head_dim, self.theta
        )


class VisionAttention(Module):
    """Multi-head self-attention with 2D RoPE and FlexAttention.

    Uses separate q/k/v projections (not fused QKV): a fused projection would
    produce ambiguous DTensor ``view`` on the TP-sharded output dim. Separate
    projections give clean per-head ColwiseParallel sharding.
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
        self.dim = config.dim
        self.num_heads = config.num_heads
        self.head_dim = config.dim // config.num_heads

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.proj = config.proj.build()
        self.flex_attention = config.inner_attention.build()

    def forward(
        self,
        x_NPD: torch.Tensor,
        *,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        num_vision, max_num_patch, _ = x_NPD.shape
        q = self.wq(x_NPD).view(num_vision, max_num_patch, -1, self.head_dim)
        k = self.wk(x_NPD).view(num_vision, max_num_patch, -1, self.head_dim)
        v = self.wv(x_NPD).view(num_vision, max_num_patch, -1, self.head_dim)

        q, k = apply_rope_2d(q, k, rope_cos, rope_sin)

        out = self.flex_attention(q, k, v, attention_masks=attention_mask)
        out = out.reshape(num_vision, max_num_patch, -1)
        return self.proj(out)


class VisionMLP(Module):
    """Two-layer feed-forward with GELU-tanh activation."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        fc1: Linear.Config
        fc2: Linear.Config
        act_fn: GELU.Config = field(
            default_factory=lambda: GELU.Config(approximate="tanh")
        )

    def __init__(self, config: Config):
        super().__init__()
        self.fc1 = config.fc1.build()
        self.fc2 = config.fc2.build()
        self.act_fn = config.act_fn.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


class VisionEncoderBlock(Module):
    """Pre-norm transformer block: norm -> attn -> residual -> norm -> mlp."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm0: LayerNorm.Config
        norm1: LayerNorm.Config
        attn: VisionAttention.Config
        mlp: VisionMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm0 = config.norm0.build()
        self.norm1 = config.norm1.build()
        self.attn = config.attn.build()
        self.mlp = config.mlp.build()

    def forward(
        self,
        x_NPD: torch.Tensor,
        *,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        x_NPD = x_NPD + self.attn(
            self.norm0(x_NPD),
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attention_mask=attention_mask,
        )
        x_NPD = x_NPD + self.mlp(self.norm1(x_NPD))
        return x_NPD


class MultiModalProjector(Module):
    """Project merged vision features to the language-model hidden size.

    Applies a per-patch LayerNorm (on ``vt_hidden_size``), then a 2-layer MLP
    over the concatenated spatial-merge features. Matches the reference
    ``K2VLMultiModalProjector``.
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
        x = merged_NMK.view(n, m, -1, self.pre_norm.normalized_shape[0])
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

        # Order of the incoming flattened patches. "raster" matches the SGLang
        # reference (direct feed); "block" matches the shared MM collator
        # (reordered to raster on entry). See ``block_to_raster_patches``.
        input_patch_order: str = "raster"

        # Sub-module configs (built in __init__).
        patch_embed_proj: Linear.Config
        pos_emb: Learnable2DInterpPosEmb.Config
        rotary_pos_emb: VisionRotaryEmbedding2D.Config
        block: VisionEncoderBlock.Config
        final_norm: LayerNorm.Config
        projector: MultiModalProjector.Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.merge_kernel_size = tuple(config.merge_kernel_size)
        self.spatial_merge_unit = self.merge_kernel_size[0] * self.merge_kernel_size[1]
        self.input_patch_order = config.input_patch_order

        self.patch_embed = config.patch_embed_proj.build()
        self.pos_emb = config.pos_emb.build()
        self.rotary_pos_emb = config.rotary_pos_emb.build()
        self.layers = ModuleDict(
            {str(idx): config.block.build() for idx in range(config.num_layers)}
        )
        self.final_norm = config.final_norm.build()
        self.projector = config.projector.build()

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
        """
        if self.input_patch_order == "block":
            pixel_values = block_to_raster_patches(
                pixel_values, grid_thw, self.merge_kernel_size[0]
            )

        num_vision, max_num_patch, _ = pixel_values.shape
        num_patch = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).to(torch.long)

        x = self.patch_embed(pixel_values)
        x = self.pos_emb(x, grid_thw)

        rope_cos, rope_sin = self.rotary_pos_emb(grid_thw, max_num_patch)

        mask_mod = get_vision_block_mask_mod(num_patch)
        attention_mask = _compiled_create_block_mask(
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
                rope_cos=rope_cos,
                rope_sin=rope_sin,
                attention_mask=attention_mask,
            )

        x = self.final_norm(x)

        # Temporal pool + spatial merge, then project to the LLM hidden size.
        kh, kw = self.merge_kernel_size
        max_merged = int(((grid_thw[:, 1] // kh) * (grid_thw[:, 2] // kw)).max().item())
        merged, _ = tpool_patch_merger(
            x, grid_thw, self.merge_kernel_size, max_merged
        )
        return self.projector(merged)
