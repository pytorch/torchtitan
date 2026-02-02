# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3-VL Vision Encoder implementation.

This module implements the Vision Transformer (ViT) encoder used in Qwen3-VL,
using FlexAttention with padded batches for efficient processing.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torchtitan.models.attention import FlexAttentionWrapper

from .args import Qwen3VLVisionEncoderArgs


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision_batched(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors.

    Args:
        q: (batch, seq, heads, head_dim)
        k: (batch, seq, heads, head_dim)
        cos: (batch, seq, head_dim)
        sin: (batch, seq, head_dim)
    """
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    # Expand cos/sin for heads dimension
    cos = cos.unsqueeze(2).float()  # (batch, seq, 1, head_dim)
    sin = sin.unsqueeze(2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class VisionRotaryEmbedding(nn.Module):
    """2D Rotary Position Embedding for Vision Transformer."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """Compute rotary embeddings for a sequence."""
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchEmbed(nn.Module):
    """3D Patch Embedding for images and videos."""

    def __init__(
        self,
        patch_size: int = 14,
        temporal_patch_size: int = 2,
        in_channels: int = 3,
        embed_dim: int = 1280,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        kernel_size = [temporal_patch_size, patch_size, patch_size]
        self.proj = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, patch_dim)
        Returns:
            (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = hidden_states.shape
        target_dtype = self.proj.weight.dtype
        # Reshape each patch to 3D
        hidden_states = hidden_states.view(
            batch_size * seq_len, self.in_channels, self.temporal_patch_size,
            self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype))
        hidden_states = hidden_states.view(batch_size, seq_len, self.embed_dim)
        return hidden_states

    def init_weights(self):
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)


class PatchMerger(nn.Module):
    """Merge spatial patches to reduce sequence length."""

    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int = 2,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size * (spatial_merge_size ** 2)

        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_size) where seq_len is divisible by spatial_merge_size^2
        Returns:
            (batch, seq_len // spatial_merge_size^2, out_hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        x = self.norm(x)
        # Reshape to merge spatial patches
        x = x.view(batch_size, seq_len // (self.spatial_merge_size ** 2), self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x

    def init_weights(self):
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.linear_fc1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.linear_fc2.weight, mean=0.0, std=0.02)
        if self.linear_fc1.bias is not None:
            nn.init.zeros_(self.linear_fc1.bias)
        if self.linear_fc2.bias is not None:
            nn.init.zeros_(self.linear_fc2.bias)


class VisionAttention(nn.Module):
    """Multi-head attention with FlexAttention for efficient batched processing."""

    def __init__(self, args: Qwen3VLVisionEncoderArgs):
        super().__init__()
        self.dim = args.dim
        self.num_heads = args.n_heads
        self.head_dim = self.dim // self.num_heads

        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)
        self.flex_attention = FlexAttentionWrapper()

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, dim)
            position_embeddings: (cos, sin) each of shape (batch, seq_len, head_dim)
            attention_mask: BlockMask for attention

        Returns:
            (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.qkv(hidden_states).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(0)  # Each: (batch, seq, heads, head_dim)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision_batched(q, k, cos, sin)

        # Reshape for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # FlexAttention
        attn_output = self.flex_attention(q, k, v, block_mask=attention_mask)

        # Reshape back: (batch, seq, dim)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.proj(attn_output)

    def init_weights(self):
        nn.init.trunc_normal_(self.qkv.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)


class VisionMLP(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, args: Qwen3VLVisionEncoderArgs):
        super().__init__()
        self.linear_fc1 = nn.Linear(args.dim, args.ffn_dim, bias=True)
        self.linear_fc2 = nn.Linear(args.ffn_dim, args.dim, bias=True)
        self.act_fn = nn.GELU()

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))

    def init_weights(self):
        nn.init.trunc_normal_(self.linear_fc1.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.linear_fc2.weight, mean=0.0, std=0.02)
        if self.linear_fc1.bias is not None:
            nn.init.zeros_(self.linear_fc1.bias)
        if self.linear_fc2.bias is not None:
            nn.init.zeros_(self.linear_fc2.bias)


class VisionTransformerBlock(nn.Module):
    """Single transformer block for vision encoder."""

    def __init__(self, args: Qwen3VLVisionEncoderArgs):
        super().__init__()
        self.norm1 = nn.LayerNorm(args.dim, eps=args.layer_norm_eps)
        self.norm2 = nn.LayerNorm(args.dim, eps=args.layer_norm_eps)
        self.attn = VisionAttention(args)
        self.mlp = VisionMLP(args)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def init_weights(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.init_weights()
        self.mlp.init_weights()


# Compiled block mask creation
_compiled_create_block_mask = torch.compile(create_block_mask)


def get_vision_block_mask_mod(seq_lens: torch.Tensor, max_seq_len: int):
    """Create a mask modifier for block-diagonal attention.

    Each image only attends to its own tokens.

    Args:
        seq_lens: (num_images,) actual sequence length per image
        max_seq_len: Maximum sequence length (padded length)
    """
    # Precompute cumulative positions for efficient lookup
    cum_lens = torch.zeros(len(seq_lens) + 1, dtype=torch.long, device=seq_lens.device)
    cum_lens[1:] = seq_lens.cumsum(0)

    def mask_mod(b, h, q_idx, kv_idx):
        # Check if both indices are within the valid range for this image
        valid_q = q_idx < seq_lens[b]
        valid_kv = kv_idx < seq_lens[b]
        return valid_q & valid_kv

    return mask_mod


class Qwen3VLVisionEncoder(nn.Module):
    """
    Qwen3-VL Vision Encoder with FlexAttention.

    Uses padded batches (N, L, D) format for efficient processing.
    """

    def __init__(self, args: Qwen3VLVisionEncoderArgs):
        super().__init__()
        self.args = args
        self.spatial_merge_size = args.spatial_merge_size
        self.patch_size = args.patch_size
        self.spatial_merge_unit = args.spatial_merge_size ** 2

        # Patch embedding
        self.patch_embed = PatchEmbed(
            patch_size=args.patch_size,
            temporal_patch_size=args.temporal_patch_size,
            in_channels=args.in_channels,
            embed_dim=args.dim,
        )

        # Position embeddings (learnable, with bilinear interpolation)
        self.num_position_embeddings = args.num_position_embeddings
        self.pos_embed = nn.Embedding(args.num_position_embeddings, args.dim)
        self.num_grid_per_side = int(args.num_position_embeddings ** 0.5)

        # Rotary embeddings for 2D positions
        head_dim = args.dim // args.n_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2, theta=args.rope_theta)

        # Transformer layers
        self.layers = nn.ModuleDict(
            {str(idx): VisionTransformerBlock(args) for idx in range(args.n_layers)}
        )

        # Main patch merger
        self.merger = PatchMerger(
            hidden_size=args.dim,
            out_hidden_size=args.out_hidden_size,
            spatial_merge_size=args.spatial_merge_size,
        )

        # DeepStack mergers for intermediate layers
        self.deepstack_visual_indexes = args.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList([
            PatchMerger(
                hidden_size=args.dim,
                out_hidden_size=args.out_hidden_size,
                spatial_merge_size=args.spatial_merge_size,
            )
            for _ in range(len(args.deepstack_visual_indexes))
        ])

    def compute_position_embeddings(
        self,
        grid_thw: torch.Tensor,
        max_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute position embeddings for padded batch.

        Args:
            grid_thw: (num_images, 3) with [t, h, w] per image
            max_seq_len: Maximum sequence length (for padding)

        Returns:
            pos_embeds: (num_images, max_seq_len, dim) learnable position embeddings
            rope_cos: (num_images, max_seq_len, head_dim) RoPE cosines
            rope_sin: (num_images, max_seq_len, head_dim) RoPE sines
        """
        num_images = grid_thw.shape[0]
        device = grid_thw.device
        merge_size = self.spatial_merge_size
        head_dim = self.args.dim // self.args.n_heads

        # Compute max height/width for RoPE freq table
        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)

        # Pre-allocate output tensors
        pos_embeds = torch.zeros(num_images, max_seq_len, self.args.dim, device=device)
        rope_embeds = torch.zeros(num_images, max_seq_len, head_dim // 2, device=device)

        for i in range(num_images):
            t, h, w = int(grid_thw[i, 0].item()), int(grid_thw[i, 1].item()), int(grid_thw[i, 2].item())
            seq_len = t * h * w

            # Compute bilinear interpolated position embeddings
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)

            h_floor, w_floor = h_idxs.int(), w_idxs.int()
            h_ceil = (h_floor + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_floor + 1).clip(max=self.num_grid_per_side - 1)

            dh, dw = h_idxs - h_floor, w_idxs - w_floor

            # Compute indices for 4 corners
            base_h, base_h_ceil = h_floor * self.num_grid_per_side, h_ceil * self.num_grid_per_side
            idx_00 = (base_h[:, None] + w_floor[None, :]).flatten().long()
            idx_01 = (base_h[:, None] + w_ceil[None, :]).flatten().long()
            idx_10 = (base_h_ceil[:, None] + w_floor[None, :]).flatten().long()
            idx_11 = (base_h_ceil[:, None] + w_ceil[None, :]).flatten().long()

            # Bilinear weights
            w_00 = ((1 - dh)[:, None] * (1 - dw)[None, :]).flatten()
            w_01 = ((1 - dh)[:, None] * dw[None, :]).flatten()
            w_10 = (dh[:, None] * (1 - dw)[None, :]).flatten()
            w_11 = (dh[:, None] * dw[None, :]).flatten()

            # Interpolated position embeddings
            pos_hw = (
                self.pos_embed(idx_00) * w_00[:, None] +
                self.pos_embed(idx_01) * w_01[:, None] +
                self.pos_embed(idx_10) * w_10[:, None] +
                self.pos_embed(idx_11) * w_11[:, None]
            )  # (h*w, dim)

            # Repeat for temporal and permute to block order
            if t > 1:
                pos_hw = pos_hw.repeat(t, 1)
            pos_hw = (
                pos_hw.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            pos_embeds[i, :seq_len] = pos_hw

            # Compute RoPE position ids in block order
            merged_h, merged_w = h // merge_size, w // merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)
            if t > 1:
                coords = coords.repeat(t, 1)

            rope_2d = freq_table[coords]  # (seq_len, 2, head_dim//4)
            rope_embeds[i, :seq_len] = rope_2d.flatten(1)  # (seq_len, head_dim//2)

        # Compute cos/sin from RoPE embeddings
        rope_embeds = torch.cat((rope_embeds, rope_embeds), dim=-1)  # (N, L, head_dim)
        rope_cos = rope_embeds.cos()
        rope_sin = rope_embeds.sin()

        return pos_embeds, rope_cos, rope_sin

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the vision encoder.

        Args:
            pixel_values: Padded patches (num_images, max_seq_len, patch_dim)
            grid_thw: Grid dimensions (num_images, 3) for [temporal, height, width]

        Returns:
            merged_hidden_states: (num_images, max_merged_seq_len, out_hidden_size)
            deepstack_features: List of features from intermediate layers
        """
        num_images, max_seq_len, _ = pixel_values.shape
        device = pixel_values.device

        # Compute actual sequence lengths per image
        seq_lens = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).to(torch.long)

        # Patch embedding
        hidden_states = self.patch_embed(pixel_values)

        # Compute position embeddings
        pos_embeds, rope_cos, rope_sin = self.compute_position_embeddings(grid_thw, max_seq_len)
        hidden_states = hidden_states + pos_embeds

        # Create attention mask for block-diagonal attention
        mask_mod = get_vision_block_mask_mod(seq_lens, max_seq_len)
        attention_mask = _compiled_create_block_mask(
            mask_mod, num_images, None, max_seq_len, max_seq_len
        )

        # Apply transformer layers with DeepStack extraction
        position_embeddings = (rope_cos, rope_sin)
        deepstack_feature_lists = []

        for layer_num, blk in self.layers.items():
            hidden_states = blk(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
            if int(layer_num) in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(int(layer_num))
                deepstack_feature = self.deepstack_merger_list[idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)

        # Final merge
        merged_hidden_states = self.merger(hidden_states)

        return merged_hidden_states, deepstack_feature_lists

    def init_weights(self):
        self.patch_embed.init_weights()
        nn.init.trunc_normal_(self.pos_embed.weight, mean=0.0, std=0.02)
        self.merger.init_weights()
        for merger in self.deepstack_merger_list:
            merger.init_weights()
        for block in self.layers.values():
            block.init_weights()
