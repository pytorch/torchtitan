# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Qwen3-VL Vision Encoder implementation.

This module implements the Vision Transformer (ViT) encoder used in Qwen3-VL,
which features 2D RoPE for spatial positional encoding with bilinear-interpolated
position embeddings for images and videos.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.varlen import varlen_attn

from .args import Qwen3VLVisionEncoderArgs

# Compiled varlen_attn for better performance
_compiled_varlen_attn = torch.compile(varlen_attn, mode="max-autotune-no-cudagraphs")


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary positional embeddings to query and key tensors for vision."""
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class VisionRotaryEmbedding(nn.Module):
    """2D Rotary Position Embedding for Vision Transformer.

    Handles spatial (height, width) positional encoding for images.
    """

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        """Compute rotary embeddings for a sequence.

        Args:
            seqlen: Maximum sequence length (max_hw)

        Returns:
            freqs: Tensor of shape (seqlen, dim // 2)
        """
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

        # 3D convolution for patch embedding (with bias as per HF implementation)
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
            hidden_states: Flattened patches (seq_len, patch_dim) where
                patch_dim = in_channels * temporal_patch_size * patch_size * patch_size

        Returns:
            (seq_len, embed_dim)
        """
        target_dtype = self.proj.weight.dtype
        # Reshape to (batch, channels, temporal, height, width)
        hidden_states = hidden_states.view(
            -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size
        )
        hidden_states = self.proj(hidden_states.to(dtype=target_dtype)).view(-1, self.embed_dim)
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
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.hidden_size = hidden_size * (spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm

        self.norm = nn.LayerNorm(
            self.hidden_size if use_postshuffle_norm else hidden_size,
            eps=1e-6,
        )
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge patches and project to output dimension.

        Args:
            x: (seq_len, hidden_size)

        Returns:
            (seq_len // (spatial_merge_size^2), out_hidden_size)
        """
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size)).view(-1, self.hidden_size)
        else:
            x = self.norm(x).view(-1, self.hidden_size)
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
    """Multi-head attention with 2D RoPE for vision."""

    def __init__(self, args: Qwen3VLVisionEncoderArgs):
        super().__init__()
        self.dim = args.dim
        self.num_heads = args.n_heads
        self.head_dim = self.dim // self.num_heads

        # Single QKV projection (as per HF implementation)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: (seq_len, dim)
            cu_seqlens: Cumulative sequence lengths for variable length attention
            max_seqlen: Maximum sequence length in the batch
            position_embeddings: (cos, sin) tensors for rotary embeddings

        Returns:
            (seq_len, dim)
        """
        seq_length = hidden_states.shape[0]

        # Single QKV projection
        qkv = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)
        query_states, key_states, value_states = qkv.permute(1, 0, 2, 3).unbind(0)

        # Apply rotary embeddings
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

        # Use varlen_attn for efficient variable-length attention
        # Input shape: (seq_len, n_heads, head_dim)
        # varlen_attn handles block-diagonal attention internally via cu_seqlens
        attn_output = varlen_attn(
            query_states,
            key_states,
            value_states,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            is_causal=False,  # Full bidirectional attention within each image
        )

        attn_output = attn_output.reshape(seq_length, -1).contiguous()
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
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states

    def init_weights(self):
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()
        self.attn.init_weights()
        self.mlp.init_weights()


class Qwen3VLVisionEncoder(nn.Module):
    """
    Qwen3-VL Vision Encoder.

    A Vision Transformer with 2D RoPE and bilinear-interpolated position embeddings
    for encoding images and videos. Supports DeepStack feature extraction.
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
            use_postshuffle_norm=False,
        )

        # DeepStack mergers for intermediate layers
        self.deepstack_visual_indexes = args.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList([
            PatchMerger(
                hidden_size=args.dim,
                out_hidden_size=args.out_hidden_size,
                spatial_merge_size=args.spatial_merge_size,
                use_postshuffle_norm=True,
            )
            for _ in range(len(args.deepstack_visual_indexes))
        ])

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Compute 2D rotary position embeddings based on grid positions.

        Args:
            grid_thw: (num_images, 3) containing [temporal, height, width]

        Returns:
            Position embeddings of shape (total_tokens, head_dim)
        """
        merge_size = self.spatial_merge_size
        device = grid_thw.device

        max_hw = int(grid_thw[:, 1:].max().item())
        freq_table = self.rotary_pos_emb(max_hw)

        total_tokens = int(torch.prod(grid_thw, dim=1).sum().item())
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw:
            num_frames, height, width = int(num_frames.item()), int(height.item()), int(width.item())
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(merge_size, device=device)
            intra_col = torch.arange(merge_size, device=device)

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset:offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # (total_tokens, 2, head_dim//4)
        embeddings = embeddings.flatten(1)  # (total_tokens, head_dim//2)
        return embeddings

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        """Bilinear interpolation of position embeddings.

        Args:
            grid_thw: (num_images, 3) containing [temporal, height, width]

        Returns:
            Position embeddings of shape (total_tokens, dim)
        """
        grid_ts, grid_hs, grid_ws = grid_thw[:, 0], grid_thw[:, 1], grid_thw[:, 2]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in zip(grid_ts, grid_hs, grid_ws):
            h, w = int(h.item()), int(w.item())
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([int(h * w) for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            t, h, w = int(t.item()), int(h.item()), int(w.item())
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)

        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the vision encoder.

        Args:
            pixel_values: Flattened patches (seq_len, patch_dim)
            grid_thw: Grid dimensions (num_images, 3) for [temporal, height, width]

        Returns:
            merged_hidden_states: (seq_len // spatial_merge_unit, out_hidden_size)
            deepstack_features: List of features from intermediate layers
        """
        # Patch embedding
        hidden_states = self.patch_embed(pixel_values)

        # Add interpolated position embeddings
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        # Compute rotary position embeddings
        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len = hidden_states.size(0)
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Compute cu_seqlens for variable length attention
        seq_lens_per_frame = grid_thw[:, 1] * grid_thw[:, 2]
        cu_seqlens = torch.repeat_interleave(
            seq_lens_per_frame, grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)
        max_seqlen = int(seq_lens_per_frame.max().item())

        # Apply transformer layers with DeepStack extraction
        deepstack_feature_lists = []
        for layer_num, blk in self.layers.items():
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                position_embeddings=position_embeddings,
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
