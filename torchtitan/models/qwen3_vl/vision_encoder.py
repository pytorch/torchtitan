# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.experimental import local_map
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torchtitan.models.common import Linear
from torchtitan.models.common.attention import FlexAttention
from torchtitan.models.common.rope import apply_rotary_emb_cos_sin
from torchtitan.protocols.module import Module, ModuleDict, ModuleList

LayerNorm = Module.from_nn_module(nn.LayerNorm)
GELU = Module.from_nn_module(nn.GELU)

_compiled_create_block_mask = torch.compile(create_block_mask)


def get_vision_block_mask_mod(num_patch: torch.Tensor, max_num_patch: int):
    """Create a mask modifier for block-diagonal attention.

    Each image only attends to its own patches.

    Args:
        num_patch: (num_vision,) actual number of patches per visual item
        max_num_patch: Maximum number of patches (padded length)
    """

    def mask_mod(b, h, q_idx, kv_idx):
        valid_q = q_idx < num_patch[b]
        valid_kv = kv_idx < num_patch[b]
        return valid_q & valid_kv

    return mask_mod


def _compute_learned_pos_embeds(
    learned_pos_embed: torch.Tensor,
    grid_thw: torch.Tensor,
    max_num_patch: int,
    num_grid_per_side: int,
    spatial_merge_size: int,
    dim: int,
) -> torch.Tensor:
    """Compute bilinear-interpolated learned position embeddings.

    Reshapes the learnable position embedding table into a 2D grid, interpolates
    it to each image's (h, w) resolution, and permutes to block order matching
    the patch layout from the collator.

    This function operates on plain tensors. When pos_embed is a DTensor,
    callers should wrap this with ``local_map`` to handle DTensor ↔ local
    conversion automatically.

    Args:
        learned_pos_embed: (num_position_embeddings, dim) learnable position embeddings
        grid_thw: (num_vision, 3) with patch counts [t, h, w] per visual item
        max_num_patch: Maximum number of patches (for padding)
        num_grid_per_side: Side length of the square position embedding grid
        spatial_merge_size: Number of patches to merge per spatial dimension
        dim: Hidden dimension

    Returns:
        learned_pos: (num_vision, max_num_patch, dim) interpolated position embeddings
    """
    num_vision = grid_thw.shape[0]
    device = grid_thw.device
    dtype = learned_pos_embed.dtype
    merge_size = spatial_merge_size

    pos_embeds = torch.zeros(num_vision, max_num_patch, dim, device=device, dtype=dtype)

    # Group images by (h, w) to batch compute position embeddings
    hw_to_indices: dict[tuple[int, int], list[int]] = {}
    for i in range(num_vision):
        h, w = int(grid_thw[i, 1].item()), int(grid_thw[i, 2].item())
        key = (h, w)
        if key not in hw_to_indices:
            hw_to_indices[key] = []
        hw_to_indices[key].append(i)

    # Reshape pos_embed to 2D grid for F.interpolate:
    # (num_position_embeddings, dim) → (1, dim, grid_side, grid_side)
    pos_grid = (
        learned_pos_embed.reshape(num_grid_per_side, num_grid_per_side, -1)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )

    for (h, w), indices in hw_to_indices.items():
        pos_hw = F.interpolate(
            pos_grid,
            size=(h, w),
            mode="bilinear",
            align_corners=True,
        )
        # (1, dim, h, w) → (h*w, dim)
        pos_hw = pos_hw.squeeze(0).permute(1, 2, 0).reshape(-1, dim).to(dtype)

        # Permute learned pos_hw from raster order to block order
        # to match the patch sequence produced by image_to_patches
        pos_hw_block = (
            pos_hw.view(h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 2, 1, 3, 4)
            .flatten(0, 3)
        )  # (h*w, dim)

        # Apply to all images with this (h, w)
        # For videos (t > 1), repeat spatial embeddings per frame;
        # temporal position encoding is handled by MRoPE in the LLM
        for i in indices:
            t = int(grid_thw[i, 0].item())
            seq_len = t * h * w
            if t > 1:
                pos_embeds[i, :seq_len] = pos_hw_block.repeat(t, 1)
            else:
                pos_embeds[i, :seq_len] = pos_hw_block

    return pos_embeds


def _compute_2d_rope_cache(
    freq_table: torch.Tensor,
    grid_thw: torch.Tensor,
    max_num_patch: int,
    spatial_merge_size: int,
    head_dim: int,
) -> torch.Tensor:
    """Compute 2D RoPE cache for vision patches.

    Builds row and col indices in block order (matching the patch layout from
    the collator), looks up separate frequency sets for each dimension, and
    concatenates them into a rope_cache for apply_rotary_emb_cos_sin.

    Args:
        freq_table: (max_hw, head_dim//4) precomputed RoPE frequencies
        grid_thw: (num_vision, 3) with patch counts [t, h, w] per visual item
        max_num_patch: Maximum number of patches (for padding)
        spatial_merge_size: Number of patches to merge per spatial dimension
        head_dim: Attention head dimension

    Returns:
        rope_cache: (num_vision, max_num_patch, 1, head_dim*2) float32 for
            apply_rotary_emb_cos_sin
    """
    num_vision = grid_thw.shape[0]
    device = grid_thw.device
    merge_size = spatial_merge_size

    rope_embeds = torch.zeros(
        num_vision, max_num_patch, head_dim // 2, device=device, dtype=torch.float32
    )

    # Group images by (h, w) to batch compute RoPE embeddings
    hw_to_indices: dict[tuple[int, int], list[int]] = {}
    for i in range(num_vision):
        h, w = int(grid_thw[i, 1].item()), int(grid_thw[i, 2].item())
        key = (h, w)
        if key not in hw_to_indices:
            hw_to_indices[key] = []
        hw_to_indices[key].append(i)

    for (h, w), indices in hw_to_indices.items():
        # Compute RoPE position ids in block order (once per unique h, w)
        # A "block" is a merge_size x merge_size group of patches that will
        # be merged into one LLM visual token. RoPE indices must follow
        # block order to match the patch layout from image_to_patches
        merged_h, merged_w = h // merge_size, w // merge_size
        row_base = (
            torch.arange(merged_h, device=device) * merge_size
        )  # starting row of each block
        col_base = (
            torch.arange(merged_w, device=device) * merge_size
        )  # starting col of each block
        intra_row = torch.arange(merge_size, device=device)
        intra_col = torch.arange(merge_size, device=device)

        # Build row and col indices in block order
        # The 4 dimensions represent: [block_row, block_col, intra_row, intra_col]
        # e.g., for merge_size=2 on a 4x4 patch grid (2x2 blocks):
        #   row_idx = [0,0,1,1, 0,0,1,1, 2,2,3,3, 2,2,3,3]
        #   col_idx = [0,1,0,1, 2,3,2,3, 0,1,0,1, 2,3,2,3]
        row_idx = (
            (row_base[:, None, None, None] + intra_row[None, None, :, None])
            .expand(merged_h, merged_w, merge_size, merge_size)
            .reshape(-1)
        )
        col_idx = (
            (col_base[None, :, None, None] + intra_col[None, None, None, :])
            .expand(merged_h, merged_w, merge_size, merge_size)
            .reshape(-1)
        )

        # 2D RoPE: row and col each get separate frequency sets, concatenated
        # (not interleaved). freq_table shape: (max_hw, head_dim//4)
        rope_row = freq_table[row_idx]  # (h*w, head_dim//4)
        rope_col = freq_table[col_idx]  # (h*w, head_dim//4)
        rope_2d = torch.cat([rope_row, rope_col], dim=-1)  # (h*w, head_dim//2)

        # Apply to all images with this (h, w)
        # For videos (t > 1), repeat spatial embeddings per frame;
        # temporal position encoding is handled by MRoPE in the LLM
        for i in indices:
            t = int(grid_thw[i, 0].item())
            seq_len = t * h * w
            if t > 1:
                rope_embeds[i, :seq_len] = rope_2d.repeat(t, 1)
            else:
                rope_embeds[i, :seq_len] = rope_2d

    # Compute cos/sin in model dtype (HF uses .float() here)
    rope_embeds = torch.cat((rope_embeds, rope_embeds), dim=-1)  # (N, L, head_dim)
    rope_cache = torch.cat([rope_embeds.cos(), rope_embeds.sin()], dim=-1).unsqueeze(
        2
    )  # (N, L, 1, head_dim*2)

    return rope_cache


class PatchEmbed(Module):
    """Patch Embedding using Linear projection.

    Since patches are already extracted by the collator, we use Linear instead of Conv3d.
    This is mathematically equivalent when Conv3d kernel_size equals input size:
    - Conv3d: (B, C, T, H, W) with kernel=C*(T,H,W) and dim kernels → (B, dim, 1, 1, 1)
    - Linear: (B, C*T*H*W) → (B, dim)
    Same weighted sum, but Linear uses efficient batched matrix multiplication.
    """

    def __init__(
        self,
        patch_size: int,
        temporal_patch_size: int,
        in_channels: int,
        embed_dim: int,
        *,
        proj: Linear.Config,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        # patch_dim matches the flattened patch from collator: (pt * ph * pw * c)
        self.patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
        self.proj = proj.build()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project patches to embeddings.

        Args:
            hidden_states: (batch, max_num_patch, patch_dim)

        Returns:
            (batch, max_num_patch, embed_dim)
        """
        return self.proj(hidden_states)


class VisionRotaryEmbedding(Module):
    """2D Rotary Position Embedding for Vision Transformer."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Re-compute inv_freq on the target device after to_empty()."""
        device = buffer_device or self.inv_freq.device
        self.inv_freq = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.float, device=device)
                / self.dim
            )
        )

    def forward(self, seqlen: int) -> torch.Tensor:
        """Compute rotary embeddings for a sequence."""
        seq = torch.arange(
            seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(seq, self.inv_freq)
        return freqs


class PatchMerger(Module):
    """Merge spatial patches to reduce sequence length.

    Args:
        hidden_size: Hidden dimension of input features
        out_hidden_size: Output hidden dimension after merging
        spatial_merge_size: Number of patches to merge per spatial dimension
        use_postshuffle_norm: If True, apply LayerNorm after spatial reshape
            (norm dim = hidden_size * spatial_merge_size^2). If False, apply
            before reshape (norm dim = hidden_size). DeepStack mergers use
            postshuffle norm; the main merger uses pre-shuffle norm
    """

    def __init__(
        self,
        hidden_size: int,
        out_hidden_size: int,
        spatial_merge_size: int,
        *,
        fc1: Linear.Config,
        fc2: Linear.Config,
        use_postshuffle_norm: bool = False,
    ):
        super().__init__()
        self.spatial_merge_size = spatial_merge_size
        self.merged_hidden_size = hidden_size * (spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm

        norm_dim = self.merged_hidden_size if use_postshuffle_norm else hidden_size
        self.norm = LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = fc1.build()
        self.act_fn = GELU(approximate="tanh")
        self.linear_fc2 = fc2.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Merge spatial patches and project to output dimension.

        Args:
            x: (batch, seq_len, hidden_size) where seq_len is divisible by spatial_merge_size^2

        Returns:
            (batch, seq_len // spatial_merge_size^2, out_hidden_size)
        """
        batch_size, seq_len, _ = x.shape
        if self.use_postshuffle_norm:
            x = x.view(
                batch_size,
                seq_len // (self.spatial_merge_size**2),
                self.merged_hidden_size,
            )
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.view(
                batch_size,
                seq_len // (self.spatial_merge_size**2),
                self.merged_hidden_size,
            )
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class VisionAttention(Module):
    """Multi-head attention with FlexAttention for efficient batched processing."""

    def __init__(
        self, dim: int, n_heads: int, *, qkv: Linear.Config, proj: Linear.Config
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = n_heads
        self.head_dim = self.dim // self.num_heads

        self.qkv = qkv.build()
        self.proj = proj.build()
        self.flex_attention = FlexAttention.Config().build()

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        """Apply multi-head attention with 2D RoPE.

        Args:
            hidden_states: (num_vision, max_num_patch, dim)
            rope_cache: (num_vision, max_num_patch, 1, head_dim*2) precomputed cos/sin
            attention_mask: BlockMask for attention

        Returns:
            (num_vision, max_num_patch, dim)
        """
        num_vision, max_num_patch, _ = hidden_states.shape

        qkv = self.qkv(hidden_states).reshape(
            num_vision, max_num_patch, 3, -1, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 1, 3, 4).unbind(
            0
        )  # Each: (num_vision, max_num_patch, heads, head_dim)

        q, k = apply_rotary_emb_cos_sin(q, k, rope_cache)

        attn_output = self.flex_attention(q, k, v, attention_masks=attention_mask)
        attn_output = attn_output.reshape(num_vision, max_num_patch, -1)
        return self.proj(attn_output)


class VisionMLP(Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, *, fc1: Linear.Config, fc2: Linear.Config):
        super().__init__()
        self.linear_fc1 = fc1.build()
        self.linear_fc2 = fc2.build()
        self.act_fn = GELU(approximate="tanh")

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class VisionTransformerBlock(Module):
    """Single transformer block for vision encoder."""

    def __init__(
        self,
        dim: int,
        n_heads: int,
        layer_norm_eps: float,
        *,
        attn_qkv: Linear.Config,
        attn_proj: Linear.Config,
        mlp_fc1: Linear.Config,
        mlp_fc2: Linear.Config,
    ):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=layer_norm_eps)
        self.norm2 = LayerNorm(dim, eps=layer_norm_eps)
        self.attn = VisionAttention(dim, n_heads, qkv=attn_qkv, proj=attn_proj)
        self.mlp = VisionMLP(fc1=mlp_fc1, fc2=mlp_fc2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            rope_cache=rope_cache,
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionEncoder(Module):
    """Qwen3-VL Vision Encoder with FlexAttention.

    Uses padded batches (N, L, D) format for efficient processing.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Configuration for Qwen3-VL Vision Encoder (ViT)."""

        dim: int = 1280
        ffn_dim: int = 5120
        n_layers: int = 32
        n_heads: int = 16

        patch_size: int = 16
        temporal_patch_size: int = 2
        in_channels: int = 3
        spatial_merge_size: int = 2

        out_hidden_size: int = 3584  # maps to LLM hidden size
        num_position_embeddings: int = 4096
        layer_norm_eps: float = 1e-6
        rope_theta: float = 10000.0

        # DeepStack: layer indices for extracting intermediate visual features
        deepstack_visual_indices: list[int] = field(default_factory=lambda: [7, 15, 23])

        # Per-layer Linear configs for vision encoder sub-modules
        patch_embed_proj: Linear.Config
        attn_qkv: Linear.Config
        attn_proj: Linear.Config
        mlp_fc1: Linear.Config
        mlp_fc2: Linear.Config
        merger_fc1: Linear.Config
        merger_fc2: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = config.spatial_merge_size**2

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.dim,
            proj=config.patch_embed_proj,
        )

        # nn.Parameter (not nn.Embedding) because we interpolate the weight directly
        self.num_position_embeddings = config.num_position_embeddings
        self.pos_embed = nn.Parameter(
            torch.empty(config.num_position_embeddings, config.dim)
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.dim // config.n_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(
            head_dim // 2, theta=config.rope_theta
        )
        # Cached RoPE freq table — recomputed only when max_hw grows
        self._cached_freq_table: torch.Tensor | None = None
        # Cached local_map wrapper for learned pos embed computation (created
        # on first use when pos_embed is a DTensor from TP/FSDP wrapping)
        self._local_map_learned_pos_fn = None

        self.layers = ModuleDict(
            {
                str(idx): VisionTransformerBlock(
                    config.dim,
                    config.n_heads,
                    config.layer_norm_eps,
                    attn_qkv=config.attn_qkv,
                    attn_proj=config.attn_proj,
                    mlp_fc1=config.mlp_fc1,
                    mlp_fc2=config.mlp_fc2,
                )
                for idx in range(config.n_layers)
            }
        )

        self.merger = PatchMerger(
            hidden_size=config.dim,
            out_hidden_size=config.out_hidden_size,
            spatial_merge_size=config.spatial_merge_size,
            fc1=config.merger_fc1,
            fc2=config.merger_fc2,
        )

        # DeepStack mergers for intermediate layers
        # DeepStack mergers use postshuffle norm (norm after spatial reshape)
        self.deepstack_visual_indices = config.deepstack_visual_indices
        self.deepstack_merger_list = ModuleList(
            [
                PatchMerger(
                    hidden_size=config.dim,
                    out_hidden_size=config.out_hidden_size,
                    spatial_merge_size=config.spatial_merge_size,
                    fc1=config.merger_fc1,
                    fc2=config.merger_fc2,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(config.deepstack_visual_indices))
            ]
        )

    def compute_position_embeddings(
        self, grid_thw: torch.Tensor, max_num_patch: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute position embeddings for padded batch.

        Delegates to two standalone helpers:
        - ``_compute_learned_pos_embeds``: bilinear-interpolated learned embeddings
        - ``_compute_2d_rope_cache``: 2D RoPE cache

        When ``pos_embed`` is a DTensor (from TP/FSDP wrapping), the learned
        embedding computation is wrapped with ``local_map`` to run interpolation
        and indexing ops on local tensors while preserving proper gradient
        placements.

        Args:
            grid_thw: (num_vision, 3) with pixel patch counts [t, h, w] per visual item
            max_num_patch: Maximum number of patches (for padding)

        Returns:
            learned_pos: (num_vision, max_num_patch, dim) learnable position embeddings
            rope_cache: (num_vision, max_num_patch, 1, head_dim*2) RoPE cache for
                apply_rotary_emb_cos_sin
        """
        head_dim = self.config.dim // self.config.n_heads

        # Get RoPE freq table, reusing cache when possible
        max_hw = int(grid_thw[:, 1:].max().item())
        if self._cached_freq_table is None or self._cached_freq_table.shape[0] < max_hw:
            self._cached_freq_table = self.rotary_pos_emb(max_hw)

        # Compute learned position embeddings
        # When pos_embed is a DTensor (from TP/FSDP wrapping), use local_map
        # so that interpolation and indexing ops run on local tensors while
        # local_map handles DTensor ↔ local conversion and gradient placements
        learned_pos_embed = self.pos_embed
        if isinstance(learned_pos_embed, DTensor):
            if self._local_map_learned_pos_fn is None:
                self._local_map_learned_pos_fn = local_map(
                    _compute_learned_pos_embeds,
                    in_placements=((Replicate(),), None, None, None, None, None),
                    out_placements=((Replicate(),),),
                    in_grad_placements=((Replicate(),), None, None, None, None, None),
                    device_mesh=learned_pos_embed.device_mesh,
                )
            learned_pos = self._local_map_learned_pos_fn(
                learned_pos_embed,
                grid_thw,
                max_num_patch,
                self.num_grid_per_side,
                self.spatial_merge_size,
                self.config.dim,
            )
        else:
            learned_pos = _compute_learned_pos_embeds(
                learned_pos_embed,
                grid_thw,
                max_num_patch,
                self.num_grid_per_side,
                self.spatial_merge_size,
                self.config.dim,
            )

        rope_cache = _compute_2d_rope_cache(
            self._cached_freq_table,
            grid_thw,
            max_num_patch,
            self.spatial_merge_size,
            head_dim,
        )

        return learned_pos, rope_cache

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Forward pass of the vision encoder.

        Processes both images and videos — each visual item is a batch of
        padded patches with a (t, h, w) grid.

        Args:
            pixel_values: Padded patches (num_vision, max_num_patch, patch_dim)
            grid_thw: Grid dimensions (num_vision, 3) for [temporal, height, width] measured in patches

        Returns:
            merged_hidden_states: (num_vision, max_merged_num_patch, out_hidden_size)
            deepstack_features: List of features from intermediate layers
        """
        num_vision, max_num_patch, _ = pixel_values.shape

        num_patch = (grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).to(torch.long)

        hidden_states = self.patch_embed(pixel_values)
        learned_pos, rope_cache = self.compute_position_embeddings(
            grid_thw, max_num_patch
        )
        hidden_states = hidden_states + learned_pos

        mask_mod = get_vision_block_mask_mod(num_patch, max_num_patch)
        attention_mask = _compiled_create_block_mask(
            mask_mod,
            num_vision,
            None,
            max_num_patch,
            max_num_patch,
            device=hidden_states.device,
        )
        deepstack_features = []

        for layer_idx, layer in self.layers.items():
            hidden_states = layer(
                hidden_states,
                rope_cache=rope_cache,
                attention_mask=attention_mask,
            )
            if int(layer_idx) in self.deepstack_visual_indices:
                idx = self.deepstack_visual_indices.index(int(layer_idx))
                deepstack_feature = self.deepstack_merger_list[idx](hidden_states)
                deepstack_features.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)

        return merged_hidden_states, deepstack_features
