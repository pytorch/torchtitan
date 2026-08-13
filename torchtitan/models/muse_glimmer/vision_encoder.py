# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer vision encoder, built on torchtitan's shared ViT components and
structured like ``torchtitan/models/qwen3_5/vision_encoder.py``.

The forward runs the whole batch padded as ``(N, P, D)`` (like qwen3_5/kimi_k2_7):
``conv1`` linear patch embed, ``grid_sample``-resampled learned position
embedding, complex 2D RoPE (:meth:`ComplexRoPE.apply_rotary_emb`), a stack of
transformer blocks, then per-image pixel-shuffle downsampling; the valid tokens
are concatenated into the returned ``(total_output_tokens, output_dim)``.

Attention uses per-image FlexAttention ``(N, P, P)`` masks (the batch dim
separates images). Most layers use windowed attention -- tokens are permuted per
row so each window is contiguous (``token_permute``) and a per-row ``win_id``
confines attention within windows, keeping the mask block-sparse -- while every
``sparse_attention_factor``-th layer and the last use full per-image
block-diagonal attention (``get_vision_block_mask_mod``).

The block stack is named ``layers`` so the shared ``apply_ac``/pipeline tooling
works unchanged.

Shape suffixes:
- N = num images in the batch (one forward)
- P = max patches per image (padded to the batch max)
- D = vision latent dim (``latent_dim``)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common import ComplexRoPE, Linear
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.common.vision_encoder import (
    compiled_create_block_mask,
    get_vision_block_mask_mod,
    VisionTransformerBlock,
)
from torchtitan.protocols.module import Module, ModuleDict


def reorder_patch_vector(
    patches: torch.Tensor, *, patch_size: int, patch_temporal: int
) -> torch.Tensor:
    """Swap the channel/temporal axes of a patch vector: the shared collator
    (reused by qwen3_5/kimi_k2_7) emits ``(c, pt, ph, pw)`` but Muse Glimmer's
    ``conv1_linear`` expects ``(pt, c, ph, pw)`` (its pretrained weight layout).

    ``patches`` is ``(..., patch_dim)`` for any leading dims (packed
    ``(n, patch_dim)`` or padded ``(N, P, patch_dim)``), where ``patch_dim ==
    c * patch_temporal * patch_size**2``; returns the same shape reordered.
    """
    *lead, patch_dim = patches.shape
    ps = patch_size
    pt = patch_temporal
    c = patch_dim // (pt * ps * ps)
    # (..., c, pt, ps, ps) -> (..., pt, c, ps, ps): swap the channel/temporal axes.
    return (
        patches.view(*lead, c, pt, ps, ps).transpose(-4, -3).reshape(*lead, patch_dim)
    )


class _VisionPosEmbed(Module):
    """Bilinearly resample the learned positional grid to (grid_h, grid_w).

    A stateless leaf module so the sharding code can wrap forward with a
    DTensor->local conversion (``grid_sample`` has no DTensor support). Under TP
    the ``pos_param`` parameter arrives as a Replicate DTensor; ``local_map``
    converts it to a local tensor, and the result is wrapped back to Replicate.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(
        self,
        pos_param: torch.Tensor,
        *,
        grid_h: int,
        grid_w: int,
        src_grid_h: int,
        src_grid_w: int,
        latent_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        dtype = pos_param.dtype
        pos_emb = (
            pos_param.view(src_grid_h, src_grid_w, latent_dim)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        inv_h = 1.0 / grid_h
        inv_w = 1.0 / grid_w
        ys = torch.linspace(-1 + inv_h, 1 - inv_h, grid_h, device=device, dtype=dtype)
        xs = torch.linspace(-1 + inv_w, 1 - inv_w, grid_w, device=device, dtype=dtype)
        pos_xy = torch.stack(torch.meshgrid(ys, xs, indexing="xy"), dim=-1).reshape(
            -1, 2
        )[None, None]
        sampled = F.grid_sample(pos_emb, pos_xy, mode="bilinear", align_corners=False)
        return sampled[0, :, 0, :].T  # [grid_h*grid_w, latent_dim]


class _VisionTokenPermute(Module):
    """Row-wise gather of a padded token batch by a per-row permutation.

    A stateless leaf module so the sharding code can wrap forward with a
    DTensor->local conversion. ``torch.gather`` rejects a mix of DTensor and
    plain tensors; running this region on local tensors (both ``x`` and ``index``
    converted by ``local_map``) sidesteps that.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Row-wise gather: ``out[i] = x[i][index[i]]``.

        Used to apply the sparse-window permutation to both the token features
        ``(N, P, D)`` and the complex RoPE cache ``(N, P, 1, head_dim//2)`` -- a
        single ``torch.gather`` over dim 1 with ``index`` broadcast over the
        trailing dims. Invoked via ``__call__`` so the TP wrapper (which
        ``Module.parallelize`` binds to ``forward``) localizes ``x`` and
        ``index`` before the gather.
        """
        trailing = x.shape[2:]
        idx = index.view(index.shape[0], index.shape[1], *([1] * len(trailing)))
        idx = idx.expand(index.shape[0], index.shape[1], *trailing)
        return torch.gather(x, 1, idx)


class _VisionRopeFreq(Module):
    """Holds the RoPE inverse-frequency table (a per-head constant).

    A leaf module with NO sharding_config on purpose: ``Module.parallelize``
    skips config-less modules, so ``inv_freq`` stays a plain tensor under TP.
    ``_make_2d_rope`` combines it with plain index tensors via ``torch.outer``,
    which rejects a mix of DTensor and plain tensors -- keeping it plain (rather
    than a Replicate DTensor on the encoder) sidesteps that. Mirrors qwen3_5's
    ``VisionRotaryEmbedding``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        rope_theta: float

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.register_buffer("inv_freq", self._compute(), persistent=False)

    def _compute(self) -> torch.Tensor:
        """RoPE inverse frequencies, shape [head_dim // 4]."""
        half_dim = self.head_dim // 2
        quarter = half_dim // 2
        return 1.0 / (
            self.rope_theta
            ** (torch.arange(0, half_dim, 2, dtype=torch.float32)[:quarter] / half_dim)
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        # After to_empty() the placeholder records the target device.
        device = buffer_device or self.inv_freq.device
        with torch.device(device):
            self.inv_freq = self._compute()


class MuseGlimmerVisionAdapter(Module):
    """Two-layer GELU MLP adapter mapping encoder features to the LLM adapter
    dimension: ``gelu(c_proj(gelu(c_fc(x))))``."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        c_fc: Linear.Config
        c_proj: Linear.Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.c_fc = config.c_fc.build()
        self.c_proj = config.c_proj.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.c_fc(x))
        x = F.gelu(self.c_proj(x))
        return x


class MuseGlimmerVisionEncoder(Module):
    """Muse Glimmer vision encoder for a batch of (variable-resolution) images.

    All images run through the transformer in one padded ``(N, P, D)`` batch with
    per-image block-diagonal masking, so images in the same forward share a pass
    but cannot attend across each other. The output is pixel-shuffle downsampled
    per image.

    ``forward`` takes padded, pre-patchified ``pixel_values``
    (``[N, P, patch_dim]``, one row per image, zero-padded to the batch's max
    patch count ``P``) and ``grid_thw`` (``[N, 3]`` = ``[1, grid_h, grid_w]`` per
    image; patchify happens in the shared collator). The whole batch is embedded
    padded (the patch vector is reordered from the collator's ``(c, pt, ph, pw)``
    layout to ``conv1``'s ``(pt, c, ph, pw)`` layout); padding rows are masked in
    attention and dropped at finalize. Returns a single
    ``[total_output_tokens, output_dim]`` tensor (valid tokens per image,
    concatenated).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        latent_dim: int
        output_dim: int  # latent_dim * downsample_factor ** 2
        num_layers: int
        num_heads: int
        head_dim: int
        patch_size: int
        patch_temporal: int
        downsample_factor: int
        sparse_attention_factor: int
        pos_emb_grid_h: int
        pos_emb_grid_w: int
        rope_theta: float = 10000.0
        conv1: Linear.Config
        ln_pre: LayerNorm.Config
        block: VisionTransformerBlock.Config
        ln_post: LayerNorm.Config
        pos_embed: Module.Config = field(default_factory=_VisionPosEmbed.Config)
        token_permute: Module.Config = field(default_factory=_VisionTokenPermute.Config)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.latent_dim = config.latent_dim
        self.output_dim = config.output_dim
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.patch_size = config.patch_size
        self.patch_temporal = config.patch_temporal
        self.downsample_factor = config.downsample_factor
        self.sparse_attention_factor = config.sparse_attention_factor
        self.pos_emb_grid_h = config.pos_emb_grid_h
        self.pos_emb_grid_w = config.pos_emb_grid_w
        self.rope_theta = config.rope_theta

        # RoPE inverse frequencies live on a config-less leaf so they stay a
        # plain tensor under TP (see _VisionRopeFreq) -- _make_2d_rope mixes them
        # with plain index tensors, which DTensor would reject.
        self.rope_freq = _VisionRopeFreq.Config(
            head_dim=self.head_dim, rope_theta=self.rope_theta
        ).build()

        self.conv1_linear = config.conv1.build()
        # Raw nn.Parameter (interpolated directly), initialized via the encoder
        # Config's param_init entry for "positional_embedding_vlm".
        self.positional_embedding_vlm = nn.Parameter(
            torch.empty(
                config.pos_emb_grid_h * config.pos_emb_grid_w, config.latent_dim
            )
        )
        self.ln_pre = config.ln_pre.build()
        self.layers = ModuleDict(
            {str(i): config.block.build() for i in range(config.num_layers)}
        )
        self.ln_post = config.ln_post.build()

        # Stateless leaf modules holding the local-tensor compute that has no
        # DTensor support (grid_sample, advanced indexing). Under TP their
        # forwards are wrapped with local_map via sharding_config; on the
        # single-device path they run on plain tensors unchanged.
        self.pos_embed = config.pos_embed.build()
        self.token_permute = config.token_permute.build()

    # ------------------------------------------------------------------
    # Positional helpers
    # ------------------------------------------------------------------

    def _make_2d_rope(
        self, grid_h: int, grid_w: int, device: torch.device
    ) -> torch.Tensor:
        """Complex 2D RoPE frequencies for a grid, shape [grid_h*grid_w, head_dim//2]."""
        inv_freq = self.rope_freq.inv_freq

        idx_h = torch.arange(1, grid_h + 1, dtype=torch.float32, device=device)
        idx_w = torch.arange(1, grid_w + 1, dtype=torch.float32, device=device)
        idx_ij_h = idx_h.unsqueeze(1).expand(-1, grid_w).reshape(-1)
        idx_ij_w = idx_w.unsqueeze(0).expand(grid_h, -1).reshape(-1)

        freq_h = torch.outer(idx_ij_h, inv_freq)
        freq_w = torch.outer(idx_ij_w, inv_freq)
        # Order: [w, h].
        freq = torch.cat([freq_w, freq_h], dim=-1)
        return torch.view_as_complex(
            torch.stack([torch.cos(freq), torch.sin(freq)], dim=-1)
        )

    def _compute_2d_rope_cache(
        self, grids: list[list[int]], max_num_patch: int, device: torch.device
    ) -> torch.Tensor:
        """Padded complex 2D-RoPE cache, grouped by unique (h, w).

        Maps ``_make_2d_rope`` over each unique grid into a padded
        ``(N, max_num_patch, 1, head_dim//2)`` complex cache (head axis = 1 for
        broadcast); padding rows are the identity ``1+0j``.
        """
        n = len(grids)
        cache = torch.ones(
            n, max_num_patch, self.head_dim // 2, device=device, dtype=torch.complex64
        )
        hw_to_indices: dict[tuple[int, int], list[int]] = {}
        for i, (_, h, w) in enumerate(grids):
            hw_to_indices.setdefault((h, w), []).append(i)
        for (h, w), idxs in hw_to_indices.items():
            emb = self._make_2d_rope(h, w, device)  # (h*w, head_dim//2) complex
            for i in idxs:
                cache[i, : h * w] = emb
        return cache.unsqueeze(2)

    def _get_pos_emb(
        self, grid_h: int, grid_w: int, device: torch.device
    ) -> torch.Tensor:
        """Bilinearly resample the learned positional grid to (grid_h, grid_w).

        Thin delegator to ``self.pos_embed`` (binds the encoder's scalars); the
        leaf holds the local-tensor compute and, under TP, the DTensor->local
        conversion of ``positional_embedding_vlm``. Returns
        ``[grid_h*grid_w, latent_dim]``.
        """
        return self.pos_embed(
            self.positional_embedding_vlm,
            grid_h=grid_h,
            grid_w=grid_w,
            src_grid_h=self.pos_emb_grid_h,
            src_grid_w=self.pos_emb_grid_w,
            latent_dim=self.latent_dim,
            device=device,
        )

    def _compute_pos_embeds(
        self, grids: list[list[int]], max_num_patch: int, device: torch.device
    ) -> torch.Tensor:
        """Padded learned positional embeddings, grouped by unique (h, w).

        Returns ``(N, max_num_patch, latent_dim)`` with padding rows left zero.
        Delegates to ``self.pos_embed`` (grid_sample leaf) once per unique grid,
        so TP local_map still localizes ``positional_embedding_vlm``.
        """
        n = len(grids)
        pos = self.positional_embedding_vlm.new_zeros(n, max_num_patch, self.latent_dim)
        hw_to_indices: dict[tuple[int, int], list[int]] = {}
        for i, (_, h, w) in enumerate(grids):
            hw_to_indices.setdefault((h, w), []).append(i)
        for (h, w), idxs in hw_to_indices.items():
            emb = self._get_pos_emb(h, w, device).to(pos.dtype)  # (h*w, D)
            for i in idxs:
                pos[i, : h * w] = emb
        return pos

    def _pixel_shuffle_downsample(
        self, x: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        """Downsample via pixel shuffle: (h r1 w r2) -> (h w r1 r2).

        Takes one image's ``(grid_h*grid_w, d)`` tokens and returns
        ``(n_out, d * f * f)``. The downsample is a fixed permutation of the patch
        grid, so it is expressed as pure view/permute/reshape rather than an
        ``arange`` gather. These ops have DTensor sharding rules, so this runs
        natively on a Replicate DTensor under TP (no ``local_map`` needed) and
        identically on plain tensors single-device.
        """
        f = self.downsample_factor
        d = x.shape[-1]
        n_out = (grid_h // f) * (grid_w // f)
        return (
            x.view(grid_h // f, f, grid_w // f, f, d)
            .permute(0, 2, 1, 3, 4)
            .reshape(n_out, f * f, d)
            .permute(0, 2, 1)
            .reshape(n_out, d * f * f)
        )

    def _get_sparse_perm_and_slens(
        self, grid_h: int, grid_w: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Group patches into pos_emb_grid_h x pos_emb_grid_w windows.

        Returns the permutation that makes each window contiguous and the
        per-window token counts.
        """
        gh, gw = self.pos_emb_grid_h, self.pos_emb_grid_w
        pad_h = math.ceil(grid_h / gh) * gh
        pad_w = math.ceil(grid_w / gw) * gw

        idx = torch.arange(grid_h * grid_w, device=device).view(grid_h, grid_w)
        idx = F.pad(idx, (0, pad_w - grid_w, 0, pad_h - grid_h), value=-1).flatten()
        idx = idx.view(pad_h // gh, gh, pad_w // gw, gw)
        idx = idx.permute(0, 2, 1, 3).reshape(-1)

        sp_perm = idx[idx != -1]
        valid = (idx != -1).view(-1, gh * gw)
        # Keep the per-window token counts on-device: they are consumed by
        # ``_compute_sparse_perm_winid`` (repeat_interleave) which runs on the
        # GPU. A ``.tolist()`` here would force a D2H sync every forward.
        sp_slens = valid.sum(dim=1).to(torch.int32)
        return sp_perm, sp_slens

    def _compute_sparse_perm_winid(
        self, grids: list[list[int]], max_num_patch: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Padded per-row window permutation, its inverse, and permuted-frame
        window ids.

        Returns ``(perm, inv, win_id)`` each ``(N, max_num_patch)``:
        - ``perm[i, :n]`` reorders image i's tokens so windows are contiguous
          (identity on the padded tail so ``gather`` stays in-bounds).
        - ``inv[i]`` is the inverse of ``perm[i]`` (identity on the tail).
        - ``win_id[i, :n]`` is the window index of each token in the *permuted*
          frame; the padded tail is ``-1`` so it never matches in the mask.
        """
        n_img = len(grids)
        perm = (
            torch.arange(max_num_patch, device=device)
            .unsqueeze(0)
            .repeat(n_img, 1)
            .clone()
        )
        inv = perm.clone()
        win_id = torch.full(
            (n_img, max_num_patch), -1, device=device, dtype=torch.int32
        )
        for i, (_, h, w) in enumerate(grids):
            n = h * w
            sp_perm, sp_slens = self._get_sparse_perm_and_slens(h, w, device)
            perm[i, :n] = sp_perm
            row_inv = torch.empty_like(sp_perm)
            row_inv[sp_perm] = torch.arange(n, device=device)
            inv[i, :n] = row_inv
            win_id[i, :n] = torch.repeat_interleave(
                torch.arange(sp_slens.numel(), device=device, dtype=torch.int32),
                sp_slens,
                output_size=n,
            )
        return perm, inv, win_id

    def _build_padded_masks(
        self,
        num_patch: torch.Tensor,
        win_id: torch.Tensor,
        max_num_patch: int,
        device: torch.device,
    ) -> tuple[BlockMask, BlockMask | None]:
        """Global (N,P,P) mask, plus the window mask when sparse is active.

        The sparse ``mask_mod`` keeps a query/key pair iff both are valid (within
        the image's patch count) AND share a window in the permuted frame.
        """
        n = num_patch.shape[0]
        global_mask = compiled_create_block_mask(
            get_vision_block_mask_mod(num_patch),  # per-image block-diagonal
            n,
            None,
            max_num_patch,
            max_num_patch,
            device=device,
        )
        sparse_mask: BlockMask | None = None
        if self.sparse_attention_factor > 1:

            def sparse_mask_mod(b, h, q_idx, kv_idx):
                valid = (q_idx < num_patch[b]) & (kv_idx < num_patch[b])
                return valid & (win_id[b, q_idx] == win_id[b, kv_idx])

            sparse_mask = compiled_create_block_mask(
                sparse_mask_mod,
                n,
                None,
                max_num_patch,
                max_num_patch,
                device=device,
            )
        return global_mask, sparse_mask

    def _layer_uses_global_attention(self, layer_idx: int) -> bool:
        """Whether layer ``layer_idx`` attends globally (vs. sparse windows).

        Global on the last layer and every ``sparse_attention_factor``-th layer,
        sparse otherwise. When ``sparse_attention_factor == 1`` this is always
        True, so every layer uses the global mask.
        """
        sf = self.sparse_attention_factor
        is_last = layer_idx == len(self.layers) - 1
        return is_last or (layer_idx + 1) % sf == 0

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        device = self.conv1_linear.weight.device
        dtype = self.conv1_linear.weight.dtype
        sf = self.sparse_attention_factor
        N, P, _ = pixel_values.shape

        assert bool((grid_thw[:, 0] == 1).all()), (
            "MuseGlimmerVisionEncoder only supports grid_thw[:, 0] == 1 "
            f"(T patches), got {grid_thw[:, 0].tolist()}"
        )
        f = self.downsample_factor
        assert bool((grid_thw[:, 1:] % f == 0).all()), (
            "MuseGlimmerVisionEncoder requires grid h/w divisible by "
            f"downsample_factor={f} (pixel-shuffle), got {grid_thw[:, 1:].tolist()}"
        )
        grids = grid_thw.tolist()  # [[t, h, w], ...]; one host sync for the forward
        num_patch = (grid_thw[:, 1] * grid_thw[:, 2]).to(torch.long)  # (N,)

        # Phase 1: padded embed. Reorder the whole (N, P, patch_dim) batch from
        # the collator's (c, pt, ps, ps) layout to conv1's (pt, c, ps, ps), then
        # conv1 -> pos -> ln_pre on (N, P, D). Padding rows stay masked downstream
        # (conv1 is bias-free; pos-emb padding is zero).
        patches = reorder_patch_vector(
            pixel_values.to(device=device, dtype=dtype),
            patch_size=self.patch_size,
            patch_temporal=self.patch_temporal,
        )
        x = self.conv1_linear(patches)  # (N, P, D)
        x = x + self._compute_pos_embeds(grids, P, device).to(dtype)
        x = self.ln_pre(x)

        rope_cache = self._compute_2d_rope_cache(grids, P, device)  # (N,P,1,hd//2)

        # Sparse permutation: reorder each row so windows are contiguous. The
        # gather runs through token_permute.__call__ -> forward so the TP
        # local_map wrapper localizes x/index before torch.gather.
        inv = None
        win_id = torch.full((N, P), -1, device=device, dtype=torch.int32)
        if sf > 1:
            perm, inv, win_id = self._compute_sparse_perm_winid(grids, P, device)
            x = self.token_permute(x, perm)
            rope_cache = self.token_permute(rope_cache, perm)

        # Phase 2: masks + transformer (block stack unchanged, now with N>1).
        global_mask, sparse_mask = self._build_padded_masks(
            num_patch, win_id, P, device
        )
        for i, block in enumerate(self.layers.values()):
            mask = global_mask if self._layer_uses_global_attention(i) else sparse_mask
            x = block(
                x,
                rope_cache=rope_cache,
                rope_apply=ComplexRoPE.apply_rotary_emb,
                attention_mask=mask,
            )

        # Phase 3: un-permute, ln_post, per-image pixel-shuffle, flatten valid tokens.
        if sf > 1:
            x = self.token_permute(x, inv)
        x = self.ln_post(x)

        all_features: list[torch.Tensor] = []
        for i, (_, grid_h, grid_w) in enumerate(grids):
            n = grid_h * grid_w
            feat = self._pixel_shuffle_downsample(x[i, :n], grid_h, grid_w)
            all_features.append(feat)
        return torch.cat(all_features, dim=0)  # (total_output_tokens, output_dim)
