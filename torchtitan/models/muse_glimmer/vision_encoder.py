# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Muse Glimmer vision encoder built on torchtitan's shared component library.

Structurally follows ``torchtitan/models/qwen3_5/vision_encoder.py`` (nested
``Config`` dataclasses, ``ModuleDict`` layer stack, shared
``Linear``/``LayerNorm``/``FlexAttention``), with Muse Glimmer-specific internals:

- patches are pre-extracted by a ``Linear`` ``conv1`` (a full-patch-kernel Conv
  expressed as a linear over flattened patches),
- a learned positional embedding is bilinearly resampled (``grid_sample``) to
  each image's grid,
- complex 2D RoPE (the same complex backend as the LLM; see
  :meth:`ComplexRoPE.apply_rotary_emb`),
- per-image block-diagonal attention with an additional sparse-window variant
  (``sparse_attention_factor``), realized as FlexAttention ``BlockMask``s built
  from segment ids (the document-mask pattern),
- pixel-shuffle spatial downsampling at the output.

The transformer block stack is named ``layers`` so the shared
``apply_ac``/pipeline tooling (which expects a ``layers`` submodule, like
``torchtitan/models/qwen3_5``) works unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common import ComplexRoPE, Linear
from torchtitan.models.common.attention import create_attention_mask
from torchtitan.models.common.nn_modules import LayerNorm
from torchtitan.models.common.vision_encoder import VisionTransformerBlock
from torchtitan.protocols.module import Module, ModuleDict


def reorder_patch_vector(
    patches: torch.Tensor, *, patch_size: int, patch_temporal: int
) -> torch.Tensor:
    """Reorder a patch vector from the shared collator's ``(c, pt, ph, pw)``
    layout to Muse Glimmer's ``(pt, c, ph, pw)`` ``conv1_linear`` layout.

    Why this exists: the shared multimodal collator (``vision_to_patches``,
    reused by qwen3_5/kimi_k2_7) emits patch vectors channel-first as
    ``(c, pt, ph, pw)``. Muse Glimmer's ``conv1_linear`` was defined with the
    temporal axis first, ``(pt, c, ph, pw)`` -- matching the pretrained
    ``conv1`` weight layout -- so we swap the c/pt axes here
    to keep ``conv1_linear`` numerics byte-identical to the old packed collator.

    Cost: a single memory-bandwidth-bound copy per forward (the ``reshape`` after
    a non-contiguous ``permute``), ~tens of MB, dwarfed by the ViT matmuls -- so
    doing it in the encoder is effectively free. If this ever needs to be truly
    zero-GPU-cost, push the reorder into the shared collator via an opt-in
    ``patch_vector_order`` param (runs on CPU dataloader workers); if Muse Glimmer
    never has to match an existing ``(pt, c, ph, pw)`` checkpoint, drop this
    entirely and define ``conv1`` natively in ``(c, pt, ph, pw)``.

    ``patches`` is ``[n, patch_dim]`` where ``patch_dim == c * patch_temporal *
    patch_size**2``. Returns ``[n, patch_dim]`` with the channel/temporal axes
    swapped so the vector matches the ordering ``conv1_linear`` expects.
    """
    n, patch_dim = patches.shape
    ps = patch_size
    pt = patch_temporal
    c = patch_dim // (pt * ps * ps)
    # (c, pt, ps, ps) -> (pt, c, ps, ps): swap the channel and temporal axes.
    return patches.view(n, c, pt, ps, ps).permute(0, 2, 1, 3, 4).reshape(n, patch_dim)


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
    """Advanced-index a token sequence by a permutation: ``x[:, index]``.

    A stateless leaf module so the sharding code can wrap forward with a
    DTensor->local conversion. Advanced indexing (``aten.index.Tensor``) rejects
    a mix of DTensor and plain tensors; running this region on local tensors
    (both ``x`` and ``index`` converted by ``local_map``) sidesteps that.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        return x[:, index]


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

    All images are embedded and concatenated into one sequence that runs through
    the transformer once with block-diagonal masking, so images in the same
    forward share a pass but cannot attend across each other. The output is
    pixel-shuffle downsampled per image.

    ``forward`` takes padded, pre-patchified ``pixel_values``
    (``[N, P, patch_dim]``, one row per image, zero-padded to the batch's max
    patch count ``P``) and ``grid_thw`` (``[N, 3]`` = ``[1, grid_h, grid_w]`` per
    image; patchify happens in the shared collator). Each row is unpadded via
    ``grid_thw`` and its patch vector reordered from the collator's
    ``(c, pt, ph, pw)`` layout to ``conv1``'s ``(pt, c, ph, pw)`` layout. Returns
    a single ``[total_output_tokens, output_dim]`` tensor.
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

    def _pixel_shuffle_downsample(
        self, x: torch.Tensor, grid_h: int, grid_w: int
    ) -> torch.Tensor:
        """Downsample via pixel shuffle: (h r1 w r2) -> (h w r1 r2).

        The downsample is a fixed permutation of the patch grid, so it is
        expressed as pure view/permute/reshape rather than an ``arange`` gather.
        These ops have DTensor sharding rules, so this runs natively on a
        Replicate DTensor under TP (no ``local_map`` needed) and identically on
        plain tensors single-device.
        """
        f = self.downsample_factor
        d = x.shape[-1]
        n_out = (grid_h // f) * (grid_w // f)
        return (
            x.squeeze(0)
            .view(grid_h // f, f, grid_w // f, f, d)
            .permute(0, 2, 1, 3, 4)
            .reshape(n_out, f * f, d)
            .permute(0, 2, 1)
            .reshape(n_out, d * f * f)
            .unsqueeze(0)
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
        # ``_block_diag_mask`` (repeat_interleave) which runs on the GPU. A
        # ``.tolist()`` here would force a D2H sync every forward.
        sp_slens = valid.sum(dim=1).to(torch.int32)
        return sp_perm, sp_slens

    def _block_diag_mask(
        self, slens: torch.Tensor, total_tokens: int, device: torch.device
    ) -> BlockMask:
        """FlexAttention block-diagonal mask over contiguous segments.

        Each token gets a segment id from ``slens``; a query attends to a key iff
        they share a segment (the document-mask pattern).
        """
        seg = torch.repeat_interleave(
            torch.arange(slens.shape[0], device=device, dtype=torch.int32),
            slens.to(torch.int32),
            # ``total_tokens`` is already known, so pass it explicitly: without
            # ``output_size`` repeat_interleave reads ``slens.sum()`` back to the
            # host to size its output, reintroducing the D2H sync.
            output_size=total_tokens,
        )

        def mask_mod(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            return seg[q_idx] == seg[kv_idx]

        return create_attention_mask(
            mask_mod, 1, None, total_tokens, total_tokens, device=device
        )

    def _embed_image(
        self,
        patches: torch.Tensor,
        grid_h: int,
        grid_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor, torch.Tensor, int, int, int, torch.Tensor | None, torch.Tensor
    ]:
        """Phase 1 for one image: conv1 -> pos emb -> ln_pre -> (sparse) perm.

        ``patches`` is the pre-patchified ``[n_tokens, patch_dim]`` slice for one
        image (patchify now happens in the collator). Returns the embedded tokens
        ``[1, n, latent_dim]``, their 2D-RoPE freqs, ``n_tokens``, ``grid_h``,
        ``grid_w``, and (under sparse attention) the per-window permutation and
        segment lengths (``None``/``[]`` otherwise).
        """
        n_tokens = grid_h * grid_w

        x = self.conv1_linear(patches.unsqueeze(0).to(device=device, dtype=dtype))
        pos_emb = self._get_pos_emb(grid_h, grid_w, device)
        x = x + pos_emb.unsqueeze(0).to(dtype)
        x = self.ln_pre(x.view(-1, self.latent_dim)).view(1, -1, self.latent_dim)

        freqs_cis = self._make_2d_rope(grid_h, grid_w, device)

        sp_perm: torch.Tensor | None = None
        sp_slens: torch.Tensor = torch.empty(0, dtype=torch.int32, device=device)
        if self.sparse_attention_factor > 1:
            sp_perm, sp_slens = self._get_sparse_perm_and_slens(grid_h, grid_w, device)
            x = self.token_permute(x, sp_perm)
            freqs_cis = freqs_cis[sp_perm]

        return x, freqs_cis, n_tokens, grid_h, grid_w, sp_perm, sp_slens

    def _finalize_image(
        self,
        x: torch.Tensor,
        grid_h: int,
        grid_w: int,
        sp_perm: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        """Phase 3 for one image: un-permute (if sparse) -> ln_post -> downsample."""
        if sp_perm is not None:
            inv_perm = torch.empty_like(sp_perm)
            inv_perm[sp_perm] = torch.arange(len(sp_perm), device=device)
            x = self.token_permute(x, inv_perm)
        x = self.ln_post(x.view(-1, self.latent_dim)).view(1, -1, self.latent_dim)
        x = self._pixel_shuffle_downsample(x, grid_h, grid_w)
        return x.squeeze(0)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        device = self.conv1_linear.weight.device
        dtype = self.conv1_linear.weight.dtype
        sf = self.sparse_attention_factor

        # Patches per image. The temporal grid dim is always 1 (temporal is
        # folded into the patch vector), so t * h * w == h * w == n_tokens.
        num_images = grid_thw.shape[0]

        # Phase 1: unpad + reorder patches per image -> embed -> concat.
        all_x: list[torch.Tensor] = []
        all_freqs: list[torch.Tensor] = []
        all_sp_slens: list[torch.Tensor] = []
        all_global_slens: list[int] = []
        per_image_meta: list[tuple[int, int, int, torch.Tensor | None]] = []

        # The unpad below slices ``pixel_values[i, :grid_h * grid_w]``, which is
        # only correct when the temporal grid dim is 1 (T folded into the patch
        # vector). Fail loudly if a caller ever passes T > 1, otherwise patches
        # would be silently dropped and the reorder/embed would misalign.
        assert bool((grid_thw[:, 0] == 1).all()), (
            "MuseGlimmerVisionEncoder only supports grid_thw[:, 0] == 1 "
            f"(T patches), got {grid_thw[:, 0].tolist()}"
        )

        grid_hw = grid_thw[:, 1:3].tolist()
        for i in range(num_images):
            grid_h, grid_w = int(grid_hw[i][0]), int(grid_hw[i][1])
            n = grid_h * grid_w
            # Padded contract: row i holds this image's patches, zero-padded to
            # P. Unpad to the real n patches, then reorder the patch vector from
            # the shared collator's (c, pt, ps, ps) layout to conv1's
            # (pt, c, ps, ps) layout so conv1_linear numerics are preserved.
            img_patches = reorder_patch_vector(
                pixel_values[i, :n],
                patch_size=self.patch_size,
                patch_temporal=self.patch_temporal,
            )
            (
                x,
                freqs_cis,
                n_tokens,
                grid_h,
                grid_w,
                sp_perm,
                sp_slens,
            ) = self._embed_image(img_patches, grid_h, grid_w, device, dtype)
            all_x.append(x.squeeze(0))
            all_freqs.append(freqs_cis)
            if sp_slens.numel() > 0:
                all_sp_slens.append(sp_slens)
            all_global_slens.append(n_tokens)
            per_image_meta.append((grid_h, grid_w, n_tokens, sp_perm))

        # Phase 2: concatenate, build masks, run the transformer once.
        x = torch.cat(all_x, dim=0).unsqueeze(0)
        freqs_cis = torch.cat(all_freqs, dim=0)
        # Named to match the shared block's ``rope_cache`` arg, but here it is not
        # a persistent cache: these 2D-RoPE freqs are recomputed every forward
        # (per image in _make_2d_rope, then concatenated). Reshape to
        # [1, total_tokens, 1, head_dim//2] to broadcast over batch(=1) and heads;
        # this reshape used to live inside MuseGlimmerVisionAttention.
        rope_cache = freqs_cis.unsqueeze(0).unsqueeze(2)
        total_tokens = x.shape[1]

        sp_slens_cat = torch.cat(all_sp_slens) if all_sp_slens else None
        global_mask = self._block_diag_mask(
            torch.tensor(all_global_slens, device=device, dtype=torch.int32),
            total_tokens,
            device,
        )
        sp_mask: BlockMask | None = (
            self._block_diag_mask(sp_slens_cat, total_tokens, device)
            if sp_slens_cat is not None
            else None
        )

        num_layers = len(self.layers)
        for i, block in enumerate(self.layers.values()):
            is_global = (i == num_layers - 1) or ((i + 1) % sf == 0)
            mask = global_mask if (is_global or sp_slens_cat is None) else sp_mask
            x = block(
                x,
                rope_cache=rope_cache,
                rope_apply=ComplexRoPE.apply_rotary_emb,
                attention_mask=mask,
            )

        # Phase 3: split per image, finalize each.
        all_features: list[torch.Tensor] = []
        offset = 0
        for grid_h, grid_w, n_tokens, sp_perm in per_image_meta:
            img_x = x[:, offset : offset + n_tokens]
            offset += n_tokens
            all_features.append(
                self._finalize_image(img_x, grid_h, grid_w, sp_perm, device)
            )

        return torch.cat(all_features, dim=0)
