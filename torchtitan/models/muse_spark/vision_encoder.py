# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NaViT encoder and adapter used by the Muse Spark Open vision tower.

Shape suffixes:
- T = packed visual tokens
- D = vision dim
- H = height
- W = width
- N = generic patch count
- P = flattened patch-vector dim
- G = spatial grid tokens
- M = merged visual tokens
- O = adapter output dim

TODO: replace Muse Glimmer vision encoder with this
"""

import math
from dataclasses import dataclass, field
from typing import cast

import spmd_types as spmd
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan.models.base.module import Module, ModuleDict

from torchtitan.models.common import ComplexRoPE, Linear
from torchtitan.models.common.nn_modules import GELU, LayerNorm
from torchtitan.models.common.vision_encoder import (
    create_block_diagonal_mask,
    RopeApply,
    VisionTransformerBlock,
)


def _identity_rope(
    query_TNH: torch.Tensor,
    key_TNH: torch.Tensor,
    rope_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    _ = rope_cache
    return query_TNH, key_TNH


def _annotate_vision_activation_type(tensor: torch.Tensor) -> torch.Tensor:
    if spmd.is_type_checking():
        return spmd.mutate_type(
            tensor,
            src=spmd.R,
            dst={"dp": spmd.V, "cp": spmd.I, "tp": spmd.I},
        )
    return tensor


def reorder_patch_vector_channel_temporal(
    patches_NP: torch.Tensor, *, patch_size: int, patch_temporal: int
) -> torch.Tensor:
    """Reorder patch vectors from ``(C, PT, PH, PW)`` to ``(PT, C, PH, PW)``.

    The shared multimodal collator emits flattened patch vectors in
    channel-first layout to match Conv3d kernels. Some pretrained ViT patch
    projections instead expect temporal-first vectors, so those encoders call
    this helper before their linearized patch projection.
    """
    num_patches, patch_dim = patches_NP.shape
    channels = patch_dim // (patch_temporal * patch_size * patch_size)
    return (
        patches_NP.view(
            num_patches,
            channels,
            patch_temporal,
            patch_size,
            patch_size,
        )
        .permute(0, 2, 1, 3, 4)
        .reshape(num_patches, patch_dim)
    )


class MuseSparkNaViTAdapter(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        c_fc: Linear.Config
        c_proj: Linear.Config
        act_fn: GELU.Config = field(default_factory=GELU.Config)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.c_fc = config.c_fc.build()
        self.c_proj = config.c_proj.build()
        self.act_fn = config.act_fn.build()

    def forward(self, value_MO: torch.Tensor) -> torch.Tensor:
        return self.act_fn(self.c_proj(self.act_fn(self.c_fc(value_MO))))


class VisionRopeFreq(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        rope_theta: float

        def __post_init__(self) -> None:
            if self.head_dim % 4:
                raise ValueError("NaViT attention head size must be divisible by four")

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.register_buffer("inv_freq", self._compute(), persistent=False)

    def _compute(self, *, device: torch.device | None = None) -> torch.Tensor:
        half_dim = self.head_dim // 2
        return self.rope_theta ** (
            -torch.arange(0, half_dim, 2, dtype=torch.float32, device=device) / half_dim
        )

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        self.inv_freq = self._compute(device=buffer_device or self.inv_freq.device)


class NaViTEncoder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        latent_dim: int
        output_dim: int
        num_layers: int
        num_heads: int
        head_dim: int
        patch_size: int
        patch_temporal: int
        downsample_factor: int
        sparse_attention_factor: int
        pos_emb_grid_h: int
        pos_emb_grid_w: int
        rope_freq: VisionRopeFreq.Config
        conv1: Linear.Config
        ln_pre: LayerNorm.Config
        block: VisionTransformerBlock.Config
        ln_post: LayerNorm.Config
        use_rope: bool = True
        grid_sample_mode: str = "bilinear"
        pos_emb_grid_order: str = "raster"

    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.output_dim != config.latent_dim * config.downsample_factor**2:
            raise ValueError(
                "NaViTEncoder output_dim must equal latent_dim * "
                "downsample_factor squared"
            )
        if config.latent_dim % config.num_heads:
            raise ValueError("vision hidden size must be divisible by attention heads")
        if config.head_dim != config.latent_dim // config.num_heads:
            raise ValueError("vision head_dim must equal latent_dim // num_heads")
        if config.patch_size <= 0:
            raise ValueError("NaViT patch size must be positive")
        if config.patch_temporal <= 0:
            raise ValueError("NaViT temporal patch size must be positive")
        if config.downsample_factor <= 0:
            raise ValueError("NaViT downsample factor must be positive")
        if config.sparse_attention_factor <= 0:
            raise ValueError("NaViT sparse attention factor must be positive")
        if config.pos_emb_grid_h <= 0 or config.pos_emb_grid_w <= 0:
            raise ValueError("NaViT positional embedding grid must be positive")
        if config.pos_emb_grid_order not in {"raster", "legacy_spark"}:
            raise ValueError("NaViT positional embedding grid order is invalid")

        self.patch_size = config.patch_size
        self.patch_temporal = config.patch_temporal
        self.downsample_factor = config.downsample_factor
        self.sparse_attention_factor = config.sparse_attention_factor
        self.pos_emb_grid_h = config.pos_emb_grid_h
        self.pos_emb_grid_w = config.pos_emb_grid_w
        self.use_rope = config.use_rope
        self.grid_sample_mode = config.grid_sample_mode
        self.pos_emb_grid_order = config.pos_emb_grid_order

        self.rope_freq = config.rope_freq.build()
        self.conv1_linear = config.conv1.build()
        self.positional_embedding_vlm = nn.Parameter(
            torch.empty(
                config.pos_emb_grid_h * config.pos_emb_grid_w,
                config.latent_dim,
            )
        )
        self.ln_pre = config.ln_pre.build()
        self.layers = ModuleDict(
            {str(index): config.block.build() for index in range(config.num_layers)}
        )
        self.ln_post = config.ln_post.build()

    def _make_2d_rope(
        self,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        inv_freq_D = self.rope_freq.inv_freq
        pos_h_H = torch.arange(1, grid_h + 1, dtype=torch.float32, device=device)
        pos_w_W = torch.arange(1, grid_w + 1, dtype=torch.float32, device=device)
        pos_h_H = _annotate_vision_activation_type(pos_h_H)
        pos_w_W = _annotate_vision_activation_type(pos_w_W)
        idx_h_G = pos_h_H.unsqueeze(1).expand(-1, grid_w).reshape(-1)
        idx_w_G = pos_w_W.unsqueeze(0).expand(grid_h, -1).reshape(-1)

        freq_h_GD = torch.outer(idx_h_G, inv_freq_D)
        freq_w_GD = torch.outer(idx_w_G, inv_freq_D)
        angles_GD = torch.cat([freq_w_GD, freq_h_GD], dim=-1)
        return torch.view_as_complex(
            torch.stack([torch.cos(angles_GD), torch.sin(angles_GD)], dim=-1)
        )

    def _get_pos_emb(
        self,
        grid_h: int,
        grid_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        position_weights = self.positional_embedding_vlm
        dtype = position_weights.dtype
        pos_emb_DHW = (
            position_weights.view(self.pos_emb_grid_h, self.pos_emb_grid_w, -1)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        inv_h = 1.0 / grid_h
        inv_w = 1.0 / grid_w
        ys_H = torch.linspace(-1 + inv_h, 1 - inv_h, grid_h, device=device, dtype=dtype)
        xs_W = torch.linspace(-1 + inv_w, 1 - inv_w, grid_w, device=device, dtype=dtype)
        # grid_sample consumes coordinates as (x, y). The default path uses
        # `ij` to keep H-major raster order and stacks `(xx, yy)`.
        # `legacy_spark` preserves the Muse Spark Open/SGLang checkpoint convention,
        # where `xy` indexing with `(y, x)` inputs is part of the learned
        # positional layout.
        if self.pos_emb_grid_order == "legacy_spark":
            pos_xy = torch.stack(
                torch.meshgrid(ys_H, xs_W, indexing="xy"),
                dim=-1,
            ).reshape(1, 1, grid_h * grid_w, 2)
        else:
            yy_HW, xx_HW = torch.meshgrid(ys_H, xs_W, indexing="ij")
            pos_xy = torch.stack((xx_HW, yy_HW), dim=-1).reshape(
                1, 1, grid_h * grid_w, 2
            )
        pos_xy = _annotate_vision_activation_type(pos_xy)
        sampled = F.grid_sample(
            pos_emb_DHW,
            pos_xy,
            mode=self.grid_sample_mode,
            align_corners=False,
        )
        return sampled[0, :, 0].transpose(0, 1)

    def _pixel_shuffle_downsample(
        self,
        value_GD: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        factor = self.downsample_factor
        hidden_dim = value_GD.shape[-1]
        return (
            value_GD.view(
                grid_h // factor,
                factor,
                grid_w // factor,
                factor,
                hidden_dim,
            )
            .permute(0, 2, 1, 3, 4)
            .reshape(-1, factor * factor, hidden_dim)
            .transpose(1, 2)
            .reshape(-1, factor * factor * hidden_dim)
        )

    def _get_sparse_perm_and_slens(
        self, grid_h: int, grid_w: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gh, gw = self.pos_emb_grid_h, self.pos_emb_grid_w
        pad_h = math.ceil(grid_h / gh) * gh
        pad_w = math.ceil(grid_w / gw) * gw

        idx = torch.arange(grid_h * grid_w, device=device).view(grid_h, grid_w)
        idx = _annotate_vision_activation_type(idx)
        idx = F.pad(idx, (0, pad_w - grid_w, 0, pad_h - grid_h), value=-1).flatten()
        idx = idx.view(pad_h // gh, gh, pad_w // gw, gw)
        idx = idx.permute(0, 2, 1, 3).reshape(-1)

        sp_perm = idx[idx != -1]
        valid = (idx != -1).view(-1, gh * gw)
        sp_slens = valid.sum(dim=1).to(torch.int32)
        return sp_perm, sp_slens

    def _embed_image(
        self,
        patches_NP: torch.Tensor,
        grid_h: int,
        grid_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        torch.Tensor, torch.Tensor, int, int, int, torch.Tensor | None, torch.Tensor
    ]:
        num_tokens = grid_h * grid_w
        patches_NP = reorder_patch_vector_channel_temporal(
            patches_NP,
            patch_size=self.patch_size,
            patch_temporal=self.patch_temporal,
        )

        x_TD = self.conv1_linear(patches_NP.to(device=device, dtype=dtype))
        pos_emb_GD = self._get_pos_emb(grid_h, grid_w, device)
        x_TD = self.ln_pre(x_TD + pos_emb_GD.to(dtype))

        freqs_cis_GD = (
            self._make_2d_rope(grid_h, grid_w, device)
            if self.use_rope
            else x_TD.new_empty((num_tokens, 0))
        )

        sp_perm: torch.Tensor | None = None
        sp_slens = torch.empty(0, dtype=torch.int32, device=device)
        if self.sparse_attention_factor > 1:
            sp_perm, sp_slens = self._get_sparse_perm_and_slens(grid_h, grid_w, device)
            x_TD = x_TD[sp_perm]
            freqs_cis_GD = freqs_cis_GD[sp_perm]

        return x_TD, freqs_cis_GD, num_tokens, grid_h, grid_w, sp_perm, sp_slens

    def _finalize_image(
        self,
        x_TD: torch.Tensor,
        grid_h: int,
        grid_w: int,
        sp_perm: torch.Tensor | None,
    ) -> torch.Tensor:
        if sp_perm is not None:
            inv_perm = torch.argsort(sp_perm)
            x_TD = x_TD[inv_perm]
        x_TD = self.ln_post(x_TD)
        return self._pixel_shuffle_downsample(x_TD, grid_h, grid_w)

    def forward(
        self,
        pixel_values_NP: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        grids = grid_thw.tolist()

        device = self.conv1_linear.weight.device
        dtype = self.conv1_linear.weight.dtype
        sf = self.sparse_attention_factor

        all_x: list[torch.Tensor] = []
        all_freqs: list[torch.Tensor] = []
        all_sp_slens: list[torch.Tensor] = []
        all_global_slens: list[int] = []
        per_frame_meta: list[tuple[int, int, int, torch.Tensor | None]] = []

        offset = 0
        for temporal, grid_h, grid_w in grids:
            num_spatial_patches = grid_h * grid_w
            for _ in range(temporal):
                frame_patches_NP = pixel_values_NP[
                    offset : offset + num_spatial_patches
                ]
                offset += num_spatial_patches
                (
                    x_TD,
                    freqs_cis_TD,
                    num_tokens,
                    grid_h,
                    grid_w,
                    sp_perm,
                    sp_slens,
                ) = self._embed_image(frame_patches_NP, grid_h, grid_w, device, dtype)
                all_x.append(x_TD)
                all_freqs.append(freqs_cis_TD)
                if sp_slens.numel() > 0:
                    all_sp_slens.append(sp_slens)
                all_global_slens.append(num_tokens)
                per_frame_meta.append((grid_h, grid_w, num_tokens, sp_perm))

        x_TD = torch.cat(all_x, dim=0)
        freqs_cis_TD = torch.cat(all_freqs, dim=0)
        rope_cache = freqs_cis_TD.unsqueeze(1)
        rope_apply = (
            cast(RopeApply, ComplexRoPE.apply_rotary_emb)
            if self.use_rope
            else _identity_rope
        )
        total_tokens = x_TD.shape[0]

        sp_slens_cat = torch.cat(all_sp_slens) if all_sp_slens else None
        global_slens = torch.tensor(
            all_global_slens,
            device=device,
            dtype=torch.int32,
        )
        global_slens = _annotate_vision_activation_type(global_slens)
        with spmd.no_typecheck():
            full_attention_mask = create_block_diagonal_mask(
                global_slens,
                total_tokens,
                device,
            )
            sparse_attention_mask = (
                create_block_diagonal_mask(
                    sp_slens_cat,
                    total_tokens,
                    device,
                )
                if sp_slens_cat is not None
                else None
            )

        num_layers = len(self.layers)
        for layer_id, layer in enumerate(self.layers.values()):
            use_full_attention = layer_id == num_layers - 1 or (layer_id + 1) % sf == 0
            if use_full_attention:
                attention_mask = full_attention_mask
            else:
                assert sparse_attention_mask is not None
                attention_mask = sparse_attention_mask
            x_TD = layer(
                x_TD,
                rope_cache=rope_cache,
                rope_apply=rope_apply,
                attention_mask=attention_mask,
            )

        outputs: list[torch.Tensor] = []
        offset = 0
        for grid_h, grid_w, num_tokens, sp_perm in per_frame_meta:
            frame_TD = x_TD[offset : offset + num_tokens]
            offset += num_tokens
            outputs.append(self._finalize_image(frame_TD, grid_h, grid_w, sp_perm))
        return torch.cat(outputs)


__all__ = [
    "MuseSparkNaViTAdapter",
    "NaViTEncoder",
    "VisionRopeFreq",
    "reorder_patch_vector_channel_temporal",
]
