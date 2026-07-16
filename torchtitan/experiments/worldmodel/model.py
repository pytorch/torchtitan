# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import itertools
import math
from collections import OrderedDict
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torchtitan.config.configs import CompileConfig, ParallelismConfig, TrainingConfig
from torchtitan.config.configurable import Configurable
from torchtitan.models.common.attention import (
    create_attention_mask,
    FlexAttention,
    ScaledDotProductAttention,
)
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.nn_modules import (
    GELU,
    Identity,
    LayerNorm,
    Linear,
    RMSNorm,
    SiLU,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.tools.logging import logger


PLAN_SIZE = 15 * 33 * 2
PLAN_HEAD_INIT_STD = 1e-3
PLAN_HEAD_INIT_LOG_SIGMA_SCALE = 5.0
TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}
TensorOrMask = torch.Tensor | BlockMask


@dataclass(kw_only=True, slots=True)
class TransformerConfig:
    n_layer: int
    n_embd: int
    n_head: int
    act: str
    attn_pdrop: float
    resid_pdrop: float
    biased_linears: bool
    prenorm: bool
    qk_norm: bool
    mlp_mult: float
    mlp_multiple_of: int
    attention_mask: str
    norm: str
    attention_impl: str
    block_size: int = field(init=False, default=0)
    attention_mask_mini_block_size: int | None = field(init=False, default=None)


@dataclass(kw_only=True, slots=True)
class SelfAttentionLinearsConfig(Configurable.Config):
    c_attn: Linear.Config = field(default_factory=lambda: linear_config(1, 1))
    c_proj: Linear.Config = field(default_factory=lambda: linear_config(1, 1))


@dataclass(kw_only=True, slots=True)
class MLPLinearsConfig(Configurable.Config):
    c_fc: Linear.Config = field(default_factory=lambda: linear_config(1, 1))
    c_proj: Linear.Config = field(default_factory=lambda: linear_config(1, 1))


FFNLinearsConfig = MLPLinearsConfig


@dataclass(kw_only=True, slots=True)
class PatchEmbedderLinearsConfig(Configurable.Config):
    linear: Linear.Config = field(default_factory=lambda: linear_config(1, 1))


@dataclass(kw_only=True, slots=True)
class ConditioningEmbedderLinearsConfig(Configurable.Config):
    mlp_in: Linear.Config = field(default_factory=lambda: linear_config(1, 1))
    mlp_out: Linear.Config = field(default_factory=lambda: linear_config(1, 1))
    to_t6: Linear.Config = field(default_factory=lambda: linear_config(1, 1))
    to_t2: Linear.Config = field(default_factory=lambda: linear_config(1, 1))


@dataclass(kw_only=True, slots=True)
class DiTBlockLinearsConfig(Configurable.Config):
    attn: SelfAttentionLinearsConfig = field(default_factory=SelfAttentionLinearsConfig)
    mlp: FFNLinearsConfig = field(default_factory=MLPLinearsConfig)


@dataclass(kw_only=True, slots=True)
class FinalLayerLinearsConfig(Configurable.Config):
    linear: Linear.Config = field(default_factory=lambda: linear_config(1, 1))


@dataclass(kw_only=True, slots=True)
class PlanHeadLinearsConfig(Configurable.Config):
    blocks: list[FFNLinearsConfig] = field(default_factory=list)
    head: Linear.Config = field(default_factory=lambda: linear_config(1, 1))


def linear_config(
    in_features: int,
    out_features: int,
    *,
    bias: bool = False,
    current: Linear.Config | None = None,
) -> Linear.Config:
    if (
        current is not None
        and current.in_features == in_features
        and current.out_features == out_features
        and current.bias == bias
    ):
        return current
    return Linear.Config(in_features=in_features, out_features=out_features, bias=bias)


def self_attention_linears_config(
    config: TransformerConfig,
    current: SelfAttentionLinearsConfig | None = None,
) -> SelfAttentionLinearsConfig:
    return SelfAttentionLinearsConfig(
        c_attn=linear_config(
            config.n_embd,
            3 * config.n_embd,
            bias=config.biased_linears,
            current=None if current is None else current.c_attn,
        ),
        c_proj=linear_config(
            config.n_embd,
            config.n_embd,
            bias=config.biased_linears,
            current=None if current is None else current.c_proj,
        ),
    )


def ffn_linears_config(
    config: TransformerConfig,
    current: FFNLinearsConfig | None = None,
) -> FFNLinearsConfig:
    hidden = mlp_hidden_dim(config.n_embd, config.mlp_mult, config.mlp_multiple_of)
    return MLPLinearsConfig(
        c_fc=linear_config(
            config.n_embd,
            hidden,
            bias=config.biased_linears,
            current=None if current is None else current.c_fc,
        ),
        c_proj=linear_config(
            hidden,
            config.n_embd,
            bias=config.biased_linears,
            current=None if current is None else current.c_proj,
        ),
    )


def conditioning_embedder_linears_config(
    input_size: int,
    hidden_size: int,
    current: ConditioningEmbedderLinearsConfig | None = None,
) -> ConditioningEmbedderLinearsConfig:
    return ConditioningEmbedderLinearsConfig(
        mlp_in=linear_config(
            input_size,
            hidden_size,
            bias=True,
            current=None if current is None else current.mlp_in,
        ),
        mlp_out=linear_config(
            hidden_size,
            hidden_size,
            bias=True,
            current=None if current is None else current.mlp_out,
        ),
        to_t6=linear_config(
            hidden_size,
            6 * hidden_size,
            bias=True,
            current=None if current is None else current.to_t6,
        ),
        to_t2=linear_config(
            hidden_size,
            2 * hidden_size,
            bias=True,
            current=None if current is None else current.to_t2,
        ),
    )


def dit_block_linears_config(
    config: TransformerConfig,
    current: DiTBlockLinearsConfig | None = None,
) -> DiTBlockLinearsConfig:
    return DiTBlockLinearsConfig(
        attn=self_attention_linears_config(
            config, None if current is None else current.attn
        ),
        mlp=ffn_linears_config(config, None if current is None else current.mlp),
    )


def plan_head_linears_config(
    config: TransformerConfig,
    current: PlanHeadLinearsConfig | None = None,
) -> PlanHeadLinearsConfig:
    current_blocks = [] if current is None else current.blocks
    return PlanHeadLinearsConfig(
        blocks=[
            ffn_linears_config(
                config, current_blocks[i] if i < len(current_blocks) else None
            )
            for i in range(config.n_layer)
        ],
        head=linear_config(
            config.n_embd,
            PLAN_SIZE,
            bias=config.biased_linears,
            current=None if current is None else current.head,
        ),
    )


def make_norm(
    name: str, normalized_shape: int, *, elementwise_affine: bool = True
) -> nn.Module:
    if name == "LayerNorm":
        return LayerNorm.Config(
            normalized_shape=normalized_shape, elementwise_affine=elementwise_affine
        ).build()
    if name == "RMSNorm":
        return RMSNorm.Config(
            normalized_shape=normalized_shape, elementwise_affine=elementwise_affine
        ).build()
    raise ValueError(f"unknown norm {name}")


def make_activation(name: str) -> nn.Module:
    if name == "GELU":
        return GELU.Config(approximate="tanh").build()
    if name == "SiLU":
        return SiLU.Config().build()
    raise ValueError(f"unknown activation {name}")


def mlp_hidden_dim(n_embd: int, mlp_mult: float, mlp_multiple_of: int) -> int:
    hidden = int(n_embd * mlp_mult)
    return mlp_multiple_of * ((hidden + mlp_multiple_of - 1) // mlp_multiple_of)


def attn_flops(config: TransformerConfig) -> int:
    head_dim = config.n_embd // config.n_head
    return (
        12
        * config.n_layer
        * config.n_head
        * head_dim
        * config.block_size
        * config.block_size
    )


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _init_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    nn.init.normal_(_local_tensor(tensor), mean=mean, std=std)


def _init_constant_(tensor: torch.Tensor, value: float) -> None:
    nn.init.constant_(_local_tensor(tensor), value)


def _init_zeros_(tensor: torch.Tensor) -> None:
    nn.init.zeros_(_local_tensor(tensor))


def _init_xavier_uniform_(tensor: torch.Tensor) -> None:
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    with torch.no_grad():
        _local_tensor(tensor).uniform_(-bound, bound)


def _init_plan_bias_(bias: torch.Tensor) -> None:
    local = _local_tensor(bias)
    with torch.no_grad():
        local.zero_()
        split = PLAN_SIZE // 2
        if not isinstance(bias, DTensor):
            local[split:].fill_(math.log(PLAN_HEAD_INIT_LOG_SIGMA_SCALE))
            return
        offset = 0
        for mesh_dim, placement in enumerate(bias.placements):
            if placement.is_shard() and placement.dim == 0:
                _, offset = placement.local_shard_size_and_offset(
                    bias.shape[0],
                    bias.device_mesh.size(mesh_dim),
                    bias.device_mesh.get_local_rank(mesh_dim),
                )
                break
        start = max(0, split - offset)
        if start < local.shape[0]:
            local[start:].fill_(math.log(PLAN_HEAD_INIT_LOG_SIGMA_SCALE))


def init_transformer_linear_weights(linear: nn.Module) -> None:
    _init_xavier_uniform_(linear.weight)
    if linear.bias is not None:
        _init_zeros_(linear.bias)


def init_mlp_weights(mlp: nn.Module, std: float = 0.02) -> None:
    for module in mlp.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            _init_normal_(module.weight, std=std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                _init_constant_(module.bias, 0)


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    x = einops.rearrange(x, "b (t n) c -> b t n c", t=shift.shape[1])
    return einops.rearrange(x * (1 + scale) + shift, "b t n c -> b (t n) c")


def gate(x: torch.Tensor, gate_value: torch.Tensor) -> torch.Tensor:
    x = einops.rearrange(x, "b (t n) c -> b t n c", t=gate_value.shape[1])
    return einops.rearrange(gate_value * x, "b t n c -> b (t n) c")


def _cast_if_autocast_enabled(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_autocast_enabled():
        return tensor.to(dtype=torch.get_autocast_dtype(tensor.device.type))
    return tensor


def _blockwise_lower_triangular_causal_mask(
    mini_block_size: int,
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    del b, h
    return q_idx // mini_block_size >= kv_idx // mini_block_size


def _last_frame_causal_mask(
    block_size: int,
    mini_block_size: int,
    b: torch.Tensor,
    h: torch.Tensor,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
) -> torch.Tensor:
    del b, h
    q_ok = q_idx > block_size - mini_block_size
    kv_ok = kv_idx < block_size - mini_block_size
    return q_ok | kv_ok


def _dense_mask(mask_fn: Callable, q_len: int, kv_len: int) -> torch.Tensor:
    q_idx = torch.arange(q_len)
    kv_idx = torch.arange(kv_len)
    ii, jj = torch.meshgrid(q_idx, kv_idx, indexing="ij")
    return mask_fn(0, 0, ii, jj)


def _mask_fn(config: TransformerConfig) -> Callable | None:
    if config.attention_mask == "NONE":
        return None
    if config.attention_mask == "BLOCKWISE_LOWER_TRIANGLE":
        if config.attention_mask_mini_block_size is None:
            raise ValueError(
                "BLOCKWISE_LOWER_TRIANGLE requires attention_mask_mini_block_size"
            )
        return partial(
            _blockwise_lower_triangular_causal_mask,
            config.attention_mask_mini_block_size,
        )
    if config.attention_mask == "LAST_FRAME_CAUSAL":
        if config.attention_mask_mini_block_size is None:
            raise ValueError(
                "LAST_FRAME_CAUSAL requires attention_mask_mini_block_size"
            )
        return partial(
            _last_frame_causal_mask,
            config.block_size,
            config.attention_mask_mini_block_size,
        )
    raise ValueError(f"unknown attention_mask {config.attention_mask}")


def build_attention_mask(
    config: TransformerConfig, device: torch.device
) -> TensorOrMask | None:
    mask_fn = _mask_fn(config)
    if mask_fn is None:
        return None
    if config.attention_impl == "FLEX":
        if config.attn_pdrop > 0.0:
            raise NotImplementedError("FLEX attention does not support dropout")
        create_mask = (
            create_block_mask if device.type == "meta" else create_attention_mask
        )
        mask = create_mask(
            mask_fn,
            B=None,
            H=None,
            Q_LEN=config.block_size,
            KV_LEN=config.block_size,
            device=device,
        )
        mask.device = device
        return mask
    if config.attention_impl == "SDPA":
        return _dense_mask(mask_fn, config.block_size, config.block_size)[
            None, None
        ].to(device=device, dtype=torch.bool)
    raise ValueError(f"unknown attention_impl {config.attention_impl}")


class PatchEmbedder(nn.Sequential):
    def __init__(
        self,
        patch_size: tuple[int, int, int],
        linears: PatchEmbedderLinearsConfig,
        norm: str,
    ):
        super().__init__(
            Rearrange(
                "b (t pt) c (h ph) (w pw) -> b (t h w) (c pt ph pw)",
                pt=patch_size[0],
                ph=patch_size[1],
                pw=patch_size[2],
            ),
            linears.linear.build(),
            make_norm(norm, linears.linear.out_features),
        )
        self.init_weights()

    def init_weights(self) -> None:
        init_transformer_linear_weights(self[1])


class ContinuousEmbedder(nn.Module):
    def __init__(self, linears: ConditioningEmbedderLinearsConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            linears.mlp_in.build(), SiLU.Config().build(), linears.mlp_out.build()
        )
        self.to_t6 = nn.Sequential(SiLU.Config().build(), linears.to_t6.build())
        self.to_t2 = nn.Sequential(SiLU.Config().build(), linears.to_t2.build())
        self.init_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.mlp(x)
        return self.to_t6(x), self.to_t2(x)

    def init_weights(self) -> None:
        init_mlp_weights(self.mlp)
        init_mlp_weights(self.to_t6)
        init_mlp_weights(self.to_t2)


class DiscreteEmbedder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        linears: ConditioningEmbedderLinearsConfig,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            Embedding.Config(
                num_embeddings=input_size, embedding_dim=hidden_size
            ).build(),
            SiLU.Config().build(),
            linears.mlp_out.build(),
        )
        self.to_t6 = nn.Sequential(SiLU.Config().build(), linears.to_t6.build())
        self.to_t2 = nn.Sequential(SiLU.Config().build(), linears.to_t2.build())
        self.init_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.mlp(x)
        return self.to_t6(x), self.to_t2(x)

    def init_weights(self) -> None:
        init_mlp_weights(self.mlp)
        init_mlp_weights(self.to_t6)
        init_mlp_weights(self.to_t2)


class TimestepEmbedder(nn.Module):
    def __init__(
        self,
        linears: ConditioningEmbedderLinearsConfig,
        *,
        frequency_embedding_size: int = 256,
        max_period: int = 10000,
        time_factor: float = 1000.0,
    ):
        super().__init__()
        if frequency_embedding_size % 2 != 0:
            raise ValueError("frequency_embedding_size must be even")
        self.mlp = nn.Sequential(
            linears.mlp_in.build(), SiLU.Config().build(), linears.mlp_out.build()
        )
        self.to_t6 = nn.Sequential(SiLU.Config().build(), linears.to_t6.build())
        self.to_t2 = nn.Sequential(SiLU.Config().build(), linears.to_t2.build())
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.time_factor = time_factor
        self.init_weights()

    def timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half, device=t.device, dtype=torch.float32)
            / half
        )
        args = self.time_factor * t.float()[..., None] * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t_emb = self.mlp(self.timestep_embedding(t).to(self.mlp[0].weight.dtype))
        return self.to_t6(t_emb), self.to_t2(t_emb)

    def init_weights(self) -> None:
        init_mlp_weights(self.mlp)
        init_mlp_weights(self.to_t6)
        init_mlp_weights(self.to_t2)


class ScaleLayer(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.empty((n_features,)))
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    def reset_parameters(self) -> None:
        _init_constant_(self.scale, 1.0)

    def init_weights(self) -> None:
        self.reset_parameters()


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig, linears: SelfAttentionLinearsConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.config = config
        self.head_dim = config.n_embd // config.n_head
        self.layer_norm = (
            make_norm(config.norm, config.n_embd)
            if config.prenorm
            else Identity.Config().build()
        )
        self.q_norm = (
            make_norm(config.norm, self.head_dim)
            if config.qk_norm
            else Identity.Config().build()
        )
        self.k_norm = (
            make_norm(config.norm, self.head_dim)
            if config.qk_norm
            else Identity.Config().build()
        )
        self.c_attn = linears.c_attn.build()
        self.c_proj = linears.c_proj.build()
        self.dropout = nn.Dropout(config.resid_pdrop)
        self.flex_attention = (
            FlexAttention.Config().build() if config.attention_impl == "FLEX" else None
        )
        self.sdpa = (
            ScaledDotProductAttention.Config().build()
            if config.attention_impl == "SDPA"
            else None
        )
        self.kv_cache: Any | None = None

    def forward(
        self, x: torch.Tensor, input_mask: TensorOrMask | None = None
    ) -> torch.Tensor:
        batch, seq_len, emb_dim = x.shape
        qkv = self.c_attn(self.layer_norm(x)).view(
            batch, seq_len, 3, self.config.n_head, self.head_dim
        )
        q, k, v = qkv.unbind(2)
        q, k = _cast_if_autocast_enabled(self.q_norm(q)), _cast_if_autocast_enabled(
            self.k_norm(k)
        )

        if self.config.attention_impl == "FLEX":
            assert self.flex_attention is not None
            y = self.flex_attention(
                q,
                k,
                v,
                attention_masks=input_mask,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        elif self.config.attention_impl == "SDPA" and input_mask is None:
            assert self.sdpa is not None
            y = self.sdpa(
                q, k, v, scale=1.0 / math.sqrt(self.head_dim), is_causal=False
            )
        elif self.config.attention_impl == "SDPA":
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=input_mask,
                dropout_p=self.config.attn_pdrop if self.training else 0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            ).transpose(1, 2)
        else:
            raise ValueError(f"unknown attention_impl {self.config.attention_impl}")
        return self.dropout(self.c_proj(y.reshape(batch, seq_len, emb_dim)))


def build_ffn(config: TransformerConfig, linears: FFNLinearsConfig) -> nn.Sequential:
    return nn.Sequential(
        OrderedDict(
            {
                "layer_norm": make_norm(config.norm, config.n_embd)
                if config.prenorm
                else Identity.Config().build(),
                "c_fc": linears.c_fc.build(),
                "act": make_activation(config.act),
                "c_proj": linears.c_proj.build(),
                "dropout": nn.Dropout(config.resid_pdrop),
            }
        )
    )


class ResidualSequential(nn.Sequential):
    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        return super().forward(input) + input


def residual_ffn(
    config: TransformerConfig, linears: FFNLinearsConfig
) -> ResidualSequential:
    return ResidualSequential(OrderedDict(build_ffn(config, linears).named_children()))


class PlanHead(nn.Module):
    def __init__(self, config: "WorldModel.Config", linears: PlanHeadLinearsConfig):
        super().__init__()
        self.mlps = nn.ModuleList(
            residual_ffn(config.plan_head, linears.blocks[i])
            for i in range(config.plan_head.n_layer)
        )
        self.head = linears.head.build()
        self.scale_layer = ScaleLayer(PLAN_SIZE)
        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for mlp in self.mlps:
            x = mlp(x)
        return self.scale_layer(self.head(x))

    def init_weights(self) -> None:
        for module in self.mlps.modules():
            if isinstance(module, nn.Linear):
                init_transformer_linear_weights(module)
        _init_normal_(self.head.weight, std=PLAN_HEAD_INIT_STD)
        if self.head.bias is not None:
            _init_plan_bias_(self.head.bias)
        self.scale_layer.init_weights()


class DiTBlock(nn.Module):
    def __init__(self, config: "WorldModel.Config", linears: DiTBlockLinearsConfig):
        super().__init__()
        self.norm1 = make_norm(
            config.transformer.norm, config.transformer.n_embd, elementwise_affine=False
        )
        self.attn = SelfAttention(config.transformer, linears.attn)
        self.norm2 = make_norm(
            config.transformer.norm, config.transformer.n_embd, elementwise_affine=False
        )
        self.mlp = build_ffn(config.transformer, linears.mlp)
        self.scale_shift_table = nn.Parameter(
            torch.empty(1, config.num_temporal_patches, 6, config.transformer.n_embd)
        )
        self.init_weights()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        input_pos: torch.Tensor | None = None,
        input_pos_t: torch.Tensor | None = None,
        input_mask: TensorOrMask | None = None,
    ) -> torch.Tensor:
        batch = x.shape[0]
        scale_shift_table = (
            self.scale_shift_table
            if input_pos_t is None
            else self.scale_shift_table[:, input_pos_t]
        )
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            scale_shift_table + t.reshape(batch, scale_shift_table.shape[1], 6, -1)
        ).chunk(6, dim=2)
        attn_input = modulate(self.norm1(x), shift_msa, scale_msa)
        if input_pos is None:
            attn_output = self.attn(attn_input, input_mask=input_mask)
        else:
            attn_output = self.attn(
                attn_input, input_pos=input_pos, input_mask=input_mask
            )
        x = x + gate(attn_output, gate_msa)
        return x + gate(
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)), gate_mlp
        )

    def reset_parameters(self) -> None:
        _init_normal_(
            self.scale_shift_table,
            mean=0.0,
            std=self.scale_shift_table.shape[-1] ** -0.5,
        )

    def init_weights(self) -> None:
        self.reset_parameters()
        for module in itertools.chain(self.attn.modules(), self.mlp.modules()):
            if isinstance(module, nn.Linear):
                init_transformer_linear_weights(module)


class FinalLayer(nn.Module):
    def __init__(self, config: "WorldModel.Config", linears: FinalLayerLinearsConfig):
        super().__init__()
        self.norm_final = make_norm(
            config.transformer.norm, config.transformer.n_embd, elementwise_affine=False
        )
        self.linear = linears.linear.build()
        self.scale_shift_table = nn.Parameter(
            torch.empty(1, config.num_temporal_patches, 2, config.transformer.n_embd)
        )
        self.init_weights()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        input_pos_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = x.shape[0]
        scale_shift_table = (
            self.scale_shift_table
            if input_pos_t is None
            else self.scale_shift_table[:, input_pos_t]
        )
        shift, scale = (
            scale_shift_table + t.reshape(batch, scale_shift_table.shape[1], 2, -1)
        ).chunk(2, dim=2)
        return self.linear(modulate(self.norm_final(x), shift, scale))

    def reset_parameters(self) -> None:
        _init_normal_(
            self.scale_shift_table,
            mean=0.0,
            std=self.scale_shift_table.shape[-1] ** -0.5,
        )

    def init_weights(self) -> None:
        self.reset_parameters()
        _init_constant_(self.linear.weight, 0)
        _init_constant_(self.linear.bias, 0)


class WorldModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        input_size: tuple[int, int, int]
        patch_size: tuple[int, int, int]
        in_channels: int
        out_channels: int
        pose_size: int
        time_factor: float
        compressor_mean: float
        compressor_std: float
        transformer: TransformerConfig
        plan_head: TransformerConfig
        experimental_pose_only_xy: bool
        x_embedder: PatchEmbedderLinearsConfig = field(init=False)
        augments_pos_ref_augment_embedder: ConditioningEmbedderLinearsConfig = field(
            init=False
        )
        ref_augment_from_augments_euler_embedder: ConditioningEmbedderLinearsConfig = (
            field(init=False)
        )
        pose_mask_embedder: ConditioningEmbedderLinearsConfig = field(init=False)
        t_embedder: ConditioningEmbedderLinearsConfig = field(init=False)
        fidx_embedder: ConditioningEmbedderLinearsConfig = field(init=False)
        blocks: list[DiTBlockLinearsConfig] = field(init=False)
        final_layer: FinalLayerLinearsConfig | None = field(init=False)
        plan_head_linears: PlanHeadLinearsConfig | None = field(init=False)

        @property
        def num_spatial_patches(self) -> int:
            return math.prod(self.input_size[1:]) // math.prod(self.patch_size[1:])

        @property
        def num_temporal_patches(self) -> int:
            return self.input_size[0] // self.patch_size[0]

        @property
        def num_patches(self) -> int:
            return self.num_spatial_patches * self.num_temporal_patches

        def __post_init__(self) -> None:
            self._sync_derived_fields()

        def _sync_derived_fields(self) -> None:
            self.transformer.block_size = self.num_patches
            self.transformer.attention_mask_mini_block_size = self.num_spatial_patches
            self.plan_head.n_embd = self.transformer.n_embd
            hidden = self.transformer.n_embd
            pose_half = self.pose_size // 2
            current_blocks = getattr(self, "blocks", [])
            current_final = getattr(self, "final_layer", None)
            current_plan = getattr(self, "plan_head_linears", None)

            self.x_embedder = PatchEmbedderLinearsConfig(
                linear=linear_config(
                    self.in_channels * math.prod(self.patch_size),
                    hidden,
                    current=getattr(getattr(self, "x_embedder", None), "linear", None),
                )
            )
            self.augments_pos_ref_augment_embedder = (
                conditioning_embedder_linears_config(
                    pose_half,
                    hidden,
                    getattr(self, "augments_pos_ref_augment_embedder", None),
                )
            )
            self.ref_augment_from_augments_euler_embedder = (
                conditioning_embedder_linears_config(
                    pose_half,
                    hidden,
                    getattr(self, "ref_augment_from_augments_euler_embedder", None),
                )
            )
            self.pose_mask_embedder = conditioning_embedder_linears_config(
                2, hidden, getattr(self, "pose_mask_embedder", None)
            )
            self.t_embedder = conditioning_embedder_linears_config(
                256, hidden, getattr(self, "t_embedder", None)
            )
            self.fidx_embedder = conditioning_embedder_linears_config(
                50, hidden, getattr(self, "fidx_embedder", None)
            )
            self.blocks = [
                dit_block_linears_config(
                    self.transformer,
                    current_blocks[i] if i < len(current_blocks) else None,
                )
                for i in range(self.transformer.n_layer)
            ]
            self.final_layer = (
                FinalLayerLinearsConfig(
                    linear=linear_config(
                        hidden,
                        math.prod(self.patch_size) * self.out_channels,
                        bias=True,
                        current=None if current_final is None else current_final.linear,
                    )
                )
                if self.out_channels > 0
                else None
            )
            self.plan_head_linears = (
                plan_head_linears_config(self.plan_head, current_plan)
                if self.plan_head.n_layer >= 0
                else None
            )

        def build(self, **kwargs: Any) -> "WorldModel":
            if kwargs:
                raise ValueError("WorldModel.Config.build does not accept kwargs")
            if self._owner is None:
                raise NotImplementedError("WorldModel.Config has no owner class")
            self._sync_derived_fields()
            return self._owner(config=copy(self))

        def update_from_config(self, *, config: Any, **kwargs: Any) -> None:
            del kwargs
            if config.parallelism.spmd_backend == "full_dtensor":
                raise ValueError("worldmodel does not support full DTensor")
            unsupported = {
                "tensor parallel": config.parallelism.tensor_parallel_degree,
                "context parallel": config.parallelism.context_parallel_degree,
                "pipeline parallel": config.parallelism.pipeline_parallel_degree,
                "expert parallel": config.parallelism.expert_parallel_degree,
            }
            for name, degree in unsupported.items():
                if degree > 1:
                    raise ValueError(f"worldmodel supports FSDP/HSDP only, not {name}")
            self._sync_derived_fields()
            config.training.seq_len = self.num_patches

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            del seq_len
            nparams = sum(p.numel() for p in model.parameters())
            return nparams, 6 * nparams + attn_flops(self.transformer) // max(
                1, self.num_patches
            )

    def __init__(self, config: Config):
        super().__init__()
        config._sync_derived_fields()
        self.config = config
        self.x_embedder = PatchEmbedder(
            config.patch_size, config.x_embedder, config.transformer.norm
        )
        pose_half = config.pose_size // 2
        self.position_scale = ScaleLayer(pose_half)
        self.euler_scale = ScaleLayer(pose_half)
        self.augments_pos_ref_augment_embedder = ContinuousEmbedder(
            config.augments_pos_ref_augment_embedder
        )
        self.ref_augment_from_augments_euler_embedder = ContinuousEmbedder(
            config.ref_augment_from_augments_euler_embedder
        )
        self.pose_mask_embedder = DiscreteEmbedder(
            2, config.transformer.n_embd, config.pose_mask_embedder
        )
        self.t_embedder = TimestepEmbedder(
            config.t_embedder, time_factor=config.time_factor
        )
        self.fidx_embedder = DiscreteEmbedder(
            50, config.transformer.n_embd, config.fidx_embedder
        )
        self.blocks = nn.ModuleList(
            DiTBlock(config, config.blocks[i])
            for i in range(config.transformer.n_layer)
        )
        self.final_layer = (
            FinalLayer(config, config.final_layer)
            if config.final_layer is not None
            else None
        )
        self.plan_head = (
            PlanHead(config, config.plan_head_linears)
            if config.plan_head_linears is not None
            else None
        )
        self.register_buffer(
            "pos_embed", torch.empty(1, config.num_patches, config.transformer.n_embd)
        )
        self.mask: TensorOrMask | None = None
        self.init_states(buffer_device=self.pos_embed.device)

    def verify_module_protocol(self) -> None:
        pass

    @torch.no_grad()
    def setup_attention_attrs(self, device: torch.device) -> None:
        if self.config.transformer.attention_mask == "NONE":
            self.mask = None
            return
        if self.mask is not None and getattr(self.mask, "device", None) == device:
            return
        self.mask = build_attention_mask(self.config.transformer, device)

    def reset_parameters(self) -> None:
        spatial_grid = (
            self.config.input_size[1] // self.config.patch_size[1],
            self.config.input_size[2] // self.config.patch_size[2],
        )
        spatial = torch.from_numpy(
            get_2d_sincos_pos_embed(self.pos_embed.shape[-1], spatial_grid)
        )
        spatial = spatial.to(
            dtype=self.pos_embed.dtype, device=self.pos_embed.device
        ).unsqueeze(0)
        spatial = einops.repeat(
            spatial, "() n d -> () (t n) d", t=self.config.num_temporal_patches
        )
        temporal = torch.from_numpy(
            get_1d_sincos_pos_embed(
                self.pos_embed.shape[-1], self.config.num_temporal_patches
            )
        )
        temporal = temporal.to(
            dtype=self.pos_embed.dtype, device=self.pos_embed.device
        ).unsqueeze(0)
        temporal = einops.repeat(
            temporal, "() t d -> () (t n) d", n=self.config.num_spatial_patches
        )
        self.pos_embed[:] = spatial + temporal

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device or self.pos_embed.device
        if isinstance(device, str):
            device = torch.device(device)
        self.setup_attention_attrs(device)

        def reset(module: nn.Module) -> None:
            module = getattr(module, "_checkpoint_wrapped_module", module)
            reset_parameters = getattr(module, "reset_parameters", None)
            if callable(reset_parameters):
                reset_parameters()

        self.apply(reset)

        def init(module: nn.Module) -> None:
            module = getattr(module, "_checkpoint_wrapped_module", module)
            init_weights = getattr(module, "init_weights", None)
            if callable(init_weights):
                init_weights()

        for module in self.children():
            if isinstance(module, nn.ModuleList):
                for child in module:
                    init(child)
            else:
                init(module)

    def scale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.config.compressor_mean) / self.config.compressor_std

    def unscale_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.config.compressor_std + self.config.compressor_mean

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        return einops.rearrange(
            x,
            "b (t h w) (c pt ph pw) -> b (t pt) c (h ph) (w pw)",
            c=self.config.out_channels,
            pt=self.config.patch_size[0],
            ph=self.config.patch_size[1],
            pw=self.config.patch_size[2],
            h=self.config.input_size[1] // self.config.patch_size[1],
            w=self.config.input_size[2] // self.config.patch_size[2],
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        augments_pos_ref_augment: torch.Tensor,
        ref_augment_from_augments_euler: torch.Tensor,
        pose_mask: torch.Tensor,
        fidx: torch.Tensor,
        return_plan: bool = True,
        input_pos_mask_pair: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        if input_pos_mask_pair is None:
            input_pos, input_mask = None, self.mask
        else:
            input_pos, input_mask = (
                input_pos_mask_pair.input_pos,
                input_pos_mask_pair.input_mask,
            )

        if self.config.experimental_pose_only_xy:
            augments_pos_ref_augment = augments_pos_ref_augment * (
                augments_pos_ref_augment.new_tensor((1.0, 1.0, 0.0))
            )
            ref_augment_from_augments_euler = ref_augment_from_augments_euler * (
                ref_augment_from_augments_euler.new_tensor((0.0, 0.0, 1.0))
            )
        pos_embed = (
            self.pos_embed[:, input_pos] if input_pos is not None else self.pos_embed
        )
        input_pos_t = (
            input_pos[:: self.config.num_spatial_patches]
            // self.config.num_spatial_patches
            if input_pos is not None
            else None
        )

        x = self.x_embedder(x) + pos_embed
        augments_pos_ref_augment = self.position_scale(augments_pos_ref_augment)
        ref_augment_from_augments_euler = self.euler_scale(
            ref_augment_from_augments_euler
        )
        t6, t2 = self.t_embedder(t)
        pos6, pos2 = self.augments_pos_ref_augment_embedder(augments_pos_ref_augment)
        euler6, euler2 = self.ref_augment_from_augments_euler_embedder(
            ref_augment_from_augments_euler
        )
        pose_mask6, pose_mask2 = self.pose_mask_embedder(pose_mask)
        fidx6, fidx2 = self.fidx_embedder(fidx)
        t6 = t6 + pos6 + euler6 + pose_mask6 + fidx6
        t2 = t2 + pos2 + euler2 + pose_mask2 + fidx2
        for block in self.blocks:
            x = block(x, t6, input_pos, input_pos_t, input_mask)
        outputs = {}
        if return_plan and self.plan_head is not None:
            outputs["plan"] = self.plan_head(x[:, -1, :])
        if self.final_layer is not None:
            outputs["sample"] = self.unpatchify(self.final_layer(x, t2, input_pos_t))
        return outputs


def parallelize_worldmodel(
    model: WorldModel,
    *,
    parallel_dims: Any,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: Any,
    dump_folder: str,
) -> WorldModel:
    if parallelism.spmd_backend == "full_dtensor":
        raise ValueError("worldmodel does not support full DTensor")
    if (
        parallel_dims.tp_enabled
        or parallel_dims.pp_enabled
        or parallel_dims.cp_enabled
        or parallel_dims.ep_enabled
    ):
        raise ValueError("worldmodel supports FSDP/HSDP only")

    if ac_config is not None:
        _apply_activation_checkpointing(model, ac_config, dump_folder=dump_folder)
    if compile_config.enable and "model" in compile_config.components:
        _apply_compile(model, compile_config)

    dp_mesh = parallel_dims.get_activated_mesh(["dp_replicate", "fsdp"])
    if dp_mesh is None:
        dp_mesh = parallel_dims.get_mesh("fsdp")
    _apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )
    logger.info(
        "Applied HSDP to the worldmodel"
        if parallel_dims.dp_replicate_enabled
        else "Applied FSDP to the worldmodel"
    )
    return model


def _apply_activation_checkpointing(
    model: WorldModel,
    ac_config: Any,
    *,
    dump_folder: str,
) -> None:
    from torchtitan.distributed.activation_checkpoint import FullAC, MemoryBudgetAC

    assert ac_config is not None
    ac_policy = ac_config.build(dump_folder=dump_folder)
    if isinstance(ac_policy, MemoryBudgetAC):
        raise ValueError(
            "worldmodel does not support memory-budget activation checkpointing"
        )

    def wrap(module: nn.Module, fqn: str) -> nn.Module:
        return ac_policy._wrap_block(module, base_fqn=fqn)

    mode = "full" if isinstance(ac_policy, FullAC) else "selective"

    for layer_id, block in model.blocks.named_children():
        model.blocks.register_module(
            layer_id,
            wrap(block, f"blocks.{layer_id}"),
        )
    if model.plan_head is not None:
        for layer_id, block in model.plan_head.mlps.named_children():
            model.plan_head.mlps.register_module(
                layer_id,
                wrap(block, f"plan_head.mlps.{layer_id}"),
            )
    logger.info(f"Applied {mode} activation checkpointing to the worldmodel")


def _apply_compile(model: WorldModel, compile_config: CompileConfig) -> None:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    for module in (
        model.x_embedder,
        model.position_scale,
        model.euler_scale,
        model.augments_pos_ref_augment_embedder,
        model.ref_augment_from_augments_euler_embedder,
        model.pose_mask_embedder,
        model.t_embedder,
        model.fidx_embedder,
    ):
        module.compile(backend=compile_config.backend, fullgraph=True)
    for block in model.blocks:
        block.compile(backend=compile_config.backend, fullgraph=True)
    if model.final_layer is not None:
        model.final_layer.compile(backend=compile_config.backend, fullgraph=True)
    if model.plan_head is not None:
        for block in model.plan_head.mlps:
            block.compile(backend=compile_config.backend, fullgraph=True)
    logger.info("Compiling worldmodel components with torch.compile")


def _apply_fsdp(
    model: WorldModel,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool,
    reshard_after_forward_policy: str,
    enable_symm_mem: bool,
) -> None:
    from torch.distributed.fsdp import (
        CPUOffloadPolicy,
        fully_shard,
        MixedPrecisionPolicy,
    )

    from torchtitan.distributed.fsdp import (
        enable_fsdp_symm_mem,
        get_fsdp_reshard_after_forward_policy,
    )

    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=True,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy,
        pp_enabled,
    )

    for module in (
        model.x_embedder,
        model.position_scale,
        model.euler_scale,
        model.augments_pos_ref_augment_embedder,
        model.ref_augment_from_augments_euler_embedder,
        model.pose_mask_embedder,
        model.t_embedder,
        model.fidx_embedder,
    ):
        fully_shard(module, **fsdp_config, reshard_after_forward=reshard_after_forward)

    for block in model.blocks:
        fully_shard(block, **fsdp_config, reshard_after_forward=reshard_after_forward)
    if model.plan_head is not None:
        for block in model.plan_head.mlps:
            fully_shard(
                block, **fsdp_config, reshard_after_forward=reshard_after_forward
            )
        fully_shard(
            model.plan_head, **fsdp_config, reshard_after_forward=reshard_after_forward
        )
    if model.final_layer is not None:
        fully_shard(
            model.final_layer,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )
    fully_shard(model, **fsdp_config)
    if enable_symm_mem:
        enable_fsdp_symm_mem(model)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: tuple[int, int]) -> np.ndarray:
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, *grid_size])
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def get_1d_sincos_pos_embed(embed_dim: int, length: int) -> np.ndarray:
    return get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(0, length)[..., None])


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    return np.concatenate(
        [
            get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]),
            get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]),
        ],
        axis=1,
    )


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    out = np.einsum("m,d->md", pos.reshape(-1), 1.0 / 10000**omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def main() -> None:
    from torchtitan.components.checkpoint import CheckpointManager

    from torchtitan.experiments.worldmodel.model_config import model_registry
    from torchtitan.tools.logging import init_logger

    parser = argparse.ArgumentParser(description="Create a tiny worldmodel checkpoint.")
    parser.add_argument("--output-dir", default="./outputs/worldmodel_debug_checkpoint")
    parser.add_argument("--flavor", default="debugmodel")
    args = parser.parse_args()

    init_logger()
    model = model_registry(args.flavor).model.build().eval()
    checkpointer = CheckpointManager.Config(
        enable=True,
        folder="checkpoint",
        interval=1,
        async_mode="disabled",
        keep_latest_k=0,
        last_save_model_only=True,
        checkpoint_id_format="",
    ).build(
        dataloader=None,
        model_parts=[model],
        optimizers=None,
        lr_schedulers=None,
        states={},
        sd_adapter=None,
        base_folder=args.output_dir,
    )
    try:
        checkpointer.save(curr_step=0, last_step=True)
    finally:
        checkpointer.close()
    print({"checkpoint_path": f"{args.output_dir}/checkpoint/0", "flavor": args.flavor})


if __name__ == "__main__":
    main()
