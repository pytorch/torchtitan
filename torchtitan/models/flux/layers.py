# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# imported from black-forest-labs/FLUX
# Shape suffixes: B = batch, L = per-sample sequence, T = packed tokens,
# D = hidden dimension, H = heads, K = query/key head dimension, V = value dimension.
import math
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
from einops import rearrange
from torch import nn, Tensor

from torchtitan.models.base.module import Module, Sequential

from torchtitan.models.common.activation import SiLU
from torchtitan.models.common.attention import (
    VarlenAttentionMetadata,
    VarlenInnerAttention,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GELU, LayerNorm, RMSNorm

from .sharding import gather_flux_kv


@spmd.local_map(
    in_types=(spmd.PartitionSpec("dp", "cp", None),),
    out_types=spmd.PartitionSpec(("dp", "cp"), None),
)
def local_pack(x_BLD: Tensor) -> Tensor:
    return x_BLD.flatten(0, 1)


@spmd.local_map(
    in_types=(spmd.PartitionSpec(("dp", "cp"), None), None),
    out_types=spmd.PartitionSpec("dp", "cp", None),
)
def local_unpack(x_TD: Tensor, batch_size: int) -> Tensor:
    return x_TD.unflatten(0, (batch_size, -1))


@spmd.local_map(
    in_types=(
        spmd.PartitionSpec(("dp", "cp"), None),
        spmd.PartitionSpec(("dp", "cp"), None),
        None,
    ),
    out_types=spmd.PartitionSpec(("dp", "cp"), None),
)
def local_concat_text_image(
    text_TD: Tensor, image_TD: Tensor, batch_size: int
) -> Tensor:
    text_BLD = text_TD.unflatten(0, (batch_size, -1))
    image_BLD = image_TD.unflatten(0, (batch_size, -1))
    return torch.cat((text_BLD, image_BLD), dim=1).flatten(0, 1)


@spmd.local_map(
    in_types=(spmd.PartitionSpec(("dp", "cp"), None), None, None),
    out_types=(
        spmd.PartitionSpec(("dp", "cp"), None),
        spmd.PartitionSpec(("dp", "cp"), None),
    ),
)
def local_split_text_image(
    combined_TD: torch.Tensor,
    text_seq_len: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_BLD = combined_TD.unflatten(0, (batch_size, -1))
    return (
        combined_BLD[:, :text_seq_len].flatten(0, 1),
        combined_BLD[:, text_seq_len:].flatten(0, 1),
    )


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "... d (i j) -> ... d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class EmbedND(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        theta: int
        axes_dim: tuple

    def __init__(self, config: Config):
        super().__init__()
        self.dim = config.dim
        self.theta = config.theta
        self.axes_dim = config.axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(-4)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Creates sinusoidal timestep embeddings.

    Args:
        t (Tensor): A 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): The dimension of the output.
        max_period (int, optional): Controls the minimum frequency of the embeddings. Default is 10000.
        time_factor (float, optional): Scaling factor for time. Default is 1000.0.

    Returns:
        Tensor: An (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    with torch.device(t.device):
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_layer: Linear.Config
        out_layer: Linear.Config
        in_dim: int
        hidden_dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.in_layer = config.in_layer.build()
        self.silu = SiLU.Config().build()
        self.out_layer = config.out_layer.build()

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class QKNorm(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        query_norm: RMSNorm.Config
        key_norm: RMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.query_norm = config.query_norm.build()
        self.key_norm = config.key_norm.build()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        qkv: Linear.Config
        proj: Linear.Config
        norm: QKNorm.Config
        num_heads: int = 8
        qkv_bias: bool = False
        inner_attention: VarlenInnerAttention.Config = field(
            default_factory=lambda: VarlenInnerAttention.Config(window_size=(-1, -1))
        )

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.qkv = config.qkv.build()
        self.norm = config.norm.build()
        self.proj = config.proj.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: Tensor,
        pe: Tensor,
        *,
        attention_metadata: VarlenAttentionMetadata,
    ) -> Tensor:
        qkv_TD = self.qkv(x_TD)
        q_THK, k_THK, v_THV = rearrange(
            qkv_TD, "T (Q H K) -> Q T H K", Q=3, H=self.num_heads
        )
        q_THK, k_THK = self.norm(q_THK, k_THK, v_THV)
        q_THK, k_THK = apply_rope(q_THK, k_THK, pe)
        k_THK, v_THV = gather_flux_kv(
            k_THK, v_THV, batch_size=attention_metadata.cu_seq_q.shape[0] - 1
        )
        out_THV = self.inner_attention(
            q_THK, k_THK, v_THV, attention_metadata=attention_metadata
        )
        return self.proj(out_THV.flatten(1))


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        lin: Linear.Config
        double: bool

    def __init__(self, config: Config):
        super().__init__()
        self.is_double = config.double
        self.multiplier = 6 if config.double else 3
        self.lin = config.lin.build()

    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(
            self.multiplier, dim=-1
        )

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        img_mod: Modulation.Config
        txt_mod: Modulation.Config
        img_attn: SelfAttention.Config
        txt_attn: SelfAttention.Config
        img_mlp_in: Linear.Config
        img_mlp_out: Linear.Config
        txt_mlp_in: Linear.Config
        txt_mlp_out: Linear.Config
        mlp_ratio: float = 4.0
        qkv_bias: bool = False
        inner_attention: VarlenInnerAttention.Config = field(
            default_factory=lambda: VarlenInnerAttention.Config(window_size=(-1, -1))
        )

    def __init__(self, config: Config):
        super().__init__()

        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        ln_cfg = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        )
        gelu_cfg = GELU.Config(approximate="tanh")

        self.img_mod = config.img_mod.build()
        self.img_norm1 = ln_cfg.build()
        self.img_attn = config.img_attn.build()

        self.img_norm2 = ln_cfg.build()
        self.img_mlp = Sequential(
            config.img_mlp_in.build(),
            gelu_cfg.build(),
            config.img_mlp_out.build(),
        )

        self.txt_mod = config.txt_mod.build()
        self.txt_norm1 = ln_cfg.build()
        self.txt_attn = config.txt_attn.build()

        self.txt_norm2 = ln_cfg.build()
        self.txt_mlp = Sequential(
            config.txt_mlp_in.build(),
            gelu_cfg.build(),
            config.txt_mlp_out.build(),
        )

        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        img_TD: Tensor,
        txt_TD: Tensor,
        vec_BD: Tensor,
        pe: Tensor,
        *,
        attention_metadata: VarlenAttentionMetadata,
    ) -> tuple[Tensor, Tensor]:
        batch_size = vec_BD.shape[0]
        img_mod1, img_mod2 = self.img_mod(vec_BD)
        txt_mod1, txt_mod2 = self.txt_mod(vec_BD)
        assert txt_mod2 is not None
        assert img_mod2 is not None

        img_modulated_BLD = local_unpack(self.img_norm1(img_TD), batch_size)
        img_modulated_TD = local_pack(
            (1 + img_mod1.scale) * img_modulated_BLD + img_mod1.shift
        )
        img_qkv_TD = self.img_attn.qkv(img_modulated_TD)
        img_q_THK, img_k_THK, img_v_THV = rearrange(
            img_qkv_TD, "T (Q H K) -> Q T H K", Q=3, H=self.num_heads
        )
        img_q_THK, img_k_THK = self.img_attn.norm(img_q_THK, img_k_THK, img_v_THV)

        txt_modulated_BLD = local_unpack(self.txt_norm1(txt_TD), batch_size)
        txt_modulated_TD = local_pack(
            (1 + txt_mod1.scale) * txt_modulated_BLD + txt_mod1.shift
        )
        txt_qkv_TD = self.txt_attn.qkv(txt_modulated_TD)
        txt_q_THK, txt_k_THK, txt_v_THV = rearrange(
            txt_qkv_TD, "T (Q H K) -> Q T H K", Q=3, H=self.num_heads
        )
        txt_q_THK, txt_k_THK = self.txt_attn.norm(txt_q_THK, txt_k_THK, txt_v_THV)

        q_THK, k_THK, v_THV = (
            local_concat_text_image(
                text.flatten(1), image.flatten(1), batch_size
            ).unflatten(1, (self.num_heads, -1))
            for text, image in (
                (txt_q_THK, img_q_THK),
                (txt_k_THK, img_k_THK),
                (txt_v_THV, img_v_THV),
            )
        )
        q_THK, k_THK = apply_rope(q_THK, k_THK, pe)
        k_THK, v_THV = gather_flux_kv(k_THK, v_THV, batch_size=batch_size)
        attn_THV = self.inner_attention(
            q_THK, k_THK, v_THV, attention_metadata=attention_metadata
        )
        txt_attn_TD, img_attn_TD = local_split_text_image(
            attn_THV.flatten(1), txt_TD.shape[0] // batch_size, batch_size
        )

        img_TD = img_TD + local_pack(
            img_mod1.gate * local_unpack(self.img_attn.proj(img_attn_TD), batch_size)
        )
        img_mlp_TD = self.img_mlp(
            local_pack(
                (1 + img_mod2.scale) * local_unpack(self.img_norm2(img_TD), batch_size)
                + img_mod2.shift
            )
        )
        img_TD = img_TD + local_pack(
            img_mod2.gate * local_unpack(img_mlp_TD, batch_size)
        )

        txt_TD = txt_TD + local_pack(
            txt_mod1.gate * local_unpack(self.txt_attn.proj(txt_attn_TD), batch_size)
        )
        txt_mlp_TD = self.txt_mlp(
            local_pack(
                (1 + txt_mod2.scale) * local_unpack(self.txt_norm2(txt_TD), batch_size)
                + txt_mod2.shift
            )
        )
        txt_TD = txt_TD + local_pack(
            txt_mod2.gate * local_unpack(txt_mlp_TD, batch_size)
        )
        return img_TD, txt_TD


class SingleStreamBlock(Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        linear1: Linear.Config
        linear2: Linear.Config
        modulation: Modulation.Config
        norm: QKNorm.Config
        mlp_ratio: float = 4.0
        qk_scale: float | None = None
        inner_attention: VarlenInnerAttention.Config = field(
            default_factory=lambda: VarlenInnerAttention.Config(window_size=(-1, -1))
        )

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = config.qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(config.hidden_size * config.mlp_ratio)
        # qkv and mlp_in
        self.linear1 = config.linear1.build()
        # proj and mlp_out
        self.linear2 = config.linear2.build()

        self.norm = config.norm.build()

        self.hidden_size = config.hidden_size
        self.pre_norm = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        ).build()

        self.mlp_act = GELU.Config(approximate="tanh").build()
        self.modulation = config.modulation.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: Tensor,
        vec_BD: Tensor,
        pe: Tensor,
        *,
        attention_metadata: VarlenAttentionMetadata,
    ) -> Tensor:
        batch_size = vec_BD.shape[0]
        mod, _ = self.modulation(vec_BD)
        x_mod_TD = local_pack(
            (1 + mod.scale) * local_unpack(self.pre_norm(x_TD), batch_size) + mod.shift
        )
        qkv_TD, mlp_TD = torch.split(
            self.linear1(x_mod_TD), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q_THK, k_THK, v_THV = rearrange(
            qkv_TD, "T (Q H K) -> Q T H K", Q=3, H=self.num_heads
        )
        q_THK, k_THK = self.norm(q_THK, k_THK, v_THV)
        q_THK, k_THK = apply_rope(q_THK, k_THK, pe)
        k_THK, v_THV = gather_flux_kv(k_THK, v_THV, batch_size=batch_size)
        attn_THV = self.inner_attention(
            q_THK, k_THK, v_THV, attention_metadata=attention_metadata
        )

        output_TD = self.linear2(
            torch.cat((attn_THV.flatten(1), self.mlp_act(mlp_TD)), dim=-1)
        )
        return x_TD + local_pack(mod.gate * local_unpack(output_TD, batch_size))


class LastLayer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        linear: Linear.Config
        adaln_linear: Linear.Config
        hidden_size: int
        patch_size: int
        out_channels: int

    def __init__(self, config: Config):
        super().__init__()
        self.norm_final = LayerNorm.Config(
            normalized_shape=config.hidden_size,
            elementwise_affine=False,
            eps=1e-6,
        ).build()
        self.linear = config.linear.build()
        self.adaLN_modulation = Sequential(
            nn.SiLU(),
            config.adaln_linear.build(),
        )

    def forward(self, x_TD: Tensor, vec_BD: Tensor) -> Tensor:
        shift_BD, scale_BD = self.adaLN_modulation(vec_BD).chunk(2, dim=1)
        x_BLD = local_unpack(self.norm_final(x_TD), vec_BD.shape[0])
        x_TD = local_pack((1 + scale_BD[:, None, :]) * x_BLD + shift_BD[:, None, :])
        return self.linear(x_TD)
