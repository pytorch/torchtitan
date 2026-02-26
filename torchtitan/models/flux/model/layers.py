# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# imported from black-forest-labs/FLUX
import math
from dataclasses import dataclass

import torch
from einops import rearrange
from torch import nn, Tensor
from torchtitan.models.common.attention import ScaledDotProductAttentionWrapper
from torchtitan.protocols.module import Module


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
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

    def init_weights(self, **kwargs):
        pass  # no learnable parameters

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


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
        in_dim: int
        hidden_dim: int

    def __init__(self, config: Config):
        super().__init__()
        self.in_layer = nn.Linear(config.in_dim, config.hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(config.hidden_dim, config.hidden_dim, bias=True)

    # pyrefly: ignore [bad-override]
    def init_weights(self, init_std: float = 0.02):
        nn.init.normal_(self.in_layer.weight, std=init_std)
        nn.init.constant_(self.in_layer.bias, 0)
        nn.init.normal_(self.out_layer.weight, std=init_std)
        nn.init.constant_(self.out_layer.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = nn.RMSNorm(dim)
        self.key_norm = nn.RMSNorm(dim)

    def init_weights(self):
        self.query_norm.reset_parameters()
        self.key_norm.reset_parameters()

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
        self.inner_attention = ScaledDotProductAttentionWrapper()

    def init_weights(self):
        for layer in (self.qkv, self.proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        self.norm.init_weights()

    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        x = self.inner_attention(q, k, v, is_causal=False)
        x = rearrange(x, "B H L D -> B L (H D)")
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def init_weights(self):
        nn.init.constant_(self.lin.weight, 0)
        nn.init.constant_(self.lin.bias, 0)

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
        mlp_ratio: float = 4.0
        qkv_bias: bool = False

    def __init__(self, config: Config):
        super().__init__()

        mlp_hidden_dim = int(config.hidden_size * config.mlp_ratio)
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.img_mod = Modulation(config.hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.img_attn = SelfAttention(
            dim=config.hidden_size, num_heads=config.num_heads, qkv_bias=config.qkv_bias
        )

        self.img_norm2 = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.img_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, config.hidden_size, bias=True),
        )

        self.txt_mod = Modulation(config.hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.txt_attn = SelfAttention(
            dim=config.hidden_size, num_heads=config.num_heads, qkv_bias=config.qkv_bias
        )

        self.txt_norm2 = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.txt_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, config.hidden_size, bias=True),
        )

        self.inner_attention = ScaledDotProductAttentionWrapper()

    # pyrefly: ignore [bad-override]
    def init_weights(self):
        # initialize all the nn.Linear submodules
        for layer in (
            self.img_mlp[0],
            self.img_mlp[2],
            self.txt_mlp[0],
            self.txt_mlp[2],
        ):
            # pyrefly: ignore [bad-argument-type]
            nn.init.xavier_uniform_(layer.weight)
            # pyrefly: ignore [bad-argument-type]
            nn.init.constant_(layer.bias, 0)

        # initialize Modulation layers, SelfAttention layers
        for layer in (self.img_attn, self.img_mod, self.txt_attn, self.txt_mod):
            layer.init_weights()

        # Reset parameters for Normalization layers
        for norm in (self.txt_norm1, self.txt_norm2, self.img_norm1, self.img_norm2):
            norm.reset_parameters()

    def forward(
        self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        q, k = apply_rope(q, k, pe)
        attn = self.inner_attention(q, k, v)
        attn = rearrange(attn, "B H L D -> B L (H D)")

        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1] :]

        # calculate the img blocks
        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img = img + img_mod2.gate * self.img_mlp(
            (1 + img_mod2.scale) * self.img_norm2(img) + img_mod2.shift
        )

        # calculate the txt blocks
        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * self.txt_mlp(
            (1 + txt_mod2.scale) * self.txt_norm2(txt) + txt_mod2.shift
        )
        return img, txt


class SingleStreamBlock(Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        num_heads: int
        mlp_ratio: float = 4.0
        qk_scale: float | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.num_heads = config.num_heads
        head_dim = config.hidden_size // config.num_heads
        self.scale = config.qk_scale or head_dim**-0.5

        self.mlp_hidden_dim = int(config.hidden_size * config.mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(
            config.hidden_size, config.hidden_size * 3 + self.mlp_hidden_dim
        )
        # proj and mlp_out
        self.linear2 = nn.Linear(
            config.hidden_size + self.mlp_hidden_dim, config.hidden_size
        )

        self.norm = QKNorm(head_dim)

        self.hidden_size = config.hidden_size
        self.pre_norm = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=1e-6
        )

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(config.hidden_size, double=False)
        self.inner_attention = ScaledDotProductAttentionWrapper()

    # pyrefly: ignore [bad-override]
    def init_weights(self):
        for layer in (self.linear1, self.linear2):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
        self.norm.init_weights()
        self.pre_norm.reset_parameters()
        self.modulation.init_weights()

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor) -> Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        q, k = apply_rope(q, k, pe)
        attn = self.inner_attention(q, k, v)
        attn = rearrange(attn, "B H L D -> B L (H D)")

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        patch_size: int
        out_channels: int

    def __init__(self, config: Config):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            config.hidden_size, elementwise_affine=False, eps=1e-6
        )
        self.linear = nn.Linear(
            config.hidden_size,
            config.patch_size * config.patch_size * config.out_channels,
            bias=True,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_size, 2 * config.hidden_size, bias=True)
        )

    # pyrefly: ignore [bad-override]
    def init_weights(self):
        # pyrefly: ignore [bad-argument-type]
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        # pyrefly: ignore [bad-argument-type]
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)
        self.norm_final.reset_parameters()

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
