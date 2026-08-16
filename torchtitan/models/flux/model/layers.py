# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# imported from black-forest-labs/FLUX
import math
from dataclasses import dataclass, field

import spmd_types as spmd
import torch
from einops import rearrange
from torch import nn, Tensor
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.distributed.utils import get_spmd_backend
from torchtitan.models.common.attention import create_attention_mask, FlexAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import GELU, LayerNorm, RMSNorm, SiLU
from torchtitan.protocols.module import Module, Sequential


def assert_token_layout(x: torch.Tensor) -> None:
    if get_spmd_backend() == "spmd_types":
        spmd.assert_type(
            x,
            {"dp": spmd.V, "cp": spmd.V},
            spmd.PartitionSpec(("dp", "cp"), *((None,) * (x.ndim - 1))),
        )


def concat_sample_sequences(
    text: torch.Tensor,
    image: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Concatenate text and image tokens within each sample."""
    with spmd.local():
        text = text.view(num_samples, -1, *text.shape[1:])
        image = image.view(num_samples, -1, *image.shape[1:])
        combined = torch.cat((text, image), dim=1).flatten(0, 1)
        assert_token_layout(combined)
    return combined


def split_sample_sequences(
    combined: torch.Tensor,
    num_samples: int,
    num_text_tokens_per_sample: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a sample-interleaved stream into text and image streams."""
    with spmd.local():
        combined = combined.view(num_samples, -1, *combined.shape[1:])
        text = combined[:, :num_text_tokens_per_sample].flatten(0, 1)
        image = combined[:, num_text_tokens_per_sample:].flatten(0, 1)
        assert_token_layout(text)
        assert_token_layout(image)
    return text, image


def repeat_sample_conditioning(
    x: torch.Tensor, num_tokens_per_sample: int
) -> torch.Tensor:
    """Repeat per-sample conditioning for each local token in that sample."""
    with spmd.local():
        x = x.repeat_interleave(num_tokens_per_sample, dim=0)
        assert_token_layout(x)
    return x


def create_flux_attention_mask(
    num_samples: int,
    num_local_tokens_per_sample: int,
    cp_degree: int,
    device: torch.device,
) -> BlockMask:
    """Create sample-isolating attention for local Q and CP-gathered K/V."""
    local_segment_ids = torch.arange(
        num_samples, device=device, dtype=torch.int32
    ).repeat_interleave(num_local_tokens_per_sample)
    global_segment_ids = local_segment_ids.repeat(cp_degree)

    def mask_mod(b, h, q_idx, kv_idx):
        return local_segment_ids[q_idx] == global_segment_ids[kv_idx]

    num_local_tokens = local_segment_ids.shape[0]
    with spmd.no_typecheck():
        return create_attention_mask(
            mask_mod,
            1,
            None,
            num_local_tokens,
            num_local_tokens * cp_degree,
            device=device,
        )


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack(
        [torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1
    )
    out = rearrange(out, "t d (i j) -> t d i j", i=2, j=2)
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
        inner_attention: FlexAttention.Config = field(
            default_factory=FlexAttention.Config
        )

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.qkv = config.qkv.build()
        self.norm = config.norm.build()
        self.proj = config.proj.build()
        self.inner_attention = config.inner_attention.build()

    def forward(self, x: Tensor, pe: Tensor, attention_mask: BlockMask) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "T (K N H) -> K T N H", K=3, N=self.num_heads)
        q, k = self.norm(q, k, v)
        q, k = apply_rope(q, k, pe)
        x = self.inner_attention(q, k, v, attention_masks=attention_mask)
        x = rearrange(x, "T N H -> T (N H)")
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


def expand_modulation(
    modulation: ModulationOut, num_tokens_per_sample: int
) -> ModulationOut:
    """Expand per-sample modulation to token-major activations."""
    return ModulationOut(
        shift=repeat_sample_conditioning(modulation.shift, num_tokens_per_sample),
        scale=repeat_sample_conditioning(modulation.scale, num_tokens_per_sample),
        gate=repeat_sample_conditioning(modulation.gate, num_tokens_per_sample),
    )


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
        out = self.lin(nn.functional.silu(vec)).chunk(self.multiplier, dim=-1)

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
        inner_attention: FlexAttention.Config = field(
            default_factory=FlexAttention.Config
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
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        attention_mask: BlockMask,
        num_samples: int,
    ) -> tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)
        assert txt_mod2 is not None
        assert img_mod2 is not None
        num_img_tokens_per_sample = img.shape[0] // num_samples
        num_txt_tokens_per_sample = txt.shape[0] // num_samples
        img_mod1 = expand_modulation(img_mod1, num_img_tokens_per_sample)
        img_mod2 = expand_modulation(img_mod2, num_img_tokens_per_sample)
        txt_mod1 = expand_modulation(txt_mod1, num_txt_tokens_per_sample)
        txt_mod2 = expand_modulation(txt_mod2, num_txt_tokens_per_sample)

        # prepare image for attention
        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "T (K N H) -> K T N H", K=3, N=self.num_heads
        )
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        # prepare txt for attention
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "T (K N H) -> K T N H", K=3, N=self.num_heads
        )
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        # run actual attention
        q = concat_sample_sequences(txt_q, img_q, num_samples)
        k = concat_sample_sequences(txt_k, img_k, num_samples)
        v = concat_sample_sequences(txt_v, img_v, num_samples)

        q, k = apply_rope(q, k, pe)
        attn = self.inner_attention(q, k, v, attention_masks=attention_mask)
        attn = rearrange(attn, "T N H -> T (N H)")
        txt_attn, img_attn = split_sample_sequences(
            attn, num_samples, num_txt_tokens_per_sample
        )

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
        linear1: Linear.Config
        linear2: Linear.Config
        modulation: Modulation.Config
        norm: QKNorm.Config
        mlp_ratio: float = 4.0
        qk_scale: float | None = None
        inner_attention: FlexAttention.Config = field(
            default_factory=FlexAttention.Config
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
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        attention_mask: BlockMask,
        num_samples: int,
    ) -> Tensor:
        mod, _ = self.modulation(vec)
        num_tokens_per_sample = x.shape[0] // num_samples
        mod = expand_modulation(mod, num_tokens_per_sample)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "T (K N H) -> K T N H", K=3, N=self.num_heads)
        q, k = self.norm(q, k, v)

        # compute attention
        q, k = apply_rope(q, k, pe)
        attn = self.inner_attention(q, k, v, attention_masks=attention_mask)
        attn = rearrange(attn, "T N H -> T (N H)")

        # compute activation in mlp stream, cat again and run second linear layer
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), dim=-1))
        return x + mod.gate * output


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
            SiLU.Config().build(),
            config.adaln_linear.build(),
        )

    def forward(self, x: Tensor, vec: Tensor, num_samples: int) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        num_tokens_per_sample = x.shape[0] // num_samples
        shift = repeat_sample_conditioning(shift, num_tokens_per_sample)
        scale = repeat_sample_conditioning(scale, num_tokens_per_sample)
        x = (1 + scale) * self.norm_final(x) + shift
        x = self.linear(x)
        return x
