# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. All Rights Reserved.

import math
import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.attention import build_attention
from torchtitan.models.moe import MoE
from torchtitan.protocols.train_spec import ModelProtocol

from .args import Qwen3NextModelArgs

from causal_conv1d import causal_conv1d_fn
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


# Adapted from https://github.com/pytorch/torchtune/blob/main/torchtune/models/qwen2/_positional_embeddings.py
def precompute_rope_cache(
    dim: int, max_seq_len: int, base: float = 1_000_000.0
) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)
    idx_theta = torch.outer(t, freqs).float()
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(rope_cache: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert ndim > 1
    _, seqlen, _, head_dim = x.shape
    rope_cache = rope_cache[0:seqlen]
    assert rope_cache.shape == (seqlen, head_dim * 2)
    shape = [-1, seqlen, 1, head_dim * 2]
    return rope_cache.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    partial_ratio: float = 1.0,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if position_ids is not None:
        rope_cache = rope_cache[position_ids]
        rope_cache = rope_cache.unsqueeze(2)  # [batch_size, seqlen, 1, head_dim * 2]
    else:
        rope_cache = reshape_for_broadcast(rope_cache, xq)
    head_dim = xq.shape[-1]
    rotary_dim = int(head_dim * partial_ratio)
    cos = rope_cache[..., :rotary_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim : head_dim + rotary_dim].to(
        dtype=xq.dtype, device=xq.device
    )
    xq_rot = (xq[..., :rotary_dim] * cos) + (rotate_half(xq[..., :rotary_dim]) * sin)
    xk_rot = (xk[..., :rotary_dim] * cos) + (rotate_half(xk[..., :rotary_dim]) * sin)
    xq_out = torch.cat([xq_rot, xq[..., rotary_dim:]], dim=-1)
    xk_out = torch.cat([xk_rot, xk[..., rotary_dim:]], dim=-1)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class ZeroCenteredRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3Next is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Attention(nn.Module):
    def __init__(self, model_args: Qwen3NextModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads
            if model_args.n_kv_heads is None
            else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.scaling = self.head_dim**-0.5
        self.partial_rotary_factor = model_args.partial_rotary_factor

        self.q_norm = ZeroCenteredRMSNorm(self.head_dim, eps=model_args.norm_eps)
        self.k_norm = ZeroCenteredRMSNorm(self.head_dim, eps=model_args.norm_eps)
        self.wq = nn.Linear(
            model_args.dim, model_args.n_heads * self.head_dim * 2, bias=False
        )
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(
            model_args.n_heads * self.head_dim, model_args.dim, bias=False
        )
        self.sdpa = build_attention(model_args.use_flex_attn, model_args.attn_mask_type)

    def init_weights(self, init_std: float):
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        self.q_norm.weight.data.zero_()
        self.k_norm.weight.data.zero_()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ):
        bs, seqlen, _ = x.shape
        xq, gate = torch.chunk(self.wq(x), 2, dim=-1)
        xk = self.wk(x)
        xv = self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)
        gate = gate.view(bs, seqlen, -1, self.head_dim)

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq, xk = apply_rotary_emb(
            xq,
            xk,
            rope_cache,
            partial_ratio=self.partial_rotary_factor,
            position_ids=position_ids,
        )

        keys = repeat_kv(xk, self.n_rep)
        values = repeat_kv(xv, self.n_rep)

        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        output = self.sdpa(xq, keys, values, scale=self.scaling)
        output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        output = output * torch.sigmoid(gate.view(bs, seqlen, -1))
        return self.wo(output)


class GatedDeltaNet(nn.Module):
    def __init__(self, model_args):
        super().__init__()
        self.linear_num_key_heads = model_args.linear_num_key_heads
        self.linear_num_value_heads = model_args.linear_num_value_heads
        self.linear_key_head_dim = model_args.linear_key_head_dim
        self.linear_value_head_dim = model_args.linear_value_head_dim
        self.key_dim = model_args.linear_key_head_dim * model_args.linear_num_key_heads
        self.value_dim = (
            model_args.linear_value_head_dim * model_args.linear_num_value_heads
        )
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.activation = model_args.hidden_act
        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            bias=False,
            kernel_size=model_args.linear_conv_kernel_dim,
            groups=self.conv_dim,
            padding=model_args.linear_conv_kernel_dim - 1,
        )
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = model_args.linear_num_value_heads * 2
        self.in_proj_qkvz = nn.Linear(model_args.dim, projection_size_qkvz, bias=False)
        self.in_proj_ba = nn.Linear(model_args.dim, projection_size_ba, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(model_args.linear_num_value_heads))
        self.A_log = nn.Parameter(
            torch.log(torch.empty(model_args.linear_num_value_heads).uniform_(0, 16))
        )
        self.norm = FusedRMSNormGated(
            model_args.linear_value_head_dim,
            eps=model_args.norm_eps,
            activation=model_args.hidden_act,
        )
        self.out_proj = nn.Linear(self.value_dim, model_args.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        position_ids: torch.Tensor | None,
    ):
        bs, seqlen, dim = x.shape
        projected_qkvz = self.in_proj_qkvz(x)
        projected_ba = self.in_proj_ba(x)
        new_shape_qkvz = projected_qkvz.size()[:-1] + (
            self.linear_num_key_heads,
            2 * self.linear_key_head_dim
            + 2
            * (
                self.linear_num_value_heads
                // self.linear_num_key_heads
                * self.linear_value_head_dim
            ),
        )
        new_shape_ba = projected_ba.size()[:-1] + (
            self.linear_num_key_heads,
            2 * (self.linear_num_value_heads // self.linear_num_key_heads),
        )
        projected_qkvz = projected_qkvz.view(*new_shape_qkvz)
        projected_ba = projected_ba.view(*new_shape_ba)
        split_qkvz = [
            self.linear_key_head_dim,
            self.linear_key_head_dim,
            (
                self.linear_num_value_heads
                // self.linear_num_key_heads
                * self.linear_value_head_dim
            ),
            (
                self.linear_num_value_heads
                // self.linear_num_key_heads
                * self.linear_value_head_dim
            ),
        ]
        split_ba = [
            (self.linear_num_value_heads // self.linear_num_key_heads),
            (self.linear_num_value_heads // self.linear_num_key_heads),
        ]
        query, key, value_pre, z_pre = torch.split(projected_qkvz, split_qkvz, dim=3)
        b_pre, a_pre = torch.split(projected_ba, split_ba, dim=3)
        b = b_pre.reshape(b_pre.size(0), b_pre.size(1), self.linear_num_value_heads)
        a = a_pre.reshape(a_pre.size(0), a_pre.size(1), self.linear_num_value_heads)
        mixed_qkv = torch.cat((query, key, value_pre), dim=-1).transpose(1, 2)
        mixed_qkv = mixed_qkv.reshape(bs, -1, seqlen)
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            activation=self.activation,
        ).transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(bs, seqlen, -1, self.linear_key_head_dim)
        key = key.reshape(bs, seqlen, -1, self.linear_key_head_dim)
        value = value.reshape(bs, seqlen, -1, self.linear_value_head_dim)
        z = z_pre.reshape(z_pre.size(0), z_pre.size(1), -1, self.linear_value_head_dim)
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        v_ratio = self.linear_num_value_heads // self.linear_num_key_heads
        if v_ratio > 1:
            query = query.repeat_interleave(v_ratio, dim=2)
            key = key.repeat_interleave(v_ratio, dim=2)
        core_attn_out, _ = chunk_gated_delta_rule(
            query, key, value, g=g, beta=beta, use_qk_l2norm_in_kernel=True
        )
        core_attn_out_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z_flat = z.reshape(-1, z.shape[-1])
        core_attn_out_flat = self.norm(core_attn_out_flat, z_flat)
        core_attn_out = core_attn_out_flat.reshape(bs, seqlen, -1)
        return self.out_proj(core_attn_out)

    def init_weights(self, init_std: float):
        self.dt_bias.data.fill_(1.0)
        self.A_log.data.uniform_(0, 16).log_()
        nn.init.trunc_normal_(self.in_proj_qkvz.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.in_proj_ba.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=init_std)
        nn.init.kaiming_uniform_(self.conv1d.weight, a=math.sqrt(5))
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, model_args: Qwen3NextModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.n_layers = model_args.n_layers
        self.layer_type = model_args.layer_types[layer_id]
        if self.layer_type == "linear_attention":
            self.attention = GatedDeltaNet(model_args)
        else:
            self.attention = Attention(model_args)
        self.moe_enabled = model_args.moe_enabled
        if self.moe_enabled and (layer_id + 1) % model_args.decoder_sparse_step == 0:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(
                dim=model_args.dim, hidden_dim=model_args.hidden_dim
            )
        self.attention_norm = ZeroCenteredRMSNorm(
            model_args.dim, eps=model_args.norm_eps
        )
        self.ffn_norm = ZeroCenteredRMSNorm(model_args.dim, eps=model_args.norm_eps)
        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ):
        x = x + self.attention(
            self.attention_norm(x), rope_cache, position_ids=position_ids
        )
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.attention_norm, self.ffn_norm):
            norm.weight.data.zero_()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device, self.n_layers)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class Qwen3NextModel(nn.Module, ModelProtocol):
    def __init__(self, model_args: Qwen3NextModelArgs):
        super().__init__()
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.eos_id = model_args.eos_id
        self.head_dim = model_args.head_dim

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)
        self.norm = ZeroCenteredRMSNorm(model_args.dim, eps=model_args.norm_eps)

        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def init_weights(self, buffer_device: torch.device | None = None):
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            layer.init_weights(buffer_device)
        self.norm.weight.data.zero_()
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        input_batch: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(h, self.rope_cache, position_ids=position_ids)
        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output
