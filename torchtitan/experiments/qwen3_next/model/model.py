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
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
    create_attention_mask,
    get_block_causal_mask_mod_by_seq_lens,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from torchtitan.models.moe import MoE
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

from .args import Qwen3NextModelArgs

try:
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from fla.modules.convolution import causal_conv1d as causal_conv1d_fn

    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    FusedRMSNormGated = None
    chunk_gated_delta_rule = None
    causal_conv1d_fn = None


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

        self.use_flex_attn = model_args.use_flex_attn
        if self.use_flex_attn:
            self.inner_attention = FlexAttentionWrapper()
        else:
            self.inner_attention = ScaledDotProductAttentionWrapper()

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
        attention_masks: AttentionMasksType | None,
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
        xk = keys.transpose(1, 2)
        xv = values.transpose(1, 2)

        if self.use_flex_attn:
            output = self.inner_attention(
                xq, xk, xv, block_mask=attention_masks["flex_attn"], scale=self.scaling
            )
        else:
            output = self.inner_attention(xq, xk, xv, scale=self.scaling)

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

    def fix_query_key_value_ordering(self, mixed_qkvz, mixed_ba):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvz` and `mixed_ba`.
        """

        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.linear_num_key_heads,
            2 * self.linear_key_head_dim
            + 2
            * self.linear_value_head_dim
            * self.linear_num_value_heads
            // self.linear_num_key_heads,
        )
        new_tensor_shape_ba = mixed_ba.size()[:-1] + (
            self.linear_num_key_heads,
            2 * self.linear_num_value_heads // self.linear_num_key_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)
        split_arg_list_qkvz = [
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
        split_arg_list_ba = [
            self.linear_num_value_heads // self.linear_num_key_heads,
            self.linear_num_value_heads // self.linear_num_key_heads,
        ]
        query, key, value, z = torch.split(mixed_qkvz, split_arg_list_qkvz, dim=3)
        b, a = torch.split(mixed_ba, split_arg_list_ba, dim=3)
        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(
            value.size(0), value.size(1), -1, self.linear_value_head_dim
        )
        z = z.reshape(z.size(0), z.size(1), -1, self.linear_value_head_dim)
        b = b.reshape(b.size(0), b.size(1), self.linear_num_value_heads)
        a = a.reshape(a.size(0), a.size(1), self.linear_num_value_heads)
        return query, key, value, z, b, a

    def forward(
        self,
        hidden_states: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        position_ids: torch.Tensor | None,
    ):
        # hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # Set up dimensions for reshapes later
        batch_size, seq_len, _ = hidden_states.shape

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = (
            x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value)
        )

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)
        mixed_qkv, _ = causal_conv1d_fn(
            x=mixed_qkv,
            weight=self.conv1d.weight.squeeze(1),
            bias=self.conv1d.bias,
            activation=self.activation,
            seq_idx=(
                attention_masks.get("seq_idx", None)
                if isinstance(attention_masks, dict)
                else None
            ),
        )

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim,
                self.key_dim,
                self.value_dim,
            ],
            dim=-1,
        )
        query = query.reshape(
            query.shape[0], query.shape[1], -1, self.linear_key_head_dim
        )
        key = key.reshape(key.shape[0], key.shape[1], -1, self.linear_key_head_dim)
        value = value.reshape(
            value.shape[0], value.shape[1], -1, self.linear_value_head_dim
        )

        beta = b.sigmoid()
        # If the model is loaded in fp16, without the .float() here, A might be -inf
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        if self.linear_num_value_heads // self.linear_num_key_heads > 1:
            query = query.repeat_interleave(
                self.linear_num_value_heads // self.linear_num_key_heads, dim=2
            )
            key = key.repeat_interleave(
                self.linear_num_value_heads // self.linear_num_key_heads, dim=2
            )

        use_packed = attention_masks is not None and "cu_seqlens" in attention_masks
        if use_packed:
            cu_seqlens = attention_masks["cu_seqlens"]
            bs, seqlen, num_heads, head_dim = query.shape
            total_len = bs * seqlen
            query_flat = query.reshape(1, total_len, num_heads, head_dim).contiguous()
            key_flat = key.reshape(1, total_len, num_heads, head_dim).contiguous()
            value_flat = value.reshape(
                1, total_len, num_heads, self.linear_value_head_dim
            ).contiguous()
            g_flat = g.reshape(1, total_len, num_heads).contiguous()
            beta_flat = beta.reshape(1, total_len, num_heads).contiguous()

            core_attn_out, _ = chunk_gated_delta_rule(
                query_flat,
                key_flat,
                value_flat,
                g=g_flat,
                beta=beta_flat,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

            core_attn_out = core_attn_out.reshape(
                bs, seqlen, num_heads, self.linear_value_head_dim
            )
        else:
            core_attn_out, _ = chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, use_qk_l2norm_in_kernel=True
            )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(
            core_attn_out.shape[0], core_attn_out.shape[1], -1
        )

        output = self.out_proj(core_attn_out)
        return output

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     rope_cache: torch.Tensor,
    #     attention_masks: AttentionMasksType | None,
    #     position_ids: torch.Tensor | None,
    # ):
    #     bs, seqlen, dim = x.shape
    #     projected_qkvz = self.in_proj_qkvz(x)
    #     projected_ba = self.in_proj_ba(x)
    #     new_shape_qkvz = projected_qkvz.size()[:-1] + (
    #         self.linear_num_key_heads,
    #         2 * self.linear_key_head_dim
    #         + 2
    #         * (
    #             self.linear_num_value_heads
    #             // self.linear_num_key_heads
    #             * self.linear_value_head_dim
    #         ),
    #     )
    #     new_shape_ba = projected_ba.size()[:-1] + (
    #         self.linear_num_key_heads,
    #         2 * (self.linear_num_value_heads // self.linear_num_key_heads),
    #     )
    #     projected_qkvz = projected_qkvz.view(*new_shape_qkvz)
    #     projected_ba = projected_ba.view(*new_shape_ba)
    #     split_qkvz = [
    #         self.linear_key_head_dim,
    #         self.linear_key_head_dim,
    #         (
    #             self.linear_num_value_heads
    #             // self.linear_num_key_heads
    #             * self.linear_value_head_dim
    #         ),
    #         (
    #             self.linear_num_value_heads
    #             // self.linear_num_key_heads
    #             * self.linear_value_head_dim
    #         ),
    #     ]
    #     split_ba = [
    #         (self.linear_num_value_heads // self.linear_num_key_heads),
    #         (self.linear_num_value_heads // self.linear_num_key_heads),
    #     ]
    #     query, key, value_pre, z_pre = torch.split(projected_qkvz, split_qkvz, dim=3)
    #     b_pre, a_pre = torch.split(projected_ba, split_ba, dim=3)
    #     b = b_pre.reshape(b_pre.size(0), b_pre.size(1), self.linear_num_value_heads)
    #     a = a_pre.reshape(a_pre.size(0), a_pre.size(1), self.linear_num_value_heads)
    #     mixed_qkv = torch.cat((query, key, value_pre), dim=-1).transpose(1, 2)
    #     mixed_qkv = mixed_qkv.reshape(bs, -1, seqlen).contiguous(memory_format=torch.channels_last)
    #     mixed_qkv = causal_conv1d_fn(
    #         mixed_qkv,
    #         self.conv1d.weight.squeeze(1),
    #         self.conv1d.bias,
    #         activation=self.activation,
    #         seq_idx=attention_masks.get("seq_idx", None) if attention_masks is not None else None,
    #     ).transpose(1, 2)

    #     query, key, value = torch.split(
    #         mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
    #     )
    #     query = query.reshape(bs, seqlen, -1, self.linear_key_head_dim)
    #     key = key.reshape(bs, seqlen, -1, self.linear_key_head_dim)
    #     value = value.reshape(bs, seqlen, -1, self.linear_value_head_dim)
    #     z = z_pre.reshape(z_pre.size(0), z_pre.size(1), -1, self.linear_value_head_dim)
    #     beta = b.sigmoid()
    #     g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    #     v_ratio = self.linear_num_value_heads // self.linear_num_key_heads
    #     if v_ratio > 1:
    #         query = query.repeat_interleave(v_ratio, dim=2).contiguous()
    #         key = key.repeat_interleave(v_ratio, dim=2).contiguous()

    #     use_packed = attention_masks is not None and "cu_seqlens" in attention_masks
    #     if use_packed:
    #         cu_seqlens = attention_masks["cu_seqlens"]
    #         bs, seqlen, num_heads, head_dim = query.shape
    #         total_len = bs * seqlen
    #         query_flat = query.reshape(1, total_len, num_heads, head_dim).contiguous()
    #         key_flat = key.reshape(1, total_len, num_heads, head_dim).contiguous()
    #         value_flat = value.reshape(1, total_len, num_heads, self.linear_value_head_dim).contiguous()
    #         g_flat = g.reshape(1, total_len, num_heads).contiguous()
    #         beta_flat = beta.reshape(1, total_len, num_heads).contiguous()

    #         core_attn_out, _ = chunk_gated_delta_rule(
    #             query_flat, key_flat, value_flat, g=g_flat, beta=beta_flat,
    #             use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens
    #         )

    #         core_attn_out = core_attn_out.reshape(bs, seqlen, num_heads, self.linear_value_head_dim)
    #     else:
    #         core_attn_out, _ = chunk_gated_delta_rule(
    #             query, key, value, g=g, beta=beta, use_qk_l2norm_in_kernel=True
    #         )

    #     core_attn_out_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    #     z_flat = z.reshape(-1, z.shape[-1])
    #     core_attn_out_flat = self.norm(core_attn_out_flat, z_flat)
    #     core_attn_out = core_attn_out_flat.reshape(bs, seqlen, -1)
    #     return self.out_proj(core_attn_out)

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     rope_cache: torch.Tensor,
    #     attention_masks: AttentionMasksType | None,
    #     position_ids: torch.Tensor | None,
    # ):
    #     bs, seqlen, dim = x.shape
    #     projected_qkvz = self.in_proj_qkvz(x)  # bs, seqlen, 2*key_dim + 2*value_dim
    #     mixed_qkv = projected_qkvz[..., :self.conv_dim]  # bs, seqlen, conv_dim
    #     z = projected_qkvz[..., self.conv_dim:]  # bs, seqlen, value_dim
    #     projected_ba = self.in_proj_ba(x)  # bs, seqlen, 2*num_value_heads
    #     b = projected_ba[..., :self.linear_num_value_heads]
    #     a = projected_ba[..., self.linear_num_value_heads:]
    #     mixed_qkv = mixed_qkv.contiguous().permute(0, 2, 1).contiguous()  # bs, conv_dim, seqlen with seqlen inn
    #     mixed_qkv = causal_conv1d_fn(
    #         mixed_qkv,
    #         self.conv1d.weight.squeeze(1),
    #         self.conv1d.bias,
    #         activation=self.activation,
    #         seq_idx=attention_masks.get("seq_idx", None) if attention_masks is not None else None,
    #     )
    #     mixed_qkv = mixed_qkv.permute(0, 2, 1)  # bs, seqlen, conv_dim
    #     query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
    #     query = query.reshape(bs, seqlen, -1, self.linear_key_head_dim)
    #     key = key.reshape(bs, seqlen, -1, self.linear_key_head_dim)
    #     value = value.reshape(bs, seqlen, -1, self.linear_value_head_dim)
    #     z = z.reshape(bs, seqlen, -1, self.linear_value_head_dim)
    #     beta = b.sigmoid()
    #     g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    #     v_ratio = self.linear_num_value_heads // self.linear_num_key_heads
    #     if v_ratio > 1:
    #         query = query.repeat_interleave(v_ratio, dim=2)
    #         key = key.repeat_interleave(v_ratio, dim=2)

    #     use_packed = attention_masks is not None and "cu_seqlens" in attention_masks
    #     if use_packed:
    #         cu_seqlens = attention_masks["cu_seqlens"]
    #         bs, seqlen, num_heads, head_dim = query.shape
    #         total_len = bs * seqlen
    #         query_flat = query.reshape(1, total_len, num_heads, head_dim)
    #         key_flat = key.reshape(1, total_len, num_heads, head_dim)
    #         value_flat = value.reshape(1, total_len, num_heads, self.linear_value_head_dim)
    #         g_flat = g.reshape(1, total_len, num_heads)
    #         beta_flat = beta.reshape(1, total_len, num_heads)

    #         core_attn_out, _ = chunk_gated_delta_rule(
    #             query_flat, key_flat, value_flat, g=g_flat, beta=beta_flat,
    #             use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens
    #         )

    #         core_attn_out = core_attn_out.reshape(bs, seqlen, num_heads, self.linear_value_head_dim)
    #     else:
    #         core_attn_out, _ = chunk_gated_delta_rule(
    #             query, key, value, g=g, beta=beta, use_qk_l2norm_in_kernel=True
    #         )

    #     core_attn_out_flat = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    #     z_flat = z.reshape(-1, z.shape[-1])
    #     core_attn_out_flat = self.norm(core_attn_out_flat, z_flat)
    #     core_attn_out = core_attn_out_flat.reshape(bs, seqlen, -1)
    #     return self.out_proj(core_attn_out)

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
        attention_masks: AttentionMasksType | None,
        position_ids: torch.Tensor | None = None,
    ):
        x = x + self.attention(
            self.attention_norm(x),
            rope_cache,
            attention_masks,
            position_ids=position_ids,
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

        have_linear_attention = any(
            lt == "linear_attention" for lt in model_args.layer_types
        )
        if have_linear_attention:
            if not HAS_FLA:
                raise ImportError(
                    "The 'flash-linear-attention' package is required for models with 'linear_attention' layers. "
                    "Please install it to proceed: `pip install flash-linear-attention`"
                )

        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
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
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
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

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        ret = {}
        mask_mods = [get_causal_mask_mod()]
        match self.model_args.attn_mask_type:
            case "causal":
                return create_attention_mask(
                    and_masks(*mask_mods),
                    1,
                    None,
                    input_batch.shape[1],
                    input_batch.shape[1],
                )
            case "block_causal":
                B, T = input_batch.shape
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))

                device = input_batch.device
                reset = torch.zeros(B, T, dtype=torch.bool, device=device)
                reset[:, 0] = True
                prev = torch.cat(
                    [
                        torch.full(
                            (B, 1),
                            tokenizer.eos_id,
                            dtype=input_batch.dtype,
                            device=device,
                        ),
                        input_batch[:, :-1],
                    ],
                    dim=1,
                )
                reset |= prev == tokenizer.eos_id
                seg = (
                    torch.cumsum(reset.to(torch.int32), dim=1, dtype=torch.int32) - 1
                ).contiguous()
                ret["seq_idx"] = seg

                cu_seqlens = [torch.tensor([0], dtype=torch.int32, device=device)]
                for b in range(B):
                    if T == 0:
                        continue
                    _, segment_counts = torch.unique(seg[b], return_counts=True)
                    segment_lengths = segment_counts.to(torch.int32)
                    cumulative_lengths = torch.cumsum(segment_lengths, dim=0)
                    offset_cumulative = cu_seqlens[-1][-1] + cumulative_lengths
                    cu_seqlens.append(offset_cumulative)
                cu_seqlens = torch.cat(cu_seqlens)
                ret["cu_seqlens"] = cu_seqlens

            # case "block_causal_by_sequence_lengths":
            #     sequence_lengths = extra_inputs.pop("sequence_lengths", None)
            #     if sequence_lengths is None:
            #         raise RuntimeError(
            #             "`sequence_lengths` required for `block_causal_by_sequence_lengths`"
            #         )
            #     B = input_batch.shape[0]
            #     mask_mods.append(
            #         get_block_causal_mask_mod_by_seq_lens(sequence_lengths)
            #     )

            # TODO: calculate seq_idx and cu_seqlens
            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.attn_mask_type}"
                )
        ret["flex_attn"] = create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )
        return ret

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(
                h,
                self.rope_cache,
                attention_masks=attention_masks,
                position_ids=position_ids,
            )
        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output
