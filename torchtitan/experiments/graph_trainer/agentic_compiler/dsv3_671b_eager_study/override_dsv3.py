# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: DSv3 transformer block whose forward calls the graph loop's kernels.

    config.override.imports = [("override_dsv3.graph_kernel_block", {...})]

It replaces the recipe's fused_mla and fused_swiglu overrides, which it applies to the block's
attention and SwiGLU configs itself (overrides of nested nodes cannot be combined). Each flag swaps
one stretch of the block for an autograd.Function from ``graph_kernels``; with every flag off the
forward is the stock block's (fused MLA attention, fused SwiGLU, local_compile off). The block runs
its attention, router, routed experts and FFNs itself, since the Functions' boundaries cross those
modules (FSDP wraps whole blocks, so all their parameters are unsharded). ``fsdp_copy_in`` also
replaces FSDP2's reduce-scatter copy-in for the block (through
``FSDPModule.set_reduce_scatter_input_fn``, see ``graph_kernels.reduce_scatter_input_fn``);
``native_copy_in`` instead installs PyTorch's experimental
``reduce_scatter_input_fn_with_native_copy``, for comparison. Only the layout the loop measured is
supported: no TP/CP, no remat.
"""

from dataclasses import dataclass
from functools import partial

import graph_kernels as gk

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp.experimental import reduce_scatter_input_fn_with_native_copy

from torchtitan.config import derive, override
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.rope import _maybe_check_max_pos
from torchtitan.models.deepseek_v3.model import DeepSeekV3TransformerBlock

from torchtitan_recipes.overrides.fused_mla import (
    _materialize_fused_mla_kv,
    _resolve_positions,
    fused_mla,
    fused_mla_q,
    FusedMLAAttention,
)
from torchtitan_recipes.overrides.fused_swiglu import fused_swiglu, FusedSwiGLU


class GraphKernelDSv3Block(DeepSeekV3TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(DeepSeekV3TransformerBlock.Config):
        attention_input: bool = True
        wq_b: bool = True
        mla_kv: bool = True
        attention_output: bool = True
        router: bool = True
        ffn: bool = True
        fsdp_copy_in: bool = True
        native_copy_in: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.use_attention_input = config.attention_input
        self.use_wq_b = config.wq_b
        self.use_mla_kv = config.mla_kv
        self.use_attention_output = config.attention_output
        self.use_router = config.router
        self.use_ffn = config.ffn
        self.use_fsdp_copy_in = config.fsdp_copy_in
        self.use_native_copy_in = config.native_copy_in
        if self.use_fsdp_copy_in and not self.use_ffn:
            raise ValueError("fsdp_copy_in needs ffn (G-007 runs in the FFN backward)")
        if self.use_fsdp_copy_in and self.use_native_copy_in:
            raise ValueError("fsdp_copy_in and native_copy_in both set the copy-in")
        self._copy_in_installed = False
        attn = self.attention
        ffn = (
            self.feed_forward.shared_experts
            if isinstance(self.feed_forward, MoE)
            else self.feed_forward
        )
        if not isinstance(attn, FusedMLAAttention) or attn.q_lora_rank == 0:
            raise ValueError("GraphKernelDSv3Block needs fused MLA attention with wq_a")
        if ffn is None or not isinstance(ffn.activation_fn, FusedSwiGLU):
            raise ValueError("GraphKernelDSv3Block needs a fused SwiGLU (shared) FFN")

    def _feed_forward(self, ffn, y, y_q, y_s):
        if not self.use_ffn:
            return ffn(y)
        if y_q is None:
            y_q, y_s = gk.quantize_rows(y)
        return gk.SwiGLUFeedForward.apply(
            y,
            y_q,
            y_s,
            ffn.w13.weight,
            ffn.w2.weight,
            gk.linear_meta(ffn.w13, ffn.w13.weight),
            gk.linear_meta(ffn.w2, ffn.w2.weight),
            self.use_fsdp_copy_in,
        )

    def forward(
        self,
        x,
        attention_metadata,
        positions=None,
        *,
        padding_mask=None,
        aux_loss_denominator=None,
    ):
        if (
            self.use_fsdp_copy_in or self.use_native_copy_in
        ) and not self._copy_in_installed:
            # FSDP wraps this block after construction, so the copy-in is installed on the
            # first forward.
            if not isinstance(self, FSDPModule):
                raise ValueError(
                    "fsdp_copy_in / native_copy_in need the block wrapped by FSDP2"
                )
            self.set_reduce_scatter_input_fn(
                gk.reduce_scatter_input_fn
                if self.use_fsdp_copy_in
                else reduce_scatter_input_fn_with_native_copy,
                recurse=False,
            )
            self._copy_in_installed = True
        attn = self.attention
        num_tokens = x.shape[0]
        if self.use_attention_input:
            q_a, kv_a, x_res = gk.AttentionInput.apply(
                x,
                self.attention_norm.weight,
                attn.wq_a.weight,
                attn.wkv_a.weight,
                gk.linear_meta(attn.wq_a, attn.wq_a.weight),
                gk.linear_meta(attn.wkv_a, attn.wkv_a.weight),
            )
        else:
            x_res = x
            y = self.attention_norm(x)
            q_a = attn.wq_a(y)
            kv_a = attn.wkv_a(y)
        q_n = attn.q_norm(q_a)
        q = gk.linear(attn.wq_b, q_n) if self.use_wq_b else attn.wq_b(q_n)
        q = q.view(num_tokens, -1, attn.qk_head_dim)
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=attn.rope.cache.shape[0] - 1)
        q = fused_mla_q(
            q.unsqueeze(0), attn.rope.cache, positions, attn.qk_nope_head_dim
        ).squeeze(0)
        kv_latent, k_pe = torch.split(
            kv_a, [attn.kv_lora_rank, attn.qk_rope_head_dim], dim=-1
        )
        positions_T = (
            torch.arange(num_tokens, device=k_pe.device, dtype=torch.int32)
            if positions is None
            else positions.reshape(-1).contiguous()
        )
        kv_c = attn.kv_norm(kv_latent)
        mla = attn.inner_attention
        if self.use_mla_kv:
            k, v = gk.MLAKeyValue.apply(
                kv_c,
                k_pe,
                mla.wkv_b.weight,
                torch.view_as_real(attn.rope.cache).contiguous(),
                _resolve_positions(positions_T, kv_c.unsqueeze(0)),
                attn.qk_nope_head_dim,
                mla.packed_kv_head_dim,
                gk.linear_meta(mla.wkv_b, mla.wkv_b.weight),
            )
            out = mla.inner_attention(
                q,
                k,
                v,
                attention_metadata=attention_metadata,
                scale=attn.softmax_scale,
            )
        else:
            out = mla(
                q,
                kv_c,
                k_pe,
                materialize_kv=partial(
                    _materialize_fused_mla_kv,
                    rope_cache=attn.rope.cache,
                    positions_T=positions_T,
                ),
                attention_metadata=attention_metadata,
                scale=attn.softmax_scale,
            )
        out_TD = out.contiguous().view(num_tokens, -1)
        if self.use_attention_output:
            h, y, y_q, y_s = gk.AttentionOutput.apply(
                out_TD,
                x_res,
                attn.wo.weight,
                self.ffn_norm.weight,
                gk.linear_meta(attn.wo, attn.wo.weight),
            )
        else:
            h = x_res + attn.wo(out_TD)
            y = self.ffn_norm(h)
            y_q = y_s = None
        moe = self.feed_forward
        if not isinstance(moe, MoE):
            return torch.add(h, self._feed_forward(moe, y, y_q, y_s))
        if self.use_router:
            scores_TK, ids_TK = gk.router_forward(
                moe.router, y, moe.expert_bias_E, padding_mask, aux_loss_denominator
            )
            routed_padding_mask = None
        else:
            scores_TK, ids_TK, _ = moe.router(
                y,
                moe.expert_bias_E,
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
            routed_padding_mask = padding_mask
        routed = moe.routed_experts(
            y, scores_TK, ids_TK, None, padding_mask_T=routed_padding_mask
        )
        shared = self._feed_forward(moe.shared_experts, y, y_q, y_s)
        return torch.add(h, torch.add(routed, shared))


@override(
    target=DeepSeekV3TransformerBlock.Config,
    description="DSv3 block calling the graph loop's verified kernels from eager",
)
def graph_kernel_block(cfg: DeepSeekV3TransformerBlock.Config, **flags):
    block = derive(cfg, GraphKernelDSv3Block.Config, **flags)
    block.attention = fused_mla(block.attention)
    for _, activation, parent, attr in list(block.traverse(SwiGLU.Config)):
        if type(activation) is SwiGLU.Config:
            setattr(parent, attr, fused_swiglu(activation))
    return block
