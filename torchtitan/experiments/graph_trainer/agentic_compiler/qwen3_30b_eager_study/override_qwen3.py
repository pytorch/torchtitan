# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Override: Qwen3 transformer block whose forward calls the graph loop's kernels.

    config.override.imports = [("override_qwen3.graph_kernel_block", {...})]

Each flag swaps one stretch of the block for an autograd.Function from ``graph_kernels``; with
every flag off the forward is the stock block's (local_compile off). The forward hands the
ffn_norm output to the router and the routed experts as separate tensors, so this block calls
the MoE's router and experts itself (FSDP wraps whole blocks, so their parameters are unsharded).
Only the layout the loop measured is supported: no TP/CP, no shared experts, no remat.
"""

from dataclasses import dataclass

import graph_kernels as gk

import torch

from torchtitan.config import derive, override
from torchtitan.models.common.moe import MoE
from torchtitan.models.qwen3.model import Qwen3TransformerBlock


class GraphKernelQwen3Block(Qwen3TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(Qwen3TransformerBlock.Config):
        attention_input: bool = True
        attention_output: bool = True
        router: bool = False

    def __init__(self, config: Config):
        super().__init__(config)
        self.use_attention_input = config.attention_input
        self.use_attention_output = config.attention_output
        self.use_router = config.router
        if (
            not isinstance(self.feed_forward, MoE)
            or self.feed_forward.shared_experts is not None
        ):
            raise ValueError(
                "GraphKernelQwen3Block supports MoE blocks without shared experts"
            )

    def forward(
        self,
        x,
        attention_metadata,
        positions=None,
        *,
        padding_mask=None,
        aux_loss_denominator=None
    ):
        attn = self.attention
        if self.use_attention_input:
            wqkv = attn.qkv_linear.wqkv
            weight = wqkv.weight
            ops = gk.mxfp8_operands(weight)
            xq, xk, xv, x_res = gk.AttentionInput.apply(
                x,
                self.attention_norm.weight,
                weight,
                ops.weight_qdata_fprop_KN,
                ops.weight_scale_fprop_swizzled,
                ops.weight_qdata_dgrad_NK,
                ops.weight_scale_dgrad_swizzled,
                attn.q_norm.weight,
                attn.k_norm.weight,
                attn.rope.cache,
                positions,
                gk.linear_meta(wqkv, weight),
            )
        else:
            x_res = x
            xq, xk, xv = attn.qkv_linear(self.attention_norm(x))
            xq, xk = attn.rope(attn.q_norm(xq), attn.k_norm(xk), positions)
        out_THV = attn.inner_attention(
            xq,
            xk,
            xv,
            attention_metadata=attention_metadata,
            scale=attn.scaling,
            enable_gqa=attn.enable_gqa,
        )
        out_TD = out_THV.contiguous().view(out_THV.shape[0], -1)
        if self.use_attention_output:
            wo = attn.wo
            ops = gk.mxfp8_operands(wo.weight)
            h, x_router, x_experts = gk.AttentionOutput.apply(
                out_TD,
                x_res,
                wo.weight,
                ops.weight_qdata_fprop_KN,
                ops.weight_scale_fprop_swizzled,
                ops.weight_qdata_dgrad_NK,
                ops.weight_scale_dgrad_swizzled,
                self.ffn_norm.weight,
                gk.linear_meta(wo, wo.weight),
            )
        else:
            h = x_res + attn.wo(out_TD)
            x_router = x_experts = self.ffn_norm(h)
        moe = self.feed_forward
        if self.use_router:
            scores_TK, ids_TK, routing_map_TE = gk.router_forward(
                moe.router,
                x_router,
                moe.expert_bias_E,
                padding_mask,
                aux_loss_denominator,
            )
        else:
            scores_TK, ids_TK, routing_map_TE = moe.router(
                x_router,
                moe.expert_bias_E,
                padding_mask_T=padding_mask,
                aux_loss_denominator=aux_loss_denominator,
            )
        ffn_out = moe.routed_experts(
            x_experts, scores_TK, ids_TK, routing_map_TE, padding_mask_T=padding_mask
        )
        return torch.add(h, ffn_out)


@override(
    target=Qwen3TransformerBlock.Config,
    description="Qwen3 block calling the graph loop's verified kernels from eager",
)
def graph_kernel_block(
    cfg: Qwen3TransformerBlock.Config,
    *,
    attention_input: bool = True,
    attention_output: bool = True,
    router: bool = False
):
    return derive(
        cfg,
        GraphKernelQwen3Block.Config,
        attention_input=attention_input,
        attention_output=attention_output,
        router=router,
    )
