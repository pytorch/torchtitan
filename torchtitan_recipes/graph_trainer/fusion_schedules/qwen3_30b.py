# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fusion plan for qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103."""

from torchtitan.experiments.graph_trainer.auto_fuser.schedule import SchedulePlan
from torchtitan.experiments.graph_trainer.auto_fuser.scheduled_kernels import (
    register_schedule,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.models.common.attention.attention import VarlenInnerAttention

from torchtitan_recipes.graph_trainer.fusion_schedules import (
    kernel_packs_root,
    one_microbatch,
)
from torchtitan_recipes.graph_trainer.fusion_schedules.benchmark import (
    graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu,
)

PROBLEM_SET = "qwen3_30b_a3b_distmoe_mxfp8_fsdp8_ep8_sm103"


def configure_schedule(plan: SchedulePlan) -> None:
    fwd, bwd = plan.indexed_graph.fwd, plan.indexed_graph.bwd
    num_layers = len(fwd.layers)
    for i in range(num_layers):
        _forward_layer(plan, i, num_layers)
        _backward_layer(plan, i, num_layers)

    loss = bwd.loss

    plan.fuse(
        "r003_loss_backward",
        [
            bwd.lm_head.t[0],
            loss._log_softmax_backward_data[0],
            loss._to_copy[0],
            loss.alias[0],
            loss.div[0],
            loss.nll_loss_backward[0],
        ],
    )

    for chunk in range(1, 8):
        plan.fuse(
            "r003_loss_backward",
            [
                bwd.lm_head.t[chunk],
                loss._log_softmax_backward_data[chunk],
                loss._to_copy[2 * chunk],
                loss.alias[2 * chunk + 1],
                loss.div[chunk],
                loss.nll_loss_backward[chunk],
            ],
        )

    loss = fwd.loss

    plan.fuse(
        "r005_final_loss_chunk",
        [
            loss._log_softmax[7],
            loss._to_copy[7],
            loss.add[7],
            loss.alias[22],
            loss.clone,
            loss.div[7],
            loss.nll_loss_forward[7],
            loss.ones_like[7],
        ],
    )

    for chunk in range(7):
        plan.fuse(
            "r006_loss_chunk",
            [
                loss._log_softmax[chunk],
                loss._to_copy[chunk],
                loss.add[chunk],
                loss.alias[2 * chunk + 8],
                loss.div[chunk],
                loss.nll_loss_forward[chunk],
                loss.ones_like[chunk],
            ],
        )


def _forward_layer(plan: SchedulePlan, i: int, num_layers: int) -> None:
    fwd = plan.indexed_graph.fwd
    layer = fwd.layers[i]
    attention = layer.attention
    rope = attention.rope
    router = layer.feed_forward.router
    aux_loss = router.aux_loss
    routed_experts = layer.feed_forward.routed_experts
    ffn_norm = layer.ffn_norm

    plan.fuse(
        "r001_qk_norm_rope",
        [
            attention.k_norm._fused_rms_norm,
            attention.k_norm.alias,
            attention.q_norm._fused_rms_norm,
            attention.q_norm.alias,
            attention.qkv_linear.clone,
            attention.qkv_linear.reshape[0],
            attention.qkv_linear.split_with_sizes,
            rope._to_copy,
            rope.add,
            rope.cat,
            rope.index,
            rope.mul,
            rope.neg,
            rope.slice,
        ],
    )

    plan.fuse(
        "r004_router_forward",
        [
            routed_experts.masked_fill,
            router._softmax,
            router.add,
            router.add_,
            router.alias,
            router.arange,
            aux_loss._to_copy,
            aux_loss.bitwise_not,
            aux_loss.clamp_min[1],
            aux_loss.div[1],
            aux_loss.linalg_vector_norm[1],
            aux_loss.mul[1],
            aux_loss.sum[0],
            aux_loss.sum[1],
            aux_loss.view[0],
            router.bitwise_and,
            router.bitwise_not,
            router.div,
            router.gather,
            router.mul,
            router.remainder,
            router.scatter_,
            router.sum,
            router.zeros_like,
        ],
    )

    if i < num_layers - 1:
        next_layer = fwd.layers[i + 1]
        wqkv = next_layer.attention.qkv_linear.wqkv

        plan.fuse(
            "r008_residual_norm_quant",
            [
                wqkv.mxfp8_quantize,
                wqkv.triton_mx_block_rearrange,
                next_layer.attention_norm._fused_rms_norm,
                next_layer.attention_norm.alias,
                layer.add[1],
            ],
        )

    plan.fuse(
        "r009_expert_w13_quant",
        [
            routed_experts._to_copy[0],
            routed_experts.new_empty[0],
            routed_experts.prepare_mxfp8_weight[0],
        ],
    )

    plan.fuse(
        "r012_router_aux_loss",
        [
            aux_loss.add_,
            aux_loss.clamp_min[0],
            aux_loss.clamp_min[2],
            aux_loss.clone,
            aux_loss.div[0],
            aux_loss.linalg_vector_norm[0],
            aux_loss.mul[0],
            aux_loss.mul[2],
            aux_loss.mul[3],
            aux_loss.mul[4],
            aux_loss.mul[5],
            aux_loss.mul[6],
            aux_loss.reciprocal,
            aux_loss.split_with_sizes,
            aux_loss.sum[2],
        ],
    )

    plan.fuse(
        "r013_expert_w2_quant",
        [
            routed_experts._to_copy[1],
            routed_experts.new_empty[1],
            routed_experts.prepare_mxfp8_weight[1],
        ],
    )

    plan.fuse(
        "r015_residual_norm",
        [
            layer.add[0],
            ffn_norm._fused_rms_norm,
            ffn_norm.alias,
        ],
    )

    if i == num_layers - 1:
        plan.fuse(
            "r015_residual_norm",
            [
                layer.add[1],
                fwd.norm._fused_rms_norm,
                fwd.norm.alias[2],
            ],
        )

    plan.fuse(
        "r018_wo_input_quant",
        [
            attention.inner_attention.alias,
            attention.reshape,
            attention.wo.mxfp8_quantize,
            attention.wo.triton_mx_block_rearrange,
        ],
    )


def _backward_layer(plan: SchedulePlan, i: int, num_layers: int) -> None:
    bwd = plan.indexed_graph.bwd
    layer = bwd.layers[i]
    backward_layer = num_layers - 1 - i
    add_offset = 10 * backward_layer
    attention = layer.attention
    rope = attention.rope
    qkv_linear = attention.qkv_linear
    wqkv = qkv_linear.wqkv
    wo = attention.wo
    routed_experts = layer.feed_forward.routed_experts
    router = layer.feed_forward.router
    gate = router.gate
    ffn_norm = layer.ffn_norm
    attention_norm = layer.attention_norm

    plan.fuse(
        "r000_qk_rope_backward",
        [
            *bwd.add[add_offset + 5 : add_offset + 9],
            attention.k_norm._fused_rms_norm_backward,
            attention.k_norm.alias,
            attention.q_norm._fused_rms_norm_backward,
            attention.q_norm.alias,
            qkv_linear.cat,
            qkv_linear.reshape[0],
            wqkv.mxfp8_quantize[0],
            wqkv.t,
            wqkv.triton_mx_block_rearrange[0],
            wqkv.triton_mx_block_rearrange[2],
            rope._to_copy,
            rope.mul,
            rope.neg,
            rope.slice,
            rope.slice_backward,
        ],
    )

    plan.fuse(
        "r002_router_backward",
        [
            *bwd.add[add_offset : add_offset + 3],
            routed_experts.masked_fill,
            router._softmax_backward_data,
            router.alias,
            router.aux_loss.div,
            router.aux_loss.gt,
            router.aux_loss.mul,
            router.aux_loss.neg,
            router.aux_loss.ones_like,
            router.aux_loss.scalar_tensor,
            router.aux_loss.sgn,
            router.aux_loss.sum,
            router.aux_loss.where,
            router.div,
            gate.__lshift__,
            *gate._to_copy[0:5],
            gate.cat[0],
            gate.permute,
            gate.split,
            gate.sub,
            router.mul,
            router.neg,
            router.new_zeros,
            router.scatter_add,
            router.sum,
        ],
    )

    plan.fuse(
        "r007_ffn_norm_backward",
        [
            bwd.add[add_offset + 3],
            bwd.add[add_offset + 4],
            wo.mxfp8_quantize[0],
            wo.t,
            wo.triton_mx_block_rearrange[0],
            wo.triton_mx_block_rearrange[2],
            ffn_norm._fused_rms_norm_backward,
            ffn_norm.alias,
        ],
    )

    if i == 0:
        plan.fuse(
            "r010_embedding_backward",
            [
                bwd.add[add_offset + 9],
                attention_norm._fused_rms_norm_backward,
                attention_norm.alias,
                bwd.tok_embeddings.embedding_dense_backward,
            ],
        )
    else:
        plan.fuse(
            "r011_attention_norm_backward",
            [
                bwd.add[add_offset + 9],
                attention_norm._fused_rms_norm_backward,
                attention_norm.alias,
            ],
        )

    for projection in (wo, wqkv):
        plan.fuse(
            "r016_projection_grad_quant",
            [
                projection.mxfp8_quantize[1],
                projection.triton_mx_block_rearrange[1],
            ],
        )

    cast_start = 9 * backward_layer + (3 if i == 0 else 5)

    plan.fuse(
        "r017_qkv_grad_cast",
        [
            *bwd._to_copy[cast_start : cast_start + 7],
            wqkv.split_with_sizes,
        ],
    )


def reference() -> GraphTrainer.Config:
    config = graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu()
    for _, attention, _, _ in config.model.traverse(VarlenInnerAttention.Config):
        attention.sdpa_backends = ["FLASH_ATTENTION"]
    return one_microbatch(config)


def scheduled() -> GraphTrainer.Config:
    config = reference()
    packs = kernel_packs_root()
    config.compile.extra_passes.append(
        register_schedule(
            configure_schedule,
            problem_dir=packs / "kernel_problems" / PROBLEM_SET,
            kernels=packs
            / "graph_trainer_kernel_packs"
            / f"{PROBLEM_SET}_fusion_finder",
        )
    )
    return config
