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

    for backward_layer, i in enumerate(reversed(range(num_layers))):
        add_offset = 10 * backward_layer
        wqkv = bwd.layers[i].attention.qkv_linear.wqkv
        qkv_linear = bwd.layers[i].attention.qkv_linear
        k_norm = bwd.layers[i].attention.k_norm
        q_norm = bwd.layers[i].attention.q_norm
        rope = bwd.layers[i].attention.rope
        plan.fuse(
            "r000_qk_rope_backward",
            [
                *bwd.add[add_offset + 5 : add_offset + 9],
                k_norm._fused_rms_norm_backward,
                k_norm.alias,
                q_norm._fused_rms_norm_backward,
                q_norm.alias,
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

    for i in range(num_layers):
        qkv_linear = fwd.layers[i].attention.qkv_linear
        k_norm = fwd.layers[i].attention.k_norm
        q_norm = fwd.layers[i].attention.q_norm
        rope = fwd.layers[i].attention.rope
        plan.fuse(
            "r001_qk_norm_rope",
            [
                k_norm._fused_rms_norm,
                k_norm.alias,
                q_norm._fused_rms_norm,
                q_norm.alias,
                qkv_linear.clone,
                qkv_linear.reshape[0],
                qkv_linear.split_with_sizes,
                rope._to_copy,
                rope.add,
                rope.cat,
                rope.index,
                rope.mul,
                rope.neg,
                rope.slice,
            ],
        )

    for backward_layer, i in enumerate(reversed(range(num_layers))):
        add_offset = 10 * backward_layer
        aux_loss = bwd.layers[i].feed_forward.router.aux_loss
        gate = bwd.layers[i].feed_forward.router.gate
        router = bwd.layers[i].feed_forward.router
        plan.fuse(
            "r002_router_backward",
            [
                *bwd.add[add_offset : add_offset + 3],
                bwd.layers[i].feed_forward.routed_experts.masked_fill,
                router._softmax_backward_data,
                router.alias,
                aux_loss.div,
                aux_loss.gt,
                aux_loss.mul,
                aux_loss.neg,
                aux_loss.ones_like,
                aux_loss.scalar_tensor,
                aux_loss.sgn,
                aux_loss.sum,
                aux_loss.where,
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
        loss = bwd.loss
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

    for i in range(num_layers):
        aux_loss = fwd.layers[i].feed_forward.router.aux_loss
        router = fwd.layers[i].feed_forward.router
        plan.fuse(
            "r004_router_forward",
            [
                fwd.layers[i].feed_forward.routed_experts.masked_fill,
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
        loss = fwd.loss
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

    for backward_layer, i in enumerate(reversed(range(num_layers))):
        add_offset = 10 * backward_layer
        wo = bwd.layers[i].attention.wo
        ffn_norm = bwd.layers[i].ffn_norm
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

    for i in range(num_layers - 1):
        wqkv = fwd.layers[i + 1].attention.qkv_linear.wqkv
        attention_norm = fwd.layers[i + 1].attention_norm
        plan.fuse(
            "r008_residual_norm_quant",
            [
                wqkv.mxfp8_quantize,
                wqkv.triton_mx_block_rearrange,
                attention_norm._fused_rms_norm,
                attention_norm.alias,
                fwd.layers[i].add[1],
            ],
        )

    for i in range(num_layers):
        routed_experts = fwd.layers[i].feed_forward.routed_experts
        plan.fuse(
            "r009_expert_w13_quant",
            [
                routed_experts._to_copy[0],
                routed_experts.new_empty[0],
                routed_experts.prepare_mxfp8_weight[0],
            ],
        )

    attention_norm = bwd.layers[0].attention_norm
    plan.fuse(
        "r010_embedding_backward",
        [
            bwd.add[479],
            attention_norm._fused_rms_norm_backward,
            attention_norm.alias,
            bwd.tok_embeddings.embedding_dense_backward,
        ],
    )

    for backward_layer, i in enumerate(reversed(range(1, num_layers))):
        add_offset = 10 * backward_layer
        attention_norm = bwd.layers[i].attention_norm
        plan.fuse(
            "r011_attention_norm_backward",
            [
                bwd.add[add_offset + 9],
                attention_norm._fused_rms_norm_backward,
                attention_norm.alias,
            ],
        )

    for i in range(num_layers):
        aux_loss = fwd.layers[i].feed_forward.router.aux_loss
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

    for i in range(num_layers):
        routed_experts = fwd.layers[i].feed_forward.routed_experts
        plan.fuse(
            "r013_expert_w2_quant",
            [
                routed_experts._to_copy[1],
                routed_experts.new_empty[1],
                routed_experts.prepare_mxfp8_weight[1],
            ],
        )

    for i in range(num_layers):
        ffn_norm = fwd.layers[i].ffn_norm
        plan.fuse(
            "r015_residual_norm",
            [
                fwd.layers[i].add[0],
                ffn_norm._fused_rms_norm,
                ffn_norm.alias,
            ],
        )
    norm = fwd.norm
    plan.fuse(
        "r015_residual_norm",
        [
            fwd.layers[47].add[1],
            norm._fused_rms_norm,
            norm.alias[2],
        ],
    )

    for i in reversed(range(num_layers)):
        wo = bwd.layers[i].attention.wo
        plan.fuse(
            "r016_projection_grad_quant",
            [
                wo.mxfp8_quantize[1],
                wo.triton_mx_block_rearrange[1],
            ],
        )
    for i in reversed(range(num_layers)):
        wqkv = bwd.layers[i].attention.qkv_linear.wqkv
        plan.fuse(
            "r016_projection_grad_quant",
            [
                wqkv.mxfp8_quantize[1],
                wqkv.triton_mx_block_rearrange[1],
            ],
        )

    for backward_layer, i in enumerate(reversed(range(1, num_layers))):
        _to_copy_offset = 9 * backward_layer
        plan.fuse(
            "r017_qkv_grad_cast",
            [
                *bwd._to_copy[_to_copy_offset + 5 : _to_copy_offset + 12],
                bwd.layers[i].attention.qkv_linear.wqkv.split_with_sizes,
            ],
        )
    plan.fuse(
        "r017_qkv_grad_cast",
        [
            *bwd._to_copy[426:433],
            bwd.layers[0].attention.qkv_linear.wqkv.split_with_sizes,
        ],
    )

    for i in range(num_layers):
        wo = fwd.layers[i].attention.wo
        plan.fuse(
            "r018_wo_input_quant",
            [
                fwd.layers[i].attention.inner_attention.alias,
                fwd.layers[i].attention.reshape,
                wo.mxfp8_quantize,
                wo.triton_mx_block_rearrange,
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
