# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fusion plan for dsv3_debug_fsdp8_sm103."""

from torchtitan.experiments.graph_trainer.auto_fuser.schedule import SchedulePlan
from torchtitan.experiments.graph_trainer.auto_fuser.scheduled_kernels import (
    register_schedule,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_recipes.graph_trainer.fusion_schedules import (
    kernel_packs_root,
    one_microbatch,
)
from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
    graph_trainer_deepseek_v3_debugmodel,
)

PROBLEM_SET = "dsv3_debug_fsdp8_sm103"


def configure_schedule(plan: SchedulePlan) -> None:
    fwd, bwd = plan.indexed_graph.fwd, plan.indexed_graph.bwd
    num_layers = len(fwd.layers)

    for i in reversed(range(num_layers)):
        inner_attention = bwd.layers[i].attention.inner_attention
        kv_norm = bwd.layers[i].attention.kv_norm
        rope = bwd.layers[i].attention.rope
        attention = bwd.layers[i].attention
        plan.fuse(
            "r000_mla_qkv_recompute",
            [
                attention.cat[0],
                inner_attention.cat[0],
                *bwd.layers[i].attention.inner_attention.inner_attention.unsqueeze[1:4],
                inner_attention.reshape[0],
                inner_attention.split_with_sizes,
                kv_norm._fused_rms_norm,
                kv_norm.alias[1],
                attention.reshape[2],
                *rope._to_copy[0:4],
                rope.clone,
                rope.index,
                rope.mul[0],
                rope.mul[1],
                attention.split_with_sizes,
            ],
        )

    for i in range(num_layers):
        rope = fwd.layers[i].attention.rope
        attention = fwd.layers[i].attention
        plan.fuse(
            "r001_mla_q_norm_rope",
            [
                attention.cat,
                fwd.layers[i].attention.inner_attention.expand,
                fwd.layers[i].attention.inner_attention.inner_attention.unsqueeze[0],
                fwd.layers[i].attention.kv_norm._fused_rms_norm,
                attention.reshape[0],
                rope._to_copy,
                rope.index,
                rope.mul,
                attention.split_with_sizes,
            ],
        )

    for backward_layer, i in enumerate(reversed(range(1, num_layers))):
        add_offset = 8 * backward_layer
        routed_experts = bwd.layers[i].feed_forward.routed_experts
        gate = bwd.layers[i].feed_forward.router.gate
        router = bwd.layers[i].feed_forward.router
        plan.fuse(
            "r002_router_backward",
            [
                bwd.add[add_offset + 2],
                bwd.add[add_offset + 3],
                routed_experts.index_put[1],
                routed_experts.mul[1],
                routed_experts.new_zeros[1],
                routed_experts.sum,
                router.div[3],
                gate.__lshift__,
                gate._to_copy,
                gate.cat[0],
                gate.permute,
                gate.split,
                gate.sub,
                router.mul[1],
                router.mul[2],
                router.neg,
                router.new_zeros,
                router.scatter_add,
                router.sigmoid_backward,
                router.sum[1],
            ],
        )

    for i in reversed(range(num_layers)):
        rope = bwd.layers[i].attention.rope
        attention = bwd.layers[i].attention
        plan.fuse(
            "r003_q_rope_backward",
            [
                attention.cat[1],
                bwd.layers[i].attention.inner_attention.inner_attention.squeeze[3],
                rope._to_copy[6],
                rope._to_copy[7],
                rope.mul[3],
                attention.slice,
                bwd.layers[i].attention.wq.t,
            ],
        )

    for i in reversed(range(num_layers)):
        inner_attention = bwd.layers[i].attention.inner_attention.inner_attention
        bwd_inner_attention = bwd.layers[i].attention.inner_attention
        rope = bwd.layers[i].attention.rope
        plan.fuse(
            "r004_kv_rope_backward",
            [
                bwd_inner_attention.cat[1],
                inner_attention.squeeze[1],
                inner_attention.squeeze[2],
                bwd_inner_attention.slice,
                bwd_inner_attention.sum,
                bwd.layers[i].attention.inner_attention.wkv_b.t,
                rope._to_copy[4],
                rope._to_copy[5],
                rope.mul[2],
                bwd.layers[i].attention.squeeze[1],
            ],
        )

    for i in range(1, num_layers):
        aux_loss = fwd.layers[i].feed_forward.router.aux_loss
        routed_experts = fwd.layers[i].feed_forward.routed_experts
        router = fwd.layers[i].feed_forward.router
        plan.fuse(
            "r005_router_forward",
            [
                routed_experts._to_copy[0],
                routed_experts.cumsum,
                routed_experts.expand,
                routed_experts.floor_divide,
                routed_experts.index,
                routed_experts.reshape[2],
                routed_experts.sort,
                routed_experts.sum,
                router.add,
                router.add_,
                aux_loss._to_copy,
                aux_loss.add_,
                aux_loss.bitwise_not,
                aux_loss.clamp_min,
                aux_loss.div,
                aux_loss.linalg_vector_norm,
                aux_loss.mul,
                aux_loss.reciprocal,
                aux_loss.sum,
                router.bitwise_and,
                router.bitwise_not,
                router.div,
                router.gather,
                router.mul,
                router.scatter_,
                router.sigmoid,
                router.sum,
                router.topk,
                router.zeros_like,
            ],
        )

    for backward_layer, i in enumerate(reversed(range(1, num_layers - 1))):
        add_offset = 8 * backward_layer
        aux_loss = bwd.layers[i].feed_forward.router.aux_loss
        routed_experts = bwd.layers[i].feed_forward.routed_experts
        router = bwd.layers[i].feed_forward.router
        attention_norm = bwd.layers[i + 1].attention_norm
        ffn_norm = bwd.layers[i].ffn_norm
        plan.fuse(
            "r006_aux_loss_backward",
            [
                bwd.add[add_offset + 6],
                bwd.add[add_offset + 7],
                bwd.add[add_offset + 9],
                attention_norm._fused_rms_norm,
                attention_norm._fused_rms_norm_backward,
                attention_norm._to_copy,
                bwd.layers[i].add,
                routed_experts._to_copy,
                routed_experts.floor_divide,
                routed_experts.gather,
                routed_experts.index,
                routed_experts.mul[2],
                routed_experts.reshape[0],
                routed_experts.sort,
                bwd.layers[i].feed_forward.routed_experts.w2.transpose[1],
                router.add,
                router.alias[1],
                aux_loss._to_copy,
                aux_loss.bitwise_not,
                aux_loss.clamp_min,
                aux_loss.div,
                aux_loss.gt,
                aux_loss.linalg_vector_norm,
                aux_loss.mul,
                aux_loss.neg,
                aux_loss.ones_like,
                aux_loss.reciprocal,
                aux_loss.scalar_tensor,
                aux_loss.sgn,
                aux_loss.sum,
                aux_loss.where,
                *router.div[0:3],
                router.gather,
                router.mul[0],
                router.sigmoid,
                router.sum[0],
                bwd.layers[i].feed_forward.shared_experts.w2.t,
                ffn_norm._fused_rms_norm,
                ffn_norm.alias[1],
            ],
        )

    for i in range(num_layers):
        inner_attention = fwd.layers[i].attention.inner_attention.inner_attention
        fwd_inner_attention = fwd.layers[i].attention.inner_attention
        plan.fuse(
            "r007_mla_kv_prep",
            [
                fwd_inner_attention.cat,
                inner_attention.unsqueeze[1],
                inner_attention.unsqueeze[2],
                fwd_inner_attention.reshape,
                fwd_inner_attention.split_with_sizes,
            ],
        )

    for i in range(1, num_layers - 1):
        routed_experts = fwd.layers[i].feed_forward.routed_experts
        plan.fuse(
            "r009_combine_residual_norm",
            [
                fwd.layers[i + 1].attention_norm._fused_rms_norm,
                fwd.layers[i].add[1],
                fwd.layers[i].feed_forward.add,
                routed_experts._to_copy[1],
                routed_experts.deterministic_scatter_add,
                routed_experts.mul[1],
            ],
        )
    routed_experts = fwd.layers[5].feed_forward.routed_experts
    plan.fuse(
        "r009_combine_residual_norm",
        [
            fwd.layers[5].add[1],
            fwd.layers[5].feed_forward.add,
            routed_experts._to_copy[1],
            routed_experts.deterministic_scatter_add,
            routed_experts.mul[1],
            fwd.norm._fused_rms_norm,
        ],
    )

    for backward_layer, i in enumerate(reversed(range(1, num_layers))):
        add_offset = 8 * backward_layer
        routed_experts = bwd.layers[i].feed_forward.routed_experts
        ffn_norm = bwd.layers[i].ffn_norm
        plan.fuse(
            "r010_ffn_norm_backward",
            [
                bwd.add[add_offset],
                bwd.add[add_offset + 4],
                bwd.add[add_offset + 5],
                bwd.layers[i].attention.wo.t,
                routed_experts.index_put[0],
                routed_experts.new_zeros[0],
                ffn_norm._fused_rms_norm_backward,
                ffn_norm._to_copy,
            ],
        )

    for i in reversed(range(1, num_layers)):
        w13 = bwd.layers[i].feed_forward.shared_experts.w13
        shared_experts = bwd.layers[i].feed_forward.shared_experts
        plan.fuse(
            "r011_swiglu_backward",
            [
                shared_experts.mul,
                shared_experts.silu,
                shared_experts.silu_backward,
                shared_experts.stack,
                shared_experts.unbind,
                w13.reshape[0],
                w13.t,
            ],
        )
    w13 = bwd.layers[0].feed_forward.w13
    feed_forward = bwd.layers[0].feed_forward
    plan.fuse(
        "r011_swiglu_backward",
        [
            feed_forward.mul,
            feed_forward.silu,
            feed_forward.silu_backward,
            feed_forward.stack,
            feed_forward.unbind,
            w13.reshape[0],
            w13.t,
        ],
    )

    for i in reversed(range(1, num_layers)):
        w13 = bwd.layers[i].feed_forward.routed_experts.w13
        routed_experts = bwd.layers[i].feed_forward.routed_experts
        plan.fuse(
            "r012_routed_swiglu_backward",
            [
                routed_experts.mul[0],
                routed_experts.mul[3],
                routed_experts.mul[4],
                routed_experts.silu,
                routed_experts.silu_backward,
                routed_experts.stack,
                routed_experts.unbind,
                w13.reshape[1],
                w13.transpose[1],
            ],
        )


def reference() -> GraphTrainer.Config:
    config = graph_trainer_deepseek_v3_debugmodel()
    return one_microbatch(config)


def scheduled() -> GraphTrainer.Config:
    config = reference()
    packs = kernel_packs_root()
    config.compile.extra_passes.append(
        register_schedule(
            configure_schedule,
            problem_dir=packs / "kernel_problems" / PROBLEM_SET,
            kernels=packs / "graph_trainer_kernel_packs" / PROBLEM_SET,
        )
    )
    return config
