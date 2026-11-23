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
    num_layers = len(plan.indexed_graph.fwd.layers)
    for i in range(num_layers):
        _forward_layer(plan, i, num_layers)
        _backward_layer(plan, i, num_layers)


def _forward_layer(plan: SchedulePlan, i: int, num_layers: int) -> None:
    fwd = plan.indexed_graph.fwd
    layer = fwd.layers[i]
    attention = layer.attention
    inner_attention = attention.inner_attention
    rope = attention.rope

    plan.fuse(
        "r001_mla_q_norm_rope",
        [
            attention.cat,
            inner_attention.expand,
            inner_attention.inner_attention.unsqueeze[0],
            attention.kv_norm._fused_rms_norm,
            attention.reshape[0],
            rope._to_copy,
            rope.index,
            rope.mul,
            attention.split_with_sizes,
        ],
    )

    if i > 0:
        routed_experts = layer.feed_forward.routed_experts
        router = layer.feed_forward.router
        aux_loss = router.aux_loss

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

        norm = fwd.layers[i + 1].attention_norm if i < num_layers - 1 else fwd.norm

        plan.fuse(
            "r009_combine_residual_norm",
            [
                norm._fused_rms_norm,
                layer.add[1],
                layer.feed_forward.add,
                routed_experts._to_copy[1],
                routed_experts.deterministic_scatter_add,
                routed_experts.mul[1],
            ],
        )

    plan.fuse(
        "r007_mla_kv_prep",
        [
            inner_attention.cat,
            inner_attention.inner_attention.unsqueeze[1],
            inner_attention.inner_attention.unsqueeze[2],
            inner_attention.reshape,
            inner_attention.split_with_sizes,
        ],
    )


def _backward_layer(plan: SchedulePlan, i: int, num_layers: int) -> None:
    bwd = plan.indexed_graph.bwd
    layer = bwd.layers[i]
    attention = layer.attention
    inner_attention = attention.inner_attention
    rope = attention.rope
    feed_forward = layer.feed_forward

    plan.fuse(
        "r000_mla_qkv_recompute",
        [
            attention.cat[0],
            inner_attention.cat[0],
            *inner_attention.inner_attention.unsqueeze[1:4],
            inner_attention.reshape[0],
            inner_attention.split_with_sizes,
            attention.kv_norm._fused_rms_norm,
            attention.kv_norm.alias[1],
            attention.reshape[2],
            *rope._to_copy[0:4],
            rope.clone,
            rope.index,
            rope.mul[0],
            rope.mul[1],
            attention.split_with_sizes,
        ],
    )

    plan.fuse(
        "r003_q_rope_backward",
        [
            attention.cat[1],
            inner_attention.inner_attention.squeeze[3],
            rope._to_copy[6],
            rope._to_copy[7],
            rope.mul[3],
            attention.slice,
            attention.wq.t,
        ],
    )

    plan.fuse(
        "r004_kv_rope_backward",
        [
            inner_attention.cat[1],
            inner_attention.inner_attention.squeeze[1],
            inner_attention.inner_attention.squeeze[2],
            inner_attention.slice,
            inner_attention.sum,
            inner_attention.wkv_b.t,
            rope._to_copy[4],
            rope._to_copy[5],
            rope.mul[2],
            attention.squeeze[1],
        ],
    )

    dense_ffn = feed_forward.shared_experts if i > 0 else feed_forward

    plan.fuse(
        "r011_swiglu_backward",
        [
            dense_ffn.mul,
            dense_ffn.silu,
            dense_ffn.silu_backward,
            dense_ffn.stack,
            dense_ffn.unbind,
            dense_ffn.w13.reshape[0],
            dense_ffn.w13.t,
        ],
    )

    if i > 0:
        backward_layer = num_layers - 1 - i
        add_offset = 8 * backward_layer
        routed_experts = feed_forward.routed_experts
        router = feed_forward.router
        gate = router.gate
        ffn_norm = layer.ffn_norm

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

        if i < num_layers - 1:
            aux_loss = router.aux_loss
            attention_norm = bwd.layers[i + 1].attention_norm

            plan.fuse(
                "r006_aux_loss_backward",
                [
                    bwd.add[add_offset - 2],
                    bwd.add[add_offset - 1],
                    bwd.add[add_offset + 1],
                    attention_norm._fused_rms_norm,
                    attention_norm._fused_rms_norm_backward,
                    attention_norm._to_copy,
                    layer.add,
                    routed_experts._to_copy,
                    routed_experts.floor_divide,
                    routed_experts.gather,
                    routed_experts.index,
                    routed_experts.mul[2],
                    routed_experts.reshape[0],
                    routed_experts.sort,
                    routed_experts.w2.transpose[1],
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
                    feed_forward.shared_experts.w2.t,
                    ffn_norm._fused_rms_norm,
                    ffn_norm.alias[1],
                ],
            )

        plan.fuse(
            "r010_ffn_norm_backward",
            [
                bwd.add[add_offset],
                bwd.add[add_offset + 4],
                bwd.add[add_offset + 5],
                attention.wo.t,
                routed_experts.index_put[0],
                routed_experts.new_zeros[0],
                ffn_norm._fused_rms_norm_backward,
                ffn_norm._to_copy,
            ],
        )

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
                routed_experts.w13.reshape[1],
                routed_experts.w13.transpose[1],
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
