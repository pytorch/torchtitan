# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fusion plan for gpt_oss_120b_mxfp8_fsdp64_ep16_sm103."""

from torchtitan.experiments.graph_trainer.auto_fuser.schedule import SchedulePlan
from torchtitan.experiments.graph_trainer.auto_fuser.scheduled_kernels import (
    KernelSource,
    register_schedule,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_recipes.graph_trainer.fusion_schedules import (
    kernel_packs_root,
    one_microbatch,
)
from torchtitan_recipes.graph_trainer.fusion_schedules.benchmark import (
    graph_trainer_gpt_oss_120b_mxfp8_64gpu,
)

PROBLEM_SET = "gpt_oss_120b_mxfp8_fsdp64_ep16_sm103"


def configure_schedule(plan: SchedulePlan) -> None:
    fwd, bwd = plan.indexed_graph.fwd, plan.indexed_graph.bwd
    num_layers = len(fwd.layers)

    for backward_layer, i in enumerate(reversed(range(num_layers))):
        add_offset = 12 * backward_layer
        w13 = bwd.layers[i].feed_forward.routed_experts.w13
        routed_experts = bwd.layers[i].feed_forward.routed_experts
        plan.fuse(
            "r000_routed_swiglu_backward",
            [
                bwd.add[add_offset],
                bwd.add[add_offset + 1],
                routed_experts.alias,
                routed_experts.gt,
                routed_experts.logical_and_,
                routed_experts.lt,
                *routed_experts.mul[2:9],
                routed_experts.scalar_tensor,
                routed_experts.sigmoid_backward,
                routed_experts.stack,
                w13.index_add,
                w13.new_zeros,
                w13.reshape[1],
                w13.slice,
                w13.transpose[0],
                routed_experts.where,
            ],
        )

    for i in reversed(range(num_layers)):
        w2 = bwd.layers[i].feed_forward.routed_experts.w2
        routed_experts = bwd.layers[i].feed_forward.routed_experts
        plan.fuse(
            "r001_expert_dispatch_backward",
            [
                routed_experts.arange[0],
                routed_experts.empty_like[0],
                routed_experts.index[0],
                routed_experts.scatter_[0],
                w2.index_add,
                w2.new_zeros,
                w2.slice,
                w2.transpose[0],
            ],
        )

    for backward_layer, i in enumerate(reversed(range(num_layers - 1))):
        add_offset = 12 * backward_layer
        aux_loss = bwd.layers[i].feed_forward.router.aux_loss
        routed_experts = bwd.layers[i].feed_forward.routed_experts
        gate = bwd.layers[i].feed_forward.router.gate
        router = bwd.layers[i].feed_forward.router
        attention_norm = bwd.layers[i + 1].attention_norm
        plan.fuse(
            "r002_router_backward",
            [
                bwd.add[add_offset + 11],
                *bwd.add[add_offset + 14 : add_offset + 17],
                attention_norm._fused_rms_norm_backward,
                attention_norm.alias,
                routed_experts._to_copy,
                routed_experts.gather,
                routed_experts.index_put[1],
                routed_experts.mul[0],
                routed_experts.mul[1],
                routed_experts.new_zeros[1],
                routed_experts.sum,
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
                *gate._to_copy[0:6],
                gate.cat[0],
                gate.permute,
                gate.split,
                gate.sub,
                gate.sum,
                router.mul,
                router.neg,
                router.new_zeros,
                router.scatter_add,
                router.sum,
            ],
        )

    for i in range(num_layers):
        w13 = fwd.layers[i].feed_forward.routed_experts.w13
        routed_experts = fwd.layers[i].feed_forward.routed_experts
        plan.fuse(
            "r003_routed_swiglu",
            [
                routed_experts.addcmul,
                routed_experts.alias,
                routed_experts.clamp,
                routed_experts.mul[0],
                routed_experts.mul[1],
                routed_experts.sigmoid,
                routed_experts.unbind[4],
                w13._to_copy,
                w13.add,
                w13.cat[1],
                w13.cat[2],
                w13.index_select,
                w13.new_zeros[1],
                w13.repeat_interleave,
                w13.reshape[1],
                w13.rsub,
            ],
        )

    for i in range(num_layers - 1):
        wqkv = fwd.layers[i + 1].attention.qkv_linear.wqkv
        routed_experts = fwd.layers[i].feed_forward.routed_experts
        attention_norm = fwd.layers[i + 1].attention_norm
        plan.fuse(
            "r004_combine_residual_norm",
            [
                wqkv.mxfp8_quantize,
                wqkv.triton_mx_block_rearrange,
                attention_norm._fused_rms_norm,
                attention_norm.alias,
                fwd.layers[i].add[1],
                routed_experts._to_copy[3],
                routed_experts.deterministic_scatter_add,
                routed_experts.mul[2],
            ],
        )

    for backward_layer, i in enumerate(reversed(range(num_layers))):
        add_offset = 12 * backward_layer
        wqkv = bwd.layers[i].attention.qkv_linear.wqkv
        qkv_linear = bwd.layers[i].attention.qkv_linear
        rope = bwd.layers[i].attention.rope
        plan.fuse(
            "r005_rope_backward",
            [
                *bwd.add[add_offset + 7 : add_offset + 11],
                qkv_linear.cat,
                qkv_linear.reshape[0],
                wqkv.mxfp8_quantize[0],
                wqkv.sum,
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
        rope = fwd.layers[i].attention.rope
        plan.fuse(
            "r006_rope_forward",
            [
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
        add_offset = 12 * backward_layer
        routed_experts = bwd.layers[i].feed_forward.routed_experts
        wo = bwd.layers[i].attention.wo
        ffn_norm = bwd.layers[i].ffn_norm
        plan.fuse(
            "r007_ffn_norm_backward",
            [
                bwd.add[add_offset + 5],
                bwd.add[add_offset + 6],
                wo.mxfp8_quantize[0],
                wo.sum,
                wo.t,
                wo.triton_mx_block_rearrange[0],
                wo.triton_mx_block_rearrange[2],
                routed_experts.index_put[0],
                routed_experts.new_zeros[0],
                ffn_norm._fused_rms_norm_backward,
                ffn_norm.alias,
            ],
        )

    for i in reversed(range(num_layers)):
        inner_attention = bwd.layers[i].attention.inner_attention
        plan.fuse(
            "r008_attention_sink_backward",
            [
                inner_attention._to_copy,
                inner_attention.alias[0],
                inner_attention.mul,
                inner_attention.neg,
                inner_attention.reshape,
                inner_attention.sigmoid_backward,
                inner_attention.sum,
                bwd.layers[i].attention.reshape,
            ],
        )

    for i in range(num_layers):
        w2 = fwd.layers[i].feed_forward.routed_experts.w2
        plan.fuse(
            "r009_expert_combine",
            [
                fwd.layers[i].feed_forward.routed_experts.index[5],
                w2._to_copy,
                w2.add,
                w2.cat[1],
                w2.cat[2],
                w2.index_select,
                w2.new_zeros[1],
                w2.repeat_interleave,
                w2.rsub,
            ],
        )

    lm_head = bwd.lm_head
    loss = bwd.loss
    plan.fuse(
        "r010_loss_backward",
        [
            lm_head.mxfp8_quantize[0],
            lm_head.t[0],
            lm_head.triton_mx_block_rearrange[0],
            lm_head.triton_mx_block_rearrange[2],
            loss._log_softmax_backward_data[0],
            loss._to_copy[0],
            loss.alias[0],
            loss.div[0],
            loss.nll_loss_backward[0],
        ],
    )
    for chunk in range(1, 8):
        lm_head = bwd.lm_head
        loss = bwd.loss
        plan.fuse(
            "r010_loss_backward",
            [
                lm_head.mxfp8_quantize[2 * chunk],
                lm_head.t[chunk],
                lm_head.triton_mx_block_rearrange[3 * chunk],
                lm_head.triton_mx_block_rearrange[3 * chunk + 2],
                loss._log_softmax_backward_data[chunk],
                loss._to_copy[2 * chunk],
                loss.alias[2 * chunk + 1],
                loss.div[chunk],
                loss.nll_loss_backward[chunk],
            ],
        )

    for i in range(num_layers):
        aux_loss = fwd.layers[i].feed_forward.router.aux_loss
        routed_experts = fwd.layers[i].feed_forward.routed_experts
        gate = fwd.layers[i].feed_forward.router.gate
        router = fwd.layers[i].feed_forward.router
        plan.fuse(
            "r011_router_forward",
            [
                routed_experts._to_copy[0],
                routed_experts.expand,
                routed_experts.floor_divide,
                routed_experts.index[0],
                routed_experts.reshape[2],
                routed_experts.reshape[10],
                routed_experts.sort,
                routed_experts.sum[0],
                routed_experts.sum[1],
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
                gate._to_copy,
                gate.add,
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
        "r012_final_loss_chunk",
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
            "r013_loss_chunk",
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

    for i in range(num_layers):
        inner_attention = fwd.layers[i].attention.inner_attention
        wo = fwd.layers[i].attention.wo
        plan.fuse(
            "r014_attention_sink",
            [
                inner_attention._to_copy,
                inner_attention.alias,
                inner_attention.mul,
                inner_attention.sigmoid,
                inner_attention.sub,
                inner_attention.transpose,
                wo.mxfp8_quantize,
                wo.triton_mx_block_rearrange,
            ],
        )

    for i in range(num_layers):
        routed_experts = fwd.layers[i].feed_forward.routed_experts
        plan.fuse(
            "r015_route_permutation",
            [
                routed_experts.add,
                routed_experts.arange[1],
                routed_experts.arange[2],
                routed_experts.empty_like,
                routed_experts.index[4],
                routed_experts.scatter_,
                routed_experts.sub[2],
            ],
        )

    for i in reversed(range(num_layers)):
        routed_experts = bwd.layers[i].feed_forward.routed_experts
        plan.fuse(
            "r016_unpermute_tokens",
            [
                routed_experts.arange[1],
                routed_experts.empty_like[1],
                routed_experts.index[1],
                routed_experts.scatter_[1],
            ],
        )

    for i in range(num_layers):
        plan.fuse(
            "r018_route_gather",
            [
                fwd.layers[i].feed_forward.routed_experts.index[1],
            ],
        )
    for i in range(num_layers):
        plan.fuse(
            "r018_route_gather",
            [
                fwd.layers[i].feed_forward.routed_experts.index[2],
            ],
        )
    for i in range(num_layers):
        plan.fuse(
            "r018_route_gather",
            [
                fwd.layers[i].feed_forward.routed_experts.index[3],
            ],
        )

    aux_loss = bwd.layers[35].feed_forward.router.aux_loss
    routed_experts = bwd.layers[35].feed_forward.routed_experts
    gate = bwd.layers[35].feed_forward.router.gate
    router = bwd.layers[35].feed_forward.router
    norm = bwd.norm
    plan.fuse(
        "r019_router_embedding_backward",
        [
            *bwd.add[2:5],
            routed_experts._to_copy,
            routed_experts.gather,
            routed_experts.index_put[1],
            routed_experts.mul[0],
            routed_experts.mul[1],
            routed_experts.new_zeros[1],
            routed_experts.sum,
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
            *gate._to_copy[0:6],
            gate.cat[0],
            gate.permute,
            gate.split,
            gate.sub,
            gate.sum,
            router.mul,
            router.neg,
            router.new_zeros,
            router.scatter_add,
            router.sum,
            norm._fused_rms_norm_backward,
            norm.alias,
        ],
    )

    routed_experts = fwd.layers[35].feed_forward.routed_experts
    lm_head = fwd.lm_head
    loss = fwd.loss
    norm = fwd.norm
    plan.fuse(
        "r020_final_combine_norm",
        [
            fwd.layers[35].add[1],
            routed_experts._to_copy[3],
            routed_experts.deterministic_scatter_add,
            routed_experts.mul[2],
            lm_head.mxfp8_quantize,
            lm_head.triton_mx_block_rearrange,
            loss.new_zeros,
            loss.slice,
            loss.split_with_sizes[0],
            loss.zeros_like,
            norm._fused_rms_norm,
            norm.alias[2],
        ],
    )

    for i in range(num_layers):
        w13 = fwd.layers[i].feed_forward.routed_experts.w13
        w2 = fwd.layers[i].feed_forward.routed_experts.w2
        routed_experts = fwd.layers[i].feed_forward.routed_experts
        plan.fuse(
            "r021_dispatch_offsets",
            [
                routed_experts.clone,
                routed_experts.cumsum,
                routed_experts.reshape[3],
                routed_experts.reshape[9],
                routed_experts.sub[0],
                routed_experts.sub[1],
                routed_experts.sum[2],
                routed_experts.sum[3],
                w13.cat[0],
                w13.new_zeros[0],
                w13.select,
                w13.slice[1],
                w13.slice[2],
                w13.sub,
                w2.cat[0],
                w2.new_zeros[0],
                w2.select,
                w2.slice,
                w2.sub,
            ],
        )


def reference() -> GraphTrainer.Config:
    config = graph_trainer_gpt_oss_120b_mxfp8_64gpu()
    return one_microbatch(config)


def scheduled() -> GraphTrainer.Config:
    config = reference()
    packs = kernel_packs_root()
    kernels = {
        "pattern_014c7f1530e0bb28a17fd596bc431e1bc5bcbb406ca5db8838c071272f7455e8/"
        "atgxg3zzd7hd3r3ty5v54y5zrdt3m7r76bm3t7dbg4cvepclff3i": KernelSource(
            path=packs
            / "graph_trainer_kernel_packs"
            / PROBLEM_SET
            / "kernels"
            / "pat_cuks2aws4thpkgjhbs5pifw2spyrysjnlfp77oeel5673nevozv"
            / "acpsy45aw6rct4jnx2sxk52j2v5owtttptwl6lxucvktfrlj5q5z.py",
            sha256="fc038532b45260020d8f34d1fac81174e8e093f73a7451c27645841845b935f3",
            entrypoint="run",
        )
    }
    config.compile.extra_passes.append(
        register_schedule(
            configure_schedule,
            problem_dir=packs / "kernel_problems" / PROBLEM_SET,
            kernels=kernels,
        )
    )
    return config
