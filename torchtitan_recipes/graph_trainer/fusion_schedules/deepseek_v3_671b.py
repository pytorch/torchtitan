# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fusion plan for dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103."""

from pathlib import Path

from torchtitan.experiments.graph_trainer.auto_fuser.schedule import SchedulePlan
from torchtitan.experiments.graph_trainer.auto_fuser.scheduled_kernels import (
    KernelSource,
    register_schedule,
)
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer

from torchtitan_recipes.graph_trainer.deepseek_v3 import (
    graph_trainer_deepseek_v3_671b_dist_moe_mxfp8,
)

from torchtitan_recipes.graph_trainer.fusion_schedules import (
    kernel_packs_root,
    one_microbatch,
)
from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
    graph_trainer_deepseek_v3_debugmodel,
)

PROBLEM_SET = "dsv3_671b_distmoe_mxfp8_fsdp256_ep64_sm103"


def configure_schedule(plan: SchedulePlan) -> None:
    fwd, bwd = plan.indexed_graph.fwd, plan.indexed_graph.bwd
    num_layers = len(fwd.layers)

    for i in range(3, num_layers):
        aux_loss = fwd.layers[i].feed_forward.router.aux_loss
        router = fwd.layers[i].feed_forward.router
        plan.fuse(
            "r000_router_forward",
            [
                fwd.layers[i].feed_forward.routed_experts.masked_fill,
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
                router.masked_fill,
                router.mul,
                router.ones_like,
                router.scatter_,
                router.sigmoid,
                router.sum,
                router.topk,
                router.zeros_like,
            ],
        )

    for i in reversed(range(3, num_layers)):
        w13 = bwd.layers[i].feed_forward.shared_experts.w13
        plan.fuse(
            "r001_w13_grad_pack",
            [
                w13.cat[1],
                w13.reshape[3],
                w13.split[1],
            ],
        )
    for i in reversed(range(3)):
        w13 = bwd.layers[i].feed_forward.w13
        plan.fuse(
            "r001_w13_grad_pack",
            [
                w13.cat[1],
                w13.reshape[3],
                w13.split[1],
            ],
        )

    for backward_layer, i in enumerate(reversed(range(3, num_layers))):
        add_offset = 8 * backward_layer
        gate = bwd.layers[i].feed_forward.router.gate
        router = bwd.layers[i].feed_forward.router
        plan.fuse(
            "r002_router_backward",
            [
                bwd.add[add_offset + 1],
                bwd.add[add_offset + 2],
                bwd.layers[i].feed_forward.routed_experts.masked_fill[2],
                router.div[3],
                gate.__lshift__,
                *gate._to_copy[0:5],
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

    for backward_layer, i in enumerate(reversed(range(3, num_layers))):
        add_offset = 8 * backward_layer
        aux_loss = bwd.layers[i].feed_forward.router.aux_loss
        router = bwd.layers[i].feed_forward.router
        plan.fuse(
            "r003_router_aux_backward",
            [
                bwd.add[add_offset],
                bwd.layers[i].feed_forward.routed_experts.masked_fill[1],
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
            ],
        )

    for i in reversed(range(num_layers)):
        wkv_b = bwd.layers[i].attention.inner_attention.wkv_b
        inner_attention = bwd.layers[i].attention.inner_attention
        wq_b = bwd.layers[i].attention.wq_b
        attention = bwd.layers[i].attention
        plan.fuse(
            "r004_mla_backward",
            [
                attention.fused_mla_q_rope_,
                inner_attention.fused_mla_kv_backward,
                inner_attention.squeeze[2],
                inner_attention.unsqueeze[3],
                inner_attention.unsqueeze[4],
                wkv_b.mxfp8_quantize[1],
                wkv_b.t[1],
                wkv_b.triton_mx_block_rearrange[1],
                wkv_b.triton_mx_block_rearrange[2],
                attention.unsqueeze[2],
                wq_b.mxfp8_quantize[1],
                wq_b.t[1],
                wq_b.triton_mx_block_rearrange[1],
                wq_b.triton_mx_block_rearrange[3],
            ],
        )

    for backward_layer, i in enumerate(reversed(range(3, num_layers))):
        add_offset = 8 * backward_layer
        wo = bwd.layers[i].attention.wo
        plan.fuse(
            "r005_ffn_norm_backward",
            [
                *bwd.add[add_offset + 3 : add_offset + 6],
                wo.mxfp8_quantize[2],
                wo.t[1],
                wo.triton_mx_block_rearrange[2],
                wo.triton_mx_block_rearrange[3],
                bwd.layers[i].ffn_norm._fused_rms_norm_backward,
            ],
        )

    for i in range(3, num_layers - 1):
        wkv_a = fwd.layers[i + 1].attention.wkv_a
        wq_a = fwd.layers[i + 1].attention.wq_a
        plan.fuse(
            "r006_combine_residual_norm",
            [
                wkv_a.mxfp8_quantize,
                wkv_a.triton_mx_block_rearrange,
                wq_a.mxfp8_quantize,
                wq_a.triton_mx_block_rearrange,
                fwd.layers[i + 1].attention_norm._fused_rms_norm,
                fwd.layers[i].add[1],
                fwd.layers[i].feed_forward.add,
            ],
        )

    for i in reversed(range(3, num_layers)):
        routed_experts = bwd.layers[i].feed_forward.routed_experts
        plan.fuse(
            "r007_expert_weight_recompute",
            [
                routed_experts.new_empty[0],
                routed_experts.prepare_mxfp8_weight[0],
            ],
        )
    for i in reversed(range(3, num_layers)):
        routed_experts = bwd.layers[i].feed_forward.routed_experts
        plan.fuse(
            "r007_expert_weight_recompute",
            [
                routed_experts.new_empty[1],
                routed_experts.prepare_mxfp8_weight[1],
            ],
        )

    for i in range(3, num_layers):
        plan.fuse(
            "r008_expert_weight_prepare",
            [
                fwd.layers[i].feed_forward.routed_experts.prepare_mxfp8_weight[0],
            ],
        )
    for i in range(3, num_layers):
        plan.fuse(
            "r008_expert_weight_prepare",
            [
                fwd.layers[i].feed_forward.routed_experts.prepare_mxfp8_weight[1],
            ],
        )

    for i in reversed(range(num_layers)):
        wkv_a = bwd.layers[i].attention.wkv_a
        wq_a = bwd.layers[i].attention.wq_a
        attention_norm = bwd.layers[i].attention_norm
        plan.fuse(
            "r009_attention_norm_recompute",
            [
                wkv_a.mxfp8_quantize[0],
                wkv_a.mxfp8_quantize[2],
                wkv_a.triton_mx_block_rearrange[0],
                wkv_a.triton_mx_block_rearrange[2],
                wq_a.mxfp8_quantize[0],
                wq_a.mxfp8_quantize[2],
                wq_a.triton_mx_block_rearrange[0],
                wq_a.triton_mx_block_rearrange[2],
                attention_norm._fused_rms_norm,
                attention_norm.alias[1],
            ],
        )

    for backward_layer, i in enumerate(reversed(range(3, num_layers - 1))):
        add_offset = 8 * backward_layer
        w2 = bwd.layers[i].feed_forward.shared_experts.w2
        plan.fuse(
            "r010_attention_norm_backward",
            [
                bwd.add[add_offset + 6],
                bwd.add[add_offset + 7],
                bwd.layers[i + 1].attention_norm._fused_rms_norm_backward,
                w2.mxfp8_quantize[0],
                w2.t,
                w2.triton_mx_block_rearrange[0],
                w2.triton_mx_block_rearrange[1],
            ],
        )
    for backward_layer, i in enumerate(reversed(range(3))):
        add_offset = 462 + 3 * backward_layer
        w2 = bwd.layers[i].feed_forward.w2
        plan.fuse(
            "r010_attention_norm_backward",
            [
                bwd.add[add_offset],
                bwd.add[add_offset + 1],
                bwd.layers[i + 1].attention_norm._fused_rms_norm_backward,
                w2.mxfp8_quantize[0],
                w2.t,
                w2.triton_mx_block_rearrange[0],
                w2.triton_mx_block_rearrange[1],
            ],
        )

    for i in reversed(range(3, num_layers)):
        w13 = bwd.layers[i].feed_forward.shared_experts.w13
        w2 = bwd.layers[i].feed_forward.shared_experts.w2
        shared_experts = bwd.layers[i].feed_forward.shared_experts
        plan.fuse(
            "r011_swiglu_backward",
            [
                shared_experts.silu_and_mul,
                shared_experts.silu_and_mul_backward,
                w13.mxfp8_quantize[2],
                w13.reshape[1],
                w13.t[1],
                w13.triton_mx_block_rearrange[2],
                w13.triton_mx_block_rearrange[3],
                w2.mxfp8_quantize[1],
                w2.triton_mx_block_rearrange[2],
            ],
        )
    for i in reversed(range(3)):
        w13 = bwd.layers[i].feed_forward.w13
        w2 = bwd.layers[i].feed_forward.w2
        feed_forward = bwd.layers[i].feed_forward
        plan.fuse(
            "r011_swiglu_backward",
            [
                feed_forward.silu_and_mul,
                feed_forward.silu_and_mul_backward,
                w13.mxfp8_quantize[2],
                w13.reshape[1],
                w13.t[1],
                w13.triton_mx_block_rearrange[2],
                w13.triton_mx_block_rearrange[3],
                w2.mxfp8_quantize[1],
                w2.triton_mx_block_rearrange[2],
            ],
        )

    for i in reversed(range(3, num_layers)):
        w13 = bwd.layers[i].feed_forward.shared_experts.w13
        ffn_norm = bwd.layers[i].ffn_norm
        plan.fuse(
            "r013_ffn_norm_recompute",
            [
                bwd.layers[i].add,
                w13.mxfp8_quantize[0],
                w13.mxfp8_quantize[1],
                w13.triton_mx_block_rearrange[0],
                w13.triton_mx_block_rearrange[1],
                ffn_norm._fused_rms_norm,
                ffn_norm.alias[1],
            ],
        )
    for i in reversed(range(3)):
        w13 = bwd.layers[i].feed_forward.w13
        ffn_norm = bwd.layers[i].ffn_norm
        plan.fuse(
            "r013_ffn_norm_recompute",
            [
                bwd.layers[i].add,
                w13.mxfp8_quantize[0],
                w13.mxfp8_quantize[1],
                w13.triton_mx_block_rearrange[0],
                w13.triton_mx_block_rearrange[1],
                ffn_norm._fused_rms_norm,
                ffn_norm.alias[1],
            ],
        )

    for i in reversed(range(num_layers)):
        wo = bwd.layers[i].attention.wo
        plan.fuse(
            "r014_wo_input_recompute",
            [
                bwd.layers[i].attention.reshape[0],
                wo.mxfp8_quantize[0],
                wo.mxfp8_quantize[1],
                wo.triton_mx_block_rearrange[0],
                wo.triton_mx_block_rearrange[1],
            ],
        )

    lm_head = bwd.lm_head
    loss = bwd.loss
    plan.fuse(
        "r015_loss_backward",
        [
            lm_head.mxfp8_quantize[0],
            lm_head.t,
            lm_head.triton_mx_block_rearrange[0],
            lm_head.triton_mx_block_rearrange[2],
            loss._log_softmax_backward_data,
            loss._to_copy,
            loss.alias,
            loss.div,
            loss.nll_loss_backward,
        ],
    )

    for i in reversed(range(3, num_layers)):
        w13 = bwd.layers[i].feed_forward.shared_experts.w13
        plan.fuse(
            "r016_w13_weight_recompute",
            [
                w13.cat[0],
                w13.t[0],
                w13.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
            ],
        )
    for i in reversed(range(3)):
        w13 = bwd.layers[i].feed_forward.w13
        plan.fuse(
            "r016_w13_weight_recompute",
            [
                w13.cat[0],
                w13.t[0],
                w13.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
            ],
        )

    for i in range(3):
        w13 = fwd.layers[i].feed_forward.w13
        plan.fuse(
            "r017_w13_weight_quant",
            [
                w13.cat,
                w13.t,
                w13.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
            ],
        )
    for i in range(3, num_layers):
        w13 = fwd.layers[i].feed_forward.shared_experts.w13
        plan.fuse(
            "r017_w13_weight_quant",
            [
                w13.cat,
                w13.t,
                w13.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
            ],
        )

    for i in range(3):
        w13 = fwd.layers[i].feed_forward.w13
        plan.fuse(
            "r020_residual_norm_quant",
            [
                fwd.layers[i].add[0],
                w13.mxfp8_quantize,
                w13.triton_mx_block_rearrange,
                fwd.layers[i].ffn_norm._fused_rms_norm,
            ],
        )
    for i in range(3):
        wq_a = fwd.layers[i + 1].attention.wq_a
        plan.fuse(
            "r020_residual_norm_quant",
            [
                wq_a.mxfp8_quantize,
                wq_a.triton_mx_block_rearrange,
                fwd.layers[i + 1].attention_norm._fused_rms_norm,
                fwd.layers[i].add[1],
            ],
        )
    for i in range(3, num_layers):
        w13 = fwd.layers[i].feed_forward.shared_experts.w13
        plan.fuse(
            "r020_residual_norm_quant",
            [
                fwd.layers[i].add[0],
                w13.mxfp8_quantize,
                w13.triton_mx_block_rearrange,
                fwd.layers[i].ffn_norm._fused_rms_norm,
            ],
        )
    lm_head = fwd.lm_head
    plan.fuse(
        "r020_residual_norm_quant",
        [
            fwd.layers[60].add[1],
            lm_head.mxfp8_quantize,
            lm_head.triton_mx_block_rearrange,
            fwd.norm._fused_rms_norm,
        ],
    )

    for i in reversed(range(num_layers)):
        wkv_a = bwd.layers[i].attention.wkv_a
        plan.fuse(
            "r021_kv_norm_backward",
            [
                bwd.layers[i].attention.cat,
                bwd.layers[i].attention.kv_norm._fused_rms_norm_backward,
                wkv_a.mxfp8_quantize[1],
                wkv_a.t[1],
                wkv_a.triton_mx_block_rearrange[1],
                wkv_a.triton_mx_block_rearrange[3],
            ],
        )

    loss = fwd.loss
    plan.fuse(
        "r022_loss_forward",
        [
            loss._log_softmax,
            loss._to_copy,
            loss.alias,
            loss.div,
            loss.nll_loss_forward,
            fwd.ones_like,
        ],
    )

    for backward_layer, i in enumerate(reversed(range(3, num_layers))):
        _to_copy_offset = 14 * backward_layer
        plan.fuse(
            "r023_wq_grad_cast",
            [
                *bwd._to_copy[_to_copy_offset + 4 : _to_copy_offset + 16],
                bwd.layers[i].attention.wq_a.split_with_sizes,
            ],
        )

    for i in range(num_layers):
        layer = fwd.layers[i]
        attention = layer.attention
        ffn = layer.feed_forward
        w2 = ffn.w2 if i < 3 else ffn.shared_experts.w2
        for linear in (
            attention.wq_a,
            attention.wq_b,
            attention.wkv_a,
            attention.inner_attention.wkv_b,
            attention.wo,
            w2,
        ):
            plan.fuse(
                "r024_weight_quant",
                [
                    linear.t,
                    linear.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
                ],
            )

    lm_head = fwd.lm_head
    plan.fuse(
        "r025_weight_recompute",
        [
            lm_head.t,
            lm_head.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
        ],
    )
    for i in reversed(range(num_layers)):
        attention = bwd.layers[i].attention
        for linear in (
            attention.wq_a,
            attention.wq_b,
            attention.wo,
            attention.wkv_a,
            attention.inner_attention.wkv_b,
        ):
            plan.fuse(
                "r025_weight_recompute",
                [
                    linear.t[0],
                    linear.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
                ],
            )

    for i in reversed(range(num_layers)):
        wq_a = bwd.layers[i].attention.wq_a
        plan.fuse(
            "r026_q_norm_backward",
            [
                bwd.layers[i].attention.q_norm._fused_rms_norm_backward,
                wq_a.mxfp8_quantize[1],
                wq_a.t[1],
                wq_a.triton_mx_block_rearrange[1],
                wq_a.triton_mx_block_rearrange[3],
            ],
        )

    for i in range(3):
        w2 = fwd.layers[i].feed_forward.w2
        plan.fuse(
            "r027_swiglu_quant",
            [
                fwd.layers[i].feed_forward.silu_and_mul,
                fwd.layers[i].feed_forward.w13.reshape[1],
                w2.mxfp8_quantize,
                w2.triton_mx_block_rearrange,
            ],
        )
    for i in range(3, num_layers):
        w2 = fwd.layers[i].feed_forward.shared_experts.w2
        plan.fuse(
            "r027_swiglu_quant",
            [
                fwd.layers[i].feed_forward.shared_experts.silu_and_mul,
                fwd.layers[i].feed_forward.shared_experts.w13.reshape[1],
                w2.mxfp8_quantize,
                w2.triton_mx_block_rearrange,
            ],
        )

    for i in reversed(range(num_layers)):
        q_norm = bwd.layers[i].attention.q_norm
        wq_b = bwd.layers[i].attention.wq_b
        plan.fuse(
            "r028_q_norm_recompute",
            [
                q_norm._fused_rms_norm,
                q_norm.alias[1],
                wq_b.mxfp8_quantize[0],
                wq_b.mxfp8_quantize[2],
                wq_b.triton_mx_block_rearrange[0],
                wq_b.triton_mx_block_rearrange[2],
            ],
        )

    for i in reversed(range(num_layers)):
        wkv_b = bwd.layers[i].attention.inner_attention.wkv_b
        kv_norm = bwd.layers[i].attention.kv_norm
        plan.fuse(
            "r029_kv_norm_recompute",
            [
                bwd.layers[i].attention.inner_attention.unsqueeze[1],
                wkv_b.mxfp8_quantize[0],
                wkv_b.triton_mx_block_rearrange[0],
                wkv_b.triton_mx_block_rearrange[3],
                kv_norm._fused_rms_norm,
                kv_norm.alias[1],
                bwd.layers[i].attention.split_with_sizes,
            ],
        )

    for i in range(num_layers):
        wkv_b = fwd.layers[i].attention.inner_attention.wkv_b
        plan.fuse(
            "r030_kv_norm_quant",
            [
                fwd.layers[i].attention.inner_attention.unsqueeze[1],
                wkv_b.mxfp8_quantize,
                wkv_b.triton_mx_block_rearrange,
                fwd.layers[i].attention.kv_norm._fused_rms_norm,
                fwd.layers[i].attention.split_with_sizes,
            ],
        )

    for i in range(num_layers):
        wo = fwd.layers[i].attention.wo
        plan.fuse(
            "r031_wo_input_quant",
            [
                fwd.layers[i].attention.reshape[1],
                wo.mxfp8_quantize,
                wo.triton_mx_block_rearrange,
            ],
        )

    for i in range(num_layers):
        wq_b = fwd.layers[i].attention.wq_b
        plan.fuse(
            "r032_q_norm_quant",
            [
                fwd.layers[i].attention.q_norm._fused_rms_norm,
                wq_b.mxfp8_quantize,
                wq_b.triton_mx_block_rearrange,
            ],
        )


def reference() -> GraphTrainer.Config:
    from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime

    config = graph_trainer_deepseek_v3_671b_dist_moe_mxfp8()
    config.parallelism.data_parallel_shard_degree = 256
    config.parallelism.expert_parallel_degree = 64
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.fsdp_defer_gradient_reduction = False
    assert isinstance(config.dist_moe_runtime, DistMoeRuntime.Config)
    config.dist_moe_runtime.scratch_capacity_factor = 64.0
    sample = graph_trainer_deepseek_v3_debugmodel()
    config.hf_assets_path = sample.hf_assets_path
    config.dataloader = sample.dataloader
    config.dataloader.max_num_documents = 32
    return one_microbatch(config)


def scheduled() -> GraphTrainer.Config:
    config = reference()
    packs = kernel_packs_root()
    w13_gradient = KernelSource(
        path=Path(__file__).with_name("w13_gradient.py"),
        sha256="7d151440763dc34bf4a6cb2caaf38f6e04853d9ed8f776cc6d149b280b8e6f26",
    )
    pattern = "pattern_cd69bfdaee6976439c1c9fab1cf86a56cefd3ff4749aba79c19330f59a099dc0"
    kernels = {
        f"{pattern}/azc77cuvwn2adu4g2mn3oaatriwvk35r5pceuxazwc4wwoilnexd": w13_gradient,
        f"{pattern}/aresq22uuuj7km43idbpztvdnafuto4jmwu2ne6i7hhteiejtvdd": w13_gradient,
    }
    config.compile.extra_passes.append(
        register_schedule(
            configure_schedule,
            problem_dir=packs / "kernel_problems" / PROBLEM_SET,
            kernels=kernels,
        )
    )
    return config
