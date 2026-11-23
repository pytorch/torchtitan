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
_NUM_DENSE_LAYERS = 3


def configure_schedule(plan: SchedulePlan) -> None:
    fwd, bwd = plan.indexed_graph.fwd, plan.indexed_graph.bwd
    num_layers = len(fwd.layers)
    for i in range(num_layers):
        _forward_layer(plan, i, num_layers)
        _backward_layer(plan, i, num_layers)

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

    lm_head = fwd.lm_head

    plan.fuse(
        "r025_weight_recompute",
        [
            lm_head.t,
            lm_head.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
        ],
    )


def _forward_layer(plan: SchedulePlan, i: int, num_layers: int) -> None:
    fwd = plan.indexed_graph.fwd
    layer = fwd.layers[i]
    attention = layer.attention
    ffn = layer.feed_forward
    ffn_norm = layer.ffn_norm
    is_moe = i >= _NUM_DENSE_LAYERS
    dense_ffn = ffn.shared_experts if is_moe else ffn

    if is_moe:
        router = ffn.router
        aux_loss = router.aux_loss
        routed_experts = ffn.routed_experts

        plan.fuse(
            "r000_router_forward",
            [
                routed_experts.masked_fill,
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

        if i < num_layers - 1:
            next_layer = fwd.layers[i + 1]

            plan.fuse(
                "r006_combine_residual_norm",
                [
                    next_layer.attention.wkv_a.mxfp8_quantize,
                    next_layer.attention.wkv_a.triton_mx_block_rearrange,
                    next_layer.attention.wq_a.mxfp8_quantize,
                    next_layer.attention.wq_a.triton_mx_block_rearrange,
                    next_layer.attention_norm._fused_rms_norm,
                    layer.add[1],
                    ffn.add,
                ],
            )

        for k in range(2):
            plan.fuse(
                "r008_expert_weight_prepare", [routed_experts.prepare_mxfp8_weight[k]]
            )

    plan.fuse(
        "r017_w13_weight_quant",
        [
            dense_ffn.w13.cat,
            dense_ffn.w13.t,
            dense_ffn.w13.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
        ],
    )

    plan.fuse(
        "r020_residual_norm_quant",
        [
            layer.add[0],
            dense_ffn.w13.mxfp8_quantize,
            dense_ffn.w13.triton_mx_block_rearrange,
            ffn_norm._fused_rms_norm,
        ],
    )

    if not is_moe:
        next_layer = fwd.layers[i + 1]

        plan.fuse(
            "r020_residual_norm_quant",
            [
                next_layer.attention.wq_a.mxfp8_quantize,
                next_layer.attention.wq_a.triton_mx_block_rearrange,
                next_layer.attention_norm._fused_rms_norm,
                layer.add[1],
            ],
        )

    if i == num_layers - 1:
        lm_head = fwd.lm_head

        plan.fuse(
            "r020_residual_norm_quant",
            [
                layer.add[1],
                lm_head.mxfp8_quantize,
                lm_head.triton_mx_block_rearrange,
                fwd.norm._fused_rms_norm,
            ],
        )

    for linear in (
        attention.wq_a,
        attention.wq_b,
        attention.wkv_a,
        attention.inner_attention.wkv_b,
        attention.wo,
        dense_ffn.w2,
    ):
        plan.fuse(
            "r024_weight_quant",
            [
                linear.t,
                linear.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
            ],
        )

    plan.fuse(
        "r027_swiglu_quant",
        [
            dense_ffn.silu_and_mul,
            dense_ffn.w13.reshape[1],
            dense_ffn.w2.mxfp8_quantize,
            dense_ffn.w2.triton_mx_block_rearrange,
        ],
    )

    plan.fuse(
        "r030_kv_norm_quant",
        [
            attention.inner_attention.unsqueeze[1],
            attention.inner_attention.wkv_b.mxfp8_quantize,
            attention.inner_attention.wkv_b.triton_mx_block_rearrange,
            attention.kv_norm._fused_rms_norm,
            attention.split_with_sizes,
        ],
    )

    plan.fuse(
        "r031_wo_input_quant",
        [
            attention.reshape[1],
            attention.wo.mxfp8_quantize,
            attention.wo.triton_mx_block_rearrange,
        ],
    )

    plan.fuse(
        "r032_q_norm_quant",
        [
            attention.q_norm._fused_rms_norm,
            attention.wq_b.mxfp8_quantize,
            attention.wq_b.triton_mx_block_rearrange,
        ],
    )


def _backward_layer(plan: SchedulePlan, i: int, num_layers: int) -> None:
    bwd = plan.indexed_graph.bwd
    layer = bwd.layers[i]
    attention = layer.attention
    wq_a = attention.wq_a
    wq_b = attention.wq_b
    wkv_a = attention.wkv_a
    wkv_b = attention.inner_attention.wkv_b
    ffn = layer.feed_forward
    ffn_norm = layer.ffn_norm
    attention_norm = layer.attention_norm
    q_norm = attention.q_norm
    kv_norm = attention.kv_norm
    is_moe = i >= _NUM_DENSE_LAYERS
    dense_ffn = ffn.shared_experts if is_moe else ffn
    backward_layer = num_layers - 1 - i

    plan.fuse(
        "r001_w13_grad_pack",
        [
            dense_ffn.w13.cat[1],
            dense_ffn.w13.reshape[3],
            dense_ffn.w13.split[1],
        ],
    )

    if is_moe:
        add_offset = 8 * backward_layer
        router = ffn.router
        aux_loss = router.aux_loss
        gate = router.gate
        routed_experts = ffn.routed_experts

        plan.fuse(
            "r002_router_backward",
            [
                bwd.add[add_offset + 1],
                bwd.add[add_offset + 2],
                routed_experts.masked_fill[2],
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

        plan.fuse(
            "r003_router_aux_backward",
            [
                bwd.add[add_offset],
                routed_experts.masked_fill[1],
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

        plan.fuse(
            "r005_ffn_norm_backward",
            [
                *bwd.add[add_offset + 3 : add_offset + 6],
                attention.wo.mxfp8_quantize[2],
                attention.wo.t[1],
                attention.wo.triton_mx_block_rearrange[2],
                attention.wo.triton_mx_block_rearrange[3],
                ffn_norm._fused_rms_norm_backward,
            ],
        )

        for k in range(2):
            plan.fuse(
                "r007_expert_weight_recompute",
                [
                    routed_experts.new_empty[k],
                    routed_experts.prepare_mxfp8_weight[k],
                ],
            )

        cast_start = 14 * backward_layer + 4

        plan.fuse(
            "r023_wq_grad_cast",
            [
                *bwd._to_copy[cast_start : cast_start + 12],
                wq_a.split_with_sizes,
            ],
        )

    plan.fuse(
        "r004_mla_backward",
        [
            attention.fused_mla_q_rope_,
            attention.inner_attention.fused_mla_kv_backward,
            attention.inner_attention.squeeze[2],
            attention.inner_attention.unsqueeze[3],
            attention.inner_attention.unsqueeze[4],
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

    if i < num_layers - 1:
        # Root adds follow backward execution order, eight per MoE layer and
        # three per dense layer. The last two feed the next attention norm's backward.
        num_dense_after = max(0, _NUM_DENSE_LAYERS - 1 - i)
        num_moe_after = backward_layer - num_dense_after
        norm_add = 8 * num_moe_after + 3 * num_dense_after - 2

        plan.fuse(
            "r010_attention_norm_backward",
            [
                bwd.add[norm_add],
                bwd.add[norm_add + 1],
                bwd.layers[i + 1].attention_norm._fused_rms_norm_backward,
                dense_ffn.w2.mxfp8_quantize[0],
                dense_ffn.w2.t,
                dense_ffn.w2.triton_mx_block_rearrange[0],
                dense_ffn.w2.triton_mx_block_rearrange[1],
            ],
        )

    plan.fuse(
        "r011_swiglu_backward",
        [
            dense_ffn.silu_and_mul,
            dense_ffn.silu_and_mul_backward,
            dense_ffn.w13.mxfp8_quantize[2],
            dense_ffn.w13.reshape[1],
            dense_ffn.w13.t[1],
            dense_ffn.w13.triton_mx_block_rearrange[2],
            dense_ffn.w13.triton_mx_block_rearrange[3],
            dense_ffn.w2.mxfp8_quantize[1],
            dense_ffn.w2.triton_mx_block_rearrange[2],
        ],
    )

    plan.fuse(
        "r013_ffn_norm_recompute",
        [
            layer.add,
            dense_ffn.w13.mxfp8_quantize[0],
            dense_ffn.w13.mxfp8_quantize[1],
            dense_ffn.w13.triton_mx_block_rearrange[0],
            dense_ffn.w13.triton_mx_block_rearrange[1],
            ffn_norm._fused_rms_norm,
            ffn_norm.alias[1],
        ],
    )

    plan.fuse(
        "r014_wo_input_recompute",
        [
            attention.reshape[0],
            attention.wo.mxfp8_quantize[0],
            attention.wo.mxfp8_quantize[1],
            attention.wo.triton_mx_block_rearrange[0],
            attention.wo.triton_mx_block_rearrange[1],
        ],
    )

    plan.fuse(
        "r016_w13_weight_recompute",
        [
            dense_ffn.w13.cat[0],
            dense_ffn.w13.t[0],
            dense_ffn.w13.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
        ],
    )

    plan.fuse(
        "r021_kv_norm_backward",
        [
            attention.cat,
            kv_norm._fused_rms_norm_backward,
            wkv_a.mxfp8_quantize[1],
            wkv_a.t[1],
            wkv_a.triton_mx_block_rearrange[1],
            wkv_a.triton_mx_block_rearrange[3],
        ],
    )

    for linear in (wq_a, wq_b, attention.wo, wkv_a, wkv_b):
        plan.fuse(
            "r025_weight_recompute",
            [
                linear.t[0],
                linear.triton_to_mxfp8_32x32_swizzle_dim0_qdata_dim01_scale,
            ],
        )

    plan.fuse(
        "r026_q_norm_backward",
        [
            q_norm._fused_rms_norm_backward,
            wq_a.mxfp8_quantize[1],
            wq_a.t[1],
            wq_a.triton_mx_block_rearrange[1],
            wq_a.triton_mx_block_rearrange[3],
        ],
    )

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

    plan.fuse(
        "r029_kv_norm_recompute",
        [
            attention.inner_attention.unsqueeze[1],
            wkv_b.mxfp8_quantize[0],
            wkv_b.triton_mx_block_rearrange[0],
            wkv_b.triton_mx_block_rearrange[3],
            kv_norm._fused_rms_norm,
            kv_norm.alias[1],
            attention.split_with_sizes,
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
