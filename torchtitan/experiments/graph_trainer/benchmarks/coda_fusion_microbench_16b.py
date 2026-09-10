# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DSV3-16B CODA eager-versus-FlexGEMM microbenchmarks.

The shapes come from an unfused GraphTrainer joint FX graph for local batch 4,
sequence length 4096, FSDP4, EP4, and TP1. The run used full activation
checkpointing and the standard EP implementation. CODA graph passes were
disabled while collecting the graph.

Run one case with::

    python -m torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_microbench_16b \
        --case f2_kv_rmsnorm

Use ``--show-source`` to print the plain eager epilogue and FlexGEMM callback.
"""

import torch

from torchtitan.experiments.graph_trainer.benchmarks.coda_fusion_microbench import (
    _bf16,
    _fp32,
    BenchmarkCase,
    eager_b1_lm_head_input_grad_cast,
    eager_b2_shared_expert_swiglu_backward,
    eager_b4_router_input_grad_add,
    eager_b5_mla_rmsnorm_backward,
    eager_b6_weight_grad_cast,
    eager_b7_attention_input_grad_merge,
    eager_f2_kv_rmsnorm,
    eager_f3_attention_output,
    eager_f3_moe_output,
    eager_f4_shared_expert_swiglu,
    make_flex_b1_lm_head_input_grad_cast,
    make_flex_b2_shared_expert_swiglu_backward,
    make_flex_b4_router_input_grad_add,
    make_flex_b5_mla_rmsnorm_backward,
    make_flex_b6_weight_grad_cast,
    make_flex_b7_attention_input_grad_merge,
    make_flex_f2_kv_rmsnorm,
    make_flex_f3_attention_output,
    make_flex_f3_moe_output,
    make_flex_f4_shared_expert_swiglu,
    run_benchmark_suite,
)


TOKENS = 4 * 4096
MODEL_WIDTH = 2048
NUM_EXPERTS = 64
KV_PROJECTION_WIDTH = 4096
KV_LOW_RANK_WIDTH = 512
KV_ROPE_WIDTH = 64
Q_PROJECTION_WIDTH = 3072
SHARED_EXPERT_WIDTH = 2816
DENSE_FFN_WIDTH = 10944
VOCAB_SIZE = 102400
LOSS_CHUNKS = 8
LOSS_CHUNK_TOKENS = TOKENS // LOSS_CHUNKS


def _make_b1_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((LOSS_CHUNK_TOKENS, VOCAB_SIZE), device),
        _bf16((VOCAB_SIZE, MODEL_WIDTH), device),
    )


def _make_b2_inputs(
    device: torch.device,
    ffn_width: int,
) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((TOKENS, MODEL_WIDTH), device),
        _bf16((MODEL_WIDTH, ffn_width), device),
        _bf16((TOKENS, ffn_width), device),
        _bf16((TOKENS, ffn_width), device),
        _bf16((TOKENS, ffn_width), device),
        _bf16((ffn_width, MODEL_WIDTH), device),
        _bf16((ffn_width, MODEL_WIDTH), device),
    )


def _make_b2_shared_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return _make_b2_inputs(device, SHARED_EXPERT_WIDTH)


def _make_b2_dense_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return _make_b2_inputs(device, DENSE_FFN_WIDTH)


def _make_b4_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _fp32((TOKENS, NUM_EXPERTS), device),
        _fp32((NUM_EXPERTS, MODEL_WIDTH), device),
        _bf16((TOKENS, MODEL_WIDTH), device),
    )


def _make_b5_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((TOKENS, KV_PROJECTION_WIDTH), device),
        _bf16((KV_PROJECTION_WIDTH, KV_LOW_RANK_WIDTH), device),
        _bf16((TOKENS, KV_LOW_RANK_WIDTH), device),
        torch.empty((TOKENS, 1), device=device, dtype=torch.float32).uniform_(0.5, 1.5),
        _bf16((KV_LOW_RANK_WIDTH,), device),
    )


def _make_b6_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((SHARED_EXPERT_WIDTH, TOKENS), device),
        _bf16((TOKENS, MODEL_WIDTH), device),
    )


def _make_b7_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((TOKENS, KV_LOW_RANK_WIDTH + KV_ROPE_WIDTH), device),
        _bf16((KV_LOW_RANK_WIDTH + KV_ROPE_WIDTH, MODEL_WIDTH), device),
        _bf16((TOKENS, Q_PROJECTION_WIDTH), device),
        _bf16((Q_PROJECTION_WIDTH, MODEL_WIDTH), device),
    )


def _make_f2_kv_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((TOKENS, MODEL_WIDTH), device),
        _bf16((MODEL_WIDTH, KV_LOW_RANK_WIDTH + KV_ROPE_WIDTH), device),
        torch.ones((KV_LOW_RANK_WIDTH,), device=device, dtype=torch.bfloat16),
        _bf16((KV_LOW_RANK_WIDTH, KV_PROJECTION_WIDTH), device),
    )


def _make_f3_projection_inputs(
    device: torch.device,
    projection_width: int,
) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((TOKENS, projection_width), device),
        _bf16((projection_width, MODEL_WIDTH), device),
        _bf16((TOKENS, MODEL_WIDTH), device),
        torch.ones((MODEL_WIDTH,), device=device, dtype=torch.bfloat16),
    )


def _make_f3_attention_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return _make_f3_projection_inputs(device, MODEL_WIDTH)


def _make_f3_dense_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return _make_f3_projection_inputs(device, DENSE_FFN_WIDTH)


def _make_f3_moe_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((TOKENS, SHARED_EXPERT_WIDTH), device),
        _bf16((SHARED_EXPERT_WIDTH, MODEL_WIDTH), device),
        _bf16((TOKENS, MODEL_WIDTH), device),
        _bf16((TOKENS, MODEL_WIDTH), device),
        torch.ones((MODEL_WIDTH,), device=device, dtype=torch.bfloat16),
    )


def _make_f4_inputs(
    device: torch.device,
    ffn_width: int,
) -> tuple[torch.Tensor, ...]:
    return (
        _bf16((TOKENS, MODEL_WIDTH), device),
        _bf16((MODEL_WIDTH, ffn_width), device),
        _bf16((MODEL_WIDTH, ffn_width), device),
    )


def _make_f4_shared_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return _make_f4_inputs(device, SHARED_EXPERT_WIDTH)


def _make_f4_dense_inputs(device: torch.device) -> tuple[torch.Tensor, ...]:
    return _make_f4_inputs(device, DENSE_FFN_WIDTH)


CASES: dict[str, BenchmarkCase] = {
    case.name: case
    for case in (
        BenchmarkCase(
            "b1_lm_head_input_grad_cast",
            "B1",
            "One chunked LM-head input-gradient BF16 store and FP32 cast",
            "(2048, 102400) @ (102400, 2048); 8 chunks per step",
            1,
            _make_b1_inputs,
            eager_b1_lm_head_input_grad_cast,
            make_flex_b1_lm_head_input_grad_cast,
        ),
        BenchmarkCase(
            "b2_shared_expert_swiglu_backward",
            "B2",
            "Shared-expert SwiGLU branch derivatives and input-gradient merge",
            "M=16384, D=2048, P=2816; three BF16 GEMMs",
            2,
            _make_b2_shared_inputs,
            eager_b2_shared_expert_swiglu_backward,
            make_flex_b2_shared_expert_swiglu_backward,
            3.1e-5,
            1e-2,
        ),
        BenchmarkCase(
            "b2_dense_ffn_swiglu_backward",
            "B2",
            "Dense first-layer SwiGLU branch derivatives and input-gradient merge",
            "M=16384, D=2048, P=10944; three BF16 GEMMs",
            2,
            _make_b2_dense_inputs,
            eager_b2_shared_expert_swiglu_backward,
            make_flex_b2_shared_expert_swiglu_backward,
            3.1e-5,
            1e-2,
        ),
        BenchmarkCase(
            "b4_router_input_grad_add",
            "B4",
            "FP32 router input-gradient GEMM, BF16 store, and expert-gradient add",
            "(16384, 64) @ (64, 2048)",
            1,
            _make_b4_inputs,
            eager_b4_router_input_grad_add,
            make_flex_b4_router_input_grad_add,
            4.9e-4,
            1e-2,
        ),
        BenchmarkCase(
            "b5_mla_kv_rmsnorm_backward",
            "B5",
            "MLA KV input-gradient projection plus RMSNorm backward",
            "(16384, 4096) @ (4096, 512)",
            1,
            _make_b5_inputs,
            eager_b5_mla_rmsnorm_backward,
            make_flex_b5_mla_rmsnorm_backward,
            0.05,
            0.02,
        ),
        BenchmarkCase(
            "b6_shared_expert_weight_grad_cast",
            "B6",
            "Frequent shared-expert BF16 weight-gradient GEMM and FP32 cast",
            "(2816, 16384) @ (16384, 2048)",
            1,
            _make_b6_inputs,
            eager_b6_weight_grad_cast,
            make_flex_b6_weight_grad_cast,
        ),
        BenchmarkCase(
            "b7_attention_input_grad_merge",
            "B7",
            "KV and direct-Q input-gradient GEMMs followed by BF16 add",
            "(16384, 576) @ (576, 2048) + (16384, 3072) @ (3072, 2048)",
            1,
            _make_b7_inputs,
            eager_b7_attention_input_grad_merge,
            make_flex_b7_attention_input_grad_merge,
        ),
        BenchmarkCase(
            "f2_kv_rmsnorm",
            "F2-KV",
            "Segmented MLA KV projection, RMSNorm, and expanded projection",
            "2048 -> 576, RMSNorm(512), 512 -> 4096 at M=16384",
            2,
            _make_f2_kv_inputs,
            eager_f2_kv_rmsnorm,
            make_flex_f2_kv_rmsnorm,
            1.6e-2,
            2e-2,
        ),
        BenchmarkCase(
            "f3_attention_output",
            "F3-A",
            "Attention WO projection, residual add, and FFN RMSNorm",
            "(16384, 2048) @ (2048, 2048)",
            1,
            _make_f3_attention_inputs,
            eager_f3_attention_output,
            make_flex_f3_attention_output,
            0.07,
            2e-2,
        ),
        BenchmarkCase(
            "f3_moe_output",
            "F3-B",
            "Shared W2 projection, routed add, residual add, and next RMSNorm",
            "(16384, 2816) @ (2816, 2048)",
            1,
            _make_f3_moe_inputs,
            eager_f3_moe_output,
            make_flex_f3_moe_output,
            0.07,
            2e-2,
        ),
        BenchmarkCase(
            "f3_dense_ffn_output",
            "F3-B",
            "Dense W2 projection, residual add, and next attention RMSNorm",
            "(16384, 10944) @ (10944, 2048)",
            1,
            _make_f3_dense_inputs,
            eager_f3_attention_output,
            make_flex_f3_attention_output,
            0.07,
            2e-2,
        ),
        BenchmarkCase(
            "f4_shared_expert_swiglu",
            "F4",
            "Shared-expert W1/W3 GEMMs with SiLU and multiply",
            "two (16384, 2048) @ (2048, 2816) GEMMs",
            2,
            _make_f4_shared_inputs,
            eager_f4_shared_expert_swiglu,
            make_flex_f4_shared_expert_swiglu,
            fast_math_flex_gemms=(0,),
        ),
        BenchmarkCase(
            "f4_dense_ffn_swiglu",
            "F4",
            "Dense first-layer W1/W3 GEMMs with SiLU and multiply",
            "two (16384, 2048) @ (2048, 10944) GEMMs",
            2,
            _make_f4_dense_inputs,
            eager_f4_shared_expert_swiglu,
            make_flex_f4_shared_expert_swiglu,
            fast_math_flex_gemms=(0,),
        ),
    )
}


def main() -> None:
    run_benchmark_suite(CASES, description=__doc__)


if __name__ == "__main__":
    main()
