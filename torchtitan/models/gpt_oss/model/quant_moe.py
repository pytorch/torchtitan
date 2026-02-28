# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Quantized wrapper for GptOssGroupedExperts, enabling ModelOpt QAT.

ModelOpt's mtq.quantize() can only wrap nn.Linear modules by default.
GptOssGroupedExperts stores expert weights as raw nn.Parameter tensors
(mlp1_weight, mlp2_weight) with torch._grouped_mm, so we need a custom
QuantModule that intercepts the forward pass and applies fake-quantization.

This is analogous to ModelOpt's _QuantGptOssExperts plugin for the HuggingFace
GPT-OSS model (plugins/huggingface.py), adapted for torchtitan's architecture:
- Weight names: mlp1_weight/mlp2_weight (not gate_up_proj/down_proj)
- Weight shape: (E, out_dim, in_dim) — no transposition needed (HF needs it)
- Compute: torch._grouped_mm (not torch.bmm)
- DTensor wrapping for tensor parallelism
"""

from functools import partial

import torch
from torch.distributed.tensor import DTensor

from .moe import (
    GptOssGroupedExperts,
    ScaleBiasForward,
    indices_padding_wrapper,
    swiglu,
)


# ---------------------------------------------------------------------------
# NVFP4 quantization configs with correct pattern ordering.
#
# ModelOpt's set_quantizer_by_cfg applies patterns in dict iteration order,
# with last-match-wins semantics. We put the catch-all disable FIRST, then
# our specific enable patterns SECOND so they take priority.
# ---------------------------------------------------------------------------

_NVFP4_QUANT_ATTRS = {
    "num_bits": (2, 1),
    "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    "enable": True,
    "pass_through_bwd": True,
}

TORCHTITAN_NVFP4_MLP_ONLY_CFG = {
    "quant_cfg": {
        # Disable all quantizers first (attention, router, output)
        "*input_quantizer": {"enable": False},
        "*weight_quantizer": {"enable": False},
        "*output_quantizer": {"enable": False},
        # Enable expert MLP quantizers (last-match-wins → overrides disable)
        "*experts.mlp1_weight_quantizer": _NVFP4_QUANT_ATTRS,
        "*experts.mlp2_weight_quantizer": _NVFP4_QUANT_ATTRS,
        "*experts.mlp1_input_quantizer": _NVFP4_QUANT_ATTRS,
        "*experts.mlp2_input_quantizer": _NVFP4_QUANT_ATTRS,
    },
    "algorithm": "max",
}

TORCHTITAN_NVFP4_MLP_WEIGHT_ONLY_CFG = {
    "quant_cfg": {
        "*input_quantizer": {"enable": False},
        "*weight_quantizer": {"enable": False},
        "*output_quantizer": {"enable": False},
        "*experts.mlp1_weight_quantizer": _NVFP4_QUANT_ATTRS,
        "*experts.mlp2_weight_quantizer": _NVFP4_QUANT_ATTRS,
    },
    "algorithm": "max",
}

# Map from config names to config dicts
TORCHTITAN_QAT_CONFIGS = {
    "NVFP4_MLP_ONLY_CFG": TORCHTITAN_NVFP4_MLP_ONLY_CFG,
    "NVFP4_MLP_WEIGHT_ONLY_CFG": TORCHTITAN_NVFP4_MLP_WEIGHT_ONLY_CFG,
}


# ---------------------------------------------------------------------------
# Quantized compute functions — copies of the originals from moe.py with
# mlp2_input_quantizer injection between SwiGLU and the second matmul.
# ---------------------------------------------------------------------------


def _run_experts_grouped_mm_quantized(
    mlp1_weight: torch.Tensor,
    mlp1_bias: torch.Tensor | None,
    mlp2_weight: torch.Tensor,
    mlp2_bias: torch.Tensor | None,
    swiglu_limit: float,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    tp_degree: int = 1,
    *,
    mlp2_input_quantizer=None,
) -> torch.Tensor:
    offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int32)
    num_tokens_per_expert_long = num_tokens_per_expert.to(torch.long)

    h = torch._grouped_mm(
        x.bfloat16(), mlp1_weight.transpose(-2, -1).bfloat16(), offs=offsets
    )

    if mlp1_bias is not None:
        b1 = mlp1_bias.repeat_interleave(num_tokens_per_expert_long, dim=0)
        tail_slack = x.shape[0] - int(offsets[-1])
        if tail_slack:
            b1 = torch.cat([b1, b1.new_zeros((tail_slack, b1.shape[-1]))], dim=0)
        h = h + b1.to(h.dtype)

    h = swiglu(h, limit=swiglu_limit)

    # Quantize intermediate activation before mlp2
    if mlp2_input_quantizer is not None:
        h = mlp2_input_quantizer(h)

    h = torch._grouped_mm(h, mlp2_weight.transpose(-2, -1).bfloat16(), offs=offsets)

    if mlp2_bias is not None:
        b2_base = mlp2_bias.repeat_interleave(num_tokens_per_expert_long, dim=0)
        b2 = ScaleBiasForward.apply(b2_base, tp_degree)
        tail_slack = x.shape[0] - int(offsets[-1])
        if tail_slack:
            b2 = torch.cat([b2, b2.new_zeros((tail_slack, b2.shape[-1]))], dim=0)
        h = h + b2.to(h.dtype)

    return h


def _run_experts_for_loop_quantized(
    mlp1_weight: torch.Tensor,
    mlp1_bias: torch.Tensor | None,
    mlp2_weight: torch.Tensor,
    mlp2_bias: torch.Tensor | None,
    swiglu_limit: float,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
    tp_degree: int = 1,
    *,
    mlp2_input_quantizer=None,
) -> torch.Tensor:
    num_tokens_per_expert = num_tokens_per_expert.tolist()
    num_padding = x.shape[0] - sum(num_tokens_per_expert)

    x = torch.split(
        x[: sum(num_tokens_per_expert)],
        split_size_or_sections=num_tokens_per_expert,
        dim=0,
    )
    out_experts_splits = []
    for expert_idx, x_expert in enumerate(x):
        h = torch.matmul(x_expert, mlp1_weight[expert_idx].transpose(-2, -1))
        if mlp1_bias is not None:
            h = h + mlp1_bias[expert_idx]
        h = swiglu(h, limit=swiglu_limit)
        # Quantize intermediate activation before mlp2
        if mlp2_input_quantizer is not None:
            h = mlp2_input_quantizer(h)
        h = torch.matmul(h, mlp2_weight[expert_idx].transpose(-2, -1))
        if mlp2_bias is not None:
            b2 = ScaleBiasForward.apply(mlp2_bias[expert_idx], tp_degree)
            h = h + b2
        out_experts_splits.append(h)
    out = torch.cat(out_experts_splits, dim=0)
    out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
    return out


# ---------------------------------------------------------------------------
# Quantized module wrapper
# ---------------------------------------------------------------------------


class _QuantGptOssGroupedExperts:
    """Quantized wrapper for GptOssGroupedExperts.

    Inserted by mtq.quantize() via QuantModuleRegistry. Creates TensorQuantizer
    modules for MLP weight and activation quantization, and overrides forward()
    to apply fake-quantization at the correct points in the compute path.

    Weight shape: (E, out_dim, in_dim) — in_dim at dim -1, so NO transposition
    needed for ModelOpt's per-block quantization (unlike the HF version).
    """

    def _setup(self):
        from modelopt.torch.quantization.nn import TensorQuantizer

        # Weight quantizers
        self.mlp1_weight_quantizer = TensorQuantizer()
        self.mlp2_weight_quantizer = TensorQuantizer()

        # Input (activation) quantizers
        self.mlp1_input_quantizer = TensorQuantizer()
        self.mlp2_input_quantizer = TensorQuantizer()

    def forward(
        self,
        x: torch.Tensor,
        num_tokens_per_expert: torch.Tensor,
    ) -> torch.Tensor:
        # --- Extract local tensors from DTensor (same as original) ---
        if isinstance(self.mlp1_weight, DTensor):
            mlp1_weight = self.mlp1_weight.to_local()
            mlp2_weight = self.mlp2_weight.to_local()
            mlp1_bias = (
                self.mlp1_bias.to_local() if self.mlp1_bias is not None else None
            )
            mlp2_bias = (
                self.mlp2_bias.to_local() if self.mlp2_bias is not None else None
            )
        else:
            mlp1_weight = self.mlp1_weight
            mlp1_bias = self.mlp1_bias
            mlp2_weight = self.mlp2_weight
            mlp2_bias = self.mlp2_bias

        # --- Fake-quantize weights ---
        mlp1_weight = self.mlp1_weight_quantizer(mlp1_weight)
        mlp2_weight = self.mlp2_weight_quantizer(mlp2_weight)

        # --- Quantize input activation ---
        x = self.mlp1_input_quantizer(x)

        # --- Determine TP degree from device mesh ---
        tp_degree = 1
        if isinstance(self.mlp1_weight, DTensor):
            mesh_dim_names = self.mlp1_weight.device_mesh.mesh_dim_names
            if "tp" in mesh_dim_names:
                tp_dim_idx = mesh_dim_names.index("tp")
                tp_degree = self.mlp1_weight.device_mesh.size(tp_dim_idx)

        # --- Dispatch to quantized compute ---
        if self.use_grouped_mm:
            run_fn = partial(
                _run_experts_grouped_mm_quantized,
                mlp2_input_quantizer=self.mlp2_input_quantizer,
            )
            if (
                not isinstance(self.mlp1_weight, DTensor)
                or "ep" not in self.mlp1_weight.device_mesh.mesh_dim_names
            ):
                run_fn = indices_padding_wrapper(run_fn)
            return run_fn(
                mlp1_weight,
                mlp1_bias,
                mlp2_weight,
                mlp2_bias,
                self.swiglu_limit,
                x,
                num_tokens_per_expert,
                tp_degree,
            )
        else:
            return _run_experts_for_loop_quantized(
                mlp1_weight,
                mlp1_bias,
                mlp2_weight,
                mlp2_bias,
                self.swiglu_limit,
                x,
                num_tokens_per_expert,
                tp_degree,
                mlp2_input_quantizer=self.mlp2_input_quantizer,
            )


def register_gpt_oss_quant_module():
    """Register GptOssGroupedExperts for ModelOpt quantization.

    Must be called before mtq.quantize() so that the module replacement
    logic can find and wrap GptOssGroupedExperts instances.
    """
    from modelopt.torch.quantization.nn.modules.quant_module import QuantModuleRegistry

    if GptOssGroupedExperts not in QuantModuleRegistry:
        QuantModuleRegistry.register(
            {GptOssGroupedExperts: "torchtitan.GptOssGroupedExperts"}
        )(_QuantGptOssGroupedExperts)
