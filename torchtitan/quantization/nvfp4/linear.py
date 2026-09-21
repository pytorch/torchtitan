# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""NVFP4 quantized linear building block.

Swaps dense ``Linear.Config`` nodes for :class:`NVFP4Linear`, which keeps a bf16
weight and quantizes activations and gradients to NVFP4 on the fly via TorchAO's
``nvfp4_training`` kernels (NVIDIA Blackwell / sm_100+, CUDA only). Under FSDP,
the NVFP4 weight operands are cached for the current unshard lifetime.

Like :class:`MXFP8LinearConverter`, this is a pure leaf swap: it inherits the
model's stock colwise/rowwise sharding and changes only the GEMM. Under tensor
parallelism the block boundary keeps its stock bf16 collectives (all-gather /
reduce-scatter); NVFP4 does not move fp4 codes over the wire.
"""

from dataclasses import dataclass, replace
from typing import Any, cast

import spmd_types as spmd
import torch
import torch.nn.functional as F
from spmd_types import SpmdType
from torch import nn
from torch.autograd.function import once_differentiable

from torchtitan.distributed.parallel_dims import MeshAxisName
from torchtitan.models.common.decoder_sharding import dense_activation_placement
from torchtitan.models.common.linear import (
    ColumnParallelLinear,
    Linear,
    RowParallelLinear,
)

from .._fsdp_tensor import _UnshardedFSDPTensor


TP = MeshAxisName.TP

# TorchAO's NVFP4 Triton kernels require each local GEMM dimension to be a
# multiple of 128.
_NVFP4_BLOCK = 128

# Fixed Random Hadamard Transform basis (the NVFP4 v1 recipe default in torchao
# and Transformer Engine). It must be identical across TP ranks -- rowwise TP
# shards the GEMM contraction dim, and the Hadamard transform only cancels
# between the two operands when both use the same sign vector. Hardcoding it
# makes every rank produce the same vector by construction (no cross-rank
# broadcast). Per-recipe dynamic sign vectors are a future extension.
_HARDCODED_SIGN_VECTOR = (
    1,
    1,
    1,
    -1,
    1,
    -1,
    -1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    -1,
)

from torchao.prototype.moe_training.nvfp4_training.nvfp4_linear import (
    _resolve_use_cutedsl,
    _rht_amax,
    _rht_quantize_row_col,
)
from torchao.prototype.mx_formats.nvfp4_tensor import per_tensor_amax_to_scale
from torchao.quantization.quantize_.common.kernel_preference import KernelPreference

from .tensor import _LinearShardedTensorWithNVFP4Compute, _quantize_nvfp4_weight


__all__ = ["NVFP4Linear"]


# Factored from
# torchao.prototype.moe_training.nvfp4_training.nvfp4_linear.nvfp4_matmul.
# Keeping the raw scaled GEMM here lets TorchTitan pass FSDP-cached operands.
def _nvfp4_scaled_mm(
    lhs_qdata: torch.Tensor,
    lhs_block_scale: torch.Tensor,
    lhs_global_scale: torch.Tensor,
    rhs_qdata: torch.Tensor,
    rhs_block_scale: torch.Tensor,
    rhs_global_scale: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return F.scaled_mm(
        lhs_qdata,
        rhs_qdata,
        scale_a=[lhs_block_scale.flatten(), lhs_global_scale],
        scale_recipe_a=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        scale_b=[rhs_block_scale.flatten(), rhs_global_scale],
        scale_recipe_b=[F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
        swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
        bias=bias,
        output_dtype=torch.bfloat16,
    )


def _nvfp4_scaled_mm_out(
    lhs_qdata: torch.Tensor,
    lhs_block_scale: torch.Tensor,
    lhs_global_scale: torch.Tensor,
    rhs_qdata: torch.Tensor,
    rhs_block_scale: torch.Tensor,
    rhs_global_scale: torch.Tensor,
    *,
    out: torch.Tensor,
) -> torch.Tensor:
    """Write one NVFP4 matrix product into caller-owned storage."""
    scaled_mm_v2 = cast(Any, torch.ops.aten)._scaled_mm_v2
    return scaled_mm_v2.out(
        lhs_qdata,
        rhs_qdata,
        [lhs_block_scale.flatten(), lhs_global_scale],
        [F.ScalingType.BlockWise1x16.value, F.ScalingType.TensorWise.value],
        [F.SwizzleType.SWIZZLE_32_4_4.value, F.SwizzleType.NO_SWIZZLE.value],
        [rhs_block_scale.flatten(), rhs_global_scale],
        [F.ScalingType.BlockWise1x16.value, F.ScalingType.TensorWise.value],
        [F.SwizzleType.SWIZZLE_32_4_4.value, F.SwizzleType.NO_SWIZZLE.value],
        None,
        torch.bfloat16,
        [],
        False,
        out=out,
    )


# Adapted from
# torchao.prototype.moe_training.nvfp4_training.nvfp4_linear.nvfp4_matmul.
# Weight quantization is removed from the autograd function and supplied by
# TorchTitan's FSDP cache; TorchAO still provides the RHT and cast kernels.
@torch._dynamo.allow_in_graph
class _NVFP4LinearFunction(torch.autograd.Function):
    """NVFP4 linear whose weight state is managed by TorchTitan FSDP."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        x: torch.Tensor,
        weight_NK: torch.Tensor,
        weight_qdata_fprop: torch.Tensor,
        weight_scale_fprop: torch.Tensor,
        weight_qdata_dgrad: torch.Tensor,
        weight_scale_dgrad: torch.Tensor,
        weight_amax: torch.Tensor,
        bias_N: torch.Tensor | None,
        sr_seed: torch.Tensor,
        sign_vector: tuple[int, ...],
        use_cutedsl: bool,
        use_fast_math: bool,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16 or weight_NK.dtype != torch.bfloat16:
            raise ValueError(
                "NVFP4Linear requires BF16 activations and weights; "
                f"got activation dtype {x.dtype} and weight dtype {weight_NK.dtype}."
            )
        if bias_N is not None and bias_N.dtype != torch.bfloat16:
            raise ValueError(
                f"NVFP4Linear requires a BF16 bias; got bias dtype {bias_N.dtype}."
            )
        if x.shape[-1] != weight_NK.shape[1]:
            raise ValueError(
                "NVFP4Linear activation and weight contraction dimensions must "
                f"match; got {x.shape[-1]} and {weight_NK.shape[1]}."
            )

        input_shape = x.shape
        x_MK = x.reshape(-1, input_shape[-1]).contiguous()
        num_rows, in_features = x_MK.shape
        out_features = weight_NK.shape[0]
        if any(value % _NVFP4_BLOCK for value in (num_rows, in_features, out_features)):
            raise ValueError(
                "NVFP4Linear requires flattened rows, local in_features, and "
                f"local out_features divisible by {_NVFP4_BLOCK}; got "
                f"{num_rows}, {in_features}, and {out_features}."
            )

        sign_vector = tuple(sign_vector)
        sign_vector_list = list(sign_vector)
        x_col_amax, x_row_amax = _rht_amax(
            x_MK,
            sign_vector_list,
            use_cutedsl,
        )
        x_col_codes, x_col_scale, x_row_codes, x_row_scale = _rht_quantize_row_col(
            x_MK,
            x_col_amax,
            x_row_amax,
            sign_vector_list,
            use_cutedsl,
            use_fast_math,
        )
        output = x.new_empty((*input_shape[:-1], out_features))
        _nvfp4_scaled_mm_out(
            x_row_codes.view(torch.float4_e2m1fn_x2),
            x_row_scale,
            per_tensor_amax_to_scale(x_row_amax),
            weight_qdata_fprop.t(),
            weight_scale_fprop,
            per_tensor_amax_to_scale(weight_amax),
            out=output.view(num_rows, out_features),
        )
        if bias_N is not None:
            output.add_(bias_N)

        has_unsharded_tensor = isinstance(weight_NK, _UnshardedFSDPTensor)
        saved_weight_tensors = (
            (weight_NK,)
            if has_unsharded_tensor
            else (weight_qdata_dgrad, weight_scale_dgrad, weight_amax)
        )
        ctx.save_for_backward(
            x_col_codes,
            x_col_scale,
            x_col_amax,
            sr_seed,
            *saved_weight_tensors,
        )
        ctx.has_unsharded_tensor = has_unsharded_tensor
        ctx.input_shape = input_shape
        ctx.has_bias = bias_N is not None
        ctx.sign_vector = sign_vector
        ctx.use_cutedsl = use_cutedsl
        ctx.use_fast_math = use_fast_math
        return output

    @staticmethod
    @once_differentiable
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        (
            x_col_codes,
            x_col_scale,
            x_col_amax,
            sr_seed,
            *saved_weight_tensors,
        ) = ctx.saved_tensors
        if ctx.has_unsharded_tensor:
            (weight_NK,) = saved_weight_tensors
            if not isinstance(weight_NK, _UnshardedFSDPTensor):
                raise RuntimeError("FSDP restored an incompatible NVFP4 weight")
            operands = weight_NK.operands
            weight_qdata_dgrad = operands.weight_qdata_dgrad
            weight_scale_dgrad = operands.weight_scale_dgrad
            weight_amax = operands.weight_amax
        else:
            (
                weight_qdata_dgrad,
                weight_scale_dgrad,
                weight_amax,
            ) = saved_weight_tensors

        grad_output_MN = grad_output.contiguous().reshape(-1, grad_output.shape[-1])
        sign_vector = list(ctx.sign_vector)
        grad_output_col_amax, grad_output_row_amax = _rht_amax(
            grad_output_MN,
            sign_vector,
            ctx.use_cutedsl,
        )
        (
            grad_output_col_codes,
            grad_output_col_scale,
            grad_output_row_codes,
            grad_output_row_scale,
        ) = _rht_quantize_row_col(
            grad_output_MN,
            grad_output_col_amax,
            grad_output_row_amax,
            sign_vector,
            ctx.use_cutedsl,
            ctx.use_fast_math,
            sr_seed=sr_seed,
        )

        grad_input_MK = _nvfp4_scaled_mm(
            grad_output_row_codes.view(torch.float4_e2m1fn_x2),
            grad_output_row_scale,
            per_tensor_amax_to_scale(grad_output_row_amax),
            weight_qdata_dgrad.t(),
            weight_scale_dgrad,
            per_tensor_amax_to_scale(weight_amax),
        )
        grad_input = grad_input_MK.reshape(ctx.input_shape)
        grad_weight_NK = _nvfp4_scaled_mm(
            grad_output_col_codes.view(torch.float4_e2m1fn_x2),
            grad_output_col_scale,
            per_tensor_amax_to_scale(grad_output_col_amax),
            x_col_codes.view(torch.float4_e2m1fn_x2).t(),
            x_col_scale,
            per_tensor_amax_to_scale(x_col_amax),
        )
        grad_bias_N = grad_output_MN.sum(dim=0) if ctx.has_bias else None
        return (
            grad_input,
            grad_weight_NK,
            None,
            None,
            None,
            None,
            None,
            grad_bias_N,
            None,
            None,
            None,
            None,
        )


spmd.register_local_autograd_function(_NVFP4LinearFunction)


class NVFP4Linear(Linear):
    """Linear with TorchTitan-owned NVFP4 autograd and FSDP operands.

    Runtime seed and RHT-sign-vector handling follows TorchAO's
    ``nvfp4_training.NVFP4Linear``; this class adapts that state to TorchTitan's
    module, sharding, and checkpoint protocols.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        """Drop-in replacement for Linear.Config that builds NVFP4Linear."""

        def __post_init__(self) -> None:
            # NVFP4's Triton kernels need every GEMM dim to be a multiple of
            # 128. in_features / out_features are known at config-build time
            # (the TP degree is not), so reject the model-dim violations up
            # front here; the quantization kernels themselves raise on the
            # per-rank local dims once TP has sharded the weight.
            for name in ("in_features", "out_features"):
                value = getattr(self, name)
                if value % _NVFP4_BLOCK:
                    raise ValueError(
                        f"NVFP4 requires {name} divisible by {_NVFP4_BLOCK}; "
                        f"got {name}={value}. NVFP4 cannot quantize this Linear; "
                        "exclude it from the converter fqns."
                    )

        def build(self, **kwargs):
            # sharding_config is attached by update_from_config before this
            # Config is built, so it is available here but not in
            # __post_init__.
            # slots=True breaks zero-arg super(), so call the parent explicitly.
            instance = Linear.Config.build(self, **kwargs)
            if instance._sharding_config is not None:
                sc = instance._sharding_config
                state_shardings = {
                    **sc.state_shardings,
                    "_sr_seed": SpmdType(
                        {
                            MeshAxisName.DP: spmd.V,
                            MeshAxisName.CP: spmd.V,
                            TP: spmd.V,
                        }
                    ),
                }
                if isinstance(
                    instance,
                    (ColumnParallelLinear, RowParallelLinear),
                ):
                    # The explicit TP class owns its collective in forward.
                    # Making the entire module local would hide that boundary.
                    instance._sharding_config = replace(
                        sc,
                        state_shardings=state_shardings,
                    )
                    return instance

                # Plain NVFP4Linear has no outer collective boundary. Fold its
                # sharding contract into a local region for the opaque autograd
                # function.
                weight_tp = sc.state_shardings["weight"].local_type.get(TP)
                rowwise = (
                    isinstance(weight_tp, spmd.Shard)
                    and weight_tp.dim == instance.weight.ndim - 1
                )
                if rowwise:
                    in_layout = dense_activation_placement(tp=spmd.S(-1), cp=spmd.S(0))
                else:
                    in_layout = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
                # Local-SPMD input layouts are keyed by the forward argument name.
                instance._sharding_config = replace(
                    sc,
                    state_shardings=state_shardings,
                    in_src_shardings={
                        **(sc.in_src_shardings or {}),
                        "input": in_layout,
                    },
                    in_dst_shardings={
                        **(sc.in_dst_shardings or {}),
                        "input": in_layout,
                    },
                    local_spmd=True,
                )
            return instance

    def __init__(self, config: Linear.Config):
        super().__init__(config)
        self.weight = nn.Parameter(
            _LinearShardedTensorWithNVFP4Compute(self.weight.data),
            requires_grad=self.weight.requires_grad,
        )
        # The Philox key is local runtime state and is not checkpointed.
        self.register_buffer("_sr_seed", None, persistent=False)
        self.register_buffer("_rht_sign_vector", None, persistent=False)
        self._rht_sign_vector_tuple = None

    def _parallelize(self, parallel_dims) -> None:
        # spmd_types returns a plain tensor when TP shards the weight. Restore
        # the FSDP extension wrapper before fully_shard() consumes it.
        super()._parallelize(parallel_dims)
        if isinstance(self.weight, _LinearShardedTensorWithNVFP4Compute):
            return
        distributed_weight = self.weight
        wrapped_weight = nn.Parameter(
            _LinearShardedTensorWithNVFP4Compute(distributed_weight.data),
            requires_grad=distributed_weight.requires_grad,
        )
        spmd.assert_type_like(wrapped_weight, distributed_weight)
        self.weight = wrapped_weight

    def _refresh_rht_sign_vector_tuple(self) -> None:
        sign_vector = self._rht_sign_vector
        if sign_vector is not None and hasattr(sign_vector, "to_local"):
            sign_vector = sign_vector.to_local()
        self._rht_sign_vector_tuple = (
            None
            if sign_vector is None or sign_vector.device.type == "meta"
            else tuple(int(value) for value in sign_vector.reshape(-1).tolist())
        )

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        self._refresh_rht_sign_vector_tuple()

    @property
    def rht_sign_vector(self) -> tuple[int, ...]:
        if self._rht_sign_vector_tuple is None:
            self._refresh_rht_sign_vector_tuple()
        if self._rht_sign_vector_tuple is None:
            raise RuntimeError("rht_sign_vector is not materialized")
        return self._rht_sign_vector_tuple

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        dev = buffer_device if buffer_device is not None else self.weight.device
        # Per-rank seed: a plain local tensor (not distributed), so each rank
        # draws its own.
        self._sr_seed = torch.randint(
            -9_223_372_036_854_775_808,
            9_223_372_036_854_775_807,
            (1,),
            dtype=torch.int64,
            device=dev,
        )
        self._rht_sign_vector = torch.tensor(
            _HARDCODED_SIGN_VECTOR,
            dtype=torch.int8,
            device=dev,
        )
        self._refresh_rht_sign_vector_tuple()

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        physical_weight = self.weight
        local_out_features = physical_weight.shape[-2]
        if local_out_features % _NVFP4_BLOCK:
            raise ValueError(
                "NVFP4 requires local out_features divisible by "
                f"{_NVFP4_BLOCK}; got {local_out_features}. Adjust the "
                "Linear out_features or TP degree so quantization blocks "
                "do not span projection boundaries."
            )
        if isinstance(physical_weight, _UnshardedFSDPTensor):
            operands = physical_weight.operands
        else:
            with torch.no_grad():
                high_precision_weight = (
                    physical_weight._tensor
                    if isinstance(physical_weight, _LinearShardedTensorWithNVFP4Compute)
                    else physical_weight
                )
                operands = _quantize_nvfp4_weight(high_precision_weight.flatten(0, -2))
        if self._sr_seed is None:
            raise RuntimeError("NVFP4 stochastic-rounding seed is not materialized")
        output = _NVFP4LinearFunction.apply(
            input,
            weight,
            operands.weight_qdata_fprop,
            operands.weight_scale_fprop,
            operands.weight_qdata_dgrad,
            operands.weight_scale_dgrad,
            operands.weight_amax,
            bias,
            self._sr_seed,
            self.rht_sign_vector,
            _resolve_use_cutedsl(KernelPreference.AUTO),
            True,
        )
        return output
