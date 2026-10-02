# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os

import torch

logger = logging.getLogger(__name__)


_batch_invariant_enabled: bool = False
_batch_invariant_extra_lib: torch.library.Library | None = None


def is_in_batch_invariant_mode() -> bool:
    """Return whether batch-invariant mode is active."""
    return _batch_invariant_enabled


def _sum_dim_batch_invariant(
    input: torch.Tensor,
    dim: list[int] | None,
    keepdim: bool = False,
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Implement ``sum.dim_IntList`` using fixed-schedule reductions."""
    from batch_invariant_ops import mean_dim

    dims = tuple(range(input.ndim)) if not dim else tuple(dim)
    dims = tuple(axis if axis >= 0 else axis + input.ndim for axis in dims)
    if len(set(dims)) != len(dims) or any(
        axis < 0 or axis >= input.ndim for axis in dims
    ):
        raise IndexError(f"invalid reduction axes {dim} for a {input.ndim}D tensor")

    output_dtype = dtype or (
        input.dtype if input.is_floating_point() or input.is_complex() else torch.int64
    )
    num_reduced_elements = math.prod(input.shape[axis] for axis in dims)
    if num_reduced_elements == 0:
        output_shape = list(input.shape)
        for axis in sorted(dims, reverse=True):
            if keepdim:
                output_shape[axis] = 1
            else:
                output_shape.pop(axis)
        return torch.zeros(output_shape, dtype=output_dtype, device=input.device)

    accumulation_dtype = torch.int64 if output_dtype == torch.bool else output_dtype
    result = (
        input if input.dtype == accumulation_dtype else input.to(accumulation_dtype)
    )
    if result.dtype in {torch.float16, torch.bfloat16, torch.float32}:
        for axis in sorted(dims, reverse=not keepdim):
            result = mean_dim(result, axis, keepdim=keepdim)
        return result * num_reduced_elements

    # Integer, boolean, complex, and float64 reductions do not need the
    # low-precision Triton kernel. A prefix scan fixes their reduction order.
    for axis in sorted(dims, reverse=not keepdim):
        result = torch.cumsum(result, dim=axis, dtype=output_dtype).select(axis, -1)
        if keepdim:
            result = result.unsqueeze(axis)
    return result.to(output_dtype)


def set_batch_invariance(enable: bool) -> None:
    """Enable batch-invariant mode for reproducible RL training.

    Delegates ATen operator overrides to the ``batch_invariant_ops`` package
    and extends them to ``sum.dim_IntList`` and ``bmm``, producing
    bit-identical results for the same input regardless of batch composition.

    On top of that, this function applies torchtitan-specific settings:
    - NCCL env vars for deterministic inter-GPU collectives
    - Disables reduced-precision reductions and TF32

    Note: callers must set ``debug.deterministic=True`` separately.
    """
    global _batch_invariant_enabled, _batch_invariant_extra_lib
    if not enable or _batch_invariant_enabled:
        return

    # Register batch-invariant ATen overrides via upstream package
    # https://github.com/thinking-machines-lab/batch_invariant_ops
    from batch_invariant_ops import enable_batch_invariant_mode as _upstream_enable
    from vllm.model_executor.determinism.batch_invariant import (  # pyrefly: ignore[missing-import]
        bmm_batch_invariant,
    )

    _upstream_enable()
    # batch_invariant_ops does not yet override sum or bmm. Compose sum from
    # its fixed-schedule mean and reuse vLLM's fixed-schedule bmm kernel.
    accelerator = torch.accelerator.current_accelerator()
    if accelerator is None:
        raise RuntimeError("batch-invariant mode requires an accelerator")
    dispatch_key = accelerator.type.upper()
    _batch_invariant_extra_lib = torch.library.Library("aten", "IMPL")
    _batch_invariant_extra_lib.impl(
        "aten::sum.dim_IntList",
        _sum_dim_batch_invariant,
        dispatch_key,
        allow_override=True,
    )
    _batch_invariant_extra_lib.impl(
        "aten::bmm",
        bmm_batch_invariant,
        dispatch_key,
        allow_override=True,
    )

    # Set NCCL env vars for deterministic inter-GPU collectives.
    # Must be set BEFORE dist.init_process_group.
    # Reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/determinism/batch_invariant.py
    os.environ["NCCL_LAUNCH_MODE"] = "GROUP"  # Fixed kernel launch ordering
    os.environ[
        "NCCL_COLLNET_ENABLE"
    ] = "0"  # Disable SHARP (non-deterministic IB HW reduce)
    os.environ[
        "NCCL_NVLS_ENABLE"
    ] = "0"  # Disable NVLink SHARP (non-deterministic NVSwitch HW reduce)
    os.environ[
        "NCCL_P2P_NET_DISABLE"
    ] = "1"  # Disable P2P to avoid transport-dependent accumulation order
    os.environ[
        "NCCL_MIN_NCHANNELS"
    ] = "1"  # Single channel to prevent split-interleave reordering
    os.environ[
        "NCCL_MAX_NCHANNELS"
    ] = "1"  # Single channel to prevent split-interleave reordering
    os.environ["NCCL_PROTO"] = "Simple"  # LL/LL128 protocols may reorder reductions
    os.environ[
        "NCCL_ALGO"
    ] = "allreduce:tree"  # Deterministic reduction order across ranks
    os.environ[
        "NCCL_NTHREADS"
    ] = "1"  # Single thread to eliminate scheduling non-determinism
    os.environ[
        "NCCL_SOCKET_NTHREADS"
    ] = "1"  # Single socket thread to eliminate scheduling non-determinism

    # Disable reduced-precision reductions: these allow cuBLAS to use
    # lower-precision accumulation that can round differently depending
    # on batch size / tile decomposition.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Disable TF32 for exact fp32 accumulation
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    _batch_invariant_enabled = True

    logger.info(
        "Batch-invariant mode enabled: mm, addmm, bmm, sum.dim_IntList, "
        "_log_softmax, mean.dim overridden with fixed-schedule kernels; "
        "reduced-precision reductions and TF32 disabled"
    )
