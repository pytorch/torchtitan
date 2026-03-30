# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Batch-invariant mode for reproducible RL training.

When enabled, replaces ``mm``, ``addmm``, ``log_softmax``, and ``mean.dim``
with Triton kernels that use a fixed tile iteration order, producing
bit-identical results for the same input regardless of batch composition.

Also disables reduced-precision reductions and TF32 to prevent
batch-size-dependent rounding.

The kernels are registered via ``torch.library.Library("aten", "IMPL")``
so they are transparent to the model code — no changes needed in the model
definition.


Usage:
    from torchtitan.experiments.rl.batch_invariant import enable_batch_invariant_mode
    enable_batch_invariant_mode()
"""

import logging
import os
from typing import Any

import torch

# https://github.com/thinking-machines-lab/batch_invariant_ops.
from batch_invariant_ops import (
    disable_batch_invariant_mode as _upstream_disable,
    enable_batch_invariant_mode as _upstream_enable,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enable / disable
# ---------------------------------------------------------------------------

_SAVED_STATE: dict[str, Any] | None = None
_enabled: bool = False


def is_batch_invariant_mode_enabled() -> bool:
    """Return whether batch-invariant mode is active."""
    return _enabled


def enable_batch_invariant_mode() -> None:
    """Enable batch-invariant mode for reproducible RL training.

    Delegates ATen operator overrides (``mm``, ``addmm``, ``_log_softmax``,
    ``mean.dim``) to the ``batch_invariant_ops`` package, which registers
    Triton kernels with a fixed tile iteration order producing bit-identical
    results for the same input regardless of batch composition.

    On top of that, this function applies torchtitan-specific settings:
    - NCCL env vars for deterministic inter-GPU collectives
    - Disables reduced-precision reductions and TF32
    - Enables deterministic algorithms

    Safe to call multiple times (idempotent).
    """
    global _SAVED_STATE, _enabled
    if _enabled:
        return

    # Register batch-invariant ATen overrides via upstream package
    _upstream_enable()

    # Save current state for restoration
    _SAVED_STATE = {
        "bf16": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "fp16": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
    }

    # Set NCCL env vars for deterministic inter-GPU collectives.
    # Must be set BEFORE dist.init_process_group.
    os.environ["NCCL_ALGO"] = "Ring"  # Fixed summation order (Tree may vary)
    os.environ["NCCL_MIN_NCHANNELS"] = "1"  # Single channel to avoid split interleaving
    os.environ["NCCL_MAX_NCHANNELS"] = "1"
    os.environ["NCCL_PROTO"] = "Simple"  # LL/LL128 may reorder reductions
    os.environ[
        "NCCL_COLLNET_ENABLE"
    ] = "0"  # Disable SHARP (non-deterministic HW reduce)
    os.environ[
        "NCCL_NVLS_ENABLE"
    ] = "0"  # Disable NVLink SHARP (non-deterministic HW reduce)

    # Disable reduced-precision reductions: these allow cuBLAS to use
    # lower-precision accumulation that can round differently depending
    # on batch size / tile decomposition.
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Disable TF32 for exact fp32 accumulation
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Enable deterministic algorithms for run-to-run reproducibility.
    torch.use_deterministic_algorithms(True)

    _enabled = True

    logger.info(
        "Batch-invariant mode enabled: mm, addmm, _log_softmax, mean.dim "
        "overridden with Triton kernels (via batch_invariant_ops); "
        "reduced-precision reductions and TF32 disabled"
    )


def disable_batch_invariant_mode() -> None:
    """Unregister batch-invariant ATen overrides and restore settings."""
    global _SAVED_STATE, _enabled

    # Unregister upstream ATen overrides
    _upstream_disable()

    # Restore saved settings
    if _SAVED_STATE is not None:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            _SAVED_STATE["bf16"]
        )
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (
            _SAVED_STATE["fp16"]
        )
        torch.backends.cuda.matmul.allow_tf32 = _SAVED_STATE["tf32_matmul"]
        torch.backends.cudnn.allow_tf32 = _SAVED_STATE["tf32_cudnn"]
        torch.use_deterministic_algorithms(_SAVED_STATE["deterministic"])
    _SAVED_STATE = None
    _enabled = False
