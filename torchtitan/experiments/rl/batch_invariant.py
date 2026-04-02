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
    from torchtitan.experiments.rl.batch_invariant import enable_batch_invariant
    enable_batch_invariant()
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enable / disable
# ---------------------------------------------------------------------------

_enabled: bool = False


def is_in_batch_invariant_mode() -> bool:
    """Return whether batch-invariant mode is active."""
    return _enabled


def enable_batch_invariant() -> None:
    """Enable batch-invariant mode for reproducible RL training.

    Delegates ATen operator overrides (``mm``, ``addmm``, ``_log_softmax``,
    ``mean.dim``) to the ``batch_invariant_ops`` package, which registers
    Triton kernels with a fixed tile iteration order producing bit-identical
    results for the same input regardless of batch composition.

    On top of that, this function applies torchtitan-specific settings:
    - NCCL env vars for deterministic inter-GPU collectives
    - Disables reduced-precision reductions and TF32

    Note: callers must set ``debug.deterministic=True`` separately (enforced
    by ``RLTrainer.Config.__post_init__``), which handles
    ``torch.use_deterministic_algorithms`` via ``set_determinism()``.
    """
    global _enabled
    if _enabled:
        return

    # Register batch-invariant ATen overrides via upstream package
    # https://github.com/thinking-machines-lab/batch_invariant_ops
    from batch_invariant_ops import enable_batch_invariant as _upstream_enable

    _upstream_enable()

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

    _enabled = True

    logger.info(
        "Batch-invariant mode enabled: mm, addmm, _log_softmax, mean.dim "
        "overridden with Triton kernels (via batch_invariant_ops); "
        "reduced-precision reductions and TF32 disabled"
    )
