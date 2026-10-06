# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared process environment defaults for TorchTitan RL jobs."""

from __future__ import annotations

import os
import sys
import warnings


ALLOCATOR_ENV_DEFAULTS: dict[str, str] = {
    # Set both names because supported torch and vLLM versions may read different
    # allocator configuration variables. Expandable segments prevent reserved but
    # unallocated memory from blocking DeepEP buffers and CUDA graph pools.
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

RL_ENV_DEFAULTS: dict[str, str] = {
    **ALLOCATOR_ENV_DEFAULTS,
    "NCCL_DEBUG": "WARN",
}

# The variable XPU launchers set to select devices. Deciding on the accelerator
# any other way means importing torch, which is exactly what these defaults have
# to precede, so the launch env is the only signal available here.
XPU_LAUNCH_ENV = "ZE_AFFINITY_MASK"


def is_xpu_launch_env() -> bool:
    """Whether this process was launched against XPU accelerators."""
    return XPU_LAUNCH_ENV in os.environ


def apply_env_defaults() -> None:
    """Apply TorchTitan RL defaults without replacing existing values."""
    defaults = dict(RL_ENV_DEFAULTS)
    if is_xpu_launch_env():
        # torch's XPU allocator reads the same two variables, and expandable
        # segments invalidate the USM pointers oneCCL registers for collectives,
        # so weight-sync transfers return corrupt data. An explicit user value is
        # still honored below.
        for key in ALLOCATOR_ENV_DEFAULTS:
            defaults.pop(key)
    elif "PYTORCH_CUDA_ALLOC_CONF" not in os.environ and "torch" in sys.modules:
        warnings.warn(
            "The 'torch' module has already been imported. Setting "
            "PYTORCH_CUDA_ALLOC_CONF may not have an effect. For best results, "
            "set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before "
            "importing torch.",
            stacklevel=2,
        )
    for key, value in defaults.items():
        os.environ.setdefault(key, value)
