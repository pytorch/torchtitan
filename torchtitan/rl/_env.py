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


RL_ENV_DEFAULTS: dict[str, str] = {
    # Expandable segments prevent reserved but unallocated memory from blocking
    # DeepEP buffers and CUDA graph pools.
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "NCCL_DEBUG": "WARN",
}


def apply_env_defaults() -> None:
    """Apply TorchTitan RL defaults without replacing existing values."""
    allocator_conf = os.environ.get("PYTORCH_ALLOC_CONF")
    legacy_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if legacy_conf is not None and legacy_conf != allocator_conf:
        raise ValueError(
            "Use PYTORCH_ALLOC_CONF and unset PYTORCH_CUDA_ALLOC_CONF; "
            "the legacy variable is only supported as an identical compatibility copy."
        )
    if allocator_conf is None and "torch" in sys.modules:
        warnings.warn(
            "The 'torch' module has already been imported. Setting "
            "the default PYTORCH_ALLOC_CONF may not have an effect. "
            "Set PYTORCH_ALLOC_CONF before importing torch.",
            stacklevel=2,
        )
    for key, value in RL_ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    # TODO: remove this mirror after updating pinned vLLM and its legacy readers.
    # 9f03b510c restores max_split_size_mb from the legacy name; fixed upstream:
    # https://github.com/vllm-project/vllm/pull/58430
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF",
        os.environ["PYTORCH_ALLOC_CONF"],
    )
