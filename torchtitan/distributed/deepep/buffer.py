# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared DeepEP buffer management.

This module provides the SINGLE source of truth for DeepEP buffer instance.
Both autotune (fused_a2a.py) and training (deepep.py) import from here to
ensure they use the SAME buffer. This is critical because DeepEP's RDMA
connections are tied to the buffer - using different buffers between
autotune and training causes timeout errors in internode mode.
"""

from typing import Optional

import torch
from torch.distributed import ProcessGroup

try:
    from deep_ep import Buffer
except ImportError as e:
    raise ImportError(
        "DeepEP is required for this module. "
        "Install from: https://github.com/deepseek-ai/deepep"
    ) from e


# Global buffer - SINGLE instance shared by autotune and training
_buffer: Optional[Buffer] = None


def get_hidden_bytes(x: torch.Tensor) -> int:
    """Calculate the number of hidden bytes for a tensor.

    Args:
        x: Input tensor with shape [..., hidden_size]

    Returns:
        Number of bytes for the hidden dimension
    """
    return x.size(1) * max(x.element_size(), 2)


def get_buffer(group: ProcessGroup, hidden_bytes: int) -> Buffer:
    """Get or create a buffer for all-to-all communication.

    This function manages a SINGLE global buffer instance. Both autotune
    and training MUST call this function to ensure they share the same
    buffer and RDMA connections.

    Args:
        group: Process group for communication
        hidden_bytes: Number of hidden bytes needed

    Returns:
        Shared communication buffer
    """
    global _buffer
    num_nvl_bytes, num_rdma_bytes = 0, 0
    for config in (
        Buffer.get_dispatch_config(group.size()),
        Buffer.get_combine_config(group.size()),
    ):
        num_nvl_bytes = max(
            config.get_nvl_buffer_size_hint(hidden_bytes, group.size()), num_nvl_bytes
        )
        num_rdma_bytes = max(
            config.get_rdma_buffer_size_hint(hidden_bytes, group.size()), num_rdma_bytes
        )

    # Allocate buffer if not existed or not enough buffer
    # NOTE: the adaptive routing configuration of the network **must be off**
    if (
        _buffer is None
        or _buffer.group != group
        or _buffer.num_nvl_bytes < num_nvl_bytes
        or _buffer.num_rdma_bytes < num_rdma_bytes
    ):
        _buffer = Buffer(group, num_nvl_bytes, num_rdma_bytes)

    return _buffer
