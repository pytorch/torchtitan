# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RemoteTensor: Abstraction for cross-process tensor access via RDMA.
"""

from typing import Any

import torch
from monarch.rdma import RDMABuffer


class RemoteTensor:
    """
    A handle to a tensor owned by another process.

    Uses RDMA transport for cross-process data transfer. The handle is
    serializable and can be sent between processes via Monarch actor messages.

    Usage:
        # Sender creates handle
        handle = RemoteTensor(tensor, owner="sender_0")

        # ... handle sent to receiver via actor message ...

        # Receiver uses handle to transfer data
        handle.read_into(local_tensor)
    """

    def __init__(self, tensor: torch.Tensor, owner: str) -> None:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        self.shape = tuple(tensor.shape)
        self.stride = tuple(tensor.stride())
        self.dtype = tensor.dtype
        self.owner = owner

        byte_view = tensor.view(torch.uint8).flatten()
        self.rdma_buffer: Any = RDMABuffer(byte_view)

    def read_into(
        self,
        dst: torch.Tensor,
        *,
        timeout: int = 3,
    ) -> None:
        """
        Copy data FROM this remote tensor INTO local dst tensor.

        Args:
            dst: Local destination tensor (must match shape/dtype)
            timeout: Timeout in seconds
        """
        byte_view = dst.view(torch.uint8).flatten()
        self.rdma_buffer.read_into(byte_view, timeout=timeout).get()
