# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import weakref

import torch
from torch import Tensor

# Rounding that wastes less pinned memory than this is accepted, e.g. for small
# allocations below pinned_max_round_threshold_mb or a small model. Power-of-two
# rounding of large moments wastes up to their size, many GiB per rank.
_PINNED_ROUNDING_SLACK_BYTES = 2**30


# TODO: Use a supported host-mapped CUDA tensor once PyTorch has one (upstream
# issue not filed yet). This relies on two undocumented behaviors: PyTorch
# trusts the device given with __cuda_array_interface__, and pinned host memory
# is usable by kernels at its host address.
class _CUDAHostMemoryView:
    """Expose a pinned CPU tensor to CUDA.

    ``torch.as_tensor`` turns this object into a CUDA tensor that aliases the
    pinned memory through ``__cuda_array_interface__``. That CUDA tensor keeps
    this object, and so the CPU tensor, alive.

    The interface has no bfloat16 type code, so it describes the memory as raw
    bytes. The caller views the bytes as the target dtype.
    """

    def __init__(self, host: Tensor, device: torch.device) -> None:
        self.host = host
        # The caching host allocator reuses the memory once ``host`` is freed, but
        # it does not see kernels that use it through this view. Wait for them
        # first. This runs before ``host`` is released. Skipped at exit, when the
        # CUDA context may already be gone.
        weakref.finalize(self, torch.accelerator.synchronize, device).atexit = False
        # CUDA Array Interface v3 (the current spec version): one uint8 ("|u1")
        # per byte, at this address, writable (read-only flag False).
        self.__cuda_array_interface__ = {
            "shape": (host.nbytes,),
            "typestr": "|u1",
            "data": (host.data_ptr(), False),
            "version": 3,
        }


def cuda_pinned_host_zeros_like(
    local: Tensor, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Return zeros shaped like the CUDA tensor ``local`` in pinned CPU memory.

    Returns the memory as a CUDA tensor and as the CPU tensor it aliases.
    ``local`` must not be empty.
    """
    # The pages land on the NUMA node this process is bound to.
    host = torch.zeros(local.shape, dtype=dtype, pin_memory=True)
    # The CUDA Array Interface normally describes device memory. Pinned host
    # memory is also addressable by CUDA kernels (unified addressing), so the
    # tensor gets an explicit device.
    raw = torch.as_tensor(_CUDAHostMemoryView(host, local.device), device=local.device)
    return raw.view(dtype).view(local.shape), host


def pinned_host_bytes() -> int:
    """Return the bytes of pinned blocks handed out by the CUDA host allocator."""
    # The stats are empty before CUDA is initialized, when nothing is pinned.
    return torch.cuda.host_memory_stats().get("active_bytes.current", 0)


def check_pinned_host_rounding(*, requested_bytes: int, bytes_before: int) -> None:
    """Raise if pinned allocations since ``bytes_before`` were rounded up.

    ``bytes_before`` is ``pinned_host_bytes()`` before the allocations. By default
    the CUDA host allocator rounds each allocation up to a power of two.
    """
    reserved_bytes = pinned_host_bytes() - bytes_before
    if reserved_bytes > requested_bytes + _PINNED_ROUNDING_SLACK_BYTES:
        raise ValueError(
            "Pinned host memory was rounded up: requested "
            f"{requested_bytes / 2**30:.2f} GiB, reserved "
            f"{reserved_bytes / 2**30:.2f} GiB. Add pinned_max_round_threshold_mb:1 "
            "to PYTORCH_ALLOC_CONF."
        )
