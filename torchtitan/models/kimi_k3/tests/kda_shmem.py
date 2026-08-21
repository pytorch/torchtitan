# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Skip helper for fla KDA kernels that outgrow a GPU's shared memory.

Under triton 3.8, fla's KDA autotuner picks a configuration requesting 109,184
bytes of dynamic shared memory when ``kda_head_dim == 64``. Consumer Blackwell
(RTX 50-series) offers ``shared_memory_per_block_optin == 101,376`` bytes, so the
launch fails with "Failed to set the allowed dynamic shared memory size". Datacenter
parts (H100/H200: 227 KB) are unaffected, and so is ``kda_head_dim == 128`` --
which is K3's actual value and what FlashKDA requires, so no production config is
affected. Only small debug flavors that shrink head_dim are.

This is an fla/triton/hardware interaction, not something to work around by
editing a validated flavor's dimensions, so the affected tests skip with the
numbers attached rather than being weakened.
"""

from __future__ import annotations

import torch

# Measured request from the failing launch under triton 3.8, kda_head_dim=64.
KDA_HEAD_DIM_64_SHMEM_BYTES = 109184


def kda_shmem_shortfall(required_bytes: int = KDA_HEAD_DIM_64_SHMEM_BYTES) -> int:
    """Bytes by which this device falls short, or 0 if it is sufficient."""
    if not torch.cuda.is_available():
        return 0
    available = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    return max(0, required_bytes - available)


def skip_reason_if_insufficient(
    required_bytes: int = KDA_HEAD_DIM_64_SHMEM_BYTES,
) -> str | None:
    """A unittest skip reason, or None when the device can run the kernel."""
    short = kda_shmem_shortfall(required_bytes)
    if not short:
        return None
    props = torch.cuda.get_device_properties(0)
    return (
        f"fla KDA kernel needs {required_bytes} B of dynamic shared memory but "
        f"{props.name} offers {props.shared_memory_per_block_optin} B "
        f"(short by {short} B). Affects kda_head_dim=64 only; K3 uses 128."
    )
