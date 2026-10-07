# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

logger = logging.getLogger(__name__)


def maybe_apply_numa_binding(device_index: int, device_type: str) -> None:
    """Bind this process to the NUMA node local to its CUDA device.

    Call this before allocating pinned CPU memory so first-touch places the
    allocation on the device-local NUMA node. Non-CUDA devices are a no-op.
    """
    if device_type != "cuda":
        return

    from torch.numa.binding import (
        _maybe_apply_numa_binding_to_current_process,
        AffinityMode,
        NumaOptions,
    )

    _maybe_apply_numa_binding_to_current_process(
        device_index=device_index,
        numa_options=NumaOptions(
            affinity_mode=AffinityMode.NODE,
            should_fall_back_if_binding_fails=True,
        ),
    )
    logger.info("NUMA binding applied for GPU %d", device_index)
