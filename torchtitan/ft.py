# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from dataclasses import dataclass
from typing import Optional

from torchtitan.config_manager import JobConfig

if importlib.util.find_spec("torchft") is not None:
    import torchft as ft

    has_torchft = True
else:
    has_torchft = False


def init_ft_manager(job: JobConfig) -> Optional["ft.Manager"]:
    """Initialize the FT manager if TorchFT is enabled.

    Args:
        job (JobConfig): The job configuration.

    Returns:
        Optional[ft.Manager]: The FT manager if TorchFT is enabled, otherwise None.
    """
    if not job.experimental.enable_torchft:
        return None

    if not has_torchft:
        raise ImportError("torchft is not installed. Please install it.")

    pg = ft.ProcessGroupBabyNCCL()
    return ft.Manager(
        pg=pg,
        min_replica_size=1,
        load_state_dict=None,
        state_dict=None,
        use_async_quorum=True,
        replica_id=f"torchtitan_ft_{job.experimental.ft_replica_id}",
    )
