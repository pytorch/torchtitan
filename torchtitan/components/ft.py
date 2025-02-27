# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from typing import Optional

from torchtitan.config_manager import JobConfig

if importlib.util.find_spec("torchft") is not None:
    import torchft as ft

    has_torchft = True
else:
    has_torchft = False


class FTManager:
    def __init__(
        self,
        manager: Optional["ft.Manager"],
        group_size: int = 1,
        replica_id: int = 0,
    ) -> None:
        self._manager = manager
        self.group_size = group_size
        self.replica_id = replica_id

    @property
    def enabled(self) -> bool:
        return self._manager is not None

    @property
    def manager(self) -> "ft.Manager":
        assert self._manager is not None
        return self._manager

    def get_dp_rank(self, dp_degree: int, dp_rank: int) -> int:
        return dp_degree * self.replica_id + dp_rank

    def get_dp_degree(self, dp_degree: int) -> int:
        return dp_degree * self.group_size


def init_ft_manager(job: JobConfig) -> FTManager:
    """Initialize the FT manager if TorchFT is enabled.

    Args:
        job (JobConfig): The job configuration.

    Returns:
        Optional[ft.Manager]: The FT manager if TorchFT is enabled, otherwise None.
    """
    if not job.experimental.enable_torchft:
        return FTManager(None)

    if not has_torchft:
        raise ImportError("torchft is not installed. Please install it.")

    if job.experimental.ft_min_replica_size < 1:
        raise ValueError("At least one FT replica is required.")

    pg = ft.ProcessGroupBabyNCCL()

    return FTManager(
        ft.Manager(
            pg=pg,
            min_replica_size=job.experimental.ft_min_replica_size,
            load_state_dict=None,
            state_dict=None,
            use_async_quorum=True,
            replica_id=f"torchtitan_ft_{job.experimental.ft_replica_id}",
        ),
        group_size=job.experimental.ft_group_size,
        replica_id=job.experimental.ft_replica_id,
    )
