# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Tuple

import torch
from torchtitan.logging import logger
from torchtitan.parallelisms import ParallelDims


def check_if_feature_in_pytorch(
    feature_name: str,
    pull_request: str,
    min_nightly_version: Optional[str] = None,
) -> None:
    if "git" in torch.__version__:  # pytorch is built from source
        # notify users to check if the pull request is included in their pytorch
        logger.warning(
            "detected that the pytorch is built from source. Please make sure the PR "
            f"({pull_request_link}) is included in pytorch for correct {feature_name}."
        )
    elif min_nightly_version is not None and torch.__version__ < min_nightly_version:
        logger.warning(
            f"detected that the pytorch version {torch.__version__} is older than "
            f"{min_nightly_version}. Please upgrade a newer version to include the "
            f"change in ({pull_request_link}) for correct {feature_name}."
        )


def get_fully_shard_mesh_dim_names(parallel_dims: ParallelDims) -> Tuple[str, ...]:
    """
    Returns the names of which mesh dims that should be passed to the fully_shard()
    function according to the `parallel_dims`. We assume that (dp_shard_enabled == True)
    or (cp_enabled == True) is True when users call this function.
    """
    # Table for composable parallelisms that fully_shard() should be applied to:
    #   parallelisms |  dp_shard_enabled|  dp_replicate_enabled|  cp_enabled|  mesh_dim_names
    #   ------------ |  --------------- |  ------------------- |  --------- |  ---------------
    #   no-parallel  |  False           |  False               |  False     |  error
    #   ddp          |  False           |  True                |  False     |  error
    #   ddp + cp     |  False           |  True                |  True      |  error
    #   cp           |  False           |  False               |  True      |  ("cp")
    #   hsdp         |  True            |  True                |  False     |  ("dp_replicate,  "dp_shard")
    #   hsdp + cp    |  True            |  True                |  True      |  ("dp_replicate", "dp_cp")
    #   fsdp         |  True            |  False               |  False     |  ("dp")
    #   fsdp + cp    |  True            |  False               |  True      |  ("dp_cp")

    if not parallel_dims.dp_shard_enabled:
        assert parallel_dims.cp_enabled, (
            "get_fully_shard_mesh_dim_names() should not be used when "
            f"{'DDP' if parallel_dims.dp_replicate_enabled else 'no parallelism'} "
            "is enabled."
        )

        if parallel_dims.dp_replicate_enabled:
            # Composability of DDP + CP is not supported.
            raise RuntimeError("Composability of DDP + CP is not supported.")

        return ("cp",)
    else:
        # FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.cp_enabled:
            if parallel_dims.dp_replicate_enabled:
                return ("dp_replicate", "dp_cp")
            else:
                return("dp_cp",)
        else:
            if parallel_dims.dp_replicate_enabled:
                return ("dp_replicate", "dp_shard")
            else:
                return ("dp",)
