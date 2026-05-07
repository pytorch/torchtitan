# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from tempfile import NamedTemporaryFile

import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import DataParallelMeshDims

from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flex_shard,
    per_param_placements,
)


@contextmanager
def single_rank_cpu_mesh() -> Iterator:
    """Create a single-rank CPU mesh for normal pytest unit tests."""
    created_pg = False
    with NamedTemporaryFile() as store:
        if not dist.is_initialized():
            dist.init_process_group(
                "gloo",
                init_method=f"file://{store.name}",
                rank=0,
                world_size=1,
                timeout=timedelta(seconds=20),
            )
            created_pg = True
        try:
            yield init_device_mesh("cpu", (1,), mesh_dim_names=("fsdp",))
        finally:
            if created_pg and dist.is_initialized():
                dist.destroy_process_group()


def flex_shard_cpu(
    model: nn.Module,
    mesh,
    buckets: list[BucketSpec] | None = None,
) -> nn.Module:
    """Apply FlexShard with CPU-compatible eager settings."""
    if buckets is None:
        buckets = [BucketSpec(["*"], reshard_after_forward=False)]
    return flex_shard(
        model,
        mesh,
        DataParallelMeshDims(shard="fsdp"),
        shard_placement_fn=per_param_placements,
        buckets=buckets,
    )


def flex_shard_tiny_model(mesh) -> nn.Sequential:
    """Return a small CPU model sharded into per-layer buckets."""
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    flex_shard_cpu(
        model,
        mesh,
        buckets=[
            BucketSpec(["0.*"], reshard_after_forward=False),
            BucketSpec(["2.*"], reshard_after_forward=False),
        ],
    )
    return model
