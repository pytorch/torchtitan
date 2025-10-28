# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager

import torch
from torch.distributed.tensor import DTensor, Replicate
from torch.utils._pytree import tree_map

from torchtitan.config import JobConfig


@contextmanager
def disable_compile(job_config: JobConfig):
    """Context manager to temporarily disable compilation."""
    original_value = job_config.compile.enable
    job_config.compile.enable = False
    try:
        yield
    finally:
        job_config.compile.enable = original_value


def parallelize_inputs(world_mesh, args, kwargs):
    def to_dtensor(tensor):
        if isinstance(tensor, torch.Tensor):
            return DTensor.from_local(tensor, world_mesh["tp"], [Replicate()])
        return tensor

    dt_args = tree_map(to_dtensor, args)

    # TODO: When using flex_attention, BlockMask would show up in kwargs,
    # and it's unclear how to convert it to DTensor. If I use to_dtensor,
    # it would fail with Dynamo Error: P2011360347
    # dt_kwargs = tree_map(to_dtensor, kwargs)

    dt_kwargs = kwargs

    return dt_args, dt_kwargs
