# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch

from autoparallel.api import AutoParallel

from torch.distributed import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims

from torchtitan.tools.logging import logger


def parallelize_llama(
    model,
    world_mesh: DeviceMesh,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    def input_fn():
        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            dp_degree = world_mesh["dp"].size()
            global_batch_size = job_config.training.local_batch_size * dp_degree
        return torch.rand(
            (global_batch_size, job_config.training.seq_len), device="cuda"
        )

    # TODO make autop work correctly with different combinations of DP, DP+TP, TP, and support DDP / HSDP
    assert (
        len(world_mesh.shape) == 2
    ), "Only support 2D mesh (DP, TP) for now- OK if one has size=1"
    assert parallel_dims.dp_shard_enabled is True, "DDP not supported yet"
    assert parallel_dims.dp_replicate_enabled is False, "DDP not supported yet"
    assert parallel_dims.cp_enabled is False, "CP not supported yet"
    assert parallel_dims.pp_enabled is False, "PP not supported yet"

    # TODO: there are multiple float8 recipes, this just hardcodes one
    enable_float8_linear = "float8" in job_config.model.converters
    if enable_float8_linear:
        import copy
        from torchao.float8.float8_linear_utils import convert_to_float8_training
        from torchao.float8.config import Float8LinearConfig
        model = convert_to_float8_training(copy.deepcopy(model), config=Float8LinearConfig())

    # bail out
    # model = model_fn()
    # return model

    autop = AutoParallel(model, input_fn, world_mesh)
    autop.add_parameter_memory_constraint(low=None, high=None)

    x_sharding = (Shard(0), Replicate())

    autop.add_input_constraints([x_sharding])
    autop.add_output_constraints([x_sharding])
    t0 = time.time()
    sharding_placement = autop.optimize_placement()
    t1 = time.time()
    logger.info(f"AutoParallel took {t1 - t0} seconds")
    parallel_mod = autop.apply_placement(sharding_placement)

    if job_config.training.compile:
        torch._inductor.config.reorder_for_peak_memory = False
        parallel_mod = torch.compile(parallel_mod, fullgraph=True)

    return parallel_mod
