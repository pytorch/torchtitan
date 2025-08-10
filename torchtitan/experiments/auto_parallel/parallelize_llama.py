# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch

from autoparallel.api import AutoParallel

from torch.distributed import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config_manager import JobConfig, TORCH_DTYPE_MAP
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
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = job_config.training.local_batch_size * dp_degree
        return (
            torch.randint(
                0,
                # job_config.training.vocab_size,
                model.vocab_size,
                (global_batch_size, job_config.training.seq_len),
                device=torch.device("cuda"),
            ),
        )

    # TODO make autop work correctly with different combinations of DP, DP+TP, TP, and support DDP / HSDP
    assert parallel_dims.dp_replicate_enabled is False, "DDP not supported yet"
    assert parallel_dims.cp_enabled is False, "CP not supported yet"
    assert parallel_dims.pp_enabled is False, "PP not supported yet"

    torch._inductor.config.bucket_all_gathers_fx_bucket_size_determinator = (
        lambda bucket_idx: 500 / parallel_dims.tp
    )
    torch._inductor.config.bucket_reduce_scatters_fx_bucket_size_determinator = (
        lambda bucket_idx: 1000 / parallel_dims.tp
    )

    # bail out
    # model = model_fn()
    # return model
    if job_config.experimental.autop_force_bf16:
        logger.info("Forcing bf16 on model")
        model = model.bfloat16()

    param_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce]
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    with AutoParallel(
        model,
        input_fn,
        world_mesh,
        mp_policy=mp_policy,
        compile=job_config.training.compile,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        possible_input_shardings = {
            # maps relative to mesh dim names used in torchtitan
            "dp_replicate": Shard(0),
            "dp_shard": Shard(0),
            "tp": Replicate(),
        }
        # only used if loss parallel is enabled
        possible_output_shardings = {
            # maps relative to mesh dim names used in torchtitan
            "dp_shard": Shard(0),
            "tp": Shard(2),
        }
        assert all(
            name in possible_input_shardings for name in world_mesh.mesh_dim_names
        ), f"Unsupported mesh dim in world mesh, only {possible_input_shardings.keys()} are supported by AutoParallel"
        x_sharding = tuple(
            possible_input_shardings[name] for name in world_mesh.mesh_dim_names
        )
        out_sharding = x_sharding
        if parallel_dims.loss_parallel_enabled:
            out_sharding = tuple(
                possible_output_shardings[name]
                for name in world_mesh.mesh_dim_names
                if name != "dp_replicate"
            )
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])
        t0 = time.time()
        sharding_placement = autop.optimize_placement()
        t1 = time.time()
        logger.info(f"AutoParallel took {t1 - t0} seconds")
        parallel_mod = autop.apply_placement(sharding_placement)

    if parallel_dims.loss_parallel_enabled:

        # current PyTorch's implementation of loss parallel assumes
        # that the DTensor has a 1d device mesh. This is not true
        # in our case, but we can work around it by adding
        # casting the output to a DTensor on a 1d device mesh.
        # We should just use AutoParallel to do this for us, but
        # it would require putting the loss inside the model as well
        def _return_as_dtensor_for_loss_parallel(module, args, output):
            return torch.distributed.tensor.DTensor.from_local(
                output, world_mesh["tp"], (Shard(2),)
            )

        # not keeping a reference to the hook, don't plan on
        # removing it at any point
        parallel_mod.register_forward_hook(_return_as_dtensor_for_loss_parallel)

    return parallel_mod
