# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
from autoparallel.api import AutoParallel
from autoparallel.auto_bucketing import configure_inductor_for_autobucketing
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.autoparallel.configs import AutoParallelCompileConfig
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def parallelize_llama(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: AutoParallelCompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """

    # TODO(whc)
    # I do this because otherwise sometimes inductor will skip re-running passes like comms reordering
    torch._inductor.config.force_disable_caches = True
    # this is necessary for working with reordering passes. Just leave it set for all the jobs for now.
    torch._inductor.config.allow_buffer_reuse = False

    # allow configuring inductor comms optimizations from torchtitan commandline
    configure_inductor_for_autobucketing(compile_config.comms_bucket_reorder_strategy)

    dense_names = ["dp_replicate", "fsdp", "tp"]
    dense_names = [
        name
        for name in dense_names
        if parallel_dims.get_optional_mesh(name) is not None
    ]
    dense_mesh = parallel_dims.get_mesh(dense_names)

    def input_fn():
        global_batch_size = training.global_batch_size
        if global_batch_size < 0:
            # This global batch size results in 1 gradient accumulation
            # step.
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = training.local_batch_size * dp_degree
        return (
            torch.randint(
                0,
                model.config.vocab_size,
                (global_batch_size, training.seq_len),
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
    if compile_config.autop_force_bf16:
        logger.info("Forcing bf16 on model")
        model = model.bfloat16()

    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    with AutoParallel(
        model,
        input_fn,
        dense_mesh,
        mp_policy=mp_policy,
        compile=compile_config,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        possible_input_shardings = {
            # maps relative to mesh dim names used in torchtitan
            "dp_replicate": Shard(0),
            "fsdp": Shard(0),
            "tp": Replicate(),
        }
        # only used if loss parallel is enabled
        possible_output_shardings = {
            # maps relative to mesh dim names used in torchtitan
            "fsdp": Shard(0),
            "tp": Shard(2),
        }
        assert all(
            name in possible_input_shardings for name in dense_mesh.mesh_dim_names
        ), f"Unsupported mesh dim in world mesh, only {possible_input_shardings.keys()} are supported by AutoParallel"
        x_sharding = tuple(
            possible_input_shardings[name] for name in dense_mesh.mesh_dim_names
        )
        out_sharding = x_sharding
        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not parallelism.disable_loss_parallel
        )
        if loss_parallel_enabled:
            out_sharding = tuple(
                possible_output_shardings[name]
                for name in dense_mesh.mesh_dim_names
                if name != "dp_replicate"
            )
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([out_sharding])
        t0 = time.time()
        sharding_placement = autop.optimize_placement(verbose=False)
        t1 = time.time()
        logger.info(f"AutoParallel took {t1 - t0} seconds")
        parallel_mod = autop.apply_placement(sharding_placement)

    if loss_parallel_enabled:

        # current PyTorch's implementation of loss parallel assumes
        # that the DTensor has a 1d device mesh. This is not true
        # in our case, but we can work around it by adding
        # casting the output to a DTensor on a 1d device mesh.
        # We should just use AutoParallel to do this for us, but
        # it would require putting the loss inside the model as well
        def _return_as_dtensor_for_loss_parallel(module, args, output):
            return torch.distributed.tensor.DTensor.from_local(
                output, dense_mesh["tp"], (Shard(2),)
            )

        # not keeping a reference to the hook, don't plan on
        # removing it at any point
        parallel_mod.register_forward_hook(_return_as_dtensor_for_loss_parallel)

    return parallel_mod
