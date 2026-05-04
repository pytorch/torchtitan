# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoParallel-based parallelization for Llama3.

Uses AutoParallelGraph to apply solver-based SPMD sharding, then feeds
the pre-built JointWithDescriptors into graph_trainer's AOT compilation
pipeline (CompiledModule + joint_graph_builder).
"""

import time

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Replicate, Shard

from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.autoparallel_api import (
    AutoParallelGraph,
    AutoParallelModelOutput,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.tools.logging import logger


def parallelize_autoparallel_llama(
    model,
    *,
    loss_fn,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply AutoParallelGraph SPMD sharding to Llama3.

    Returns a sharded model carrying AutoParallel train-step metadata for
    graph_trainer's aot_fx_trace path.
    """
    if compile_config.mode != "aot_fx_trace":
        raise ValueError(
            "AutoParallel graph_trainer requires compile.mode=aot_fx_trace"
        )

    if loss_fn is None:
        raise ValueError("AutoParallel graph_trainer requires a configured loss_fn")

    if parallel_dims.dp_replicate_enabled:
        raise ValueError("AutoParallel Llama3 does not support DDP yet")
    if parallel_dims.cp_enabled:
        raise ValueError("AutoParallel Llama3 does not support CP yet")
    if parallel_dims.pp_enabled:
        raise ValueError("AutoParallel Llama3 does not support PP yet")

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
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = training.local_batch_size * dp_degree
        tokens = torch.randint(
            0,
            model.config.vocab_size,
            (global_batch_size, training.seq_len),
            device=torch.device("cuda"),
        )
        return tokens

    param_dtype = TORCH_DTYPE_MAP[training.mixed_precision_param]
    reduce_dtype = TORCH_DTYPE_MAP[training.mixed_precision_reduce]
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallel_dims.pp_enabled,
    )

    possible_input_shardings = {
        "dp_replicate": Shard(0),
        "fsdp": Shard(0),
        "tp": Replicate(),
    }
    unsupported_axes = [
        name
        for name in dense_mesh.mesh_dim_names
        if name not in possible_input_shardings
    ]
    if unsupported_axes:
        raise ValueError(
            "Unsupported mesh axis for AutoParallel Llama3: "
            f"{unsupported_axes}. Supported axes: "
            f"{tuple(possible_input_shardings.keys())}"
        )
    x_sharding = tuple(
        possible_input_shardings[name] for name in dense_mesh.mesh_dim_names
    )

    loss_parallel_enabled = (
        parallel_dims.tp_enabled and not parallelism.disable_loss_parallel
    )
    if not loss_parallel_enabled:
        raise ValueError(
            "AutoParallel Llama3 graph_trainer currently requires loss parallel "
            "so the model-only graph can return vocab-sharded DTensor logits."
        )
    output_sharding = tuple(
        Shard(2) if name == "tp" else Shard(0) for name in dense_mesh.mesh_dim_names
    )

    with AutoParallelGraph(
        model,
        input_fn,
        dense_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([output_sharding])

        t0 = time.time()
        sharding_placement = autop.optimize_placement(verbose=False)
        t1 = time.time()
        logger.info(f"AutoParallelGraph took {t1 - t0:.2f} seconds")

        model_output = AutoParallelModelOutput(
            output_mesh=parallel_dims.get_mesh("tp"),
            output_placements=(Shard(2),),
            sharded_output_axis=2,
        )
        parallel_mod = autop.apply_placement_for_fx_module(
            sharding_placement,
            model_output=model_output,
        )

    model = apply_compile(
        parallel_mod,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
    return model
