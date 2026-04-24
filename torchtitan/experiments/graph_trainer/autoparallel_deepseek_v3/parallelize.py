# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoParallel-based parallelization for DeepSeek V3.

Uses AutoParallelGraph to apply solver-based SPMD sharding and return a
JointWithDescriptors for graph_trainer's compilation pipeline.
"""

import time

import torch
from autoparallel.auto_bucketing import configure_inductor_for_autobucketing
from torch.distributed.tensor.placement_types import Shard

from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.autoparallel_api import AutoParallelGraph
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def _set_torchtitan_fields(orig, new):
    if hasattr(new, "layers") and isinstance(new.layers, torch.nn.ModuleDict):
        for block in new.layers.values():
            block.moe_enabled = hasattr(block, "moe")


def _preserve_moe_attributes(original_model, parallel_model):
    """Preserve MoE attributes (moe_enabled, load_balance_coeff) from original."""

    def get_moe_modules(model):
        moe_modules = []
        if hasattr(model, "layers"):
            blocks = (
                model.layers.values()
                if isinstance(model.layers, torch.nn.ModuleDict)
                else []
            )
            for block in blocks:
                if hasattr(block, "moe"):
                    moe_modules.append(block.moe)
        return moe_modules

    for orig_moe, par_moe in zip(
        get_moe_modules(original_model), get_moe_modules(parallel_model)
    ):
        if hasattr(orig_moe, "moe_enabled"):
            par_moe.moe_enabled = orig_moe.moe_enabled
        if hasattr(orig_moe, "load_balance_coeff"):
            par_moe.load_balance_coeff = orig_moe.load_balance_coeff


def parallelize_deepseekv3(
    model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply AutoParallelGraph SPMD sharding to DeepSeek V3.

    Returns a parallelized model with _joint_with_descriptors attached
    for graph_trainer's compilation pipeline.
    """
    torch._inductor.config.force_disable_caches = True
    torch._inductor.config.allow_buffer_reuse = False

    comms_strategy = getattr(compile_config, "comms_bucket_reorder_strategy", "aten")
    configure_inductor_for_autobucketing(comms_strategy)

    sparse_names = ["dp_replicate", "efsdp", "ep", "etp"]
    sparse_names = [
        name
        for name in sparse_names
        if parallel_dims.get_optional_mesh(name) is not None
    ]
    sparse_mesh = parallel_dims.get_mesh(sparse_names)

    assert sparse_mesh.ndim == 2, "AP dsv3.py's local_map is specialized on 2 dims"

    for layer in model.layers.values():
        if layer.moe_enabled:
            layer.moe.mesh = sparse_mesh
            layer.moe.axis_name = "ep"

    def input_fn():
        global_batch_size = training.global_batch_size
        if global_batch_size < 0:
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = training.local_batch_size * dp_degree
        return (
            torch.randint(
                0,
                model.model_args.vocab_size,
                (global_batch_size, training.seq_len),
                device=torch.device("cuda"),
            ),
        )

    with AutoParallelGraph(
        model,
        input_fn,
        sparse_mesh,
        compile=False,
        dynamic=True,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        x_sharding = (Shard(0), Shard(0))
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        t0 = time.time()
        sharding_placement = autop.optimize_placement()
        t1 = time.time()
        logger.info(f"AutoParallelGraph took {t1 - t0:.2f} seconds")

        result = autop.apply_placement(sharding_placement)
        parallel_mod = result["parallel_model"]

    _set_torchtitan_fields(model, parallel_mod)
    _preserve_moe_attributes(model, parallel_mod)

    parallel_mod._joint_with_descriptors = result["joint_with_descriptors"]

    return parallel_mod
