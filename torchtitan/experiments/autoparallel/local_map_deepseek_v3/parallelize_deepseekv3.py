# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import time

import torch
from autoparallel.api import AutoParallel
from autoparallel.auto_bucketing import configure_inductor_for_autobucketing

from torch.distributed.tensor.placement_types import Shard
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.autoparallel.configs import AutoParallelCompileConfig
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan.tools.logging import logger


# TODO: Autoparallel should transparently wrap the original nn.Module
# but I don't know how to do that.
def set_torchtitan_fields(orig, new):
    assert isinstance(new.layers, torch.nn.ModuleDict)
    for block in new.layers.values():
        block.moe_enabled = hasattr(block, "moe")


def parallelize_deepseekv3(
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
    Apply Autoparallel to the model

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

    # Build the sparse mesh for MoE expert parallelism
    # Filter to only include enabled mesh dimensions
    sparse_names = ["dp_replicate", "efsdp", "ep", "etp"]
    sparse_names = [
        name
        for name in sparse_names
        if parallel_dims.get_optional_mesh(name) is not None
    ]
    sparse_mesh = parallel_dims.get_mesh(sparse_names)

    # Update me when changing dsv3.py
    assert sparse_mesh.ndim == 2, "AP dsv3.py's local_map is specialized on 2 dims"

    # Provide AP MoE with mesh
    for layer in model.layers.values():
        if layer.moe_enabled:
            layer.moe.mesh = sparse_mesh
            layer.moe.axis_name = "ep"

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

    should_compile = compile_config.enable
    if should_compile:
        # TODO: support more options in AP API
        assert compile_config.components == ["model"]
        assert compile_config.backend == "inductor"

    mp_policy = None
    with AutoParallel(
        model,
        input_fn,
        sparse_mesh,
        mp_policy=mp_policy,
        compile=should_compile,
        dynamic=True,
    ) as autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        x_sharding = (Shard(0), Shard(0))
        loss_parallel_enabled = (
            parallel_dims.tp_enabled and not parallelism.disable_loss_parallel
        )
        assert not loss_parallel_enabled
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])
        t0 = time.time()
        sharding_placement = autop.optimize_placement()
        t1 = time.time()
        logger.info(f"AutoParallel took {t1 - t0} seconds")
        parallel_mod = autop.apply_placement(sharding_placement)

    set_torchtitan_fields(model, parallel_mod)

    if loss_parallel_enabled:

        # current PyTorch's implementation of loss parallel assumes
        # that the DTensor has a 1d device mesh. This is not true
        # in our case, but we can work around it by adding
        # casting the output to a DTensor on a 1d device mesh.
        # We should just use AutoParallel to do this for us, but
        # it would require putting the loss inside the model as well
        def _return_as_dtensor_for_loss_parallel(module, args, output):
            return torch.distributed.tensor.DTensor.from_local(
                output, sparse_mesh["etp"], (Shard(2),)
            )

        # not keeping a reference to the hook, don't plan on
        # removing it at any point
        parallel_mod.register_forward_hook(_return_as_dtensor_for_loss_parallel)

    _preserve_moe_attributes(model, parallel_mod)

    return parallel_mod


def _preserve_moe_attributes(original_model, parallel_model):
    """
    Preserve MoE custom attributes from the original model to the parallel model.
    This is only needed for attributes that aren't used in the graph, so they aren't
    lifted as graph inputs and fetched by the pre-graph runtime wrapper.

    `moe_enabled` and `load_balance_coeff` are used later in the optimizer to identify
    this block as a moe block. This should be safe as they are read-only.
    """

    def get_moe_modules(model):
        """Extract all MoE modules from the model."""
        moe_modules = []
        if hasattr(model, "layers"):
            if isinstance(model.layers, torch.nn.ModuleDict):
                # regular torchtitan structure
                blocks = model.layers.values()
            else:
                # autoparallel might change structure
                blocks = (
                    model.layers.children() if hasattr(model.layers, "children") else []
                )

            for block in blocks:
                if (
                    hasattr(block, "moe_enabled")
                    and block.moe_enabled
                    and hasattr(block, "moe")
                ):
                    moe_modules.append(block.moe)
                elif hasattr(block, "moe"):  # fallback for autoparallel
                    moe_modules.append(block.moe)
        return moe_modules

    original_moe_modules = get_moe_modules(original_model)
    parallel_moe_modules = get_moe_modules(parallel_model)

    # Copy custom attributes from original to parallel MoE modules
    # This is fine to do since these attributes are read only
    for orig_moe, par_moe in zip(original_moe_modules, parallel_moe_modules):
        if hasattr(orig_moe, "moe_enabled"):
            par_moe.load_balance_coeff = orig_moe.load_balance_coeff

        # Copy load_balance_coeff
        if hasattr(orig_moe, "load_balance_coeff"):
            par_moe.load_balance_coeff = orig_moe.load_balance_coeff
