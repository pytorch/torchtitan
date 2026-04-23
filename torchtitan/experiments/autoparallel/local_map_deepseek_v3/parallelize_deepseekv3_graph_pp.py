# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn
from autoparallel.api_pp import AutoParallelPP
from autoparallel.auto_bucketing import configure_inductor_for_autobucketing
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Replicate, Shard
from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.autoparallel.configs import AutoParallelCompileConfig
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def set_torchtitan_fields(orig: nn.Module, new: nn.Module) -> None:
    """Set torchtitan-specific fields on the parallelized model."""
    if hasattr(new, "layers") and isinstance(new.layers, torch.nn.ModuleDict):
        for block in new.layers.values():
            block.moe_enabled = hasattr(block, "moe")


def _preserve_moe_attributes(
    original_model: nn.Module, parallel_model: nn.Module
) -> None:
    """
    Preserve MoE custom attributes from the original model to the parallel model.
    This is only needed for attributes that aren't used in the graph, so they aren't
    lifted as graph inputs and fetched by the pre-graph runtime wrapper.

    `moe_enabled` and `load_balance_coeff` are used later in the optimizer to identify
    this block as a moe block. This should be safe as they are read-only.
    """

    def get_moe_modules(model: nn.Module) -> list[nn.Module]:
        moe_modules = []
        if hasattr(model, "layers"):
            if isinstance(model.layers, torch.nn.ModuleDict):
                blocks = model.layers.values()
            else:
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
                elif hasattr(block, "moe"):
                    moe_modules.append(block.moe)
        return moe_modules

    original_moe_modules = get_moe_modules(original_model)
    parallel_moe_modules = get_moe_modules(parallel_model)

    for orig_moe, par_moe in zip(original_moe_modules, parallel_moe_modules):
        if hasattr(orig_moe, "moe_enabled"):
            par_moe.moe_enabled = orig_moe.moe_enabled
        if hasattr(orig_moe, "load_balance_coeff"):
            par_moe.load_balance_coeff = orig_moe.load_balance_coeff


def _set_moe_mesh(model: nn.Module, mesh: DeviceMesh, axis_name: str) -> None:
    """Set MoE mesh/axis_name on model layers, handling ModelWithLoss wrapper.

    This must be called on the deep-copied model (autop.model) with the original
    (non-deep-copied) mesh to avoid broken _pg_registry from DeviceMesh pickling.
    """
    target = model
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        target = model.model
    if hasattr(target, "layers"):
        layers = target.layers
        if isinstance(layers, torch.nn.ModuleDict):
            layer_iter = layers.values()
        elif hasattr(layers, "children"):
            layer_iter = layers.children()
        else:
            layer_iter = []
        for layer in layer_iter:
            if hasattr(layer, "moe_enabled") and layer.moe_enabled:
                layer.moe.mesh = mesh
                layer.moe.axis_name = axis_name


def parallelize_deepseekv3(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: AutoParallelCompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    # GraphPP-specific (passed by graph_pipeline_llm):
    input_fn: Optional[Callable] = None,
    spmd_mesh: Optional[DeviceMesh] = None,
    stage_idx: int = 0,
    num_stages: int = 1,
    has_loss: bool = False,
) -> nn.Module:
    """
    Apply AutoParallelPP to a pipeline stage module.

    Mirrors the structure of the existing local_map_deepseek_v3/parallelize_deepseekv3
    but uses AutoParallelPP instead of AutoParallel. The graph callables and metadata
    are attached to the returned module as _graph_callables and _graph_meta attributes.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # --- Same as existing parallelize_deepseekv3 ---
    torch._inductor.config.force_disable_caches = True
    torch._inductor.config.allow_buffer_reuse = False
    configure_inductor_for_autobucketing(compile_config.comms_bucket_reorder_strategy)

    # Build mesh if not provided by caller
    if spmd_mesh is None:
        sparse_names = ["dp_replicate", "efsdp", "ep", "etp"]
        sparse_names = [
            name
            for name in sparse_names
            if parallel_dims.get_optional_mesh(name) is not None
        ]
        spmd_mesh = parallel_dims.get_mesh(sparse_names)

    assert input_fn is not None, "input_fn must be provided for AutoParallelPP"

    # --- AutoParallelPP (replaces AutoParallel) ---
    # NOTE: AutoParallel.__init__ does copy.deepcopy(model). DeviceMesh.__getstate__
    # strips _pg_registry (ProcessGroups can't be pickled), and __setstate__ only
    # reconstructs each mesh's OWN PG names — but sub-mesh PGs are stored in the
    # root mesh's registry at construction time. After deep-copy the root mesh's
    # registry is missing sub-mesh PG entries, causing "PG not found" during
    # torch.compile tracing. To avoid this, we set the MoE mesh AFTER the deep-copy
    # using the original (non-copied) spmd_mesh which has the correct _pg_registry.
    autop = AutoParallelPP(
        model,
        input_fn,
        spmd_mesh,
        dynamic=True,
        compile=False,
        reshard_after_forward=False,
        repeated_subgraphs=True,
    )
    _set_moe_mesh(autop.model, spmd_mesh, "ep")
    with autop:
        autop.add_parameter_memory_constraint(low=None, high=None)

        x_sharding = (Shard(0), Shard(0))
        if has_loss:
            autop.add_input_constraints([x_sharding, x_sharding])
            autop.add_output_constraints([(Replicate(), Replicate())])
        else:
            autop.add_input_constraints([x_sharding])
            autop.add_output_constraints([x_sharding])

        sharding_placement = autop.optimize_placement(verbose=False)

        graph_passes = ["split_fsdp_collectives"]
        if stage_idx > 0:
            graph_passes.append("split_dI_dW")

        cache = autop.apply_placement_pp(
            sharding_placement=sharding_placement, graph_passes=graph_passes
        )
        pp_mod = autop.parallel_model

    # --- Post-processing (same as existing) ---
    # Get the original model for attribute preservation
    # (handle ModelWithLoss wrapper)
    orig_model = model.model if hasattr(model, "model") else model
    set_torchtitan_fields(orig_model, pp_mod)
    _preserve_moe_attributes(orig_model, pp_mod)

    # --- GraphPP-specific: attach graph artifacts to module ---
    pp_mod._graph_callables = cache["graph_callables"]
    pp_mod._graph_meta = cache["graph_meta"]

    logger.info(
        f"Applied AutoParallelPP to stage {stage_idx}/{num_stages} "
        f"with graph passes {graph_passes}"
    )

    return pp_mod
