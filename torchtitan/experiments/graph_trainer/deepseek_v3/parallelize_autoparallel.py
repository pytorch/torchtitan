# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
AutoParallel-based parallelization for DeepSeek V3.

Uses AutoParallelGraph to apply solver-based SPMD sharding on AutoParallel's
local_map DSv3 model (whose ops the solver supports), then feeds the pre-built
JWD into graph_trainer's AOT pipeline. Requires a 2D sparse mesh (EFSDP+EP).

The torchtitan DSv3 model is replaced with AutoParallel's DeepSeekV3Model
because the solver doesn't support torchtitan's token_dispatcher ops
(aten::div.Tensor_mode). The two models share the same hierarchical config
layout via duck typing.
"""

import time

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor.placement_types import Shard

from torchtitan.config import (
    ActivationCheckpointConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.autoparallel_api import AutoParallelGraph
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.tools.logging import logger


def _load_autoparallel_dsv3_dependency():
    """Load the temporary AutoParallel DSv3 integration dependency."""
    try:
        from autoparallel._testing.models.dsv3 import (
            annotate_deepseekv3_for_graph_trainer,
            DeepSeekV3Model,
        )
    except ImportError as exc:
        raise ImportError(
            "AutoParallel graph_trainer DeepSeek V3 currently depends on "
            "autoparallel._testing.models.dsv3. Move that model and annotation "
            "helper into a supported AutoParallel namespace before treating this "
            "route as a stable production dependency."
        ) from exc
    return DeepSeekV3Model, annotate_deepseekv3_for_graph_trainer


def _set_torchtitan_fields(parallel_model):
    if hasattr(parallel_model, "layers") and isinstance(
        parallel_model.layers, torch.nn.ModuleDict
    ):
        for block in parallel_model.layers.values():
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


def parallelize_autoparallel_deepseekv3(
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
    """Apply AutoParallelGraph SPMD sharding to DeepSeek V3.

    Returns a sharded model carrying AutoParallel train-step metadata.
    Requires a 2D sparse mesh (EFSDP+EP).
    """
    if compile_config.mode != "aot_fx_trace":
        raise ValueError(
            "AutoParallel graph_trainer requires compile.mode=aot_fx_trace"
        )

    sparse_names = ["dp_replicate", "efsdp", "ep"]
    sparse_names = [
        name
        for name in sparse_names
        if parallel_dims.get_optional_mesh(name) is not None
    ]
    sparse_mesh = parallel_dims.get_mesh(sparse_names)

    if sparse_mesh.ndim != 2:
        raise ValueError(
            "AutoParallel DeepSeek V3 requires a 2D sparse mesh with EFSDP and EP "
            f"axes, but got mesh axes {sparse_mesh.mesh_dim_names}"
        )

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
    (
        APDeepSeekV3Model,
        annotate_deepseekv3_for_graph_trainer,
    ) = _load_autoparallel_dsv3_dependency()

    # Use AutoParallel's DSv3 model: torchtitan's token_dispatcher uses
    # aten::div.Tensor_mode which the AP solver doesn't support yet.
    # The AP model accepts torchtitan's config via duck typing (same
    # hierarchical attribute paths).
    with torch.device("meta"):
        ap_model = APDeepSeekV3Model(
            model.config,
            mesh=sparse_mesh,
            compute_dtype=param_dtype,
        )

    def input_fn():
        global_batch_size = training.global_batch_size
        if global_batch_size < 0:
            dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
            global_batch_size = training.local_batch_size * dp_degree
        tokens = torch.randint(
            0,
            ap_model.model_args.vocab_size,
            (global_batch_size, training.seq_len),
            device=torch.device("cuda"),
        )
        return tokens

    x_sharding = (Shard(0), Shard(0))
    if parallel_dims.tp_enabled and not parallelism.disable_loss_parallel:
        raise ValueError(
            "AutoParallel DeepSeek V3 graph_trainer does not support TP loss "
            "parallel yet. Pass --parallelism.disable_loss_parallel."
        )

    autop = AutoParallelGraph(
        ap_model,
        input_fn,
        sparse_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        dynamic=True,
    )

    annotate_deepseekv3_for_graph_trainer(autop.model)

    with autop:
        autop.add_parameter_memory_constraint(low=None, high=None)
        autop.add_input_constraints([x_sharding])
        autop.add_output_constraints([x_sharding])

        t0 = time.time()
        sharding_placement = autop.optimize_placement()
        t1 = time.time()
        logger.info(f"AutoParallelGraph took {t1 - t0:.2f} seconds")

        # The solved output is logically batch-sharded over EFSDP and EP
        # (Shard(0), Shard(0)). Those axes are data-parallel factors for
        # loss computation, so graph_trainer can consume each rank's local
        # logits as a plain tensor and pair them with local labels. Only TP
        # vocab sharding needs a DTensor output boundary for loss_parallel().
        parallel_mod = autop.apply_placement_for_fx_module(sharding_placement)

    _set_torchtitan_fields(parallel_mod)
    _preserve_moe_attributes(ap_model, parallel_mod)

    model = apply_compile(
        parallel_mod,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )
    return model
