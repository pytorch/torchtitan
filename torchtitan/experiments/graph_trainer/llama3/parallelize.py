# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_ac_regions,
    annotate_module_fqns,
    apply_graph_ac,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.llama3.model import GraphTrainerLlama3Model
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.models.llama3.parallelize import apply_tp
from torchtitan.tools.logging import logger


def annotate_llama(model: GraphTrainerLlama3Model) -> None:
    """Attach annotations to FX graph nodes with ``torch.fx.traceback.annotate_fn``

    - Module FQN annotation: Tags each submodule's forward with its
      fully-qualified name so that the transformer_block_bucketing pass
      can identify which graph nodes belong to which module.
    - AC region annotation: Tags each transformer block's forward with a unique
      ac_region_id so that apply_sac_pass can assign per-block ac_graph_id
      boundaries for the min-cut partitioner.
    """
    annotate_module_fqns(model)
    annotate_ac_regions(model)


def parallelize_llama(
    model: GraphTrainerLlama3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    **kwargs,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    annotate_llama(model)

    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        apply_tp(
            model,
            tp_mesh,
            enable_loss_parallel=not parallelism.disable_loss_parallel,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    if ac_config.mode != "none":
        apply_graph_ac(compile_config, ac_config)

    # apply data parallel
    if (
        parallel_dims.dp_replicate_enabled
        or parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
    ):
        if parallel_dims.dp_replicate_enabled:
            if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
                dp_mesh_dim_names = ["dp_replicate", "fsdp"]
                dp_mode = "hybrid_shard"
            else:
                dp_mesh_dim_names = ["dp_replicate"]
                dp_mode = "replicate"
        else:
            dp_mesh_dim_names = ["fsdp"]
            dp_mode = "fully_shard"

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

        model = data_parallel(
            model,
            parallel_dims.get_mesh(dp_mesh_dim_names),
            mode=dp_mode,
            mp_policy=mp_policy,
        )
        logger.info(
            "Applied Data Parallel (simple_fsdp) (dp mode=%s) to the model", dp_mode
        )

    # Apply compilation based on mode
    model = apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )

    return model
