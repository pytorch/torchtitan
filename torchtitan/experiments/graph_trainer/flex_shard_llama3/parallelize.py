# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.fsdp import DataParallelMeshDims

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.flex_shard import (
    BucketSpec,
    flex_shard,
    MixedPrecisionPolicy,
    per_param_placements,
)
from torchtitan.experiments.graph_trainer.flex_shard_llama3.model import (
    FlexShardLlama3Model,
)
from torchtitan.tools.logging import logger


def _validate_flex_shard_eager_only(
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    compile_config: CompileConfig,
) -> None:
    if compile_config.enable or getattr(compile_config, "mode", None) is not None:
        raise ValueError(
            "FlexShard currently supports eager execution only. "
            "Set --compile.mode None and leave --compile.enable false."
        )
    if getattr(compile_config, "precompile_artifact_dir", ""):
        raise ValueError("FlexShard does not support precompile artifacts yet.")
    if not parallel_dims.dp_shard_enabled:
        raise ValueError(
            "FlexShard requires data-parallel sharding; set "
            "--parallelism.data_parallel_shard_degree to a value greater than 1."
        )

    unsupported = []
    if parallel_dims.dp_replicate_enabled:
        unsupported.append("hybrid data parallel")
    if parallel_dims.tp_enabled:
        unsupported.append("tensor parallel")
    if parallel_dims.cp_enabled:
        unsupported.append("context parallel")
    if parallel_dims.pp_enabled:
        unsupported.append("pipeline parallel")
    if parallel_dims.ep_enabled:
        unsupported.append("expert parallel")
    if unsupported:
        raise ValueError(
            "FlexShard eager-only path currently supports only FSDP-style "
            f"data-parallel sharding; unsupported: {', '.join(unsupported)}."
        )
    if training.enable_cpu_offload:
        raise ValueError("FlexShard CPU offload is not supported in the eager-only path.")


def parallelize_llama_flex_shard(
    model: FlexShardLlama3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """
    Apply eager-only FlexShard data parallelism to Llama.
    """
    _validate_flex_shard_eager_only(
        parallel_dims=parallel_dims,
        training=training,
        compile_config=compile_config,
    )

    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=False,
            base_folder=dump_folder,
        )

    fsdp_mesh = parallel_dims.get_mesh("fsdp")
    dp_mesh_dims = DataParallelMeshDims(shard="fsdp")
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )
    reshard_after_fwd = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, parallel_dims.pp_enabled
    )
    buckets = (
        [
            BucketSpec(
                ["tok_embeddings.*"],
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_fwd,
            ),
        ]
        + [
            BucketSpec(
                [f"layers.{i}.*"],
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_fwd,
            )
            for i in range(len(model.layers))
        ]
        + [
            BucketSpec(
                ["norm.*", "lm_head.*", "output.*"],
                mp_policy=mp_policy,
                reshard_after_forward=reshard_after_fwd,
            ),
        ]
    )

    flex_shard(
        model,
        fsdp_mesh,
        dp_mesh_dims,
        shard_placement_fn=per_param_placements,
        buckets=buckets,
    )
    logger.info("Applied eager FlexShard data parallelism to the model")

    return model
