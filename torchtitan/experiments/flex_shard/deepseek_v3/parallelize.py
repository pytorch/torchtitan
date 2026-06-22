# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

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
    disable_flex_shard_gradient_division,
    flex_shard,
)
from torchtitan.experiments.flex_shard.flex_shard.bucket_storage import (
    MixedPrecisionPolicy,
)
from torchtitan.experiments.flex_shard.grad_norm import (
    install_flex_shard_grad_norm_clipping,
)
from torchtitan.models.deepseek_v3 import DeepSeekV3Model
from torchtitan.models.llama4.parallelize import apply_moe_ep_tp
from torchtitan.tools.logging import logger

from .placement_policy import DeepSeekV3FlexShardPolicy


def parallelize_deepseekv3(
    model: DeepSeekV3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """Apply the experimental eager FlexShard path to DeepSeek V3."""
    _validate_supported_parallelisms(
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

    if parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_mesh("ep"),
            enable_sp=parallelism.enable_sequence_parallel,
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )
    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    dp_mesh = parallel_dims.get_mesh("fsdp")
    efsdp_mesh = parallel_dims.get_mesh("efsdp") if parallel_dims.ep_enabled else None

    _apply_flex_shard(
        model,
        dp_mesh=dp_mesh,
        efsdp_mesh=efsdp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        placement_policy=DeepSeekV3FlexShardPolicy(),
    )

    logger.info("Applied experimental eager FlexShard to the DeepSeek V3 model")
    return model


def _validate_supported_parallelisms(
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    compile_config: CompileConfig,
) -> None:
    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "The experimental FlexShard DeepSeek V3 path does not support PP yet."
        )
    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "The experimental FlexShard DeepSeek V3 path does not support TP yet."
        )
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "The experimental FlexShard DeepSeek V3 path does not support CP yet."
        )
    if parallel_dims.dp_replicate_enabled:
        raise NotImplementedError(
            "The experimental FlexShard DeepSeek V3 path does not support HSDP yet."
        )
    if training.enable_cpu_offload:
        raise NotImplementedError(
            "FlexShard eager training does not support CPU offload yet."
        )
    if compile_config.enable and "model" in compile_config.components:
        raise NotImplementedError(
            "This FlexShard training entry point is eager-only; disable model compile."
        )


def _apply_flex_shard(
    model: nn.Module,
    *,
    dp_mesh: DeviceMesh,
    efsdp_mesh: DeviceMesh | None,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    reshard_after_forward_policy: str,
    placement_policy: DeepSeekV3FlexShardPolicy,
) -> None:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy,
        pp_enabled,
    )
    reshard_last = reshard_after_forward_policy == "always"
    buckets = placement_policy.build_buckets(
        model,
        dp_mesh=dp_mesh,
        efsdp_mesh=efsdp_mesh,
        mp_policy=mp_policy,
        reshard_after_forward=reshard_after_forward,
        reshard_last=reshard_last,
    )

    flex_shard(model, buckets=buckets)
    disable_flex_shard_gradient_division(model)
    install_flex_shard_grad_norm_clipping()
