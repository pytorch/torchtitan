# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import minimal_async_ep
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.fsdp import apply_fsdp_to_decoder
from torchtitan.distributed.full_dtensor import (
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
    validate_config,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.common.token_dispatcher import MinimalAsyncEPTokenDispatcher
from torchtitan.models.deepseek_v3 import DeepSeekV3Model
from torchtitan.tools.utils import device_module, device_type


def _validate_minimal_async_ep_config(
    dispatcher_config: MinimalAsyncEPTokenDispatcher.Config,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
) -> None:
    if parallelism.spmd_backend == "full_dtensor":
        raise ValueError("MinimalAsyncEP does not support full_dtensor SPMD.")
    if parallel_dims.ep <= 1:
        raise ValueError("MinimalAsyncEP requires expert_parallel_degree > 1.")
    if dispatcher_config.num_experts % parallel_dims.ep != 0:
        raise ValueError(
            f"MinimalAsyncEP num_experts ({dispatcher_config.num_experts}) must be "
            f"divisible by expert_parallel_degree ({parallel_dims.ep})."
        )
    if parallel_dims.tp != 1:
        raise ValueError(
            "MinimalAsyncEP does not support tensor or sequence parallelism."
        )
    if parallelism.enable_sequence_parallel:
        raise ValueError("MinimalAsyncEP does not support sequence parallelism.")
    if parallel_dims.cp != 1:
        raise ValueError("MinimalAsyncEP does not support context parallelism.")
    if parallel_dims.pp != 1:
        raise ValueError("MinimalAsyncEP does not support pipeline parallelism.")
    if parallel_dims.dp_replicate != 1:
        raise ValueError(
            "MinimalAsyncEP requires data_parallel_replicate_degree == 1. Got "
            f"{parallel_dims.dp_replicate}."
        )
    if parallel_dims.dp_shard % parallel_dims.ep != 0:
        raise ValueError(
            "MinimalAsyncEP requires data_parallel_shard_degree to be a multiple "
            "of expert_parallel_degree. Got "
            f"data_parallel_shard_degree={parallel_dims.dp_shard}, "
            f"expert_parallel_degree={parallel_dims.ep}."
        )


def _init_minimal_async_ep_buffer(
    model: DeepSeekV3Model,
    *,
    dispatcher_config: MinimalAsyncEPTokenDispatcher.Config,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
) -> None:
    minimal_async_ep.init_buffer(
        group=parallel_dims.get_mesh("ep").get_group(),
        hidden_dim=model.config.dim,
        max_tokens_per_rank=training.local_batch_size * training.seq_len,
        num_local_experts=dispatcher_config.num_experts // parallel_dims.ep,
        top_k=dispatcher_config.top_k,
        dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        device=torch.device(device_type, device_module.current_device()),
    )

def _maybe_init_minimal_async_ep_buffer(
    model: DeepSeekV3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    ac_config: ActivationCheckpointConfig,
    memory_policy: str | None = None,
) -> None:
    moe_config = next((l.moe for l in model.config.layers if l.moe is not None), None)
    if moe_config is None:
        return

    dispatcher_config = moe_config.experts.token_dispatcher
    if not isinstance(dispatcher_config, MinimalAsyncEPTokenDispatcher.Config):
        return

    if ac_config.mode != "full" and memory_policy != "full":
        raise ValueError(
            "MinimalAsyncEP requires full recompute: set "
            "--activation_checkpoint.mode full for eager training or "
            "--compile.memory_policy full for graph_trainer."
        )

    _validate_minimal_async_ep_config(
        dispatcher_config,
        parallel_dims=parallel_dims,
        parallelism=parallelism,
    )
    _init_minimal_async_ep_buffer(
        model,
        dispatcher_config=dispatcher_config,
        parallel_dims=parallel_dims,
        training=training,
    )


# Adapted from llama4/infra/parallelize.py
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
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """
    _maybe_init_minimal_async_ep_buffer(
        model,
        parallel_dims=parallel_dims,
        training=training,
        parallelism=parallelism,
        ac_config=ac_config,
    )

    if parallelism.spmd_backend == "full_dtensor":
        validate_config(parallel_dims, model)
        model.parallelize(parallel_dims)
    else:
        # CP: wrap inner attention forward BEFORE parallelize() so CP logic
        # runs inside the local_map boundary on local tensors.
        if parallel_dims.cp_enabled:
            apply_cp_to_forward(
                # pyrefly: ignore [missing-attribute]
                [block.attention.inner_attention for block in model.layers.values()],
                parallel_dims.get_mesh("cp"),
            )
        if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
            model.parallelize(parallel_dims)

    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

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

    if model_compile_enabled:
        apply_compile(model, compile_config)

    if parallelism.spmd_backend == "full_dtensor":
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    else:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)
        dp_mesh_dims = None
        edp_mesh = None
        edp_mesh_dims = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = (
                ["dp_replicate", "efsdp"]
                if parallel_dims.dp_replicate_enabled
                else ["efsdp"]
            )
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model
