# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

from torchtitan.models.common.token_dispatcher import (
    DeepEPTokenDispatcher,
    HybridEPTokenDispatcher,
    MinimalAsyncEPTokenDispatcher,
)


def update_ep_token_dispatcher_config(model_config: Any, config: Any) -> None:
    """Validate and fill EP token dispatcher configs from runtime config."""
    from torchtitan.config import ParallelismConfig, TORCH_DTYPE_MAP
    from torchtitan.distributed.activation_checkpoint import FullAC
    from torchtitan.trainer import Trainer

    assert hasattr(
        config, "parallelism"
    ), "config passed to update_from_config must provide a parallelism field."
    parallelism = config.parallelism
    assert isinstance(parallelism, ParallelismConfig), (
        "config.parallelism must be a ParallelismConfig, got "
        f"{type(parallelism).__name__}."
    )

    dispatcher_cfgs = []
    for layer_cfg in model_config.layers:
        moe_cfg = getattr(layer_cfg, "moe", None)
        if moe_cfg is None:
            continue
        dispatcher_cfg = moe_cfg.routed_experts.token_dispatcher
        if isinstance(
            dispatcher_cfg,
            (
                DeepEPTokenDispatcher.Config,
                HybridEPTokenDispatcher.Config,
                MinimalAsyncEPTokenDispatcher.Config,
            ),
        ):
            dispatcher_cfgs.append(dispatcher_cfg)

    if not dispatcher_cfgs:
        return

    if parallelism.expert_parallel_degree == 1:
        dispatcher_name = type(dispatcher_cfgs[0]).__qualname__
        raise ValueError(
            f"{dispatcher_name} requires expert parallelism "
            "(expert_parallel_degree > 1)."
        )

    minimal_async_cfgs = [
        cfg
        for cfg in dispatcher_cfgs
        if isinstance(cfg, MinimalAsyncEPTokenDispatcher.Config)
    ]
    if minimal_async_cfgs:
        if parallelism.spmd_backend == "full_dtensor":
            raise ValueError("MinimalAsyncEP does not support full_dtensor SPMD.")
        if parallelism.tensor_parallel_degree != 1:
            raise ValueError(
                "MinimalAsyncEP does not support tensor or sequence parallelism."
            )
        if parallelism.context_parallel_degree != 1:
            raise ValueError("MinimalAsyncEP does not support context parallelism.")
        if parallelism.pipeline_parallel_degree != 1:
            raise ValueError("MinimalAsyncEP does not support pipeline parallelism.")
        for num_experts in {cfg.num_experts for cfg in minimal_async_cfgs}:
            if num_experts % parallelism.expert_parallel_degree != 0:
                raise ValueError(
                    f"MinimalAsyncEP num_experts ({num_experts}) must be "
                    "divisible by expert_parallel_degree "
                    f"({parallelism.expert_parallel_degree})."
                )

        if not isinstance(config, Trainer.Config):
            raise ValueError(
                "MinimalAsyncEP requires a Trainer.Config-compatible runtime config "
                "to set hidden_dim, tokens_per_rank, and dtype."
            )

        memory_policy = getattr(config.compile, "memory_policy", None)
        if (
            not isinstance(config.activation_checkpoint, FullAC.Config)
            and memory_policy != "full"
        ):
            raise ValueError(
                "MinimalAsyncEP requires full recompute: set "
                "activation-checkpoint:full for eager training or "
                "--compile.memory_policy full for graph_trainer."
            )

    if not isinstance(config, Trainer.Config):
        return

    num_tokens_per_rank = config.training.local_batch_size * config.training.seq_len
    dtype = TORCH_DTYPE_MAP[config.training.mixed_precision_param]
    for dispatcher_cfg in dispatcher_cfgs:
        if isinstance(dispatcher_cfg, DeepEPTokenDispatcher.Config):
            dispatcher_cfg.hidden_dim = model_config.dim
        elif isinstance(dispatcher_cfg, HybridEPTokenDispatcher.Config):
            dispatcher_cfg.hidden_dim = model_config.dim
            dispatcher_cfg.num_tokens_per_rank = num_tokens_per_rank
        elif isinstance(dispatcher_cfg, MinimalAsyncEPTokenDispatcher.Config):
            dispatcher_cfg.hidden_dim = model_config.dim
            dispatcher_cfg.tokens_per_rank = num_tokens_per_rank
            dispatcher_cfg.dtype = dtype


__all__ = ["update_ep_token_dispatcher_config"]
