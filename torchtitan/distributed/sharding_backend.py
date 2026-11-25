# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""How data-parallel parameters are sharded over ``dp_shard``.

``parallelism.fsdp_backend`` selects a ``ShardingBackend`` by its config type.
Each backend owns the operations that depend on how the parameters are sharded,
so callers don't check which backend is in use. ``FullyShardBackend``, the
default, applies FSDP2's ``fully_shard``; ``FlexShardBackend``
(``torchtitan/distributed/flex_shard_backend.py``) applies FlexShard.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from typing import Any, TYPE_CHECKING, TypeAlias

import torch
import torch.nn as nn

from torchtitan.config.configurable import Configurable

if TYPE_CHECKING:
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.optim import Optim
    from torchtitan.config import DebugConfig, PrecisionConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.distributed.parallelism_context import ParallelismContext
    from torchtitan.models.base.model import BaseModel
    from torchtitan.models.common.decoder import Decoder

__all__ = ["FullyShardBackend", "PartialBackwards", "ShardingBackend"]

# Called with ``run_backward`` and ``outer_backward_follows``, as
# ``fsdp_partial_backwards`` (torchtitan/distributed/fsdp_lifecycle.py) is, and
# yields the same hooks.
PartialBackwards: TypeAlias = Callable[..., AbstractContextManager[Any]]


class ShardingBackend(Configurable, ABC):
    """Shard data-parallel parameters, and run what depends on their sharding."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(self, config: Config) -> None:
        self.config = config

    def validate_model(
        self,
        model_config: BaseModel.Config,
        *,
        parallelism: ParallelismConfig,
        debug: DebugConfig,
    ) -> None:
        """Reject the model, parallelism and debug settings this backend
        doesn't support. ``validate_model_training_config`` calls this."""

    def validate_trainer(
        self,
        *,
        checkpointer: CheckpointManager.Config | None,
        optim: Optim.Config,
        trainer: type | None,
    ) -> None:
        """Reject the trainer components this backend doesn't support.
        ``TrainingEngine.Config.__post_init__`` calls this for every trainer,
        with ``trainer`` the class that owns its config."""

    @abstractmethod
    def apply_to_decoder(
        self,
        model: Decoder,
        *,
        parallelism_context: ParallelismContext,
        precision: PrecisionConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        """Shard a decoder's parameters over its data-parallel mesh."""

    @abstractmethod
    def clip_grad_norm_(
        self,
        parameters: list[torch.nn.Parameter],
        max_norm: float,
        *,
        foreach: bool | None,
        parallelism_context: ParallelismContext,
    ) -> torch.Tensor:
        """Clip the grads of ``parameters`` by their total L2 norm, reduced
        over every rank that holds part of them and over the PP mesh of
        ``parallelism_context``, and return that norm."""

    @abstractmethod
    def partial_backwards(self, model: nn.Module, module_fqn: str) -> PartialBackwards:
        """The context that keeps ``model``'s submodule ``module_fqn``
        unsharded while it runs on its own, in several forwards and backwards,
        as ``ChunkedLossWrapper`` runs ``lm_head``. Call after sharding."""

    @abstractmethod
    def reshard_after_eval(self, model_parts: list[nn.Module]) -> None:
        """Finish the forwards an eval step leaves open, e.g. ``lm_head``'s
        under ``ChunkedLossWrapper``, and reshard the parameters. Eval has no
        backward to do it."""


class FullyShardBackend(ShardingBackend):
    """PyTorch's FSDP2 ``fully_shard``, configured by the ``fsdp_*`` fields of
    ``ParallelismConfig``."""

    @dataclass(kw_only=True, slots=True)
    class Config(ShardingBackend.Config):
        pass

    def apply_to_decoder(
        self,
        model: Decoder,
        *,
        parallelism_context: ParallelismContext,
        precision: PrecisionConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torchtitan.config import TORCH_DTYPE_MAP
        from torchtitan.distributed.fsdp import (
            apply_fsdp_to_decoder,
            resolve_fsdp_mesh,
            resolve_sparse_fsdp_mesh,
        )

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallelism_context)
        apply_fsdp_to_decoder(
            model,
            dp_mesh=dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[precision.param_dtype],
            reduce_dtype=TORCH_DTYPE_MAP[precision.reduce_dtype],
            pp_enabled=parallelism_context.pp_enabled,
            cpu_offload=parallelism.fsdp_enable_cpu_offloading,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallelism_context.ep,
            edp_mesh=edp_mesh,
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh_dims=edp_mesh_dims,
            symm_mem_scope=parallelism.fsdp_symm_mem_scope,
            unshard_in_current_stream=parallelism.fsdp_unshard_in_current_stream,
        )

    def clip_grad_norm_(
        self,
        parameters: list[torch.nn.Parameter],
        max_norm: float,
        *,
        foreach: bool | None,
        parallelism_context: ParallelismContext,
    ) -> torch.Tensor:
        from torchtitan.distributed import utils as dist_utils

        return dist_utils.clip_grad_norm_(
            parameters,
            max_norm,
            foreach=foreach,
            pp_mesh=parallelism_context.get_optional_mesh("pp"),
        )

    def partial_backwards(self, model: nn.Module, module_fqn: str) -> PartialBackwards:
        from torchtitan.distributed.fsdp_lifecycle import fsdp_partial_backwards

        return partial(fsdp_partial_backwards, model.get_submodule(module_fqn))

    def reshard_after_eval(self, model_parts: list[nn.Module]) -> None:
        from torchtitan.distributed.fsdp import post_validate_fsdp_for_accelerator_graph

        post_validate_fsdp_for_accelerator_graph(model_parts)
