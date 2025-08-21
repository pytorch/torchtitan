# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from contextlib import nullcontext
from datetime import timedelta
from typing import Callable, ContextManager, Optional, TYPE_CHECKING, Union

import torch
import torch.distributed as dist

import torch.nn as nn
from torch.distributed._composable.fsdp.fully_shard import FSDPModule
from torch.distributed.distributed_c10d import ReduceOp
from torchtitan.components.ft.config import FaultTolerance as FTConfig
from torchtitan.tools.logging import logger

if importlib.util.find_spec("torchft") is not None:
    import torchft as ft

    if TYPE_CHECKING:
        from torchft import local_sgd

    has_torchft = True
else:
    has_torchft = False


class FTManager:
    def __init__(
        self,
        ft_config: FTConfig,
    ) -> None:
        if not ft_config.enable:
            self._manager = None
            return

        if not has_torchft:
            raise ImportError("torchft is not installed. Please install it.")

        process_group_timeout = timedelta(
            milliseconds=ft_config.process_group_timeout_ms
        )
        if ft_config.process_group == "gloo":
            pg = ft.ProcessGroupGloo(timeout=process_group_timeout)
        elif ft_config.process_group == "nccl":
            pg = ft.ProcessGroupNCCL(timeout=process_group_timeout)
        else:
            raise ValueError(f"Unsupported process group: {ft_config.process_group}")

        # If the training method is specific, then the quorum should be synchronous
        self.use_async_quorum = ft_config.semi_sync_method is None

        self._manager = ft.Manager(
            pg=pg,
            min_replica_size=ft_config.min_replica_size,
            load_state_dict=None,
            state_dict=None,
            use_async_quorum=self.use_async_quorum,
            replica_id=f"torchtitan_ft_{ft_config.replica_id}",
        )
        self.group_size = ft_config.group_size
        self.replica_id = ft_config.replica_id

        if self.use_async_quorum:
            self.replicate_pg = ft.process_group.ManagedProcessGroup(self._manager)
            self.replicate_pg.register("dp_replicate")

    @property
    def enabled(self) -> bool:
        return self._manager is not None

    @property
    def manager(self) -> "ft.Manager":
        assert self._manager is not None
        return self._manager

    def get_dp_info(self, dp_degree: int, dp_rank: int) -> tuple[int, int]:
        if self.enabled:
            return dp_degree * self.group_size, dp_degree * self.replica_id + dp_rank
        else:
            return dp_degree, dp_rank

    def maybe_set_all_reduce_hook(self, model_parts: list[torch.nn.Module]) -> None:
        if self.enabled and self.use_async_quorum:

            def all_reduce_hook(output):
                dist.all_reduce(output, group=self.replicate_pg, op=ReduceOp.AVG)

            def apply_set_all_reduce_hook(m):
                if isinstance(m, FSDPModule):
                    m.set_all_reduce_hook(all_reduce_hook)

            for model_part in model_parts:
                model_part.apply(apply_set_all_reduce_hook)

    @property
    def loss_sync_pg(
        self,
    ) -> Optional["ft.process_group.ManagedProcessGroup"]:
        if self.enabled and self.use_async_quorum:
            return self.replicate_pg
        else:
            # skip loss sync when using semi-sync training
            return None


def maybe_semi_sync_training(
    ft_config: FTConfig,
    ft_manager: FTManager,
    model: torch.nn.Module,
    n_layers: int,
    optimizer: torch.optim.Optimizer,
    fragment_fn: Optional[Callable[..., list[nn.Module]]] = None,
) -> ContextManager[Union["local_sgd.DiLoCo", "local_sgd.LocalSGD", None]]:
    """
    If TorchFT is enabled and the config is set, use semi_sync_method
    """
    semi_sync_method = ft_config.semi_sync_method
    if ft_config.enable and semi_sync_method is not None:
        from torchft import local_sgd

        assert (
            ft_manager._manager is not None
        ), "FTManager must be enabled to use semi-sync training."
        logger.info(
            f"using fragment function to split model: {fragment_fn is not None}"
        )
        if semi_sync_method.lower() == "diloco":
            if fragment_fn:
                model_parts = fragment_fn(model, ft_config, n_layers)
            else:
                model_parts = [model]

            # Create the outer optimizer based on the inner optimizer parameters.
            outer_optimizers = []
            for model in model_parts:
                params = [p for p in model.parameters() if p.requires_grad]
                outer_optimizer = torch.optim.SGD(
                    params, lr=0.7, momentum=0.9, nesterov=True
                )
                outer_optimizers.append(outer_optimizer)

            return local_sgd.DiLoCo(
                manager=ft_manager._manager,
                model_fragments=model_parts,
                inner_optimizer=optimizer,
                outer_optimizer=outer_optimizers,
                sync_every=ft_config.sync_steps,
                should_quantize=ft_config.should_quantize,
                fragment_sync_delay=ft_config.fragment_sync_delay,
                fragment_update_alpha=ft_config.fragment_update_alpha,
            )
        elif semi_sync_method.lower() == "local_sgd":
            return local_sgd.LocalSGD(
                manager=ft_manager._manager,
                model=model,
                optimizer=optimizer,
                sync_every=ft_config.sync_steps,
            )
        else:
            raise ValueError(
                f"Unknown training method: {semi_sync_method}, only 'diloco' and 'local_sgd' are supported."
            )
    return nullcontext()
