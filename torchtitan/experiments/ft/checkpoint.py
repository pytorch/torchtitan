# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FT-aware CheckpointManager subclass.

Adds TorchFT fault tolerance support on top of the base CheckpointManager:
- Per-replica dataloader checkpointing (``_ft_save`` / ``_ft_load``)
- ``participating_rank`` guards so only one replica saves the full checkpoint
- ``ft_manager.set_state_dict_fns`` for in-memory state transfer between replicas
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.components.checkpoint import (
    AsyncMode,
    CheckpointManager,
    DATALOADER,
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
    TRAIN_STATE,
)
from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.experiments.ft.manager import FTManager
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection


class FTCheckpointManager(CheckpointManager):
    """CheckpointManager with TorchFT fault tolerance support.

    There are two types of checkpoints when TorchFT is enabled:

    1. **Full persistent checkpoint** — saved by the replica with
       ``ft_manager.participating_rank() == 0``. Contains model, optimizer,
       lr_scheduler, dataloader, and train_state.

    2. **Per-replica checkpoint** — contains only the dataloader and is
       saved/loaded by all replicas to/from their own folder, prefixed with
       the ``ft_replica_id``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CheckpointManager.Config):
        enable_ft_dataloader_checkpoints: bool = True
        """
        Warning: Disabling this can have fault tolerant replicas training
        over the same data multiple times. Use it with caution if training
        over the same data is acceptable.

        Used to enable checkpointing the dataloader index for fault tolerant
        training with torchft. If enabled, data loader state is checkpointed.
        Otherwise, replicas will train over the same data multiple times,
        which can result in overfitting.
        """

    def __init__(
        self,
        config: Config,
        *,
        dataloader: BaseDataLoader | None,
        model_parts: list[nn.Module],
        optimizers: OptimizersContainer,
        lr_schedulers: LRSchedulersContainer,
        states: dict[str, Any],
        sd_adapter: BaseStateDictAdapter | None,
        base_folder: str = "",
        ft_manager: FTManager | None = None,
    ) -> None:
        # Initialize the base checkpoint manager (without FT)
        super().__init__(
            config,
            dataloader=dataloader,
            model_parts=model_parts,
            optimizers=optimizers,
            lr_schedulers=lr_schedulers,
            states=states,
            sd_adapter=sd_adapter,
            base_folder=base_folder,
        )

        self.ft_manager = (
            ft_manager.manager if ft_manager and ft_manager.enabled else None
        )
        self.enable_ft_dataloader_checkpoints = (
            self.ft_manager and config.enable_ft_dataloader_checkpoints
        )

        if self.ft_manager and not self.enable_ft_dataloader_checkpoints:
            logger.warning(
                "Fault tolerance is enabled but enable_ft_dataloader_checkpoints "
                "is False. This means replicas can retrain over the same data "
                "multiple times, which can result in overfitting."
            )

        if not self.enable:
            return

        if self.ft_manager:
            optimizers.init_cache_state_dict()

            def state_dict():
                ret = {}
                for k, v in self.states.items():
                    if k in {MODEL, OPTIMIZER, LR_SCHEDULER, TRAIN_STATE}:
                        ret[k] = v.state_dict()
                return ret

            def load_state_dict(state_dict):
                assert state_dict is not None
                for k, v in state_dict.items():
                    self.states[k].load_state_dict(v)

            # pyrefly: ignore [missing-attribute]
            self.ft_manager.set_state_dict_fns(load_state_dict, state_dict)
            assert ft_manager is not None
            self.ft_replica_id = ft_manager.replica_id

        # FT may need staging even without async_with_pinned_mem
        if self.enable_ft_dataloader_checkpoints:
            self.enable_staging = True
            self.ft_states = {DATALOADER: dataloader}

            # FT needs gloo pg for async dataloader checkpoints
            if self.pg is None:
                self.pg = cast(dist.ProcessGroup, dist.new_group(backend="gloo"))

    @torch.no_grad()
    def save(self, curr_step: int, last_step: bool = False) -> None:
        # FT dataloader checkpoint is saved every step (not gated by interval)
        # to minimize data replay on replica failure.
        if self.enable_ft_dataloader_checkpoints:
            self._ft_save(curr_step)

        if not self.enable_ft_dataloader_checkpoints or (
            self.ft_manager
            # pyrefly: ignore [missing-attribute]
            and self.ft_manager.participating_rank() == 0
        ):
            super().save(curr_step, last_step)
        elif self.enable_ft_dataloader_checkpoints:
            assert self.ft_manager is not None
            logger.info(
                "Replica %d doesn't save checkpoint.",
                # pyrefly: ignore [missing-attribute]
                self.ft_manager.participating_rank(),
            )

    @torch.no_grad()
    def load(self, step: int = -1) -> bool:
        if self.enable_ft_dataloader_checkpoints:
            self._ft_load()
        return super().load(step)

    def _states_to_load(self, model_only: bool) -> dict[str, Any]:
        states = super()._states_to_load(model_only)
        if self.enable_ft_dataloader_checkpoints:
            states.pop(DATALOADER, None)
        return states

    def _async_wait(self) -> None:
        # _ft_save() always uses AsyncMode.ASYNC (regardless of self.async_mode),
        # so save_future can exist even when self.async_mode is DISABLED. The base
        # class would incorrectly raise in that case, so we override to handle it.
        if self.save_future is None:
            return
        self.save_future.result()
        # ASYNC_WITH_PINNED_MEM: the stager manages the future's lifecycle;
        # all other modes (ASYNC, DISABLED with FT) should clear the future.
        if self.async_mode != AsyncMode.ASYNC_WITH_PINNED_MEM:
            self.save_future = None

    def _should_purge(self) -> bool:
        if not super()._should_purge():
            return False
        if self.enable_ft_dataloader_checkpoints:
            # pyrefly: ignore [missing-attribute]
            return bool(self.ft_manager and self.ft_manager.participating_rank() == 0)
        return True

    def _ft_folder(self) -> str:
        return os.path.join(self.folder, f"ft-replicat-{self.ft_replica_id}")

    def _ft_save(self, step: int) -> None:
        begin = time.monotonic()
        self._async_wait()
        checkpoint_id = self._create_checkpoint_id(step, folder=self._ft_folder())
        self.save_future = self.dcp_save(
            self.ft_states, checkpoint_id=checkpoint_id, async_mode=AsyncMode.ASYNC
        )
        logger.info(f"Staging ft checkpoint took {time.monotonic() - begin} secs.")

    def _ft_load(self) -> None:
        step = self._find_load_step(folder=self._ft_folder())
        if step == -1:
            return

        begin = time.monotonic()
        logger.info(f"Loading the FT checkpoint at step {step}.")
        checkpoint_id = self._create_checkpoint_id(step, folder=self._ft_folder())
        self.dcp_load(
            self.ft_states,
            checkpoint_id=checkpoint_id,
            from_hf=False,
            from_quantized=False,
        )
        GarbageCollection.collect("GC collection for checkpoint loading.")
        logger.info(
            f"Finished loading the ft checkpoint in "
            f"{time.monotonic() - begin:.2f} seconds."
        )
