# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch.nn as nn
from torch_checkpointing.barriers import TCPStoreBarrierConfig
from torch_checkpointing.checkpoint_manager import (
    CheckpointManager as BackendCheckpointManager,
)
from torch_checkpointing.checkpoint_writer import CheckpointWriterConfig
from torch_checkpointing.config import AsyncCheckpointSaverConfig
from torch_checkpointing.default_resharder import DefaultResharder
from torch_checkpointing.schema import ItemSpec
from torch_checkpointing.staging import CheckpointStagerConfig
from torch_checkpointing.storage.base_storage import Storage
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.tools import filesystem

from .base import (
    BaseCheckpointManager,
    DATALOADER,
    LR_SCHEDULER,
    MODEL,
    ModelWrapper,
    OPTIMIZER,
)

DEFAULT_TORCH_CHECKPOINTING_BARRIER_TCPSTORE_PORT = 43001
_DEFAULT_BARRIER_INIT_TIMEOUT_SEC = 60
_DEFAULT_BARRIER_TIMEOUT_SEC = 600


class _BackendCheckpointStorage:
    """``CheckpointStorage`` backed by a ``torch_checkpointing`` ``Storage``.

    Path probes have to go through the same ``Storage`` the backend saves and
    loads with, or a caller-supplied remote storage would be written by the
    backend and read by something else.

    ``Storage`` has no ``isfile``, so it is composed from the two halves it does
    have. ``exists`` covers files and directories alike, so excluding
    directories leaves exactly the existing non-directory entries.

    ``Path`` would mangle a remote URI -- it collapses the double slash in
    ``gs://bucket/x`` -- but every path arriving here is joined off
    ``checkpoint.folder`` or ``checkpoint.initial_load_path``, and the manager
    rejects a remote value for either at construction. So there is nothing left
    to guard against by the time a path reaches this class.
    """

    def __init__(self, storage: Storage) -> None:
        self._storage = storage

    def isdir(self, path: str) -> bool:
        return self._storage.isdir(Path(path))

    def isfile(self, path: str) -> bool:
        target = Path(path)
        return self._storage.exists(target) and not self._storage.isdir(target)

    def listdir(self, path: str) -> list[str]:
        return self._storage.ls(Path(path))

    def remove(self, path: str) -> None:
        self._storage.rmdir(Path(path))


def _item_specs() -> dict[str, ItemSpec]:
    resharder = DefaultResharder()
    return {
        MODEL: ItemSpec(
            requires_copy=True,
            resharder=resharder,
            required=False,
        ),
        OPTIMIZER: ItemSpec(
            requires_copy=True,
            resharder=resharder,
            required=False,
        ),
    }


def _default_backend_config() -> BackendCheckpointManager.Config:
    barrier_timeout_sec = _DEFAULT_BARRIER_TIMEOUT_SEC
    save_config = AsyncCheckpointSaverConfig(
        writer_config=CheckpointWriterConfig(
            checkpoint_write_barrier_timeout_sec=barrier_timeout_sec,
            barrier_config=TCPStoreBarrierConfig(
                master_address=os.environ.get("MASTER_ADDR", "localhost"),
                tcpstore_port=DEFAULT_TORCH_CHECKPOINTING_BARRIER_TCPSTORE_PORT,
                timeout_barrier_init_sec=_DEFAULT_BARRIER_INIT_TIMEOUT_SEC,
                use_checkpoint_barrier_tcpstore_libuv=True,
            ),
        ),
        staging_config=CheckpointStagerConfig(use_pinned_memory=True),
        wait_timeout_secs=barrier_timeout_sec,
    )
    return BackendCheckpointManager.Config(
        items=_item_specs(),
        default=ItemSpec(requires_copy=False),
        save=save_config,
    )


class TorchCheckpointingManager(BaseCheckpointManager):
    """TorchTitan checkpoint manager backed by ``torch_checkpointing``."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseCheckpointManager.Config):
        pass

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
    ) -> None:
        self.enable = config.enable
        if not self.enable:
            return
        self.save_future = None

        self.folder = filesystem.join(base_folder, config.folder)
        # Checked here, not just in the storage adapter: a save runs no path
        # probe when retention is off, so it would otherwise reach the backend
        # and be mangled by Path() rather than failing.
        for label, candidate in (
            ("checkpoint.folder", self.folder),
            ("checkpoint.initial_load_path", config.initial_load_path),
        ):
            if candidate and filesystem.is_remote(candidate):
                raise ValueError(
                    f"{label} is a remote URI ({candidate!r}); remote URIs are "
                    "not yet supported by torch_checkpointing. Use the DCP "
                    "checkpoint manager for remote storage."
                )
        self.interval = config.interval
        self.states = states
        self.states.update(
            {
                MODEL: ModelWrapper(model_parts),
                OPTIMIZER: optimizers,
                DATALOADER: dataloader,
                LR_SCHEDULER: lr_schedulers,
            }
        )

        self.load_only = config.load_only
        self.exclude_from_loading = config.exclude_from_loading
        self.initial_load_path = config.initial_load_path
        self.initial_load_model_only = config.initial_load_model_only
        self.initial_load_in_hf = config.initial_load_in_hf
        self.initial_load_in_hf_quantized = config.initial_load_in_hf_quantized
        self.enable_first_step_checkpoint = config.enable_first_step_checkpoint
        self.last_save_model_only = config.last_save_model_only
        self.last_save_in_hf = config.last_save_in_hf
        self.export_dtype = TORCH_DTYPE_MAP[config.export_dtype]
        self.keep_latest_k = config.keep_latest_k
        self.sd_adapter = sd_adapter
        if self.last_save_in_hf and self.sd_adapter is None:
            raise ValueError(
                "checkpoint.last_save_in_hf is True, but sd_adapter is not provided."
            )

        manager_config = _default_backend_config()
        storage_config = manager_config.storage_config or LocalFileSystemStorageConfig()
        self._storage = _BackendCheckpointStorage(storage_config.create_storage())
        self._manager = manager_config.build()

    def __del__(self) -> None:
        self.close()

    # Save and load routing land in later changes; this one only plumbs config.
    # The methods are stubbed rather than omitted because BaseCheckpointManager
    # declares them abstract, so a partial implementation cannot be instantiated.

    def _load(self, step: int = -1) -> bool:
        raise NotImplementedError(
            "TorchCheckpointingManager does not implement load() yet."
        )

    def _save(self, curr_step: int, last_step: bool = False) -> bool:
        raise NotImplementedError(
            "TorchCheckpointingManager does not implement save() yet."
        )

    def _wait_for_saving(self) -> None:
        raise NotImplementedError(
            "TorchCheckpointingManager does not implement saving yet."
        )

    def _maybe_wait_for_staging(self) -> None:
        raise NotImplementedError(
            "TorchCheckpointingManager does not implement maybe_wait_for_staging() yet."
        )

    def _parse_step(self, checkpoint_name: str) -> int | None:
        raise NotImplementedError(
            "TorchCheckpointingManager does not implement checkpoint discovery yet."
        )

    def _close(self) -> None:
        # hasattr: __del__ -> close() can reach here on a partially constructed
        # object if __init__ raised after setting enable but before building the
        # backend manager.
        if hasattr(self, "_manager"):
            self._manager.close()
