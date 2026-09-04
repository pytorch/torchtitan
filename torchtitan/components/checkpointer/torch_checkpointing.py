# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict_saver import _stateful_to_state_dict
from torch_checkpointing.barriers import TCPStoreBarrierConfig
from torch_checkpointing.checkpoint_manager import (
    CheckpointManager as BackendCheckpointManager,
)
from torch_checkpointing.checkpoint_writer import CheckpointWriterConfig
from torch_checkpointing.config import (
    AsyncCheckpointSaverConfig,
    CheckpointSaverConfig,
    SyncCheckpointSaverConfig,
)
from torch_checkpointing.default_resharder import DefaultResharder
from torch_checkpointing.distributed_metadata import (
    METADATA_FILE_NAME as TORCH_CHECKPOINTING_METADATA_FILE_NAME,
)
from torch_checkpointing.schema import ItemSpec
from torch_checkpointing.staging import CheckpointStagerConfig
from torch_checkpointing.storage.base_storage import Storage, StorageConfig
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torchtitan.components.data.loader import BaseDataLoader
from torchtitan.components.optimizer import LRSchedulersContainer, OptimizersContainer
from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.tools import filesystem
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import GarbageCollection

from .base import (
    BaseCheckpointManager,
    DATALOADER,
    LR_SCHEDULER,
    MODEL,
    ModelWrapper,
    OPTIMIZER,
    purge_thread,
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


def _writer_config(*, use_barrier: bool) -> CheckpointWriterConfig:
    return CheckpointWriterConfig(
        checkpoint_write_barrier_timeout_sec=_DEFAULT_BARRIER_TIMEOUT_SEC,
        barrier_config=(
            TCPStoreBarrierConfig(
                master_address=os.environ.get("MASTER_ADDR", "localhost"),
                tcpstore_port=DEFAULT_TORCH_CHECKPOINTING_BARRIER_TCPSTORE_PORT,
                timeout_barrier_init_sec=_DEFAULT_BARRIER_INIT_TIMEOUT_SEC,
                use_checkpoint_barrier_tcpstore_libuv=True,
            )
            if use_barrier
            else None
        ),
    )


def _async_save_config() -> AsyncCheckpointSaverConfig:
    return AsyncCheckpointSaverConfig(
        writer_config=_writer_config(use_barrier=True),
        staging_config=CheckpointStagerConfig(use_pinned_memory=True),
        wait_timeout_secs=_DEFAULT_BARRIER_TIMEOUT_SEC,
    )


def _sync_save_config(*, use_barrier: bool = True) -> SyncCheckpointSaverConfig:
    return SyncCheckpointSaverConfig(
        writer_config=_writer_config(use_barrier=use_barrier),
        wait_timeout_secs=_DEFAULT_BARRIER_TIMEOUT_SEC,
    )


def _default_backend_config(
    save_config: CheckpointSaverConfig,
    *,
    storage_config: StorageConfig | None = None,
) -> BackendCheckpointManager.Config:
    return BackendCheckpointManager.Config(
        items=_item_specs(),
        default=ItemSpec(requires_copy=False),
        save=save_config,
        storage_config=storage_config,
    )


class TorchCheckpointingManager(BaseCheckpointManager):
    """TorchTitan checkpoint manager backed by ``torch_checkpointing``.

    Args:
        storage_config: Backend storage for reading and writing checkpoints.
            Defaults to the local filesystem. An init parameter rather than a
            ``Config`` field because ``Configurable.Config`` is Tyro-parsed and
            a backend storage object is not a command-line surface; callers that
            need remote storage pass it programmatically.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseCheckpointManager.Config):
        def __post_init__(self) -> None:
            BaseCheckpointManager.Config.__post_init__(self)
            if self.last_save_in_hf:
                raise ValueError(
                    "TorchCheckpointingManager does not support last_save_in_hf yet."
                )

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
        storage_config: StorageConfig | None = None,
    ) -> None:
        self.enable = config.enable
        if not self.enable:
            return
        self.save_future: Future[Any] | None = None
        self.purge_thread: threading.Thread | None = None

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
        self.purge_exempt = (
            config.purge_exempt.build() if config.purge_exempt is not None else None
        )

        save_config = (
            _sync_save_config(use_barrier=False)
            if self.load_only
            else _async_save_config()
        )
        manager_config = _default_backend_config(
            save_config,
            storage_config=storage_config,
        )
        self._manager_config = manager_config
        storage_config = (
            self._manager_config.storage_config or LocalFileSystemStorageConfig()
        )
        self._storage = _BackendCheckpointStorage(storage_config.create_storage())
        self._prewarmed = False

        self.sd_adapter = sd_adapter
        if self.last_save_in_hf and self.sd_adapter is None:
            raise ValueError(
                "checkpoint.last_save_in_hf is True, but sd_adapter is not provided."
            )

        self._manager = self._manager_config.build()

        if self.keep_latest_k > 0:
            self.purge_queue: queue.Queue[str | None] = queue.Queue()
            self.purge_thread = threading.Thread(
                target=purge_thread,
                args=(self.purge_queue, self._storage.remove),
                daemon=True,
            )
            self.purge_thread.start()

        logger.info(
            "Checkpointing active. Checkpoints will be loaded from and saved "
            f"to {self.folder}"
        )

    def __del__(self) -> None:
        # __init__ can fail before the backend manager is built. In that case,
        # this object owns no backend resources to close.
        if hasattr(self, "_manager"):
            self.close()

    # Load routing lands in a later change.
    def _load(self, step: int = -1) -> bool:
        raise NotImplementedError(
            "TorchCheckpointingManager does not implement load() yet."
        )

    @sl.log_trace_span("checkpoint_save")
    @torch.no_grad()
    def _save(self, curr_step: int, last_step: bool = False) -> bool:
        should_save = self._should_save(curr_step, last_step)
        # Prewarm on a step we are not saving, so the first real save does not
        # pay for pinned-buffer allocation.
        if not should_save and self._should_prewarm():
            self._manager.prewarm_staging(_stateful_to_state_dict(self.states))
            self._prewarmed = True
        if not should_save:
            return False

        sl.add_step_tag("checkpoint_save")
        self.maybe_wait_for_saving()
        # Always preserve the current step's published and staging directories.
        self._purge_stale_checkpoints(
            saving_step=curr_step,
            staging_dir_prefix=(
                self._manager_config.save.writer_config.temp_dir_prefix
            ),
        )

        if last_step:
            self._save_last_step(curr_step)
        else:
            self.save_future = self._manager.save(
                self._create_checkpoint_id(curr_step),
                _stateful_to_state_dict(self.states),
            )
            self._prewarmed = True

        return True

    def _is_valid_checkpoint(self, checkpoint_dir: str) -> bool:
        return self._storage.isfile(
            filesystem.join(checkpoint_dir, TORCH_CHECKPOINTING_METADATA_FILE_NAME)
        )

    def _maybe_wait_for_staging(self) -> None:
        # BaseCheckpointManager.close() calls this to wait for in-flight staging.
        # If _save_last_step already closed the backend manager, that close drained
        # staging but left this lock usable, so acquiring it here cannot hang.
        with self._manager.lock():
            pass

    def _wait_for_saving(self) -> None:
        # Clear the active save before waiting so close() does not retry a failure.
        save_future = self.save_future
        assert save_future is not None
        self.save_future = None
        save_future.result(timeout=self._manager_config.save.wait_timeout_secs)

    def _close(self) -> None:
        try:
            self.maybe_wait_for_saving()
        finally:
            try:
                if self.purge_thread is not None and self.purge_thread.is_alive():
                    self.purge_queue.put(None)
                    self.purge_thread.join()
            finally:
                # _save_last_step may already have closed the manager; the
                # backend's close() returns immediately when it has.
                self._manager.close()

    def _save_last_step(self, curr_step: int) -> None:
        if self.last_save_model_only:
            model_state = self.states[MODEL].state_dict()
            # Cast floating-point tensors to the export dtype and preserve other
            # buffers.
            model_state = {
                key: value.to(self.export_dtype)
                if isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and value.dtype != self.export_dtype
                else value
                for key, value in model_state.items()
            }
            states: dict[str, Any] = {MODEL: model_state}
            logger.info(
                f"Saving a model only checkpoint in {self.export_dtype} "
                f"at last step, step {curr_step}."
            )
        else:
            states = self.states
            logger.info(f"Saving a full checkpoint at last step, step {curr_step}.")

        # The final save must land before the process exits, so retire the async
        # manager and write synchronously through a fresh one.
        self._manager.close()
        manager = _default_backend_config(
            _sync_save_config(),
            storage_config=self._manager_config.storage_config,
        ).build()
        try:
            manager.save(
                self._create_checkpoint_id(curr_step),
                _stateful_to_state_dict(states),
            )
        finally:
            manager.close()
        GarbageCollection.collect("GC collection invoked by checkpointer.")

    def _should_prewarm(self) -> bool:
        return self.enable and not self._prewarmed and not self.load_only
