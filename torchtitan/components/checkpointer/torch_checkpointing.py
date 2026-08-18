# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import logging
import os
import queue
import re
import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict_saver import _stateful_to_state_dict
from torch_checkpointing.barriers import TCPStoreBarrierConfig
from torch_checkpointing.checkpoint_layout import LayoutInfo, SafetensorsSerialization
from torch_checkpointing.checkpoint_manager import (
    CheckpointManager as BackendCheckpointManager,
)
from torch_checkpointing.checkpoint_writer import CheckpointWriterConfig
from torch_checkpointing.config import (
    AsyncCheckpointSaverConfig,
    SyncCheckpointSaverConfig,
)
from torch_checkpointing.distributed_metadata import (
    METADATA_FILE_NAME as TORCH_CHECKPOINTING_METADATA_FILE_NAME,
)
from torch_checkpointing.dtensor_resharder import DTensorResharder
from torch_checkpointing.hf.consolidation import consolidate_hf_safetensors_checkpoint
from torch_checkpointing.logging_utils import checkpoint_logging_context
from torch_checkpointing.schema import ItemSpec
from torch_checkpointing.staging import CheckpointStagerConfig
from torch_checkpointing.storage.base_storage import Storage, StorageConfig
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torchtitan.components.dataloader import BaseDataLoader
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

# Index the HF consolidation writes at the root of a final export; the
# backend names it after the checkpoint item it consolidated.
_HF_INDEX_FILE_NAME = f"{MODEL}.safetensors.index.json"

# Logger the backend emits its checkpoint events and metrics on.
_BACKEND_LOGGER_NAME = "torch_checkpointing"


def _step_dir_pattern(temp_dir_prefix: str) -> re.Pattern[str]:
    """Match the two directory names the backend writes into checkpoint.folder.

    Those are the published ``step-N`` and, while a save is still in flight, the
    same name under ``temp_dir_prefix``. Anything else in the folder was not
    written by us, so we never delete it.

    The prefix is read from the writer config rather than hardcoded: the backend
    exposes it as ``CheckpointWriterConfig.temp_dir_prefix``, so a caller can
    change it and a hardcoded pattern would then quietly stop recognizing
    in-flight saves.
    """
    return re.compile(rf"(?P<tmp>{re.escape(temp_dir_prefix)})?step-(?P<step>\d+)")


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


def _init_subprocess_logging(
    output_dir: str,
    init_fn: Callable[..., None] | None,
    init_args: tuple[Any, ...],
) -> None:
    """Re-establish structured logging inside the async save subprocess.

    The subprocess does not inherit the parent's logging handlers, so its
    checkpoint records would otherwise be lost.
    """
    sl.init_structured_logger(source="training", output_dir=output_dir)
    # Handlers are not the only thing missing. A fresh subprocess has no logging
    # configuration at all, so this logger sits at NOTSET and inherits root's
    # default of WARNING. Logger.info() tests that level before it builds a
    # record, so the backend's INFO checkpoint events would be dropped there --
    # ahead of every handler, including the one installed just below. Raising
    # the level is what makes that handler reachable at all.
    #
    # min(), not a bare setLevel(INFO): someone debugging a checkpoint problem
    # may have set this logger to DEBUG, and lowering it back to INFO would
    # quietly discard the verbosity they asked for.
    backend_logger = logging.getLogger(_BACKEND_LOGGER_NAME)
    backend_logger.setLevel(min(backend_logger.getEffectiveLevel(), logging.INFO))
    sl.install_forwarding_structured_logging_handler(_BACKEND_LOGGER_NAME)
    if init_fn is not None:
        init_fn(*init_args)


def _with_structured_logging(
    config: BackendCheckpointManager.Config,
    output_dir: str,
) -> BackendCheckpointManager.Config:
    """Forward the backend's own log records into TorchTitan's structured log.

    No-op when structured logging is not active, or when saves are synchronous
    and therefore already run in this process with the handler installed. Any
    existing ``subprocess_init_fn`` is chained rather than replaced.
    """
    if not sl.install_forwarding_structured_logging_handler(_BACKEND_LOGGER_NAME):
        return config
    if not isinstance(config.save, AsyncCheckpointSaverConfig):
        return config
    return replace(
        config,
        subprocess_init_fn=_init_subprocess_logging,
        subprocess_init_args=(
            output_dir,
            config.subprocess_init_fn,
            config.subprocess_init_args,
        ),
    )


def _item_specs() -> dict[str, ItemSpec]:
    resharder = DTensorResharder()
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


def _with_sync_save(
    config: BackendCheckpointManager.Config,
    *,
    use_barrier: bool = True,
) -> BackendCheckpointManager.Config:
    """Return ``config`` with its async saver swapped for a synchronous one.

    Used for the final save, which must complete before the process exits, and
    for load-only runs, where no save is expected and the barrier would block
    against ranks that never save.
    """
    writer_config = copy.deepcopy(config.save.writer_config)
    if not use_barrier:
        writer_config = replace(writer_config, barrier_config=None)
    return replace(
        config,
        save=SyncCheckpointSaverConfig(
            writer_config=writer_config,
            wait_timeout_secs=config.save.wait_timeout_secs,
        ),
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
                    f"{label} is a remote URI ({candidate!r}), which "
                    "torch_checkpointing cannot address. Use the DCP checkpoint "
                    "manager for remote storage."
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

        manager_config = _default_backend_config()
        if self.load_only:
            manager_config = _with_sync_save(manager_config, use_barrier=False)
        elif (
            dist.is_initialized()
            and dist.get_world_size() > 1
            and manager_config.save.writer_config.barrier_config is None
        ):
            raise ValueError(
                "TorchCheckpointingManager requires a checkpoint barrier for "
                "multi-rank saves."
            )
        # An explicit storage_config wins, and is pushed into the backend config
        # so saves and loads use it too, not just our own path probes.
        if storage_config is not None:
            manager_config = replace(manager_config, storage_config=storage_config)
        manager_config = _with_structured_logging(manager_config, base_folder)
        self._manager_config = manager_config
        self._step_dir_pattern = _step_dir_pattern(
            manager_config.save.writer_config.temp_dir_prefix
        )
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
        # The backend stamps its own events from this context and carries it
        # into the async save subprocess, so without it every forwarded backend
        # metric reports step=None.
        checkpoint_logging_context.update(step=curr_step)
        self.maybe_wait_for_saving()
        # Purge before issuing this step's save, while the folder holds only
        # settled state: the previous save has been awaited and the next has not
        # started, so nothing here can be mistaken for an in-flight checkpoint.
        self._purge_stale_checkpoints(is_save_in_flight=False)

        if last_step:
            self._save_last_step(curr_step)
        else:
            self.save_future = self._manager.save(
                self._create_checkpoint_id(curr_step),
                _stateful_to_state_dict(self.states),
            )
            self._prewarmed = True

        return True

    def _parse_step(self, filename: str) -> tuple[int, bool] | None:
        match = self._step_dir_pattern.fullmatch(filename)
        if match is None:
            return None
        return int(match.group("step")), bool(match.group("tmp"))

    def _is_valid_checkpoint(self, checkpoint_id: str) -> bool:
        # Either shape this manager publishes. A resumable checkpoint has the
        # backend's metadata at its root. A final HF export does not: its
        # backend metadata sits in the nested "sharded" directory the shards
        # were written to, and the root holds the consolidated HF files. Probing
        # only for the former would classify a finished export as abandoned and
        # let the next run's retention delete it.
        return self._storage.isfile(
            filesystem.join(checkpoint_id, TORCH_CHECKPOINTING_METADATA_FILE_NAME)
        ) or self._storage.isfile(filesystem.join(checkpoint_id, _HF_INDEX_FILE_NAME))

    def _maybe_wait_for_staging(self) -> None:
        # Acquiring the backend lock is what blocks until staging for the last
        # save has drained; there is no separate staging future to await. Safe
        # after _save_last_step has closed the manager: the lock is a no-op once
        # nothing is staging.
        with self._manager.lock():
            pass

    def _wait_for_saving(self) -> None:
        # Cleared before awaiting so a failed save is not retried on close().
        # Narrowing for the type checker: maybe_wait_for_saving only dispatches
        # here when save_future is set. Cleared before awaiting so a failed save
        # is not retried on close().
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
                # hasattr: __del__ -> close() can reach here on a partially
                # constructed object if __init__ raised after setting enable but
                # before building the backend manager. No guard against closing
                # twice is needed -- _save_last_step may already have closed it,
                # and the backend's close() returns immediately when it has.
                if hasattr(self, "_manager"):
                    self._manager.close()

    def _save_last_step(self, curr_step: int) -> None:
        if self.last_save_model_only:
            model_state = self.states[MODEL].state_dict()
            # Matches the DCP manager (#4166): convert per tensor rather than
            # gating on export_dtype != float32, which skipped the conversion
            # entirely for BF16 training exporting to FP32, and cast integer and
            # boolean buffers when it did run.
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
        checkpoint_id = self._create_checkpoint_id(curr_step)
        self._manager.close()
        manager_config = _with_sync_save(self._manager_config)
        input_checkpoint_id = checkpoint_id
        if self.last_save_in_hf:
            assert self.sd_adapter is not None
            states = {MODEL: self.sd_adapter.to_hf(states[MODEL])}
            # Ranks write safetensors shards into a nested directory; the
            # pre-finalize callback consolidates them up into checkpoint_id, so
            # the published checkpoint is HF-layout rather than sharded.
            input_checkpoint_id = filesystem.join(checkpoint_id, "sharded")
            item_specs = dict(manager_config.items)
            model_spec = item_specs.get(
                MODEL,
                ItemSpec(requires_copy=True, required=False),
            )
            item_specs[MODEL] = replace(
                model_spec,
                layout=LayoutInfo(
                    f"{MODEL}_{{rank}}.safetensors",
                    SafetensorsSerialization(),
                ),
            )
            fqn_to_index_mapping = self.sd_adapter.fqn_to_index_mapping
            hf_storage_config = (
                manager_config.storage_config
                or LocalFileSystemStorageConfig(use_direct_io=False)
            )
            manager_config = replace(
                manager_config,
                items=item_specs,
                # The backend hands the callback the directory the shards were
                # actually written to -- its staging directory when a write
                # barrier is configured, the final path otherwise -- so
                # consolidate from that path as given, deriving nothing from it.
                pre_finalize_callback=lambda staged, _event_logger: (
                    consolidate_hf_safetensors_checkpoint(
                        staged,
                        output_dir=checkpoint_id,
                        item_key=MODEL,
                        fqn_to_index_mapping=fqn_to_index_mapping,
                        storage_config=hf_storage_config,
                    )
                ),
            )
        manager = manager_config.build()
        try:
            manager.save(
                input_checkpoint_id,
                _stateful_to_state_dict(states),
            )
        finally:
            manager.close()
        GarbageCollection.collect("GC collection invoked by checkpointer.")

    def _should_prewarm(self) -> bool:
        return self.enable and not self._prewarmed and not self.load_only
