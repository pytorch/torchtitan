# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import queue
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.stateful import Stateful

from torch.distributed.tensor import DTensor
from torchtitan.config import Configurable
from torchtitan.tools import filesystem
from torchtitan.tools.logging import logger

MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"
PostCheckpointCallback = Callable[[int, str], None]
CallbackTrigger = Callable[[int], bool]
RetentionExemption = Callable[[int], bool]


@dataclass(frozen=True, slots=True)
class _PostCheckpointRegistration:
    """One callback and the checkpoint policies associated with it."""

    callback: PostCheckpointCallback
    callback_trigger: CallbackTrigger | None
    retention_exemption: RetentionExemption | None


def purge_thread(
    purge_queue: queue.Queue[str | None],
    remove_path: Callable[[str], None],
) -> None:
    """Thread to purge the old checkpoints.

    This is only used when keep_latest_k > 0.

    Args:
        purge_queue (queue.Queue): The queue to receive paths to purge and the
            ``None`` shutdown sentinel.
    """
    try:
        while True:
            path = purge_queue.get()
            if path is None:
                return
            logger.info("Checkpointer is deleting %s.", path)
            begin = time.monotonic()
            # A single failed deletion (e.g. a transient remote error) must not
            # kill this daemon thread; otherwise keep_latest_k would silently
            # stop purging for the rest of the run.
            try:
                remove_path(path)
            except Exception as error:
                logger.warning(
                    "Checkpointer failed to delete %s: %s. Skipping.", path, error
                )
                continue
            logger.info(
                "Checkpointer deleted %s in %.2f seconds.",
                path,
                time.monotonic() - begin,
            )
    finally:
        logger.info("Destroying the purge thread.")


def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Whether ``a`` and ``b`` are backed by the same storage.

    For ``DTensor`` the local shard's storage is compared via ``_local_tensor``
    rather than ``to_local()``, which is autograd-aware; this is a read-only
    identity check on the local storage.
    """
    if isinstance(a, DTensor):
        a = a._local_tensor
    if isinstance(b, DTensor):
        b = b._local_tensor
    return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()


class ModelWrapper(Stateful):
    """
    A wrapper for `nn.Module` (or a list of modules) that provides a unified `Stateful`
    interface for distributed checkpointing.

    This class serves two purposes:
        1. Flattening/Aggregation: It combines the state dicts of multiple
           different modules (like individual chunks in Pipeline Parallelism)
           into a single flat view so checkpointing code can interact
           with them through a unified interface.
        2. Stable-storage caching: It caches the flattened state dict and, on
           every `state_dict()` call, returns tensors backed by the same
           storage. Async DCP staging may cache pinned host buffers keyed by the
           source storage, so keeping the storage stable lets it reuse those
           buffers across saves (the fast checkpoint path). Parameter tensors
           already satisfy this because the cached view shares the parameter
           storage; tensors produced by module `state_dict` hooks (e.g. one that
           splits a fused parameter) may be freshly allocated each call, so they
           are refreshed in place to keep their storage stable while their values
           track the current parameters.

    Notes:
        - Calling `load_state_dict` updates the underlying modules and
        refreshes the cached state_dict.
        - The model architecture should not be structurally modified (e.g.,
        changing keys or replacing tensor references) after wrapping, or the
        cache will become stale.
    """

    def __init__(self, model: nn.Module | list[nn.Module]) -> None:
        self.model = [model] if isinstance(model, nn.Module) else model
        self.cached_state_dict = self._get_state_dict()

    def _get_state_dict(self) -> dict[str, Any]:
        # TorchTitan already makes model state_dict keys canonical.
        return {k: v for model in self.model for k, v in model.state_dict().items()}

    def state_dict(self) -> dict[str, Any]:
        # Recompute the state dict so hook-produced tensors reflect the current
        # parameters, then merge into the cache without changing storage objects.
        for key, value in self._get_state_dict().items():
            cached = self.cached_state_dict.get(key)
            if (
                cached is None
                or cached.shape != value.shape
                or cached.dtype != value.dtype
            ):
                self.cached_state_dict[key] = value
            elif not _shares_storage(cached, value):
                cached.copy_(value)
        return self.cached_state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        # strict=False because state_dict is the flattened checkpoint dict, which
        # mixes model FQN keys with non-model keys (optimizer, lr_scheduler, ...).
        for model in self.model:
            model.load_state_dict(state_dict, strict=False)
        # Refresh the cache so state_dict() reflects the freshly loaded values.
        self.cached_state_dict = self._get_state_dict()


@runtime_checkable
class CheckpointStorage(Protocol):
    """The path operations a checkpoint manager needs from its storage.

    Managers differ in how they read and write checkpoint bytes, but they ask
    the same handful of questions about paths: is this a checkpoint directory,
    did this metadata file land, which steps are on disk, delete this one. This
    protocol is the whole of that surface, so policies like retention and
    latest-step discovery can live on ``BaseCheckpointManager`` without knowing
    which backend answers them.

    Paths are ``str`` rather than ``Path`` because a checkpoint id may be a
    remote URI (``gs://...``) that ``Path`` would mangle -- it collapses the
    double slash. Carrying ``str`` keeps the vocabulary lossless; whether a
    given implementation can actually reach a remote URI is up to that
    implementation, which should reject what it cannot address rather than
    silently rewrite it.

    ``runtime_checkable`` so implementations can assert conformance in their
    tests. It only checks that the method names exist, which is enough to catch
    a rename that would otherwise surface as an ``AttributeError`` mid-save.
    """

    def isdir(self, path: str) -> bool:
        """Whether ``path`` is an existing directory."""
        ...

    def isfile(self, path: str) -> bool:
        """Whether ``path`` is an existing entry that is not a directory."""
        ...

    def listdir(self, path: str) -> list[str]:
        """The entry names directly under the directory ``path``."""
        ...

    def remove(self, path: str) -> None:
        """Recursively delete the directory ``path``."""
        ...


class BaseCheckpointManager(Configurable, ABC):
    """Contract every TorchTitan checkpoint manager implements.

    Subclasses must declare their own nested ``Config``, even an empty one:
    ``Configurable.__init_subclass__`` sets ``Config._owner`` on the class that
    declares it and ``build()`` constructs ``_owner``, so a subclass that
    inherited ``Config`` unchanged would build this abstract base instead.
    """

    enable: bool
    save_future: Future | None
    folder: str
    keep_latest_k: int
    _storage: CheckpointStorage

    # State for the callback invoked after a checkpoint is fully persisted.
    _save_pending_finish: bool = False
    _post_checkpoint_callback_completion: Future[None] | None = None
    _post_checkpoint_registration: _PostCheckpointRegistration | None = None

    # A disabled manager returns early from ``__init__`` without setting up any
    # state, so none of its attributes exist. The public entry points below own
    # that check once, on behalf of every implementation, and dispatch to the
    # ``_``-prefixed hooks only when enabled. Subclasses override the hooks, not
    # these methods: overriding a public method here would silently bypass the
    # guard and run against uninitialized state.

    def register_post_checkpoint_callback(
        self,
        callback: PostCheckpointCallback,
        *,
        callback_trigger: CallbackTrigger | None = None,
        retention_exemption: RetentionExemption | None = None,
    ) -> None:
        """Register one post-checkpoint callback and its checkpoint policies."""
        self._post_checkpoint_registration = _PostCheckpointRegistration(
            callback=callback,
            callback_trigger=callback_trigger,
            retention_exemption=retention_exemption,
        )

    def _is_checkpoint_requested_by_callback_trigger(self, step: int) -> bool:
        """Whether the configured callback trigger requests ``step``."""
        registration = self._post_checkpoint_registration
        return (
            registration is not None
            and registration.callback_trigger is not None
            and registration.callback_trigger(step)
        )

    def _is_retention_exempt(self, step: int) -> bool:
        """Whether the configured exemption protects ``step`` from deletion."""
        registration = self._post_checkpoint_registration
        return (
            registration is not None
            and registration.retention_exemption is not None
            and registration.retention_exemption(step)
        )

    def _run_post_checkpoint_callback(self, step: int, checkpoint_id: str) -> None:
        """Invoke the callback configured for a completed checkpoint save."""
        registration = self._post_checkpoint_registration
        if registration is not None:
            registration.callback(step, checkpoint_id)

    def _track_checkpoint_save(self, step: int, checkpoint_id: str) -> None:
        """Start the post-persistence lifecycle of a save that was just issued.

        A synchronous save is already durable here, so its callback runs
        immediately and the lifecycle finishes before this returns. An
        asynchronous save is still in flight: its callback runs later, on the
        completion thread of ``save_future``, and ``maybe_wait_for_saving`` is
        what finishes the lifecycle -- including whatever a concrete manager
        defers to ``_finish_checkpoint_save``, such as retention cleanup.
        """
        assert not self._save_pending_finish
        self._save_pending_finish = True

        if self.save_future is None:
            try:
                self._run_post_checkpoint_callback(step, checkpoint_id)
            finally:
                self._finish_checkpoint_save()
            return

        post_checkpoint_callback_completion: Future[None] = Future()
        self._post_checkpoint_callback_completion = (
            post_checkpoint_callback_completion
        )
        self.save_future.add_done_callback(
            lambda future: self._run_post_checkpoint_callback_after_async_save(
                future, post_checkpoint_callback_completion, step, checkpoint_id
            )
        )

    def _run_post_checkpoint_callback_after_async_save(
        self,
        save_future: Future,
        post_checkpoint_callback_completion: Future[None],
        step: int,
        checkpoint_id: str,
    ) -> None:
        """Run the callback on the async-save completion thread.

        The separate completion future lets the training thread wait for the
        callback and receive its exception even though ``add_done_callback``
        does not return a future for the callback itself.
        """
        try:
            if not save_future.cancelled() and save_future.exception() is None:
                self._run_post_checkpoint_callback(step, checkpoint_id)
        except BaseException as error:
            post_checkpoint_callback_completion.set_exception(error)
        else:
            post_checkpoint_callback_completion.set_result(None)

    def _finish_checkpoint_save(self) -> None:
        """Finish bookkeeping once persistence and callbacks are both done.

        Concrete managers may extend this point with work that must happen only
        after post-checkpoint consumers have received the checkpoint, such as
        retention cleanup.
        """
        if not self._save_pending_finish:
            return
        self._save_pending_finish = False

    def load(self, step: int = -1) -> bool:
        """Restore state from ``step``, or the latest checkpoint when ``-1``."""
        if not self.enable:
            return False
        return self._load(step)

    def save(self, curr_step: int, last_step: bool = False) -> bool:
        """Persist state for ``curr_step``."""
        if not self.enable:
            return False
        if self.save_future is not None and self.save_future.done():
            self.maybe_wait_for_saving()
        return self._save(curr_step, last_step)

    def maybe_wait_for_staging(self) -> None:
        """Block until asynchronous staging for the last save completes."""
        if not self.enable:
            return
        self._maybe_wait_for_staging()

    def close(self) -> None:
        """Release background threads and other resources."""
        if not self.enable:
            return
        self.maybe_wait_for_saving()
        self._close()

    def maybe_wait_for_saving(self) -> None:
        """Block until the last asynchronous save completes.

        A manager with no asynchronous save in flight leaves ``save_future`` at
        ``None`` and never reaches ``_wait_for_saving``.
        """
        if not self.enable or self.save_future is None:
            return
        self._wait_for_saving()
        try:
            if self._post_checkpoint_callback_completion is not None:
                self._post_checkpoint_callback_completion.result()
        finally:
            self._post_checkpoint_callback_completion = None
            self._finish_checkpoint_save()

    @abstractmethod
    def _wait_for_saving(self) -> None:
        """Await ``save_future`` and clear it. Only called when it is set."""

    @abstractmethod
    def _load(self, step: int = -1) -> bool:
        """Implement ``load``. Only called when checkpointing is enabled."""

    @abstractmethod
    def _save(self, curr_step: int, last_step: bool = False) -> bool:
        """Implement ``save``. Only called when checkpointing is enabled."""

    @abstractmethod
    def _maybe_wait_for_staging(self) -> None:
        """Implement ``maybe_wait_for_staging``. Only called when enabled."""

    @abstractmethod
    def _close(self) -> None:
        """Implement ``close``. Only called when checkpointing is enabled."""

    def _should_purge(self) -> bool:
        """Whether this rank should purge stale checkpoints."""
        return (
            self.keep_latest_k > 0
            and dist.get_rank() == 0
            and self._storage.isdir(self.folder)
        )

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Checkpoint policies shared by concrete TorchTitan checkpoint managers."""

        enable: bool = False
        """Whether to enable checkpoint"""

        folder: str = "checkpoint"
        """Checkpoint folder, relative to the trainer dump folder."""

        interval: int = 500
        """Checkpointing interval in steps."""

        initial_load_path: str | None = None
        """Optional checkpoint path used when the output checkpoint folder is empty."""

        initial_load_model_only: bool = True
        """Whether an initial checkpoint restores only model state."""

        initial_load_in_hf: bool = False
        """Whether the initial checkpoint uses Hugging Face safetensors."""

        initial_load_in_hf_quantized: bool = False
        """Whether the initial Hugging Face checkpoint uses quantized keys."""

        last_save_model_only: bool = True
        """Whether the final checkpoint contains only model state."""

        last_save_in_hf: bool = False
        """Whether the final model-only checkpoint uses Hugging Face safetensors."""

        export_dtype: Literal["float16", "bfloat16", "float32"] = "float32"
        """Model dtype used by a final model-only checkpoint."""

        keep_latest_k: int = 10
        """Number of recent checkpoints to retain, or zero to retain all."""

        load_step: int = -1
        """Load the checkpoint at the specified step. If -1, load the latest
        checkpoint."""

        exclude_from_loading: list[str] = field(default_factory=list)
        """Non-model state keys excluded from loading."""

        enable_first_step_checkpoint: bool = False
        """Whether to save immediately after the first training step."""

        create_seed_checkpoint: bool = False
        """Whether to initialize and save an unsharded seed checkpoint."""

        load_only: bool = False
        """Whether to permit loads while disabling all saves."""

        def __post_init__(self) -> None:
            if not self.folder.strip():
                raise ValueError("The 'folder' field cannot be empty.")
            if self.interval < 1:
                raise ValueError("Checkpoint interval needs to be at least 1 step.")
            if self.load_step < -1:
                raise ValueError("load_step must be -1 or non-negative.")
            if self.keep_latest_k < 0:
                raise ValueError("keep_latest_k cannot be negative.")
            if self.keep_latest_k == 1:
                raise ValueError(
                    "We need to maintain at least 2 checkpoint replicas, "
                    "as the last one may be in the process of being saved."
                )
            if MODEL in self.exclude_from_loading:
                raise ValueError(f"{MODEL} key shouldn't be in exclude_from_loading.")
            if (
                OPTIMIZER in self.exclude_from_loading
                and LR_SCHEDULER not in self.exclude_from_loading
            ):
                raise ValueError(
                    f"{LR_SCHEDULER} must be excluded when {OPTIMIZER} is excluded."
                )

            if self.initial_load_path:
                self.initial_load_path = self.initial_load_path.strip()
                if not (
                    self.initial_load_path.startswith("/")
                    or filesystem.is_remote(self.initial_load_path)
                ):
                    raise ValueError(
                        "initial_load_path must be an absolute path or a remote "
                        f"URI (e.g. gs://...): {self.initial_load_path}"
                    )
            if self.initial_load_in_hf and not self.initial_load_model_only:
                raise ValueError("initial_load_in_hf requires initial_load_model_only.")
            if self.initial_load_in_hf_quantized and not (
                self.initial_load_in_hf and self.initial_load_path
            ):
                raise ValueError(
                    "initial_load_in_hf_quantized requires initial_load_in_hf "
                    "and initial_load_path."
                )
            if self.last_save_in_hf and not self.last_save_model_only:
                raise ValueError("last_save_in_hf requires last_save_model_only=True.")

            # Remote (fsspec) checkpoint IO supports only the native DCP format.
            # HF safetensors read/write to a remote URI is not implemented, so
            # reject the combination up front instead of failing deep in DCP.
            if self.last_save_in_hf and filesystem.is_remote(self.folder):
                raise ValueError(
                    "last_save_in_hf is not supported with a remote "
                    f"checkpoint.folder: {self.folder}"
                )
            if (
                self.initial_load_in_hf
                and self.initial_load_path
                and filesystem.is_remote(self.initial_load_path)
            ):
                raise ValueError(
                    "initial_load_in_hf is not supported with a remote "
                    f"initial_load_path: {self.initial_load_path}"
                )

            if self.load_only and self.enable_first_step_checkpoint:
                logger.warning(
                    "checkpoint.load_only is True; enable_first_step_checkpoint "
                    "will be ignored."
                )
            if self.initial_load_model_only and not self.initial_load_path:
                logger.warning(
                    "initial_load_model_only=True has no effect without "
                    "an initial_load_path."
                )
