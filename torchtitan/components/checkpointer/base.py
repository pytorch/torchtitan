# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import queue
import re
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

import torch
import torch.distributed as dist
import torch.nn as nn
import tyro
from torch.distributed.checkpoint.stateful import Stateful
from torch.distributed.tensor import DTensor

from torchtitan.config import Configurable, Function
from torchtitan.tools import filesystem
from torchtitan.tools.logging import logger

MODEL = "model"
OPTIMIZER = "optimizer"
LR_SCHEDULER = "lr_scheduler"
DATALOADER = "dataloader"
TRAIN_STATE = "train_state"


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

    For ``DTensor`` the local shard is compared via ``_local_tensor`` rather
    than ``to_local()``, which is autograd-aware. The dispatcher-level alias
    check also supports wrapper subclasses without directly accessible storage.
    """
    if isinstance(a, DTensor):
        a = a._local_tensor
    if isinstance(b, DTensor):
        b = b._local_tensor
    # pyrefly: ignore [missing-attribute]
    return torch._C._is_alias_of(a, b)


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
    load_only: bool
    interval: int
    enable_first_step_checkpoint: bool
    staging_future: Future | None
    save_future: Future | None
    folder: str
    keep_latest_k: int
    purge_exempt: Callable[[int], bool] | None = None
    purge_thread: threading.Thread | None
    purge_queue: queue.Queue[str | None]
    _storage: CheckpointStorage

    _STEP_DIR_PATTERN = r"step-(0|[1-9]\d*)"

    # A disabled manager returns early from ``__init__`` without setting up any
    # state, so none of its attributes exist. The public entry points below own
    # that check once, on behalf of every implementation, and dispatch to the
    # ``_``-prefixed hooks only when enabled. Subclasses override the hooks, not
    # these methods: overriding a public method here would silently bypass the
    # guard and run against uninitialized state.

    def load(self, step: int = -1) -> bool:
        """Restore state from ``step``, or the latest checkpoint when ``-1``."""
        if not self.enable:
            return False
        return self._load(step)

    def save(self, curr_step: int, last_step: bool = False) -> bool:
        """Persist state for ``curr_step``."""
        if not self.enable:
            return False
        return self._save(curr_step, last_step)

    def maybe_wait_for_staging(self) -> None:
        """Block until asynchronous staging for the last save completes."""
        if not self.enable:
            return
        self._maybe_wait_for_staging()

    def close(self) -> None:
        """Release background threads and other resources."""
        # getattr rather than a plain attribute read: ``__del__`` calls close(),
        # and it can run on a partially constructed object whose ``__init__``
        # raised before assigning ``enable``.
        if not getattr(self, "enable", False):
            return
        try:
            self.maybe_wait_for_staging()
            self.maybe_wait_for_saving()
        finally:
            self._close()

    def maybe_wait_for_saving(self) -> None:
        """Block until the last asynchronous save completes.

        A manager with no asynchronous save in flight leaves ``save_future`` at
        ``None`` and never reaches ``_wait_for_saving``.
        """
        if not self.enable or getattr(self, "save_future", None) is None:
            return
        self._wait_for_saving()

    @abstractmethod
    def _wait_for_saving(self) -> None:
        """Await ``save_future`` and clear it. Only called when it is set."""

    # Policies shared by every manager. These depend only on config fields that
    # BaseCheckpointManager.Config declares, not on how a backend reads or
    # writes bytes, so they live here rather than once per backend.

    def _should_save(self, curr_step: int, last_step: bool = False) -> bool:
        """Whether ``curr_step`` is a checkpointing step."""
        if not self.enable or self.load_only:
            return False
        if curr_step == 1 and self.enable_first_step_checkpoint:
            return True
        return last_step or curr_step % self.interval == 0

    def _create_checkpoint_id(self, step: int, folder: str = "") -> str:
        """Standardized checkpoint path, e.g. ``checkpoints/step-100``."""
        folder = folder or self.folder
        return filesystem.join(folder, f"step-{step}")

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

    def _is_purge_exempt(self, step: int) -> bool:
        """Whether the configured exemption protects ``step`` from deletion."""
        return self.purge_exempt is not None and self.purge_exempt(step)

    def _parse_step(self, dirname: str) -> int | None:
        """Parse a canonical ``step-N`` checkpoint directory name."""
        match = re.fullmatch(self._STEP_DIR_PATTERN, dirname)
        return None if match is None else int(match.group(1))

    @abstractmethod
    def _is_valid_checkpoint(self, checkpoint_dir: str) -> bool:
        """Whether ``checkpoint_dir`` holds a checkpoint this manager can load.

        A directory whose save was interrupted exists but has no metadata, so
        resuming from it would fail; this is what keeps it out of
        ``_find_load_step``.
        """

    def _find_load_step(self, folder: str = "") -> int:
        """The highest step in ``folder`` that can actually be loaded.

        Args:
            folder: Directory to scan. Defaults to ``self.folder``.

        Returns:
            The step number, or -1 when the folder holds no loadable checkpoint.

        Note:
            This is not remote friendly: it issues one listdir plus a metadata
            probe per step folder, each a network round trip on remote (fsspec)
            storage instead of a single batched listing. Acceptable for now
            since it only runs once at load time.
        """
        folder = folder or self.folder
        if not self._storage.isdir(folder):
            return -1

        valid_steps = []
        for dirname in self._storage.listdir(folder):
            step = self._parse_step(dirname)
            if step is None:
                continue
            if self._is_valid_checkpoint(filesystem.join(folder, dirname)):
                valid_steps.append(step)
        return max(valid_steps) if valid_steps else -1

    def _purge_stale_checkpoints(
        self,
        *,
        saving_step: int,
        staging_dir_prefix: str | None = None,
    ) -> None:
        """Delete abandoned entries and reserve one retained slot for this save."""
        if self._should_purge():
            saving_dirnames = {f"step-{saving_step}"}
            if staging_dir_prefix:
                saving_dirnames.add(f"{staging_dir_prefix}step-{saving_step}")

            staging_pattern = (
                re.compile(rf"{re.escape(staging_dir_prefix)}step-(0|[1-9]\d*)")
                if staging_dir_prefix
                else None
            )
            checkpoints: list[tuple[int, str]] = []
            abandoned: list[str] = []

            for dirname in self._storage.listdir(self.folder):
                if dirname in saving_dirnames:
                    continue

                checkpoint_dir = filesystem.join(self.folder, dirname)
                # torch_checkpointing uses this pattern for staging directories.
                if staging_pattern and staging_pattern.fullmatch(dirname):
                    abandoned.append(checkpoint_dir)
                    continue

                step = self._parse_step(dirname)
                if step is None:
                    continue
                if self._is_valid_checkpoint(checkpoint_dir):
                    checkpoints.append((step, checkpoint_dir))
                else:
                    abandoned.append(checkpoint_dir)

            checkpoints.sort()
            num_to_keep = self.keep_latest_k - 1
            num_to_purge = max(0, len(checkpoints) - num_to_keep)
            for step, checkpoint_dir in checkpoints[:num_to_purge]:
                if self._is_purge_exempt(step):
                    logger.info(
                        "Checkpointer is preserving checkpoint %s outside "
                        "keep_latest_k.",
                        checkpoint_dir,
                    )
                    continue
                assert self.purge_thread is not None
                self.purge_queue.put(checkpoint_dir)

            for checkpoint_dir in abandoned:
                assert self.purge_thread is not None
                self.purge_queue.put(checkpoint_dir)

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

        purge_exempt: Annotated[Function.Config | None, tyro.conf.Suppress] = None
        """Optional predicate that exempts checkpoint steps from purging."""

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
