# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Publish full emergency DCP candidates with explicitly bound replica cursors."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException

from torchtitan.components.checkpointer import (
    CheckpointManager,
    DATALOADER,
    EMA,
    LR_SCHEDULER,
    MODEL,
    OPTIMIZER,
    TRAIN_STATE,
)
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from .control import atomic_json, LastWordsExitError, read_json, sync_checkpoint_tree
from .rng import RNG_SCHEMA
from .validation import (
    checkpoint_state,
    checkpoint_torchft_state,
    contained,
    recovery_preflight,
    validate_rng_cursor,
    validate_storage_ranges,
)

logger = logging.getLogger(__name__)


class LastWordsCheckpointManager(TorchFTCheckpointManager):
    CURSOR_INDEX_LIMIT = 64
    elastic_enabled = False

    @dataclass(kw_only=True, slots=True)
    class Config(TorchFTCheckpointManager.Config):
        pass

    def __init__(self, *args, **kwargs):
        self.last_words = None
        self.last_words_signature = ""
        self.last_words_state = None
        self.last_words_cursor_current = False
        self.last_words_on_stop = None
        self._last_words_pending_cursor = None
        self._last_words_pending_binding = None
        self._last_words_control_pg = None
        self._last_words_cursor_anchors = {}
        self._last_words_upload_error = None
        self.elastic_enabled = False
        self.rng_state = None
        super().__init__(*args, **kwargs)

    def _rng_metadata(self):
        if self.rng_state is None:
            return {}
        return dict(rng_schema=RNG_SCHEMA, rng_shards=self.last_words.world_size)

    def _cursor_record(self, state):
        return dict(
            state=dict(state),
            signature=self.last_words_signature,
            **self._rng_metadata(),
        )

    def _restore_rng(self):
        if self.rng_state is not None:
            self.rng_state.restore()

    def enable_elastic_state(self):
        if self.ft_manager is None:
            raise ValueError("Elastic checkpointing requires a TorchFT manager")
        self.elastic_enabled = True
        # The native metadata is also required in ordinary periodic checkpoints.
        self.states["torchft"] = self.ft_manager

    def _save(self, curr_step, last_step=False):
        if self.last_words is None:
            return super()._save(curr_step, last_step)
        writer = not self.enable_ft_dataloader_checkpoints or (
            self.ft_manager and self.ft_manager.participating_rank() == 0
        )
        if self.last_words is not None:
            if self.last_words_stop_requested():
                assert self.last_words_on_stop is not None
                return self.last_words_on_stop()
            if (
                self.last_words_state is None
                or self.last_words_state["step"] != curr_step
            ):
                return False
            # Trainer.save is outside train_step's exception handler. Drain the
            # previous cursor on every shard before _ft_save can start another
            # upload or one shard can enter a periodic full-save collective.
            try:
                uploaded = self._drain_last_words_upload()
            except (Exception, CheckpointException) as error:
                self.last_words.request(
                    f"Unsafe background checkpoint failure: {error}"
                )
                self.last_words.publish("ineligible")
                raise
            if not uploaded:
                self.last_words.request("Completed cursor storage upload failed")
                assert self.last_words_on_stop is not None
                return self.last_words_on_stop()
            if self.enable_ft_dataloader_checkpoints:
                self._ft_save(curr_step)
                if writer and self._should_save(curr_step, last_step):
                    # Keep non-periodic cursors asynchronous. A periodic writer
                    # must coordinate a newly failed cursor before its full save.
                    try:
                        uploaded = self._drain_last_words_upload()
                    except (Exception, CheckpointException) as error:
                        self.last_words.request(
                            f"Unsafe background checkpoint failure: {error}"
                        )
                        self.last_words.publish("ineligible")
                        raise
                    if not uploaded:
                        self.last_words.request(
                            "Completed cursor storage upload failed"
                        )
                        assert self.last_words_on_stop is not None
                        return self.last_words_on_stop()
        # Preserve TorchFT's full-checkpoint writer election; its per-replica
        # side channel has already completed and must not be launched twice.
        saved = False
        if writer:
            saved = CheckpointManager._save(self, curr_step, last_step)
        if self.last_words is not None and saved and dist.get_rank() == 0:
            assert self.last_words_state is not None
            path = (
                Path(self._create_checkpoint_id(curr_step)) / "last_words_binding.json"
            )
            binding = {
                **self._rng_metadata(),
                "signature": self.last_words_signature,
                "run_id": self.last_words.config.run_id,
                "state": dict(self.last_words_state),
                "anchors": {
                    str(replica): str(
                        anchor.resolve().relative_to(Path(self.folder).resolve())
                    )
                    for replica, anchor in self._last_words_cursor_anchors.items()
                },
            }
            if self.elastic_enabled:
                assert self.ft_manager is not None
                binding["torchft"] = dict(self.ft_manager.state_dict())
            if self.save_future is None:
                atomic_json(path, binding, durable=True)
            else:
                self._last_words_pending_binding = (path, binding)
        return saved

    def last_words_stop_requested(self):
        assert self.last_words is not None
        # All shards take the same branch before entering FSDP/EP or cursor DCP.
        stop = torch.tensor(int(self.last_words.requested), dtype=torch.int32)
        assert self._last_words_control_pg is not None
        dist.all_reduce(stop, op=dist.ReduceOp.MAX, group=self._last_words_control_pg)
        return bool(stop.item())

    def _ft_save(self, step):
        if self.last_words is not None:
            self.maybe_wait_for_saving()
            marker = (
                Path(self._create_checkpoint_id(step, folder=self._ft_folder()))
                / "last_words_cursor.json"
            )
            if marker.is_file():
                if read_json(marker)["state"] != self.last_words_state:
                    raise LastWordsExitError(
                        "Refusing to overwrite an immutable cursor version"
                    )
                return
        super()._ft_save(step)
        if self.last_words is None:
            return
        assert self.last_words_state is not None
        state = dict(self.last_words_state)
        path = (
            Path(self._create_checkpoint_id(step, folder=self._ft_folder()))
            / "last_words_cursor.json"
        )
        assert self.save_future is not None
        self._last_words_pending_cursor = (path, state)
        upload = self.save_future

        def persist_cursor():
            upload.result()
            if dist.get_rank() == 0:
                sync_checkpoint_tree(path.parent, root=Path(self.folder))
                atomic_json(path, self._cursor_record(state), durable=True)
                self._update_cursor_index(path, state)

        self.save_future = self.last_words._durability_executor.submit(persist_cursor)
        self._last_words_cursor_background = True

    def _update_cursor_index(self, marker, state):
        path = marker.parent.parent / "last_words_cursor_index.json"
        try:
            value: dict[str, Any] = (
                read_json(path)
                if path.is_file()
                else {"signature": self.last_words_signature, "entries": {}}
            )
            if value["signature"] != self.last_words_signature or not isinstance(
                value["entries"], dict
            ):
                raise ValueError("Invalid cursor index header")
            if any(
                Path(name).name != name
                or not name.startswith("step-")
                or not isinstance(entry, dict)
                or any(
                    type(entry.get(key)) is not int or entry[key] < 0
                    for key in ("step", "last_words_num_updates")
                )
                for name, entry in value["entries"].items()
            ):
                raise ValueError("Invalid cursor index entry")
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("Rebuilding unreadable cursor index at %s", path)
            value = {"signature": self.last_words_signature, "entries": {}}
            for previous in path.parent.glob("step-*/last_words_cursor.json"):
                try:
                    record = read_json(previous)
                    if record["signature"] == self.last_words_signature:
                        value["entries"][previous.parent.name] = record["state"]
                except (OSError, ValueError, KeyError):
                    continue
        value["entries"][marker.parent.name] = dict(state)
        # This is a bounded lookup hint; all published cursor versions remain available to the scan fallback.
        newest = sorted(
            value["entries"],
            key=lambda name: (
                value["entries"][name]["last_words_num_updates"],
                value["entries"][name]["step"],
            ),
            reverse=True,
        )[: self.CURSOR_INDEX_LIMIT]
        value["entries"] = {name: value["entries"][name] for name in newest}
        atomic_json(path, value, durable=True)

    def _wait_for_saving(self):
        future = self.save_future
        try:
            super()._wait_for_saving()
            if self._last_words_pending_cursor is not None:
                path, state = self._last_words_pending_cursor
                if dist.get_rank() == 0 and not getattr(
                    self, "_last_words_cursor_background", False
                ):
                    sync_checkpoint_tree(path.parent, root=Path(self.folder))
                    atomic_json(path, self._cursor_record(state), durable=True)
                self._last_words_pending_cursor = None
                self._last_words_cursor_background = False
            if self._last_words_pending_binding is not None:
                path, binding = self._last_words_pending_binding
                atomic_json(path, binding, durable=True)
                self._last_words_pending_binding = None
        except (Exception, CheckpointException) as error:
            if self.last_words is not None:
                # Only an ended cursor upload with storage errors can fall back.
                # A failed full save or communication/staging error is unsafe.
                failures = (
                    error.failures.values()
                    if isinstance(error, CheckpointException)
                    else [(error, None)]
                )
                safe = (
                    self._last_words_pending_cursor is not None
                    and self._last_words_pending_binding is None
                    and future is not None
                    and future.done()
                    and self.staging_future is None
                    and all(isinstance(failure[0], OSError) for failure in failures)
                )
                self._last_words_upload_error = (safe, str(error))
                self._last_words_pending_cursor = None
                self._last_words_pending_binding = None
                if future is not None and future.done():
                    self.save_future = None
            raise

    def _ft_folder(self):
        folder = super()._ft_folder()
        if self.last_words is None:
            return folder
        branch = Path(folder) / self.last_words.config.run_id
        if self.elastic_enabled:
            branch = branch / f"generation-{self.last_words.generation}"
        return str(branch)

    def _cursor_candidates(self, replica, state, *, run_id=None, anchors=None):
        assert self.last_words is not None
        source_root = Path(self.folder) / f"ft-replicat-{replica}"
        run_id = self.last_words.config.run_id if run_id is None else run_id
        anchors = self._last_words_cursor_anchors if anchors is None else anchors
        available = []
        branch = source_root / run_id
        index = branch / "last_words_cursor_index.json"
        markers = []
        if index.is_file():
            try:
                value = read_json(index)
                if value["signature"] != self.last_words_signature:
                    raise ValueError("Cursor index has another signature")
                eligible = [
                    (entry["last_words_num_updates"], entry["step"], name)
                    for name, entry in value["entries"].items()
                    if entry["last_words_num_updates"]
                    <= state["last_words_num_updates"]
                    and entry["step"] <= state["step"]
                ]
                for _, _, name in sorted(eligible, reverse=True):
                    if Path(name).name != name or not name.startswith("step-"):
                        raise ValueError("Invalid indexed cursor path")
                    candidate = branch / name
                    validate_storage_ranges(candidate)
                    marker = candidate / "last_words_cursor.json"
                    cursor = read_json(marker)
                    if (
                        cursor["signature"] != self.last_words_signature
                        or cursor["state"] != value["entries"][name]
                    ):
                        raise ValueError("Cursor index disagrees with published marker")
                    markers = [marker]
                    break
            except (Exception, CheckpointException) as error:
                logger.warning(
                    "Cursor index unavailable; scanning published cursors: %s", error
                )
        if not markers:
            markers = list(branch.glob("step-*/last_words_cursor.json"))
        if self.elastic_enabled:
            markers.extend(branch.glob("generation-*/step-*/last_words_cursor.json"))
            markers.extend(branch.glob("initial/step-*/last_words_cursor.json"))
        if replica in anchors:
            markers.append(anchors[replica] / "last_words_cursor.json")
        for marker in markers:
            if not marker.is_file():
                continue
            try:
                contained(Path(self.folder), marker.parent)
                validate_storage_ranges(marker.parent)
                cursor = read_json(marker)
                if self.rng_state is not None:
                    validate_rng_cursor(
                        marker.parent, cursor, self.last_words.world_size
                    )
            except (Exception, CheckpointException) as error:
                logger.warning("Ignoring incomplete cursor %s: %s", marker, error)
                continue
            if (
                cursor["signature"] == self.last_words_signature
                and cursor["state"]["last_words_num_updates"]
                <= state["last_words_num_updates"]
                and cursor["state"]["step"] <= state["step"]
                and (marker.parent / ".metadata").is_file()
            ):
                available.append(
                    (
                        cursor["state"]["last_words_num_updates"],
                        cursor["state"]["step"],
                        marker.parent,
                        cursor,
                    )
                )
        return available

    def initialize_elastic_cursors(self, state, *, make_cursor, initial_replicas):
        """Materialize all logical slots before their data can be consumed."""
        control = self.last_words
        assert control is not None
        self.maybe_wait_for_staging()
        self._drain_last_words_upload()
        for replica in range(control.num_replicas):
            if self.rng_state is not None:
                owner = (
                    replica if replica in initial_replicas else min(initial_replicas)
                )
                if control.replica_id != owner:
                    continue
            elif control.replica_id != min(initial_replicas):
                continue
            if self._cursor_candidates(replica, state):
                continue
            if state["step"] != 0 or state["last_words_num_updates"] != 0:
                raise LastWordsExitError(
                    "Cannot invent a missing slot cursor after training has begun"
                )
            target = (
                Path(self.folder)
                / f"ft-replicat-{replica}"
                / control.config.run_id
                / "initial/step-0"
            )
            cursor = (
                self.ft_states
                if replica == control.replica_id
                else make_cursor(replica)
            )
            try:
                dcp.save(cursor, checkpoint_id=target, process_group=self.pg)
            finally:
                if cursor is not self.ft_states:
                    cursor[DATALOADER].close()
            if control.rank == 0:
                sync_checkpoint_tree(target, root=Path(self.folder))
                atomic_json(
                    target / "last_words_cursor.json",
                    self._cursor_record(state),
                    durable=True,
                )
            if dist.is_initialized():
                dist.barrier(group=self._last_words_control_pg)
        deadline = time.monotonic() + control.config.timeout_seconds
        while True:
            if all(
                self._cursor_candidates(replica, state)
                for replica in range(control.num_replicas)
            ):
                return
            if time.monotonic() >= deadline:
                raise LastWordsExitError(
                    "Timed out preparing complete logical slot cursors"
                )
            _ = control.requested
            time.sleep(0.1)

    def restore_elastic_cursor(self, state):
        control = self.last_words
        assert control is not None
        self.maybe_wait_for_staging()
        self._drain_last_words_upload()
        available = self._cursor_candidates(control.replica_id, state)
        if not available:
            raise LastWordsExitError(
                "No compatible safe cursor for this logical replica"
            )
        _, _, source, _ = max(
            available, key=lambda entry: (entry[0], entry[1], str(entry[2]))
        )
        dcp.load(self.ft_states, checkpoint_id=source, process_group=self.pg)
        self._restore_rng()
        self.last_words_cursor_current = True
        return source

    @staticmethod
    def _checkpoint_state(path):
        # Inspect only scalar progress, without mutating live training objects
        # or using a collective before every rank has selected its candidate.
        return checkpoint_state(path)

    def _periodic_recovery(self, step):
        assert self.last_words is not None
        path = Path(self._create_checkpoint_id(step))
        if not (path / ".metadata").is_file():
            raise LastWordsExitError("Periodic checkpoint metadata is missing")
        validate_storage_ranges(path)
        if step == 0:
            return {}, {}
        binding_path = path / "last_words_binding.json"
        if not binding_path.is_file():
            raise LastWordsExitError(
                "Periodic checkpoint has no bound cursor recovery chain"
            )
        binding = read_json(binding_path)
        if binding["signature"] != self.last_words_signature:
            raise LastWordsExitError("Periodic checkpoint configuration differs")
        if self.rng_state is not None and binding.get("rng_schema") != RNG_SCHEMA:
            raise LastWordsExitError(
                "Periodic checkpoint has no bound RNG recovery chain"
            )
        state = self._checkpoint_state(path)
        if state["step"] != step or (
            "state" in binding
            and any(binding["state"][key] != value for key, value in state.items())
        ):
            raise LastWordsExitError("Periodic checkpoint binding progress differs")
        if self.elastic_enabled and checkpoint_torchft_state(path) != binding.get(
            "torchft"
        ):
            raise LastWordsExitError(
                "Periodic checkpoint has no matching TorchFT progress"
            )
        anchors = {
            int(replica): contained(Path(self.folder), Path(self.folder) / anchor)
            for replica, anchor in binding["anchors"].items()
        }
        selected = {}
        for replica in range(self.last_words.num_replicas):
            available = self._cursor_candidates(
                replica, state, run_id=binding["run_id"], anchors=anchors
            )
            if not available:
                raise LastWordsExitError(
                    "Periodic model checkpoint has no compatible safe replica cursor"
                )
            _, _, cursor_path, _ = max(
                available, key=lambda entry: (entry[0], entry[1], str(entry[2]))
            )
            selected[replica] = cursor_path
        return state, selected

    def _periodic_candidates(self, step):
        paths = (
            [Path(self._create_checkpoint_id(step))]
            if step != -1
            else Path(self.folder).glob("step-*")
        )
        available = []
        for path in paths:
            suffix = path.name.removeprefix("step-")
            if not suffix.isdecimal():
                continue
            candidate_step = int(suffix)
            try:
                state, anchors = self._periodic_recovery(candidate_step)
            except (Exception, CheckpointException) as error:
                if step != -1:
                    raise LastWordsExitError(
                        f"Requested periodic checkpoint is not recoverable: {error}"
                    ) from error
                logger.warning(
                    "Ignoring unpublished or incompatible periodic checkpoint %s: %s",
                    path,
                    error,
                )
                continue
            available.append((candidate_step, state, anchors))
        return available

    def _import_cursor_anchors(self, anchors):
        assert self.last_words is not None
        imported = {}
        error = ""
        for replica, source in anchors.items():
            if source.resolve().is_relative_to(Path(self.folder).resolve()):
                imported[replica] = source
                continue
            target = (
                Path(self.folder)
                / f"ft-replicat-{self.last_words.replica_id}"
                / self.last_words.config.run_id
                / "resume-anchors"
                / f"replica-{replica}"
            )
            imported[replica] = target
            if self.last_words.rank == 0:
                try:
                    shutil.copytree(source, target)
                    sync_checkpoint_tree(target, root=Path(self.folder))
                except Exception as caught:
                    error = str(caught)
        if dist.is_initialized():
            failures: list[str | None] = [None] * dist.get_world_size(self.pg)
            dist.all_gather_object(failures, error, group=self.pg)
            error = next((failure for failure in failures if failure), "")
        if error:
            raise LastWordsExitError(f"Failed importing resume cursors: {error}")
        return imported

    def _load_periodic(self, step):
        assert self.last_words is not None
        candidates = self._periodic_candidates(step)
        if candidates:
            step, _, anchors = max(candidates, key=lambda entry: entry[0])
        elif self._find_load_step() != -1:
            raise LastWordsExitError(
                "No published periodic checkpoint has a complete recovery chain"
            )
        else:
            anchors = {}
        imported = self._import_cursor_anchors(anchors) if anchors else {}
        # Existing step-0 initial states predate the native metadata extension.
        metadata = None
        if self.elastic_enabled and step == 0:
            metadata = self.states.pop("torchft", None)
        try:
            self._last_words_loading_periodic = True
            loaded = CheckpointManager.load(self, step)
        finally:
            self._last_words_loading_periodic = False
            if metadata is not None:
                self.states["torchft"] = metadata
        if loaded:
            state = self.states[TRAIN_STATE].state_dict()
            if state["step"] == 0:
                if self.elastic_enabled:
                    assert self.ft_manager is not None
                    self.ft_manager.load_state_dict(dict(step=0, batches_committed=0))
                return loaded
            if not anchors:
                raise LastWordsExitError(
                    "Initial full checkpoint has no bound cursor recovery chain"
                )
            self._last_words_cursor_anchors = imported
            dcp.load(
                self.ft_states,
                checkpoint_id=self._last_words_cursor_anchors[
                    self.last_words.replica_id
                ],
                process_group=self.pg,
            )
            self._restore_rng()
        return loaded

    @torch.no_grad()
    def load(self, step=-1):
        control = self.last_words
        if control is None:
            return super().load(step)
        root = Path(self.folder) / "last_words"
        requested = control.config.resume_path
        paths = (
            [Path(requested)]
            if requested
            else list(root.glob("*/replica-*/manifest.json"))
        )
        if not requested and self.elastic_enabled:
            paths.extend(root.glob("*/events/event-*/replica-*/manifest.json"))
        candidates = []
        for path in paths:
            manifest_path = path / "manifest.json" if path.is_dir() else path
            if not manifest_path.is_file():
                if requested:
                    raise ValueError("Requested last-words candidate is not published")
                continue
            try:
                manifest = read_json(manifest_path)
                if manifest["signature"] != self.last_words_signature:
                    raise ValueError(
                        "Last-words checkpoint topology or data configuration differs"
                    )
                if self.elastic_enabled and "torchft" not in manifest:
                    raise ValueError(
                        "Elastic recovery requires checkpointed TorchFT progress"
                    )
                if (
                    self.rng_state is not None
                    and manifest.get("rng_schema") != RNG_SCHEMA
                ):
                    raise ValueError(
                        "Last-words checkpoint has no bound RNG recovery chain"
                    )
                if step != -1 and manifest["state"]["step"] != step:
                    continue
                base = manifest_path.parent
                recovery_preflight(base, manifest, control.num_replicas)
            except (Exception, CheckpointException) as error:
                if requested:
                    raise ValueError(
                        f"Requested last-words candidate is not recoverable: {error}"
                    ) from error
                logger.warning(
                    "Ignoring incomplete last-words candidate %s: %s",
                    manifest_path,
                    error,
                )
                continue
            candidates.append(
                (
                    manifest["state"]["last_words_num_updates"],
                    -manifest["replica_id"],
                    manifest_path,
                    manifest,
                )
            )
        if not candidates:
            if requested:
                raise ValueError("No matching last-words candidate")
            return self._load_periodic(step)
        _, _, manifest_path, manifest = max(
            candidates, key=lambda entry: (entry[0], entry[1], str(entry[2]))
        )
        # A later periodic checkpoint takes precedence over an older emergency save.
        if not requested and step == -1:
            periodic = self._periodic_candidates(-1)
            if (
                periodic
                and max(periodic, key=lambda entry: entry[0])[0]
                > manifest["state"]["step"]
            ):
                return self._load_periodic(-1)
        base = manifest_path.parent
        anchors = {
            int(replica): base / cursor["path"]
            for replica, cursor in manifest["cursors"].items()
        }
        imported = self._import_cursor_anchors(anchors)
        cursor = manifest["cursors"][str(control.replica_id)]
        states = self._full_states()
        dcp.load(states, checkpoint_id=base / "model", process_group=self.pg)
        self.states[MODEL].load_state_dict(states)
        if self.elastic_enabled:
            assert self.ft_manager is not None
            self.ft_manager.load_state_dict(states["torchft"])
        dcp.load(
            self.ft_states, checkpoint_id=base / cursor["path"], process_group=self.pg
        )
        self._restore_rng()
        self._last_words_cursor_anchors = imported
        logger.warning(
            "Loaded last words model=%s cursor replica=%d state=%s",
            manifest["state"],
            control.replica_id,
            cursor["state"],
        )
        return True

    def _load_checkpoint(self, states, checkpoint_id, *, from_hf, from_quantized):
        if getattr(self, "_last_words_loading_periodic", False):
            # Exact replica cursors are restored from the validated binding, not
            # from TorchFT's independently selected latest per-replica cursor.
            return CheckpointManager._load_checkpoint(
                self,
                states,
                checkpoint_id,
                from_hf=from_hf,
                from_quantized=from_quantized,
            )
        return super()._load_checkpoint(
            states, checkpoint_id, from_hf=from_hf, from_quantized=from_quantized
        )

    def _full_states(self):
        self.states[OPTIMIZER]._refresh_cached_state_dict()
        states = {
            key: self.states[key]
            for key in (MODEL, OPTIMIZER, LR_SCHEDULER, TRAIN_STATE)
        }
        if EMA in self.states:
            states[EMA] = self.states[EMA]
        if self.elastic_enabled:
            assert self.ft_manager is not None
            states["torchft"] = dict(self.ft_manager.state_dict())
        return self._flattened_model_states_sd(states)

    def _drain_last_words_upload(self):
        try:
            self.maybe_wait_for_saving()
        except (Exception, CheckpointException):
            if self._last_words_upload_error is None:
                raise
        error = self._last_words_upload_error
        failed = torch.tensor(
            0 if error is None else (1 if error[0] else 2), dtype=torch.int32
        )
        if dist.is_initialized():
            # All uploader futures have ended before probing their group.
            dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=self.pg)
        if failed.item() == 2:
            raise LastWordsExitError(
                "Background checkpoint failed outside a completed cursor storage upload"
            )
        if failed.item() == 1:
            logger.warning(
                "Using previously published replica cursors after failed upload: %s",
                error,
            )
        return failed.item() == 0

    def save_last_words(self, state):
        control = self.last_words
        assert control is not None
        control.wait_frozen(state)
        self.last_words_state = dict(state)
        # Complete staging/upload before freezing the cursor versions.
        self.maybe_wait_for_staging()
        if self._drain_last_words_upload() and self.last_words_cursor_current:
            self._ft_save(state["step"])
            self._drain_last_words_upload()
        # Advertise only replicas with every required, non-future cursor.
        cursor_error = ""
        if control.rank == 0:
            try:
                for replica in range(control.num_replicas):
                    if not self._cursor_candidates(replica, state):
                        raise LastWordsExitError(
                            f"A safe cursor is missing for replica {replica}"
                        )
            except Exception as error:
                cursor_error = str(error)
        if dist.is_initialized():
            preparation = [cursor_error]
            dist.broadcast_object_list(preparation, group=self.pg, group_src=0)
            cursor_error = preparation[0]
        if cursor_error:
            control.publish("ineligible")
            raise LastWordsExitError(
                f"Emergency cursor preparation failed: {cursor_error}"
            )
        write, selected = control.await_save_authorization(
            state, self.last_words_signature
        )
        if not write:
            return selected
        if dist.is_initialized():
            # Every shard has independently checked the same immutable grant.
            dist.barrier(group=self._last_words_control_pg)
        target = (
            control.checkpoint_target(control.replica_id)
            if self.elastic_enabled
            else Path(self.folder)
            / "last_words"
            / control.config.run_id
            / f"replica-{control.replica_id}"
        )
        if not self.elastic_enabled or control.rank == 0:
            target.mkdir(parents=True, exist_ok=not self.elastic_enabled)
        if self.elastic_enabled and dist.is_initialized():
            dist.barrier(group=self._last_words_control_pg)
        control.write_authorization_binding(target)
        cursors = {}
        cursor_error = ""
        if control.rank == 0:
            try:
                for replica in range(control.num_replicas):
                    available = self._cursor_candidates(replica, state)
                    if not available:
                        raise LastWordsExitError(
                            f"A safe cursor is missing for replica {replica}"
                        )
                    _, _, source, cursor = max(
                        available, key=lambda entry: (entry[0], entry[1], str(entry[2]))
                    )
                    relative = f"cursors/replica-{replica}"
                    shutil.copytree(source, target / relative)
                    cursors[str(replica)] = {"path": relative, "state": cursor["state"]}
            except Exception as error:
                cursor_error = str(error)
        if dist.is_initialized():
            preparation = [cursor_error]
            dist.broadcast_object_list(preparation, group=self.pg, group_src=0)
            cursor_error = preparation[0]
        if cursor_error:
            raise LastWordsExitError(
                f"Emergency cursor preparation failed: {cursor_error}"
            )
        # self.pg is replica-local Gloo, never a TorchFT cross-replica group.
        states = self._full_states()
        dcp.save(states, checkpoint_id=target / "model", process_group=self.pg)
        if control.rank == 0:
            sync_checkpoint_tree(target, root=Path(self.folder))
            if not cursor_error:
                atomic_json(
                    target / "manifest.json",
                    {
                        **self._rng_metadata(),
                        "format_version": 1,
                        "run_id": control.config.run_id,
                        "replica_id": control.replica_id,
                        "signature": self.last_words_signature,
                        "state": state,
                        "cursors": cursors,
                        "fault_reports": [
                            read_json(path)
                            for path in control.root.glob("request-*.json")
                        ],
                        **(
                            dict(
                                torchft=states["torchft"],
                                **{
                                    key: control._coordinator_grant[key]
                                    for key in ("event_id", "membership_epoch")
                                },
                            )
                            if self.elastic_enabled
                            else {}
                        ),
                    },
                    durable=True,
                )
            atomic_json(
                target / "publication.json",
                {"success": True, "error": ""},
                durable=True,
            )
        control.wait_publication(target)
        control.wait_save_result()
        logger.warning("Last-words DCP saved at %s", target)
        return target
