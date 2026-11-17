# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import logging
import os
import queue
import tempfile
import unittest
import uuid
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest import mock

import fsspec
import torch
import torch.nn as nn

import torchtitan.components.checkpointer.torch_checkpointing as manager_module
from safetensors.torch import save_file
from torch.distributed.checkpoint.stateful import Stateful
from torch_checkpointing.barriers import DefaultStoreBarrierConfig
from torch_checkpointing.checkpoint_layout import SafetensorsSerialization
from torch_checkpointing.checkpoint_manager import (
    CheckpointManager as BackendCheckpointManager,
)
from torch_checkpointing.checkpoint_writer import CallbackArgs, CheckpointWriterConfig
from torch_checkpointing.config import (
    AsyncCheckpointSaverConfig,
    SyncCheckpointSaverConfig,
)
from torch_checkpointing.default_resharder import DefaultResharder
from torch_checkpointing.hf.metadata import (
    HuggingFaceSafetensorsDistributedMetadataFormat,
)
from torch_checkpointing.logging_utils import checkpoint_logging_context
from torch_checkpointing.schema import ItemSpec
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torch_checkpointing.storage.fsspec_storage import FsspecStorageConfig
from torchtitan.components.checkpointer import (
    CheckpointManager,
    CheckpointStorage,
    EMA,
    MODEL,
    ModelWrapper,
    OPTIMIZER,
)
from torchtitan.components.checkpointer.torch_checkpointing import (
    _async_save_config,
    _BackendCheckpointStorage,
    _default_backend_config,
    TorchCheckpointingManager,
)
from torchtitan.config import CallableFunction
from torchtitan.models.base.module import Module
from torchtitan.models.common.nn_modules import RMSNorm


class _BackendManager:
    def __init__(self) -> None:
        self.closed = False
        self.lock_calls = 0
        self.load_calls = []
        self.load_result: Any = None
        self.prewarm_calls = []
        self.save_calls = []
        self.save_result = Future()

    def save(self, checkpoint_id, checkpoint):
        # The real backend wraps ids in Path(), which mangles "gs://".
        assert "://" not in checkpoint_id, checkpoint_id
        self.save_calls.append((checkpoint_id, checkpoint))
        return self.save_result

    def prewarm_staging(self, checkpoint) -> None:
        self.prewarm_calls.append(checkpoint)

    def lock(self):
        self.lock_calls += 1
        return nullcontext()

    def load(self, checkpoint_id, into=None, **kwargs):
        assert "://" not in checkpoint_id, checkpoint_id
        self.load_calls.append((checkpoint_id, into, kwargs))
        if self.load_result is None:
            return dict(into)

        result = dict(into)
        for key, value in self.load_result.items():
            target = into[key]
            if isinstance(target, dict) and isinstance(value, dict):
                target.clear()
                target.update(value)
                result[key] = target
            else:
                result[key] = value
        return result

    def close(self) -> None:
        self.closed = True


class _Stateful(Stateful):
    def __init__(self, value) -> None:
        self.value = value

    def state_dict(self):
        return {"value": self.value}

    def load_state_dict(self, state_dict) -> None:
        self.value = state_dict["value"]


class _StateDictAdapter:
    def __init__(self, hf_assets_path: str | None = None) -> None:
        self.fqn_to_index_mapping = {"hf_weight": 1}
        self.hf_assets_path = hf_assets_path
        self.from_hf_calls = []
        self.to_hf_calls = []
        self.to_hf_results = []

    def to_hf(self, state_dict):
        self.to_hf_calls.append(state_dict)
        result = {"hf_weight": state_dict["weight"]}
        self.to_hf_results.append(result)
        return result

    def from_hf(self, state_dict):
        self.from_hf_calls.append(state_dict)
        return {"weight": state_dict["hf_weight"]}


class TorchCheckpointingManagerTest(unittest.TestCase):
    def _build_manager(
        self,
        config: TorchCheckpointingManager.Config,
        *,
        storage_config=None,
        base_folder: str = "/tmp",
        model_parts=None,
        optimizers=None,
        ema=None,
        extra_states=None,
        sd_adapter=None,
    ) -> tuple[TorchCheckpointingManager, _BackendManager]:
        backend_manager = _BackendManager()
        with mock.patch.object(
            BackendCheckpointManager.Config,
            "build",
            return_value=backend_manager,
        ):
            manager = config.build(
                model_parts=model_parts or [nn.Linear(2, 2)],
                optimizers=optimizers or _Stateful("optimizer"),
                lr_schedulers=_Stateful("scheduler"),
                ema=ema,
                extra_states=extra_states or {"train_state": _Stateful("train")},
                sd_adapter=sd_adapter,
                base_folder=base_folder,
                storage_config=storage_config,
            )
        return manager, backend_manager

    def test_config_builds_independent_manager(self) -> None:
        config = TorchCheckpointingManager.Config(
            keep_latest_k=0,
            initial_load_model_only=False,
            purge_exempt=CallableFunction.Config(fn=lambda step: step % 2 == 0),
        )

        manager, backend_manager = self._build_manager(config)

        self.assertIsInstance(manager, TorchCheckpointingManager)
        self.assertNotIsInstance(manager, CheckpointManager)
        self.assertTrue(manager._is_purge_exempt(2))
        self.assertFalse(manager._is_purge_exempt(3))
        manager.close()
        self.assertTrue(backend_manager.closed)

    def test_config_to_dict_is_json_serializable(self) -> None:
        config_dict = TorchCheckpointingManager.Config().to_dict()

        json.dumps(config_dict)
        self.assertNotIn("checkpoint_manager", config_dict)

    def test_del_ignores_manager_whose_construction_failed(self) -> None:
        manager = TorchCheckpointingManager.__new__(TorchCheckpointingManager)
        manager.save_future = None
        manager.purge_thread = None

        manager.__del__()

    def test_default_backend_configuration_owns_schema_and_barrier(self) -> None:
        backend_config = _default_backend_config(_async_save_config())

        self.assertIsInstance(backend_config.save, AsyncCheckpointSaverConfig)
        self.assertTrue(backend_config.save.staging_config.use_pinned_memory)
        self.assertEqual(set(backend_config.items), {MODEL, OPTIMIZER, EMA})
        for spec in backend_config.items.values():
            self.assertTrue(spec.requires_copy)
            self.assertFalse(spec.required)
            self.assertIsInstance(spec.resharder, DefaultResharder)
        self.assertIsNotNone(backend_config.default)
        self.assertFalse(backend_config.default.requires_copy)
        barrier_config = backend_config.save.writer_config.barrier_config
        self.assertIsInstance(barrier_config, DefaultStoreBarrierConfig)

    def test_remote_folder_uses_fsspec_storage_and_scheme_less_save_ids(self) -> None:
        # The backend wraps every path in Path(), which would turn
        # "gs://bucket/x" into "gs:/bucket/x". It is addressed without the
        # scheme, and the protocol travels in the storage config instead.
        config = TorchCheckpointingManager.Config(
            folder="gs://bucket/run/checkpoint/",
            interval=3,
            keep_latest_k=0,
            initial_load_model_only=False,
        )
        storage = mock.Mock()
        with mock.patch.object(
            FsspecStorageConfig, "create_storage", return_value=storage
        ):
            manager, backend_manager = self._build_manager(config)

        self.assertEqual("gs://bucket/run/checkpoint/", manager.folder)
        storage_config = manager._manager_config.storage_config
        self.assertIsInstance(storage_config, FsspecStorageConfig)
        self.assertEqual("gs", storage_config.protocol)

        self.assertTrue(manager.save(curr_step=3))
        self.assertEqual(
            "bucket/run/checkpoint/step-3", backend_manager.save_calls[0][0]
        )
        backend_manager.save_result.set_result(None)

        # The final save goes through a fresh synchronous backend manager.
        sync_manager = _BackendManager()
        sync_manager.save_result = None
        with mock.patch.object(
            BackendCheckpointManager.Config, "build", return_value=sync_manager
        ):
            self.assertTrue(manager.save(curr_step=6, last_step=True))
        self.assertEqual(
            "bucket/run/checkpoint/exports/step-6", sync_manager.save_calls[0][0]
        )

        # Retention deletes through the same storage, by scheme-less path.
        manager._storage.remove("gs://bucket/run/checkpoint/step-3")
        storage.rmdir.assert_called_once_with(Path("bucket/run/checkpoint/step-3"))
        manager.close()

    def test_remote_folder_is_probed_and_loaded_by_scheme_less_path(self) -> None:
        storage = mock.Mock()
        storage.ls.return_value = ["step-7"]
        storage.exists.return_value = True
        storage.isdir.side_effect = lambda path: not str(path).endswith("metadata.pkl")
        config = TorchCheckpointingManager.Config(
            folder="gs://bucket/run/checkpoint",
            keep_latest_k=0,
            initial_load_model_only=False,
            load_only=True,
        )
        with mock.patch.object(
            FsspecStorageConfig, "create_storage", return_value=storage
        ):
            manager, backend_manager = self._build_manager(config)

        self.assertTrue(manager.load())

        storage.ls.assert_called_with(Path("bucket/run/checkpoint"))
        self.assertEqual(
            "bucket/run/checkpoint/step-7", backend_manager.load_calls[0][0]
        )
        manager.close()

    def test_storage_reads_each_path_from_the_storage_its_scheme_names(self) -> None:
        # Dropping the scheme and reading through the folder's storage would
        # turn gs://bucket/hf into bucket/hf on S3.
        folder_storage_config = FsspecStorageConfig(protocol="s3")
        folder_storage, local_storage, gcs_storage = (mock.Mock() for _ in range(3))
        with (
            mock.patch.object(
                FsspecStorageConfig,
                "create_storage",
                autospec=True,
                side_effect=lambda config: {"s3": folder_storage, "gs": gcs_storage}[
                    config.protocol
                ],
            ),
            mock.patch.object(
                LocalFileSystemStorageConfig,
                "create_storage",
                return_value=local_storage,
            ),
        ):
            adapter = _BackendCheckpointStorage(folder_storage_config, protocol="s3")
            adapter.isdir("s3://bucket/run/step-1")
            adapter.isdir("/data/hf")
            adapter.isdir("gs://bucket/hf")

        folder_storage.isdir.assert_called_once_with(Path("bucket/run/step-1"))
        local_storage.isdir.assert_called_once_with(Path("/data/hf"))
        gcs_storage.isdir.assert_called_once_with(Path("bucket/hf"))
        # The backend managers that load these paths get the same storage.
        self.assertIs(
            folder_storage_config, adapter.storage_config_for("s3://bucket/run")
        )
        self.assertEqual(
            LocalFileSystemStorageConfig(), adapter.storage_config_for("/data/hf")
        )
        self.assertEqual(
            FsspecStorageConfig(protocol="gs"),
            adapter.storage_config_for("gs://bucket/hf"),
        )

    def test_remote_folder_round_trip(self) -> None:
        """Save and resume through a URI folder with the real backend."""
        with tempfile.TemporaryDirectory() as base_folder:
            folder = f"file://{base_folder}/checkpoint"

            def build(model: nn.Module, **overrides) -> TorchCheckpointingManager:
                return TorchCheckpointingManager.Config(
                    folder=folder,
                    keep_latest_k=0,
                    initial_load_model_only=False,
                    last_save_model_only=False,
                    **overrides,
                ).build(
                    model_parts=[model],
                    optimizers=_Stateful("optimizer"),
                    lr_schedulers=_Stateful("scheduler"),
                    ema=None,
                    extra_states={"train_state": _Stateful("train")},
                    sd_adapter=None,
                )

            saved = nn.Linear(2, 2)
            manager = build(saved)
            try:
                self.assertTrue(manager.save(curr_step=5, last_step=True))
            finally:
                manager.close()
            self.assertIsInstance(
                manager._manager_config.storage_config, FsspecStorageConfig
            )

            restored = nn.Linear(2, 2)
            loader = build(restored, load_only=True)
            try:
                self.assertTrue(loader.load())
            finally:
                loader.close()
            torch.testing.assert_close(restored.weight, saved.weight)

    def test_caller_supplied_storage_config_wins_for_a_remote_folder(self) -> None:
        storage_config = mock.Mock()
        storage_config.create_storage.return_value = mock.Mock()
        config = TorchCheckpointingManager.Config(
            folder="gs://bucket/run/checkpoint",
            keep_latest_k=0,
        )

        manager, _ = self._build_manager(config, storage_config=storage_config)

        self.assertIs(storage_config, manager._manager_config.storage_config)
        manager.close()

        # A storage the folder's URI cannot be on would silently drop the
        # scheme: gs://bucket/run becomes the local relative path bucket/run.
        for mismatched in (
            LocalFileSystemStorageConfig(),
            FsspecStorageConfig(protocol="s3"),
        ):
            with self.subTest(storage_config=mismatched):
                with self.assertRaisesRegex(ValueError, "storage_config"):
                    self._build_manager(config, storage_config=mismatched)

        # The reverse: a local folder on remote storage would turn
        # /tmp/checkpoint into gs://tmp/checkpoint. fsspec's local filesystem
        # holds it as is.
        local_config = TorchCheckpointingManager.Config(keep_latest_k=0)
        with self.assertRaisesRegex(ValueError, "storage_config"):
            self._build_manager(
                local_config, storage_config=FsspecStorageConfig(protocol="gs")
            )
        manager, _ = self._build_manager(
            local_config, storage_config=FsspecStorageConfig(protocol="file")
        )
        manager.close()

    def _memory_bucket(self) -> str:
        memory_fs = fsspec.filesystem("memory")
        bucket = f"/bucket-{uuid.uuid4().hex}"
        self.addCleanup(
            lambda: memory_fs.rm(bucket, recursive=True)
            if memory_fs.exists(bucket)
            else None
        )
        return bucket

    def test_hf_initial_load_reads_local_assets_with_a_remote_folder(self) -> None:
        """Fine-tune from HF weights on local disk while checkpointing remotely."""
        with tempfile.TemporaryDirectory() as hf_assets_path:
            expected_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
            shard = "model-00001-of-00001.safetensors"
            save_file(
                {"hf_weight": expected_weight}, os.path.join(hf_assets_path, shard)
            )
            with open(
                os.path.join(hf_assets_path, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump({"weight_map": {"hf_weight": shard}}, f)
            model = nn.Linear(2, 2, bias=False)
            manager = TorchCheckpointingManager.Config(
                keep_latest_k=0,
                initial_load_model_only=True,
                initial_load_in_hf=True,
            ).build(
                model_parts=[model],
                optimizers=_Stateful("optimizer"),
                lr_schedulers=_Stateful("scheduler"),
                ema=None,
                extra_states={"train_state": _Stateful("train")},
                sd_adapter=_StateDictAdapter(hf_assets_path=hf_assets_path),
                base_folder=f"memory://{self._memory_bucket()}",
            )
            try:
                self.assertTrue(manager.load())
            finally:
                manager.close()

            torch.testing.assert_close(model.weight.detach(), expected_weight)

    def test_initial_load_path_on_other_storage_than_the_folder(self) -> None:
        """Start from a local native checkpoint while checkpointing remotely."""

        def build(model: nn.Module, base_folder: str, **overrides):
            return TorchCheckpointingManager.Config(keep_latest_k=0, **overrides).build(
                model_parts=[model],
                optimizers=_Stateful("optimizer"),
                lr_schedulers=_Stateful("scheduler"),
                ema=None,
                extra_states={"train_state": _Stateful("train")},
                sd_adapter=None,
                base_folder=base_folder,
            )

        with tempfile.TemporaryDirectory() as local_dump:
            saved = nn.Linear(2, 2)
            saver = build(saved, local_dump, initial_load_model_only=False)
            try:
                self.assertTrue(saver.save(curr_step=5, last_step=True))
            finally:
                saver.close()

            restored = nn.Linear(2, 2)
            loader = build(
                restored,
                f"memory://{self._memory_bucket()}",
                initial_load_model_only=True,
                initial_load_path=saver._create_last_step_id(5),
            )
            try:
                self.assertTrue(loader.load())
            finally:
                loader.close()

            torch.testing.assert_close(restored.weight, saved.weight)

    def test_remote_folder_hf_round_trip(self) -> None:
        """Export to HF and load it back on a remote folder with the real backend.

        The folder is relative to a remote dump folder, as with
        ``--output-dir gs://...``.
        """
        memory_fs = fsspec.filesystem("memory")
        bucket = f"/hf-round-trip-{uuid.uuid4().hex}"
        self.addCleanup(
            lambda: memory_fs.rm(bucket, recursive=True)
            if memory_fs.exists(bucket)
            else None
        )
        base_folder = f"memory://{bucket}"
        export = f"{bucket}/checkpoint/exports/step-5"

        def build(model: nn.Module, adapter, **overrides) -> TorchCheckpointingManager:
            return TorchCheckpointingManager.Config(keep_latest_k=0, **overrides).build(
                model_parts=[model],
                optimizers=_Stateful("optimizer"),
                lr_schedulers=_Stateful("scheduler"),
                ema=None,
                extra_states={"train_state": _Stateful("train")},
                sd_adapter=adapter,
                base_folder=base_folder,
            )

        saved = nn.Linear(2, 2, bias=False)
        manager = build(
            saved,
            _StateDictAdapter(),
            initial_load_model_only=False,
            last_save_model_only=True,
            last_save_in_hf=True,
        )
        try:
            self.assertTrue(manager.save(curr_step=5, last_step=True))
        finally:
            manager.close()
        self.assertTrue(memory_fs.isfile(f"{export}/model-00001-of-00001.safetensors"))
        self.assertTrue(memory_fs.isfile(f"{export}/model.safetensors.index.json"))

        restored = nn.Linear(2, 2, bias=False)
        loader = build(
            restored,
            _StateDictAdapter(hf_assets_path=f"memory://{export}"),
            initial_load_model_only=True,
            initial_load_in_hf=True,
        )
        try:
            self.assertTrue(loader.load())
        finally:
            loader.close()
        torch.testing.assert_close(restored.weight, saved.weight)

    def test_storage_config_is_an_init_parameter_not_a_config_field(self) -> None:
        # Backend storage is a live object, so it is passed programmatically
        # rather than declared as serializable configuration.
        field_names = {
            field.name for field in dataclasses.fields(TorchCheckpointingManager.Config)
        }
        self.assertNotIn("storage_config", field_names)

        storage = mock.Mock()
        storage_config = mock.Mock()
        storage_config.create_storage.return_value = storage
        config = TorchCheckpointingManager.Config(
            keep_latest_k=0,
            initial_load_model_only=False,
            load_only=True,
        )

        manager, _ = self._build_manager(config, storage_config=storage_config)

        # Path probes go through the same storage, wrapped to answer the
        # CheckpointStorage protocol, and it is pushed into the backend config
        # so saves and loads use it too.
        self.assertIsInstance(manager._storage, CheckpointStorage)
        manager._storage.isdir("/somewhere")
        storage.isdir.assert_called_once_with(Path("/somewhere"))
        self.assertIs(storage_config, manager._manager_config.storage_config)
        manager.close()

    def test_save_obeys_cadence_and_tracks_backend_future(self) -> None:
        config = TorchCheckpointingManager.Config(
            interval=3,
            keep_latest_k=0,
            initial_load_model_only=False,
        )
        manager, backend_manager = self._build_manager(config)

        self.assertFalse(manager.save(curr_step=1))
        self.assertEqual([], backend_manager.save_calls)

        self.assertTrue(manager.save(curr_step=3))
        self.assertEqual(1, len(backend_manager.save_calls))
        checkpoint_id, checkpoint = backend_manager.save_calls[0]
        self.assertEqual("/tmp/checkpoint/step-3", checkpoint_id)
        self.assertEqual("train", checkpoint["train_state"]["value"])
        self.assertIs(backend_manager.save_result, manager.save_future)

        backend_manager.save_result.set_result(None)
        manager.maybe_wait_for_saving()
        self.assertIsNone(manager.save_future)
        manager.close()

    def test_prewarm_runs_once_before_first_scheduled_save(self) -> None:
        config = TorchCheckpointingManager.Config(
            interval=10,
            keep_latest_k=0,
            initial_load_model_only=False,
        )
        manager, backend_manager = self._build_manager(config)

        self.assertFalse(manager.save(curr_step=1))
        self.assertFalse(manager.save(curr_step=2))

        self.assertEqual(1, len(backend_manager.prewarm_calls))
        self.assertEqual(set(manager.states), set(backend_manager.prewarm_calls[0]))
        manager.close()

    def test_ema_is_tracked_and_resharded_when_configured(self) -> None:
        """EMA has to reach this backend's state dict and carry a resharder.
        Without the resharder it would fall through to the no-resharder
        default and silently fail to reshard across world sizes, unlike model
        and optimizer state."""
        from torchtitan.components.checkpointer.base import EMA
        from torchtitan.components.checkpointer.torch_checkpointing import _item_specs

        config = TorchCheckpointingManager.Config(keep_latest_k=0)
        ema = _Stateful("ema")
        manager, _ = self._build_manager(config, ema=ema)
        self.assertIs(manager.states[EMA], ema)

        specs = _item_specs()
        self.assertIn(EMA, specs)
        self.assertIsNotNone(specs[EMA].resharder)
        manager.close()

    def test_ema_absent_when_not_configured(self) -> None:
        from torchtitan.components.checkpointer.base import EMA

        config = TorchCheckpointingManager.Config(keep_latest_k=0)
        manager, _ = self._build_manager(config)
        self.assertNotIn(EMA, manager.states)
        manager.close()

    def test_load_only_uses_synchronous_backend(self) -> None:
        config = TorchCheckpointingManager.Config(
            keep_latest_k=0,
            initial_load_model_only=False,
            load_only=True,
        )

        manager, _ = self._build_manager(config)

        self.assertIsInstance(manager._manager_config.save, SyncCheckpointSaverConfig)
        self.assertIsNone(manager._manager_config.save.writer_config.barrier_config)
        manager.close()

    def test_load_only_does_not_construct_checkpoint_barrier(self) -> None:
        config = TorchCheckpointingManager.Config(
            keep_latest_k=0,
            initial_load_model_only=False,
            load_only=True,
        )

        with mock.patch.object(
            DefaultStoreBarrierConfig,
            "create_barrier",
            side_effect=AssertionError("checkpoint barrier constructed"),
        ):
            manager = config.build(
                model_parts=[nn.Linear(2, 2)],
                optimizers=_Stateful("optimizer"),
                lr_schedulers=_Stateful("scheduler"),
                ema=None,
                extra_states={"train_state": _Stateful("train")},
                sd_adapter=None,
                base_folder="/tmp",
            )

        manager.close()

    def test_staging_wait_uses_backend_lock(self) -> None:
        config = TorchCheckpointingManager.Config(
            keep_latest_k=0,
            initial_load_model_only=False,
        )
        manager, backend_manager = self._build_manager(config)

        manager.maybe_wait_for_staging()

        self.assertEqual(1, backend_manager.lock_calls)
        manager.close()

    def test_save_wait_uses_configured_timeout(self) -> None:
        config = TorchCheckpointingManager.Config(
            keep_latest_k=0,
            initial_load_model_only=False,
        )
        manager, _ = self._build_manager(config)
        save_future = mock.Mock()
        manager.save_future = save_future

        manager.maybe_wait_for_saving()

        save_future.result.assert_called_once_with(
            timeout=manager._manager_config.save.wait_timeout_secs
        )
        self.assertIsNone(manager.save_future)
        manager.close()

    def test_close_releases_resources_when_save_fails(self) -> None:
        config = TorchCheckpointingManager.Config(
            keep_latest_k=2,
            initial_load_model_only=False,
        )
        manager, backend_manager = self._build_manager(config)
        backend_manager.save_result.set_exception(RuntimeError("save failed"))
        manager.save_future = backend_manager.save_result

        with self.assertRaisesRegex(RuntimeError, "save failed"):
            manager.close()

        self.assertTrue(backend_manager.closed)
        self.assertFalse(manager.purge_thread.is_alive())
        self.assertIsNone(manager.save_future)

    def test_purge_runs_before_this_step_save_is_issued(self) -> None:
        # Purging while a save is in flight would let it see, and delete, that
        # save's own temporary directory.
        config = TorchCheckpointingManager.Config(
            interval=1,
            keep_latest_k=0,
            initial_load_model_only=False,
        )
        manager, backend_manager = self._build_manager(config)
        calls = []
        backend_manager.save = lambda checkpoint_id, checkpoint: (
            calls.append("save"),
            backend_manager.save_result,
        )[1]

        with mock.patch.object(
            manager,
            "_purge_stale_checkpoints",
            side_effect=lambda *, saving_step, staging_dir_prefix=None: calls.append(
                ("purge", saving_step, staging_dir_prefix)
            ),
        ):
            self.assertTrue(manager.save(curr_step=1))

        self.assertEqual(
            [
                (
                    "purge",
                    1,
                    manager._manager_config.save.writer_config.temp_dir_prefix,
                ),
                "save",
            ],
            calls,
        )
        backend_manager.save_result.set_result(None)
        manager.close()

    def _purge_manager(self, keep_latest_k: int, entries: list[str]):
        manager = TorchCheckpointingManager.__new__(TorchCheckpointingManager)
        manager.keep_latest_k = keep_latest_k
        manager.folder = "/checkpoint"
        manager.purge_queue = queue.Queue()
        manager.purge_thread = object()
        manager._storage = mock.Mock(spec=CheckpointStorage)
        manager._storage.isdir.return_value = True
        manager._storage.listdir.return_value = entries
        manager._storage.isfile.return_value = True
        return manager

    def _purge(self, manager, *, saving_step: int) -> set[str]:
        manager._purge_stale_checkpoints(
            saving_step=saving_step,
            staging_dir_prefix="tmp_",
        )
        purged = set()
        while not manager.purge_queue.empty():
            purged.add(manager.purge_queue.get_nowait())
        return purged

    @mock.patch("torch.distributed.get_rank", return_value=0)
    def test_purge_reserves_a_slot_for_the_imminent_save(self, _rank) -> None:
        # keep_latest_k=3 with a save about to be issued leaves 2 on disk.
        manager = self._purge_manager(3, ["step-1", "step-2", "step-3"])

        self.assertEqual({"/checkpoint/step-1"}, self._purge(manager, saving_step=4))

    @mock.patch("torch.distributed.get_rank", return_value=0)
    def test_purge_treats_metadata_less_directories_as_abandoned(self, _rank) -> None:
        # A step-N other than the current save with no metadata is the residue
        # of a save that died. It is removed instead of occupying a retained slot.
        manager = self._purge_manager(2, ["step-1", "step-2", "step-3"])
        manager._storage.isfile.return_value = False

        self.assertEqual(
            {
                "/checkpoint/step-1",
                "/checkpoint/step-2",
                "/checkpoint/step-3",
            },
            self._purge(manager, saving_step=4),
        )
        manager._storage.remove.assert_not_called()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    def test_purge_deletes_abandoned_temporaries_without_spending_a_slot(
        self, _rank
    ) -> None:
        # Two saves failed before their rename. They must not push step-1, the
        # only checkpoint this run can resume from, out of the retained set.
        manager = self._purge_manager(3, ["step-1", "tmp_step-2", "tmp_step-3"])

        # step-1 survives, and the purge thread removes the temporaries.
        self.assertEqual(
            {"/checkpoint/tmp_step-2", "/checkpoint/tmp_step-3"},
            self._purge(manager, saving_step=4),
        )
        manager._storage.remove.assert_not_called()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    def test_purge_preserves_the_current_staging_directory(self, _rank) -> None:
        manager = self._purge_manager(2, ["tmp_step-2", "tmp_step-3"])

        self.assertEqual(
            {"/checkpoint/tmp_step-2"}, self._purge(manager, saving_step=3)
        )
        manager._storage.remove.assert_not_called()

    @mock.patch("torch.distributed.get_rank", return_value=0)
    def test_purge_ignores_directories_the_backend_did_not_write(self, _rank) -> None:
        manager = self._purge_manager(
            2, ["step-1", "step-2", "step-3.partial", "step-4-notes", "notes-step-5"]
        )

        self.assertEqual({"/checkpoint/step-1"}, self._purge(manager, saving_step=6))

    def test_last_step_export_converts_only_mismatched_float_tensors(self) -> None:
        # Regression for the two defects #4166 fixed on the DCP side: gating on
        # export_dtype != float32 skipped BF16-to-FP32 exports entirely, and the
        # blanket cast turned integer and boolean buffers into floats. Parameters
        # in preserve_dtype_params keep their dtype.
        cases = (("float32", torch.float32), ("bfloat16", torch.bfloat16))
        for export_dtype, expected in cases:
            with self.subTest(export_dtype=export_dtype):
                config = TorchCheckpointingManager.Config(
                    keep_latest_k=0,
                    initial_load_model_only=False,
                    last_save_model_only=True,
                    export_dtype=export_dtype,
                )
                manager, _ = self._build_manager(config)
                sync_manager = _BackendManager()
                sync_manager.save_result = None
                model = Module()
                model.weight = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
                model.register_buffer("step_count", torch.ones(2, dtype=torch.int64))
                model.register_buffer("mask", torch.ones(2, dtype=torch.bool))
                model.norm = RMSNorm.Config(
                    normalized_shape=2, preserve_dtype_params=("weight",)
                ).build()
                manager.states[MODEL] = ModelWrapper(model)

                with mock.patch.object(
                    BackendCheckpointManager.Config,
                    "build",
                    return_value=sync_manager,
                ):
                    self.assertTrue(manager.save(curr_step=5, last_step=True))

                saved = sync_manager.save_calls[0][1][MODEL]
                self.assertEqual(expected, saved["weight"].dtype)
                self.assertEqual(torch.int64, saved["step_count"].dtype)
                self.assertEqual(torch.bool, saved["mask"].dtype)
                self.assertEqual(torch.float32, saved["norm.weight"].dtype)
                manager.close()

    def test_last_step_uses_synchronous_manager_and_model_only_payload(self) -> None:
        config = TorchCheckpointingManager.Config(
            keep_latest_k=0,
            initial_load_model_only=False,
            last_save_model_only=True,
        )
        manager, backend_manager = self._build_manager(config)
        sync_manager = _BackendManager()
        sync_manager.save_result = None

        with mock.patch.object(
            BackendCheckpointManager.Config,
            "build",
            autospec=True,
            return_value=sync_manager,
        ) as build:
            self.assertTrue(manager.save(curr_step=5, last_step=True))

        self.assertTrue(backend_manager.closed)
        self.assertEqual(1, len(sync_manager.save_calls))
        checkpoint_id, checkpoint = sync_manager.save_calls[0]
        self.assertEqual("/tmp/checkpoint/exports/step-5", checkpoint_id)
        self.assertEqual({MODEL}, set(checkpoint))
        self.assertTrue(sync_manager.closed)
        sync_config = build.call_args.args[0]
        self.assertIsNone(sync_config.pre_finalize_callback)
        manager.close()

    def test_save_stamps_the_step_on_backend_events(self) -> None:
        # The backend reads this context when it builds its own events and
        # exports it to the async save subprocess. Without it every forwarded
        # backend metric carries step=None, which makes them hard to line up
        # against the training step they belong to.
        config = TorchCheckpointingManager.Config(
            interval=1,
            keep_latest_k=0,
            initial_load_model_only=False,
        )
        manager, backend_manager = self._build_manager(config)
        self.addCleanup(checkpoint_logging_context.import_context, {})

        self.assertTrue(manager.save(curr_step=7))

        self.assertEqual(7, checkpoint_logging_context.get("step"))
        backend_manager.save_result.set_result(None)
        manager.close()

    def test_hf_consolidation_uses_the_path_the_backend_supplies(self) -> None:
        """Drive a real backend save and check what pre_finalize_callback gets.

        Every other test here mocks the backend, so they cannot catch the
        callback's path contract changing underneath us -- which it has. This
        asserts against the installed torch_checkpointing: whatever directory
        the writer names, that is where the shards are, so the callback must
        consolidate from it verbatim.
        """

        def _pre_finalize_callback_ensure_nonempty_dir(args: CallbackArgs) -> None:
            self.assertTrue(os.path.isdir(args.path))
            self.assertGreater(
                len(os.listdir(args.path)),
                0,
                msg=f"callback was handed {args.path!r}, which holds no shards",
            )

        with tempfile.TemporaryDirectory() as root:
            checkpoint_id = os.path.join(root, "step-1", "sharded")
            config = BackendCheckpointManager.Config(
                default=ItemSpec(requires_copy=False),
                save=SyncCheckpointSaverConfig(writer_config=CheckpointWriterConfig()),
                # O_DIRECT alignment support varies across CI filesystems and
                # is unrelated to the callback-path contract under test.
                storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
                pre_finalize_callback=_pre_finalize_callback_ensure_nonempty_dir,
            )
            manager = config.build()
            try:
                manager.save(checkpoint_id, {MODEL: torch.ones(2)})
            finally:
                manager.close()

    def test_a_finished_hf_export_is_a_valid_checkpoint(self) -> None:
        # A final HF export keeps the backend's metadata in its nested "sharded"
        # directory and the consolidated files at the root. Recognising only the
        # backend metadata would mark a finished export abandoned, and the next
        # run's pre-save retention deletes abandoned directories outright.
        manager = TorchCheckpointingManager.__new__(TorchCheckpointingManager)
        manager._storage = _BackendCheckpointStorage(
            LocalFileSystemStorageConfig(use_direct_io=False),
            protocol=None,
        )

        for marker, is_resumable in (
            ("metadata.pkl", True),
            ("model.safetensors.index.json", False),
            ("model.safetensors", False),
        ):
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as step:
                open(os.path.join(step, marker), "wb").close()
                self.assertTrue(manager._is_valid_checkpoint(step))
                self.assertEqual(is_resumable, manager._is_resumable_checkpoint(step))

        with tempfile.TemporaryDirectory() as step:
            self.assertFalse(manager._is_valid_checkpoint(step))

    def test_subprocess_logging_initializes_and_delegates(self) -> None:
        calls = []
        init_fn = mock.Mock(side_effect=lambda *_args: calls.append("existing"))
        structured_logger_init_fn = mock.Mock(
            side_effect=lambda: calls.append("structured")
        )
        manager_module._init_subprocess_logging(
            structured_logger_init_fn,
            init_fn,
            ("argument",),
        )

        structured_logger_init_fn.assert_called_once_with()
        init_fn.assert_called_once_with("argument")
        self.assertEqual(["existing", "structured"], calls)

    def _init_subprocess_logging(self) -> None:
        manager_module._init_subprocess_logging(mock.Mock(), None, ())

    def test_subprocess_logging_only_overrides_suppressed_inherited_level(self) -> None:
        root_logger = logging.getLogger()
        backend_logger = logging.getLogger(manager_module.CHECKPOINTING_LOGGER_NAME)
        self.addCleanup(root_logger.setLevel, root_logger.level)
        self.addCleanup(backend_logger.setLevel, backend_logger.level)
        for root_level, backend_level, expected_level in (
            (logging.WARNING, logging.NOTSET, logging.INFO),
            (logging.WARNING, logging.DEBUG, logging.DEBUG),
            (logging.WARNING, logging.WARNING, logging.WARNING),
            (logging.DEBUG, logging.NOTSET, logging.NOTSET),
        ):
            with self.subTest(root_level=root_level, backend_level=backend_level):
                root_logger.setLevel(root_level)
                backend_logger.setLevel(backend_level)

                self._init_subprocess_logging()

                self.assertEqual(expected_level, backend_logger.level)

    def test_async_backend_config_composes_subprocess_logging_initializer(self) -> None:
        original_init_fn = mock.Mock()
        structured_logger_init_fn = mock.Mock()

        with mock.patch.object(
            manager_module.sl,
            "get_structured_logger_subprocess_init_fn",
            return_value=structured_logger_init_fn,
        ):
            backend_config = _default_backend_config(
                _async_save_config(),
                subprocess_init_fn=original_init_fn,
                subprocess_init_args=("argument",),
            )

        self.assertIs(
            backend_config.subprocess_init_fn,
            manager_module._init_subprocess_logging,
        )
        self.assertEqual(
            (
                structured_logger_init_fn,
                original_init_fn,
                ("argument",),
            ),
            backend_config.subprocess_init_args,
        )

    @mock.patch.object(
        manager_module,
        "consolidate_hf_safetensors_checkpoint",
        create=True,
    )
    def test_hf_final_save_converts_and_consolidates_before_commit(
        self,
        consolidate,
    ) -> None:
        adapter = _StateDictAdapter()
        config = TorchCheckpointingManager.Config(
            keep_latest_k=0,
            initial_load_model_only=False,
            last_save_model_only=True,
            last_save_in_hf=True,
        )
        storage_config = mock.Mock()
        storage_config.create_storage.return_value = mock.Mock()
        backend_manager = _BackendManager()
        sync_manager = _BackendManager()
        sync_manager.save_result = None
        with mock.patch.object(
            BackendCheckpointManager.Config,
            "build",
            autospec=True,
            side_effect=[backend_manager, sync_manager],
        ) as build:
            manager = config.build(
                model_parts=[nn.Linear(2, 2)],
                optimizers=_Stateful("optimizer"),
                lr_schedulers=_Stateful("scheduler"),
                ema=None,
                extra_states={"train_state": _Stateful("train")},
                sd_adapter=adapter,
                base_folder="/tmp",
                storage_config=storage_config,
            )
            self.assertTrue(manager.save(curr_step=5, last_step=True))

        sync_config = build.call_args_list[1].args[0]
        self.assertEqual(
            "/tmp/checkpoint/exports/step-5/sharded",
            sync_manager.save_calls[0][0],
        )
        checkpoint = sync_manager.save_calls[0][1]
        self.assertEqual({MODEL}, set(checkpoint))
        self.assertEqual({"hf_weight"}, set(checkpoint[MODEL]))
        torch.testing.assert_close(
            checkpoint[MODEL]["hf_weight"],
            manager.states[MODEL].state_dict()["weight"],
        )
        model_spec = sync_config.items[MODEL]
        self.assertIsInstance(
            model_spec.layout.serialization_format,
            SafetensorsSerialization,
        )
        self.assertEqual(f"{MODEL}_{{rank}}.safetensors", model_spec.layout.file_path)

        # The backend hands the callback the directory the shards were actually
        # written to -- its staging directory when a barrier is configured. Feed
        # that in and assert it is consolidated as given, with no derivation.
        self.assertIsNotNone(sync_config.save.writer_config.barrier_config)
        # Built from the writer's public config rather than the backend's
        # private _temp_dir_path helper, so a rename upstream cannot break
        # collection of this module the way CheckpointWriter.TMP_PREFIX did.
        save_path = Path(sync_manager.save_calls[0][0])
        prefix = sync_config.save.writer_config.temp_dir_prefix
        staged = save_path.parent / f"{prefix}{save_path.name}"
        sync_config.pre_finalize_callback(
            CallbackArgs(
                path=str(staged),
                storage=mock.Mock(),
                event_logger=mock.Mock(),
            )
        )
        consolidate.assert_called_once_with(
            "/tmp/checkpoint/exports/step-5/tmp_sharded",
            output_dir="/tmp/checkpoint/exports/step-5",
            item_key=MODEL,
            fqn_to_index_mapping=adapter.fqn_to_index_mapping,
            storage_config=storage_config,
        )
        manager.close()

    def test_hf_final_save_rejects_frozen_params(self) -> None:
        """State dict adapters drop LoRA adapters, so the export would be the base."""
        model = nn.Linear(2, 2)
        model.bias.requires_grad_(False)
        config = TorchCheckpointingManager.Config(
            last_save_model_only=True, last_save_in_hf=True
        )
        with self.assertRaisesRegex(ValueError, "frozen parameters"):
            config.build(
                model_parts=[model],
                optimizers=_Stateful("optimizer"),
                lr_schedulers=_Stateful("scheduler"),
                ema=None,
                extra_states={},
                sd_adapter=_StateDictAdapter(),
                storage_config=mock.Mock(),
            )

    def test_hf_load_uses_temporary_model_only_manager_and_restores_model(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_id = os.path.join(base_folder, "hf_checkpoint")
            os.makedirs(checkpoint_id)
            with open(
                os.path.join(checkpoint_id, "model.safetensors.index.json"),
                "w",
            ):
                pass
            model = nn.Linear(2, 2, bias=False)
            adapter = _StateDictAdapter(hf_assets_path=checkpoint_id)
            config = TorchCheckpointingManager.Config(
                keep_latest_k=0,
                initial_load_model_only=True,
                initial_load_in_hf=True,
            )
            backend_manager = _BackendManager()
            hf_manager = _BackendManager()
            expected_weight = torch.full_like(model.weight, 3)
            hf_manager.load_result = {
                MODEL: {"hf_weight": expected_weight},
            }

            with mock.patch.object(
                BackendCheckpointManager.Config,
                "build",
                autospec=True,
                side_effect=[backend_manager, hf_manager],
            ) as build:
                manager = config.build(
                    model_parts=[model],
                    optimizers=_Stateful("optimizer"),
                    lr_schedulers=_Stateful("scheduler"),
                    ema=None,
                    extra_states={"train_state": _Stateful("train")},
                    sd_adapter=adapter,
                    base_folder=base_folder,
                )

                self.assertTrue(manager.load())

            self.assertEqual([], backend_manager.load_calls)
            self.assertEqual(1, len(hf_manager.load_calls))
            loaded_id, into, kwargs = hf_manager.load_calls[0]
            self.assertEqual(checkpoint_id, loaded_id)
            self.assertIs(adapter.to_hf_results[0], into[MODEL])
            self.assertEqual(
                {
                    "strict": True,
                    "metadata_format": HuggingFaceSafetensorsDistributedMetadataFormat,
                },
                kwargs,
            )
            self.assertEqual(1, len(adapter.from_hf_calls))
            self.assertIs(
                expected_weight,
                adapter.from_hf_calls[0]["hf_weight"],
            )
            torch.testing.assert_close(model.weight, expected_weight)

            hf_config = build.call_args_list[1].args[0]
            self.assertIsInstance(
                manager._manager_config.save,
                AsyncCheckpointSaverConfig,
            )
            self.assertIsInstance(hf_config.save, SyncCheckpointSaverConfig)
            self.assertIsNone(hf_config.save.writer_config.barrier_config)
            self.assertEqual({MODEL}, set(hf_config.items))
            self.assertIsNone(hf_config.default)
            # The HF spec derives every field from the model spec, resharder
            # included: DefaultResharder reads safetensors sources directly.
            self.assertIs(
                hf_config.items[MODEL].resharder,
                manager._manager_config.items[MODEL].resharder,
            )
            self.assertEqual(
                manager._manager_config.items[MODEL].requires_copy,
                hf_config.items[MODEL].requires_copy,
            )
            self.assertEqual(
                manager._manager_config.items[MODEL].layout,
                hf_config.items[MODEL].layout,
            )
            self.assertEqual(
                manager._manager_config.items[MODEL].required,
                hf_config.items[MODEL].required,
            )
            self.assertIsInstance(
                manager._manager_config.items[MODEL].resharder,
                DefaultResharder,
            )
            self.assertIs(
                manager._manager_config.storage_config,
                hf_config.storage_config,
            )
            self.assertTrue(hf_manager.closed)
            self.assertFalse(backend_manager.closed)
            manager.close()
            self.assertTrue(backend_manager.closed)

    def test_hf_load_reads_a_real_safetensors_export(self) -> None:
        """Load an actual export end to end, with no backend mocks."""
        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_id = os.path.join(base_folder, "hf_checkpoint")
            os.makedirs(checkpoint_id)
            expected_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
            shard = "model-00001-of-00001.safetensors"
            save_file(
                {"hf_weight": expected_weight}, os.path.join(checkpoint_id, shard)
            )
            with open(
                os.path.join(checkpoint_id, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump({"weight_map": {"hf_weight": shard}}, f)
            model = nn.Linear(2, 2, bias=False)
            config = TorchCheckpointingManager.Config(
                keep_latest_k=0,
                initial_load_model_only=True,
                initial_load_in_hf=True,
            )
            manager = config.build(
                model_parts=[model],
                optimizers=_Stateful("optimizer"),
                lr_schedulers=_Stateful("scheduler"),
                ema=None,
                extra_states={"train_state": _Stateful("train")},
                sd_adapter=_StateDictAdapter(hf_assets_path=checkpoint_id),
                base_folder=base_folder,
                storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
            )
            try:
                self.assertTrue(manager.load())
            finally:
                manager.close()

            torch.testing.assert_close(model.weight.detach(), expected_weight)

    def test_hf_load_leaves_parameters_the_export_does_not_carry(self) -> None:
        """A LoRA model loads its base weights from an HF export of the base model."""

        class _LoRAModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.zeros(2, 2))
                # to_hf has no HF name for this, as for a new LoRA adapter.
                self.lora_a = nn.Parameter(torch.full((2, 2), 7.0))

        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_id = os.path.join(base_folder, "hf_checkpoint")
            os.makedirs(checkpoint_id)
            expected_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
            save_file(
                {"hf_weight": expected_weight},
                os.path.join(checkpoint_id, "model.safetensors"),
            )
            model = _LoRAModel()
            config = TorchCheckpointingManager.Config(
                keep_latest_k=0,
                initial_load_model_only=True,
                initial_load_in_hf=True,
            )
            manager = config.build(
                model_parts=[model],
                optimizers=_Stateful("optimizer"),
                lr_schedulers=_Stateful("scheduler"),
                ema=None,
                extra_states={"train_state": _Stateful("train")},
                sd_adapter=_StateDictAdapter(hf_assets_path=checkpoint_id),
                base_folder=base_folder,
                storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
            )
            try:
                self.assertTrue(manager.load())
            finally:
                manager.close()

            torch.testing.assert_close(model.weight.detach(), expected_weight)
            torch.testing.assert_close(model.lora_a.detach(), torch.full((2, 2), 7.0))

    def test_hf_load_closes_temporary_manager_when_load_fails(self) -> None:
        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_id = os.path.join(base_folder, "hf_checkpoint")
            os.makedirs(checkpoint_id)
            with open(os.path.join(checkpoint_id, "model.safetensors"), "wb"):
                pass
            adapter = _StateDictAdapter()
            config = TorchCheckpointingManager.Config(
                keep_latest_k=0,
                initial_load_path=checkpoint_id,
                initial_load_model_only=True,
                initial_load_in_hf=True,
                load_only=True,
            )
            backend_manager = _BackendManager()
            hf_manager = _BackendManager()
            hf_manager.load = mock.Mock(side_effect=RuntimeError("load failed"))

            with mock.patch.object(
                BackendCheckpointManager.Config,
                "build",
                autospec=True,
                side_effect=[backend_manager, hf_manager],
            ):
                manager = config.build(
                    model_parts=[nn.Linear(2, 2)],
                    optimizers=_Stateful("optimizer"),
                    lr_schedulers=_Stateful("scheduler"),
                    ema=None,
                    extra_states={"train_state": _Stateful("train")},
                    sd_adapter=adapter,
                    base_folder=base_folder,
                )

                with self.assertRaisesRegex(RuntimeError, "load failed"):
                    manager.load()

            self.assertTrue(hf_manager.closed)
            self.assertEqual([], adapter.from_hf_calls)
            manager.close()

    def test_hf_load_rejects_quantized_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_id = os.path.join(base_folder, "hf_checkpoint")
            os.makedirs(checkpoint_id)
            with open(os.path.join(checkpoint_id, "model.safetensors"), "wb"):
                pass
            config = TorchCheckpointingManager.Config(
                keep_latest_k=0,
                initial_load_path=checkpoint_id,
                initial_load_model_only=True,
                initial_load_in_hf=True,
                initial_load_in_hf_quantized=True,
                load_only=True,
            )
            manager, backend_manager = self._build_manager(
                config,
                base_folder=base_folder,
                sd_adapter=_StateDictAdapter(),
            )

            with self.assertRaisesRegex(ValueError, "quantized"):
                manager.load()

            self.assertEqual([], backend_manager.load_calls)
            manager.close()

    def test_native_load_restores_model_and_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_id = os.path.join(base_folder, "checkpoint", "step-5")
            os.makedirs(checkpoint_id)
            with open(os.path.join(checkpoint_id, "metadata.pkl"), "wb"):
                pass
            hf_export_id = os.path.join(base_folder, "checkpoint", "step-8")
            os.makedirs(hf_export_id)
            with open(os.path.join(hf_export_id, "model.safetensors"), "wb"):
                pass
            hf_checkpoint_id = os.path.join(base_folder, "hf_checkpoint")
            os.makedirs(hf_checkpoint_id)
            with open(os.path.join(hf_checkpoint_id, "model.safetensors"), "wb"):
                pass
            model = nn.Linear(2, 2, bias=False)
            optimizer = _Stateful("optimizer")
            adapter = _StateDictAdapter()
            config = TorchCheckpointingManager.Config(
                folder="checkpoint",
                keep_latest_k=0,
                initial_load_path=hf_checkpoint_id,
                initial_load_model_only=True,
                initial_load_in_hf=True,
                load_only=True,
            )
            manager, backend_manager = self._build_manager(
                config,
                base_folder=base_folder,
                model_parts=[model],
                optimizers=optimizer,
                sd_adapter=adapter,
            )
            expected_weight = torch.full_like(model.weight, 3)
            backend_manager.load_result = {
                MODEL: {"weight": expected_weight},
                OPTIMIZER: {"value": "restored"},
            }

            self.assertTrue(manager.load())

            torch.testing.assert_close(model.weight, expected_weight)
            self.assertEqual("restored", optimizer.value)
            self.assertEqual(checkpoint_id, backend_manager.load_calls[0][0])
            self.assertEqual(set(manager.states), set(backend_manager.load_calls[0][1]))
            self.assertEqual({"strict": True}, backend_manager.load_calls[0][2])
            self.assertEqual([], adapter.to_hf_calls)
            manager.close()

    def test_native_load_requires_every_requested_key(self) -> None:
        # The backend skips absent keys by default, which would leave those
        # parameters at their initialized values and quietly resume from a model
        # that is not the one that was saved.
        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_id = os.path.join(base_folder, "checkpoint", "step-5")
            os.makedirs(checkpoint_id)
            with open(os.path.join(checkpoint_id, "metadata.pkl"), "wb"):
                pass
            config = TorchCheckpointingManager.Config(
                folder="checkpoint",
                keep_latest_k=0,
                initial_load_model_only=False,
                load_only=True,
            )
            manager, backend_manager = self._build_manager(
                config,
                base_folder=base_folder,
            )
            backend_manager.load = mock.Mock(
                side_effect=RuntimeError(
                    f"Checkpoint at {checkpoint_id} is missing keys: "
                    f"['{MODEL}::weight']"
                )
            )

            with self.assertRaisesRegex(RuntimeError, "is missing keys"):
                manager.load(step=5)

            self.assertIs(True, backend_manager.load.call_args.kwargs["strict"])
            manager.close()

    def test_explicit_load_step_rejects_incomplete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as base_folder:
            os.makedirs(os.path.join(base_folder, "checkpoint", "step-5"))
            config = TorchCheckpointingManager.Config(
                folder="checkpoint",
                keep_latest_k=0,
                initial_load_model_only=False,
                load_only=True,
            )
            manager, backend_manager = self._build_manager(
                config,
                base_folder=base_folder,
            )

            with self.assertRaisesRegex(
                ValueError, "step-5 is not a complete checkpoint"
            ):
                manager.load(step=5)

            self.assertEqual([], backend_manager.load_calls)
            manager.close()

    def test_initial_load_path_rejects_hf_export(self) -> None:
        # The backend's own check accepts a final HF export as a valid
        # checkpoint, but a native load cannot read it.
        with tempfile.TemporaryDirectory() as base_folder:
            initial_load_path = os.path.join(base_folder, "initial", "step-5")
            os.makedirs(initial_load_path)
            with open(
                os.path.join(initial_load_path, f"{MODEL}.safetensors.index.json"), "w"
            ):
                pass
            config = TorchCheckpointingManager.Config(
                folder="checkpoint",
                keep_latest_k=0,
                initial_load_path=initial_load_path,
                load_only=True,
            )
            manager, backend_manager = self._build_manager(
                config,
                base_folder=base_folder,
            )

            with self.assertRaisesRegex(
                ValueError, "step-5 is not a complete checkpoint"
            ):
                manager.load()

            self.assertEqual([], backend_manager.load_calls)
            manager.close()

    def test_load_latest_step_zero_loads_only_model_state(self) -> None:
        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_id = os.path.join(base_folder, "checkpoint", "step-0")
            os.makedirs(checkpoint_id)
            with open(os.path.join(checkpoint_id, "metadata.pkl"), "wb"):
                pass
            config = TorchCheckpointingManager.Config(
                folder="checkpoint",
                keep_latest_k=0,
                initial_load_model_only=False,
                load_only=True,
            )
            manager, backend_manager = self._build_manager(
                config,
                base_folder=base_folder,
            )

            self.assertTrue(manager.load())

            self.assertEqual({MODEL}, set(backend_manager.load_calls[0][1]))
            manager.close()

    def test_model_only_load_reseeds_ema_from_loaded_weights(self) -> None:
        # Same contract as the DCP CheckpointManager: an EMA that was not
        # restored is reseeded from the weights that were.
        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_id = os.path.join(base_folder, "checkpoint", "step-0")
            os.makedirs(checkpoint_id)
            with open(os.path.join(checkpoint_id, "metadata.pkl"), "wb"):
                pass
            # initial_load_model_only only applies to initial_load_path; the
            # step-0 seed checkpoint is loaded model-only regardless.
            config = TorchCheckpointingManager.Config(
                folder="checkpoint",
                keep_latest_k=0,
                initial_load_model_only=False,
                load_only=True,
            )
            ema = mock.create_autospec(Stateful, instance=True)
            ema.state_dict.return_value = {}
            manager, backend_manager = self._build_manager(
                config,
                base_folder=base_folder,
                ema=ema,
            )

            self.assertTrue(manager.load())

            self.assertEqual({MODEL}, set(backend_manager.load_calls[0][1]))
            ema.load_state_dict.assert_called_once_with({})
            manager.close()

    def _resume_step_5_with_ema(self, **config_kwargs):
        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_id = os.path.join(base_folder, "checkpoint", "step-5")
            os.makedirs(checkpoint_id)
            with open(os.path.join(checkpoint_id, "metadata.pkl"), "wb"):
                pass
            config = TorchCheckpointingManager.Config(
                folder="checkpoint",
                keep_latest_k=0,
                load_only=True,
                **config_kwargs,
            )
            ema = mock.create_autospec(Stateful, instance=True)
            # Non-empty, so restoring it is distinguishable from a reseed ({}).
            ema.state_dict.return_value = {"num_updates": 5}
            manager, backend_manager = self._build_manager(
                config,
                base_folder=base_folder,
                ema=ema,
            )
            self.assertTrue(manager.load())
            manager.close()
            return set(backend_manager.load_calls[0][1]), ema

    def test_ema_excluded_from_resume_is_reseeded(self) -> None:
        loaded, ema = self._resume_step_5_with_ema(exclude_from_loading=["ema"])

        self.assertNotIn(EMA, loaded)
        ema.load_state_dict.assert_called_once_with({})

    def test_full_resume_keeps_restored_ema(self) -> None:
        # Reseeding here would wipe the EMA history restored with the checkpoint.
        loaded, ema = self._resume_step_5_with_ema()

        self.assertIn(EMA, loaded)
        self.assertNotIn(mock.call({}), ema.load_state_dict.call_args_list)

    def test_load_latest_uses_configured_storage(self) -> None:
        # The backend Storage the adapter wraps: step-7 is a directory holding a
        # metadata.pkl file.
        storage = mock.Mock()
        storage.ls.return_value = ["step-7"]
        storage.exists.return_value = True
        storage.isdir.side_effect = lambda path: not str(path).endswith("metadata.pkl")
        storage_config = mock.Mock()
        storage_config.create_storage.return_value = storage
        config = TorchCheckpointingManager.Config(
            folder="checkpoint",
            keep_latest_k=0,
            initial_load_model_only=False,
            load_only=True,
        )
        manager, backend_manager = self._build_manager(
            config,
            storage_config=storage_config,
            base_folder="/custom",
        )

        self.assertTrue(manager.load())

        self.assertEqual(
            "/custom/checkpoint/step-7",
            backend_manager.load_calls[0][0],
        )
        manager.close()

    def test_load_latest_ignores_incomplete_checkpoint_directories(self) -> None:
        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_folder = os.path.join(base_folder, "checkpoint")
            for step in (2, 5):
                checkpoint_id = os.path.join(checkpoint_folder, f"step-{step}")
                os.makedirs(checkpoint_id)
                with open(os.path.join(checkpoint_id, "metadata.pkl"), "wb"):
                    pass
            incomplete_checkpoint_id = os.path.join(checkpoint_folder, "step-8")
            os.makedirs(incomplete_checkpoint_id)
            # An interrupted save leaves its metadata behind, so the temporary
            # directory looks complete. Resuming from it would build the id
            # "step-9", which does not exist.
            # Neither is a checkpoint id we could reconstruct: "tmp_step-9" is
            # an interrupted save whose metadata already landed, so it looks
            # complete, and "step-99.partial" is not a name we write at all.
            # Matching either one loosely resolves to a step that does not exist.
            for decoy in ("tmp_step-9", "step-99.partial"):
                decoy_id = os.path.join(checkpoint_folder, decoy)
                os.makedirs(decoy_id)
                with open(os.path.join(decoy_id, "metadata.pkl"), "wb"):
                    pass
            config = TorchCheckpointingManager.Config(
                folder="checkpoint",
                keep_latest_k=0,
                initial_load_model_only=False,
                load_only=True,
            )
            manager, backend_manager = self._build_manager(
                config,
                base_folder=base_folder,
            )

            self.assertTrue(manager.load())

            self.assertEqual(
                os.path.join(checkpoint_folder, "step-5"),
                backend_manager.load_calls[0][0],
            )
            manager.close()

    def test_load_latest_on_remote_folder_ignores_incomplete_checkpoint_directories(
        self,
    ) -> None:
        bucket = f"bucket-{uuid.uuid4().hex}"
        memory_fs = fsspec.filesystem("memory")
        self.addCleanup(
            lambda: memory_fs.rm(bucket, recursive=True)
            if memory_fs.exists(bucket)
            else None
        )
        memory_folder = f"{bucket}/checkpoint"
        for step in (2, 5):
            memory_fs.pipe_file(f"{memory_folder}/step-{step}/metadata.pkl", b"")
        memory_fs.makedirs(f"{memory_folder}/step-8", exist_ok=True)
        for decoy in ("tmp_step-9", "step-99.partial"):
            memory_fs.pipe_file(f"{memory_folder}/{decoy}/metadata.pkl", b"")
        config = TorchCheckpointingManager.Config(
            folder=f"memory://{memory_folder}",
            keep_latest_k=0,
            initial_load_model_only=False,
            load_only=True,
        )
        manager, backend_manager = self._build_manager(config)

        self.assertTrue(manager.load())

        self.assertEqual(
            f"{memory_folder}/step-5",
            backend_manager.load_calls[0][0],
        )
        manager.close()
