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
from concurrent.futures import Future
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn

import torchtitan.components.checkpointer.torch_checkpointing as manager_module
from torch.distributed.checkpoint.stateful import Stateful
from torch_checkpointing.barriers import TCPStoreBarrierConfig
from torch_checkpointing.checkpoint_layout import SafetensorsSerialization
from torch_checkpointing.checkpoint_manager import (
    CheckpointManager as BackendCheckpointManager,
)
from torch_checkpointing.checkpoint_writer import CheckpointWriterConfig
from torch_checkpointing.config import (
    AsyncCheckpointSaverConfig,
    SyncCheckpointSaverConfig,
)
from torch_checkpointing.default_resharder import DefaultResharder
from torch_checkpointing.logging_utils import checkpoint_logging_context
from torch_checkpointing.schema import ItemSpec
from torch_checkpointing.storage.filesystem import LocalFileSystemStorageConfig
from torchtitan.components.checkpointer import (
    BaseCheckpointManager,
    CheckpointManager,
    CheckpointStorage,
    MODEL,
    OPTIMIZER,
)
from torchtitan.components.checkpointer.torch_checkpointing import (
    _async_save_config,
    _default_backend_config,
    DEFAULT_TORCH_CHECKPOINTING_BARRIER_TCPSTORE_PORT,
    TorchCheckpointingManager,
)
from torchtitan.config import Function


class _BackendManager:
    def __init__(self) -> None:
        self.closed = False
        self.lock_calls = 0
        self.load_calls = []
        self.load_result = None
        self.prewarm_calls = []
        self.save_calls = []
        self.save_result = Future()

    def save(self, checkpoint_id, checkpoint):
        self.save_calls.append((checkpoint_id, checkpoint))
        return self.save_result

    def prewarm_staging(self, checkpoint) -> None:
        self.prewarm_calls.append(checkpoint)

    def lock(self):
        self.lock_calls += 1
        return nullcontext()

    def load(self, checkpoint_id, into=None, **kwargs):
        self.load_calls.append((checkpoint_id, into, kwargs))
        return self.load_result if self.load_result is not None else into

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
    def __init__(self) -> None:
        self.fqn_to_index_mapping = {"hf_weight": 1}
        self.to_hf_calls = []

    def to_hf(self, state_dict):
        self.to_hf_calls.append(state_dict)
        return {"hf_weight": state_dict["weight"]}


class TorchCheckpointingManagerTest(unittest.TestCase):
    def _build_manager(
        self,
        config: TorchCheckpointingManager.Config,
        *,
        storage_config=None,
        base_folder: str = "/tmp",
        model_parts=None,
        optimizers=None,
        states=None,
    ) -> tuple[TorchCheckpointingManager, _BackendManager]:
        backend_manager = _BackendManager()
        with mock.patch.object(
            BackendCheckpointManager.Config,
            "build",
            return_value=backend_manager,
        ):
            manager = config.build(
                dataloader=None,
                model_parts=model_parts or [nn.Linear(2, 2)],
                optimizers=optimizers or _Stateful("optimizer"),
                lr_schedulers=_Stateful("scheduler"),
                states=states or {"train_state": _Stateful("train")},
                sd_adapter=None,
                base_folder=base_folder,
                storage_config=storage_config,
            )
        return manager, backend_manager

    def test_config_builds_independent_manager(self) -> None:
        config = TorchCheckpointingManager.Config(
            enable=True,
            keep_latest_k=0,
            initial_load_model_only=False,
            purge_exempt=Function.Config(fn=lambda step: step % 2 == 0),
        )

        manager, backend_manager = self._build_manager(config)

        self.assertIsInstance(manager, TorchCheckpointingManager)
        self.assertNotIsInstance(manager, CheckpointManager)
        self.assertTrue(manager._is_purge_exempt(2))
        self.assertFalse(manager._is_purge_exempt(3))
        manager.close()
        self.assertTrue(backend_manager.closed)

    def test_config_adds_no_checkpoint_options(self) -> None:
        field_names = {
            field.name for field in dataclasses.fields(TorchCheckpointingManager.Config)
        }
        base_field_names = {
            field.name for field in dataclasses.fields(BaseCheckpointManager.Config)
        }

        self.assertEqual(field_names, base_field_names)

    def test_config_to_dict_is_json_serializable(self) -> None:
        config_dict = TorchCheckpointingManager.Config().to_dict()

        json.dumps(config_dict)
        self.assertNotIn("checkpoint_manager", config_dict)

    def test_disabled_manager_lifecycle_is_noop(self) -> None:
        config = TorchCheckpointingManager.Config(
            enable=False,
            initial_load_model_only=False,
        )
        manager, _ = self._build_manager(config)

        self.assertFalse(manager.load())
        self.assertFalse(manager.save(curr_step=1))
        self.assertIsNone(manager.maybe_wait_for_staging())
        manager.close()

    def test_del_ignores_manager_whose_construction_failed(self) -> None:
        manager = TorchCheckpointingManager.__new__(TorchCheckpointingManager)
        manager.enable = True
        manager.save_future = None
        manager.purge_thread = None

        manager.__del__()

    @mock.patch.dict("os.environ", {"MASTER_ADDR": "checkpoint-host"})
    def test_default_backend_configuration_owns_schema_and_barrier(self) -> None:
        backend_config = _default_backend_config(_async_save_config())

        self.assertIsInstance(backend_config.save, AsyncCheckpointSaverConfig)
        self.assertTrue(backend_config.save.staging_config.use_pinned_memory)
        self.assertEqual(set(backend_config.items), {MODEL, OPTIMIZER})
        for spec in backend_config.items.values():
            self.assertTrue(spec.requires_copy)
            self.assertFalse(spec.required)
            self.assertIsInstance(spec.resharder, DefaultResharder)
        self.assertIsNotNone(backend_config.default)
        self.assertFalse(backend_config.default.requires_copy)
        barrier_config = backend_config.save.writer_config.barrier_config
        self.assertIsInstance(barrier_config, TCPStoreBarrierConfig)
        self.assertEqual(barrier_config.master_address, "checkpoint-host")
        self.assertEqual(
            barrier_config.tcpstore_port,
            DEFAULT_TORCH_CHECKPOINTING_BARRIER_TCPSTORE_PORT,
        )

    def test_remote_checkpoint_paths_are_rejected_at_construction(self) -> None:
        # Path() would turn "gs://bucket/x" into "gs:/bucket/x". Rejecting the
        # two roots up front is what lets every path derived from them be
        # converted without a further check -- and a save with retention off
        # reaches the backend having run no probe that could have caught it.
        for field, kwargs in (
            ("checkpoint.folder", {"folder": "gs://bucket/checkpoint"}),
            (
                "checkpoint.initial_load_path",
                {"initial_load_path": "gs://bucket/pretrained"},
            ),
        ):
            with self.subTest(field=field):
                config = TorchCheckpointingManager.Config(
                    enable=True,
                    keep_latest_k=0,
                    initial_load_model_only=True,
                    **kwargs,
                )
                with self.assertRaisesRegex(ValueError, rf"{field}.*not yet supported"):
                    self._build_manager(config)

    def test_storage_config_is_an_init_parameter_not_a_config_field(self) -> None:
        # Backend storage is passed programmatically rather than declared on
        # Config, which is Tyro-parsed and not the place for a storage object.
        field_names = {
            field.name for field in dataclasses.fields(TorchCheckpointingManager.Config)
        }
        self.assertNotIn("storage_config", field_names)

        storage = mock.Mock()
        storage_config = mock.Mock()
        storage_config.create_storage.return_value = storage
        config = TorchCheckpointingManager.Config(
            enable=True,
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

    def test_legacy_config_has_no_backend_selector(self) -> None:
        config = CheckpointManager.Config()

        self.assertFalse(hasattr(config, "save_backend"))
        self.assertFalse(hasattr(config, "load_backend"))

    def test_save_obeys_cadence_and_tracks_backend_future(self) -> None:
        config = TorchCheckpointingManager.Config(
            enable=True,
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
            enable=True,
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

    def test_load_only_uses_synchronous_backend(self) -> None:
        config = TorchCheckpointingManager.Config(
            enable=True,
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
            enable=True,
            keep_latest_k=0,
            initial_load_model_only=False,
            load_only=True,
        )

        with mock.patch.object(
            TCPStoreBarrierConfig,
            "create_barrier",
            side_effect=AssertionError("checkpoint barrier constructed"),
        ):
            manager = config.build(
                dataloader=None,
                model_parts=[nn.Linear(2, 2)],
                optimizers=_Stateful("optimizer"),
                lr_schedulers=_Stateful("scheduler"),
                states={"train_state": _Stateful("train")},
                sd_adapter=None,
                base_folder="/tmp",
            )

        manager.close()

    def test_staging_wait_uses_backend_lock(self) -> None:
        config = TorchCheckpointingManager.Config(
            enable=True,
            keep_latest_k=0,
            initial_load_model_only=False,
        )
        manager, backend_manager = self._build_manager(config)

        manager.maybe_wait_for_staging()

        self.assertEqual(1, backend_manager.lock_calls)
        manager.close()

    def test_save_wait_uses_configured_timeout(self) -> None:
        config = TorchCheckpointingManager.Config(
            enable=True,
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
            enable=True,
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
            enable=True,
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
        # blanket cast turned integer and boolean buffers into floats.
        cases = (("float32", torch.float32), ("bfloat16", torch.bfloat16))
        for export_dtype, expected in cases:
            with self.subTest(export_dtype=export_dtype):
                config = TorchCheckpointingManager.Config(
                    enable=True,
                    keep_latest_k=0,
                    initial_load_model_only=False,
                    last_save_model_only=True,
                    export_dtype=export_dtype,
                )
                manager, _ = self._build_manager(config)
                sync_manager = _BackendManager()
                sync_manager.save_result = None
                model = _Stateful(None)
                model.state_dict = lambda: {
                    "weight": torch.ones(2, dtype=torch.bfloat16),
                    "step_count": torch.ones(2, dtype=torch.int64),
                    "mask": torch.ones(2, dtype=torch.bool),
                }
                manager.states[MODEL] = model

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
                manager.close()

    def test_last_step_uses_synchronous_manager_and_model_only_payload(self) -> None:
        config = TorchCheckpointingManager.Config(
            enable=True,
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
        self.assertEqual("/tmp/checkpoint/step-5", checkpoint_id)
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
            enable=True,
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
        received: list[str] = []
        with tempfile.TemporaryDirectory() as root:
            checkpoint_id = os.path.join(root, "step-1", "sharded")
            config = BackendCheckpointManager.Config(
                default=ItemSpec(requires_copy=False),
                save=SyncCheckpointSaverConfig(
                    writer_config=CheckpointWriterConfig(barrier_config=None)
                ),
                # O_DIRECT alignment support varies across CI filesystems and
                # is unrelated to the callback-path contract under test.
                storage_config=LocalFileSystemStorageConfig(use_direct_io=False),
                pre_finalize_callback=lambda path, _logger: received.append(path),
            )
            manager = config.build()
            try:
                manager.save(checkpoint_id, {MODEL: torch.ones(2)})
            finally:
                manager.close()

            self.assertEqual(1, len(received))
            self.assertTrue(
                os.listdir(received[0]),
                f"callback was handed {received[0]!r}, which holds no shards",
            )

    def test_a_finished_hf_export_is_a_valid_checkpoint(self) -> None:
        # A final HF export keeps the backend's metadata in its nested "sharded"
        # directory and the consolidated files at the root. Recognising only the
        # backend metadata would mark a finished export abandoned, and the next
        # run's pre-save retention deletes abandoned directories outright.
        manager = TorchCheckpointingManager.__new__(TorchCheckpointingManager)
        manager._storage = mock.Mock(spec=CheckpointStorage)

        for marker in ("metadata.pkl", "model.safetensors.index.json"):
            with self.subTest(marker=marker):
                manager._storage.isfile.side_effect = (
                    lambda path, marker=marker: path.endswith(marker)
                )
                self.assertTrue(manager._is_valid_checkpoint("/tmp/checkpoint/step-5"))

        manager._storage.isfile.side_effect = None
        manager._storage.isfile.return_value = False
        self.assertFalse(manager._is_valid_checkpoint("/tmp/checkpoint/step-5"))

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
            enable=True,
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
                dataloader=None,
                model_parts=[nn.Linear(2, 2)],
                optimizers=_Stateful("optimizer"),
                lr_schedulers=_Stateful("scheduler"),
                states={"train_state": _Stateful("train")},
                sd_adapter=adapter,
                base_folder="/tmp",
                storage_config=storage_config,
            )
            self.assertTrue(manager.save(curr_step=5, last_step=True))

        sync_config = build.call_args_list[1].args[0]
        self.assertEqual(
            "/tmp/checkpoint/step-5/sharded",
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
        sync_config.pre_finalize_callback(str(staged), mock.Mock())
        consolidate.assert_called_once_with(
            "/tmp/checkpoint/step-5/tmp_sharded",
            output_dir="/tmp/checkpoint/step-5",
            item_key=MODEL,
            fqn_to_index_mapping=adapter.fqn_to_index_mapping,
            storage_config=storage_config,
        )
        manager.close()

    def test_native_load_restores_model_and_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_id = os.path.join(base_folder, "checkpoint", "step-5")
            os.makedirs(checkpoint_id)
            with open(os.path.join(checkpoint_id, "metadata.pkl"), "wb"):
                pass
            model = nn.Linear(2, 2, bias=False)
            optimizer = _Stateful("optimizer")
            config = TorchCheckpointingManager.Config(
                enable=True,
                folder="checkpoint",
                keep_latest_k=0,
                initial_load_model_only=False,
                load_only=True,
            )
            manager, backend_manager = self._build_manager(
                config,
                base_folder=base_folder,
                model_parts=[model],
                optimizers=optimizer,
            )
            expected_weight = torch.full_like(model.weight, 3)
            backend_manager.load_result = {
                "weight": expected_weight,
                OPTIMIZER: {"value": "restored"},
            }

            self.assertTrue(manager.load(step=5))

            torch.testing.assert_close(model.weight, expected_weight)
            self.assertEqual("restored", optimizer.value)
            self.assertEqual(checkpoint_id, backend_manager.load_calls[0][0])
            self.assertEqual(set(manager.states), set(backend_manager.load_calls[0][1]))
            self.assertEqual({"strict": True}, backend_manager.load_calls[0][2])
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
                enable=True,
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

    def test_load_latest_step_zero_loads_only_model_state(self) -> None:
        with tempfile.TemporaryDirectory() as base_folder:
            checkpoint_id = os.path.join(base_folder, "checkpoint", "step-0")
            os.makedirs(checkpoint_id)
            with open(os.path.join(checkpoint_id, "metadata.pkl"), "wb"):
                pass
            config = TorchCheckpointingManager.Config(
                enable=True,
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
            enable=True,
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
                enable=True,
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
