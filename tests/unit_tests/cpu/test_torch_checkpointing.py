# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import unittest
from unittest import mock

import torch.nn as nn
from torch_checkpointing.barriers import TCPStoreBarrierConfig
from torch_checkpointing.checkpoint_manager import (
    CheckpointManager as BackendCheckpointManager,
)
from torch_checkpointing.config import AsyncCheckpointSaverConfig
from torch_checkpointing.dtensor_resharder import DTensorResharder

from torchtitan.components.checkpointer import (
    BaseCheckpointManager,
    CheckpointManager,
    MODEL,
    OPTIMIZER,
)
from torchtitan.components.checkpointer.torch_checkpointing import (
    _default_backend_config,
    DEFAULT_TORCH_CHECKPOINTING_BARRIER_TCPSTORE_PORT,
    TorchCheckpointingManager,
)


class _BackendManager:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class TorchCheckpointingManagerTest(unittest.TestCase):
    def _build_manager(
        self, config: TorchCheckpointingManager.Config
    ) -> tuple[TorchCheckpointingManager, _BackendManager]:
        backend_manager = _BackendManager()
        with mock.patch.object(
            BackendCheckpointManager.Config,
            "build",
            return_value=backend_manager,
        ):
            manager = config.build(
                dataloader=None,
                model_parts=[nn.Linear(2, 2)],
                optimizers=object(),
                lr_schedulers=object(),
                states={},
                sd_adapter=None,
                base_folder="/tmp",
            )
        return manager, backend_manager

    def test_config_builds_independent_manager(self) -> None:
        config = TorchCheckpointingManager.Config(
            enable=True,
            keep_latest_k=0,
            initial_load_model_only=False,
        )

        manager, backend_manager = self._build_manager(config)

        self.assertIsInstance(manager, TorchCheckpointingManager)
        self.assertNotIsInstance(manager, CheckpointManager)
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

    @mock.patch.dict("os.environ", {"MASTER_ADDR": "checkpoint-host"})
    def test_default_backend_configuration_owns_schema_and_barrier(self) -> None:
        backend_config = _default_backend_config()

        self.assertIsInstance(backend_config.save, AsyncCheckpointSaverConfig)
        self.assertTrue(backend_config.save.staging_config.use_pinned_memory)
        self.assertEqual(set(backend_config.items), {MODEL, OPTIMIZER})
        for spec in backend_config.items.values():
            self.assertTrue(spec.requires_copy)
            self.assertFalse(spec.required)
            self.assertIsInstance(spec.resharder, DTensorResharder)
        self.assertIsNotNone(backend_config.default)
        self.assertFalse(backend_config.default.requires_copy)
        barrier_config = backend_config.save.writer_config.barrier_config
        self.assertIsInstance(barrier_config, TCPStoreBarrierConfig)
        self.assertEqual(barrier_config.master_address, "checkpoint-host")
        self.assertEqual(
            barrier_config.tcpstore_port,
            DEFAULT_TORCH_CHECKPOINTING_BARRIER_TCPSTORE_PORT,
        )

    def test_legacy_config_has_no_backend_selector(self) -> None:
        config = CheckpointManager.Config()

        self.assertFalse(hasattr(config, "save_backend"))
        self.assertFalse(hasattr(config, "load_backend"))
