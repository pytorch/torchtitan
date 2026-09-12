# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
import shutil
import tempfile
import time
import unittest
from concurrent.futures import Future
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed.checkpoint as dist_checkpoint
import torch.nn as nn
from torch.utils.data import DataLoader

from torchtitan.components.optimizer import LRSchedulersContainer, ParamGroupConfig
from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager
from torchtitan.experiments.torchft.optimizer import TorchFTOptimizersContainer


class FakeOptimizersContainer:
    def __init__(self):
        self._fake_param = torch.tensor([1.0], dtype=torch.float32)

    def state_dict(self):
        return {"fake_param": self._fake_param}

    def load_state_dict(self, sd: dict):
        if "fake_param" in sd:
            self._fake_param = sd["fake_param"]

    def init_cache_state_dict(self):
        pass


class FakeLRSchedulersContainer:
    def state_dict(self):
        return {}

    def load_state_dict(self, sd: dict):
        pass


class FakeDataLoader(DataLoader):
    def __init__(self):
        super().__init__(dataset=[], batch_size=1)

    def state_dict(self):
        return {}

    def load_state_dict(self, sd: dict):
        pass


class DummyFuture:
    def __new__(cls):
        # Return a Mock that mimics Future instead of an instance of this class
        # That allows isinstance(DummyFuture, Future) to pass
        instance = mock.Mock(spec=Future)
        instance.result = mock.Mock()

        return instance


def fake_async_save(*args, **kwargs):
    return DummyFuture()


class DummyFTManager:
    """Mimics TorchFTManager for testing without requiring torchft."""

    def __init__(self, enabled=True, replica_id=0, participating_rank=0):
        self._enabled = enabled
        self.replica_id = replica_id
        if enabled:
            self.manager = mock.MagicMock()
            self.manager.participating_rank.return_value = participating_rank
        else:
            self.manager = None

    @property
    def enabled(self):
        return self._enabled


class TestFTCheckpointManager(unittest.TestCase):
    def setUp(self):
        self.base_temp_dir = tempfile.mkdtemp()
        self.test_folder = os.path.join(self.base_temp_dir, self._testMethodName)
        os.makedirs(self.test_folder, exist_ok=True)
        self.model_parts = [nn.Linear(2, 2)]
        self.states = {"trainer": torch.tensor([1.2347])}
        self.optimizers = FakeOptimizersContainer()
        self.lr_schedulers = FakeLRSchedulersContainer()
        self.data_loader = FakeDataLoader()
        self.ft_manager = DummyFTManager(enabled=True, participating_rank=0)
        self.patcher_group = mock.patch(
            "torch.distributed.new_group", return_value="pg"
        )
        self.patcher_group.start()
        # Patch process group destruction
        self.patcher_destroy = mock.patch("torch.distributed.destroy_process_group")
        self.patcher_destroy.start()

    def tearDown(self):
        self.patcher_group.stop()
        self.patcher_destroy.stop()
        shutil.rmtree(self.base_temp_dir)
        time.sleep(0.1)

    @mock.patch("torch.cuda.Stream")
    @mock.patch.object(
        dist_checkpoint,
        "async_save",
        side_effect=fake_async_save,
    )
    def test_torchft_async_save_calls_maybe_wait_for_saving(
        self,
        mock_async_save,
        mock_cuda_stream,
    ):
        """
        Test that with FT enabled, AsyncMode.ASYNC via FT triggers correct waits.
        """
        config = TorchFTCheckpointManager.Config(
            enable=True,
            async_mode="async",
            folder=self.test_folder,
            interval=1,
            keep_latest_k=0,
            last_save_model_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_only=False,
            enable_ft_dataloader_checkpoints=True,
        )
        manager = TorchFTCheckpointManager(
            config,
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            sd_adapter=None,
            base_folder=self.test_folder,
            ft_manager=self.ft_manager,
        )

        # Initially no future
        self.assertIsNone(manager.save_future)
        manager.save(curr_step=5, last_step=False)
        self.assertIsNotNone(manager.save_future)

        manager.save_future.result.assert_not_called()
        prev_future = manager.save_future
        manager.save(curr_step=6, last_step=False)
        prev_future.result.assert_called_once()
        self.assertIsNotNone(manager.save_future)
        manager.save_future.result.assert_not_called()

        manager.close()

    def _manager(self, participating_rank: int) -> TorchFTCheckpointManager:
        config = TorchFTCheckpointManager.Config(
            enable=True,
            async_mode="disabled",
            folder=self.test_folder,
            interval=1,
            keep_latest_k=0,
            last_save_model_only=False,
            export_dtype="float32",
            exclude_from_loading=[],
            initial_load_path=None,
            initial_load_model_only=False,
            enable_ft_dataloader_checkpoints=True,
        )
        return TorchFTCheckpointManager(
            config,
            dataloader=self.data_loader,
            model_parts=self.model_parts,
            optimizers=self.optimizers,
            lr_schedulers=self.lr_schedulers,
            states=self.states,
            sd_adapter=None,
            base_folder=self.test_folder,
            ft_manager=DummyFTManager(
                enabled=True, participating_rank=participating_rank
            ),
        )

    @mock.patch("torch.cuda.Stream")
    @mock.patch.object(dist_checkpoint, "async_save", side_effect=fake_async_save)
    def test_save_returns_whether_the_full_checkpoint_was_written(
        self,
        mock_async_save,
        mock_cuda_stream,
    ):
        # BaseCheckpointManager.save returns _save's result, so this override has
        # to report a bool. The per-replica dataloader checkpoint is a side
        # channel and does not count as writing the checkpoint.
        with mock.patch.object(dist_checkpoint, "save"):
            participating = self._manager(participating_rank=0)
            self.assertIs(True, participating.save(curr_step=5))
            participating.close()

            bystander = self._manager(participating_rank=1)
            self.assertIs(False, bystander.save(curr_step=5))
            bystander.close()

    def test_load_restores_ft_checkpoint_before_main_checkpoint(self):
        manager = self._manager(participating_rank=0)
        checkpoint_id = manager._create_checkpoint_id(5)
        os.makedirs(checkpoint_id)
        open(os.path.join(checkpoint_id, ".metadata"), "w").close()
        calls = []
        ft_grad_enabled = []

        def load_ft_checkpoint():
            calls.append("ft")
            ft_grad_enabled.append(torch.is_grad_enabled())

        with mock.patch.object(
            manager,
            "_ft_load",
            side_effect=load_ft_checkpoint,
        ), mock.patch.object(
            manager,
            "_load_checkpoint",
            side_effect=lambda *_args, **_kwargs: calls.append("main"),
        ):
            self.assertTrue(manager.load())

        self.assertEqual(["ft", "main"], calls)
        self.assertEqual([False], ft_grad_enabled)
        manager.close()

    def test_disabled_load_does_not_restore_ft_checkpoint(self):
        manager = TorchFTCheckpointManager.__new__(TorchFTCheckpointManager)
        manager.enable = False

        with mock.patch.object(manager, "_ft_load") as ft_load:
            self.assertFalse(manager.load())

        ft_load.assert_not_called()

    def _build_replica(self, replica_id):
        model = nn.Linear(1, 1, bias=False)
        ft_manager = DummyFTManager(replica_id=replica_id)
        ft_manager.use_async_quorum = True
        ft_manager.manager.should_commit.return_value = True
        optimizers = TorchFTOptimizersContainer(
            TorchFTOptimizersContainer.Config(
                implementation="for-loop",
                param_groups=[
                    ParamGroupConfig(
                        pattern=r".*",
                        optimizer_name="AdamW",
                        optimizer_kwargs={"lr": 0.08, "weight_decay": 0.0},
                    )
                ],
            ),
            model_parts=[model],
            ft_manager=ft_manager,
        )
        schedulers = LRSchedulersContainer.Config(warmup_steps=0).build(
            optimizers=optimizers, training_steps=8
        )
        checkpoint = TorchFTCheckpointManager(
            TorchFTCheckpointManager.Config(
                enable=True,
                folder=os.path.join(self.test_folder, str(replica_id)),
                keep_latest_k=0,
                initial_load_model_only=False,
                enable_ft_dataloader_checkpoints=False,
            ),
            dataloader=None,
            model_parts=[model],
            optimizers=optimizers,
            lr_schedulers=schedulers,
            states={},
            sd_adapter=None,
            ft_manager=ft_manager,
        )
        self.addCleanup(checkpoint.close)
        (
            load_state_dict,
            state_dict,
        ) = ft_manager.manager.set_state_dict_fns.call_args.args
        return SimpleNamespace(
            model=model,
            optimizer=optimizers,
            scheduler=schedulers,
            state_dict=state_dict,
            load_state_dict=load_state_dict,
        )

    def test_optimizer_cache_metadata_changes_only_after_explicit_refresh(self):
        replica = self._build_replica(replica_id=0)
        cached_state = replica.optimizer.state_dict()
        cached_tensors = {
            key: cached_state[key]
            for key in (
                "state.weight.step",
                "state.weight.exp_avg",
                "state.weight.exp_avg_sq",
            )
        }

        param_group = replica.optimizer.optimizers[0].param_groups[0]
        param_group["lr"] = 0.04
        param_group["weight_decay"] = 0.1
        exported_state = replica.optimizer.state_dict()

        self.assertIs(exported_state, cached_state)
        self.assertEqual(exported_state["param_groups.weight.lr"], 0.08)
        self.assertEqual(exported_state["param_groups.weight.weight_decay"], 0.0)

        replica.optimizer._refresh_cached_state_dict()

        self.assertIs(replica.optimizer.state_dict(), cached_state)
        self.assertEqual(cached_state["param_groups.weight.lr"], 0.04)
        self.assertEqual(cached_state["param_groups.weight.weight_decay"], 0.1)
        for key, tensor in cached_tensors.items():
            with self.subTest(state_key=key):
                self.assertIs(cached_state[key], tensor)

    def test_joining_replica_restores_healthy_replica_learning_rate(self):
        healthy = self._build_replica(replica_id=0)
        joining = self._build_replica(replica_id=1)
        # Advance beyond the LR captured by a newly initialized optimizer cache.
        healthy.model.weight.grad = torch.ones_like(healthy.model.weight)
        for _ in range(3):
            healthy.optimizer.step()
            healthy.scheduler.step()
        healthy_lr = healthy.optimizer.optimizers[0].param_groups[0]["lr"]
        self.assertNotEqual(
            joining.optimizer.optimizers[0].param_groups[0]["lr"], healthy_lr
        )

        # Use the registered recovery callback and an independent transfer payload.
        payload = copy.deepcopy(healthy.state_dict())
        self.assertEqual(payload["optimizer"]["param_groups.weight.lr"], healthy_lr)
        joining.load_state_dict(payload)

        self.assertEqual(
            joining.optimizer.optimizers[0].param_groups[0]["lr"], healthy_lr
        )

    def test_first_rejoined_update_matches_healthy_replica(self):
        healthy = self._build_replica(replica_id=0)
        joining = self._build_replica(replica_id=1)
        healthy.model.weight.grad = torch.ones_like(healthy.model.weight)
        for _ in range(3):
            healthy.optimizer.step()
            healthy.scheduler.step()

        joining.load_state_dict(copy.deepcopy(healthy.state_dict()))
        joining.model.weight.grad = torch.ones_like(joining.model.weight)
        healthy.optimizer.step()
        joining.optimizer.step()

        torch.testing.assert_close(
            joining.model.weight, healthy.model.weight, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
