# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A pipeline rank's saves under a CheckpointPolicy, on gloo: every gradient stays bitwise
whether the saves stay, go to host memory, are parked on another rank or are moved as a plan
over the profiled step picks, and the rank store's blocks never reach the policy."""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.distributed.activation_storage import (
    _load_transfer_engine,
    ActivationStorage,
    cpu_offload_all,
    RemoteBackend,
)
from torchtitan.models.kimi_k3.pipeline_parallel.activations import (
    BackwardPrefetch,
    compute_actions,
    PPMemoryConfig,
    PPMemoryController,
)
from torchtitan.models.kimi_k3.pipeline_parallel.cache import PPRankLocalCache
from torchtitan.models.kimi_k3.pipeline_parallel.layout import infer_block_layout_tables
from torchtitan.models.kimi_k3.pipeline_parallel.stage import (
    _grad_send_wait_points,
    _GradSendWaits,
    AttnResPipelineStage,
)

TOKENS, DIM, LAYERS, BLOCK, MICROBATCHES = 8, 16, 8, 2, 4


def _has_mooncake() -> bool:
    try:
        _load_transfer_engine()
    except ImportError:
        return False
    return True


class _CopyBackend:
    def put(self, tensor: torch.Tensor, stream) -> torch.Tensor:
        return tensor.clone()

    def get(self, payload: torch.Tensor, out: torch.Tensor, stream) -> None:
        out.copy_(payload)

    def free(self, payload: torch.Tensor) -> None:
        pass


class _Layer(nn.Module):
    def __init__(self, layer: int) -> None:
        super().__init__()
        self.layer = layer
        self.w = nn.Linear(DIM, DIM, dtype=torch.float64)

    def forward(self, hidden: torch.Tensor, blocks: list[torch.Tensor]):
        if self.layer % BLOCK == 0:
            blocks = [*blocks, hidden]
        stack = torch.stack(blocks, dim=1)
        weights = torch.softmax(stack @ self.w.weight[0], dim=1)
        mixed = (weights.unsqueeze(-1) * stack).sum(1)
        return torch.tanh(self.w(mixed)) + hidden, blocks


class _Stage(nn.Module):
    def __init__(self, layers: list[int], *, first: bool, last: bool) -> None:
        super().__init__()
        self.first, self.last = first, last
        self.layers = nn.ModuleDict(
            {
                str(layer): checkpoint_wrapper(
                    _Layer(layer), checkpoint_impl=CheckpointImpl.NO_REENTRANT
                )
                for layer in layers
            }
        )

    def forward(self, hidden: torch.Tensor, blocks: list[torch.Tensor] | None = None):
        if self.first:
            blocks = []
        assert blocks is not None
        for layer in self.layers.values():
            hidden, blocks = layer(hidden, blocks)
        if self.last:
            return hidden.sum(-1)
        return hidden, blocks


def _held_storages(store: PPRankLocalCache) -> set[int]:
    return {
        block.untyped_storage().data_ptr()
        for mb in range(MICROBATCHES)
        for block in store.blocks(mb).values()
    }


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(output, target)


class TestPipelineActivationStorage(DTensorTestBase):
    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def world_size(self) -> int:
        return 2

    def _train_pipeline(self, saves: str | None):
        torch.manual_seed(0)
        split = [[0, 1], [2, 3], [4, 5], [6, 7]]
        modules = [
            _Stage(layers, first=s == 0, last=s == 3) for s, layers in enumerate(split)
        ]
        mine = list(range(self.rank, 4, self.world_size))
        stages = [
            AttnResPipelineStage(modules[s], s, 4, torch.device("cpu")) for s in mine
        ]
        schedule = ScheduleInterleaved1F1B(
            list(stages), n_microbatches=MICROBATCHES, loss_fn=_loss, scale_grads=False
        )
        stage_to_rank = dict(stages[0].stage_index_to_group_rank)
        layout = infer_block_layout_tables(
            stage_to_rank=stage_to_rank,
            n_layers=LAYERS,
            layers_per_block=BLOCK,
            layer_to_stage={
                layer: s for s, layers in enumerate(split) for layer in layers
            },
        )
        asked: list[bool] = []
        stores: list[PPRankLocalCache] = []

        def policy(tensor: torch.Tensor, chunk) -> CheckpointPolicy:
            asked.append(
                tensor.untyped_storage().data_ptr() in _held_storages(stores[0])
            )
            if saves == "offload" or (saves == "remote" and self.rank == 1):
                return CheckpointPolicy.MUST_CPU_OFFLOAD
            return CheckpointPolicy.PREFER_SAVE

        storage = prefetch = None
        if saves is not None:
            backends = {"host": _CopyBackend()}
            if saves == "remote":
                backends["remote"] = RemoteBackend(
                    stages[0].group,
                    dests={1: 0},
                    spans={1: 1 << 24},
                    staging_bytes=1 << 22,
                    device=torch.device("cpu"),
                )
            storage = ActivationStorage(
                torch.device("cpu"),
                policy,
                backends,
                route=(lambda chunk: "remote") if saves == "remote" else None,
                min_tensor_bytes=0,
            )
            prefetch = BackwardPrefetch(
                storage, compute_actions(schedule.pipeline_order[self.rank])
            )
        store = PPRankLocalCache(storage)
        stores.append(store)
        waits = _GradSendWaits(
            _grad_send_wait_points(schedule.pipeline_order, stage_to_rank, self.rank)
        )
        for stage in stages:
            stage.set_routing(
                layout, store, wait_sends_at_backward=True, grad_send_waits=waits
            )
            if storage is not None:
                storage.register_stage(stage.stage_index, stage.submod.layers)
                stage.set_activation_storage(storage, prefetch)
        inputs = torch.randn(MICROBATCHES * TOKENS, DIM, dtype=torch.float64)
        targets = torch.randn(MICROBATCHES * TOKENS, dtype=torch.float64)
        losses: list[torch.Tensor] = []
        if 3 in mine:
            schedule.step(target=targets, losses=losses)
        else:
            schedule.step(inputs)
        grads = {
            (stage.stage_index, name): p.grad.clone()
            for stage in stages
            for name, p in stage.submod.named_parameters()
        }
        return grads, [loss.detach() for loss in losses], storage, asked

    @with_comms
    def test_routed_saves_leave_every_gradient_bitwise(self):
        reference, ref_losses, _, _ = self._train_pipeline(None)
        for saves in ("keep", "offload"):
            grads, losses, storage, asked = self._train_pipeline(saves)
            self.assertEqual(reference.keys(), grads.keys())
            for key, grad in grads.items():
                torch.testing.assert_close(grad, reference[key], rtol=0, atol=0)
            for a, b in zip(losses, ref_losses, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert storage is not None
            self.assertFalse(storage._chunks)
            self.assertEqual(storage.stats["pinned_bytes"], 0)
            self.assertGreater(storage.stats["pinned_peak_bytes"], 0)
            self.assertTrue(asked)
            self.assertFalse(any(asked))
            if saves == "offload":
                self.assertGreater(storage.stats["host_bytes"], 0)
            else:
                self.assertEqual(storage.stats["host_bytes"], 0)

    @unittest.skipUnless(_has_mooncake(), "needs mooncake-transfer-engine")
    @with_comms
    def test_saves_parked_on_another_rank_leave_every_gradient_bitwise(self):
        reference, ref_losses, _, _ = self._train_pipeline(None)
        grads, losses, storage, _ = self._train_pipeline("remote")
        for key, grad in grads.items():
            torch.testing.assert_close(grad, reference[key], rtol=0, atol=0)
        for a, b in zip(losses, ref_losses, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert storage is not None
        self.assertFalse(storage._chunks)
        if self.rank == 1:
            self.assertGreater(storage.stats["remote_bytes"], 0)

    def _train_planned(self, memory: PPMemoryConfig | None, steps: int = 2):
        torch.manual_seed(0)
        split = [[0, 1], [2, 3], [4, 5], [6, 7]]
        modules = [
            _Stage(layers, first=s == 0, last=s == 3) for s, layers in enumerate(split)
        ]
        mine = list(range(self.rank, 4, self.world_size))
        stages = [
            AttnResPipelineStage(modules[s], s, 4, torch.device("cpu")) for s in mine
        ]
        schedule = ScheduleInterleaved1F1B(
            list(stages), n_microbatches=MICROBATCHES, loss_fn=_loss, scale_grads=False
        )
        stage_to_rank = dict(stages[0].stage_index_to_group_rank)
        layout = infer_block_layout_tables(
            stage_to_rank=stage_to_rank,
            n_layers=LAYERS,
            layers_per_block=BLOCK,
            layer_to_stage={
                layer: s for s, layers in enumerate(split) for layer in layers
            },
        )
        storage = controller = None
        if memory is not None:
            storage = ActivationStorage(
                torch.device("cpu"),
                cpu_offload_all({LAYERS - 1})
                if memory.cpu_offload != "none"
                else lambda tensor, chunk: CheckpointPolicy.PREFER_SAVE,
                {"host": _CopyBackend()},
                min_tensor_bytes=0,
            )
            controller = PPMemoryController(
                memory,
                storage,
                group=stages[0].group,
                rank=self.rank,
                orders=schedule.pipeline_order,
                device=torch.device("cpu"),
                skip_layers={LAYERS - 1},
            )
            controller.attach(schedule)
        store = PPRankLocalCache(storage)
        waits = _GradSendWaits(
            _grad_send_wait_points(schedule.pipeline_order, stage_to_rank, self.rank)
        )
        for stage in stages:
            stage.set_routing(
                layout, store, wait_sends_at_backward=True, grad_send_waits=waits
            )
            if storage is not None:
                storage.register_stage(stage.stage_index, stage.submod.layers)
                stage.set_activation_storage(storage, controller)
        history = []
        for _ in range(steps):
            inputs = torch.randn(MICROBATCHES * TOKENS, DIM, dtype=torch.float64)
            targets = torch.randn(MICROBATCHES * TOKENS, dtype=torch.float64)
            losses: list[torch.Tensor] = []
            if 3 in mine:
                schedule.step(target=targets, losses=losses)
            else:
                schedule.step(inputs)
            grads = {
                (stage.stage_index, name): p.grad.clone()
                for stage in stages
                for name, p in stage.submod.named_parameters()
            }
            for stage in stages:
                stage.submod.zero_grad(set_to_none=True)
            history.append((grads, [loss.detach() for loss in losses]))
        return history, storage, controller

    @with_comms
    def test_planned_moves_leave_every_gradient_bitwise(self):
        reference, _, _ = self._train_planned(None)
        history, storage, controller = self._train_planned(
            PPMemoryConfig(cpu_offload="planned", target_gib=0.0)
        )
        for (grads, losses), (ref_grads, ref_losses) in zip(
            history, reference, strict=True
        ):
            self.assertEqual(ref_grads.keys(), grads.keys())
            for key, grad in grads.items():
                torch.testing.assert_close(grad, ref_grads[key], rtol=0, atol=0)
            for a, b in zip(losses, ref_losses, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert storage is not None and controller is not None
        assert controller.plan is not None
        self.assertFalse(storage._chunks)
        self.assertEqual(storage.off_device_bytes, 0)
        self.assertTrue([key for key in controller.plan.backend if key[0] == self.rank])
        self.assertGreater(storage.stats["host_bytes"], 0)

    @unittest.skipUnless(_has_mooncake(), "needs mooncake-transfer-engine")
    @with_comms
    def test_balance_alone_parks_on_the_other_rank_and_leaves_every_gradient_bitwise(
        self,
    ):
        reference, _, _ = self._train_planned(None)
        # Rank 0 peaks at 5 KiB and rank 1 at 3 KiB; their mean leaves no room for a pool's slack.
        history, storage, controller = self._train_planned(
            PPMemoryConfig(balance=True, target_gib=4.5 * 2**10 / 2**30)
        )
        for (grads, losses), (ref_grads, ref_losses) in zip(
            history, reference, strict=True
        ):
            self.assertEqual(ref_grads.keys(), grads.keys())
            for key, grad in grads.items():
                torch.testing.assert_close(grad, ref_grads[key], rtol=0, atol=0)
            for a, b in zip(losses, ref_losses, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert storage is not None and controller is not None
        assert controller.plan is not None
        self.assertTrue(controller.plan.dests)
        self.assertEqual(set(controller.plan.backend.values()), {"remote"})
        self.assertEqual(storage.stats["host_bytes"], 0)
        if self.rank in controller.plan.dests:
            self.assertGreater(storage.stats["remote_bytes"], 0)
