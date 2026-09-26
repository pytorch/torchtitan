# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A pipeline rank's saves routed through one activation storage, on gloo: every gradient
stays bitwise, and the rank store's blocks never reach the storage policy."""

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

from torchtitan.distributed.activation_storage import ActivationStorage
from torchtitan.models.kimi_k3.pipeline_parallel.cache import PPRankLocalCache
from torchtitan.models.kimi_k3.pipeline_parallel.layout import infer_block_layout_tables
from torchtitan.models.kimi_k3.pipeline_parallel.stage import (
    _grad_send_wait_points,
    _GradSendWaits,
    AttnResPipelineStage,
)

TOKENS, DIM, LAYERS, BLOCK, MICROBATCHES = 8, 16, 8, 2, 4


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

    def _train_pipeline(self, manager: str | None):
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

        def policy(tensor: torch.Tensor, chunk) -> str:
            asked.append(
                tensor.untyped_storage().data_ptr() in _held_storages(stores[0])
            )
            return "copy"

        storage = None
        if manager == "keep":
            storage = ActivationStorage(torch.device("cpu"), {}, lambda t, c: None)
        elif manager == "move":
            storage = ActivationStorage(
                torch.device("cpu"),
                {"copy": _CopyBackend()},
                policy,
                min_tensor_bytes=0,
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
                stage.set_activation_storage(storage)
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
        for manager in ("keep", "move"):
            grads, losses, storage, asked = self._train_pipeline(manager)
            self.assertEqual(reference.keys(), grads.keys())
            for key, grad in grads.items():
                torch.testing.assert_close(grad, reference[key], rtol=0, atol=0)
            for a, b in zip(losses, ref_losses, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            assert storage is not None
            self.assertFalse(storage._chunks)
            self.assertEqual(storage.stats["pinned_bytes"], 0)
            self.assertGreater(storage.stats["pinned_peak_bytes"], 0)
            if manager == "move":
                self.assertGreater(storage.stats["copy_bytes"], 0)
                self.assertTrue(asked)
                self.assertFalse(any(asked))
