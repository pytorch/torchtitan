# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The stage's carrier handling, on CPU: routing, the rank store, and the swap of
core's stages for AttnRes ones."""

import unittest

import torch
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import Schedule1F1B, ScheduleInterleaved1F1B
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.models.kimi_k3.pipeline_parallel import _swap_in_attn_res_stages
from torchtitan.models.kimi_k3.pipeline_parallel.cache import PPRankLocalCache
from torchtitan.models.kimi_k3.pipeline_parallel.stage import (
    _outgoing_blocks,
    AttnResPipelineStage,
)


class TestCarrier(unittest.TestCase):
    def test_payload_is_the_routed_blocks_of_the_model_list(self):
        blocks = [torch.randn(4, 8, requires_grad=True) for _ in range(3)]
        payload = _outgoing_blocks(blocks, [0, 1, 2], [1, 2])
        self.assertEqual(len(payload), 2)
        self.assertIs(payload[0], blocks[1])
        self.assertIs(payload[1], blocks[2])
        self.assertEqual(_outgoing_blocks(blocks, [0, 1, 2], []), [])
        with self.assertRaisesRegex(ValueError, "routing expects"):
            _outgoing_blocks(blocks, [0, 1], [1])
        with self.assertRaisesRegex(ValueError, "not among"):
            _outgoing_blocks(blocks, [0, 1, 2], [3])

    def test_store_releases_blocks_one_at_a_time(self):
        store = PPRankLocalCache()
        for b in range(3):
            store.put(0, b, torch.full((4, 2), float(b)))
        store.release(0, [1])
        self.assertEqual(sorted(store.blocks(0)), [0, 2])
        store.release(0, [0, 2])
        self.assertEqual(store.blocks(0), {})
        store.put(1, 0, torch.zeros(4, 2))
        store.release(1)
        self.assertEqual(store.blocks(1), {})

    def test_store_accumulates_deposits_and_counts_empty_ones(self):
        store = PPRankLocalCache()
        first = torch.ones(4, 2)
        store.deposit(0, 0, first)
        store.deposit(0, 0, torch.ones(4, 2))
        store.deposit(0, 0, None)
        self.assertTrue(torch.equal(first, torch.ones(4, 2)))
        self.assertTrue(store.has_deposits(0))
        grad, count = store.collect(0, 0)
        self.assertEqual(count, 3)
        assert grad is not None
        self.assertTrue(torch.equal(grad, torch.full((4, 2), 2.0)))
        self.assertFalse(store.has_deposits(0))
        self.assertEqual(store.collect(0, 0), (None, 0))
        store.deposit(0, 1, None)
        self.assertTrue(store.has_deposits(0))
        self.assertEqual(store.collect(0, 1), (None, 1))


class TestForwardOnly(unittest.TestCase):
    def test_eval_backward_routes_nothing_and_forgets_the_chunk(self):
        stage = AttnResPipelineStage.__new__(AttnResPipelineStage)
        stage._has_backward = False
        stage.fwd_cache = {0: ((torch.zeros(1),), [])}
        stage.bwd_cache = {}
        stage._held_in = {0: [0]}
        stage._delta_in = {0: [1]}
        stage._fwd_send_works = {}
        stage._activations = None
        AttnResPipelineStage.backward_one_chunk(stage, 0)
        self.assertEqual(
            (stage.fwd_cache, stage._held_in, stage._delta_in), ({}, {}, {})
        )


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return output.sum()


def _get_mesh(*args, **kwargs):
    return None


class TestStageSwap(DTensorTestBase):
    """Core's stages are rebuilt as AttnRes stages on the schedule, one rank on gloo."""

    @property
    def device_type(self) -> str:
        return "cpu"

    @property
    def world_size(self) -> int:
        return 1

    @staticmethod
    def _stage(index: int, num_stages: int) -> PipelineStage:
        return PipelineStage(
            nn.Linear(2, 2),
            index,
            num_stages,
            torch.device("cpu"),
            get_mesh=_get_mesh,
        )

    def _assert_rebuilt(self, old: PipelineStage, new: AttnResPipelineStage) -> None:
        self.assertIsInstance(new, AttnResPipelineStage)
        self.assertIsNot(new, old)
        # The same module object, so the model parts core returned stay valid.
        self.assertIs(new.submod, old.submod)
        self.assertEqual(
            (new.stage_index, new.num_stages, new.device),
            (old.stage_index, old.num_stages, old.device),
        )
        self.assertIs(new.group, old.group)
        self.assertIs(new._mesh_cache._get_mesh_cb, old._mesh_cache._get_mesh_cb)
        self.assertEqual(new.stage_index_to_group_rank, old.stage_index_to_group_rank)

    @with_comms
    def test_single_stage_schedule(self):
        old = self._stage(0, 1)
        schedule = Schedule1F1B(old, n_microbatches=1, loss_fn=_loss)
        (new,) = _swap_in_attn_res_stages(schedule)
        self._assert_rebuilt(old, new)
        self.assertIs(schedule._stage, new)

    @with_comms
    def test_multi_stage_schedule(self):
        old = [self._stage(index, 2) for index in range(2)]
        schedule = ScheduleInterleaved1F1B(old, n_microbatches=2, loss_fn=_loss)
        new = _swap_in_attn_res_stages(schedule)
        self.assertEqual(len(new), 2)
        for old_stage, new_stage in zip(old, new, strict=True):
            self._assert_rebuilt(old_stage, new_stage)
        self.assertEqual(schedule._stages, new)


if __name__ == "__main__":
    unittest.main()
