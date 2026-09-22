# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The stage's carrier handling, on CPU: assembly, routing, the gradient split, and
the swap of core's stages for AttnRes ones."""

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
from torchtitan.models.kimi_k3.pipeline_parallel.stage import (
    assemble_stack,
    AttnResPipelineStage,
    pack_outgoing_delta,
    PPRankLocalCache,
    split_stack_grad,
)


class TestCarrier(unittest.TestCase):
    def test_assembly_orders_blocks_and_hands_back_a_leaf(self):
        T, D = 4, 8
        hidden = torch.randn(T, D)
        delta = torch.randn(T, 1, D, requires_grad=True)  # block 2 on the wire
        store = {0: torch.randn(T, D), 1: torch.randn(T, D)}
        stack, order = assemble_stack(hidden, delta, [2], store)
        self.assertEqual(order, [0, 1, 2])
        self.assertTrue(stack.is_leaf and stack.requires_grad)
        self.assertTrue(torch.equal(stack[:, 0], store[0]))
        self.assertTrue(torch.equal(stack[:, 2], delta[:, 0]))
        empty, order = assemble_stack(hidden, hidden.new_zeros(T, 0, D), [], {})
        self.assertEqual((tuple(empty.shape), order), ((T, 0, D), []))
        with self.assertRaisesRegex(ValueError, "routing expects"):
            assemble_stack(hidden, delta, [2, 3], store)

    def test_payload_is_the_routed_columns_of_the_model_stack(self):
        T, D = 4, 8
        stack_out = torch.randn(T, 3, D, requires_grad=True)
        payload = pack_outgoing_delta(stack_out, [0, 1, 2], [1, 2])
        self.assertEqual(tuple(payload.shape), (T, 2, D))
        self.assertTrue(torch.equal(payload[:, 0], stack_out[:, 1]))
        self.assertTrue(payload.requires_grad)
        self.assertEqual(
            tuple(pack_outgoing_delta(stack_out, [0, 1, 2], []).shape), (T, 0, D)
        )

    def test_gradient_split_sends_the_received_and_deposits_the_stored(self):
        T, D = 4, 8
        grad_stack = torch.randn(T, 3, D)
        like = torch.zeros(T, D)
        grad_delta, deposits = split_stack_grad(grad_stack, [0, 1, 2], [2], like)
        self.assertEqual(tuple(grad_delta.shape), (T, 1, D))
        self.assertTrue(grad_delta.is_contiguous())
        self.assertTrue(torch.equal(grad_delta[:, 0], grad_stack[:, 2]))
        self.assertEqual(set(deposits), {0, 1})
        self.assertTrue(torch.equal(deposits[1], grad_stack[:, 1]))
        grad_delta, deposits = split_stack_grad(None, [0], [0], like)
        self.assertTrue(torch.equal(grad_delta, torch.zeros(T, 1, D)))
        self.assertEqual(deposits, {})

    def test_store_accumulates_deposits_and_releases_blocks_separately(self):
        store = PPRankLocalCache()
        store.put(0, 0, torch.zeros(4, 2))
        store.deposit(0, 0, torch.ones(4, 2))
        store.deposit(0, 0, torch.ones(4, 2))
        store.release(0)
        self.assertEqual(store.blocks(0), {})
        self.assertTrue(store.has_deposits(0))
        grad, count = store.collect(0, 0)
        self.assertEqual(count, 2)
        assert grad is not None
        self.assertTrue(torch.equal(grad, torch.full((4, 2), 2.0)))
        self.assertFalse(store.has_deposits(0))
        self.assertEqual(store.collect(0, 0), (None, 0))


class TestForwardOnly(unittest.TestCase):
    def test_eval_backward_routes_nothing_and_forgets_the_chunk(self):
        stage = AttnResPipelineStage.__new__(AttnResPipelineStage)
        stage._has_backward = False
        stage.fwd_cache = {0: ((torch.zeros(1),), [])}
        stage.bwd_cache = {}
        stage._order = {0: [0, 1]}
        stage._delta_in = {0: [1]}
        AttnResPipelineStage.backward_one_chunk(stage, 0)
        self.assertEqual((stage.fwd_cache, stage._order, stage._delta_in), ({}, {}, {}))


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
