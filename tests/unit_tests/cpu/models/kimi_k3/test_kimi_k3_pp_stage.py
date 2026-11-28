# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The stage's carrier handling, on CPU: assembly, routing, the gradient split, and
the swap of core's stages for AttnRes ones."""

import unittest
from contextlib import contextmanager
from unittest.mock import Mock

import torch
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining._utils import InferenceMode
from torch.distributed.pipelining.schedules import Schedule1F1B, ScheduleInterleaved1F1B
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.models.kimi_k3.pipeline_parallel import _swap_in_attn_res_stages
from torchtitan.models.kimi_k3.pipeline_parallel.cache import PPRankLocalCache
from torchtitan.models.kimi_k3.pipeline_parallel.stage import (
    _assemble_stack,
    _pack_outgoing_delta,
    _split_stack_grad,
    AttnResPipelineStage,
)


class TestCarrier(unittest.TestCase):
    def test_assembly_orders_blocks_and_hands_back_a_leaf(self):
        T, D = 4, 8
        hidden = torch.randn(T, D)
        delta = torch.randn(T, 1, D, requires_grad=True)  # block 2 on the wire
        store = {0: torch.randn(T, D), 1: torch.randn(T, D)}
        stack, order = _assemble_stack(hidden, delta, [2], store)
        self.assertEqual(order, [0, 1, 2])
        self.assertTrue(stack.is_leaf and stack.requires_grad)
        self.assertTrue(torch.equal(stack[:, 0], store[0]))
        self.assertTrue(torch.equal(stack[:, 2], delta[:, 0]))
        empty, order = _assemble_stack(hidden, hidden.new_zeros(T, 0, D), [], {})
        self.assertEqual((tuple(empty.shape), order), ((T, 0, D), []))
        with self.assertRaisesRegex(ValueError, "routing expects"):
            _assemble_stack(hidden, delta, [2, 3], store)

    def test_payload_is_the_routed_columns_of_the_model_stack(self):
        T, D = 4, 8
        stack_out = torch.randn(T, 3, D, requires_grad=True)
        payload = _pack_outgoing_delta(stack_out, [0, 1, 2], [1, 2])
        self.assertEqual(tuple(payload.shape), (T, 2, D))
        self.assertTrue(torch.equal(payload[:, 0], stack_out[:, 1]))
        self.assertTrue(payload.requires_grad)
        self.assertEqual(
            tuple(_pack_outgoing_delta(stack_out, [0, 1, 2], []).shape), (T, 0, D)
        )

    def test_gradient_split_sends_the_received_and_deposits_the_stored(self):
        T, D = 4, 8
        grad_stack = torch.randn(T, 3, D)
        like = torch.zeros(T, D)
        grad_delta, deposits = _split_stack_grad(grad_stack, [0, 1, 2], [2], like)
        self.assertEqual(tuple(grad_delta.shape), (T, 1, D))
        self.assertTrue(grad_delta.is_contiguous())
        self.assertTrue(torch.equal(grad_delta[:, 0], grad_stack[:, 2]))
        self.assertEqual(set(deposits), {0, 1})
        self.assertTrue(torch.equal(deposits[1], grad_stack[:, 1]))
        grad_delta, deposits = _split_stack_grad(None, [0], [0], like)
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
        stage._forward_chunk_states = {0: object()}
        stage.bwd_cache = {}
        stage._order = {0: [0, 1]}
        stage._delta_in = {0: [1]}
        AttnResPipelineStage.backward_one_chunk(stage, 0)
        self.assertEqual(
            (stage._forward_chunk_states, stage._order, stage._delta_in),
            ({}, {}, {}),
        )


def _loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return output.sum()


def _get_mesh(*args, **kwargs):
    return None


def _linear_stage_metadata(num_stages: int) -> dict[int, dict]:
    """Static metadata for ``nn.Linear(2, 2)`` stages with one-token microbatches."""
    value = torch.empty(1, 2, device="meta").requires_grad_()
    return {
        index: {
            "input_args": (value,),
            "output_args": (value,),
            "input_grads": (None,) if index == 0 else (value.detach(),),
            "output_grads": (None,) if index == num_stages - 1 else (value.detach(),),
        }
        for index in range(num_stages)
    }


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
        # The declared metadata reaches the rebuilt stage, so no inference pass runs.
        self.assertFalse(InferenceMode.needs_dynamic(new._user_meta, True))

    @with_comms
    def test_forward_enters_registered_pipeline_context(self):
        """Dist-MoE selects each microbatch's activation slot through this context."""
        entered = []

        @contextmanager
        def forward_context(info):
            entered.append((info.stage_index, info.microbatch_index))
            yield

        stage = AttnResPipelineStage(
            nn.Linear(2, 2),
            0,
            1,
            torch.device("cpu"),
            get_mesh=_get_mesh,
            **_linear_stage_metadata(1)[0],
        )
        handle = stage.register_forward_context(forward_context)
        stage.set_routing(Mock(stage_to_rank={0: 0}), PPRankLocalCache())

        stage.forward_one_chunk(2, (torch.ones(1, 2),))
        handle.remove()
        stage.forward_one_chunk(3, (torch.ones(1, 2),))

        self.assertEqual(entered, [(0, 2)])

    @with_comms
    def test_single_stage_schedule(self):
        old = self._stage(0, 1)
        schedule = Schedule1F1B(old, n_microbatches=1, loss_fn=_loss)
        (new,) = _swap_in_attn_res_stages(schedule, _linear_stage_metadata(1))
        self._assert_rebuilt(old, new)
        self.assertIs(schedule._stage, new)

    @with_comms
    def test_multi_stage_schedule(self):
        old = [self._stage(index, 2) for index in range(2)]
        schedule = ScheduleInterleaved1F1B(old, n_microbatches=2, loss_fn=_loss)
        new = _swap_in_attn_res_stages(schedule, _linear_stage_metadata(2))
        self.assertEqual(len(new), 2)
        for old_stage, new_stage in zip(old, new, strict=True):
            self._assert_rebuilt(old_stage, new_stage)
        self.assertEqual(schedule._stages, new)


if __name__ == "__main__":
    unittest.main()
