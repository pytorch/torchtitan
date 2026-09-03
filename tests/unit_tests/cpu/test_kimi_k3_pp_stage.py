# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The stage's carrier handling, on CPU: assembly, routing, the gradient split."""

import unittest

import torch

from torchtitan.models.kimi_k3.pipeline_stage import (
    assemble_stack,
    RankStore,
    route_payload,
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
        payload = route_payload(stack_out, [0, 1, 2], [1, 2])
        self.assertEqual(tuple(payload.shape), (T, 2, D))
        self.assertTrue(torch.equal(payload[:, 0], stack_out[:, 1]))
        self.assertTrue(payload.requires_grad)
        self.assertEqual(
            tuple(route_payload(stack_out, [0, 1, 2], []).shape), (T, 0, D)
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
        store = RankStore()
        store.put(0, 0, torch.zeros(4, 2))
        store.deposit(0, 0, torch.ones(4, 2))
        store.deposit(0, 0, torch.ones(4, 2))
        store.release(0)
        self.assertEqual(store.blocks(0), {})
        self.assertTrue(store.has_deposits(0))
        grad, count = store.collect(0, 0)
        self.assertEqual(count, 2)
        self.assertTrue(torch.equal(grad, torch.full((4, 2), 2.0)))
        self.assertFalse(store.has_deposits(0))
        self.assertEqual(store.collect(0, 0), (None, 0))


if __name__ == "__main__":
    unittest.main()
