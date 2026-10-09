# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU tests, also runnable with torchrun on real CUDA/NCCL process groups.

CHUNKED_LOSS_TEST_DEVICE=cuda torchrun --standalone --nproc-per-node=2 \
    -m torchtitan.experiments.graph_trainer.tests.test_chunked_loss_model
"""

import os
import tempfile
import unittest
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import TestCase

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    disable_active_parametrization,
    local_gradient_accumulation,
    MixedPrecisionPolicy,
)
from torchtitan.experiments.graph_trainer.tests.test_chunked_loss import (
    _WeightedTwoOutputLoss,
)


class TestChunkedLossModel(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        device_type = os.environ.get("CHUNKED_LOSS_TEST_DEVICE", "cpu")
        if device_type == "cuda":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        cls.device = (
            torch.device("cuda", torch.cuda.current_device())
            if device_type == "cuda"
            else torch.device("cpu")
        )
        cls.owns_process_group = not dist.is_initialized()
        cls.store_dir = tempfile.TemporaryDirectory()
        if cls.owns_process_group:
            kwargs = {}
            if "RANK" not in os.environ:
                kwargs = dict(
                    init_method=f"file://{cls.store_dir.name}/store",
                    rank=0,
                    world_size=1,
                )
            dist.init_process_group(
                backend="nccl" if device_type == "cuda" else "gloo", **kwargs
            )
        cls.mesh = init_device_mesh(
            device_type, (dist.get_world_size(),), mesh_dim_names=("dp",)
        )

    @classmethod
    def tearDownClass(cls):
        if cls.owns_process_group:
            dist.destroy_process_group()
        cls.store_dir.cleanup()
        super().tearDownClass()

    def _make_case(
        self,
        num_chunks=4,
        dtype=torch.float32,
        bias=False,
        multi=False,
        mesh=None,
        mode="fully_shard",
        reduce_dtype=torch.float32,
    ):
        torch.manual_seed(123)
        head = nn.Linear(8, 16, bias=bias, device=self.device)
        reference = deepcopy(head)
        for module in (head, reference):
            data_parallel(
                module,
                mesh if mesh is not None else self.mesh,
                mode=mode,
                param_shard_placements={},
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=dtype, reduce_dtype=reduce_dtype
                ),
            )
        options = {"num_chunks": num_chunks}
        if multi:
            options["loss_fn"] = _WeightedTwoOutputLoss.Config()
        loss = ChunkedLossWrapperWithParamGrads(
            ChunkedLossWrapperWithParamGrads.Config(**options)
        )
        loss.set_lm_head(head)
        baseline = ChunkedLossWrapper(ChunkedLossWrapper.Config(**options))
        baseline.set_lm_head(reference)
        # Different data per rank exercises a real reduction, not just copies.
        torch.manual_seed(456 + dist.get_rank())
        hidden = torch.randn(16, 8, device=self.device, dtype=dtype)
        labels = torch.randint(0, 16, (16,), device=self.device)
        return head, reference, loss, baseline, hidden, labels

    def test_gradients_and_forward_match_per_chunk_sync(self):
        for dtype in (torch.float32, torch.bfloat16):
            for num_chunks in (1, 4):
                with self.subTest(dtype=dtype, num_chunks=num_chunks):
                    head, reference, loss, baseline, h, y = self._make_case(
                        num_chunks, dtype, bias=True
                    )
                    actual_h = h.detach().requires_grad_()
                    expected_h = h.detach().clone().requires_grad_()
                    value, _ = loss(actual_h, y, 16.0)
                    actual = torch.autograd.grad(value, (actual_h, *head.parameters()))
                    expected_value, _ = baseline(expected_h, y, 16.0)
                    expected_value.backward()
                    expected = (
                        expected_h.grad,
                        *(p.grad for p in reference.parameters()),
                    )
                    self.assertEqual(value, expected_value)
                    for a, b in zip(actual, expected, strict=True):
                        self.assertEqual(a, b, atol=2e-6, rtol=2e-5)
                    self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_scaled_backward_preserves_existing_grads_across_microbatches(self):
        head, reference, loss, baseline, h, y = self._make_case()
        expected_total = None
        for scale in (0.25, 0.75):
            hh = h.detach().clone().requires_grad_()
            value, _ = loss(hh, y)
            (value * scale).backward()
            ref_h = h.detach().clone().requires_grad_()
            reference.zero_grad(set_to_none=True)
            expected_value, _ = baseline(ref_h, y)
            expected_value.backward()
            expected = next(reference.parameters()).grad * scale
            expected_total = (
                expected if expected_total is None else expected_total + expected
            )
            self.assertEqual(hh.grad, ref_h.grad * scale)
            self.assertEqual(next(head.parameters()).grad, expected_total)

    def test_lower_precision_reduction_preserves_forward_weights(self):
        head, _, loss, baseline, h, y = self._make_case(
            dtype=torch.float32, reduce_dtype=torch.bfloat16
        )
        expected_weight = head.weight.detach()
        with local_gradient_accumulation(head):
            self.assertEqual(head.weight, expected_weight, atol=0, rtol=0)
        actual_h = h.detach().requires_grad_()
        expected_h = h.detach().clone().requires_grad_()
        value, _ = loss(actual_h, y, 16.0)
        expected_value, _ = baseline(expected_h, y, 16.0)
        self.assertEqual(value, expected_value, atol=0, rtol=0)
        actual_h_grad = torch.autograd.grad(value, actual_h)[0]
        expected_value.backward()
        self.assertEqual(actual_h_grad, expected_h.grad, atol=0, rtol=0)

    def test_trace_has_one_reduction_and_replays_updated_weights(self):
        for num_chunks in (1, 4, 8):
            with self.subTest(num_chunks=num_chunks):
                head, _, loss, _, h, y = self._make_case(num_chunks, torch.bfloat16)
                h.requires_grad_()

                def step(hidden, labels):
                    value, _ = loss(hidden, labels, 16.0)
                    grads = torch.autograd.grad(value, (hidden, *head.parameters()))
                    # make_fx does not flatten DTensor outputs; trace the local
                    # shards, as GraphTrainer's tracer does for graph outputs.
                    return (value, grads[0], *(g.to_local() for g in grads[1:]))

                gm = make_fx(step)(h, y)
                rs = [
                    n
                    for n in gm.graph.nodes
                    if n.target
                    is torch.ops._c10d_functional.reduce_scatter_tensor.default
                ]
                ag = [
                    n
                    for n in gm.graph.nodes
                    if n.target
                    is torch.ops._c10d_functional.all_gather_into_tensor.default
                ]
                expected_count = int(dist.get_world_size() > 1)
                self.assertEqual(len(rs), expected_count)
                self.assertEqual(len(ag), expected_count)
                for node in rs:
                    self.assertEqual(node.args[0].meta["val"].dtype, torch.float32)
                for node in ag:
                    self.assertEqual(node.args[0].meta["val"].dtype, torch.bfloat16)
                for _ in range(2):
                    expected = step(h, y)
                    actual = gm(h, y)
                    for a, b in zip(actual, expected, strict=True):
                        self.assertEqual(a, b)
                    with torch.no_grad():
                        for p in head.parameters():
                            p.add_(0.03125)
                self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_multi_output_accumulates_before_reduction(self):
        head, reference, loss, baseline, h, y = self._make_case(multi=True)
        inputs = (h.detach().clone().requires_grad_(), (h * 0.5).requires_grad_())
        ref_inputs = tuple(x.detach().clone().requires_grad_() for x in inputs)
        value, _ = loss(inputs, (y, y), 16.0)
        actual = torch.autograd.grad(value, (*inputs, *head.parameters()))
        expected_value, _ = baseline(ref_inputs, (y, y), 16.0)
        expected_value.backward()
        expected = (
            *(x.grad for x in ref_inputs),
            *(p.grad for p in reference.parameters()),
        )
        self.assertEqual(value, expected_value)
        for a, b in zip(actual, expected, strict=True):
            self.assertEqual(a, b, atol=2e-6, rtol=2e-5)

    def test_replicated_and_hybrid_gradient_sync(self):
        meshes = [("replicate", self.mesh)]
        if dist.get_world_size() % 2 == 0:
            meshes.append(
                (
                    "hybrid_shard",
                    init_device_mesh(
                        self.device.type,
                        (2, dist.get_world_size() // 2),
                        mesh_dim_names=("dp_replicate", "dp_shard"),
                    ),
                )
            )
        for mode, mesh in meshes:
            with self.subTest(mode=mode):
                head, reference, loss, baseline, h, y = self._make_case(
                    dtype=torch.bfloat16, mode=mode, mesh=mesh
                )
                h.requires_grad_()
                value, _ = loss(h, y, 16.0)
                actual = torch.autograd.grad(value, (h, *head.parameters()))
                ref_h = h.detach().clone().requires_grad_()
                expected_value, _ = baseline(ref_h, y, 16.0)
                expected_value.backward()
                expected = (ref_h.grad, *(p.grad for p in reference.parameters()))
                self.assertEqual(value, expected_value)
                for a, b in zip(actual, expected, strict=True):
                    self.assertEqual(a, b, atol=2e-6, rtol=2e-5)

    def test_nested_scopes_and_exception_restore_parameter_access(self):
        head, _, loss, _, h, y = self._make_case()
        with local_gradient_accumulation(head) as outer:
            first = head.weight
            with local_gradient_accumulation(head):
                self.assertIsNot(head.weight, first)
            self.assertIs(head.weight, first)
            with disable_active_parametrization():
                with disable_active_parametrization():
                    self.assertIs(head.weight, head._parameters["weight"])
                self.assertIs(head.weight, head._parameters["weight"])
            self.assertIs(head.weight, first)
            self.assertEqual(len(outer.parameters), 1)

        def fail(*args, **kwargs):
            raise RuntimeError("test loss failure")

        loss.loss_fn = fail
        with self.assertRaisesRegex(RuntimeError, "test loss failure"):
            loss(h.requires_grad_(), y)
        self.assertIsNone(loss._local_gradients)
        for p in head.__dict__["_simple_fsdp_parametrizations"].values():
            self.assertIsNone(p.local_gradients)
        self.assertIsNot(head.weight, first)
        self.assertTrue(all(p.grad is None for p in head.parameters()))

    def test_validation_and_frozen_parameters(self):
        head, reference, loss, baseline, h, y = self._make_case()
        with torch.no_grad():
            actual, _ = loss(h, y)
            expected, _ = baseline(h, y)
        self.assertEqual(actual, expected)
        self.assertFalse(actual.requires_grad)
        for module in (head, reference):
            module.requires_grad_(False)
        hh = h.detach().requires_grad_()
        ref_h = h.detach().clone().requires_grad_()
        actual, _ = loss(hh, y)
        expected, _ = baseline(ref_h, y)
        actual.backward()
        expected.backward()
        self.assertEqual(hh.grad, ref_h.grad)
        self.assertTrue(all(p.grad is None for p in head.parameters()))


if __name__ == "__main__":
    unittest.main()
