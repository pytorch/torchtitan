# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.graph_trainer.make_fx_tracer import trace_train_step

_HAS_TORCHAO = importlib.util.find_spec("torchao") is not None


def _count_op(gm: torch.fx.GraphModule, target) -> int:
    return sum(
        1
        for node in gm.graph.nodes
        if node.op == "call_function" and node.target is target
    )


class TestApplySimpleFSDPSingleRank(unittest.TestCase):
    """Verify simple_fsdp's MixedPrecisionPolicy actually casts params at NGPU=1."""

    def setUp(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12358",
                world_size=1,
                rank=0,
            )

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    @patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
    def test_param_cast_to_bf16_at_ngpu_1(self):
        """With ``mixed_precision_param=bfloat16``, the parametrized weight must
        yield bf16 — and a forward must run in bf16 — even when fsdp /
        dp_replicate / ep are all disabled. Without the unconditional
        simple_fsdp wrap, parameters silently stay in fp32 on a single GPU and
        any downstream bf16-only kernel (e.g. MXFP8) breaks.
        """
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=1,
        )
        training = TrainingConfig(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        )

        model = nn.Linear(8, 8)
        self.assertEqual(model.weight.dtype, torch.float32)

        model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)

        # Parametrization replaces ``weight`` access with a bf16 cast via
        # ``redistribute(forward_dtype=...)``.
        self.assertEqual(model.weight.dtype, torch.bfloat16)

        # Underlying storage stays in fp32 (the cast is applied per forward),
        # confirming this is true mixed precision rather than a one-shot
        # downcast that would lose master-weight precision.
        self.assertEqual(model._parameters["weight"].dtype, torch.float32)

        # End-to-end: forward against the parametrized weight produces bf16
        # activations.
        x = torch.randn(2, 8, dtype=torch.bfloat16)
        y = model(x)
        self.assertEqual(y.dtype, torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
@unittest.skipUnless(_HAS_TORCHAO, "torchao required")
class TestMXFP8GraphCaptureSingleRank(unittest.TestCase):
    """Verify MXFP8's __torch_function__ rewrites fire under simple_fsdp's
    parametrization, so the captured FX graph contains ``aten._scaled_mm``
    for MXFP8-converted linears instead of plain ``aten.mm``.
    """

    def setUp(self):
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="tcp://localhost:12359",
                world_size=1,
                rank=0,
            )
        torch.manual_seed(0)

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_scaled_mm_in_graph(self):
        """A forward through MXFP8Linear + simple_fsdp must capture
        aten._scaled_mm and no plain aten.mm for that linear."""
        from torchtitan.components.quantization.mx import MXFP8Linear

        # MXFP8 scaled_mm requires dims divisible by 16. Keep shapes small.
        in_features, out_features = 64, 64
        batch = 32

        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=1,
        )
        training = TrainingConfig(
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        )

        class _Wrap(nn.Module):
            def __init__(self, mxfp8_linear):
                super().__init__()
                self.lin = mxfp8_linear

            def forward(self, x):
                return self.lin(x)

        mxfp8_linear = (
            MXFP8Linear.Config(
                in_features=in_features, out_features=out_features, bias=False
            )
            .build()
            .cuda()
        )
        model = _Wrap(mxfp8_linear)
        model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)

        x = torch.randn(batch, in_features, dtype=torch.bfloat16, device="cuda")

        def forward_fn(model, x):
            return model(x)

        traced = trace_train_step(forward_fn)(model, x)

        num_scaled_mm = _count_op(traced.gm, torch.ops.aten._scaled_mm.default)
        num_mm = _count_op(traced.gm, torch.ops.aten.mm.default)

        self.assertGreaterEqual(
            num_scaled_mm,
            1,
            "Expected at least one aten._scaled_mm in the captured graph — "
            "MXFP8's __torch_function__ did not fire at F.linear.",
        )
        self.assertEqual(
            num_mm,
            0,
            f"Expected no plain aten.mm in the captured graph, found {num_mm}. "
            "simple_fsdp's parametrization is stripping the MXFP8 wrapper "
            "before F.linear so the graph falls back to plain mm.",
        )


if __name__ == "__main__":
    unittest.main()
