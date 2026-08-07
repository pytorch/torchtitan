# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.config.configs import TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import apply_simple_fsdp
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    _distribute_dtensor,
    MixedPrecisionPolicy,
    ReplicateComputation,
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
class TestReplicateComputationUnevenShard(unittest.TestCase):
    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()

    def _run_rms_norm_weight_replicate_compute(self, *, dp: int, tp: int) -> None:
        dim = 256
        dist.init_process_group("fake", rank=0, world_size=dp * tp)
        mesh = init_device_mesh(
            "cuda",
            (dp, tp),
            mesh_dim_names=("dp_shard", "tp"),
        )
        tp_mesh = mesh["tp"]
        dp_mesh = mesh["dp_shard"]

        param = _distribute_dtensor(
            distribute_tensor(
                torch.randn(dim, device="cuda"),
                tp_mesh,
                [Replicate()],
            ),
            dp_mesh,
            [Shard(0)],
        )
        rc = ReplicateComputation(
            dp_mesh,
            (Shard(0),),
            "fully_shard",
            MixedPrecisionPolicy(),
        )

        weight = rc.replicate_compute(param)
        weight_local = weight.to_local() if isinstance(weight, DTensor) else weight
        self.assertEqual(weight_local.shape, (dim,))

        input = torch.randn(4, dim, device="cuda")
        torch.rms_norm(input, (dim,), weight_local, 1e-5)

    def test_rms_norm_weight_unpads_uneven_dp_tp_shard(self):
        for dp, tp in ((3, 2), (9, 8)):
            with self.subTest(dp=dp, tp=tp):
                try:
                    self._run_rms_norm_weight_replicate_compute(dp=dp, tp=tp)
                finally:
                    if dist.is_initialized():
                        dist.destroy_process_group()


if __name__ == "__main__":
    unittest.main()
