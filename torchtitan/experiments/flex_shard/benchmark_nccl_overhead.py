# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Benchmark: FlexShard all-gather NCCL overhead.

Measures FlexShard's parametrization-based all-gather path on a simple
multi-layer model.

Requires 2+ GPUs. Run with:
    torchrun --standalone --nproc_per_node=2 -m pytest \
        torchtitan/experiments/flex_shard/benchmark_nccl_overhead.py -xvs
"""

import copy
import unittest

import torch
from torch.distributed.fsdp import DataParallelMeshDims
from torch.testing._internal.common_fsdp import FSDPTest

from torchtitan.distributed import ParallelDims
from torchtitan.experiments.flex_shard import (
    flex_shard,
    lift_params_to_global_spmd_mesh,
    Shard,
)

WARMUP = 5
ITERS = 20
LAYERS = 4
DIM = 256


class BenchmarkNCCLOverhead(FSDPTest):
    """Benchmark per-parameter vs batched all-gather NCCL overhead."""

    def _make_model_and_inputs(self):
        model = torch.nn.Sequential(*[torch.nn.Linear(DIM, DIM) for _ in range(LAYERS)])
        inputs = torch.randn(32, DIM).cuda()
        return model, inputs

    def _benchmark(self, model, inputs):
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(WARMUP):
            optim.zero_grad()
            model(inputs).sum().backward()
            optim.step()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(ITERS):
            optim.zero_grad()
            model(inputs).sum().backward()
            optim.step()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / ITERS

    def _count_nccl_ops(self, model, inputs):
        """Count all_gather ops in one fwd+bwd via torch.profiler."""
        optim = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(WARMUP):
            optim.zero_grad()
            model(inputs).sum().backward()
            optim.step()
        torch.cuda.synchronize()

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
        ) as prof:
            optim.zero_grad()
            model(inputs).sum().backward()
            optim.step()

        count = 0
        for evt in prof.key_averages():
            if "all_gather" in evt.key.lower():
                count += evt.count
        return count

    def test_nccl_overhead(self):
        parallel_dims = ParallelDims(
            dp_shard=-1,
            dp_replicate=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=self.world_size,
        )
        mesh = parallel_dims.get_mesh("fsdp")
        model_base, inputs = self._make_model_and_inputs()

        model = copy.deepcopy(model_base)
        # With only the fsdp axis, this named mesh is the full SPMD mesh.
        lift_params_to_global_spmd_mesh(model, mesh)
        flex_shard(
            model,
            mesh,
            DataParallelMeshDims(shard="fsdp"),
            shard_placement_fn={"*": Shard(0)},
        )
        ms = self._benchmark(model, inputs)
        ops = self._count_nccl_ops(model, inputs)

        print(f"\n{'Mode':<20} {'ms/iter':>10} {'all-gathers':>15}")
        print(f"{'-' * 20} {'-' * 10} {'-' * 15}")
        print(f"{'flex_shard':<20} {ms:>10.3f} {ops:>15}")


if __name__ == "__main__":
    unittest.main()
