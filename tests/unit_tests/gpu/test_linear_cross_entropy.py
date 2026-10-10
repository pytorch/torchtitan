# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common.linear import Linear


def _run_fsdp(rank, world_size, port, capture):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, device_id=torch.device("cuda", rank)
    )
    try:
        mesh = init_device_mesh("cuda", (world_size,))
        torch.manual_seed(42)
        heads = [
            Linear.Config(in_features=256, out_features=1024, bias=True)
            .build()
            .to(device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        ]
        heads[1].load_state_dict(heads[0].state_dict())
        x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(1024, (64,), device="cuda")
        labels[:16] = -100  # Entire ignored chunk must remain finite.
        denominator = torch.tensor(137, device="cuda")
        apply_local_compile(["loss"])
        results = []
        for joint, head in zip((False, True), heads, strict=True):
            fully_shard(
                head,
                mesh=mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
            )
            loss_fn = ChunkedLossWrapper.Config(
                num_chunks=4, linear_cross_entropy=joint
            ).build()
            loss_fn.set_lm_head(head)
            inputs = x.clone().requires_grad_()

            def step():
                head.zero_grad(set_to_none=False)
                if inputs.grad is not None:
                    inputs.grad.zero_()
                loss, _ = loss_fn(inputs, labels, denominator)
                loss.backward()
                return loss

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    loss = step()
            torch.cuda.current_stream().wait_stream(stream)
            if capture:
                del loss
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    loss = step()
                for _ in range(3):
                    graph.replay()
            torch.cuda.synchronize()
            results.append(
                (
                    loss.detach().clone(),
                    inputs.grad.clone(),
                    head.weight.grad.full_tensor(),
                    head.bias.grad.full_tensor(),
                )
            )
            if capture:
                # Release captured NCCL work before destroying the process group.
                del graph
        for reference, actual in zip(*results, strict=True):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, reference, rtol=0.02, atol=1e-5)
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("capture", [False, True])
def test_compiled_projection_ce_fsdp(capture):
    mp.spawn(_run_fsdp, args=(2, get_free_port(), capture), nprocs=2, join=True)
