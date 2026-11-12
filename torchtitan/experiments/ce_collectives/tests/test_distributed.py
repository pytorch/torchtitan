# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run with pytest or torchrun --standalone --nproc-per-node=4 this_file.py."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from torchtitan.experiments.ce_collectives.collectives import (
    CEAllToAllReduceScatter,
    CEMulticastAllGather,
)
from torchtitan.experiments.ce_collectives.configs import CECollectivesConfig
from torchtitan.experiments.ce_collectives.fsdp import enable_fsdp_ce_collectives


def _check_collective(group: dist.ProcessGroup, device: torch.device) -> None:
    comm = CEMulticastAllGather(group)
    stream = torch.cuda.Stream(device=device)
    rank, world_size = group.rank(), group.size()
    for dtype in (torch.bfloat16, torch.float32, torch.uint8):
        shard_numel = 1 << 16
        with torch.cuda.stream(stream):
            output = comm.allocate(
                (world_size * shard_numel,), dtype=dtype, device=device
            )
            shard = output.narrow(0, rank * shard_numel, shard_numel)

            def initialize(step: int) -> None:
                output.fill_(0)
                shard.fill_(rank + 1 + step * 7)

            initialize(0)
            for _ in range(3):
                comm(output, shard, group)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            comm(output, shard, group, async_op=True)
        for step in range(1, 5):
            with torch.cuda.stream(stream):
                initialize(step)
                graph.replay()
            stream.synchronize()
            expected = torch.arange(
                1 + step * 7,
                1 + step * 7 + world_size,
                device=device,
                dtype=torch.int32,
            ).to(dtype)
            expected = expected[:, None].expand(world_size, shard_numel)
            torch.testing.assert_close(
                output.view_as(expected), expected, rtol=0, atol=0
            )

        if dtype == torch.bfloat16:
            dist.barrier(group=group)
            if rank == 0:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    graph.replay()
                    torch.cuda.synchronize(device)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "trace.json"
                    profile.export_chrome_trace(str(path))
                    events = json.loads(path.read_text())["traceEvents"]
                kernels = [event for event in events if event.get("cat") == "kernel"]
                copies = [
                    event
                    for event in events
                    if event.get("cat") in ("gpu_memcpy", "gpu_memcpy2")
                ]
                assert len(kernels) == 2, kernels
                assert all(
                    "barrier_kernel" in event["name"]
                    and event["args"]["grid"] == [1, 1, 1]
                    for event in kernels
                )
                assert len(copies) == 1, copies
                assert copies[0]["args"]["bytes"] == shard_numel * 2
            else:
                graph.replay()
                torch.cuda.synchronize(device)
            dist.barrier(group=group)
        del graph, shard, output


def _check_fsdp(
    group: dist.ProcessGroup, device: torch.device, num_domains: int = 1
) -> None:
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 16),
    ).to(device)
    reference = copy.deepcopy(model)
    mesh = init_device_mesh("cuda", (group.size(),))
    layers = [module for module in model if isinstance(module, nn.Linear)]
    for layer in layers:
        fully_shard(layer, mesh=mesh, reshard_after_forward=True)
    fully_shard(model, mesh=mesh)
    assert enable_fsdp_ce_collectives(
        model,
        CECollectivesConfig(
            all_gather=num_domains == 1, ranks_per_domain=group.size() // num_domains
        ),
    ) == len(layers)
    for index, layer in enumerate(layers):
        if index + 1 < len(layers):
            layer.set_modules_to_forward_prefetch([layers[index + 1]])
        if index:
            layer.set_modules_to_backward_prefetch([layers[index - 1]])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    static_input = torch.empty(8, 128, device=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))

    def reference_step() -> torch.Tensor:
        reference_optimizer.zero_grad(set_to_none=True)
        expected = reference(static_input)
        expected.square().mean().backward()
        for parameter in reference.parameters():
            dist.all_reduce(parameter.grad, group=group)
            parameter.grad.div_(group.size())
        reference_optimizer.step()
        return expected.detach()

    def fsdp_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        result = model(static_input)
        result.square().mean().backward()
        optimizer.step()
        return result

    def check_parameters() -> None:
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(
                actual.full_tensor(), expected, rtol=2e-5, atol=2e-6
            )
            torch.testing.assert_close(
                actual.grad.full_tensor(), expected.grad, rtol=2e-5, atol=2e-6
            )

    for step in range(3):
        with torch.cuda.stream(stream):
            static_input.normal_(mean=group.rank() * 0.1 + step * 0.03)
            expected = reference_step()
            result = fsdp_step()
        stream.synchronize()
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
        check_parameters()
        print(f"CE_FSDP_EAGER_PASS rank={group.rank()} step={step}", flush=True)

    # Exercise allocator reuse with queued steps and deliberately uneven CPU
    # progress. Per-step host synchronization would hide lifetime mistakes.
    for step in range(8):
        if group.rank() == step % group.size():
            time.sleep(0.005)
        with torch.cuda.stream(stream):
            static_input.normal_(mean=group.rank() * 0.1 + step * 0.03)
            expected = reference_step()
            result = fsdp_step()
    stream.synchronize()
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
    check_parameters()
    print(f"CE_FSDP_QUEUED_PASS rank={group.rank()}", flush=True)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = fsdp_step()
    for step in range(4):
        with torch.cuda.stream(stream):
            static_input.normal_(mean=group.rank() * 0.2 + step * 0.05)
            expected = reference_step()
            graph.replay()
        stream.synchronize()
        torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-6)
        check_parameters()
        print(f"CE_FSDP_GRAPH_PASS rank={group.rank()} step={step}", flush=True)


def _check_reduce_scatter(group: dist.ProcessGroup, device: torch.device) -> None:
    for num_domains in (1, 2):
        comm = CEAllToAllReduceScatter(
            group, ranks_per_domain=group.size() // num_domains
        )
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for chunk_numel in (3, 1024, 65536):
                stream = torch.cuda.Stream(device=device)
                with torch.cuda.stream(stream):
                    output = comm.allocate((chunk_numel,), dtype=dtype, device=device)
                    source = comm.allocate(
                        (group.size() * chunk_numel,), dtype=dtype, device=device
                    )
                    pattern = (torch.arange(source.numel(), device=device) % 13 - 6).to(
                        dtype
                    )
                    for op in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
                        source.copy_(pattern + group.rank() % 5)
                        for _ in range(3):
                            comm(output, source, group, op)
                        expected = source.to(torch.float32, copy=True)
                        dist.all_reduce(expected, group=group)
                        expected = expected.chunk(group.size())[group.rank()]
                        if op == dist.ReduceOp.AVG:
                            expected /= group.size()
                        torch.testing.assert_close(
                            output, expected.to(dtype), rtol=0, atol=0
                        )
                        stream.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            comm(output, source, group, op, async_op=True)
                        for step in range(4):
                            source.copy_(pattern + group.rank() % 5 + step)
                            graph.replay()
                            expected_step = expected + (
                                step if op == dist.ReduceOp.AVG else step * group.size()
                            )
                            torch.testing.assert_close(
                                output, expected_step.to(dtype), rtol=0, atol=0
                            )
                        stream.synchronize()
                print(
                    f"CE_RS_PASS rank={group.rank()} domains={num_domains} dtype={dtype} chunk={chunk_numel}",
                    flush=True,
                )


def _worker() -> None:
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=240))
    try:
        group = dist.group.WORLD
        assert group is not None
        _check_collective(group, device)
        print(f"CE_COLLECTIVE_PASS rank={group.rank()}", flush=True)
        _check_reduce_scatter(group, device)
        _check_fsdp(group, device)
        _check_fsdp(group, device, num_domains=2)
        dist.barrier(group=group)
        print(
            "CE_COLLECTIVES_RESULT="
            + json.dumps(
                {
                    "rank": group.rank(),
                    "world_size": group.size(),
                    "status": "PASS",
                    "torch": torch.__version__,
                    "nccl": torch.cuda.nccl.version(),
                    "gpu": torch.cuda.get_device_name(device),
                }
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
def test_ce_collectives() -> None:
    if torch.cuda.device_count() < 4 or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Requires four multicast-capable NVIDIA GPUs in one NVLink domain")
    pytest.importorskip("cuda.bindings.driver")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=4",
            __file__,
        ],
        check=True,
        timeout=600,
    )


if __name__ == "__main__":
    _worker()
