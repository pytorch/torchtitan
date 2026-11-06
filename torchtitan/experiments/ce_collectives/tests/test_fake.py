# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run with pytest or python this_file.py; needs one CUDA GPU and no peers.

Under a fake process group, the CE collectives must write the same values as
the fake backend's NCCL collectives they replace.
"""

import copy
import subprocess
import sys
from itertools import product

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from torchtitan.experiments.ce_collectives.collectives import (
    CEAllToAllReduceScatter,
    CEMulticastAllGather,
    CEWorkspace,
)
from torchtitan.experiments.ce_collectives.configs import CECollectivesConfig
from torchtitan.experiments.ce_collectives.fsdp import enable_fsdp_ce_collectives

# A nonzero rank checks every shard offset; it is lane 1 of domain 1 when four
# ranks form two domains.
WORLD_SIZE = 4
RANK = 3


def _replay(graph: torch.cuda.CUDAGraph, stream: torch.cuda.Stream) -> None:
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()


def _check_all_gather(group: dist.ProcessGroup, device: torch.device) -> None:
    comm = CEMulticastAllGather(group)
    stream = torch.cuda.Stream(device=device)
    for dtype in (torch.bfloat16, torch.float32, torch.uint8):
        for shard_numel in (3, 1 << 16):
            with torch.cuda.stream(stream):
                output = comm.allocate(
                    (group.size() * shard_numel,), dtype=dtype, device=device
                )
                shard = output.narrow(0, group.rank() * shard_numel, shard_numel)
                pattern = torch.arange(shard_numel, device=device) % 97
                expected = torch.empty_like(output)

                def run(step: int) -> None:
                    output.fill_(1)
                    shard.copy_(pattern + step)
                    dist.all_gather_into_tensor(expected, shard, group=group)

                run(0)
                comm(output, shard, group)
            stream.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                comm(output, shard, group, async_op=True)
            for step in range(1, 3):
                with torch.cuda.stream(stream):
                    run(step)
                _replay(graph, stream)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _check_reduce_scatter(group: dist.ProcessGroup, device: torch.device) -> None:
    stream = torch.cuda.Stream(device=device)
    for num_domains, bounded in product((1, 2), (False, True)):
        comm = CEAllToAllReduceScatter(
            group, ranks_per_domain=group.size() // num_domains
        )
        # Force multiple chunks, including a partial last chunk, so fake
        # receives exercise the same shard offsets as the real transport.
        workspace = (
            CEWorkspace(comm.local_group, size_bytes=group.size() * 127 * 4)
            if bounded
            else None
        )
        for dtype in (torch.float32, torch.bfloat16, torch.float16):
            for chunk_numel in (3, 1024):
                with torch.cuda.stream(stream):
                    output = comm.allocate((chunk_numel,), dtype=dtype, device=device)
                    source = comm.allocate(
                        (group.size() * chunk_numel,), dtype=dtype, device=device
                    )
                    pattern = (torch.arange(source.numel(), device=device) % 13 - 6).to(
                        dtype
                    )
                    expected = torch.empty_like(output)
                for op in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):

                    def run(step: int, op: dist.ReduceOp = op) -> None:
                        source.copy_(pattern + step)
                        output.fill_(float("nan"))
                        # The fake reduce-scatter returns this rank's chunk for
                        # any op; AVG divides it by the group size, as zero
                        # contributions from the peers would.
                        dist.reduce_scatter_tensor(
                            expected, source, op=dist.ReduceOp.SUM, group=group
                        )
                        if op == dist.ReduceOp.AVG:
                            expected.copy_(expected.float() / group.size())

                    with torch.cuda.stream(stream):
                        run(0)
                        comm(output, source, group, op, workspace=workspace)
                    stream.synchronize()
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        comm(
                            output,
                            source,
                            group,
                            op,
                            async_op=True,
                            workspace=workspace,
                        )
                    for step in range(1, 3):
                        with torch.cuda.stream(stream):
                            run(step)
                        _replay(graph, stream)
                        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _check_fsdp(device: torch.device, num_domains: int) -> None:
    """CE and the fake backend's NCCL path must train identically.

    Both force SUM reductions: the fake backend ignores the AVG/PREMUL_SUM ops
    that FSDP would otherwise request, while CE always lets FSDP divide.
    """
    mesh = init_device_mesh("cuda", (WORLD_SIZE,))
    torch.manual_seed(42)
    base = nn.Sequential(
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 128),
        nn.SiLU(),
        nn.Linear(128, 16),
    ).to(device)

    def build(use_ce: bool) -> tuple[nn.Module, torch.optim.Optimizer]:
        model = copy.deepcopy(base)
        layers = [module for module in model if isinstance(module, nn.Linear)]
        for layer in layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=True)
        fully_shard(model, mesh=mesh)
        for index, layer in enumerate(layers):
            if index + 1 < len(layers):
                layer.set_modules_to_forward_prefetch([layers[index + 1]])
            if index:
                layer.set_modules_to_backward_prefetch([layers[index - 1]])
        if use_ce:
            config = CECollectivesConfig(
                all_gather=num_domains == 1,
                ranks_per_domain=WORLD_SIZE // num_domains,
            )
            assert enable_fsdp_ce_collectives(model, config) == len(layers)
        else:
            for layer in layers:
                layer.set_force_sum_reduction_for_comms(True)
        return model, torch.optim.SGD(model.parameters(), lr=0.05)

    expected_model, expected_optimizer = build(use_ce=False)
    model, optimizer = build(use_ce=True)
    static_input = torch.empty(8, 128, device=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))

    def step(model: nn.Module, optimizer: torch.optim.Optimizer) -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        result = model(static_input)
        result.square().mean().backward()
        optimizer.step()
        return result

    def check(result: torch.Tensor, expected: torch.Tensor) -> None:
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        for actual, reference in zip(
            model.parameters(), expected_model.parameters(), strict=True
        ):
            torch.testing.assert_close(
                actual.to_local(), reference.to_local(), rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual.grad.to_local(), reference.grad.to_local(), rtol=0, atol=0
            )
            assert actual.grad.to_local().abs().sum() > 0

    for index in range(3):
        with torch.cuda.stream(stream):
            static_input.normal_(mean=index * 0.03)
            expected = step(expected_model, expected_optimizer).detach()
            result = step(model, optimizer)
        stream.synchronize()
        check(result, expected)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = step(model, optimizer)
    for index in range(3):
        with torch.cuda.stream(stream):
            static_input.normal_(mean=index * 0.05)
            expected = step(expected_model, expected_optimizer).detach()
        _replay(graph, stream)
        check(result, expected)


def _worker() -> None:
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dist.init_process_group("fake", rank=RANK, world_size=WORLD_SIZE)
    try:
        group = dist.group.WORLD
        assert group is not None and dist.get_backend(group) == "fake"
        _check_all_gather(group, device)
        _check_reduce_scatter(group, device)
        for num_domains in (1, 2):
            _check_fsdp(device, num_domains)
        print("CE_FAKE_PASS", flush=True)
    finally:
        dist.destroy_process_group()


def test_ce_collectives_fake_process_group() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Requires one CUDA GPU")
    pytest.importorskip("cuda.bindings.driver")
    # A separate process keeps the global fake process group out of other tests.
    completed = subprocess.run(
        [sys.executable, __file__],
        check=True,
        timeout=300,
        capture_output=True,
        text=True,
    )
    assert "CE_FAKE_PASS" in completed.stdout, completed.stderr


if __name__ == "__main__":
    _worker()
