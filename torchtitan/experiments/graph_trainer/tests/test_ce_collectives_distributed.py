# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run with torchrun --standalone --nproc-per-node=2 this_file.py."""

import json
import os
import subprocess
import sys
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

from torchtitan.experiments.ce_collectives.collectives import (
    CEAllToAllReduceScatter,
    CEWorkspace,
)
from torchtitan.experiments.ce_collectives.configs import CECollectivesConfig
from torchtitan.experiments.graph_trainer.ce_collectives import ce_collectives_pass
from torchtitan.experiments.graph_trainer.simple_fsdp import FSDP_PARAM_FQNS_META


def _graph(group, kind, *, workspace_size_mib=10):
    graph = torch.fx.Graph()
    source = graph.placeholder("source")
    out = graph.placeholder("out")
    c10d = torch.ops._c10d_functional
    if kind == "out":
        node = graph.call_function(
            c10d.all_gather_into_tensor_out.default,
            (source, group.size(), group.group_name),
            {"out": out},
        )
    elif kind == "all_gather":
        node = graph.call_function(
            c10d.all_gather_into_tensor.default,
            (source, group.size(), group.group_name),
        )
    else:
        node = graph.call_function(
            c10d.reduce_scatter_tensor.default,
            (source, kind, group.size(), group.group_name),
        )
    node.meta["custom"] = {FSDP_PARAM_FQNS_META: ("layers.0.weight",)}
    wait = graph.call_function(c10d.wait_tensor.default, (node,))
    graph.output(wait)
    return ce_collectives_pass(
        torch.fx.GraphModule({}, graph),
        config=CECollectivesConfig(),
        workspace_size_mib=workspace_size_mib,
    )


def _check_capture_memory_reuse(group, device):
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    shard_numel = 1024 * 1024
    ce = torch.ops.graph_trainer_ce
    wait = torch.ops._c10d_functional.wait_tensor.default

    def step():
        total = torch.zeros((), device=device)
        for _ in range(16):
            source = torch.full((shard_numel,), group.rank() + 1.0, device=device)
            pending = ce.all_gather_into_tensor(source, group.size(), group.group_name)
            del source
            # Reallocation before wait must not overwrite an in-flight input.
            torch.empty(shard_numel, device=device).fill_(-1)
            total.add_(wait(pending).sum())
            del pending
            source = torch.full(
                (shard_numel * group.size(),), group.rank() + 1.0, device=device
            )
            pending = ce.reduce_scatter_tensor(
                source,
                "avg",
                group.size(),
                group.group_name,
                ranks_per_domain=group.size(),
            )
            del source
            torch.empty(shard_numel * group.size(), device=device).fill_(-1)
            total.add_(wait(pending).sum())
            del pending
        return total

    expected = 16 * shard_numel * (group.size() + 1) ** 2 / 2
    with torch.cuda.stream(stream):
        for _ in range(2):
            assert step().item() == expected
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=stream):
        output = step()
    graph_bytes = sum(
        segment["total_size"]
        for segment in torch.cuda.memory_snapshot()
        if segment["segment_pool_id"] == capture.pool()
    )
    # Memory must reflect live buffers, not the sum across all 32 collectives.
    assert graph_bytes < 128 * 1024**2, graph_bytes
    with torch.cuda.stream(stream):
        for _ in range(10):
            capture.replay()
            assert output.item() == expected
    print(
        f"CE_CAPTURE_MEMORY rank={group.rank()} graph_bytes={graph_bytes}", flush=True
    )


def _check_reduction_matches_full_workspace(group, device):
    comm = CEAllToAllReduceScatter(group, ranks_per_domain=group.size())
    workspace = CEWorkspace(group, size_bytes=256 * 1024)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for shard_numel in (3, 65536, 262147):
                source = torch.randn(
                    group.size() * shard_numel, device=device, dtype=dtype
                )
                expected = torch.empty(shard_numel, device=device, dtype=dtype)
                actual = torch.empty_like(expected)
                for reduction in (dist.ReduceOp.SUM, dist.ReduceOp.AVG):
                    comm(expected, source, group, reduction)
                    comm(actual, source, group, reduction, workspace=workspace)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        comm(expected, source, group, reduction)
                        comm(actual, source, group, reduction, workspace=workspace)
                    for _ in range(3):
                        source.add_(0.125)
                        graph.replay()
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    stream.synchronize()
    print(f"CE_CHUNK_NUMERICS rank={group.rank()} status=PASS", flush=True)


def _worker():
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=timedelta(seconds=180))
    group = dist.group.WORLD
    assert group is not None
    try:
        for kind in ("all_gather", "out", "sum", "avg"):
            for compiled in (False, True):
                for dtype in (torch.float32, torch.bfloat16):
                    gm = _graph(group, kind, workspace_size_mib=32 if compiled else 10)
                    fn = torch.compile(gm, fullgraph=True) if compiled else gm
                    stream = torch.cuda.Stream(device=device)
                    stream.wait_stream(torch.cuda.current_stream(device))
                    # Cross workspace boundaries and exercise a partial final chunk.
                    shard_numel = 3 * 1024**2 + 17
                    is_gather = kind in ("all_gather", "out")
                    numel = shard_numel if is_gather else shard_numel * group.size()
                    with torch.cuda.stream(stream):
                        out = torch.empty(
                            shard_numel * group.size(), device=device, dtype=dtype
                        )
                        source = (
                            out.narrow(0, shard_numel * group.rank(), shard_numel)
                            if kind == "out"
                            else torch.empty(numel, device=device, dtype=dtype)
                        )
                        pattern = (torch.arange(numel, device=device) % 17).to(dtype)

                        def initialize(step, source=source, pattern=pattern):
                            source.copy_(pattern + group.rank() + step)

                        def expected(
                            step,
                            is_gather=is_gather,
                            dtype=dtype,
                            shard_numel=shard_numel,
                            pattern=pattern,
                            kind=kind,
                        ):
                            if is_gather:
                                return (
                                    torch.arange(
                                        group.size(), device=device, dtype=dtype
                                    )
                                    .add(step)
                                    .repeat_interleave(shard_numel)
                                ) + pattern.repeat(group.size())
                            shard_pattern = pattern.chunk(group.size())[group.rank()]
                            value = (
                                group.size() * step
                                + group.size() * (group.size() - 1) / 2
                            )
                            if kind == "avg":
                                return shard_pattern + value / group.size()
                            return shard_pattern * group.size() + value

                        for step in range(3):
                            initialize(step)
                            result = fn(source, out)
                            torch.testing.assert_close(
                                result, expected(step), rtol=0, atol=0
                            )
                        # Queued calls exercise scratch reuse while ranks and streams
                        # make independent progress, without per-step host waits.
                        queued = []
                        for step in range(3, 15):
                            initialize(step)
                            queued.append(fn(source, out).clone())
                        for step, result in enumerate(queued, 3):
                            torch.testing.assert_close(
                                result, expected(step), rtol=0, atol=0
                            )
                    stream.synchronize()
                    capture = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(capture, stream=stream):
                        result = fn(source, out)
                    for step in range(15, 25):
                        with torch.cuda.stream(stream):
                            initialize(step)
                            capture.replay()
                            torch.testing.assert_close(
                                result, expected(step), rtol=0, atol=0
                            )
                    stream.synchronize()
                    print(
                        f"CE_GRAPH_PASS rank={group.rank()} kind={kind} compiled={compiled} dtype={dtype}",
                        flush=True,
                    )
        _check_capture_memory_reuse(group, device)
        _check_reduction_matches_full_workspace(group, device)
        dist.barrier()
        print(
            "CE_GRAPH_RESULT=" + json.dumps({"rank": group.rank(), "status": "PASS"}),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
def test_ce_graph_collectives():
    if torch.cuda.device_count() < 2 or torch.cuda.get_device_capability() < (9, 0):
        pytest.skip("Requires two multicast-capable NVIDIA GPUs")
    pytest.importorskip("cuda.bindings.driver")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            __file__,
        ],
        check=True,
        timeout=600,
    )


if __name__ == "__main__":
    _worker()
