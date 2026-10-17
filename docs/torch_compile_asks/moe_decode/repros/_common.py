# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the MoE decode repros: Qwen3.5-35B-A3B routed-expert shapes and CUDA-graph timers."""
import statistics

import torch

NUM_EXPERTS, DIM, HIDDEN, TOP_K = 256, 2048, 512, 8  # E, D, F, routes per token
HBM_BYTES_PER_US = 8e6  # 8 TB/s


def header():
    print(
        f"torch {torch.__version__} ({torch.version.git_version[:10]}), {torch.cuda.get_device_name()}"
    )


def routed_inputs(num_tokens, out_features, in_features, seed=0):
    """Expert weights ``(E, N, K)``, routed rows sorted by expert ``(T*TOP_K, K)``, ``offs`` and counts per expert.

    Routing is uniform random top-8 of 256 experts, so T=34 touches ~171 experts with ~1 row each.
    """
    gen = torch.Generator(device="cuda").manual_seed(seed + num_tokens)
    expert_ids = (
        torch.rand(num_tokens, NUM_EXPERTS, device="cuda", generator=gen)
        .topk(TOP_K, dim=-1)
        .indices
    )
    counts = torch.bincount(expert_ids.flatten(), minlength=NUM_EXPERTS)
    weight = (
        torch.randn(
            NUM_EXPERTS, out_features, in_features, device="cuda", generator=gen
        )
        * 0.02
    ).bfloat16()
    rows = torch.randn(
        int(counts.sum()), in_features, device="cuda", generator=gen
    ).bfloat16()
    return weight, rows, counts.cumsum(0).to(torch.int32), counts


def weight_roofline_us(counts, out_features, in_features):
    """Time to read the weights of every expert that got at least one row, at 8 TB/s."""
    return int((counts > 0).sum()) * out_features * in_features * 2 / HBM_BYTES_PER_US


def _capture(fn):
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def graph_us(fn, replays=100, rounds=3):
    """CUDA-graph replay time of ``fn`` in us: min over rounds of the median replay."""
    graph = _capture(fn)
    medians = []
    for _ in range(rounds):
        times = []
        for _ in range(replays):
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end) * 1000)
        medians.append(statistics.median(times))
    return min(medians)


def graph_kernels(fn, replays=20):
    """Kernels of one CUDA-graph replay of ``fn``, in launch order: ``[(min us over replays, name)]``."""
    graph = _capture(fn)
    graph.replay()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    kernels = sorted(
        (e.start_ns(), e.duration_ns() / 1000, e.name())
        for e in prof.profiler.kineto_results.events()
        if e.device_type() == torch.autograd.DeviceType.CUDA
    )
    per_replay = len(kernels) // replays
    return [
        (min(kernels[r * per_replay + i][1] for r in range(replays)), kernels[i][2])
        for i in range(per_replay)
    ]


def counting_sort_dispatch(x_TD, scores_TK, expert_ids_TK):
    """TorchTitan's inference MoE dispatch: sort the T*K routes by expert and gather their token rows.

    Returns the routed rows ``(T*K, D)``, each row's token, each row's score, and each route's row (for combine).

    Example:
        >>> # ids [[2, 0], [0, 1]] -> expert order routes 1, 2, 3, 0; rows of routes 3, 0, 1, 2
    """
    expert_ids_N = expert_ids_TK.reshape(-1)
    one_hot_NE = (
        expert_ids_N[:, None] == torch.arange(NUM_EXPERTS, device=x_TD.device)
    ).to(torch.int32)
    rank_NE = torch.cumsum(one_hot_NE, dim=0)
    counts_E = rank_NE[-1]
    starts_E = torch.cumsum(counts_E, dim=0) - counts_E
    row_of_route_N = (
        starts_E[expert_ids_N] + rank_NE.gather(1, expert_ids_N[:, None]).squeeze(1) - 1
    ).long()
    route_of_row_N = torch.empty_like(row_of_route_N)
    route_of_row_N[row_of_route_N] = torch.arange(
        expert_ids_N.numel(), device=x_TD.device
    )
    token_of_row_N = route_of_row_N // TOP_K
    return (
        x_TD[token_of_row_N],
        token_of_row_N,
        scores_TK.reshape(-1)[route_of_row_N],
        row_of_route_N,
    )


def router_inputs(num_tokens, seed=0):
    """Tokens ``(T, D)`` bf16, top-8 scores and expert ids ``(T, 8)`` from random logits."""
    gen = torch.Generator(device="cuda").manual_seed(seed + num_tokens)
    x_TD = torch.randn(num_tokens, DIM, device="cuda", generator=gen).bfloat16()
    scores_TK, expert_ids_TK = torch.rand(
        num_tokens, NUM_EXPERTS, device="cuda", generator=gen
    ).topk(TOP_K, dim=-1)
    return x_TD, scores_TK, expert_ids_TK
