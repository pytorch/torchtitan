# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile

import grain.python as grain
import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchtitan.components.data.balancing import document_attention_cost
from torchtitan.components.data.global_balance import (
    GlobalPlanConfig,
    GlobalPlanIterDataset,
    plan_window,
)
from torchtitan.components.data.types import TokenizedTrainingMicrobatch

T = 64  # tokens per row


def _row(seed: int) -> TokenizedTrainingMicrobatch:
    """A row of T tokens split into 1-3 documents; tokens encode the seed."""
    rng = np.random.default_rng(seed)
    cuts = sorted(
        rng.choice(np.arange(1, T), size=int(rng.integers(0, 3)), replace=False)
    )
    bounds = [0, *cuts, T]
    positions = torch.cat(
        [torch.arange(b - a) for a, b in zip(bounds[:-1], bounds[1:])]
    )
    tokens = torch.full((T,), seed, dtype=torch.int64) + torch.arange(T)
    return TokenizedTrainingMicrobatch(
        input=tokens,
        labels=tokens + 1,
        positions=positions,
        padding_mask=torch.zeros(T, dtype=torch.bool),
        loss_token_counts=torch.tensor([T]),
        routing_token_counts=torch.tensor([T]),
    )


def _rows_for(dp_rank: int, n: int) -> list[TokenizedTrainingMicrobatch]:
    return [_row(1000 * dp_rank + i) for i in range(n)]


def test_plan_window_properties():
    rng = np.random.default_rng(0)
    for n, w in [(1, 4), (4, 8), (8, 8), (24, 32)]:
        costs = rng.random((n, w))
        assign = plan_window(costs)
        flat = [src for step in assign for src in step]
        assert sorted(flat) == [
            (s, i) for s in range(n) for i in range(w)
        ]  # a permutation
        order = sorted(
            ((costs[s, i], s, i) for s in range(n) for i in range(w)),
            key=lambda x: (-x[0], x[1], x[2]),
        )
        for k, step in enumerate(assign):
            group = {(s, i) for _, s, i in order[k * n : (k + 1) * n]}
            assert set(step) == group  # step k holds the k-th cost group
            owners = {s for s, _ in group}
            kept = sum(1 for d, (s, _) in enumerate(step) if s == d)
            assert kept == len(owners)  # as many rows stay local as possible


def _worker(rank, world, dp, init_file, out_dir, windows, w):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    cp = world // dp
    dp_rank = rank // cp
    rows = _rows_for(dp_rank, windows * w)
    parent = grain.MapDataset.source(rows).to_iter_dataset()
    ds = GlobalPlanIterDataset(
        parent,
        config=GlobalPlanConfig(window_size=w),
        dp_rank=dp_rank,
        dp_world_size=dp,
    )
    it = iter(ds)
    got = [next(it) for _ in range(windows * w)]
    torch.save(
        [(g.input, g.labels, g.positions, g.padding_mask) for g in got],
        os.path.join(out_dir, f"r{rank}.pt"),
    )
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("dp,cp", [(4, 2), (3, 1)])
def test_exchange_matches_plan(dp, cp):
    windows, w = 2, 4
    world = dp * cp
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(
            _worker,
            args=(world, dp, os.path.join(d, "init"), d, windows, w),
            nprocs=world,
            join=True,
        )
        got = {
            r: torch.load(os.path.join(d, f"r{r}.pt"), weights_only=True)
            for r in range(world)
        }
    src = {s: _rows_for(s, windows * w) for s in range(dp)}
    for win in range(windows):
        costs = np.array(
            [
                [document_attention_cost(src[s][win * w + i]) for i in range(w)]
                for s in range(dp)
            ]
        )
        assign = plan_window(costs)
        for k, step in enumerate(assign):
            for dpr, (s, i) in enumerate(step):
                exp = src[s][win * w + i]
                for c in range(cp):
                    inp, lab, pos, pad = got[dpr * cp + c][win * w + k]
                    assert torch.equal(inp, exp.input) and inp.dtype == exp.input.dtype
                    assert torch.equal(lab, exp.labels)
                    assert torch.equal(pos, exp.positions)
                    assert torch.equal(pad, exp.padding_mask)
