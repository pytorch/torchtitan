# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import os
import random
import tempfile

import grain.python as grain
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchtitan.components.data.balancing import (
    assign_sorted_bins_keep_local,
    GlobalBalancer,
    MicrobatchBalancingIterDataset,
    plan_window,
)

WINDOW = 4
NUM_ROWS = 2 * WINDOW


def _row(dp_rank: int, i: int) -> dict:
    """A multimodal-like row: tensors of several shapes and dtypes, and a tuple."""
    return {
        "id": (dp_rank, i),
        "cost": random.Random(100 * dp_rank + i).random(),
        "pixels": torch.full((i + 1, 3), 100.0 * dp_rank + i),
        "mask": torch.arange(i + 2) % 2 == 0,
    }


def _cost(row: dict) -> float:
    return row["cost"]


def _dataset(dp_rank: int, dp: int) -> MicrobatchBalancingIterDataset:
    balancer = GlobalBalancer.Config(window_size=WINDOW, cost_fn=_cost).build(
        dp_rank=dp_rank, dp_world_size=dp
    )
    rows = [_row(dp_rank, i) for i in range(NUM_ROWS)]
    source = grain.MapDataset.source(rows).to_iter_dataset()
    return MicrobatchBalancingIterDataset(source, balancer=balancer)


def _worker(rank, world, dp, tmp, resume_at):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp}/init", rank=rank, world_size=world
    )
    ds = _dataset(rank // (world // dp), dp)
    it = iter(ds)
    got = list(itertools.islice(it, resume_at))
    if resume_at:
        state = it.get_state()
        it = iter(ds)
        it.set_state(state)
    got += list(itertools.islice(it, NUM_ROWS - resume_at))
    torch.save(got, f"{tmp}/r{rank}.pt")
    dist.destroy_process_group()


def test_plan_window():
    # Step 0's two heaviest rows (9 and 8) are both on rank 0: 8 moves to rank 1.
    # Step 1's (4 and 3) are both on rank 1: 3 moves to rank 0.
    assert plan_window([[9, 1, 8, 2], [3, 2, 1, 4]]) == [
        [(0, 0), (0, 2)],
        [(1, 0), (1, 3)],
        [(0, 3), (1, 1)],
        [(0, 1), (1, 2)],
    ]


def test_keep_local_assignment():
    # Sorted bins 9, 8 | 4, 3 held by ranks 0, 0 | 1, 1: each row keeps one bin
    # on its owner and moves the other to the free rank.
    assert assign_sorted_bins_keep_local([0, 0, 1, 1], dp_degree=2) == [[0, 1], [3, 2]]


@pytest.mark.parametrize("dp,cp,resume_at", [(4, 2, 0), (2, 1, 5)])
def test_each_rank_trains_its_planned_rows(dp, cp, resume_at):
    world = dp * cp
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(world, dp, tmp, resume_at), nprocs=world)
        got = [torch.load(f"{tmp}/r{r}.pt", weights_only=False) for r in range(world)]

    moved = 0
    for w in range(0, NUM_ROWS, WINDOW):
        costs = [[_row(s, w + i)["cost"] for i in range(WINDOW)] for s in range(dp)]
        for k, step in enumerate(plan_window(costs)):
            for d, (s, i) in enumerate(step):
                expected = _row(s, w + i)
                moved += s != d
                for c in range(cp):  # CP peers get the same row
                    row = got[d * cp + c][w + k]
                    assert row["id"] == expected["id"]
                    assert torch.equal(row["pixels"], expected["pixels"])
                    assert torch.equal(row["mask"], expected["mask"])
    assert moved > 0


def _dead_peer_worker(rank, tmp):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp}/init", rank=rank, world_size=2
    )
    ds = _dataset(rank, 2)
    if rank == 0:
        with pytest.raises(RuntimeError, match="GlobalBalancer exchange failed"):
            next(iter(ds))
    os._exit(0)  # rank 1 dies before the first window; skip process group cleanup


def test_dead_peer_raises():
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_dead_peer_worker, args=(tmp,), nprocs=2)
