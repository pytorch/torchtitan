# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import random
import tempfile

import grain.python as grain
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchtitan.components.data.balancing import (
    GlobalBalancer,
    MicrobatchBalancingIterDataset,
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


def _global_rank(dp_rank: int, cp_rank: int, dp: int, cp: int, cp_outer: bool) -> int:
    # Either mesh order: the balancer only sees the DP group's ranks.
    return cp_rank * dp + dp_rank if cp_outer else dp_rank * cp + cp_rank


def _worker(rank, dp, cp, cp_outer, tmp, resume_at):
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp}/init", rank=rank, world_size=dp * cp
    )
    dp_rank, cp_rank = (rank % dp, rank // dp) if cp_outer else divmod(rank, cp)
    # What the trainer passes: the global ranks of this rank's DP group.
    dp_ranks = [_global_rank(d, cp_rank, dp, cp, cp_outer) for d in range(dp)]
    balancer = GlobalBalancer.Config(window_size=WINDOW, cost_fn=_cost).build(
        dp_rank=dp_rank, dp_world_size=dp, dp_ranks=dp_ranks
    )
    rows = [_row(dp_rank, i) for i in range(NUM_ROWS)]
    source = grain.MapDataset.source(rows).to_iter_dataset()
    ds = MicrobatchBalancingIterDataset(source, balancer=balancer)
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
    assert GlobalBalancer._plan_window([[9, 1, 8, 2], [3, 2, 1, 4]]) == [
        [(0, 0), (0, 2)],
        [(1, 0), (1, 3)],
        [(0, 3), (1, 1)],
        [(0, 1), (1, 2)],
    ]


@pytest.mark.parametrize(
    "dp,cp,cp_outer,resume_at",
    [(4, 2, False, 0), (2, 2, True, 0), (2, 1, False, 5)],
)
def test_each_rank_trains_its_planned_rows(dp, cp, cp_outer, resume_at):
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(dp, cp, cp_outer, tmp, resume_at), nprocs=dp * cp)
        got = [torch.load(f"{tmp}/r{r}.pt", weights_only=False) for r in range(dp * cp)]

    moved = 0
    for w in range(0, NUM_ROWS, WINDOW):
        costs = [[_row(s, w + i)["cost"] for i in range(WINDOW)] for s in range(dp)]
        for k, step in enumerate(GlobalBalancer._plan_window(costs)):
            for d, (s, i) in enumerate(step):
                expected = _row(s, w + i)
                moved += s != d
                for c in range(cp):  # CP peers get the same row
                    row = got[_global_rank(d, c, dp, cp, cp_outer)][w + k]
                    assert row["id"] == expected["id"]
                    assert torch.equal(row["pixels"], expected["pixels"])
                    assert torch.equal(row["mask"], expected["mask"])
    assert moved > 0
