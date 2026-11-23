# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-process gloo tests for microbatch balancing with group_size > 1."""

import os
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import grain.python as grain
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchtitan.components.data import balancing
from torchtitan.components.data.balancing import (
    MicrobatchBalancingConfig,
    MicrobatchBalancingIterDataset,
    plan_window,
)

WINDOW = 4
WINDOWS = 2


@dataclass
class _Microbatch:
    """A multimodal-like microbatch: tensors of several dtypes and shapes,
    nested ``model_kwargs`` and non-tensor fields."""

    input: torch.Tensor
    padding_mask: torch.Tensor
    cost: float
    model_kwargs: dict[str, Any] = field(default_factory=dict)


def _microbatch(dp_rank: int, i: int) -> _Microbatch:
    seed = 1000 * dp_rank + i
    g = torch.Generator().manual_seed(seed)
    tokens = 16 + seed % 7  # variable shapes
    kwargs: dict[str, Any] = {"source": f"r{dp_rank}-{i}"}
    if seed % 2:
        kwargs["pixel_values"] = torch.rand(seed % 3 + 1, 3, 4, 4, generator=g)
        kwargs["grid"] = [torch.tensor([1, 2, 2]), (seed, None)]
    return _Microbatch(
        input=torch.randint(0, 1 << 40, (1, tokens), generator=g),
        padding_mask=torch.rand(1, tokens, generator=g) < 0.2,
        cost=float(torch.rand(1, generator=g)),
        model_kwargs=kwargs,
    )


def _local_microbatches(dp_rank: int) -> list[_Microbatch]:
    return [_microbatch(dp_rank, i) for i in range(WINDOW * WINDOWS)]


def _cost(mb: _Microbatch) -> float:
    return mb.cost


def _worker(rank, world, dp, group_size, init_file, out_dir, resume_at):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world
    )
    dp_rank = rank // (world // dp)
    config = MicrobatchBalancingConfig(
        window_size=WINDOW, group_size=group_size, cost_fn=_cost
    )
    parent = grain.MapDataset.source(_local_microbatches(dp_rank)).to_iter_dataset()
    ds = MicrobatchBalancingIterDataset(
        parent, config=config, dp_rank=dp_rank, dp_world_size=dp
    )
    it = iter(ds)
    got = [next(it) for _ in range(resume_at)]
    if resume_at:
        state = it.get_state()
        it = iter(ds)
        it.set_state(state)
    got += [next(it) for _ in range(WINDOW * WINDOWS - resume_at)]
    torch.save(got, os.path.join(out_dir, f"r{rank}.pt"))
    dist.barrier()
    dist.destroy_process_group()


def _assert_same(a: Any, b: Any) -> None:
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor)
        assert a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same(x, y)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            _assert_same(a[key], b[key])
    elif isinstance(a, _Microbatch):
        assert isinstance(b, _Microbatch)
        _assert_same(vars(a), vars(b))
    else:
        assert a == b


@pytest.mark.parametrize(
    "dp,cp,group_size,resume_at",
    [(4, 2, None, 0), (4, 2, 2, 0), (3, 1, None, 0), (4, 1, 2, 6)],
)
def test_ranks_receive_planned_microbatches(dp, cp, group_size, resume_at):
    world = dp * cp
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(
            _worker,
            args=(world, dp, group_size, os.path.join(d, "init"), d, resume_at),
            nprocs=world,
            join=True,
        )
        got = {
            r: torch.load(os.path.join(d, f"r{r}.pt"), weights_only=False)
            for r in range(world)
        }
    size = group_size or dp
    src = {s: _local_microbatches(s) for s in range(dp)}
    for w in range(WINDOWS):
        for g0 in range(0, dp, size):
            costs = [
                [src[g0 + s][w * WINDOW + i].cost for i in range(WINDOW)]
                for s in range(size)
            ]
            for k, step in enumerate(plan_window(costs)):
                for d, (s, i) in enumerate(step):
                    expected = src[g0 + s][w * WINDOW + i]
                    for c in range(cp):
                        _assert_same(expected, got[(g0 + d) * cp + c][w * WINDOW + k])


def _timeout_worker(rank, init_file):
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    balancing.FIRST_WINDOW_TIMEOUT = timedelta(seconds=2)
    parent = grain.MapDataset.source(_local_microbatches(rank)).to_iter_dataset()
    ds = MicrobatchBalancingIterDataset(
        parent,
        config=MicrobatchBalancingConfig(
            window_size=WINDOW, group_size=None, cost_fn=_cost
        ),
        dp_rank=rank,
        dp_world_size=2,
    )
    if rank == 0:  # rank 1 never reaches the first window
        with pytest.raises(RuntimeError, match="timed out"):
            next(iter(ds))
    os._exit(0)  # skip destroy_process_group on the broken group


def test_exchange_times_out_with_a_clear_error():
    with tempfile.TemporaryDirectory() as d:
        mp.spawn(_timeout_worker, args=(os.path.join(d, "init"),), nprocs=2, join=True)
