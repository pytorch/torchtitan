# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils._pytree import tree_map
from torchtitan.components.checkpointer.rng import RNG_STATE
from torchtitan.tools.rng import preserved_rng

from tests.unit_tests.cpu.components.checkpointer.test_rng_checkpoint import (
    draw,
    manager,
    seed,
    step,
)


def _cuda_restart_worker(rank, folder, rendezvous, mode, async_mode):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.set_device(rank)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group(
        "nccl",
        init_method=Path(rendezvous).as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    checkpoint = None
    try:
        seed(50 if mode == "save" else 300)
        checkpoint, model, optimizer, scheduler, loader = manager(
            folder, async_mode=async_mode, device=f"cuda:{rank}"
        )
        seed(50 + rank if mode == "save" else 300 + rank)
        if mode == "save":
            for _ in range(2):
                step(model, optimizer, scheduler, loader)
            checkpoint.save(2)
            checkpoint.maybe_wait_for_staging()
            with preserved_rng():
                draw(f"cuda:{rank}")
        else:
            step(model, optimizer, scheduler, loader)
            if mode == "negative":
                checkpoint.exclude_from_loading = [RNG_STATE]
            checkpoint.load(2)
        results = [step(model, optimizer, scheduler, loader) for _ in range(3)]
        checkpoint.close()
        results = tree_map(
            lambda value: value.cpu() if isinstance(value, torch.Tensor) else value,
            results,
        )
        torch.save(results, Path(folder) / f"{mode}-{rank}.pt")
    finally:
        if checkpoint is not None:
            checkpoint.close()
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
@pytest.mark.parametrize("async_mode", ["disabled", "async", "async_with_pinned_mem"])
def test_cuda_distributed_restart(tmp_path, async_mode):
    for mode in ("save", "resume", "negative"):
        mp.spawn(
            _cuda_restart_worker,
            args=(str(tmp_path), str(tmp_path / f"rdzv-{mode}"), mode, async_mode),
            nprocs=2,
        )
    for rank in range(2):
        results = {
            mode: torch.load(tmp_path / f"{mode}-{rank}.pt", weights_only=False)
            for mode in ("save", "resume", "negative")
        }
        torch.testing.assert_close(results["save"], results["resume"], rtol=0, atol=0)
        assert results["save"][0]["draw"][0] != results["negative"][0]["draw"][0]
