# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
End-to-end validation for per-direction PP communicators wired into
``torch.distributed.pipelining.PipelineStage`` (``p2p_per_direction=True``).

It runs a real 4-stage ``Schedule1F1B`` pipeline through the actual
``PipelineStage`` + schedule code -- once with the single shared communicator
(default) and once with per-direction communicators -- and checks:

* both runs complete (no deadlock), and
* the per-stage loss and gradient norm are **bitwise identical** between the two
  runs. Per-direction P2P only changes which communicator carries the bytes, not
  the math, so identical numerics is the correctness bar (cf. CLAUDE.md:
  non-computation changes must produce identical loss).

Needs >= 4 GPUs. Run directly or via pytest:

    python tests/test_pp_per_direction_pipeline_stage.py
    pytest tests/test_pp_per_direction_pipeline_stage.py
"""

from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn as nn

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

DIM = 16
MICRO_BATCH = 4
N_MICROBATCHES = 4
GLOBAL_BATCH = MICRO_BATCH * N_MICROBATCHES
MASTER_PORT = 29613


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(DIM, DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin(x))


def _loss_fn(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(output, target)


def _worker(rank: int, world_size: int, backend_mode: str, per_direction: bool, q):
    import torch.distributed as dist
    from torch.distributed.pipelining import PipelineStage
    from torch.distributed.pipelining.schedules import Schedule1F1B

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(MASTER_PORT)

    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)

    if backend_mode == "torchcomms":
        import torchcomms  # noqa: F401
        import torch.distributed.config as dist_config

        dist_config.use_torchcomms = True

    # device_id binds the default PG so split_group can duplicate the PP comm.
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, device_id=device
    )

    # Deterministic, reproducible init/data across both runs.
    torch.manual_seed(1234 + rank)
    torch.use_deterministic_algorithms(True, warn_only=False)

    module = _Layer().to(device)
    example = torch.randn(MICRO_BATCH, DIM, device=device)
    stage = PipelineStage(
        module,
        rank,
        world_size,
        device,
        input_args=(example,),
        group=dist.group.WORLD,
        p2p_per_direction=per_direction,
    )
    schedule = Schedule1F1B(stage, n_microbatches=N_MICROBATCHES, loss_fn=_loss_fn)

    # Fixed input on the first stage, fixed target on the last stage.
    gen = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(GLOBAL_BATCH, DIM, device=device, generator=gen)
    target = torch.randn(GLOBAL_BATCH, DIM, device=device, generator=gen)

    losses: list[torch.Tensor] = []
    if rank == 0:
        schedule.step(x)
    elif rank == world_size - 1:
        schedule.step(target=target, losses=losses)
    else:
        schedule.step()
    torch.cuda.synchronize()

    # Report this stage's loss (last stage only) and gradient norm.
    loss_val = (
        float(torch.stack([loss_.detach() for loss_ in losses]).sum().item())
        if losses
        else float("nan")
    )
    grad_norm = float(
        torch.norm(
            torch.stack(
                [
                    p.grad.detach().norm()
                    for p in module.parameters()
                    if p.grad is not None
                ]
            )
        ).item()
        if any(p.grad is not None for p in module.parameters())
        else 0.0
    )
    q.put((rank, loss_val, grad_norm))

    dist.barrier()
    dist.destroy_process_group()


def _run(backend_mode: str, world_size: int, per_direction: bool, timeout_s: float):
    """Spawn a pipeline run; return ('completed'|'deadlock'|'crashed', results)."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [
        ctx.Process(
            target=_worker, args=(r, world_size, backend_mode, per_direction, q)
        )
        for r in range(world_size)
    ]
    for p in procs:
        p.start()

    results: dict[int, tuple[float, float]] = {}
    deadline = time.time() + timeout_s
    while time.time() < deadline and len(results) < world_size:
        try:
            rank, loss_val, grad_norm = q.get(timeout=1.0)
            results[rank] = (loss_val, grad_norm)
        except Exception:
            if all(not p.is_alive() for p in procs) and q.empty():
                break

    # Give procs a moment to reach their final barrier / exit.
    end = time.time() + 10
    while time.time() < end and any(p.is_alive() for p in procs):
        time.sleep(0.5)

    alive = [p for p in procs if p.is_alive()]
    if alive:
        status = "deadlock"
    elif all(p.exitcode == 0 for p in procs):
        status = "completed"
    else:
        status = "crashed"
    for p in alive:
        p.terminate()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
    return status, results


def _compare(backend_mode: str, world_size: int = 4, timeout_s: float = 90.0):
    base_status, base = _run(backend_mode, world_size, False, timeout_s)
    fix_status, fix = _run(backend_mode, world_size, True, timeout_s)
    return base_status, base, fix_status, fix


# ------------------------------- pytest -------------------------------------- #


def _require_n_gpus(n: int):
    import pytest

    if not torch.cuda.is_available() or torch.cuda.device_count() < n:
        pytest.skip(f"needs >= {n} GPUs")


def _assert_match(base, fix):
    assert set(base) == set(fix), f"rank sets differ: {set(base)} vs {set(fix)}"
    for rank in base:
        b_loss, b_gn = base[rank]
        f_loss, f_gn = fix[rank]
        # Bitwise identical: per-direction P2P is a non-computation change.
        if b_loss == b_loss:  # not NaN (only the last stage reports a loss)
            assert b_loss == f_loss, f"rank {rank} loss {b_loss} != {f_loss}"
        assert b_gn == f_gn, f"rank {rank} grad_norm {b_gn} != {f_gn}"


def test_per_direction_matches_single_comm_nccl():
    _require_n_gpus(4)
    base_status, base, fix_status, fix = _compare("nccl")
    assert base_status == "completed", f"baseline run did not complete: {base_status}"
    assert fix_status == "completed", f"per-direction run did not complete: {fix_status}"
    _assert_match(base, fix)


def test_per_direction_matches_single_comm_torchcomms():
    _require_n_gpus(4)
    try:
        import torchcomms  # noqa: F401
    except ImportError:
        import pytest

        pytest.skip("torchcomms not installed")
    base_status, base, fix_status, fix = _compare("torchcomms")
    assert base_status == "completed", f"baseline run did not complete: {base_status}"
    assert fix_status == "completed", f"per-direction run did not complete: {fix_status}"
    _assert_match(base, fix)


if __name__ == "__main__":
    backend = sys.argv[1] if len(sys.argv) > 1 else "nccl"
    bs, base, fs, fix = _compare(backend)
    print(f"==> [{backend}] single-comm : {bs.upper()}  {base}", flush=True)
    print(f"==> [{backend}] per-direction: {fs.upper()}  {fix}", flush=True)
    ok = bs == "completed" and fs == "completed"
    if ok:
        try:
            _assert_match(base, fix)
            print("==> numerics MATCH (bitwise)", flush=True)
        except AssertionError as e:
            print(f"==> numerics MISMATCH: {e}", flush=True)
            ok = False
    sys.exit(0 if ok else 1)
