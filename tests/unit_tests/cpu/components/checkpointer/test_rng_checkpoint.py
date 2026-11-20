# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
import random
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from torchtitan.components.checkpointer.base import DATALOADER, OPTIMIZER
from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.checkpointer.rng import RNG_STATE, RngState
from torchtitan.tools.rng import preserved_rng, RngSnapshot


class Loader:
    """Restoration deliberately consumes all global streams, like iterator setup."""

    def __init__(self):
        self.position = 0

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]
        draw()


def draw():
    return random.random(), float(np.random.random()), torch.rand(3)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def assert_draw_equal(left, right):
    assert left[:2] == right[:2]
    torch.testing.assert_close(left[2], right[2], rtol=0, atol=0)


def manager(folder, *, enabled=True, asynchronous=False):
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Dropout(0.5), torch.nn.Linear(4, 1)
    )
    if dist.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    loader = Loader()
    checkpoint = CheckpointManager(
        CheckpointManager.Config(
            folder=str(folder),
            interval=1,
            keep_latest_k=0,
            last_save_model_only=False,
            enable_rng_state=enabled,
            async_mode="async" if asynchronous else "disabled",
        ),
        model_parts=[model],
        optimizers=optimizer,
        lr_schedulers=scheduler,
        extra_states={DATALOADER: loader},
        ema=None,
        sd_adapter=None,
    )
    return checkpoint, model, optimizer, scheduler, loader


def step(model, optimizer, scheduler, loader):
    a, b, x = draw()
    optimizer.zero_grad()
    loss = model(x * (a + b)).square().sum()
    loss.backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    optimizer.step()
    scheduler.step()
    loader.position += 1
    return {
        "draw": (a, b, x),
        "loss": loss.detach(),
        "gradients": gradients,
        "parameters": [p.detach().clone() for p in model.parameters()],
        "optimizer": pickle.loads(pickle.dumps(optimizer.state_dict())),
        "scheduler": scheduler.state_dict(),
        "position": loader.position,
    }


def test_capture_is_immutable_and_restore_is_explicit():
    seed(1)
    rng = RngState()
    state = rng.state_dict()
    expected = draw()
    seed(99)
    rng.load_state_dict(state)
    unrelated = draw()
    assert unrelated[0] != expected[0]
    rng.restore()
    assert_draw_equal(draw(), expected)


def test_validation_preserves_rng_on_exception():
    seed(1)
    snapshot = RngSnapshot.capture()
    expected = draw()
    snapshot.restore()
    with pytest.raises(RuntimeError), preserved_rng():
        draw()
        raise RuntimeError("validation failed")
    assert_draw_equal(draw(), expected)


def test_legacy_and_evaluation_loads(tmp_path, caplog):
    checkpoint, *objects = manager(tmp_path, enabled=False)
    checkpoint.save(1)
    checkpoint.close()
    checkpoint, model, optimizer, scheduler, loader = manager(tmp_path)
    try:
        assert checkpoint.load(1)
        assert "without RNG continuity" in caplog.text
        checkpoint.save(2)
        # Removing optimizer means evaluation can load without rank-RNG validation.
        checkpoint.exclude_from_loading = [OPTIMIZER, "lr_scheduler"]
        with mock.patch.object(RngState, "can_load", side_effect=AssertionError):
            assert checkpoint.load(2)
        checkpoint.exclude_from_loading = [RNG_STATE]
        with mock.patch.object(RngState, "can_load", side_effect=AssertionError):
            assert checkpoint.load(2)
    finally:
        checkpoint.close()


def test_incompatible_rank_or_device_layout_skips_rng(tmp_path, caplog):
    rng = RngState()
    assert not rng.can_load({f"{RNG_STATE}.rank_0", f"{RNG_STATE}.rank_1"})
    rng.load_state_dict(rng.state_dict())
    with mock.patch.object(RngSnapshot, "compatible", return_value=False):
        rng.restore()
    assert rng.pending is None
    assert "without RNG continuity" in caplog.text


def test_weights_only_export_omits_rng(tmp_path):
    checkpoint, *_ = manager(tmp_path)
    try:
        checkpoint.last_save_model_only = True
        checkpoint.save(1, last_step=True)
        keys = (
            dcp.FileSystemReader(str(tmp_path / "exports" / "step-1"))
            .read_metadata()
            .state_dict_metadata
        )
        assert not any(key.startswith(RNG_STATE) for key in keys)
    finally:
        checkpoint.close()


def restart_worker(rank, folder, rendezvous, mode, asynchronous):
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2
    )
    checkpoint = None
    try:
        seed(50 if mode == "save" else 300)
        checkpoint, model, optimizer, scheduler, loader = manager(
            folder, asynchronous=asynchronous
        )
        seed(50 + rank if mode == "save" else 300 + rank)
        if mode == "save":
            for _ in range(2):
                step(model, optimizer, scheduler, loader)
            checkpoint.save(2)
            # Validation after save must not alter the continuation.
            with preserved_rng():
                draw()
        else:
            # Allocate optimizer state so the ordinary optimizer Stateful template
            # requests its tensors from DCP. Production containers do this too.
            step(model, optimizer, scheduler, loader)
            if mode == "negative":
                checkpoint.exclude_from_loading = [RNG_STATE]
            checkpoint.load(2)
        results = [step(model, optimizer, scheduler, loader) for _ in range(3)]
        checkpoint.close()
        with open(Path(folder) / f"{mode}-{rank}.pkl", "wb") as handle:
            pickle.dump(results, handle)
    finally:
        if checkpoint is not None:
            checkpoint.close()
        dist.destroy_process_group()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_fresh_process_distributed_restart(tmp_path, asynchronous):
    for mode in ("save", "resume", "negative"):
        mp.spawn(
            restart_worker,
            args=(str(tmp_path), str(tmp_path / f"rdzv-{mode}"), mode, asynchronous),
            nprocs=2,
        )
    for rank in range(2):
        results = {}
        for mode in ("save", "resume", "negative"):
            with open(tmp_path / f"{mode}-{rank}.pkl", "rb") as handle:
                results[mode] = pickle.load(handle)
        torch.testing.assert_close(results["save"], results["resume"], rtol=0, atol=0)
        assert results["save"][0]["draw"][0] != results["negative"][0]["draw"][0]
    keys = dcp.FileSystemReader(tmp_path / "step-2").read_metadata().state_dict_metadata
    assert {f"{RNG_STATE}.rank_0", f"{RNG_STATE}.rank_1"} <= set(keys)


def test_remote_checkpoint_restores_rng():
    import uuid

    checkpoint, *_ = manager(f"memory://rng-test-{uuid.uuid4()}")
    try:
        seed(4)
        checkpoint.save(1)
        expected = draw()
        seed(9)
        assert checkpoint.load(1)
        assert_draw_equal(draw(), expected)
    finally:
        checkpoint.close()
        checkpoint._storage.remove(checkpoint.folder)


def incompatible_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2
    )
    try:
        rng = RngState()
        seed(1 + rank)
        rng.load_state_dict(rng.state_dict())
        seed(10 + rank)
        snapshot = RngSnapshot.capture()
        expected = draw()
        snapshot.restore()
        with mock.patch.object(RngSnapshot, "compatible", return_value=rank == 0):
            rng.restore()
        assert_draw_equal(draw(), expected)
    finally:
        dist.destroy_process_group()


def test_one_incompatible_rank_skips_rng_on_all_ranks(tmp_path):
    mp.spawn(incompatible_worker, args=(str(tmp_path / "rdzv"),), nprocs=2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_rng_through_manager(tmp_path):
    torch.cuda.init()
    checkpoint, *_ = manager(tmp_path)
    try:
        torch.cuda.manual_seed_all(19)
        checkpoint.save(1)
        expected = torch.rand(8, device="cuda")
        torch.cuda.manual_seed_all(91)
        checkpoint.load(1)
        torch.testing.assert_close(
            torch.rand(8, device="cuda"), expected, rtol=0, atol=0
        )
    finally:
        checkpoint.close()


def test_disabled_feature_keeps_custom_state_contract(tmp_path):
    checkpoint, *_ = manager(tmp_path, enabled=False)
    custom = Loader()
    checkpoint.states[RNG_STATE] = custom
    try:
        custom.position = 7
        checkpoint.save(1)
        custom.position = 99
        checkpoint.load(1)
        assert custom.position == 7
    finally:
        checkpoint.close()


def test_torchft_rejects_global_rank_rng_checkpointing():
    from torchtitan.experiments.torchft.checkpoint import TorchFTCheckpointManager

    with pytest.raises(ValueError, match="per-replica checkpoint participation"):
        TorchFTCheckpointManager.Config(enable_rng_state=True)
