# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pickle
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from torchtitan.components.checkpointer.base import DATALOADER, OPTIMIZER
from torchtitan.components.checkpointer.dcp import CheckpointManager
from torchtitan.components.checkpointer.rng import RNG_STATE, RngSnapshot, RngState


class Loader:
    """Restoration deliberately consumes the Torch CPU stream, like iterator setup."""

    def __init__(self):
        self.position = 0

    def state_dict(self):
        return {"position": self.position}

    def load_state_dict(self, state):
        self.position = state["position"]
        draw()


def draw(device="cpu"):
    return torch.rand(3, device=device)


def manager(folder, *, enabled=True, async_mode="disabled", device="cpu"):
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 4), torch.nn.Dropout(0.5), torch.nn.Linear(4, 1)
    ).to(device)
    if dist.is_initialized():
        device_ids = (
            [torch.device(device).index]
            if torch.device(device).type == "cuda"
            else None
        )
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids)
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
            async_mode=async_mode,
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
    x = draw(next(model.parameters()).device)
    optimizer.zero_grad()
    loss = model(x).square().sum()
    loss.backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    optimizer.step()
    scheduler.step()
    loader.position += 1
    return {
        "draw": x,
        "loss": loss.detach(),
        "gradients": gradients,
        "parameters": [p.detach().clone() for p in model.parameters()],
        "optimizer": pickle.loads(pickle.dumps(optimizer.state_dict())),
        "scheduler": scheduler.state_dict(),
        "position": loader.position,
    }


def test_capture_is_immutable_and_restore_is_explicit():
    torch.manual_seed(1)
    rng = RngState()
    state = rng.state_dict()
    expected = draw()
    torch.manual_seed(99)
    rng.load_state_dict(state)
    unrelated = draw()
    assert not torch.equal(unrelated, expected)
    rng.restore()
    torch.testing.assert_close(draw(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("case", ["legacy", "world_size", "backend"])
def test_strict_resume_and_explicit_rng_exclusion(tmp_path, case):
    source, model, _, _, loader = manager(tmp_path, enabled=case != "legacy")
    loader.position = 7
    weights = model[0].weight.detach().clone()
    try:
        snapshot = RngSnapshot.capture()
        if case == "world_size":
            source._rng_state.world_size = 2
        elif case == "backend":
            snapshot.accelerator = "different_backend"
        with mock.patch.object(RngSnapshot, "capture", return_value=snapshot):
            source.save(1)
    finally:
        source.close()

    checkpoint, model, _, _, loader = manager(tmp_path)
    try:
        if case == "legacy":
            with pytest.raises(dcp.CheckpointException, match="Missing key.*rng_state"):
                checkpoint.load(1)
        else:
            with pytest.raises(ValueError, match="world size|accelerator backend"):
                checkpoint.load(1)

        checkpoint.exclude_from_loading = [RNG_STATE]
        loader.position = 0
        with torch.no_grad():
            model[0].weight.zero_()
        torch.manual_seed(42)
        draw()  # Loader restoration consumes one draw.
        expected = draw()
        torch.manual_seed(42)
        assert checkpoint.load(1)
        assert loader.position == 7
        torch.testing.assert_close(model[0].weight, weights, rtol=0, atol=0)
        torch.testing.assert_close(draw(), expected, rtol=0, atol=0)
    finally:
        checkpoint.close()


def test_partial_load_excluding_optimizer_requires_rng_exclusion(tmp_path):
    checkpoint, *_ = manager(tmp_path)
    try:
        checkpoint.save(1)
        checkpoint.exclude_from_loading = [OPTIMIZER, "lr_scheduler"]
        with pytest.raises(ValueError, match="rng_state"):
            checkpoint.load(1)
        checkpoint.exclude_from_loading.append(RNG_STATE)
        assert checkpoint.load(1)
    finally:
        checkpoint.close()


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
        torch.manual_seed(50 if mode == "save" else 300)
        checkpoint, model, optimizer, scheduler, loader = manager(
            folder, async_mode="async" if asynchronous else "disabled"
        )
        torch.manual_seed(50 + rank if mode == "save" else 300 + rank)
        if mode == "save":
            for _ in range(2):
                step(model, optimizer, scheduler, loader)
            checkpoint.save(2)
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
        assert not torch.equal(
            results["save"][0]["draw"], results["negative"][0]["draw"]
        )
    keys = dcp.FileSystemReader(tmp_path / "step-2").read_metadata().state_dict_metadata
    assert {f"{RNG_STATE}.rank_0", f"{RNG_STATE}.rank_1"} <= set(keys)


def test_remote_checkpoint_restores_rng():
    import uuid

    checkpoint, *_ = manager(f"memory://rng-test-{uuid.uuid4()}")
    try:
        torch.manual_seed(4)
        checkpoint.save(1)
        expected = draw()
        torch.manual_seed(9)
        assert checkpoint.load(1)
        torch.testing.assert_close(draw(), expected, rtol=0, atol=0)
    finally:
        checkpoint.close()
        checkpoint._storage.remove(checkpoint.folder)


def incompatible_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2
    )
    try:
        rng = RngState()
        torch.manual_seed(1 + rank)
        rng.load_state_dict(rng.state_dict())
        torch.manual_seed(10 + rank)
        snapshot = RngSnapshot.capture()
        expected = draw()
        snapshot.restore()
        with mock.patch.object(RngSnapshot, "compatible", return_value=rank == 0):
            with pytest.raises(ValueError, match="accelerator backend"):
                rng.restore()
        torch.testing.assert_close(draw(), expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_one_incompatible_rank_fails_before_restoring_any_rank(tmp_path):
    mp.spawn(incompatible_worker, args=(str(tmp_path / "rdzv"),), nprocs=2)


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
