# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Persistent optimizer effects must be inside an authorized transaction."""

import copy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torchtitan.components.optim import AdamW, EMA, LRSchedulersContainer
from torchtitan.experiments.torchft.optimizer import (
    TorchFTOptim,
    TorchFTOptimizersContainer,
)
from torchtitan.last_words.control import LastWordsExitError
from torchtitan.last_words.training import LastWordsTrainingCoordinator


def assert_state_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_state_equal(left, right)
    else:
        assert actual == expected


@pytest.fixture
def transaction():
    torch.manual_seed(42)
    model = torch.nn.Linear(2, 1)
    model.register_buffer("routing_bias", torch.zeros(1))
    events = []
    manager = SimpleNamespace(authorized=False)

    def vote():
        events.append("vote")
        return manager.authorized

    manager.should_commit = vote
    manager.report_error = Mock()
    context = SimpleNamespace(pp_enabled=False, get_optional_mesh=lambda axis: None)
    config = TorchFTOptim.Config(
        optimizer=TorchFTOptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=".*", lr=0.01, fused=False)]
        ),
        lr_scheduler=LRSchedulersContainer.Config(warmup_steps=0),
        ema=EMA.Config(decays=[0.5]),
    )
    optim = TorchFTOptim(
        config,
        model_parts=[model],
        parallelism_context=context,
        training_steps=10,
        pp_has_last_stage=True,
    )
    ft_manager = SimpleNamespace(manager=manager, use_async_quorum=True)
    optim.configure_fault_tolerance(ft_manager)
    optim.optimizers.configure_fault_tolerance(ft_manager)
    optim.optimizers.init_cache_state_dict()
    control = SimpleNamespace(updating=False)

    def updating():
        control.updating = True
        events.append("updating")

    control.before_optimizer_update = updating
    optim.last_words = control

    def persistent_hook(*args):
        events.append("persistent_hook")
        model.routing_bias.add_(1)

    optim.optimizers.register_step_pre_hook(persistent_hook)
    loss = model(torch.tensor([[0.5, -0.25]])).square().sum()
    loss.backward()
    return SimpleNamespace(
        model=model,
        optim=optim,
        manager=manager,
        control=control,
        loss=loss,
        events=events,
    )


def test_rejected_vote_preserves_all_persistent_state(transaction):
    item = transaction
    before = copy.deepcopy(
        (
            item.model.state_dict(),
            item.optim.optimizers.state_dict(),
            item.optim.lr_schedulers.state_dict(),
            item.optim.ema.state_dict(),
        )
    )
    norm = item.optim.step(item.loss, current_step=1)
    assert torch.isfinite(norm)
    assert not item.optim.update_committed
    assert not item.control.updating
    assert item.events == ["vote"]
    after = (
        item.model.state_dict(),
        item.optim.optimizers.state_dict(),
        item.optim.lr_schedulers.state_dict(),
        item.optim.ema.state_dict(),
    )
    assert_state_equal(after, before)


def test_authorized_update_votes_once_before_persistent_hook(transaction):
    item = transaction
    item.manager.authorized = True
    before = copy.deepcopy(item.model.state_dict())
    scheduler_before = copy.deepcopy(item.optim.lr_schedulers.state_dict())
    ema_before = copy.deepcopy(item.optim.ema.state_dict())
    item.optim.step(item.loss, current_step=1)
    assert item.optim.update_committed
    assert item.events == ["vote", "updating", "persistent_hook"]
    assert item.model.routing_bias.item() == 1
    assert not torch.equal(item.model.weight, before["weight"])
    assert item.optim.lr_schedulers.state_dict() != scheduler_before
    assert any(
        not torch.equal(item.optim.ema.state_dict()[key], value)
        for key, value in ema_before.items()
    )
    assert not item.optim.optimizers.last_words_pre_authorized


def test_hook_failure_keeps_updating_and_does_not_advance_scheduler(transaction):
    item = transaction
    item.manager.authorized = True
    before = copy.deepcopy(item.optim.lr_schedulers.state_dict())

    def fail(*args):
        raise RuntimeError("persistent hook interrupted")

    item.optim.optimizers.register_step_pre_hook(fail)
    with pytest.raises(RuntimeError, match="hook interrupted"):
        item.optim.step(item.loss, current_step=1)
    assert item.control.updating
    assert not item.optim.update_committed
    assert not item.optim.optimizers.last_words_pre_authorized
    assert item.model.routing_bias.item() == 1
    assert_state_equal(item.optim.lr_schedulers.state_dict(), before)


@pytest.mark.parametrize(
    "unsafe", ["updating", "restoration_incomplete", "commit_unknown"]
)
def test_incomplete_transaction_never_saves_or_retries(unsafe):
    coordinator = LastWordsTrainingCoordinator.__new__(LastWordsTrainingCoordinator)
    coordinator.completed = {"step": 2, "ntokens_seen": 8, "last_words_num_updates": 2}
    coordinator.completed_train_state = dict(coordinator.completed)
    coordinator.iterator = iter([torch.tensor(1)])
    from torchtitan.last_words.rng import RNGCheckpointState

    coordinator.rng_state = RNGCheckpointState(torch.device("cpu"), 0)
    coordinator.completed_rng_state = None
    coordinator.elastic = False
    coordinator.num_updates = 2
    coordinator.manager = SimpleNamespace(
        begin_step=Mock(),
        restoration_incomplete=False,
        commit_unknown=False,
        commit_authorized=False,
        discard_interrupted_step=Mock(),
    )
    coordinator.control = SimpleNamespace(request=Mock(), publish=Mock())
    coordinator.checkpointer = SimpleNamespace(
        last_words_stop_requested=lambda: False,
        last_words_cursor_current=True,
        save_last_words=Mock(),
    )
    coordinator.engine = SimpleNamespace(
        begin_last_words_attempt=Mock(),
        optim=SimpleNamespace(optimizers=SimpleNamespace()),
        recover_last_words_attempt=Mock(),
    )

    def interrupted(batches):
        if unsafe == "updating":
            coordinator.updating = True
        else:
            setattr(coordinator.manager, unsafe, True)
        raise RuntimeError("injected transaction interruption")

    with pytest.raises(LastWordsExitError, match="Incomplete or unknown transaction"):
        coordinator.run_step(interrupted, coordinator.iterator)
    coordinator.checkpointer.save_last_words.assert_not_called()
    coordinator.engine.recover_last_words_attempt.assert_not_called()
    coordinator.manager.discard_interrupted_step.assert_not_called()
    assert coordinator.num_updates == 2


@pytest.mark.parametrize("committed", [False, True])
def test_engine_only_counts_completed_update(committed):
    from torchtitan.experiments.torchft.trainer import FaultTolerantTrainingEngine

    engine = FaultTolerantTrainingEngine.__new__(FaultTolerantTrainingEngine)
    engine.num_completed_steps = 2
    engine.loss = torch.tensor(1.0)
    engine.checkpointer = SimpleNamespace(maybe_wait_for_staging=Mock())
    engine.optim = SimpleNamespace(
        step=Mock(return_value=torch.tensor(0.5)), update_committed=committed
    )
    observed = []
    engine.last_words = SimpleNamespace(
        update_completed=lambda: observed.append(engine.num_completed_steps)
    )
    engine.optim_step()
    assert engine.num_completed_steps == 2 + int(committed)
    assert observed == ([3] if committed else [])


@pytest.fixture
def checkpoint(transaction, tmp_path):
    import torch.distributed as dist
    from torchtitan.last_words.checkpoint import LastWordsCheckpointManager

    dist.init_process_group(
        "gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1
    )
    item = transaction
    item.manager.authorized = True
    item.optim.step(item.loss, current_step=1)
    progress = {"step": 1, "ntokens_seen": 2, "last_words_num_updates": 1}
    cursor = {"position": 1}

    def stateful(values):
        def load(saved):
            values.clear()
            values.update(copy.deepcopy(saved))

        return SimpleNamespace(
            state_dict=lambda: copy.deepcopy(values), load_state_dict=load
        )

    manager = LastWordsCheckpointManager.Config(
        folder="checkpoint",
        interval=1,
        keep_latest_k=0,
        enable_ft_dataloader_checkpoints=False,
        last_save_model_only=False,
    ).build(
        model_parts=[item.model],
        optimizers=item.optim.optimizers,
        lr_schedulers=item.optim.lr_schedulers,
        ema=item.optim.ema,
        extra_states={
            "train_state": stateful(progress),
            "dataloader": stateful(cursor),
        },
        sd_adapter=None,
        base_folder=str(tmp_path),
        ft_manager=None,
    )
    try:
        assert type(manager) is LastWordsCheckpointManager
        yield manager, item, progress, cursor
    finally:
        manager.close()
        dist.destroy_process_group()


def test_real_dcp_restores_model_optimizer_scheduler_ema_and_progress(checkpoint):
    manager, item, progress, cursor = checkpoint
    expected = copy.deepcopy(
        (
            item.model.state_dict(),
            item.optim.optimizers.state_dict(),
            item.optim.lr_schedulers.state_dict(),
            item.optim.ema.state_dict(),
            progress,
            cursor,
        )
    )
    assert manager.save(1)
    with torch.no_grad():
        for parameter in item.model.parameters():
            parameter.add_(7)
        item.model.routing_bias.add_(7)
    item.optim.lr_schedulers.step()
    item.optim.ema.step(2)
    progress["step"], cursor["position"] = 99, 99
    assert manager.load(1)
    actual = (
        item.model.state_dict(),
        item.optim.optimizers.state_dict(),
        item.optim.lr_schedulers.state_dict(),
        item.optim.ema.state_dict(),
        progress,
        cursor,
    )
    assert_state_equal(actual, expected)


def test_emergency_payload_includes_auxiliary_and_native_progress(checkpoint):
    manager, item, progress, cursor = checkpoint
    manager.elastic_enabled = True
    manager.ft_manager = SimpleNamespace(
        state_dict=lambda: {"step": 1, "batches_committed": 2}
    )
    payload = manager._full_states()
    assert "model" not in payload
    assert "weight" in payload and "routing_bias" in payload
    assert {
        "optimizer",
        "lr_scheduler",
        "ema",
        "train_state",
        "torchft",
    } <= payload.keys()
    assert payload["torchft"] == {"step": 1, "batches_committed": 2}
    assert payload["train_state"].state_dict() == progress
    # Logical replica cursors are bound and saved separately from the model.
    assert "dataloader" not in payload


def test_truncated_dcp_shard_is_rejected_before_restore(checkpoint):
    from pathlib import Path

    from torchtitan.last_words.validation import validate_storage_ranges

    manager, item, progress, cursor = checkpoint
    assert manager.save(1)
    path = Path(manager.folder) / "step-1"
    validate_storage_ranges(path)
    shard = next(path.glob("*.distcp"))
    with shard.open("r+b") as stream:
        stream.truncate(shard.stat().st_size - 1)
    with pytest.raises(ValueError, match="truncated"):
        validate_storage_ranges(path)


def test_active_replica_publishes_its_own_initial_rng(checkpoint):
    import json
    from pathlib import Path

    import torch.distributed.checkpoint as dcp
    from torchtitan.last_words.rng import RNG_STATE, RNGCheckpointState

    manager, item, progress, cursor = checkpoint
    zero = dict(step=0, ntokens_seen=0, last_words_num_updates=0)
    own = RNGCheckpointState(torch.device("cpu"), 0)
    own.capture()
    expected = torch.rand(4)
    manager.rng_state = own
    manager.elastic_enabled = True
    manager.last_words_signature = "initial-active-rng"
    manager.last_words = SimpleNamespace(
        replica_id=1,
        num_replicas=2,
        rank=0,
        world_size=1,
        requested=False,
        config=SimpleNamespace(run_id="initial-rng", timeout_seconds=5),
    )
    manager.ft_states = {RNG_STATE: own}
    # The first active replica has already published. Replica 1 must publish
    # its own stream rather than wait for the first replica to invent one.
    other_path = Path(manager.folder) / "ft-replicat-0/initial-rng/initial/step-0"
    dcp.save({RNG_STATE: own}, checkpoint_id=other_path)
    (other_path / "last_words_cursor.json").write_text(
        json.dumps(manager._cursor_record(zero))
    )
    manager.initialize_elastic_cursors(
        zero,
        make_cursor=Mock(side_effect=AssertionError("No dormant slots")),
        initial_replicas=[0, 1],
    )
    own_path = Path(manager.folder) / "ft-replicat-1/initial-rng/initial/step-0"
    assert own_path.is_dir()
    own.capture()
    dcp.load(manager.ft_states, checkpoint_id=own_path)
    own.restore()
    assert torch.equal(torch.rand(4), expected)


def test_retention_recognizes_recoverable_rng_bound_periodic_checkpoints(checkpoint):
    import json
    from pathlib import Path

    import torch.distributed.checkpoint as dcp
    from torchtitan.last_words.job import retention_plan
    from torchtitan.last_words.rng import RNG_STATE, RNGCheckpointState

    manager, item, progress, cursor = checkpoint
    root = Path(manager.folder)
    for step in range(1, 4):
        progress.update(step=step, last_words_num_updates=step)
        assert manager.save(step)
        anchors = {}
        for replica in range(2):
            path = root / f"ft-replicat-{replica}/retention/step-{step}"
            rng = RNGCheckpointState(torch.device("cpu"), 0)
            rng.capture()
            dcp.save({RNG_STATE: rng}, checkpoint_id=path)
            record = dict(
                state=dict(progress),
                signature="retention-rng",
                rng_schema=1,
                rng_shards=1,
            )
            (path / "last_words_cursor.json").write_text(json.dumps(record))
            anchors[str(replica)] = str(path.relative_to(root))
        binding = dict(
            signature="retention-rng",
            run_id="retention",
            state=dict(progress),
            anchors=anchors,
            rng_schema=1,
            rng_shards=1,
        )
        (root / f"step-{step}/last_words_binding.json").write_text(json.dumps(binding))
    plan = retention_plan(root, keep=2, active_seconds=0, num_replicas=2)
    assert plan["delete"] == [str(root / "step-1")]
    assert str(root / "step-2") in plan["protected"]
    assert str(root / "step-3") in plan["protected"]
    assert not plan["notes"]
    # Planning never deletes either retained data or the proposed old version.
    assert all((root / f"step-{step}/.metadata").is_file() for step in range(1, 4))
