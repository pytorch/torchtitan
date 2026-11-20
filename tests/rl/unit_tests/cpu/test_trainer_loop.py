# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `Controller._trainer_loop`: what each trainer DP replica's leader receives,
and the model state dict push and pull.

The push and pull of each step's policy overlap the next step's forward/backward.
These tests run the loop with fake actors (no GPU / Monarch / TorchStore) and check
the ordering that makes the overlap safe.
"""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from torchtitan.rl.actors.system import TrainerHandle
from torchtitan.rl.controller.controller import Controller, ValidationConfig
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout.packed_group import pack_rollout_group, PackedRolloutGroup
from torchtitan.rl.rollout.types import (
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
    RolloutTurnID,
)
from torchtitan.rl.trainer.batcher import StepPlan

TRAINER_PUSH_KEY = "timing/weight_sync/trainer_push_model_state_dict"
GENERATOR_PULL_KEY = "timing/weight_sync/generator_pull_model_state_dict"
# The fake trainer's two last-stage ranks spend this long in their forward_backward handlers:
# the rank whose call arrived last waits least. A third rank without the last stage returns
# no metrics and leaves early (EARLY_STAGE_HANDLER_S); it must not count.
TRAINER_HANDLER_S = (0.02, 0.035)
EARLY_STAGE_HANDLER_S = 0.001


class _RankValues:
    """Stands in for a Monarch ValueMesh; every rank holds `value` unless `per_rank` is given."""

    def __init__(self, value=None, *, per_rank=None):
        self._values = per_rank if per_rank is not None else [value]

    def get(self, rank):
        return self._values[rank]

    def values(self):
        return self._values


class _Endpoint:
    """Stands in for a Monarch endpoint; `on_call` runs on each call."""

    def __init__(self, on_call=None):
        self._on_call = on_call

    async def call(self, *args, **kwargs):
        if self._on_call is not None:
            return await self._on_call(*args, **kwargs)

    async def call_one(self, *args, **kwargs):
        return await self.call(*args, **kwargs)


def _controller(
    events: list[str], *, on_push=None, on_pull=None, on_forward_backward=None
) -> Controller:
    """A controller with only the state `_trainer_loop` reads; each actor call appends to `events`."""
    version = 0

    async def receive_replica_column(column):
        return 0.0

    async def forward_backward(**kwargs):
        events.append(f"fwd_bwd {version + 1}")
        if on_forward_backward is not None:
            await on_forward_backward()
        return _RankValues(
            per_rank=[
                *[({"loss/mean": 1.0}, handler_s) for handler_s in TRAINER_HANDLER_S],
                (None, EARLY_STAGE_HANDLER_S),
            ]
        )

    async def optim_step(**kwargs):
        nonlocal version
        version += 1
        events.append(f"optim {version}")
        return _RankValues(SimpleNamespace(policy_version=version, metrics={}))

    async def push():
        events.append("push start")
        if on_push is not None:
            await on_push()
        events.append("push end")

    async def pull(policy_version):
        events.append(f"pull {policy_version}")
        if on_pull is not None:
            await on_pull()

    async def release_slots(group_ids, *, reason):
        events.append(f"release_slots {group_ids} {reason}")

    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(
            target_offpolicy_steps=1,
            max_offpolicy_steps=None,
            num_prompts_per_train_step=5,
            validation=ValidationConfig(),
        )
    )
    controller.start_step = 0
    controller._trainer_policy_version = 0
    controller._validation_tasks = []
    controller._has_pending_validation = False
    # The groups `_batch_queue`'s batches train, as the batcher loop leaves them.
    controller._packed_groups_by_id = {
        group_id: _packed_group(group_id, num_rollouts=1) for group_id in range(50)
    }
    trainer = SimpleNamespace(
        sync_log_step=_Endpoint(),
        receive_replica_column=_Endpoint(receive_replica_column),
        forward_backward=_Endpoint(forward_backward),
        optim_step=_Endpoint(optim_step),
        push_model_state_dict=_Endpoint(push),
    )
    # DP degree 1: every trainer rank is at DP rank 0.
    controller.trainer_dp_degree = 1
    controller.trainer = TrainerHandle(
        all_ranks=trainer,
        rank0=trainer,
        by_dp_rank=[trainer],
        leader_by_dp_rank=[trainer],
    )
    controller.generator_router = SimpleNamespace(
        sync_log_step=_Endpoint(), pull_model_state_dict=_Endpoint(pull)
    )
    controller.batcher = SimpleNamespace(
        sync_log_step=SimpleNamespace(broadcast=lambda step: None),
    )

    async def sync_log_step(step):
        pass

    controller._rollouter = SimpleNamespace(
        sync_log_step=sync_log_step,
        commit_training_sample_ids=lambda group_ids: None,
        state_dict=lambda: {},
    )
    controller._group_buffer = SimpleNamespace(
        metrics=lambda: [], release_slots=release_slots
    )
    controller.logged_metrics = []
    controller.metrics_processor = SimpleNamespace(
        log=lambda *, step, is_validation, metrics: controller.logged_metrics.append(
            {metric.key: metric for metric in metrics}
        )
    )
    return controller


def _packed_group(group_id: int, *, num_rollouts: int) -> PackedRolloutGroup:
    """A group of `num_rollouts` one-turn rollouts, each with 2 prompt and 3 completion tokens."""
    rollouts = [
        Rollout(
            group_id=group_id,
            rollout_id=rollout_id,
            status=RolloutStatus.COMPLETED,
            turns=[
                RolloutTurn(
                    rollout_id=RolloutTurnID(
                        group_id=group_id, rollout_id=rollout_id, turn_id=0
                    ),
                    prompt_prefix_len=0,
                    prompt_delta_token_ids=np.array([1, 2], dtype=np.int32),
                    completion_token_ids=np.array([3, 4, 5], dtype=np.int32),
                    completion_logprobs=np.zeros(3, dtype=np.float32),
                    min_policy_version=0,
                    max_policy_version=0,
                )
            ],
            reward=float(rollout_id),
            advantage=1.0,
        )
        for rollout_id in range(num_rollouts)
    ]
    _, packed = pack_rollout_group(RolloutGroup(group_id=group_id, rollouts=rollouts))
    return packed


def _spec(group_id: int, rollout_id: int) -> SimpleNamespace:
    """All `_trainer_loop` reads of a spec: which rollout it comes from."""
    return SimpleNamespace(
        rollout_id=RolloutTurnID(group_id=group_id, rollout_id=rollout_id, turn_id=0)
    )


def _batch(
    trainable_group_ids: list[int],
    microbatch_groups: list[list[list[list[SimpleNamespace]]]],
    *,
    untrainable_group_id: int,
) -> StepPlan:
    return StepPlan(
        microbatch_groups=microbatch_groups,
        global_loss_token_counts=torch.tensor([1]),
        global_routing_token_counts=torch.tensor([1]),
        metrics=[],
        group_ids=[*trainable_group_ids, untrainable_group_id],
        trainable_group_ids=trainable_group_ids,
        min_policy_versions=[0],
    )


def _batch_queue(num_batches: int) -> asyncio.Queue:
    """Batch `i` trains groups 5i..5i+4 (num_prompts_per_train_step=5), one microbatch of their
    rollout 0, plus one untrainable group."""
    queue: asyncio.Queue = asyncio.Queue()
    for batch_index in range(num_batches):
        trainable_group_ids = list(range(5 * batch_index, 5 * batch_index + 5))
        queue.put_nowait(
            _batch(
                trainable_group_ids,
                [[[[_spec(group_id, 0) for group_id in trainable_group_ids]]]],
                untrainable_group_id=100 + batch_index,
            )
        )
    return queue


def _run(controller: Controller, *, num_training_steps: int) -> None:
    controller.config.async_loop.num_training_steps = num_training_steps
    asyncio.run(
        controller._trainer_loop(
            _batch_queue(num_training_steps), num_training_steps=num_training_steps
        )
    )


def test_push_then_pull_then_release_after_each_optimizer_step() -> None:
    events: list[str] = []
    _run(_controller(events), num_training_steps=2)

    assert events == [
        "fwd_bwd 1",
        "optim 1",
        # Step 1's push -> pull -> release starts in the background. The per-DP-rank
        # forward_backward calls run as tasks, so with these instant fakes it finishes first ...
        "push start",
        "push end",
        "pull 1",
        # Only the trained groups: the batcher loop released the untrainable one already.
        "release_slots [0, 1, 2, 3, 4] trained",
        "fwd_bwd 2",
        # ... and lands before step 2's optimizer step.
        "optim 2",
        "push start",
        "push end",
        "pull 2",
        "release_slots [5, 6, 7, 8, 9] trained",
    ]


def test_optimizer_step_waits_for_the_previous_push() -> None:
    async def slow_push():
        for _ in range(5):
            await asyncio.sleep(0)

    events: list[str] = []
    _run(_controller(events, on_push=slow_push), num_training_steps=2)

    # Forward/backward 2 overlaps push 1; optimizer step 2 waits for it to finish.
    assert events.index("fwd_bwd 2") < events.index("push end")
    assert events.index("push end") < events.index("optim 2")


def test_next_push_waits_for_the_previous_pull() -> None:
    async def slow_pull():
        for _ in range(5):
            await asyncio.sleep(0)

    events: list[str] = []
    _run(_controller(events, on_pull=slow_pull), num_training_steps=2)

    # Push 2 must not overwrite the TorchStore key while pull 1 reads it.
    second_push = [i for i, event in enumerate(events) if event == "push start"][1]
    assert events.index("release_slots [0, 1, 2, 3, 4] trained") < second_push


def test_last_push_and_pull_finish_before_the_loop_returns() -> None:
    events: list[str] = []
    _run(_controller(events), num_training_steps=1)

    assert events[-3:] == [
        "push end",
        "pull 1",
        "release_slots [0, 1, 2, 3, 4] trained",
    ]


def test_push_and_pull_times_are_logged_one_step_late() -> None:
    controller = _controller([])
    _run(controller, num_training_steps=2)

    first, second = controller.logged_metrics
    # Step 1 has no previous push or pull.
    assert first[TRAINER_PUSH_KEY].value.value == 0.0
    assert first[GENERATOR_PULL_KEY].value.value == 0.0
    # Step 2 logs step 1's push and pull, which have run by then.
    assert second[TRAINER_PUSH_KEY].value.value > 0.0
    assert second[GENERATOR_PULL_KEY].value.value > 0.0


def test_forward_backward_transfer_is_the_wall_time_minus_the_handler_time() -> None:
    async def slow_forward_backward():
        await asyncio.sleep(0.05)

    controller = _controller([], on_forward_backward=slow_forward_backward)
    _run(controller, num_training_steps=1)

    [metrics] = controller.logged_metrics
    wall_s = metrics["timing/step/forward_backward"].value.value
    transfer_s = metrics["timing/hop/forward_backward/transfer"].value.value
    assert wall_s >= 0.05
    # The shortest handler: the rank that got its call last bounds the transfer.
    assert transfer_s == pytest.approx(wall_s - min(TRAINER_HANDLER_S), abs=5e-3)
    assert metrics["timing/hop/forward_backward/handler"].value.value == max(
        TRAINER_HANDLER_S
    )


def test_the_column_send_is_timed_apart_from_forward_backward() -> None:
    """`timing/step/forward_backward` (and the perf ratios read from it) times training only;
    sending the leaders their columns has its own step timer."""
    controller = _controller([])

    async def slow_receive_replica_column(column):
        await asyncio.sleep(0.05)
        return 0.0

    [leader] = controller.trainer.leader_by_dp_rank
    leader.receive_replica_column = _Endpoint(slow_receive_replica_column)
    _run(controller, num_training_steps=1)

    [metrics] = controller.logged_metrics
    assert metrics["timing/step/send_replica_columns"].value.value >= 0.05
    assert metrics["timing/step/forward_backward"].value.value < 0.05


def test_push_failure_fails_the_loop() -> None:
    async def failing_push():
        raise RuntimeError("push failed")

    controller = _controller([], on_push=failing_push)
    with pytest.raises(RuntimeError, match="push failed"):
        _run(controller, num_training_steps=2)


def test_failed_validation_fails_training() -> None:
    """freq=1: the validation started at step 1 fails, and step 2's check raises its error."""

    async def validate(*, step: int) -> list[m.Metric]:
        raise RuntimeError("validation failed")

    controller = _controller([])
    controller.config.async_loop.validation = ValidationConfig(freq=1)
    controller.validate = validate
    with pytest.raises(RuntimeError, match="validation failed"):
        _run(controller, num_training_steps=3)


def test_each_leader_receives_its_column_and_only_the_rollouts_it_names() -> None:
    received = {}
    forward_backward_calls = []

    def leader(dp_rank: int) -> SimpleNamespace:
        async def receive_replica_column(column):
            received[dp_rank] = column
            return 0.001

        return SimpleNamespace(receive_replica_column=_Endpoint(receive_replica_column))

    async def forward_backward(**kwargs):
        forward_backward_calls.append(kwargs)
        return _RankValues(({"loss/mean": 1.0}, 0.0))

    controller = _controller([])
    group_7 = _packed_group(7, num_rollouts=3)
    group_8 = _packed_group(8, num_rollouts=2)
    controller._packed_groups_by_id = {7: group_7, 8: group_8, 9: object()}
    replica = SimpleNamespace(
        forward_backward=_Endpoint(forward_backward),
        **{
            name: getattr(controller.trainer.all_ranks, name)
            for name in ("sync_log_step", "optim_step", "push_model_state_dict")
        },
    )
    controller.trainer_dp_degree = 2
    controller.trainer = TrainerHandle(
        all_ranks=replica,
        rank0=replica,
        by_dp_rank=[replica, replica],
        leader_by_dp_rank=[leader(0), leader(1)],
    )
    controller.config.async_loop.num_prompts_per_train_step = 2
    # Two groups of two microbatches per DP rank. Rollout 7/1 lands on both ranks; group 8
    # only on rank 1; rollout 7/2 nowhere.
    plan_grid = [
        [[[_spec(7, 0)], [_spec(7, 1), _spec(8, 1)]]],
        [[[_spec(7, 1)], [_spec(8, 0)]]],
    ]
    queue: asyncio.Queue = asyncio.Queue()
    queue.put_nowait(_batch([7, 8], plan_grid, untrainable_group_id=100))
    controller.config.async_loop.num_training_steps = 1

    asyncio.run(controller._trainer_loop(queue, num_training_steps=1))

    assert [received[0].microbatch_groups, received[1].microbatch_groups] == [
        [[[_spec(7, 0)]], [[_spec(7, 1)]]],
        [[[_spec(7, 1), _spec(8, 1)]], [[_spec(8, 0)]]],
    ]

    def rollout_ids(column) -> dict[int, tuple[int, ...]]:
        return {group.group_id: group.rollout_ids for group in column.rollout_groups}

    assert rollout_ids(received[0]) == {7: (0, 1)}
    assert rollout_ids(received[1]) == {7: (1,), 8: (0, 1)}
    # Each rank's forward_backward carries no tensor but the global counts.
    assert (
        forward_backward_calls
        == [
            {
                "num_microbatch_groups": 2,
                "global_loss_token_counts": torch.tensor([1]),
                "global_routing_token_counts": torch.tensor([1]),
            }
        ]
        * 2
    )
    # The leaders hold the sent groups now; the untrainable group 9 is not this step's.
    assert list(controller._packed_groups_by_id) == [9]
    [metrics] = controller.logged_metrics
    column_bytes = metrics["payload/replica_column_packed_bytes"].value
    assert column_bytes.value == (
        group_7.select({0, 1}).num_bytes
        + group_7.select({1}).num_bytes
        + group_8.num_bytes
    )
    assert column_bytes.count == 2
    assert metrics["timing/hop/receive_replica_column/handler"].value.value == (
        pytest.approx(0.002)
    )
