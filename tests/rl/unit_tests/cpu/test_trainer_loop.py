# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `Controller._trainer_loop`: what each trainer DP replica's leader receives,
and the model state dict push and pull.

Each step's push finishes before the next step's forward/backward, and its pull
overlaps that forward/backward. These tests run the loop with fake actors (no GPU /
Monarch / TorchStore) and check that ordering.
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


class _RankValues:
    """Stands in for a Monarch ValueMesh."""

    def __init__(self, value):
        self._value = value

    def get(self, rank):
        return self._value

    def values(self):
        return [self._value]


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
        pass

    async def start_reading(*, num_microbatch_groups):
        events.append(f"read ahead {version + 2}")

    async def forward_backward(**kwargs):
        events.append(f"fwd_bwd {version + 1}")
        if on_forward_backward is not None:
            await on_forward_backward()
        return _RankValues({"loss/mean": 1.0})

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

    async def release_active_groups(count, *, reason):
        events.append(f"release {count} {reason}")

    controller = object.__new__(Controller)
    controller.config = SimpleNamespace(
        async_loop=SimpleNamespace(
            target_offpolicy_steps=1,
            max_offpolicy_steps=None,
            num_prompts_per_train_step=5,
            validation=ValidationConfig(),
        ),
        rollouter=SimpleNamespace(worker=SimpleNamespace(rdma_min_group_bytes=None)),
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
        start_reading=_Endpoint(start_reading),
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
        release_groups=lambda group_ids: events.append(f"release_groups {group_ids}"),
    )
    controller._group_buffer = SimpleNamespace(
        metrics=lambda: [], release_active_groups=release_active_groups
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
        # Batch 2 is queued: its readers start before step 1's optimizer step.
        "read ahead 2",
        "optim 1",
        "push start",
        "push end",
        # Step 1's pull -> release runs in the background. The per-DP-rank forward_backward
        # calls run as tasks, so with these instant fakes the pull finishes first.
        "pull 1",
        "release 5 trained",
        "fwd_bwd 2",
        "optim 2",
        "push start",
        "push end",
        "pull 2",
        "release 5 trained",
    ]


def test_trained_groups_go_to_the_workers_once_their_step_releases_their_slots() -> None:
    """When groups can stay in the rollout workers, a step's trained group ids go to every
    worker only after its `forward_backward` returned (every leader read is done) and the pull
    released their slots. By value (the other tests), nothing goes to the workers."""
    events: list[str] = []
    controller = _controller(events)
    controller.config.rollouter.worker.rdma_min_group_bytes = 1

    _run(controller, num_training_steps=2)

    # Batch 0 trains groups 0..4, batch 1 groups 5..9.
    releases = [event for event in events if event.startswith("release_groups")]
    assert releases == [
        "release_groups [0, 1, 2, 3, 4]",
        "release_groups [5, 6, 7, 8, 9]",
    ]
    assert events.index("release_groups [0, 1, 2, 3, 4]") > events.index("fwd_bwd 1")
    assert (
        events.index("release_groups [0, 1, 2, 3, 4]")
        == events.index("release 5 trained") + 1
    )


def test_the_next_steps_columns_go_out_during_forward_backward() -> None:
    """With plan 2 already queued when forward_backward 1 starts, its leaders get their columns
    while it runs, and every rank starts reading after it returns and before optim_step 1; step 2
    then trains on the started readers without sending the columns again."""
    events: list[str] = []

    async def slow_forward_backward():
        await asyncio.sleep(0.01)
        events.append("fwd_bwd returns")

    controller = _controller(events, on_forward_backward=slow_forward_backward)
    leader = controller.trainer.leader_by_dp_rank[0]
    received_group_ids: list[list[int]] = []

    async def receive_replica_column(column):
        received_group_ids.append([group.group_id for group in column.rollout_groups])
        events.append(f"column {column.rollout_groups[0].group_id}")

    leader.receive_replica_column = _Endpoint(receive_replica_column)

    _run(controller, num_training_steps=2)

    assert events[0] == "column 0"
    assert events.index("column 5") < events.index("fwd_bwd returns")
    assert (
        events.index("fwd_bwd returns")
        < events.index("read ahead 2")
        < events.index("optim 1")
    )
    assert received_group_ids == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    read_ahead = [
        logged["trainer/read_ahead"].value.value for logged in controller.logged_metrics
    ]
    assert read_ahead == [0.0, 1.0]
    # What stays exposed of the read-ahead is timed in the step that does it. The last step
    # times an empty one, so every step's accounting (`compute_perf_ratio_metrics`) has it.
    assert all(
        "timing/step/read_ahead" in logged for logged in controller.logged_metrics
    )


def test_a_failed_forward_backward_cancels_the_next_steps_send() -> None:
    """If forward_backward raises while the next step's columns are going out, the send is
    cancelled instead of outliving the loop."""

    async def failing_forward_backward():
        # Plan 2 was taken before the gather, and its send is under way.
        await asyncio.sleep(0.01)
        raise RuntimeError("forward_backward failed")

    controller = _controller([], on_forward_backward=failing_forward_backward)
    sent_group_ids: list[int] = []

    async def run() -> None:
        never = asyncio.Event()

        async def receive_replica_column(column):
            sent_group_ids.append(column.rollout_groups[0].group_id)
            if len(sent_group_ids) > 1:
                await never.wait()  # step 2's column never lands

        controller.trainer.leader_by_dp_rank[0].receive_replica_column = _Endpoint(
            receive_replica_column
        )
        controller.config.async_loop.num_training_steps = 2
        with pytest.raises(RuntimeError, match="forward_backward failed"):
            await controller._trainer_loop(_batch_queue(2), num_training_steps=2)
        for _ in range(3):
            await asyncio.sleep(0)
        assert sent_group_ids == [0, 5]
        assert [
            task for task in asyncio.all_tasks() if task is not asyncio.current_task()
        ] == []

    asyncio.run(run())


def test_no_read_ahead_past_the_last_step() -> None:
    events: list[str] = []
    controller = _controller(events)
    queue = _batch_queue(3)
    controller.config.async_loop.num_training_steps = 2

    asyncio.run(controller._trainer_loop(queue, num_training_steps=2))

    assert [event for event in events if event.startswith("read ahead")] == [
        "read ahead 2"
    ]
    assert queue.qsize() == 1


def test_a_shutdown_taken_during_read_ahead_stops_the_next_step() -> None:
    events: list[str] = []
    controller = _controller(events)
    queue = _batch_queue(1)
    queue.put_nowait(None)
    controller.config.async_loop.num_training_steps = 3

    asyncio.run(controller._trainer_loop(queue, num_training_steps=3))

    assert [event for event in events if event.startswith(("fwd_bwd", "read"))] == [
        "fwd_bwd 1"
    ]


def test_a_read_ahead_batchs_policy_age_uses_the_version_it_trains_at() -> None:
    """Batch 2 is taken during step 1, before optimizer step 1, but its policy age is computed
    when step 2 consumes it, against policy 1."""
    events: list[str] = []
    controller = _controller(events)

    _run(controller, num_training_steps=2)

    ages = [
        logged["train_batch/policy_age"].value.value
        / logged["train_batch/policy_age"].value.count
        for logged in controller.logged_metrics
    ]
    assert ages == [0.0, 1.0]


def test_forward_backward_waits_for_the_previous_push() -> None:
    async def slow_push():
        for _ in range(5):
            await asyncio.sleep(0)

    events: list[str] = []
    _run(_controller(events, on_push=slow_push), num_training_steps=2)

    # Forward/backward 2 blocks the trainer's event loop, so push 1 must finish first.
    assert events.index("push end") < events.index("fwd_bwd 2")


def test_next_push_waits_for_the_previous_pull() -> None:
    async def slow_pull():
        for _ in range(5):
            await asyncio.sleep(0)

    events: list[str] = []
    _run(_controller(events, on_pull=slow_pull), num_training_steps=2)

    # Push 2 must not overwrite the TorchStore key while pull 1 reads it.
    second_push = [i for i, event in enumerate(events) if event == "push start"][1]
    assert events.index("release 5 trained") < second_push


def test_last_push_and_pull_finish_before_the_loop_returns() -> None:
    events: list[str] = []
    _run(_controller(events), num_training_steps=1)

    assert events[-3:] == ["push end", "pull 1", "release 5 trained"]


def test_push_time_is_logged_at_its_step_and_pull_time_one_step_late() -> None:
    controller = _controller([])
    _run(controller, num_training_steps=2)

    first, second = controller.logged_metrics
    # Step 1 logs its own push; it has no previous pull.
    assert first[TRAINER_PUSH_KEY].value.value > 0.0
    assert first[GENERATOR_PULL_KEY].value.value == 0.0
    # Step 2 logs step 1's pull, which has run by then.
    assert second[GENERATOR_PULL_KEY].value.value > 0.0


def test_the_column_send_is_timed_apart_from_forward_backward() -> None:
    """`timing/step/forward_backward` (and the perf ratios read from it) times training only;
    sending the leaders their columns has its own step timer."""
    controller = _controller([])

    async def slow_receive_replica_column(column):
        await asyncio.sleep(0.05)

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

        return SimpleNamespace(receive_replica_column=_Endpoint(receive_replica_column))

    async def forward_backward(**kwargs):
        forward_backward_calls.append(kwargs)
        return _RankValues({"loss/mean": 1.0})

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
