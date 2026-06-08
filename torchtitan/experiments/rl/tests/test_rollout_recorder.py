# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for `torchtitan.experiments.rl.rollout_recorder`."""

from __future__ import annotations

import json

from torchtitan.experiments.rl.rollout import (
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.rollout_recorder import (
    KeepExtremeRewardsFilter,
    RolloutSampleRecorder,
)


def _recorder(tmp_path, **config_kwargs) -> RolloutSampleRecorder:
    return RolloutSampleRecorder.Config(**config_kwargs).build(dump_dir=str(tmp_path))


def _turn(*, completion_logprobs: list[float], policy_version: int = 1) -> RolloutTurn:
    return RolloutTurn(
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[4, 5],
        completion_logprobs=completion_logprobs,
        policy_version=policy_version,
        prompt_messages=[{"role": "user", "content": "sort: b a c"}],
        completion_message={"role": "assistant", "content": "a b c"},
        env_messages=[{"role": "user", "content": "ok"}],
    )


def _group(group_idx: int, *, rewards: list[float | None]) -> RolloutGroup:
    group_id = f"step=1/group={group_idx}"
    return RolloutGroup(
        group_id=group_id,
        rollouts=[
            Rollout(
                group_id=group_id,
                sample_id=f"{group_id}/sample={s}",
                status=RolloutStatus.COMPLETED,
                turns=[_turn(completion_logprobs=[-0.5, -1.5])],
                reward=reward,
                reward_breakdown={} if reward is None else {"correct": reward},
            )
            for s, reward in enumerate(rewards)
        ],
    )


def _read_lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# --- filter ---


def test_filter_picks_highest_and_lowest_per_group() -> None:
    # k=2: the two -1 (lowest) + the two 0.9 (highest); the unscored None is skipped.
    group = _group(0, rewards=[0.9, 0.9, 0.2, None, -0.3, -1.0, -1.0])
    picked = KeepExtremeRewardsFilter.Config(k=2).build()([group])
    assert sorted(r.reward for r in picked) == [-1.0, -1.0, 0.9, 0.9]


def test_filter_dedupes_small_groups() -> None:
    group = _group(0, rewards=[0.3, 0.7])
    assert len(KeepExtremeRewardsFilter.Config(k=1).build()([group])) == 2
    assert len(KeepExtremeRewardsFilter.Config(k=5).build()([group])) == 2  # no dups


def test_default_filter_logs_only_highest_and_lowest_per_group(tmp_path) -> None:
    recorder = _recorder(tmp_path)  # default filter: KeepExtremeRewardsFilter, k=1
    recorder.record(
        step=1,
        is_validation=False,
        rollout_groups=[_group(0, rewards=[0.1, 0.9, 0.5, 0.2])],
    )
    records = _read_lines(tmp_path / "rollout_samples.jsonl")
    assert sorted(r["reward"] for r in records) == [0.1, 0.9]  # lowest + highest only


def test_logs_all_groups_no_cap(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    recorder.record(
        step=1,
        is_validation=False,
        rollout_groups=[_group(i, rewards=[0.1, 0.9]) for i in range(3)],
    )
    # default k=1 keeps both rollouts of each 2-rollout group; all 3 groups logged -> 6
    assert len(_read_lines(tmp_path / "rollout_samples.jsonl")) == 6


# --- record schema (the whole Rollout, dumped raw) ---


def test_record_dumps_the_rollout_minus_token_arrays(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    recorder.record(
        step=3, is_validation=True, rollout_groups=[_group(0, rewards=[1.0])]
    )

    (record,) = _read_lines(tmp_path / "rollout_samples.jsonl")
    assert record["step"] == 3 and record["is_validation"] is True
    assert record["group_id"] == "step=1/group=0"
    assert record["sample_id"] == "step=1/group=0/sample=0"
    assert record["status"] == "completed"
    assert record["reward"] == 1.0
    assert record["reward_breakdown"] == {"correct": 1.0}

    (turn,) = record["turns"]
    assert turn["policy_version"] == 1
    assert turn["completion_message"] == {"role": "assistant", "content": "a b c"}
    assert turn["env_messages"] == [{"role": "user", "content": "ok"}]
    # token-id / logprob arrays are large, so they are dropped unless opted in.
    assert "prompt_token_ids" not in turn
    assert "completion_token_ids" not in turn
    assert "completion_logprobs" not in turn


def test_log_tensors_and_logprobs_opt_in(tmp_path) -> None:
    recorder = _recorder(tmp_path, log_tensors=True, log_logprobs=True)
    recorder.record(
        step=1, is_validation=False, rollout_groups=[_group(0, rewards=[1.0])]
    )
    (turn,) = _read_lines(tmp_path / "rollout_samples.jsonl")[0]["turns"]
    assert turn["prompt_token_ids"] == [1, 2, 3]
    assert turn["completion_token_ids"] == [4, 5]
    assert turn["completion_logprobs"] == [-0.5, -1.5]


def test_appends_across_calls(tmp_path) -> None:
    recorder = _recorder(tmp_path)
    recorder.record(
        step=0, is_validation=True, rollout_groups=[_group(0, rewards=[1.0])]
    )
    recorder.record(
        step=1, is_validation=False, rollout_groups=[_group(0, rewards=[1.0])]
    )
    records = _read_lines(tmp_path / "rollout_samples.jsonl")
    assert [r["is_validation"] for r in records] == [True, False]


# --- edge cases ---


def _record_one(tmp_path, rollout: Rollout) -> dict:
    """Record a single rollout (wrapped in a one-rollout group) and return its record."""
    recorder = _recorder(tmp_path)
    group = RolloutGroup(group_id=rollout.group_id, rollouts=[rollout])
    recorder.record(step=1, is_validation=False, rollout_groups=[group])
    (record,) = _read_lines(tmp_path / "rollout_samples.jsonl")
    return record


def test_empty_turns_rollout(tmp_path) -> None:
    # A rollout with no turns (e.g. a prompt too long): no turns, no crash.
    rollout = Rollout(
        group_id="step=1/group=0",
        sample_id="step=1/group=0/sample=0",
        status=RolloutStatus.TRUNCATED_PROMPT_TOO_LONG,
        turns=[],
        reward=0.0,
    )
    record = _record_one(tmp_path, rollout)
    assert record["turns"] == []
    assert record["reward"] == 0.0


def test_turn_metrics_are_excluded_not_serialized(tmp_path) -> None:
    # Metric/MetricValue objects aren't JSON-serializable; they must be dropped, not crash.
    from torchtitan.experiments.rl.observability import metrics as m

    turn = _turn(completion_logprobs=[-0.5, -1.5])
    turn.metrics = [m.Metric("generator/output_tokens", m.Sum(2.0))]
    rollout = Rollout(
        group_id="step=1/group=0",
        sample_id="step=1/group=0/sample=0",
        status=RolloutStatus.COMPLETED,
        turns=[turn],
        reward=1.0,
    )
    record = _record_one(tmp_path, rollout)
    assert "metrics" not in record["turns"][0]


def test_none_policy_version_and_completion_message(tmp_path) -> None:
    turn = RolloutTurn(
        prompt_token_ids=[1, 2],
        completion_token_ids=[],
        completion_logprobs=[],
        policy_version=None,
        completion_message=None,
    )
    rollout = Rollout(
        group_id="step=1/group=0",
        sample_id="step=1/group=0/sample=0",
        status=RolloutStatus.ERROR,
        turns=[turn],
        reward=0.0,
    )
    record = _record_one(tmp_path, rollout)
    assert record["turns"][0]["policy_version"] is None
    assert record["turns"][0]["completion_message"] is None
