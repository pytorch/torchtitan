# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for RL metric helpers + controller-subroutine outputs.

These tests do **not** start Monarch, vLLM, W&B, or distributed
process groups. Controller subroutines are invoked as static / instance
methods on plain dataclasses.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

import torch

from torchtitan.experiments.rl.grpo import (
    _RL_TRAIN_HEADLINE_METRIC_PATTERNS,
    _RL_VALIDATION_HEADLINE_METRIC_PATTERNS,
    GRPOLoss,
    RLTrainer,
)
from torchtitan.experiments.rl.loss.types import LossOutput
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.observability.grpo_metrics import (
    _population_std,
    build_reward_component_metrics,
)
from torchtitan.experiments.rl.types import Completion, Step, Trajectory


# ---------------------------------------------------------------------------
# _population_std
# ---------------------------------------------------------------------------


class TestPopulationStd:
    def test_empty_returns_nan(self) -> None:
        assert math.isnan(_population_std([]))

    def test_single_value_zero(self) -> None:
        assert _population_std([5.0]) == 0.0

    def test_matches_population_std(self) -> None:
        # Population std (ddof=0) of [1,2,3,4] = sqrt(1.25)
        assert _population_std([1.0, 2.0, 3.0, 4.0]) == pytest.approx(
            math.sqrt(1.25), abs=1e-7
        )


# ---------------------------------------------------------------------------
# build_reward_component_metrics
# ---------------------------------------------------------------------------


def _step(rewards: dict[str, float]) -> Step:
    return Step(rewards=rewards, done=True)


class TestBuildRewardComponentMetrics:
    def test_one_metric_per_observed_name(self) -> None:
        steps = [
            _step({"correctness": 1.0, "format": 0.5}),
            _step({"correctness": 0.0, "format": 1.0}),
        ]
        metrics = build_reward_component_metrics("reward/component", steps)
        keys = {entry.key for entry in metrics}
        assert keys == {
            "reward/component/correctness",
            "reward/component/format",
        }
        for entry in metrics:
            assert isinstance(entry.reduction, m.Mean)

    def test_components_observed_in_some_steps_only(self) -> None:
        # `format` only appears in step #2 — it should average over that
        # one step (no zero-fill).
        steps = [_step({"correctness": 1.0}), _step({"format": 0.5})]
        metrics = build_reward_component_metrics("reward/component", steps)
        agg = m.aggregate_metrics(metrics)
        assert agg["reward/component/correctness/mean"] == 1.0
        assert agg["reward/component/format/mean"] == 0.5

    def test_empty_steps(self) -> None:
        assert build_reward_component_metrics("reward/component", []) == []

    def test_prefix_controls_namespace(self) -> None:
        steps = [_step({"correctness": 1.0})]
        metrics = build_reward_component_metrics("validation/reward/component", steps)
        assert metrics[0].key == "validation/reward/component/correctness"


# ---------------------------------------------------------------------------
# Controller subroutines: _collect_rollouts, _build_episodes, validate
# ---------------------------------------------------------------------------


def _completion(
    prompt_idx: int,
    prompt_len: int,
    response_len: int,
    finish_reason: str | None = "stop",
) -> Completion:
    return Completion(
        policy_version=0,
        prompt_idx=prompt_idx,
        prompt_token_ids=list(range(prompt_len)),
        text="x" * response_len,
        token_ids=list(range(response_len)),
        token_logprobs=[0.0] * response_len,
        finish_reason=finish_reason,
    )


class _FakeEnv:
    """Minimal env stub: ``step(text)`` returns a preset reward dict."""

    def __init__(self, rewards: dict[str, float]):
        self.prompt = "p"
        self._rewards = rewards

    def step(self, text: str) -> Step:
        return _step(self._rewards)


def _build_collect_rollouts_inputs(self_obj):
    """Build a hollow RLTrainer instance + completions wired into the
    fake generator — without spawning real meshes.
    """

    completions = [
        _completion(prompt_idx=0, prompt_len=4, response_len=10),
        _completion(prompt_idx=0, prompt_len=4, response_len=20),
        _completion(prompt_idx=1, prompt_len=6, response_len=15),
    ]

    class _RewardEnvBuilder:
        @staticmethod
        def build(*, step, group_idx):
            return _FakeEnv({"correctness": float(group_idx), "format": 0.5})

    self_obj.config = MagicMock()
    self_obj.config.env = _RewardEnvBuilder
    self_obj.generator = MagicMock()
    # `_get_rank_0_value` is the layer that strips Monarch's ValueMesh,
    # so just make it return whatever it's handed.
    self_obj._get_rank_0_value = lambda value, has_gpus=True: completions
    return completions


class TestCollectRollouts:
    def test_emits_expected_metric_keys(self) -> None:
        controller = RLTrainer.__new__(RLTrainer)
        completions = _build_collect_rollouts_inputs(controller)
        trajectories, rollout_metrics = controller._collect_rollouts(
            num_groups=2, step=0
        )
        assert len(trajectories) == len(completions)
        agg = m.aggregate_metrics(rollout_metrics)
        # Length keys: Mean+Max for prompt/response, Max-only for total.
        assert "rollout/response_length/mean" in agg
        assert "rollout/response_length/max" in agg
        assert "rollout/prompt_length/mean" in agg
        assert "rollout/prompt_length/max" in agg
        assert "rollout/total_length/max" in agg
        # Reward-component keys derived from env step output (now under
        # the top-level ``reward/`` namespace).
        assert "reward/component/correctness/mean" in agg
        assert "reward/component/format/mean" in agg

    def test_truncation_rate(self) -> None:
        """``rollout/truncation_rate`` averages
        ``finish_reason == 'length'`` over completions."""
        controller = RLTrainer.__new__(RLTrainer)
        completions = [
            _completion(0, 4, 10, finish_reason="length"),
            _completion(0, 4, 10, finish_reason="stop"),
            _completion(1, 4, 10, finish_reason="length"),
            _completion(1, 4, 10, finish_reason="length"),
        ]

        controller.config = MagicMock()
        controller.config.env = MagicMock()
        controller.config.env.build = lambda *, step, group_idx: _FakeEnv({"r": 1.0})
        controller.generator = MagicMock()
        controller._get_rank_0_value = lambda value, has_gpus=True: completions

        _, rollout_metrics = controller._collect_rollouts(num_groups=2, step=0)
        agg = m.aggregate_metrics(rollout_metrics)
        # 3 of 4 completions hit max_tokens.
        assert agg["rollout/truncation_rate/mean"] == pytest.approx(0.75)

    def test_total_length_uses_per_episode_max(self) -> None:
        """``rollout/total_length/max`` must be ``max(prompt+response per
        episode)``, **not** ``max(prompt) + max(response)`` — the latter
        may combine two different episodes."""
        controller = RLTrainer.__new__(RLTrainer)

        # Carefully chosen so per-side maxes don't align: the longest
        # prompt has the shortest response, etc.
        completions = [
            _completion(prompt_idx=0, prompt_len=10, response_len=2),  # total 12
            _completion(prompt_idx=1, prompt_len=2, response_len=10),  # total 12
            _completion(prompt_idx=2, prompt_len=5, response_len=5),  # total 10
        ]
        controller.config = MagicMock()
        controller.config.env = MagicMock()
        controller.config.env.build = lambda *, step, group_idx: _FakeEnv({"r": 1.0})
        controller.generator = MagicMock()
        controller._get_rank_0_value = lambda value, has_gpus=True: completions

        _, rollout_metrics = controller._collect_rollouts(num_groups=3, step=0)
        agg = m.aggregate_metrics(rollout_metrics)
        per_side_max_sum = max(c.prompt_token_ids[-1] + 1 for c in completions) + max(
            c.token_ids[-1] + 1 for c in completions
        )  # = 10 + 10 = 20
        actual_max = max(
            len(c.prompt_token_ids) + len(c.token_ids) for c in completions
        )  # = 12
        assert agg["rollout/total_length/max"] == actual_max
        assert agg["rollout/total_length/max"] < per_side_max_sum


def _trajectory(
    sample_idx: int, prompt_len: int, response_len: int, reward: float
) -> Trajectory:
    completion = _completion(sample_idx, prompt_len, response_len)
    return Trajectory(
        sample_idx=sample_idx,
        transitions=[(completion, _step({"r": reward}))],
    )


class TestBuildEpisodes:
    def test_emits_expected_metric_keys(self) -> None:
        trajectories = [
            _trajectory(0, 4, 5, reward=1.0),
            _trajectory(0, 4, 5, reward=0.0),
            _trajectory(1, 4, 5, reward=0.5),
            _trajectory(1, 4, 5, reward=0.5),
        ]
        episodes, episode_metrics = RLTrainer._build_episodes(trajectories)
        assert len(episodes) == 4
        agg = m.aggregate_metrics(episode_metrics)
        # Stats expansion (5 sub-keys each, top-level reward/advantage).
        for prefix in ("reward", "advantage"):
            for sub in ("max", "mean", "min", "std", "sum"):
                assert f"{prefix}/_{sub}" in agg
        # Group std + degenerate fraction live under reward/.
        assert "reward/group_std/mean" in agg
        assert "reward/group_std/max" in agg
        assert "reward/zero_std_frac" in agg
        # num_prompts/num_episodes were dropped — make sure they don't
        # creep back in.
        assert "rollout/num_prompts" not in agg
        assert "rollout/num_episodes" not in agg

    def test_degenerate_group_fraction(self) -> None:
        # Two groups: both constant => fraction == 1.0.
        all_constant = [
            _trajectory(0, 4, 5, reward=1.0),
            _trajectory(0, 4, 5, reward=1.0),
            _trajectory(1, 4, 5, reward=0.5),
            _trajectory(1, 4, 5, reward=0.5),
        ]
        _, em = RLTrainer._build_episodes(all_constant)
        agg = m.aggregate_metrics(em)
        assert agg["reward/zero_std_frac"] == 1.0

        # Mixed: one group constant (degenerate), one group varied.
        mixed = [
            _trajectory(0, 4, 5, reward=1.0),
            _trajectory(0, 4, 5, reward=1.0),
            _trajectory(1, 4, 5, reward=0.0),
            _trajectory(1, 4, 5, reward=1.0),
        ]
        _, em = RLTrainer._build_episodes(mixed)
        agg = m.aggregate_metrics(em)
        assert agg["reward/zero_std_frac"] == 0.5

        # Both groups have variance => 0.0.
        none_constant = [
            _trajectory(0, 4, 5, reward=0.0),
            _trajectory(0, 4, 5, reward=1.0),
            _trajectory(1, 4, 5, reward=0.5),
            _trajectory(1, 4, 5, reward=1.5),
        ]
        _, em = RLTrainer._build_episodes(none_constant)
        agg = m.aggregate_metrics(em)
        assert agg["reward/zero_std_frac"] == 0.0


# ---------------------------------------------------------------------------
# RLTrainer.Config wiring
# ---------------------------------------------------------------------------


class TestRLTrainerConfigWiring:
    def test_metrics_default_uses_headline_patterns(self) -> None:
        cfg = RLTrainer.Config()
        assert (
            cfg.metrics.train_console_allow_list == _RL_TRAIN_HEADLINE_METRIC_PATTERNS
        )
        assert (
            cfg.metrics.validation_console_allow_list
            == _RL_VALIDATION_HEADLINE_METRIC_PATTERNS
        )

    def test_metrics_defaults_are_independent_copies(self) -> None:
        """Mutating one Config's allow lists must not bleed into other
        instances or into the module constants."""
        cfg = RLTrainer.Config()
        cfg.metrics.train_console_allow_list.append("X")
        cfg.metrics.validation_console_allow_list.append("Y")
        assert "X" not in _RL_TRAIN_HEADLINE_METRIC_PATTERNS
        assert "Y" not in _RL_VALIDATION_HEADLINE_METRIC_PATTERNS
        # And a fresh Config still has the pristine defaults.
        fresh = RLTrainer.Config()
        assert "X" not in fresh.metrics.train_console_allow_list
        assert "Y" not in fresh.metrics.validation_console_allow_list

    def test_metrics_default_wandb_disabled(self) -> None:
        cfg = RLTrainer.Config()
        assert cfg.metrics.enable_wandb is False


# ---------------------------------------------------------------------------
# GRPOLoss bridge
# ---------------------------------------------------------------------------


class TestGRPOLossBridge:
    def test_returns_loss_output_with_token_weighted_sums(self) -> None:
        loss_fn = GRPOLoss(GRPOLoss.Config(clip_eps=0.2))
        # Two samples with **unequal** response lengths, so the
        # token-weighted bridge differs from naive sample averaging.
        policy_logprobs = [
            torch.zeros(2, requires_grad=True),
            torch.zeros(8, requires_grad=True),
        ]
        advantages = torch.tensor([1.0, -1.0])

        out = loss_fn(policy_logprobs=policy_logprobs, advantages=advantages)
        assert isinstance(out, LossOutput)
        # Numerator + denominator are SUMs (token-weighted), not means.
        assert out.num_valid_tokens.item() == 10.0
        for key in (
            "loss/total",
            "loss/pg",
            "loss/ratio/mean",
            "loss/ratio/clipped_frac",
        ):
            assert key in out.token_mean_metric_sums

    def test_loss_total_diverges_from_loss_item_under_unequal_lengths(self) -> None:
        """Document v8/v9 transitional behavior. Under unequal response
        lengths the bridged ``loss/total`` differs from
        ``LossOutput.loss.item()``; this is intentional and the
        loss-migration PR converges them. A future change cannot silently
        re-couple them without updating this test.
        """
        loss_fn = GRPOLoss(GRPOLoss.Config(clip_eps=0.2))
        # Different per-sample logprobs so per-sample pg_loss differs;
        # response lengths differ so the token-weighted sum disagrees
        # with the unweighted sample mean.
        policy_logprobs = [
            torch.full((2,), 0.1, requires_grad=True),
            torch.full((8,), 0.0, requires_grad=True),
        ]
        advantages = torch.tensor([1.0, -1.0])

        out = loss_fn(policy_logprobs=policy_logprobs, advantages=advantages)
        # `loss/total/mean` (after dividing by num_valid_tokens) ≠ loss
        # under unequal lengths.
        token_weighted_loss_total = (
            out.token_mean_metric_sums["loss/total"].item()
            / out.num_valid_tokens.item()
        )
        assert not math.isclose(
            token_weighted_loss_total,
            out.loss.item(),
            rel_tol=1e-4,
            abs_tol=1e-6,
        )


# ---------------------------------------------------------------------------
# Trainer reducers (single-DP fast paths)
# ---------------------------------------------------------------------------


def _stub_trainer_for_reducers(dp_size: int):
    """Build a minimal stand-in for `PolicyTrainer` so we can exercise
    the reducer methods on a CPU box without spawning Monarch / NCCL.
    """
    # Late import: importing `PolicyTrainer` triggers monarch/torchtitan
    # actor wiring at module load time, which is fine for CPU.
    from torchtitan.experiments.rl.actors.trainer import PolicyTrainer

    inst = PolicyTrainer.__new__(PolicyTrainer)
    inst.dp_size = dp_size
    inst.device = torch.device("cpu")
    inst.parallel_dims = MagicMock()
    inst.parallel_dims.get_optional_mesh = MagicMock(return_value=None)
    return inst


class TestReducerFastPaths:
    def test_loss_reducer_single_dp(self) -> None:
        trainer = _stub_trainer_for_reducers(dp_size=1)
        # 4 valid tokens; numerator 12.0 ⇒ mean 3.0
        sums = {"loss/total": torch.tensor(12.0)}
        out = trainer.reduce_token_mean_loss_metrics(
            sums, num_valid_tokens=torch.tensor(4.0)
        )
        assert out == {"loss/total": pytest.approx(3.0)}

    def test_loss_reducer_empty(self) -> None:
        trainer = _stub_trainer_for_reducers(dp_size=1)
        out = trainer.reduce_token_mean_loss_metrics(
            {}, num_valid_tokens=torch.tensor(4.0)
        )
        assert out == {}

    def test_verification_reducer_single_dp_identical(self) -> None:
        trainer = _stub_trainer_for_reducers(dp_size=1)
        out = trainer.reduce_verification_metrics(
            {
                "train/logprob_diff/mean": 0.001,
                "train/logprob_diff/max": 0.005,
                "train/logprob/bitwise_identical": True,
            },
            num_valid_tokens=torch.tensor(4.0),
        )
        assert out["train/logprob_diff/mean"] == pytest.approx(0.001)
        assert out["train/logprob_diff/max"] == pytest.approx(0.005)
        assert out["train/logprob/bitwise_identical"] == 1.0

    def test_verification_reducer_single_dp_not_identical(self) -> None:
        trainer = _stub_trainer_for_reducers(dp_size=1)
        out = trainer.reduce_verification_metrics(
            {
                "train/logprob_diff/mean": 0.0,
                "train/logprob_diff/max": 0.0,
                "train/logprob/bitwise_identical": False,
            },
            num_valid_tokens=torch.tensor(4.0),
        )
        assert out["train/logprob/bitwise_identical"] == 0.0

    def test_loss_reducer_simulated_two_dp(self) -> None:
        """Mock funcol.all_reduce as a lambda that doubles tensors so we
        can verify the pack layout end-to-end without NCCL."""
        trainer = _stub_trainer_for_reducers(dp_size=2)
        trainer.parallel_dims.get_optional_mesh = MagicMock(return_value="batch")

        with patch(
            "torchtitan.experiments.rl.actors.trainer.funcol.all_reduce",
            side_effect=lambda t, op, mesh: t * 2,
        ):
            sums = {
                "loss/total": torch.tensor(6.0),  # local numerator
                "loss/pg": torch.tensor(2.0),
            }
            out = trainer.reduce_token_mean_loss_metrics(
                sums, num_valid_tokens=torch.tensor(3.0)
            )
        # SUM doubled both numerators and denominator => same ratio:
        # loss/total: 12 / 6 = 2.0; loss/pg: 4 / 6 ≈ 0.6667
        assert out["loss/total"] == pytest.approx(2.0)
        assert out["loss/pg"] == pytest.approx(2.0 / 3.0)
