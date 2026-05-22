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

from torchtitan.experiments.rl.grpo import _prepare_reward_metrics, GRPOLoss, RLTrainer
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import Completion, Step, Trajectory


# ---------------------------------------------------------------------------
# _prepare_reward_metrics
# ---------------------------------------------------------------------------


def _step(rewards: dict[str, float]) -> Step:
    return Step(rewards=rewards, done=True)


def _reward_trajectory(rewards: dict[str, float], sample_idx: int = 0) -> Trajectory:
    """Single-turn trajectory with a fake completion + the given rewards."""
    fake_completion = Completion(
        policy_version=0,
        prompt_idx=sample_idx,
        text="",
        token_ids=[],
        token_logprobs=[],
    )
    return Trajectory(
        sample_idx=sample_idx,
        prompt_token_ids=[],
        transitions=[(fake_completion, _step(rewards))],
    )


class TestBuildRewardMetrics:
    def test_one_metric_per_observed_name(self) -> None:
        trajectories = [
            _reward_trajectory({"correctness": 1.0, "format": 0.5}, sample_idx=0),
            _reward_trajectory({"correctness": 0.0, "format": 1.0}, sample_idx=1),
        ]
        metrics = _prepare_reward_metrics("reward/component", trajectories)
        keys = {entry.key for entry in metrics}
        assert keys == {
            "reward/component/correctness",
            "reward/component/format",
        }
        for entry in metrics:
            assert isinstance(entry.value, m.Mean)

    def test_components_observed_in_some_trajectories_only(self) -> None:
        # `format` only appears in the second trajectory — it should
        # average over that one entry (no zero-fill).
        trajectories = [
            _reward_trajectory({"correctness": 1.0}, sample_idx=0),
            _reward_trajectory({"format": 0.5}, sample_idx=1),
        ]
        metrics = _prepare_reward_metrics("reward/component", trajectories)
        agg = m.MetricsProcessor._aggregate_metrics(metrics)
        assert agg["reward/component/correctness/mean"] == 1.0
        assert agg["reward/component/format/mean"] == 0.5

    def test_empty_input(self) -> None:
        assert _prepare_reward_metrics("reward/component", []) == []

    def test_prefix_controls_namespace(self) -> None:
        trajectories = [_reward_trajectory({"correctness": 1.0}, sample_idx=0)]
        metrics = _prepare_reward_metrics("validation/reward/component", trajectories)
        assert metrics[0].key == "validation/reward/component/correctness"


# ---------------------------------------------------------------------------
# Controller subroutines: _collect_rollouts, _build_episodes, validate
# ---------------------------------------------------------------------------


def _completion(
    prompt_idx: int,
    response_len: int,
    finish_reason: str | None = "stop",
    *,
    policy_version: int = 0,
) -> Completion:
    return Completion(
        policy_version=policy_version,
        prompt_idx=prompt_idx,
        text="x" * response_len,
        token_ids=list(range(response_len)),
        token_logprobs=[0.0] * response_len,
        finish_reason=finish_reason,
    )


class _FakeEnv:
    """Minimal env stub: step(text) returns a preset reward dict."""

    def __init__(self, rewards: dict[str, float], prompt: str = "p"):
        self.prompt = prompt
        self._rewards = rewards

    def step(self, text: str) -> Step:
        return _step(self._rewards)


def _build_collect_rollouts_inputs(self_obj):
    """Build a hollow RLTrainer instance + completions wired into the
    fake generator — without spawning real meshes.
    """

    completions = [
        _completion(prompt_idx=0, response_len=10),
        _completion(prompt_idx=0, response_len=20),
        _completion(prompt_idx=1, response_len=15),
    ]

    class _RewardEnvBuilder:
        @staticmethod
        def build(*, step, group_idx):
            return _FakeEnv({"correctness": float(group_idx), "format": 0.5})

    self_obj.config = MagicMock()
    self_obj.config.env = _RewardEnvBuilder
    self_obj.tokenizer = MagicMock()
    self_obj.tokenizer.encode.side_effect = lambda prompt, **_: [ord(prompt)]
    self_obj.generator = MagicMock()
    # `_get_rank_0_value` is the layer that strips Monarch's ValueMesh,
    # so just make it return whatever it's handed.
    self_obj._get_rank_0_value = lambda value, has_gpus=True: (completions, [])
    return completions


class TestCollectRollouts:
    def test_passes_token_ids_to_generator(self) -> None:
        """Controller tokenizes env prompts and hands the IDs (not strings)
        to ``generator.generate.call``."""
        controller = RLTrainer.__new__(RLTrainer)
        _build_collect_rollouts_inputs(controller)
        controller._collect_rollouts(num_groups=2, step=0)
        # _FakeEnv.prompt == "p"; encode side_effect returns [ord(prompt)] = [112].
        controller.generator.generate.call.assert_called_once_with([[112], [112]])

    def test_emits_expected_metric_keys(self) -> None:
        controller = RLTrainer.__new__(RLTrainer)
        completions = _build_collect_rollouts_inputs(controller)
        trajectories, rollout_metrics = controller._collect_rollouts(
            num_groups=2, step=0
        )
        assert len(trajectories) == len(completions)
        agg = m.MetricsProcessor._aggregate_metrics(rollout_metrics)
        # Length keys: Mean+Max for prompt/response, Max-only for total.
        assert "rollout/response_length/mean" in agg
        assert "rollout/response_length/max" in agg
        assert "rollout/prompt_length/mean" in agg
        assert "rollout/prompt_length/max" in agg
        assert "rollout/total_length/max" in agg
        # Reward-component keys derived from env step output (now under
        # the top-level reward/ namespace).
        assert "reward/component/correctness/mean" in agg
        assert "reward/component/format/mean" in agg

    def test_truncation_rate(self) -> None:
        """rollout/truncation_rate averages
        finish_reason == 'length' over completions."""
        controller = RLTrainer.__new__(RLTrainer)
        completions = [
            _completion(0, 10, finish_reason="length"),
            _completion(0, 10, finish_reason="stop"),
            _completion(1, 10, finish_reason="length"),
            _completion(1, 10, finish_reason="length"),
        ]

        controller.config = MagicMock()
        controller.config.env = MagicMock()
        controller.config.env.build = lambda *, step, group_idx: _FakeEnv({"r": 1.0})
        controller.tokenizer = MagicMock()
        controller.tokenizer.encode.side_effect = lambda prompt, **_: [ord(prompt)]
        controller.generator = MagicMock()
        controller._get_rank_0_value = lambda value, has_gpus=True: (completions, [])

        _, rollout_metrics = controller._collect_rollouts(num_groups=2, step=0)
        agg = m.MetricsProcessor._aggregate_metrics(rollout_metrics)
        # 3 of 4 completions hit max_tokens.
        assert agg["rollout/truncation_rate/mean"] == pytest.approx(0.75)

    def test_total_length_uses_per_episode_max(self) -> None:
        """rollout/total_length/max must be max(prompt+response per
        episode), **not** max(prompt) + max(response) — the latter
        may combine two different episodes."""
        controller = RLTrainer.__new__(RLTrainer)

        # Carefully chosen so per-side maxes don't align: the longest
        # prompt has the shortest response, etc. The tokenizer mock below
        # produces token lists of length == len(prompt), so the env prompts
        # control prompt-side lengths.
        env_prompts = {0: "x" * 10, 1: "x" * 2, 2: "x" * 5}
        completions = [
            _completion(prompt_idx=0, response_len=2),  # total 10 + 2 = 12
            _completion(prompt_idx=1, response_len=10),  # total 2 + 10 = 12
            _completion(prompt_idx=2, response_len=5),  # total 5 + 5 = 10
        ]
        controller.config = MagicMock()
        controller.config.env = MagicMock()
        controller.config.env.build = lambda *, step, group_idx: _FakeEnv(
            {"r": 1.0}, prompt=env_prompts[group_idx]
        )
        controller.tokenizer = MagicMock()
        controller.tokenizer.encode.side_effect = lambda prompt, **_: list(
            prompt.encode()
        )
        controller.generator = MagicMock()
        controller._get_rank_0_value = lambda value, has_gpus=True: (completions, [])

        trajectories, rollout_metrics = controller._collect_rollouts(
            num_groups=3, step=0
        )
        agg = m.MetricsProcessor._aggregate_metrics(rollout_metrics)
        per_side_max_sum = max(len(t.prompt_token_ids) for t in trajectories) + max(
            len(c.token_ids) for c in completions
        )  # = 10 + 10 = 20
        actual_max = max(
            len(t.prompt_token_ids) + len(c.token_ids)
            for t, c in zip(trajectories, completions, strict=True)
        )  # = 12
        assert agg["rollout/total_length/max"] == actual_max
        assert agg["rollout/total_length/max"] < per_side_max_sum


def _trajectory(
    sample_idx: int,
    prompt_len: int,
    response_len: int,
    reward: float,
    *,
    policy_version: int = 0,
) -> Trajectory:
    completion = _completion(sample_idx, response_len, policy_version=policy_version)
    return Trajectory(
        sample_idx=sample_idx,
        prompt_token_ids=list(range(prompt_len)),
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
        agg = m.MetricsProcessor._aggregate_metrics(episode_metrics)
        # SummaryStats expansion (5 sub-keys each, top-level reward/advantage).
        for prefix in ("reward", "advantage"):
            for sub in ("max", "mean", "min", "std", "sum"):
                assert f"{prefix}/_{sub}" in agg
        # Group std + degenerate fraction live under reward/.
        assert "reward/group_std/mean" in agg
        assert "reward/group_std/max" in agg
        assert "reward/zero_std_frac" in agg
        # Per-rollout policy version distribution (min/max only).
        assert "rollout/policy_version/mean" not in agg
        assert "rollout/policy_version/min" in agg
        assert "rollout/policy_version/max" in agg
        # num_prompts/num_episodes were dropped: make sure they don't
        # creep back in.
        assert "rollout/num_prompts" not in agg
        assert "rollout/num_episodes" not in agg

    def test_policy_version_metrics_single_version(self) -> None:
        """When all rollouts came from the same policy version, min == max."""
        single_version = [
            _trajectory(0, 4, 5, reward=1.0, policy_version=5),
            _trajectory(1, 4, 5, reward=0.5, policy_version=5),
        ]
        _, em = RLTrainer._build_episodes(single_version)
        agg = m.MetricsProcessor._aggregate_metrics(em)
        assert agg["rollout/policy_version/min"] == 5.0
        assert agg["rollout/policy_version/max"] == 5.0

    def test_policy_version_metrics_mixed_versions(self) -> None:
        """Mixed rollout versions emit min and max."""
        mixed_versions = [
            _trajectory(0, 4, 5, reward=1.0, policy_version=2),
            _trajectory(1, 4, 5, reward=0.5, policy_version=4),
        ]
        _, em = RLTrainer._build_episodes(mixed_versions)
        agg = m.MetricsProcessor._aggregate_metrics(em)
        assert agg["rollout/policy_version/min"] == 2.0
        assert agg["rollout/policy_version/max"] == 4.0

    def test_degenerate_group_fraction(self) -> None:
        # Two groups: both constant => fraction == 1.0.
        all_constant = [
            _trajectory(0, 4, 5, reward=1.0),
            _trajectory(0, 4, 5, reward=1.0),
            _trajectory(1, 4, 5, reward=0.5),
            _trajectory(1, 4, 5, reward=0.5),
        ]
        _, em = RLTrainer._build_episodes(all_constant)
        agg = m.MetricsProcessor._aggregate_metrics(em)
        assert agg["reward/zero_std_frac"] == 1.0

        # Mixed: one group constant (degenerate), one group varied.
        mixed = [
            _trajectory(0, 4, 5, reward=1.0),
            _trajectory(0, 4, 5, reward=1.0),
            _trajectory(1, 4, 5, reward=0.0),
            _trajectory(1, 4, 5, reward=1.0),
        ]
        _, em = RLTrainer._build_episodes(mixed)
        agg = m.MetricsProcessor._aggregate_metrics(em)
        assert agg["reward/zero_std_frac"] == 0.5

        # Both groups have variance => 0.0.
        none_constant = [
            _trajectory(0, 4, 5, reward=0.0),
            _trajectory(0, 4, 5, reward=1.0),
            _trajectory(1, 4, 5, reward=0.5),
            _trajectory(1, 4, 5, reward=1.5),
        ]
        _, em = RLTrainer._build_episodes(none_constant)
        agg = m.MetricsProcessor._aggregate_metrics(em)
        assert agg["reward/zero_std_frac"] == 0.0


# ---------------------------------------------------------------------------
# RLTrainer.Config wiring
# ---------------------------------------------------------------------------


class TestRLTrainerConfigWiring:
    """Use the canonical `rl_grpo_qwen3_0_6b` registry config so the test
    matches a real production config and stays insulated from any future
    tightening of VLLMGenerator/PolicyTrainer field validators."""

    def test_metrics_default_uses_factory(self) -> None:
        from torchtitan.experiments.rl.config_registry import rl_grpo_qwen3_0_6b

        cfg = rl_grpo_qwen3_0_6b()
        baseline = m.MetricsProcessor.Config()
        assert cfg.metrics.console_log_keys_train == baseline.console_log_keys_train
        assert (
            cfg.metrics.console_log_keys_validation
            == baseline.console_log_keys_validation
        )

    def test_metrics_defaults_are_independent_copies(self) -> None:
        """Mutating one Config's allow lists must not bleed into other instances."""
        from torchtitan.experiments.rl.config_registry import rl_grpo_qwen3_0_6b

        cfg = rl_grpo_qwen3_0_6b()
        cfg.metrics.console_log_keys_train.append("X")
        cfg.metrics.console_log_keys_validation.append("Y")
        # A fresh Config still has the pristine defaults.
        fresh = rl_grpo_qwen3_0_6b()
        assert "X" not in fresh.metrics.console_log_keys_train
        assert "Y" not in fresh.metrics.console_log_keys_validation

    def test_metrics_default_wandb_enabled(self) -> None:
        from torchtitan.experiments.rl.config_registry import rl_grpo_qwen3_0_6b

        cfg = rl_grpo_qwen3_0_6b()
        assert cfg.metrics.enable_wandb is True
        assert cfg.metrics.enable_tensorboard is False


# ---------------------------------------------------------------------------
# GRPOLoss bridge
# ---------------------------------------------------------------------------


class TestGRPOLossBridge:
    def test_loss_keeps_gradient(self) -> None:
        """`loss` must remain differentiable so `.backward()` works.
        Regression test for `_token_weighted_mean` accidentally detaching."""
        loss_fn = GRPOLoss(GRPOLoss.Config(clip_eps=0.2))
        policy_logprobs = [
            torch.zeros(2, requires_grad=True),
            torch.zeros(8, requires_grad=True),
        ]

        loss, _loss_metrics = loss_fn(
            policy_logprobs=policy_logprobs,
            advantages=torch.tensor([1.0, -1.0]),
            num_global_valid_tokens=torch.tensor(10.0),
        )

        assert loss.requires_grad
        assert loss.grad_fn is not None
        loss.backward()
        assert all(sample.grad is not None for sample in policy_logprobs)

    def test_returns_loss_and_pre_normalized_metrics(self) -> None:
        loss_fn = GRPOLoss(GRPOLoss.Config(clip_eps=0.2))
        # Two samples with unequal response lengths.
        policy_logprobs = [
            torch.zeros(2, requires_grad=True),
            torch.zeros(8, requires_grad=True),
        ]
        advantages = torch.tensor([1.0, -1.0])
        # Single-rank case: global == local valid tokens.
        num_global_valid_tokens = torch.tensor(10.0)

        loss, loss_metrics = loss_fn(
            policy_logprobs=policy_logprobs,
            advantages=advantages,
            num_global_valid_tokens=num_global_valid_tokens,
        )
        assert isinstance(loss, torch.Tensor)
        assert isinstance(loss_metrics, dict)
        for key in ("loss/mean", "loss/ratio/mean", "loss/ratio/clipped_frac"):
            assert key in loss_metrics

    def test_loss_is_token_weighted_sum_over_global_tokens(self) -> None:
        """loss = sum_i(sample_loss_i * num_tokens_i) / num_global_valid_tokens.

        Under unequal response lengths this differs from a naive sample mean.
        """
        loss_fn = GRPOLoss(GRPOLoss.Config(clip_eps=0.2))
        policy_logprobs = [
            torch.full((2,), 0.1, requires_grad=True),
            torch.full((8,), 0.0, requires_grad=True),
        ]
        advantages = torch.tensor([1.0, -1.0])
        num_global_valid_tokens = torch.tensor(10.0)

        loss, loss_metrics = loss_fn(
            policy_logprobs=policy_logprobs,
            advantages=advantages,
            num_global_valid_tokens=num_global_valid_tokens,
        )
        # loss/mean metric is the same value as loss (both pre-normalized).
        assert math.isclose(
            loss_metrics["loss/mean"].item(),
            loss.item(),
            rel_tol=1e-6,
        )

        # And it is NOT equal to the unweighted sample mean of policy gradient
        # losses, which is what the prior implementation used.
        per_sample_mean_logprobs = torch.stack(
            [sample_logprobs.mean() for sample_logprobs in policy_logprobs]
        )
        ratio = torch.exp(per_sample_mean_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
        sample_policy_gradient_losses = -torch.min(
            ratio * advantages, clipped_ratio * advantages
        )
        unweighted_sample_mean = float(sample_policy_gradient_losses.mean().item())
        assert not math.isclose(
            loss.item(), unweighted_sample_mean, rel_tol=1e-4, abs_tol=1e-6
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
    def test_single_dp_identical(self) -> None:
        # Pre-normalized values pass through SUM-reduce unchanged on a single
        # rank (no mesh -> no all-reduce -> values are exactly what we passed).
        trainer = _stub_trainer_for_reducers(dp_size=1)
        out = trainer.reduce_forward_backward_metrics(
            sum_reduced_metrics={
                "loss/mean": torch.tensor(3.0),
                "bit_wise/logprob_diff/mean": torch.tensor(0.001),
                "bit_wise/ratio_tokens_different/mean": torch.tensor(0.0),
            },
            max_reduced_metrics={"bit_wise/logprob_diff/max": torch.tensor(0.005)},
        )
        assert out["loss/mean"] == pytest.approx(3.0)
        assert out["bit_wise/logprob_diff/mean"] == pytest.approx(0.001)
        assert out["bit_wise/logprob_diff/max"] == pytest.approx(0.005)
        assert out["bit_wise/ratio_tokens_different/mean"] == 0.0

    def test_unbiased_sum_reduction_across_ranks(self) -> None:
        """Two ranks contribute pre-normalized shares; SUM-reducing
        reconstructs the global value.

        Rank 0 shares: loss/mean=10/15 (token-weighted local share).
        Rank 1 shares: loss/mean=30/15.
        SUM-reduce: 40/15 = 2.667 (the global token-weighted mean).
        """
        trainer = _stub_trainer_for_reducers(dp_size=2)
        trainer.parallel_dims.get_optional_mesh = MagicMock(return_value="loss")

        rank0_share = torch.tensor([10.0 / 15.0], dtype=torch.float32)
        rank1_share = torch.tensor([30.0 / 15.0], dtype=torch.float32)

        def fake_all_reduce(t, *, reduceOp, group):
            if t.numel() == 1 and t.dtype == torch.float32:
                return rank0_share + rank1_share
            return t

        with patch(
            "torchtitan.experiments.rl.actors.trainer.funcol.all_reduce",
            side_effect=fake_all_reduce,
        ):
            out = trainer.reduce_forward_backward_metrics(
                sum_reduced_metrics={"loss/mean": rank0_share[0]},
                max_reduced_metrics={"bit_wise/logprob_diff/max": torch.tensor(0.0)},
            )
        assert out["loss/mean"] == pytest.approx(40.0 / 15.0)

    def test_max_reduce_path(self) -> None:
        """MAX-reduced metrics compose via elementwise max across ranks.

        Patches funcol.all_reduce to dispatch on reduceOp: SUM doubles
        (simulating two ranks contributing equal shares); MAX takes the
        elementwise max with a higher second-rank value.
        """
        import torch.distributed.distributed_c10d as c10d

        trainer = _stub_trainer_for_reducers(dp_size=2)
        trainer.parallel_dims.get_optional_mesh = MagicMock(return_value="loss")

        rank1_max = torch.tensor([0.006], dtype=torch.float32)

        def fake_all_reduce(t, *, reduceOp, group):
            if reduceOp == c10d.ReduceOp.SUM.name:
                return t * 2
            if reduceOp == c10d.ReduceOp.MAX.name:
                return torch.maximum(t, rank1_max)
            raise AssertionError(f"unexpected reduceOp={reduceOp!r}")

        with patch(
            "torchtitan.experiments.rl.actors.trainer.funcol.all_reduce",
            side_effect=fake_all_reduce,
        ):
            out = trainer.reduce_forward_backward_metrics(
                sum_reduced_metrics={"loss/mean": torch.tensor(0.5)},
                max_reduced_metrics={"bit_wise/logprob_diff/max": torch.tensor(0.003)},
            )
        # SUM doubled: 0.5 + 0.5 = 1.0. MAX = max(0.003, 0.006) = 0.006.
        assert out["loss/mean"] == pytest.approx(1.0)
        assert out["bit_wise/logprob_diff/max"] == pytest.approx(0.006)

    def test_sum_only_skips_max_collective(self) -> None:
        """max_reduced_metrics={} must not crash and must not call the
        MAX collective; the SUM bucket is still reduced normally."""
        import torch.distributed.distributed_c10d as c10d

        trainer = _stub_trainer_for_reducers(dp_size=2)
        trainer.parallel_dims.get_optional_mesh = MagicMock(return_value="loss")

        seen_ops: list[str] = []

        def fake_all_reduce(t, *, reduceOp, group):
            seen_ops.append(reduceOp)
            if reduceOp == c10d.ReduceOp.SUM.name:
                return t * 2
            raise AssertionError(f"unexpected reduceOp={reduceOp!r}")

        with patch(
            "torchtitan.experiments.rl.actors.trainer.funcol.all_reduce",
            side_effect=fake_all_reduce,
        ):
            out = trainer.reduce_forward_backward_metrics(
                sum_reduced_metrics={"loss/mean": torch.tensor(0.5)},
                max_reduced_metrics={},
            )
        assert seen_ops == [c10d.ReduceOp.SUM.name]
        assert out == {"loss/mean": pytest.approx(1.0)}

    def test_max_only_skips_sum_collective(self) -> None:
        """sum_reduced_metrics={} must not crash and must not call the
        SUM collective; the MAX bucket is still reduced normally."""
        import torch.distributed.distributed_c10d as c10d

        trainer = _stub_trainer_for_reducers(dp_size=2)
        trainer.parallel_dims.get_optional_mesh = MagicMock(return_value="loss")

        seen_ops: list[str] = []
        rank1_max = torch.tensor([0.006], dtype=torch.float32)

        def fake_all_reduce(t, *, reduceOp, group):
            seen_ops.append(reduceOp)
            if reduceOp == c10d.ReduceOp.MAX.name:
                return torch.maximum(t, rank1_max)
            raise AssertionError(f"unexpected reduceOp={reduceOp!r}")

        with patch(
            "torchtitan.experiments.rl.actors.trainer.funcol.all_reduce",
            side_effect=fake_all_reduce,
        ):
            out = trainer.reduce_forward_backward_metrics(
                sum_reduced_metrics={},
                max_reduced_metrics={
                    "bit_wise/logprob_diff/max": torch.tensor(0.003),
                },
            )
        assert seen_ops == [c10d.ReduceOp.MAX.name]
        assert out == {"bit_wise/logprob_diff/max": pytest.approx(0.006)}

    def test_both_empty_returns_empty(self) -> None:
        """Both buckets empty: no collectives called, empty dict returned."""
        trainer = _stub_trainer_for_reducers(dp_size=2)
        trainer.parallel_dims.get_optional_mesh = MagicMock(return_value="loss")
        with patch(
            "torchtitan.experiments.rl.actors.trainer.funcol.all_reduce",
            side_effect=AssertionError("should not be called"),
        ):
            out = trainer.reduce_forward_backward_metrics(
                sum_reduced_metrics={},
                max_reduced_metrics={},
            )
        assert out == {}
