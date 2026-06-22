# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Coding-agent (Claude Code) rollouter for R2E-Gym SWE tasks.

Unlike the env-driven rollouters (search_r1, alphabet_sort), the agent loop runs
*inside the sandbox*: Claude Code drives tool calls / file edits itself and talks
to the on-box Anthropic adapter for each model turn. So this rollouter overrides
``run_group_rollouts`` to, per sibling:

    1. open an adapter session keyed by the rollout id;
    2. boot a fresh sandbox from the task image + install the toolchain;
    3. run ``claude -p`` against the adapter (via the Daytona file-relay bridge);
    4. capture the agent's patch (``git diff``) and grade it in a clean sandbox;
    5. drain the adapter's captured turns into ``RolloutTurn``s and stamp the
       grade onto the last turn's ``env_rewards`` (read back by ``RewardR2E``).

The standard rubric + advantage + ``rollout_to_episodes`` path then applies: each
turn's prompt exactly extends ``prev_prompt + prev_completion`` (the adapter uses
Token-In-Token-Out bridging), so a whole trajectory packs into one episode with
assistant tokens trained and prompt / tool-result tokens masked out.

Knobs read from env (the launcher sets these; see ``run_swe_r2e_*.sh``):
  ``SHIM_BIND_HOST`` / ``SHIM_PORT``  adapter bind address (default 127.0.0.1:18001)
  ``SWE_TIME_BUDGET_SEC``             per-agent wallclock (default 1200)
  ``SWE_EVAL_TIMEOUT_SEC``            per-eval test run (default 400)
  ``SWE_MAX_CONTEXT_LEN``             model context budget for per-turn gen cap
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from renderers import Renderer

from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.swe_r2e.data import SWER2EDataset, SWER2ESample
from torchtitan.experiments.rl.examples.swe_r2e.env import SWER2EEnv
from torchtitan.experiments.rl.examples.swe_r2e.grading import evaluate_r2e
from torchtitan.experiments.rl.examples.swe_r2e.rubric import R2E_REWARD_KEY, RewardR2E
from torchtitan.experiments.rl.harness import (
    AnthropicAdapter,
    boot_agent_sandbox,
    git_diff,
    run_claude_code,
)
from torchtitan.experiments.rl.rollout.advantage import AdvantageEstimator
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout.types import (
    GenerateFn,
    Rollout,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.rubrics import Rubric

if TYPE_CHECKING:
    # Type-only: importing the generator module pulls in vLLM at import time.
    from torchtitan.experiments.rl.actors.generator import SamplingConfig

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val and val.strip() else default


class SWER2ERollouter(Rollouter):
    """Drives Claude Code in a sandbox per sibling, then grades the patch (R2E)."""

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: SWER2EDataset.Config = field(
            default_factory=lambda: SWER2EDataset.Config(seed=42)
        )
        validation_dataset: SWER2EDataset.Config = field(
            default_factory=lambda: SWER2EDataset.Config(seed=99, shuffle=False)
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[RewardR2E.Config(weight=1.0)],
                # An errored / timed-out agent has no valid patch -> no learning signal.
                error_reward=0.0,
                truncation_reward=0.0,
            )
        )
        # Placeholder env (the agent loop runs in-sandbox; see env.py).
        message_env: SWER2EEnv.Config = field(default_factory=SWER2EEnv.Config)
        token_env: TokenEnv.Config = field(default_factory=TokenEnv.Config)
        advantage: AdvantageEstimator.Config = field(
            default_factory=lambda: AdvantageEstimator.Config(should_std_normalize=True)
        )

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._shim_host = os.environ.get("SHIM_BIND_HOST", "127.0.0.1")
        self._shim_port = _env_int("SHIM_PORT", 18001)
        self._time_budget_sec = _env_int("SWE_TIME_BUDGET_SEC", 1200)
        self._eval_timeout_sec = _env_int("SWE_EVAL_TIMEOUT_SEC", 400)
        self._max_context_tokens = _env_int("SWE_MAX_CONTEXT_LEN", 32768)
        # Whole-rollout wall-clock guard: agent budget + eval + boot/diff buffer.
        self._guard_sec = self._time_budget_sec + self._eval_timeout_sec + 300
        self._adapter: AnthropicAdapter | None = None
        self._adapter_lock = asyncio.Lock()

    async def _ensure_adapter(self, renderer: Renderer) -> AnthropicAdapter:
        if self._adapter is None:
            async with self._adapter_lock:
                if self._adapter is None:
                    adapter = AnthropicAdapter(
                        renderer=renderer, host=self._shim_host, port=self._shim_port
                    )
                    await adapter.start()
                    self._adapter = adapter
        return self._adapter

    async def run_group_rollouts(
        self,
        *,
        generate_fn: GenerateFn,
        sample: SWER2ESample,
        group_id: str,
        group_size: int,
        sampling: "SamplingConfig",
        renderer: Renderer,
    ) -> RolloutGroup:
        """Run + grade one prompt group of Claude Code coding-agent rollouts."""
        adapter = await self._ensure_adapter(renderer)

        rollouts = await asyncio.gather(
            *(
                self._run_claude_rollout(
                    adapter=adapter,
                    generate_fn=generate_fn,
                    sample=sample,
                    group_id=group_id,
                    rollout_id=f"{group_id}/sample={i}",
                    sampling=sampling,
                    renderer=renderer,
                )
                for i in range(group_size)
            )
        )

        # Standard scoring + advantage path (mirrors Rollouter.run_group_rollouts).
        outputs = await self.score_group(rollouts, sample)
        for rollout, output in zip(rollouts, outputs, strict=True):
            rollout.reward = output.reward
            rollout.reward_breakdown = output.reward_breakdown

        group = RolloutGroup(group_id=group_id, rollouts=rollouts)
        advantages = self.advantage_estimator(group)
        for rollout, advantage in zip(group.rollouts, advantages, strict=True):
            rollout.advantage = advantage
        return group

    async def _run_claude_rollout(
        self,
        *,
        adapter: AnthropicAdapter,
        generate_fn: GenerateFn,
        sample: SWER2ESample,
        group_id: str,
        rollout_id: str,
        sampling: "SamplingConfig",
        renderer: Renderer,
    ) -> Rollout:
        """Boot a sandbox, run Claude Code against the adapter, grade the patch.

        Always returns a ``Rollout`` (errors are caught and marked terminal) so one
        bad sibling never fails the whole group.
        """
        adapter.open_session(
            rollout_id,
            generate_fn=generate_fn,
            sampling=sampling,
            routing_session_id=rollout_id,
            max_context_tokens=self._max_context_tokens,
        )

        status = RolloutStatus.ERROR
        reward = 0.0
        solved = False
        applied = False
        diff_text = ""
        try:
            async with asyncio.timeout(self._guard_sec):
                async with boot_agent_sandbox(sample.image) as sb:
                    await run_claude_code(
                        sb,
                        workdir=sample.workdir,
                        session_id=rollout_id,
                        adapter_url=adapter.url,
                        time_budget_sec=self._time_budget_sec,
                        problem_statement=sample.problem_statement,
                        pre_commands=sample.pre_commands,
                    )
                    diff_text = await git_diff(sb, sample.workdir, tracked_only=True)

                reward, solved, applied = await evaluate_r2e(
                    image=sample.image,
                    workdir=sample.workdir,
                    diff_text=diff_text,
                    r2e=sample.r2e,
                    pre_commands=sample.pre_commands,
                    timeout_sec=self._eval_timeout_sec,
                )
                status = RolloutStatus.COMPLETED
        except (TimeoutError, asyncio.TimeoutError):
            logger.warning("[swe_r2e] %s: wall-clock guard fired", rollout_id)
            status = RolloutStatus.ERROR_TIMEOUT
        except Exception:
            logger.exception("[swe_r2e] %s: rollout failed", rollout_id)
            status = RolloutStatus.ERROR
        finally:
            captured = await adapter.finish_session(rollout_id)

        # Drop empty-completion turns so rollout_to_episodes only sees trainable
        # turns (a non-final empty completion would otherwise raise).
        turns: list[RolloutTurn] = [
            RolloutTurn(
                prompt_token_ids=ct.prompt_token_ids,
                completion_token_ids=ct.completion_token_ids,
                completion_logprobs=ct.completion_logprobs,
                policy_version=ct.policy_version,
            )
            for ct in captured
            if ct.completion_token_ids
        ]

        if not turns:
            # No trainable tokens (agent never produced a usable turn).
            status = RolloutStatus.ERROR
        else:
            turns[-1].env_rewards = {
                R2E_REWARD_KEY: float(reward),
                "r2e_solved": float(solved),
                "r2e_applied": float(applied),
            }

        logger.info(
            "[swe_r2e] %s: status=%s reward=%.2f solved=%s applied=%s turns=%d",
            rollout_id,
            status,
            reward,
            solved,
            applied,
            len(turns),
        )
        self._maybe_dump_trace(
            rollout_id=rollout_id,
            sample=sample,
            captured=captured,
            renderer=renderer,
            status=str(status),
            reward=reward,
            solved=solved,
            applied=applied,
            diff_text=diff_text,
        )
        return Rollout(
            group_id=group_id,
            sample_id=rollout_id,
            status=status,
            turns=turns,
        )

    def _maybe_dump_trace(
        self,
        *,
        rollout_id: str,
        sample: SWER2ESample,
        captured: list,
        renderer: Renderer,
        status: str,
        reward: float,
        solved: bool,
        applied: bool,
        diff_text: str,
    ) -> None:
        """Write a human-readable per-rollout training trace when
        ``SWE_ROLLOUT_DUMP_DIR`` is set: the R2E task, grade, and every captured
        turn's decoded model completion + token lengths + finish reason. This is
        the *training view* (what the model generated and trains on each turn),
        complementing Claude Code's own stream-json agent trace under
        ``SWE_TRAJECTORY_DUMP_DIR``. Best-effort; never raises into the rollout."""
        dump_dir = os.environ.get("SWE_ROLLOUT_DUMP_DIR", "")
        if not dump_dir:
            return
        try:
            tokenizer = getattr(renderer, "tokenizer", None) or getattr(
                renderer, "_tokenizer", None
            )

            def _decode(ids: list[int]) -> str:
                if tokenizer is None or not ids:
                    return ""
                return tokenizer.decode(ids, skip_special_tokens=False)

            record = {
                "rollout_id": rollout_id,
                "instance_id": sample.instance_id,
                "image": sample.image,
                "status": status,
                "reward": reward,
                "solved": solved,
                "applied": applied,
                "num_turns": len(captured),
                "diff": diff_text,
                "turns": [
                    {
                        "turn": i,
                        "prompt_tokens": len(ct.prompt_token_ids),
                        "completion_tokens": len(ct.completion_token_ids),
                        "finish_reason": ct.finish_reason,
                        "extends_previous": ct.extends_previous,
                        # The model's generated text this turn (the trained tokens).
                        "completion_text": _decode(ct.completion_token_ids),
                    }
                    for i, ct in enumerate(captured)
                ],
            }
            os.makedirs(dump_dir, exist_ok=True)
            safe = rollout_id.replace("/", "_")
            path = os.path.join(dump_dir, f"{safe}.json")
            with open(path, "w") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
            logger.info("[swe_r2e] rollout trace dumped: %s", path)
        except Exception as e:
            logger.warning("[swe_r2e] rollout trace dump failed: %s", e)
