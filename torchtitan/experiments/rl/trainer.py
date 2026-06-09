# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RL trainer used for synchronous grpo training.
"""

import asyncio
import logging
import math
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

# must run before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchstore as ts
from monarch.actor import ProcMesh
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.config import CompileConfig, Configurable
from torchtitan.experiments.rl.actors.generator import (
    Completion,
    SamplingConfig,
    VLLMGenerator,
)
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.batcher import Batcher
from torchtitan.experiments.rl.environment import TokenEnv, TokenEnvOutput
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.renderer import RendererConfig
from torchtitan.experiments.rl.rollout import (
    last_completion_text,
    prepare_rollout_metrics,
    Rollout,
    rollout_to_episode,
    RolloutGroup,
    RolloutStatus,
    RolloutTurn,
)
from torchtitan.experiments.rl.rollout.rollouter import Rollouter
from torchtitan.experiments.rl.rollout_recorder import RolloutSampleRecorder
from torchtitan.experiments.rl.generator_router import GeneratorRouter, RouteContext
from torchtitan.experiments.rl.types import Episode
from torchtitan.observability import structured_logger as sl
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


class GRPOLoss(Configurable):
    """Per-token clipped surrogate loss for GRPO.

    Computes the PPO-style clipped objective at the token level::

        ratio_t = exp(policy_logprob_t - ref_logprob_t)     # π_θ / π_old
        clipped_t = clamp(ratio_t, 1 - ε, 1 + ε)
        loss_t = -min(ratio_t * A_t, clipped_t * A_t)

    The final scalar loss is the sum of per-token losses over loss
    positions (where ``loss_mask == 1``), divided by
    ``num_global_valid_tokens`` (total loss positions across all
    microbatches and DP ranks).  This normalization ensures that
    gradient accumulation across microbatches produces the same
    result as a single large-batch forward pass.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_eps: float = 0.2
        """PPO clipping epsilon for the probability ratio."""

    def __init__(self, config: Config):
        self.clip_eps = config.clip_eps

    def __call__(
        self,
        policy_logprobs: torch.Tensor,
        generator_logprobs: torch.Tensor,
        loss_mask: torch.Tensor,
        advantages: torch.Tensor,
        num_global_valid_tokens: int,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute per-token GRPO clipped surrogate loss.

        Args:
            policy_logprobs: [B, L] log π_θ(a_t | s_t) from the current policy.
            generator_logprobs: [B, L] log π_old(a_t | s_t) from the sampling policy.
            loss_mask: [B, L] bool mask; True for response tokens.
            advantages: [B, L] per-token advantages (0.0 for prompt/padding).
            num_global_valid_tokens: total response tokens across all microbatches
                and DP ranks; used as the loss denominator so gradient
                accumulation is equivalent to a single large-batch step.

        Returns:
            (loss, metrics) where loss is a scalar tensor and metrics is a
            dict of scalar tensors pre-normalized for SUM reduction across
            DP ranks.
        """
        # Per-token importance sampling ratio: π_θ / π_old
        log_ratio = policy_logprobs - generator_logprobs
        ratio = torch.exp(log_ratio)

        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        token_pg_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        masked_loss = token_pg_loss * loss_mask
        loss_denominator = max(num_global_valid_tokens, 1)
        loss = masked_loss.sum() / loss_denominator

        with torch.no_grad():
            masked_ratio = ratio * loss_mask
            metrics = {
                "loss/mean": loss.detach(),
                "loss/ratio_mean": masked_ratio.sum() / loss_denominator,
                "loss/ratio_clipped_frac": (
                    (torch.abs(ratio - clipped_ratio) > 1e-6).float() * loss_mask
                ).sum()
                / loss_denominator,
            }

        return loss, metrics


def _log_samples(rollout_groups: list[RolloutGroup]) -> None:
    """Log the first scored, trainable rollout per group for debugging."""
    for group in rollout_groups:
        rollout = next(
            (
                r
                for r in group.rollouts
                if r.reward is not None and r.turns and r.turns[0].completion_token_ids
            ),
            None,
        )
        if rollout is None:
            continue
        logger.info("  [%s] reward=%+.1f", group.group_id, rollout.reward)
        logger.info(
            "       A: %s",
            last_completion_text(rollout)[:300].replace("\n", " ").strip(),
        )


def _sample_id(group_id: str, sample_idx: int) -> str:
    return f"{group_id}/sample={sample_idx}"


@dataclass(kw_only=True, slots=True)
class _RolloutGroupState:
    """A prompt group's working state across one rollout collection call."""

    group_id: str
    sample: object
    envs: list[TokenEnv]  # [group_size]
    env_init_outputs: list[TokenEnvOutput] = field(default_factory=list)
    completions: list[Completion] | None = None


class RLTrainer(Configurable):
    """Top-level RL training orchestrator.

    Owns a `PolicyTrainer` actor (gradient updates), a `VLLMGenerator` actor
    (sampling), and a `Rollouter` (datasets + rubric + env construction). Each
    training step samples groups of rollouts, scores them via the rollouter's rubric,
    builds GRPO advantages, and syncs trainer weights to the generator.

    Example:

        cfg = config_registry.rl_grpo_qwen3_0_6b_varlen()
        trainer = cfg.build()
        await trainer.setup_async()
        await trainer.train()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Top-level config for RL training."""

        model_spec: ModelSpec | None = None
        """Model specification shared by trainer and generator.
        Set programmatically via config_registry (not from CLI)."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Path to HF assets folder (model weights, tokenizer, config files)."""

        num_steps: int = 10
        """Number of RL training steps."""

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts (temp weights, logs, etc.)."""

        num_groups_per_rollout_batch: int = 5
        """GRPO groups collected per rollout batch; a train step may collect several batches
        until the token target is met. Rollouts per batch = `num_groups_per_rollout_batch * group_size`."""
        # TODO(continuous-batching): this knob exists because we collect to a token budget
        # in discrete sync batches; async/continuous batching streams may change this logic

        group_size: int = 8
        """Sibling rollouts sampled per dataset row (the GRPO group). The generator
        is always called with `n=1`; prompts are pre-expanded by `group_size`."""

        num_validation_samples: int = 20
        """Number of held-out prompts scored greedily (temp=0, n=1) per validation pass."""

        rollouter: Rollouter.Config
        """The rollouter: its datasets, envs, and rubric."""
        # TODO: support multiple rollouters for data mixing.

        renderer: RendererConfig
        """Message-to-token renderer config."""

        log_samples: bool = False
        """Log first completion per episode during training and validation."""

        rollout_recorder: RolloutSampleRecorder.Config = field(
            default_factory=RolloutSampleRecorder.Config
        )
        """JSONL recorder to save sampled rollouts to disk for further inspection and debugging."""

        compile: CompileConfig = field(default_factory=CompileConfig)
        """torch.compile config shared by trainer and generator."""

        batcher: Batcher.Config = field(default_factory=Batcher.Config)
        """Batcher config: local_batch_size, seq_len."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=GRPOLoss.Config())
        )
        """PolicyTrainer config. Controls optimizer, training, parallelism."""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

        router: GeneratorRouter.Config = field(default_factory=GeneratorRouter.Config)
        """Generator routing strategy configuration."""

        metrics: m.MetricsProcessor.Config = field(
            default_factory=m.MetricsProcessor.Config
        )

        def __post_init__(self):
            if self.generator.checkpoint.enable:
                raise ValueError(
                    "Generator checkpoint must be disabled in the RL loop "
                    "(weights are synced from the trainer via TorchStore). "
                    "Set generator.checkpoint.enable=False."
                )
            if self.trainer.debug.batch_invariant:
                if not self.trainer.debug.deterministic:
                    raise ValueError("batch_invariant requires deterministic=True")
                # TODO: Replace trainer dtype constraint to use mixed
                #  training enabled by FSDP.
                if self.trainer.training.dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 training dtype, "
                        f"got {self.trainer.training.dtype!r}"
                    )
                if self.generator.model_dtype != "bfloat16":
                    raise ValueError(
                        f"batch_invariant requires bfloat16 generator dtype, "
                        f"got {self.generator.model_dtype!r}"
                    )
                if self.trainer.parallelism.enable_sequence_parallel:
                    raise ValueError(
                        "batch_invariant mode doesn't support SP now. "
                        "SP uses reduce-scatter which only supports Ring in NCCL "
                        "and has not been validated for determinism."
                    )

    def __init__(self, config: Config):
        self.config = config
        self.trainer = None
        self.generator_router: GeneratorRouter | None = None
        self._proc_meshes = []
        self.metrics_processor: m.MetricsProcessor = config.metrics.build(
            log_dir=config.dump_folder,
            job_config=config.to_dict(),
        )
        self.renderer = config.renderer.build(tokenizer_path=config.hf_assets_path)

        # Renderer stop tokens are injected into the generator at spawn
        self._stop_token_ids = list(self.renderer.get_stop_token_ids())
        self._sampling = config.generator.sampling
        # TODO: pass our own tokenizer to the renderer and read pad/eos off it
        # once `renderers` supports bring-your-own-tokenizer
        # (https://github.com/PrimeIntellect-ai/renderers/pull/70).
        # Until then, reach into the renderer's tokenizer for the pad id (eos doubles as pad).
        self.batcher = Batcher(
            config.batcher, pad_id=self.renderer._tokenizer.eos_token_id
        )
        self._rollouter: Rollouter = config.rollouter.build()
        self.rollout_recorder = config.rollout_recorder.build(
            dump_dir=config.dump_folder
        )

    async def close(self):
        """Best-effort: tear down actors, close metric backends, then stop proc meshes."""
        logger.info("Closing: tearing down actors and process meshes.")

        if self.trainer is not None:
            try:
                await self.trainer.close.call()
            except Exception:
                logger.exception("trainer.close failed")

        if self.generator_router is not None:
            close_results = await self.generator_router.fanout(
                "close",
                return_exceptions=True,
            )
            for idx, result in enumerate(close_results):
                if isinstance(result, BaseException):
                    actor_name = (
                        "generator" if len(close_results) == 1 else f"generator[{idx}]"
                    )
                    logger.error(
                        "%s.close failed",
                        actor_name,
                        exc_info=(type(result), result, result.__traceback__),
                    )

        try:
            self.metrics_processor.close()
        except Exception:
            logger.exception("metrics_processor close failed")

        for i, mesh in enumerate(self._proc_meshes):
            try:
                await mesh.stop()
            except Exception:
                logger.exception("mesh.stop[%d] failed", i)
        self._proc_meshes = []

    def _get_rank_0_value(self, result):
        """Extract rank 0 result from a Monarch ValueMesh.

        Monarch actor endpoints return results from all ranks in the mesh.
        This method picks out rank 0's result. This should be used in cases
        where all ranks return the same result.
        """
        return result.get(0)

    def _shard_episodes(self, episodes: list[Episode]) -> list[list[Episode]]:
        """Round-robin partition episodes across DP ranks."""
        return [
            [episodes[i] for i in range(rank, len(episodes), self.trainer_dp_degree)]
            for rank in range(self.trainer_dp_degree)
        ]

    @sl.log_trace_span("setup_async")
    async def setup_async(
        self,
        *,
        trainer_mesh: ProcMesh,
        generator_meshes: list[ProcMesh],
    ):
        """Spawn Monarch actors on separate meshes and initialize weights.

        Kept separate from ``__init__`` because actor spawning, torch
        elastic env setup, TorchStore initialization, and the initial
        weight push/pull are all ``await``-based runtime side effects
        that cannot run in a synchronous constructor.

        The trainer and generator meshes are provisioned by the caller
        (see ``create_proc_mesh``) on disjoint GPUs; this method only
        spawns the actors on them and synchronizes initial weights from
        trainer to generator. Must be called before :meth:`train`.

        Args:
            trainer_mesh: ProcMesh the trainer actor is spawned on.
            generator_meshes: ProcMesh objects the generator actors are spawned on.
        """
        # Thread pool for TokenEnv's asyncio.to_thread renderer calls — one worker per
        # concurrent rollout, capped by CPUs.
        max_concurrent_rollouts = max(
            self.config.num_groups_per_rollout_batch * self.config.group_size,
            self.config.num_validation_samples,
        )
        max_workers = max(1, min(max_concurrent_rollouts, os.cpu_count() or 1))
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(max_workers=max_workers)
        )

        config = self.config
        if not generator_meshes:
            raise ValueError("setup_async requires at least one generator mesh")

        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        # TODO(observability): the mesh_spawn span wraps ~80 LoC of branching
        # provisioner logic. Pull a PerHostProvisioner.spawn_meshes(...) helper and
        # shrink this span to a single call.
        with sl.log_trace_span("mesh_spawn"):
            # Store proc meshes for cleanup
            self._proc_meshes = [trainer_mesh, *generator_meshes]

            await setup_torch_elastic_env_async(trainer_mesh)
            for mesh in generator_meshes:
                await setup_torch_elastic_env_async(mesh)

            # Spawn actors on their respective meshes
            self.trainer = trainer_mesh.spawn(
                "trainer",
                PolicyTrainer,
                config.trainer,
                model_spec=config.model_spec,
                hf_assets_path=config.hf_assets_path,
                generator_dtype=config.generator.model_dtype,
                compile_config=config.compile,
                output_dir=config.dump_folder,
            )

            generators = []
            for idx, mesh in enumerate(generator_meshes):
                actor_name = (
                    "generator" if len(generator_meshes) == 1 else f"generator_{idx}"
                )
                generator = mesh.spawn(
                    actor_name,
                    VLLMGenerator,
                    config.generator,
                    model_spec=config.model_spec,
                    model_path=config.hf_assets_path,
                    compile_config=config.compile,
                    max_num_seqs=max(
                        config.num_groups_per_rollout_batch * config.group_size,
                        config.num_validation_samples,
                    ),
                    output_dir=config.dump_folder,
                    stop_token_ids=self._stop_token_ids,
                )
                generators.append(generator)
            self.generator_router = GeneratorRouter(
                config.router,
                generators=generators,
            )

        # Initialize TorchStore for weight sync between trainer and generator.
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # LocalRankStrategy: routes each process to a storage volume based on
        #   LOCAL_RANK, so colocated processes share the same volume.
        # https://github.com/meta-pytorch/torchstore
        with sl.log_trace_span("torchstore_init"):
            await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # Initial weight sync from trainer to generator
        with sl.log_trace_span("trainer_push_model_state_dict"):
            await self.trainer.push_model_state_dict.call()
        with sl.log_trace_span("generator_pull_model_state_dict"):
            await self.generator_router.sync_weights(0)

    @sl.log_trace_span("_collect_rollouts")
    async def _collect_rollouts(
        self,
        *,
        is_validation: bool,
        num_groups: int,
        group_size: int,
        sampling: SamplingConfig,
        step: int,
        group_offset: int,
    ) -> tuple[list[RolloutGroup], list[m.Metric]]:
        """Sample examples, run rollouts, score groups, and emit metrics.

        Steps:
        1. Sample one example per group from the train / validation dataset
        2. Create N envs per example
        3. Reset every env to get each rollout's first prompt
        4. Split groups by init status
        5. Run one batched `generate` over valid groups (n=1; pre-expanded)
        6. For each group: step generated rollouts if needed, then score with `rollouter.score_group`

        Args:
            is_validation: Sample from the validation dataset (else train).
            num_groups: Number of prompt groups to collect this call.
            group_size: Sibling rollouts per group (prompts pre-expanded; generator runs n=1).
            sampling: SamplingConfig for the generate call.
            step: Training step, tagged into group_id / sample_id for metrics + debugging.
            group_offset: Starting group index so group_ids stay unique across rounds in a step.

        Returns:
            Scored rollout groups plus rollout/generator metrics. Per-group
            rollout/scoring failures are logged and dropped.

        TODO(continuous-batching): once available, run rollouts independently
        instead of batching one `generate` over all prompts at once.
        """

        generation_metrics_prefix = (
            "validation_generator" if is_validation else "generator"
        )
        rollout_metrics_prefix = "validation" if is_validation else "rollout"

        # 1. Get one sample per group from the train / validation dataset;
        # 2. create N envs per sample.
        group_states: list[_RolloutGroupState] = []
        for group_idx in range(num_groups):
            sample = (
                self._rollouter.get_validation_sample()
                if is_validation
                else self._rollouter.get_training_sample()
            )
            group_id = f"step={step}/group={group_offset + group_idx}"
            envs = self._rollouter.make_env_group(
                sample=sample, group_size=group_size, renderer=self.renderer
            )
            group_states.append(
                _RolloutGroupState(
                    group_id=group_id,
                    sample=sample,
                    envs=envs,
                )
            )

        generation_metrics: list[m.Metric] = []
        # 3. Reset every env to get its first prompt.
        env_init_outputs_per_group_state = await asyncio.gather(  # [G][group_size]
            *(
                asyncio.gather(*(env.init() for env in group_state.envs))
                for group_state in group_states
            )
        )
        for group_state, env_init_outputs in zip(
            group_states, env_init_outputs_per_group_state, strict=True
        ):
            group_state.env_init_outputs = env_init_outputs

        # 4. Skip invalid group states and collect the rest for one batched
        # generate call. Reset status is treated as a group-level invariant:
        # all siblings are valid, or the whole group state is skipped.
        valid_group_states: list[_RolloutGroupState] = []
        num_skipped_groups = 0
        for group_state in group_states:
            is_valid = all(
                not env_init_output.status.is_terminal()
                for env_init_output in group_state.env_init_outputs
            )
            if is_valid:
                valid_group_states.append(group_state)
            else:
                num_skipped_groups += 1
                # TODO: log skipped prompts so they remain debuggable.
                await asyncio.gather(
                    *(env.close() for env in group_state.envs),
                    return_exceptions=True,
                )

        # Prepare generate requests
        prompt_token_ids: list[list[int]] = []  # [num_valid_samples][prompt_tokens]
        request_ids: list[str] = []  # [num_valid_samples]
        for group_state in valid_group_states:
            for sample_idx, env_init_output in enumerate(group_state.env_init_outputs):
                prompt_token_ids.append(env_init_output.next_prompt_token_ids or [])
                # TODO(multi-turn): make request_id unique per turn (append a turn index). Today it is
                # per-sample, so multiple generate calls within one multi-turn rollout would reuse the id.
                request_ids.append(_sample_id(group_state.group_id, sample_idx))

        # 5. Run one batched generate over valid group states (n=1; pre-expanded).
        # TODO: pass the remaining budget (max_rollout_tokens - len(prompt)) to the
        # sampling_config, to limit generation length in one turn.
        completions: list[Completion] = []
        if valid_group_states:
            completions, generation_metrics = self._get_rank_0_value(
                await self.generator_router.route(
                    "generate",
                    prompt_token_ids,
                    request_ids=request_ids,
                    sampling_config=sampling,
                    metrics_prefix=generation_metrics_prefix,
                    ctx=RouteContext(est_cost=len(prompt_token_ids)),
                )
            )
            returned_ids = [completion.request_id for completion in completions]
            if returned_ids != request_ids:
                raise RuntimeError(
                    f"generator returned request_ids {returned_ids}, "
                    f"expected {request_ids}"
                )

        # 6. After the batch-level generate returns, finish each group state
        # independently: step generated rollouts, score the group, and append
        # the result.
        # TODO(continuous-batching): group completions by group_id parsed from request_id instead of
        # relying on returned order; under CB completions won't come back in request order.
        completion_offset = 0
        for group_state in valid_group_states:
            next_completion_offset = completion_offset + len(
                group_state.env_init_outputs
            )
            group_state.completions = completions[
                completion_offset:next_completion_offset
            ]
            completion_offset = next_completion_offset

        finished_rollout_groups: list[RolloutGroup | None] = await asyncio.gather(
            *(
                self._run_group_rollout(
                    group_state=group_state,
                )
                for group_state in valid_group_states
            )
        )

        # Compute Metrics
        rollout_groups = [
            rollout_group
            for rollout_group in finished_rollout_groups
            if rollout_group is not None
        ]
        num_failed_groups = (
            num_skipped_groups + len(finished_rollout_groups) - len(rollout_groups)
        )

        generation_metrics.append(
            m.Metric(
                f"{rollout_metrics_prefix}/group_failures",
                m.Sum(float(num_failed_groups)),
            )
        )
        rollout_metrics = prepare_rollout_metrics(
            rollout_metrics_prefix,
            [
                rollout
                for rollout_group in rollout_groups
                for rollout in rollout_group.rollouts
            ],
        )
        rollout_metrics += generation_metrics
        return rollout_groups, rollout_metrics

    @sl.log_trace_span("_run_group_rollout")
    async def _run_group_rollout(
        self,
        *,
        group_state: _RolloutGroupState,
    ) -> RolloutGroup | None:
        """Step generated rollouts, score the group, and return it.

        Args:
            group_state: One prompt group's envs, prompt steps, and rollout slots.

        Returns:
            Scored rollout group, or `None` if this group failed and should be dropped.
        """
        try:
            if group_state.completions is None:
                raise RuntimeError(f"group {group_state.group_id} has no completions")

            rollouts: list[Rollout] = await asyncio.gather(
                *(
                    self._run_single_rollout(
                        group_id=group_state.group_id,
                        sample_id=_sample_id(group_state.group_id, sample_idx),
                        env=env,
                        env_init_output=env_init_output,
                        completion=completion,
                    )
                    for sample_idx, (env, env_init_output, completion) in enumerate(
                        zip(
                            group_state.envs,
                            group_state.env_init_outputs,
                            group_state.completions,
                            strict=True,
                        )
                    )
                )
            )

            outputs = await self._rollouter.score_group(rollouts, group_state.sample)
            for rollout, output in zip(rollouts, outputs, strict=True):
                rollout.reward = output.reward
                rollout.reward_breakdown = output.reward_breakdown
            return RolloutGroup(group_id=group_state.group_id, rollouts=rollouts)
        except Exception:
            # TODO: add better logging so they are debuggable
            logger.exception(
                "group %s rollout/scoring failed; dropping", group_state.group_id
            )
            return None
        finally:
            await asyncio.gather(
                *(env.close() for env in group_state.envs),
                return_exceptions=True,
            )

    @sl.log_trace_span("_run_single_rollout")
    async def _run_single_rollout(
        self,
        *,
        group_id: str,
        sample_id: str,
        env: TokenEnv,
        env_init_output: TokenEnvOutput,
        completion: Completion,
    ) -> Rollout:
        """Step one env with its completion into a `Rollout`. On failure, return the
        turns collected so far with an `ERROR` status.

        Reward is left unset; the controller scores via `rollouter.score_group(...)`
        afterward and fills `reward` / `reward_breakdown`.

        Args:
            group_id: Prompt-group ID; siblings share it for advantage centering.
            sample_id: Unique rollout id.
            env: The env for this rollout.
            env_init_output: env output whose prompt produced this completion.
            completion: Generator completion for this env's prompt.

        Returns:
            One unscored Rollout.
        """
        rollout_turns: list[RolloutTurn] = []
        try:
            env_output = await env.step(completion)
            rollout_turns.append(
                RolloutTurn(
                    prompt_token_ids=env_init_output.next_prompt_token_ids or [],
                    prompt_messages=env_init_output.next_prompt_messages or [],
                    completion_token_ids=completion.token_ids,
                    completion_logprobs=completion.token_logprobs,
                    policy_version=completion.policy_version,
                    completion_message=env_output.completion_message,
                    env_messages=env_output.env_messages,
                    env_rewards=env_output.env_rewards,
                )
            )
            status = env_output.status
            # TODO(multi-turn): while not status.is_terminal(): generate → step → append turn.
            if not status.is_terminal():
                raise RuntimeError(
                    f"env {sample_id} returned a non-terminal turn; "
                    "the controller does not yet support multi-turn rollouts."
                )
        except Exception:
            logger.exception(
                "rollout %s failed; keeping %d turn(s) as ERROR",
                sample_id,
                len(rollout_turns),
            )
            status = RolloutStatus.ERROR
        return Rollout(
            group_id=group_id, sample_id=sample_id, status=status, turns=rollout_turns
        )

    @staticmethod
    @sl.log_trace_span("_build_episodes")
    def _build_episodes(
        rollout_groups: list[RolloutGroup],
    ) -> tuple[list[Episode], list[m.Metric]]:
        """Build train episodes and GRPO advantages from scored rollout groups.

        Centers each group's rewards by its mean, skips rollouts without
        training tokens, and emits reward/advantage metrics.

        Args:
            rollout_groups: Scored rollout groups from one collection round.

        Returns:
            Train episodes plus episode-level metrics.
        """
        # Mean-baseline advantage per group
        episodes: list[Episode] = []
        group_stds: list[float] = []
        for group in rollout_groups:
            # Drop the whole group if any sibling has no trainable tokens; we
            # need one turn with assistant tokens to build an episode.
            if any(
                not rollout.turns or not rollout.turns[0].completion_token_ids
                for rollout in group.rollouts
            ):
                logger.warning(
                    "group %s has an untrainable rollout; dropping the group",
                    group.group_id,
                )
                continue

            rewards = [rollout.reward for rollout in group.rollouts]
            group_mean = sum(rewards) / len(rewards)
            group_stds.append(statistics.pstdev(rewards))

            for rollout in group.rollouts:
                rollout.advantage = rollout.reward - group_mean
                episodes.append(rollout_to_episode(rollout))

        num_groups = len(rollout_groups)
        zero_std_frac = (
            sum(1 for s in group_stds if s == 0.0) / num_groups if num_groups else 0.0
        )
        episode_metrics: list[m.Metric] = [
            m.Metric(
                "reward",
                m.SummaryStats.from_list([ep.reward for ep in episodes]),
            ),
            m.Metric(
                "advantage",
                m.SummaryStats.from_list([ep.advantage for ep in episodes]),
            ),
            m.Metric("reward/group_std", m.Mean.from_list(group_stds)),
            m.Metric("reward/group_std", m.Max.from_list(group_stds)),
            m.Metric("reward/zero_std_frac", m.NoReduce(zero_std_frac)),
        ]

        # Per-rollout policy versions. We log max/min in case episodes come
        # from multiple rollout versions.
        policy_versions = [episode.policy_version for episode in episodes]
        if policy_versions:
            episode_metrics.extend(
                [
                    m.Metric(
                        "rollout/policy_version", m.Min.from_list(policy_versions)
                    ),
                    m.Metric(
                        "rollout/policy_version", m.Max.from_list(policy_versions)
                    ),
                ]
            )
        return episodes, episode_metrics

    # TODO: we currently determine num_validation_samples
    # but what if i want to run the entire dataset?
    @sl.log_trace_span("validate")
    async def validate(self, *, step: int) -> list[m.Metric]:
        """Run greedy validation on held-out prompts.

        Args:
            step: Training step this validation pass belongs to (0 for the
                pre-training pass); tagged into logged rollout samples.

        Returns:
            Validation rollout metrics, generation metrics, and validation
            timing.
        """
        # TODO: investigate using pass@k for validation.
        t_validate_start = time.perf_counter()
        num_samples = self.config.num_validation_samples
        greedy = replace(self._sampling, temperature=0.0, top_p=1.0)

        rollout_groups, validation_metrics = await self._collect_rollouts(
            is_validation=True,
            num_groups=num_samples,
            group_size=1,
            sampling=greedy,
            step=step,
            group_offset=0,
        )
        rollouts = [rollout for group in rollout_groups for rollout in group.rollouts]

        if self.config.log_samples:
            _log_samples(rollout_groups)
        self.rollout_recorder.record(
            step=step, is_validation=True, rollout_groups=rollout_groups
        )

        validation_metrics.append(
            m.Metric("validation/num_samples", m.NoReduce(float(len(rollouts))))
        )

        t_validate_s = time.perf_counter() - t_validate_start
        validation_metrics.append(m.Metric("timing/validate", m.NoReduce(t_validate_s)))
        return validation_metrics

    async def train(self):
        num_steps = self.config.num_steps
        num_groups = self.config.num_groups_per_rollout_batch
        logger.info(f"Pre-training validation; then {num_steps} steps of RL training")

        # collect validation metrics before training
        # so we can compare before/after
        pre_validation_metrics = await self.validate(step=0)
        self.metrics_processor.log(
            step=0,
            metrics=pre_validation_metrics,
            is_validation=True,
        )
        pre_validation_agg = m.MetricsProcessor._aggregate_metrics(
            pre_validation_metrics
        )

        sl.log_trace_instant("training_start")

        for step in range(1, num_steps + 1):
            sl.set_step(step)
            # Propagate the step counter to actors for structured logging.
            await self.trainer.sync_log_step.call(step)
            await self.generator_router.fanout("sync_log_step", step)

            t_step_start = time.perf_counter()

            # --- rollouts ---
            # Collect rollouts until total response tokens reach the
            # token budget. The Batcher then packs, truncates to
            # global_batch_size rows, and splits into microbatches.
            t_rollout_start = time.perf_counter()
            rollout_groups: list[RolloutGroup] = []
            rollout_metrics: list[m.Metric] = []
            collected_tokens = 0
            group_offset = 0
            # num_tokens_target (= global_batch_size * seq_len) is the stop
            # condition for collected tokens before a train step can proceed.
            # NOTE: this is a proxy — packing adds padding to fill fixed-length
            # rows, so actual token consumption may exceed collected_tokens.
            num_tokens_target = self.batcher.num_tokens_target(self.trainer_dp_degree)
            while collected_tokens < num_tokens_target:
                new_rollout_groups, new_metrics = await self._collect_rollouts(
                    is_validation=False,
                    num_groups=num_groups,
                    group_size=self.config.group_size,
                    sampling=self._sampling,
                    step=step,
                    group_offset=group_offset,
                )
                rollout_groups.extend(new_rollout_groups)
                rollout_metrics.extend(new_metrics)
                # Both prompt length and completion length are counted.
                collected_tokens += sum(
                    len(t.prompt_token_ids) + len(t.completion_token_ids) - 1
                    for group in new_rollout_groups
                    for r in group.rollouts
                    for t in r.turns
                )
                group_offset += num_groups

            episodes, episode_metrics = self._build_episodes(rollout_groups)
            t_rollout_s = time.perf_counter() - t_rollout_start

            if self.config.log_samples:
                _log_samples(rollout_groups)
            self.rollout_recorder.record(
                step=step, is_validation=False, rollout_groups=rollout_groups
            )

            # --- train ---
            t_train_start = time.perf_counter()
            with sl.log_trace_span("batcher_batch"):
                (
                    microbatches,
                    num_global_valid_tokens,
                    packing_metrics,
                ) = self.batcher.batch(episodes, dp_degree=self.trainer_dp_degree)

            # Aggregate metrics across gradient-accumulation microbatches.
            # "/mean" and "/frac" metrics are pre-normalized by
            # num_global_valid_tokens, so summing reconstructs the global
            # value.  "/max" metrics take the max across microbatches.
            fwd_bwd_metrics: dict[str, float] = {}
            for microbatch in microbatches:
                with sl.log_trace_span("trainer_forward_backward_call"):
                    mb_metrics = self._get_rank_0_value(
                        await self.trainer.forward_backward.call(
                            microbatch, num_global_valid_tokens
                        )
                    )
                    for k, v in mb_metrics.items():
                        if k not in fwd_bwd_metrics:
                            fwd_bwd_metrics[k] = v
                        elif k.endswith("/max"):
                            fwd_bwd_metrics[k] = max(fwd_bwd_metrics[k], v)
                        elif k.endswith(("/mean", "/frac")):
                            fwd_bwd_metrics[k] += v
            with sl.log_trace_span("trainer_optim_step_call"):
                optim_output = self._get_rank_0_value(
                    await self.trainer.optim_step.call()
                )
            trainer_policy_version = optim_output.policy_version
            optimizer_metrics = optim_output.metrics
            t_train_s = time.perf_counter() - t_train_start

            # --- weight sync ---
            # TODO: we should have `push_model_state_dict` return `trainer_policy_version`
            # instead of having `trainer.optim_step` return it
            t_push_start = time.perf_counter()
            with sl.log_trace_span("trainer_push_model_state_dict"):
                await self.trainer.push_model_state_dict.call()
            t_weight_sync_push_s = time.perf_counter() - t_push_start
            with sl.log_trace_span("generator_pull_model_state_dict"):
                await self.generator_router.sync_weights(trainer_policy_version)
            t_weight_sync_total_s = time.perf_counter() - t_push_start
            t_step_s = time.perf_counter() - t_step_start
            # --- divergence check before any logging ---
            if not math.isfinite(fwd_bwd_metrics["loss/mean"]):
                logger.error("Loss is NaN/Inf; training diverged")
                break

            # --- Prepare metrics ---
            total_tokens = sum(
                len(ep.prompt_token_ids) + len(ep.completion_token_ids)
                for ep in episodes
            )

            step_metrics: list[m.Metric] = []

            step_metrics += rollout_metrics
            step_metrics += episode_metrics

            # Actor metrics are already globally reduced and aggregated
            # across microbatches; NoReduce passes them through.
            step_metrics += [
                m.Metric(k, m.NoReduce(v)) for k, v in fwd_bwd_metrics.items()
            ]
            step_metrics += [
                m.Metric(k, m.NoReduce(v)) for k, v in optimizer_metrics.items()
            ]
            step_metrics += packing_metrics

            # timing metrics
            for key, value in [
                ("timing/step", t_step_s),
                ("timing/rollout", t_rollout_s),
                ("timing/train", t_train_s),
                ("timing/weight_sync/push", t_weight_sync_push_s),
                ("timing/weight_sync/total", t_weight_sync_total_s),
            ]:
                step_metrics.append(m.Metric(key, m.NoReduce(value)))

            step_metrics.append(
                m.Metric("perf/tokens_per_second", m.NoReduce(total_tokens / t_step_s))
            )

            self.metrics_processor.log(
                step=step, metrics=step_metrics, is_validation=False
            )

        post_validation_metrics = await self.validate(step=num_steps)
        self.metrics_processor.log(
            step=num_steps,
            metrics=post_validation_metrics,
            is_validation=True,
        )
        post_validation_agg = m.MetricsProcessor._aggregate_metrics(
            post_validation_metrics
        )

        # Side-by-side pre/post summary so the before/after improvement is
        # visible without scrolling back through the train loop.
        reward_keys = sorted(
            k
            for k in set(pre_validation_agg) | set(post_validation_agg)
            if "reward" in k
        )
        logger.info("=" * 60)
        logger.info("Validation summary (pre / post):")
        for key in reward_keys:
            pre = pre_validation_agg.get(key, float("nan"))
            post = post_validation_agg.get(key, float("nan"))
            logger.info(f"  {key}:  {pre:+.3f}  /  {post:+.3f}")
        logger.info("=" * 60)
