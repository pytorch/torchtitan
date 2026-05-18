# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
RL training loop using Monarch Actors.

This demonstrates:
1. Distributed actor architecture with VLLMGenerator (vLLM) and PolicyTrainer (TorchTitan)
   running on separate GPU meshes
2. Weight synchronization across meshes: trainer gathers full (unsharded) weights,
   generator reshards to match its own parallelism layout via distribute_tensor
3. Envs driven rollouts; reward and advantage computation live inline
   in the controller.

Command to run:
python3 torchtitan/experiments/rl/grpo.py \
    --module rl --config rl_grpo_qwen3_0_6b \
    --hf_assets_path=<path_to_model_checkpoint>
"""

import asyncio
import logging
import math
import os
import time
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

# must run before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torchstore as ts
from monarch.actor import this_host
from monarch.spmd import setup_torch_elastic_env_async

from torchtitan.components.dataloading.utils import pack
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import (
    CompileConfig,
    ConfigManager,
    Configurable,
    ParallelismConfig,
)
from torchtitan.experiments.rl.actors.generator import SamplingConfig, VLLMGenerator
from torchtitan.experiments.rl.actors.trainer import PolicyTrainer
from torchtitan.experiments.rl.observability import metrics as m
from torchtitan.experiments.rl.types import (
    Completion,
    Episode,
    Step,
    TrainBatch,
    Trajectory,
)
from torchtitan.protocols.model_spec import ModelSpec

logger = logging.getLogger(__name__)


class GRPOLoss(Configurable):
    """Clipped GRPO surrogate loss operating on [B, L] tensors."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        clip_eps: float = 0.2
        """PPO clipping epsilon for the probability ratio."""

    def __init__(self, config: Config):
        self.clip_eps = config.clip_eps

    def __call__(
        self,
        policy_logprobs: torch.Tensor,
        response_mask: torch.Tensor,
        advantages: torch.Tensor,
        global_valid_tokens: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Per-token GRPO clipped surrogate, summed locally and normalized by the
        global response-token count.

        Each DP rank contributes ``local_sum / global_valid_tokens``; the per-rank
        losses sum to the correct global mean, matching the SFT trainer's pattern
        (``torchtitan/trainer.py`` ~line 720). Using the local count here would
        bias gradient accumulation across uneven ranks.
        """
        # Per-token log ratio (masked to response tokens only)
        ratio = torch.exp(policy_logprobs * response_mask)

        unclipped_loss = ratio * advantages
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
        clipped_loss = clipped_ratio * advantages

        denom = max(global_valid_tokens, 1)
        pg_loss = (
            -(torch.min(unclipped_loss, clipped_loss) * response_mask).sum() / denom
        )

        # Local diagnostics — use local count to avoid divide-by-zero on
        # all-pad ranks and to keep these per-rank-meaningful.
        local_count = max(int(response_mask.sum().item()), 1)
        metrics = {
            "pg_loss": pg_loss.item(),
            "ratio_mean": (ratio * response_mask).sum().item() / local_count,
            "ratio_clipped_frac": (
                (torch.abs(ratio - clipped_ratio) > 1e-6) & response_mask.bool()
            )
            .float()
            .sum()
            .item()
            / local_count,
        }
        return pg_loss, metrics


class Provisioner:
    """Allocates non-overlapping GPU ranges for Monarch proc meshes.

    In non-colocated mode, the trainer and generator run on separate GPU
    meshes (e.g. GPUs 0-3 for training, GPUs 4-7 for generation). Each
    call to ``allocate(n)`` reserves the next *n* GPUs and returns a
    bootstrap callable that sets ``CUDA_VISIBLE_DEVICES`` before CUDA
    initializes in the spawned process, ensuring each mesh only sees its
    own devices.
    """

    def __init__(self, total_gpus: int = 8):
        self.total_gpus = total_gpus
        self.next_gpu = 0

    @property
    def available(self) -> int:
        return self.total_gpus - self.next_gpu

    def allocate(self, num_gpus: int) -> Callable[[], None]:
        if num_gpus > self.available:
            raise RuntimeError(
                f"Requested {num_gpus} GPUs but only {self.available} "
                f"available (total={self.total_gpus}, allocated={self.next_gpu})"
            )
        gpu_ids = list(range(self.next_gpu, self.next_gpu + num_gpus))
        self.next_gpu += num_gpus

        def _bootstrap():
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
            # TODO: Remove once Monarch/PyTorch fixes concurrent import during unpickling.
            import torch  # noqa: F401

        return _bootstrap


def _mean_rewards(steps: list[Step]) -> dict[str, float]:
    """Per-component mean reward across a list of Steps."""
    return {k: sum(s.rewards[k] for s in steps) / len(steps) for k in steps[0].rewards}


def _format_rewards(components: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:+.3f}" for k, v in components.items())


def _format_validation(result: dict) -> str:
    return (
        f"mean_reward={result['mean_reward']:+.3f} "
        f"({_format_rewards(result['components'])})"
    )


def _log_samples(items: list[Episode] | list[Completion]) -> None:
    """Log the first sample per prompt for debugging."""
    seen_prompts: set[int] = set()
    for item in items:
        if item.prompt_idx in seen_prompts:
            continue
        seen_prompts.add(item.prompt_idx)
        reward_str = f" reward={item.reward:+.1f}" if hasattr(item, "reward") else ""
        logger.info(f"  [prompt {item.prompt_idx}]{reward_str}")
        logger.info(f"       A: {item.text[:300].replace(chr(10), ' ').strip()}")


def _prepare_reward_metrics(
    prefix: str,
    trajectories: list[Trajectory],
) -> list[m.Metric]:
    """One ``Mean`` metric per observed reward component across trajectories.

    Example::

        trajectories = [
            Trajectory(
                sample_idx=0,
                prompt_token_ids=p0,
                transitions=[(c0, Step(rewards={"correctness": 1.0, "format": 0.5}, done=True))],
            ),
            Trajectory(
                sample_idx=1,
                prompt_token_ids=p1,
                transitions=[(c1, Step(rewards={"correctness": 0.0}, done=True))],
            ),
        ]
        _prepare_reward_metrics("reward/component", trajectories)
        # -> [
        #      Metric("reward/component/correctness", Mean(sum=1.0, count=2)),  # 0.5
        #      Metric("reward/component/format",      Mean(sum=0.5, count=1)),  # 0.5 - "format" only in trajectory 0
        #    ]
    """
    values_by_name: dict[str, list[float]] = defaultdict(list)
    for trajectory in trajectories:
        for _completion, step in trajectory.transitions:
            for name, value in step.rewards.items():
                values_by_name[name].append(float(value))
    return [
        m.Metric(f"{prefix}/{name}", m.Mean.from_list(values))
        for name, values in sorted(values_by_name.items())
    ]


class Batcher(Configurable):
    """Packs episodes into a global batch split across DP ranks and grad accum steps.

    The number of gradient
    accumulation steps is derived as
    ``global_batch_size // (local_batch_size * dp_degree)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        local_batch_size: int = 1
        """Per-DP-rank batch size (packed rows per forward pass)."""

        global_batch_size: int = -1
        """Total packed rows per optimizer step. ``-1`` (default) auto-sizes
        to fit all episodes. When set explicitly, must be a multiple of
        ``local_batch_size * dp_degree``; excess rows are truncated."""

        seq_length: int = 2048
        """Tokens per packed row. Must not exceed the model's intrinsic
        ``rope.max_seq_len`` — validated in ``ModelArgs.update_from_config``."""

        input_ids_pad_value: int = 0
        """Token id used to pad packed sequences. Any in-vocab id is safe; these
        positions have ``response_mask == 0`` and contribute 0 to the loss."""

    def __init__(self, config: Config):
        self.local_batch_size = config.local_batch_size
        self.global_batch_size = config.global_batch_size
        self.seq_length = config.seq_length
        self.input_ids_pad_value = config.input_ids_pad_value

    def batch(
        self,
        episodes: list[Episode],
        *,
        dp_degree: int,
    ) -> tuple[list[list[TrainBatch]], int]:
        """Pack episodes into a structured global batch.

        Returns:
            per_step_batches: shape ``[gradient_accumulation_steps][dp_degree]``,
                each entry is a ``TrainBatch`` with ``local_batch_size`` rows.
            global_valid_tokens: total response tokens across the global batch
                (excludes padding rows). Use this to normalize the loss so that
                gradient accumulation matches a single large-batch step.
        """
        chunk = self.local_batch_size * dp_degree
        packed_rows = list(self._pack_episodes(episodes))

        if self.global_batch_size < 0:
            # Auto-size: round packed-row count up to next multiple of chunk,
            # preserving the original "use all episodes" behavior.
            global_batch_size = max(
                ((len(packed_rows) + chunk - 1) // chunk) * chunk, chunk
            )
        else:
            global_batch_size = self.global_batch_size
            assert global_batch_size > 0
            assert global_batch_size % chunk == 0, (
                f"global_batch_size ({global_batch_size}) must be a multiple of "
                f"local_batch_size ({self.local_batch_size}) * dp_degree "
                f"({dp_degree})"
            )
            if len(packed_rows) > global_batch_size:
                logger.warning(
                    f"Episodes packed into {len(packed_rows)} rows, exceeding "
                    f"global_batch_size {global_batch_size}; truncating. "
                    f"Consider increasing global_batch_size or reducing "
                    f"num_prompts_per_step."
                )
                packed_rows = packed_rows[:global_batch_size]
        gradient_accumulation_steps = global_batch_size // chunk

        if len(packed_rows) < global_batch_size:
            pad_count = global_batch_size - len(packed_rows)
            packed_rows.extend(self._pad_row() for _ in range(pad_count))

        global_valid_tokens = sum(
            int(row["response_mask"].sum().item()) for row in packed_rows
        )

        # Split into [gradient_accumulation_steps][dp_degree] TrainBatches,
        # each holding local_batch_size rows.
        per_step_batches: list[list[TrainBatch]] = []
        for step in range(gradient_accumulation_steps):
            step_batches: list[TrainBatch] = []
            for rank in range(dp_degree):
                start = (step * dp_degree + rank) * self.local_batch_size
                end = start + self.local_batch_size
                step_batches.append(self.collate(packed_rows[start:end]))
            per_step_batches.append(step_batches)

        return per_step_batches, global_valid_tokens

    def _pack_episodes(self, episodes: list[Episode]) -> Iterator[dict]:
        """Pack episodes into [1, seq_length] rows via shared pack()."""

        def _episode_samples() -> Iterator[dict]:
            for ep in episodes:
                prompt_len = len(ep.prompt_token_ids)
                response_len = len(ep.token_ids)
                yield {
                    "input_ids": ep.prompt_token_ids + ep.token_ids,
                    "ref_logprobs": [0.0] * prompt_len + ep.token_logprobs,
                    "response_mask": [0.0] * prompt_len + [1.0] * response_len,
                    "advantages": [0.0] * prompt_len + [ep.advantage] * response_len,
                }

        token_keys = ["input_ids", "ref_logprobs", "response_mask", "advantages"]
        yield from pack(
            _episode_samples(),
            token_keys=token_keys,
            max_seq_length=self.seq_length,
            pad_values={
                "input_ids": self.input_ids_pad_value,
                "ref_logprobs": 0.0,
                "response_mask": 0.0,
                "advantages": 0.0,
            },
        )

    def _pad_row(self) -> dict:
        """Construct a fully-padded row. response_mask is all zeros so this
        contributes 0 to ``global_valid_tokens`` and 0 to the loss."""
        L = self.seq_length
        return {
            "input_ids": torch.full((1, L), self.input_ids_pad_value, dtype=torch.long),
            "ref_logprobs": torch.zeros((1, L), dtype=torch.float32),
            "response_mask": torch.zeros((1, L), dtype=torch.float32),
            "advantages": torch.zeros((1, L), dtype=torch.float32),
            "positions": torch.arange(L, dtype=torch.long).unsqueeze(0),
            "seq_lens": [],
        }

    @staticmethod
    def collate(rows: list[dict]) -> TrainBatch:
        """Stack packed rows into a single [B, L] TrainBatch."""
        return TrainBatch(
            token_ids=torch.cat([r["input_ids"] for r in rows]),
            positions=torch.cat([r["positions"] for r in rows]),
            ref_logprobs=torch.cat([r["ref_logprobs"] for r in rows]),
            response_mask=torch.cat([r["response_mask"] for r in rows]),
            advantages=torch.cat([r["advantages"] for r in rows]),
        )


class RLTrainer(Configurable):
    """Top-level RL training orchestrator."""

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

        num_prompts_per_step: int = 5
        """Number of distinct prompts (= GRPO groups) drawn per training step.

        The total episodes per step is ``num_prompts_per_step * group_size``,
        where ``group_size`` is ``generator.sampling.n`` (completions per prompt).
        """

        num_validation_samples: int = 20
        """Number of held-out prompts scored greedily (temp=0, n=1) per validation pass."""

        env: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Env config for training rollouts."""

        validation_env: Configurable.Config = field(default=None)  # type: ignore[assignment]
        """Env config for validation rollouts."""

        log_samples: bool = False
        """Log first completion per episode during training and validation."""

        compile: CompileConfig = field(default_factory=CompileConfig)
        """torch.compile config shared by trainer and generator."""

        batcher: Batcher.Config = field(default_factory=Batcher.Config)

        metrics: m.MetricsProcessor.Config = field(
            default_factory=m.MetricsProcessor.Config
        )
        """Batcher config: local_batch_size, global_batch_size, seq_length."""

        trainer: PolicyTrainer.Config = field(
            default_factory=lambda: PolicyTrainer.Config(loss=GRPOLoss.Config())
        )
        """PolicyTrainer config. Controls optimizer, training, parallelism."""

        generator: VLLMGenerator.Config = field(default_factory=VLLMGenerator.Config)
        """VLLMGenerator actor configuration (vLLM engine, sampling)."""

        def __post_init__(self):
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
        self.generator = None
        self._proc_meshes = []
        self.metrics_processor: m.MetricsProcessor = config.metrics.build(
            log_dir=config.dump_folder,
            job_config=config.to_dict(),
        )
        # TODO: Replace this single-turn tokenizer with renderer
        self.tokenizer = HuggingFaceTokenizer(tokenizer_path=config.hf_assets_path)
        self._batcher = config.batcher.build()

    async def close(self):
        """Best-effort: tear down actors, then stop proc meshes."""
        logger.info("Closing: tearing down actors and process meshes.")
        for actor_name, actor in (
            ("trainer", self.trainer),
            ("generator", self.generator),
        ):
            if actor is None:
                continue
            try:
                await actor.close.call()
            except Exception:
                logger.exception("%s.close failed", actor_name)

        for i, mesh in enumerate(self._proc_meshes):
            try:
                await mesh.stop()
            except Exception:
                logger.exception("mesh.stop[%d] failed", i)
        self._proc_meshes = []

    def _get_rank_0_value(self, result, has_gpus: bool = True):
        """Extract rank 0 result, handling both single and multi-node meshes.

        Monarch actor endpoints return results from all ranks in the mesh.
        This picks out rank 0's result by indexing into the host and GPU
        dimensions as needed (multi-node meshes have an extra host dimension).
        This should be used in cases where all ranks return the same result.
        """
        kwargs = {}
        if self._multi_node:
            kwargs["hosts"] = 0
        if has_gpus:
            kwargs["gpus"] = 0
        return result.item(**kwargs)

    @staticmethod
    def _compute_world_size(p: ParallelismConfig) -> int:
        """Compute world size from all parallel dimensions."""
        dp_shard = max(p.data_parallel_shard_degree, 1)
        return (
            p.data_parallel_replicate_degree
            * dp_shard
            * p.tensor_parallel_degree
            * p.pipeline_parallel_degree
            * p.context_parallel_degree
        )

    async def setup(
        self,
        *,
        host_mesh=None,
        trainer_nodes: int | None = None,
        generator_nodes: int | None = None,
        gpus_per_node: int | None = None,
    ):
        """Spawn Monarch actors on separate meshes and initialize weights.

        Creates separate GPU meshes for trainer and generator and
        synchronizes initial weights from trainer to generator. Must be
        called before :meth:`train`.

        Args:
            host_mesh: Optional multi-node HostMesh. When provided,
                whole nodes are dedicated to trainer vs generator
                roles instead of partitioning GPUs on a single host.
            trainer_nodes: Number of nodes for the trainer (required when
                host_mesh is provided).
            generator_nodes: Number of nodes for the generator (required when
                host_mesh is provided).
            gpus_per_node: GPUs per node, assumed to be the same across all
                nodes (no heterogeneous node configurations). Required when
                host_mesh is provided.
        """
        config = self.config

        self.trainer_world_size = self._compute_world_size(config.trainer.parallelism)
        self.generator_world_size = self._compute_world_size(
            config.generator.parallelism
        )
        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        total_gpus = self.trainer_world_size + self.generator_world_size
        logger.info(
            f"{self.generator_world_size} generator GPUs + "
            f"{self.trainer_world_size} trainer GPUs = {total_gpus} total"
        )

        self._multi_node = host_mesh is not None

        if host_mesh is not None:
            # Multi-node mode: dedicate whole nodes to trainer vs generator
            if (
                trainer_nodes is None
                or generator_nodes is None
                or gpus_per_node is None
            ):
                raise ValueError(
                    "trainer_nodes, generator_nodes, and gpus_per_node are "
                    "required when host_mesh is provided"
                )
            # Validate that world sizes are evenly divisible by node counts
            assert self.trainer_world_size % trainer_nodes == 0, (
                f"trainer_world_size ({self.trainer_world_size}) must be "
                f"evenly divisible by trainer_nodes ({trainer_nodes})"
            )
            assert self.generator_world_size % generator_nodes == 0, (
                f"generator_world_size ({self.generator_world_size}) must be "
                f"evenly divisible by generator_nodes ({generator_nodes})"
            )

            # Compute GPUs per node for each role based on the config's
            # world size and number of nodes allocated to that role
            trainer_gpus_per_node = self.trainer_world_size // trainer_nodes
            generator_gpus_per_node = self.generator_world_size // generator_nodes

            trainer_host_mesh = host_mesh.slice(hosts=slice(0, trainer_nodes))
            generator_host_mesh = host_mesh.slice(
                hosts=slice(trainer_nodes, trainer_nodes + generator_nodes)
            )

            # Use Provisioner to set CUDA_VISIBLE_DEVICES so each role
            # only sees its own GPUs and doesn't conflict with other
            # processes on the node
            trainer_provisioner = Provisioner(total_gpus=gpus_per_node)
            generator_provisioner = Provisioner(total_gpus=gpus_per_node)

            trainer_mesh = trainer_host_mesh.spawn_procs(
                per_host={"gpus": trainer_gpus_per_node},
                bootstrap=trainer_provisioner.allocate(trainer_gpus_per_node),
            )
            generator_mesh = generator_host_mesh.spawn_procs(
                per_host={"gpus": generator_gpus_per_node},
                bootstrap=generator_provisioner.allocate(generator_gpus_per_node),
            )
        else:
            # Single-node mode: partition GPUs on this_host() via
            # CUDA_VISIBLE_DEVICES
            provisioner = Provisioner(total_gpus=total_gpus)
            trainer_mesh = this_host().spawn_procs(
                per_host={"gpus": self.trainer_world_size},
                bootstrap=provisioner.allocate(self.trainer_world_size),
            )
            generator_mesh = this_host().spawn_procs(
                per_host={"gpus": self.generator_world_size},
                bootstrap=provisioner.allocate(self.generator_world_size),
            )

        # Store proc meshes for cleanup
        self._proc_meshes = [trainer_mesh, generator_mesh]

        await setup_torch_elastic_env_async(trainer_mesh)
        await setup_torch_elastic_env_async(generator_mesh)

        # Spawn actors on their respective meshes
        self.trainer = trainer_mesh.spawn(
            "trainer",
            PolicyTrainer,
            config.trainer,
            model_spec=config.model_spec,
            hf_assets_path=config.hf_assets_path,
            generator_dtype=config.generator.model_dtype,
            compile_config=config.compile,
        )

        self.generator = generator_mesh.spawn(
            "generator",
            VLLMGenerator,
            config.generator,
            model_spec=config.model_spec,
            model_path=config.hf_assets_path,
            compile_config=config.compile,
            max_num_seqs=max(
                config.num_prompts_per_step * config.generator.sampling.n,
                config.num_validation_samples,
            ),
        )

        # Initialize TorchStore for weight sync between trainer and generator.
        # StorageVolumes are spawned on the trainer mesh so they are colocated
        # with the weight source for faster data access in the non-RDMA path.
        # LocalRankStrategy: routes each process to a storage volume based on
        #   LOCAL_RANK, so colocated processes share the same volume.
        # https://github.com/meta-pytorch/torchstore
        await ts.initialize(mesh=trainer_mesh, strategy=ts.LocalRankStrategy())

        # push weights from trainer
        self.trainer.push_model_state_dict.call().get()
        # pull weights for policy version 0 (initial weights)
        self.generator.pull_model_state_dict.call(0).get()

    def _collect_rollouts(self, num_groups: int, step: int) -> list[Trajectory]:
        """Collect group rollouts: one single-use env per group, scored and returned."""
        envs = [
            self.config.env.build(step=step, group_idx=i) for i in range(num_groups)
        ]
        # TODO: Add a check max_tokens = min(max_tokens, context_window - model_input.length)
        # and pass max_tokens to the generator call or skip the call if max_tokens<=0.
        # Do the same for validation.
        tokenized_prompts = [
            self.tokenizer.encode(env.prompt, add_bos=True, add_eos=False)
            for env in envs
        ]
        completions, generation_metrics = self._get_rank_0_value(
            self.generator.generate.call(tokenized_prompts).get()
        )
        trajectories: list[Trajectory] = []
        for c in completions:
            step_result = envs[c.prompt_idx].step(c.text)
            trajectories.append(
                Trajectory(
                    sample_idx=c.prompt_idx,
                    prompt_token_ids=tokenized_prompts[c.prompt_idx],
                    transitions=[(c, step_result)],
                )
            )
        return trajectories

    @staticmethod
    def _build_episodes(trajectories: list[Trajectory]) -> list[Episode]:
        """Group trajectories by sample, apply mean-baseline advantage, flatten to Episodes."""
        groups: dict[int, list[Trajectory]] = {}
        for t in trajectories:
            groups.setdefault(t.sample_idx, []).append(t)

        episodes: list[Episode] = []
        for sample_idx, group in groups.items():
            rewards = [t.total_reward for t in group]
            group_mean = sum(rewards) / len(rewards)
            for t in group:
                # Single-turn: exactly one (completion, step) per trajectory.
                c, _ = t.transitions[0]
                episodes.append(
                    Episode(
                        policy_version=c.policy_version,
                        prompt_idx=sample_idx,
                        prompt_token_ids=t.prompt_token_ids,
                        text=c.text,
                        token_ids=c.token_ids,
                        token_logprobs=c.token_logprobs,
                        reward=t.total_reward,
                        advantage=t.total_reward - group_mean,
                    )
                )
        return episodes

    async def validate(self) -> dict:
        """Run validation on held-out prompts using greedy sampling.
        TODO: investigate using pass@k."""
        num_samples = self.config.num_validation_samples
        envs = [
            self.config.validation_env.build(step=0, group_idx=i)
            for i in range(num_samples)
        ]
        greedy = SamplingConfig(
            n=1,
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.config.generator.sampling.max_tokens,
        )

        tokenized_prompts: list[list[int]] = [
            self.tokenizer.encode(env.prompt, add_bos=True, add_eos=False)
            for env in envs
        ]
        completions, generation_metrics = self._get_rank_0_value(
            self.generator.generate.call(
                tokenized_prompts,
                sampling_config=greedy,
                metrics_prefix="validation_generator",
            ).get()
        )

        steps = [env.step(completions[i].text) for i, env in enumerate(envs)]

        if self.config.log_samples:
            _log_samples(completions)

        components = _mean_rewards(steps)
        return {
            "mean_reward": sum(components.values()),
            "components": components,
            "total": num_samples,
        }

    async def train(self):
        num_steps = self.config.num_steps
        num_groups = self.config.num_prompts_per_step
        logger.info(f"Pre-training validation; then {num_steps} steps of RL training")
        pre_validation = await self.validate()
        logger.info(f"Pre:  {_format_validation(pre_validation)}")

        for step in range(num_steps):
            # Cancellation point for Ctrl-C (KeyboardInterrupt) handling.
            # This yields to the event loop to check for cancellation, which
            # doesn't happen with `.get` calls.
            # TODO: investigate replacing `.get()` with `await
            await asyncio.sleep(0)

            step_start = time.perf_counter()

            # --- Collect data and create episodes --- #
            trajectories = self._collect_rollouts(num_groups, step=step)
            episodes = self._build_episodes(trajectories)

            if self.config.log_samples:
                _log_samples(episodes)

            # --- Pack and train --- #
            # Batcher produces a fixed global_batch_size of packed rows,
            # padded as needed, and pre-split into [grad_accum_steps][dp_degree]
            # local TrainBatches. global_valid_tokens excludes padding rows.
            per_step_batches, global_valid_tokens = self._batcher.batch(
                episodes, dp_degree=self.trainer_dp_degree
            )

            all_fwd_bwd_metrics = []
            for step_batches in per_step_batches:
                fwd_bwd_metrics = self._get_rank_0_value(
                    self.trainer.forward_backward.call(
                        step_batches, global_valid_tokens
                    ).get()
                )
                all_fwd_bwd_metrics.append(fwd_bwd_metrics)
            optim_metrics = self._get_rank_0_value(self.trainer.optim_step.call().get())
            metrics = {
                **all_fwd_bwd_metrics[-1],
                **optim_metrics,
                "global_valid_tokens": global_valid_tokens,
            }

            # --- Weight sync --- #
            t0 = time.perf_counter()
            self.trainer.push_model_state_dict.call().get()
            t_push = time.perf_counter() - t0
            self.generator.pull_model_state_dict.call(metrics["policy_version"]).get()
            t_sync = time.perf_counter() - t0
            logger.info(f"Weight sync: push={t_push:.3f}s, total={t_sync:.3f}s")

            # --- Logging --- #
            steps = [t.transitions[0][1] for t in trajectories]
            components = _mean_rewards(steps)
            avg_tokens = sum(len(ep.token_ids) for ep in episodes) / len(episodes)
            logger.info(
                f"Step {step:2d} | Loss: {metrics['loss']:+.4f} | "
                f"Reward: {sum(components.values()):+.3f} ({_format_rewards(components)}) | "
                f"Avg tokens: {avg_tokens:>3.0f} | "
                f"Logprob diff: mean={metrics['logprob_diff_mean']:.4e}, "
                f"max={metrics['logprob_diff_max']:.4e} | "
                f"Time: {time.perf_counter() - step_start:.1f}s"
            )

            if not math.isfinite(metrics["loss"]):
                logger.error("Loss is NaN/Inf; training diverged")
                break

        logger.info("Post-training validation")
        post_validation = await self.validate()
        logger.info(
            f"Summary:\n  Pre:  {_format_validation(pre_validation)}\n"
            f"  Post: {_format_validation(post_validation)}"
        )


async def main():
    config = ConfigManager().parse_args()
    rl_trainer = config.build()
    try:
        await rl_trainer.setup()
        await rl_trainer.train()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Interrupted; attempting graceful shutdown...")
    finally:
        await rl_trainer.close()


if __name__ == "__main__":
    asyncio.run(main())
