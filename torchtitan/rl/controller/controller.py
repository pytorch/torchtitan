# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TLDR:
_data_input_loop -> _rollout_loop -> _batcher_loop -> training_batch_queue -> _trainer_loop
           |               ^                    ^
           v               |                    |
           +-------RolloutGroupWorkBuffer-------+

Detailed diagram:

_data_input_loop                                      _rollout_loop[N] (group workers)
+--------------------------------------------------+  +--------------------------------------------------+
| group_buffer.wait_for_slot()                     |  | work = group_buffer.claim_next()                  |
| index, sample = rollouter.get_training_sample()  |  | group = rollouter.run_group_rollouts(work.sample) |
| work = RolloutGroupWork(group_id, sample)        |  | group_buffer.finalize_work(group)                 |
| group_buffer.add_work(work)                      |  +-----------------------+--------------------------+
+-----------------------+--------------------------+                          ^ |
                        |                                                     | |
                        | adds work entry                                     | | updates same entry
                        v                                                     | v
RolloutGroupWorkBuffer
+---------------------------------------------------------------------------------------------------------------------+
| active slots = (target_offpolicy_steps + 1) * num_prompts_per_train_step, or the adaptive buffer's demand           |
|                                                                                                                     |
| caller            group_buffer call                                            state / active slot                  |
| _data_input_loop  add_work(RolloutGroupWork)                                   WAITING; slot acquired               |
| _rollout_loop[N]  claim_next()                                                 WAITING -> INFLIGHT                  |
| _rollout_loop[N]  finalize_work(RolloutGroup)                                  INFLIGHT -> FINALIZED                |
| _batcher_loop     RolloutGroup = take_finalized()                              FINALIZED -> taken (slot still held) |
|                                                                                adaptive: too-old INFLIGHT cancelled |
| _rollout_loop[N]  release_cancelled(group_id)                                  cancelled group's task ended; freed  |
| _batcher_loop     release_active_groups(1, "untrainable_group")                slot released                        |
| _trainer_loop     record_step_start(trainer_policy_version)                    adaptive: demand updated             |
| _trainer_loop     release_active_groups(num_prompts_per_train_step, "trained")  slots released after state dict pull |
+---------------------------------------------------------------------------------------------------------------------+
                                                  |
                                                  | group = group_buffer.take_finalized()
                                                  v
_batcher_loop
+----------------------------------------------------------------------------------------+
| batch, trainable = batcher.add_group.call_one(group)                                   |
| if no trainable samples: group_buffer.release_active_groups(1, "untrainable_group")    |
| training_batch_queue.put(TrainerStepBatch)                                             |
+-----------------------+------------------------------+---------------------------------+
                        |  ^                           |
          add group     |  | maybe_training_batch      | put batch
                        v  |                           v
Batcher (own process)                                training_batch_queue
+-------------------------------------------------+   +----------------------------------------------+
| build_from_group -> add_training_samples        |   | size 1; holds TrainerStepBatch | None        |
| pack at num_prompts_per_train_step               |   +---------------------+------------------------+
+-------------------------------------------------+                         |
                                                                         | packed = training_batch_queue.get()
                                                                         v
_trainer_loop
+----------------------------------------------------------------------------------------------------------+
| train batch -> optim -> push/pull weights -> buffer.release_active_groups(num_prompts_per_train_step)     |
+----------------------------------------------------------------------------------------------------------+

Backpressure (each loop: what it consumes/produces, and what gates each side):
_data_input_loop
  produces: RolloutGroupWork into group_buffer
    waits for:    a free active slot (group_buffer.wait_for_slot)
    unblocked by: _trainer_loop release_active_groups(num_prompts_per_train_step, "trained") after the pull
                  (and _batcher_loop release_active_groups(1,"untrainable_group"))
_rollout_loop[N]
  consumes: a WAITING RolloutGroupWork (group_buffer.claim_next)
    waits for:    a claimable WAITING entry
    unblocked by: _data_input_loop group_buffer.add_work()
  produces: RolloutGroup (group_buffer.finalize_work)
    waits for:    nothing (admits its own claimed slot)
    unblocked by: n/a

_batcher_loop
  consumes: the oldest FINALIZED group inside the window (group_buffer.take_finalized)
    waits for:    a group inside the window becoming FINALIZED (any group when windowed_fifo_batches is None)
    unblocked by: _rollout_loop[N] group_buffer.finalize_work()
  produces: TrainerStepBatch (training_batch_queue.put)
    waits for:    a free training_batch_queue slot (maxsize=1)
    unblocked by: _trainer_loop training_batch_queue.get()
_trainer_loop
  consumes: a TrainerStepBatch (training_batch_queue.get)
    waits for:    a TrainerStepBatch in the queue
    unblocked by: _batcher_loop training_batch_queue.put()
"""

import asyncio
import copy
import json
import logging
import math
import os
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any

# PYTORCH_CUDA_ALLOC_CONF is set in torchtitan/rl/__init__.py (before torch is imported)
# and in train.py; see the note there.
import torch  # noqa: F401

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.components.renderer import RendererConfig
from torchtitan.components.tokenizer import HuggingFaceTokenizer
from torchtitan.config import apply_overrides, Configurable, OverrideConfig
from torchtitan.observability import structured_logger as sl
from torchtitan.rl.actors.system import ActorSystem, TrainerHandle
from torchtitan.rl.controller.metrics import (
    compute_host_memory_metrics,
    compute_perf_ratio_metrics,
    compute_policy_age_metrics,
    compute_rollout_metrics,
    MetricsTimer,
)
from torchtitan.rl.controller.work_buffer import (
    AdaptiveRolloutGroupWorkBuffer,
    DEFAULT_SEQS_PER_GENERATOR_DP_SHARD,
    RolloutGroupWork,
    RolloutGroupWorkBuffer,
)
from torchtitan.rl.generator.generator import Generator
from torchtitan.rl.generator.http_client import GeneratorClient
from torchtitan.rl.generator.types import (
    EXTRA_VALIDATION_GROUP_ID_STRIDE,
    OVERLAPPED_VALIDATION_FIRST_GROUP_ID,
    SamplingConfig,
)
from torchtitan.rl.observability import metrics as m
from torchtitan.rl.rollout import RolloutGroup
from torchtitan.rl.rollout.recorder import RolloutSampleRecorder
from torchtitan.rl.rollout.rollouter import Rollouter
from torchtitan.rl.rollout.types import GenerateFn, ReleaseSessionFn
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter
from torchtitan.rl.trainer.batcher import Batcher, TrainerStepBatch
from torchtitan.rl.trainer.controller_state import DATALOADER_STATE_KEY
from torchtitan.rl.trainer.losses import ScoreCenteringLoss
from torchtitan.rl.trainer.trainer import Trainer
from torchtitan.tools.garbage_collector import GarbageCollector

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class ValidationConfig:
    """Held-out validation that runs before training, every `freq` steps, and after the last step.

    Example, `freq=25`, `overlap_training=True`, 100 steps:
        step 0:   pass 0 starts on policy 0; training starts at once.
        step 7:   pass 0 ends and is logged at step 7; its slowest rollouts sampled up to policy 6.
        step 25:  pass 25 starts on policy 25 or later. Had pass 0 still been running, pass 25 would
                  start at the end of the step in which pass 0 ends.
        step 100: a pass that ends in step 100, or is still running, is logged at step 100; the
                  final pass then logs at step 101.
    """

    steps: int = 20
    """Maximum prompts per pass. -1 consumes one finite source pass; 0 disables."""

    freq: int | None = None
    """Also validate after every `freq` train steps; None validates only before and after."""

    overlap_training: bool = False
    """False: training waits for each pass, so all its rollouts sample one policy.
    True: the pre-training and periodic passes run beside training on the same generators:
    - at the lowest vLLM priority, so on busy generators a pass takes longer;
    - a rollout's later tokens can sample a newer policy (`validation/max_policy_version`);
    - a failed pass logs `validation/pass_failures`; training continues.
    Keep False for a clean pre-training score."""

    temperature: float = 1.0
    """Sampling temperature. 1.0 samples the model's distribution as is; 0.0 is greedy and needs
    `num_samples_per_prompt=1`."""

    num_samples_per_prompt: int = 1
    """Sibling rollouts sampled per prompt. `validation_reward` averages all of them; set 4 for
    lower-noise validation. With n > 1, `validation_reward/best_of_{n}` averages each prompt's
    best reward."""

    def __post_init__(self) -> None:
        if self.steps < -1:
            raise ValueError("validation steps must be -1 or non-negative")
        if self.freq is not None and self.freq <= 0:
            raise ValueError(f"validation freq must be positive, got {self.freq}")
        if self.temperature < 0:
            raise ValueError(
                f"validation temperature must be non-negative, got {self.temperature}"
            )
        if self.num_samples_per_prompt < 1:
            raise ValueError(
                "validation num_samples_per_prompt must be >= 1, got "
                f"{self.num_samples_per_prompt}"
            )
        if self.temperature == 0 and self.num_samples_per_prompt > 1:
            raise ValueError(
                "greedy validation (temperature=0) repeats the same rollout; set "
                f"num_samples_per_prompt=1, got {self.num_samples_per_prompt}"
            )


@dataclass(kw_only=True, slots=True)
class ExtraValidationConfig:
    """A named validation set beside `AsyncLoopConfig.validation`, e.g. a slow benchmark.

    Its passes always overlap training, each set keeps at most one pass running, and passes of
    different sets run concurrently. Unlike the main set it has no pre/post-training blocking pass:
    a pass starts at the first step and then every `freq` steps; a pass still running when training
    ends is awaited and logged at the last step. Its generator requests get the lowest vLLM priority
    (`vllm_priority`), so a long pass never delays training rollouts. Its tasks come from the
    rollouter's validation set of the same name. Metrics go under `validation/{name}/...`:

        validation/{name}_reward/mean   mean reward over every sample (= pass@1 for 0/1 rewards)
        validation/{name}/best_of_{k}   each prompt's best sample, averaged (k > 1)
        validation/{name}/pass_at_{k}   same value, logged only with `binary_reward`
    """

    name: str
    """Validation set name; the rollouter must have a validation set of this name."""

    freq: int
    """Start a pass every `freq` train steps (if the previous one has ended)."""

    steps: int = -1
    """Maximum prompts per pass; -1 runs the whole set."""

    temperature: float = 1.0
    """Sampling temperature."""

    num_samples_per_prompt: int = 1
    """Sibling rollouts sampled per prompt."""

    binary_reward: bool = False
    """Rewards are 0/1, so `best_of_k` is pass@k and is also logged as `pass_at_{k}`."""

    def __post_init__(self) -> None:
        if not self.name or "/" in self.name:
            raise ValueError(
                f"extra validation name must be non-empty without '/', got {self.name!r}"
            )
        if self.freq <= 0:
            raise ValueError(f"extra validation freq must be positive, got {self.freq}")
        if self.steps == 0 or self.steps < -1:
            raise ValueError("extra validation steps must be -1 or positive")
        if self.temperature < 0:
            raise ValueError("extra validation temperature must be non-negative")
        if self.num_samples_per_prompt < 1:
            raise ValueError("extra validation num_samples_per_prompt must be >= 1")
        if self.temperature == 0 and self.num_samples_per_prompt > 1:
            raise ValueError(
                "greedy validation (temperature=0) repeats the same rollout; set "
                "num_samples_per_prompt=1"
            )


@dataclass(kw_only=True, slots=True)
class AsyncLoopConfig(Configurable.Config):
    num_training_steps: int = 10
    """Optimizer steps to run."""

    num_prompts_per_train_step: int = 8
    """Global number of prompt groups, across all DPs, whose surviving rollouts compose
    one train step (the global_batch_size, in groups)."""

    num_samples_per_prompt: int = 8
    """Sibling rollouts sampled per prompt (the GRPO group)."""

    target_offpolicy_steps: int = 3
    """Target steady-state offpolicy steps used to set the active buffer size to
    `(S + 1) * P`. Observed offpolicy steps are not guaranteed to equal this
    target: when rollout generation is the bottleneck, the buffer may not fill
    and observed offpolicy steps will be lower. A finite `windowed_fifo_batches` bounds
    how far a slow group may exceed this target; None leaves it unbounded. See
    ``torchtitan/rl/controller/WINDOWED_FIFO.md`` for details."""

    windowed_fifo_batches: int | None = None
    """FIFO look-ahead window in train batches.

    None (the default) is greedy: the batcher takes the oldest finished group
    anywhere in the buffer. An unfinished older group does not block a younger
    finished group. Maximum policy age is unbounded.

    Set to `n >= 1` to limit consumption to `n * P` group ids from the oldest
    group in the buffer. Maximum policy age is bounded by
    `target_offpolicy_steps + n`. A value of 1 is FIFO by batch. See
    ``torchtitan/rl/controller/WINDOWED_FIFO.md``."""

    group_buffer: RolloutGroupWorkBuffer.Config | AdaptiveRolloutGroupWorkBuffer.Config = field(
        default_factory=RolloutGroupWorkBuffer.Config
    )
    """Which buffer paces generation:
    - `RolloutGroupWorkBuffer` (default): `(target_offpolicy_steps + 1) * P` slots, windowed by
      `windowed_fifo_batches`.
    - `AdaptiveRolloutGroupWorkBuffer`: slots learned from the run, under its own age knobs. It ignores
      `windowed_fifo_batches`, and `target_offpolicy_steps` only sets
      `train_batch/pct_samples_over_target_age`."""
    batcher: Batcher.Config = field(default_factory=Batcher.Config)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    extra_validations: list[ExtraValidationConfig] = field(default_factory=list)
    """Named validation sets beside `validation`, each on its own schedule (see
    `ExtraValidationConfig`)."""

    def __post_init__(self) -> None:
        extra_names = [extra.name for extra in self.extra_validations]
        if len(set(extra_names)) != len(extra_names):
            raise ValueError(
                f"extra_validations names must be unique, got {extra_names}"
            )
        if self.num_prompts_per_train_step < 1:
            raise ValueError(
                "num_prompts_per_train_step must be >= 1, got "
                f"{self.num_prompts_per_train_step}"
            )
        if self.target_offpolicy_steps < 0:
            raise ValueError(
                f"target_offpolicy_steps must be >= 0, got {self.target_offpolicy_steps}"
            )
        if self.windowed_fifo_batches is not None and self.windowed_fifo_batches < 1:
            raise ValueError(
                f"windowed_fifo_batches must be None or >= 1, got {self.windowed_fifo_batches}"
            )

    @property
    def max_active_rollout_groups(self) -> int:
        return (self.target_offpolicy_steps + 1) * self.num_prompts_per_train_step

    @property
    def window_size(self) -> int | None:
        """FIFO look-ahead window in group ids, `windowed_fifo_batches * P`; None means no window."""
        if self.windowed_fifo_batches is None:
            return None
        return self.windowed_fifo_batches * self.num_prompts_per_train_step

    @property
    def max_concurrent_rollout_groups(self) -> int:
        """Most groups generating at once: every active slot of the fixed buffer, or the adaptive
        buffer's `generation_capacity`. Sizes the rollout workers and vLLM's `max_num_seqs`.
        Raises if the adaptive default needs the generator layout (no `target_offpolicy_steps`);
        the controller then uses `resolve_max_concurrent_rollout_groups`."""
        return self.resolve_max_concurrent_rollout_groups()

    def resolve_max_concurrent_rollout_groups(
        self, num_generator_dp_shards: int | None = None
    ) -> int:
        """`max_concurrent_rollout_groups`, given the generator layout when the default needs it.

        Example:
            # adaptive buffer, no target, 24 generator DP shards, 16 samples per prompt
            async_loop.resolve_max_concurrent_rollout_groups(num_generator_dp_shards=24)  # 768
        """
        if isinstance(self.group_buffer, AdaptiveRolloutGroupWorkBuffer.Config):
            return self.group_buffer.resolve_generation_capacity(
                self.num_prompts_per_train_step,
                num_generator_dp_shards=num_generator_dp_shards,
                num_samples_per_prompt=self.num_samples_per_prompt,
            )
        return self.max_active_rollout_groups

    @property
    def max_offpolicy_steps(self) -> int | None:
        """Return the worst case consume-time offpolicy bound, or None without a window.

        For active buffer size `B`, window size `W`, and prompts per train step
        `P`, the bound is `(B + W - 2) // P`, which is `S + windowed_fifo_batches`.
        The adaptive buffer drops groups past its own `max_offpolicy_steps` instead.
        """
        if isinstance(self.group_buffer, AdaptiveRolloutGroupWorkBuffer.Config):
            return self.group_buffer.max_offpolicy_steps
        if self.window_size is None:
            return None
        return (
            self.max_active_rollout_groups + self.window_size - 2
        ) // self.num_prompts_per_train_step


def _default_actor_system_config() -> ActorSystem.Config:
    # Imported here so the controller depends on Monarch only when it is the configured
    # actor system.
    from torchtitan.rl.actors.monarch.system import MonarchActorSystem

    return MonarchActorSystem.Config()


class Controller(Configurable):
    """Top-level RL async training orchestrator.

    Owns a `Trainer` actor (gradient updates), `Generator` actors
    (sampling), and a `Rollouter` (datasets + rubric + env construction),
    placed on processes by its `ActorSystem` (`Config.actor_system`).

    Check the docstring at the top of the file for more details.

    Example:

        config = recipes.rl_grpo_qwen3_0_6b_varlen()
        controller = config.build()
        try:
            await controller.setup_async()
            await controller.run()
        finally:
            await controller.close()
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Top-level config for RL training."""

        hf_assets_path: str = "./tests/assets/tokenizer"
        """Path to HF assets folder (model weights, tokenizer, config files)."""

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts (temp weights, logs, etc.)."""

        async_loop: AsyncLoopConfig = field(default_factory=AsyncLoopConfig)
        """How the data->rollout->batch->train loop is sized and coordinated."""

        rollouter: Rollouter.Config
        """The rollouter: its datasets, envs, and rubric."""
        # TODO: support multiple rollouters for data mixing.

        tokenizer: HuggingFaceTokenizer.Config = field(
            default_factory=HuggingFaceTokenizer.Config
        )
        """Tokenizer loaded from `hf_assets_path`."""

        renderer: RendererConfig
        """The model's chat template; renders messages to token ids and parses completions
        back. E.g. `from_renderers(Qwen3RendererConfig(enable_thinking=False))`."""

        rollout_recorder: RolloutSampleRecorder.Config = field(
            default_factory=RolloutSampleRecorder.Config
        )
        """JSONL recorder to save sampled rollouts to disk for further inspection and debugging."""

        trainer: Trainer.Config
        """Trainer config. Controls the trainer model, optimizer, training, parallelism."""

        # TODO: put generator, num generators and generator router in a separate config
        generator: Generator.Config
        """Generator actor configuration (generator model, vLLM engine, sampling)."""

        num_generators: int = 1
        """Number of generator replicas to spawn as separate proc meshes.

        This is distinct from intra-generator parallelism controlled by
        ``generator.parallelism``. Total generator GPU/process usage is
        ``num_generators * generator_world_size``.
        """

        generator_router: InterGeneratorRouter.Config = field(
            default_factory=InterGeneratorRouter.Config
        )
        """Generator routing strategy configuration."""

        actor_system: ActorSystem.Config = field(
            default_factory=_default_actor_system_config
        )
        """The actor framework that places the trainer, generators, router and rollout
        workers on processes, selected by config type, e.g.
        ``MonarchActorSystem.Config()`` (default)."""

        # TODO: rename it to metrics_processor
        metrics: m.MetricsProcessor.Config = field(
            default_factory=m.MetricsProcessor.Config
        )

        override: OverrideConfig = field(default_factory=OverrideConfig)
        """Registered overrides applied to this whole config tree, including the
        trainer and generator models, when the controller is constructed."""

        def maybe_log(self) -> None:
            debug = self.trainer.debug
            config_dict = self.to_dict()
            if debug.print_config:
                logger.info(
                    f"Running with configs: {json.dumps(config_dict, indent=2, ensure_ascii=False)}"
                )

            if debug.save_config_file is not None:
                config_file = os.path.join(self.dump_folder, debug.save_config_file)
                os.makedirs(os.path.dirname(config_file), exist_ok=True)
                with open(config_file, "w") as file:
                    json.dump(config_dict, file, indent=2)
                logger.info(f"Saved job configs to {config_file}")

        def __post_init__(self):
            trainer_model_cls = type(self.trainer.model)
            generator_model_cls = type(self.generator.model)
            if trainer_model_cls is not generator_model_cls:
                raise ValueError(
                    "The trainer and generator must use the same model class, got "
                    f"{trainer_model_cls.__qualname__} and "
                    f"{generator_model_cls.__qualname__}."
                )
            if self.num_generators < 1:
                raise ValueError(
                    f"num_generators must be at least 1, got {self.num_generators}"
                )
            if self.generator.checkpointer is not None:
                raise ValueError(
                    "Generator checkpoint must be disabled in the RL loop "
                    "(weights are synced from the trainer via TorchStore). "
                    "Set generator.checkpointer=None."
                )
            if self.trainer.training.num_tokens_per_train_step != -1:
                warnings.warn(
                    "trainer.training.num_tokens_per_train_step is ignored by "
                    "the RL loop; configure async_loop.num_prompts_per_train_step "
                    "to control optimizer-step boundaries.",
                    stacklevel=2,
                )
            if (
                self.async_loop.batcher.rdma_min_tensor_bytes is not None
                and not self.trainer.training.preprocess_microbatch_groups_lazily
            ):
                warnings.warn(
                    "async_loop.batcher.rdma_min_tensor_bytes is set without "
                    "trainer.training.preprocess_microbatch_groups_lazily=True: trainer "
                    "ranks read all microbatches before forward/backward, with no overlap.",
                    stacklevel=2,
                )
            # TODO: read RDMA microbatches in pipeline-step groups.
            if (
                self.async_loop.batcher.rdma_min_tensor_bytes is not None
                and self.trainer.parallelism.pipeline_parallel_degree > 1
            ):
                raise ValueError(
                    "async_loop.batcher.rdma_min_tensor_bytes does not support "
                    "trainer pipeline parallelism yet."
                )
            if self.trainer.parallelism.enable_sequence_parallel:
                sp_degree = self.trainer.parallelism.tensor_parallel_degree
                max_context_length = self.trainer.training.max_context_length
                if sp_degree > 1 and max_context_length % sp_degree != 0:
                    raise ValueError(
                        "training.max_context_length "
                        f"({max_context_length}) must be divisible "
                        f"by sequence parallel degree ({sp_degree})."
                    )

            loss_config = self.trainer.loss
            if isinstance(loss_config, ChunkedLossWrapper.Config):
                loss_config = loss_config.loss_fn
            uses_score_centering = isinstance(loss_config, ScoreCenteringLoss.Config)
            if uses_score_centering != (self.generator.sampling.num_topk_logprobs > 0):
                raise ValueError(
                    "ScoreCenteringLoss needs generator.sampling.num_topk_logprobs > 0, "
                    "and other losses need 0 (they take no top-k inputs). Got "
                    f"{type(loss_config).__qualname__} with num_topk_logprobs="
                    f"{self.generator.sampling.num_topk_logprobs}."
                )

            # TODO: add a check so that all seq_len related variables make sense
            # e.g. rollout max length cannot be larger than the model max_seq_len
            # or the packing len, etc.

            if self.trainer.debug.batch_invariant:
                if torch.version.hip is not None:
                    raise ValueError(
                        "batch_invariant mode is not supported on ROCm: the varlen "
                        "attention path cannot force num_splits=1 (rejected by ROCm), "
                        "so split-k reductions are non-deterministic."
                    )
                if not self.trainer.debug.deterministic:
                    raise ValueError("batch_invariant requires deterministic=True")
                # The trainer forward must compute in bf16 to match the bf16
                # generator, via FSDP mixed precision. The trainer always wraps
                # the model in FSDP (even at data_parallel_shard_degree=1, where
                # FSDP acts purely as a mixed-precision boundary), so
                # mixed_precision_param == "bfloat16" casts the fp32 master
                # weights to bf16 for the forward before any matmul.
                if self.trainer.training.mixed_precision_param != "bfloat16":
                    raise ValueError(
                        "batch_invariant requires the trainer forward to compute "
                        "in bfloat16 to match the generator. Set "
                        "training.mixed_precision_param='bfloat16' (fp32 master "
                        "weights, bf16-cast forward via FSDP mixed precision). "
                        "Got mixed_precision_param="
                        f"{self.trainer.training.mixed_precision_param!r}."
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
        # Copy each model so overrides never write through a model config the
        # recipe shares between the trainer and generator.
        config.trainer.model = copy.deepcopy(config.trainer.model)
        config.generator.model = copy.deepcopy(config.generator.model)
        if config.override.imports:
            apply_overrides(config.override, config)
        config.maybe_log()
        self.actor_system: ActorSystem | None = None
        self.trainer: TrainerHandle | None = None
        self.generator_router: InterGeneratorRouter | None = None
        # The batcher's single-actor handle.
        self.batcher: Any | None = None
        # Resume step (0 = fresh); set in setup_async from the loaded checkpoint.
        self.start_step = 0
        self.metrics_processor: m.MetricsProcessor = config.metrics.build(
            log_dir=config.dump_folder,
            job_config=config.to_dict(),
        )
        self.tokenizer = config.tokenizer.build(tokenizer_path=config.hf_assets_path)
        self.renderer = config.renderer.build(tokenizer=self.tokenizer)

        # Carry the base seed and renderer stop tokens on the sampling config so
        # the generator reads them off each request; the rollouter offsets the
        # seed per sample. Avoids the generator depending on request_id format.
        self._sampling = replace(
            config.generator.sampling,
            seed=config.generator.debug.seed,
            stop_token_ids=list(self.renderer.get_stop_token_ids()),
        )
        self._rollouter: Rollouter = config.rollouter.build()
        self.rollout_recorder = config.rollout_recorder.build(
            dump_dir=config.dump_folder
        )
        # With `validation.overlap_training`: the pass running beside training.
        self._validation_task: asyncio.Task[list[m.Metric]] | None = None
        # Extra validation sets: the running pass of each, by name; and sets due to start one.
        self._extra_validation_tasks: dict[str, asyncio.Task[list[m.Metric]]] = {}
        self._extra_validation_requested: set[str] = set()

    async def close(self):
        """Best-effort: tear down actors, close metric backends, then stop the actor system."""
        logger.info("Closing: tearing down actors and process meshes.")

        # Still running only if run() crashed; finish cancelling it before closing the rollouter it uses.
        if self._validation_task is not None:
            self._validation_task.cancel()
            await asyncio.gather(self._validation_task, return_exceptions=True)
        extra_tasks = list(self._extra_validation_tasks.values())
        for task in extra_tasks:
            task.cancel()
        await asyncio.gather(*extra_tasks, return_exceptions=True)

        if self.trainer is not None:
            try:
                await self.trainer.all_ranks.close.call()
            except Exception:
                logger.exception("trainer.close failed")

        # Close generators before the rollouter, so in-flight rollout groups fail fast; see `Rollouter.close`.
        if self.generator_router is not None:
            try:
                close_results = await self.generator_router.close_generators.call_one()
                for idx, result in enumerate(close_results):
                    if isinstance(result, BaseException):
                        actor_name = (
                            "generator"
                            if len(close_results) == 1
                            else f"generator[{idx}]"
                        )
                        logger.error(
                            "%s.close failed",
                            actor_name,
                            exc_info=(type(result), result, result.__traceback__),
                        )
            except Exception:
                logger.exception("generator_router.close_generators failed")

        try:
            await self._rollouter.close()
        except Exception:
            logger.exception("rollouter.close failed")

        try:
            self.metrics_processor.close()
        except Exception:
            logger.exception("metrics_processor close failed")

        if self.actor_system is not None:
            await self.actor_system.stop()

    def _get_rank_0_value(self, result):
        """Extract rank 0 result from a Monarch ValueMesh.

        Monarch actor endpoints return results from all ranks in the mesh.
        This method picks out rank 0's result. This should be used in cases
        where all ranks return the same result.
        """
        return result.get(0)

    @sl.log_trace_span("setup_async")
    async def setup_async(self, *, actor_system: ActorSystem | None = None):
        """Spawn the services through the actor system and initialize weights.

        Kept separate from ``__init__`` because provisioning processes, spawning
        actors, TorchStore initialization, and the initial model state dict
        push/pull are all ``await``-based runtime side effects that cannot run in
        a synchronous constructor.

        Provisions and spawns the trainer, generators, router, batcher and rollout
        workers through the actor system, and synchronizes initial weights from
        trainer to generator. Must be called before :meth:`run`.

        Args:
            actor_system: An actor system the launcher built, e.g. a
                ``MonarchActorSystem`` on hosts it allocated; None builds
                ``config.actor_system``.
        """
        config = self.config
        generator_dp_degree = max(config.generator.parallelism.data_parallel_degree, 1)
        num_generator_dp_shards = config.num_generators * generator_dp_degree

        # Peak concurrent rollout sequences (groups * num_samples_per_prompt, or the validation pass);
        # sizes max_num_seqs below. With periodic or overlapped validation, training rollouts keep
        # generating during a pass, so the peak is the sum.
        async_loop = self.config.async_loop
        self._max_concurrent_rollout_groups = (
            async_loop.resolve_max_concurrent_rollout_groups(num_generator_dp_shards)
        )
        num_training_rollouts = (
            self._max_concurrent_rollout_groups * async_loop.num_samples_per_prompt
        )
        validation = async_loop.validation
        num_validation_rollouts = (
            len(self._rollouter.get_validation_samples(validation.steps))
            * validation.num_samples_per_prompt
        )
        rollout_concurrency = (
            num_training_rollouts + num_validation_rollouts
            if validation.freq is not None or validation.overlap_training
            else max(num_training_rollouts, num_validation_rollouts)
        )
        # Extra validation passes always overlap training (and each other).
        rollout_concurrency += sum(
            len(self._rollouter.get_validation_samples(extra.steps, extra.name))
            * extra.num_samples_per_prompt
            for extra in async_loop.extra_validations
        )

        trainer_parallelism = config.trainer.parallelism
        dp_shard = max(trainer_parallelism.data_parallel_shard_degree, 1)
        self.trainer_dp_degree = (
            trainer_parallelism.data_parallel_replicate_degree * dp_shard
        )

        # Ceiling (not target) for the generator's max_num_seqs: the per-generator
        # upper bound on concurrently scheduled sequences. vLLM may admit fewer if KV
        # is tight; this also sets CUDA-graph capture sizes.
        max_num_seqs = min(
            math.ceil(rollout_concurrency / num_generator_dp_shards),
            DEFAULT_SEQS_PER_GENERATOR_DP_SHARD,
        )

        logger.info(
            "max_num_seqs=%d per generator (rollout_concurrency=%d / generator_dp_shards=%d)",
            max_num_seqs,
            rollout_concurrency,
            num_generator_dp_shards,
        )

        with sl.log_trace_span("mesh_spawn"):
            if actor_system is None:
                actor_system = config.actor_system.build()
            self.actor_system = actor_system
            await actor_system.provision(
                trainer_parallelism=config.trainer.parallelism,
                generator_parallelism=config.generator.parallelism,
                num_generators=config.num_generators,
            )
            self.trainer = await actor_system.spawn_trainer(
                config.trainer,
                dp_degree=self.trainer_dp_degree,
                hf_assets_path=config.hf_assets_path,
                generator_dtype=config.generator.model_dtype,
                max_num_documents=config.async_loop.batcher.max_num_documents,
                output_dir=config.dump_folder,
            )
            # TODO: torch.compile with aot_eager backend (inductor crashes the vLLM engine on the shared model path).
            self.generator_router = await actor_system.spawn_generator_router(
                config.generator_router,
                generator_config=config.generator,
                model_path=config.hf_assets_path,
                max_num_seqs=max_num_seqs,
                output_dir=config.dump_folder,
            )
            self.batcher = await actor_system.spawn_batcher(
                async_loop.batcher,
                num_tokens_per_microbatch_per_dp_rank=(
                    config.trainer.training.num_tokens_per_microbatch_per_dp_rank
                ),
                max_context_length=config.trainer.training.max_context_length,
                num_prompts_per_train_step=async_loop.num_prompts_per_train_step,
                dp_degree=self.trainer_dp_degree,
                num_pp_microbatches=(
                    config.trainer.parallelism.num_pp_microbatches
                    if config.trainer.parallelism.pipeline_parallel_degree > 1
                    else 1
                ),
                pad_id=self.tokenizer.eos_id,
                temperature=self._sampling.temperature,
                num_topk_logprobs=self._sampling.num_topk_logprobs,
                output_dir=config.dump_folder,
                enable_structured_logging=config.trainer.debug.enable_structured_logging,
                replay_routed_experts=config.generator.engine.return_routed_experts,
            )

            await self._rollouter.setup_async(
                actor_system=actor_system,
                tokenizer_config=config.tokenizer,
                renderer_config=config.renderer,
                hf_assets_path=config.hf_assets_path,
                training_sample_builder=config.async_loop.batcher.training_sample_builder,
            )

        # Initialize TorchStore, which carries the model state dict from the trainer to the generators.
        with sl.log_trace_span("torchstore_init"):
            await actor_system.initialize_torchstore()

        # Resume: __init__ ran CheckpointManager.load(); read back the restored
        # policy version and controller-owned data state.
        self.start_step, controller_state = await asyncio.gather(
            self.trainer.rank0.get_policy_version.call_one(),
            self.trainer.rank0.get_controller_state.call_one(),
        )
        if controller_state is not None:
            dataloader_state = controller_state[DATALOADER_STATE_KEY]
            self._rollouter.load_state_dict(dataloader_state)
            logger.info("Restored RL dataloader state")
            # TODO: Replay cached finished rollouts instead of regenerating all
            # pending prompts after resume.
        elif self.start_step > 0:
            logger.warning(
                "Resumed at step %d without RL data state; the prompt stream "
                "restarts from the beginning.",
                self.start_step,
            )
        if self.start_step > 0:
            logger.info(f"Resuming RL training from step {self.start_step}")

        # Start each generator's engine loop on all ranks once, before any
        # rank-0-only generate / pull (rank 0 drives the followers through this
        # loop, so every rank must be running it first).
        with sl.log_trace_span("generator_start_engine_loop"):
            await self.generator_router.start_engine_loop.call_one()

        # Initial push and pull: only the trainer loads weights; generators pull them at start_step.
        with sl.log_trace_span("trainer_push_model_state_dict"):
            await self.trainer.all_ranks.push_model_state_dict.call()
        with sl.log_trace_span("generator_pull_model_state_dict"):
            await self.generator_router.pull_model_state_dict.call_one(self.start_step)

    # TODO: fold validation into a Validator(Configurable) the controller attaches, instead of these methods.
    @sl.log_trace_span("_collect_validation_rollouts")
    async def _collect_validation_rollouts(
        self,
        *,
        samples: list[object],
        sampling: SamplingConfig,
        step: int,
        overlapped: bool = False,
        extra: ExtraValidationConfig | None = None,
        extra_index: int = 0,
    ) -> tuple[list[RolloutGroup], list[m.Metric]]:
        """Run each held-out prompt `num_samples_per_prompt` times concurrently and emit validation metrics.

        `extra` selects an extra validation set (`extra_index` is its position in
        `extra_validations`): its own group-id range and metric prefix `validation/{name}`.

        Example: 2 prompts x 2 samples with rewards [[1, 0], [0, 0]]:

            validation_reward/_mean           0.25   # mean over all 4 rollouts
            validation_reward/best_of_2/mean  0.5    # best rollout per prompt, averaged (n > 1 only)
        """
        # Negative ids keep validation disjoint from training group ids, so their request_ids can't
        # collide in the shared engine (e.g. post-validation). Overlapped and extra passes get ids
        # at or below OVERLAPPED_VALIDATION_FIRST_GROUP_ID, hence the lowest vLLM priority
        # (`vllm_priority`); each extra set has its own range.
        if extra is None:
            num_samples_per_prompt = (
                self.config.async_loop.validation.num_samples_per_prompt
            )
            first_group_id = OVERLAPPED_VALIDATION_FIRST_GROUP_ID if overlapped else -1
            prefix = "validation"
            best_of_prefix = "validation_reward"
            generator_prefix = "validation_generator"
        else:
            num_samples_per_prompt = extra.num_samples_per_prompt
            first_group_id = (
                OVERLAPPED_VALIDATION_FIRST_GROUP_ID
                - (extra_index + 1) * EXTRA_VALIDATION_GROUP_ID_STRIDE
            )
            prefix = best_of_prefix = f"validation/{extra.name}"
            generator_prefix = f"validation_{extra.name}_generator"
        group_ids = [first_group_id - i for i in range(len(samples))]
        client = GeneratorClient(
            router=self.generator_router, metrics_prefix=generator_prefix
        )
        # TODO(naming): reserve "sample" for TrainingSample; rename the rollouter's raw-prompt "sample" -> "prompt"/"data_input".
        group_results = await asyncio.gather(
            *(
                self._rollouter.run_group_rollouts(
                    generate_fn=client,
                    release_session_fn=client.release_session,
                    sample=sample,
                    group_id=group_id,
                    group_size=num_samples_per_prompt,
                    sampling=sampling,
                )
                for group_id, sample in zip(group_ids, samples, strict=True)
            ),
            return_exceptions=True,
        )
        # Validation group ids are reused every validation, so their cache salts must not outlive it.
        await self.generator_router.release_groups.call_one(group_ids)

        # Keep the groups that succeeded; log + count the ones that raised.
        rollout_groups: list[RolloutGroup] = []
        num_failed_groups = 0
        for group_id, result in zip(group_ids, group_results, strict=True):
            if isinstance(result, BaseException):
                logger.error(
                    f"{prefix} group {group_id} (step={step}) failed; dropping",
                    exc_info=(type(result), result, result.__traceback__),
                )
                num_failed_groups += 1
                continue
            rollout_groups.append(result)

        rollouts = [rollout for group in rollout_groups for rollout in group.rollouts]
        metrics = compute_rollout_metrics(prefix=prefix, rollouts=rollouts)
        metrics.append(
            m.Metric(f"{prefix}/group_failures", m.Sum(float(num_failed_groups)))
        )
        # Policies the pass sampled. One, unless the pass overlaps training: then a weight sync
        # during a rollout makes its later tokens sample a newer policy.
        turns = [turn for rollout in rollouts for turn in rollout.turns]
        is_mixed_policy = [
            rollout.turns[-1].max_policy_version > rollout.turns[0].min_policy_version
            for rollout in rollouts
            if rollout.turns
        ]
        metrics += [
            m.Metric(
                f"{prefix}/min_policy_version",
                m.Min.from_list([turn.min_policy_version for turn in turns]),
            ),
            m.Metric(
                f"{prefix}/max_policy_version",
                m.Max.from_list([turn.max_policy_version for turn in turns]),
            ),
            m.Metric(
                f"{prefix}/mixed_policy_rollouts", m.Mean.from_list(is_mixed_policy)
            ),
        ]
        if num_samples_per_prompt > 1:
            # Equals pass@n when rewards are 0/1.
            best_rewards = [
                max(rollout.reward for rollout in group.rollouts)
                for group in rollout_groups
            ]
            metrics.append(
                m.Metric(
                    f"{best_of_prefix}/best_of_{num_samples_per_prompt}",
                    m.Mean.from_list(best_rewards),
                )
            )
            if extra is not None and extra.binary_reward:
                metrics.append(
                    m.Metric(
                        f"{prefix}/pass_at_{num_samples_per_prompt}",
                        m.Mean.from_list(best_rewards),
                    )
                )
        return rollout_groups, metrics

    @sl.log_trace_span("validate")
    async def validate(self, *, step: int, overlapped: bool = False) -> list[m.Metric]:
        """Run validation on held-out prompts.

        Args:
            step: Training step this validation pass belongs to (0 for the
                pre-training pass); tagged into logged rollout samples.
            overlapped: The pass runs while training keeps stepping, so its wall time is
                logged as `timing/validation/overlapped`, not `timing/validation/blocking`,
                and its generator requests get the lowest vLLM priority.

        Returns:
            Validation rollout metrics, generation metrics, and validation
            timing.
        """
        t_validate_start = time.perf_counter()
        steps = self.config.async_loop.validation.steps
        if steps == 0:  # skip validation (e.g. loss guard CI)
            return []
        # In a thread: building the dataset would otherwise block the training loops.
        samples = await asyncio.to_thread(self._rollouter.get_validation_samples, steps)
        # Validation rollouts are never trained, so skip the top-k transport.
        sampling = replace(
            self._sampling,
            temperature=self.config.async_loop.validation.temperature,
            num_topk_logprobs=0,
        )

        rollout_groups, validation_metrics = await self._collect_validation_rollouts(
            samples=samples, sampling=sampling, step=step, overlapped=overlapped
        )

        self.rollout_recorder.record(is_validation=True, rollout_groups=rollout_groups)

        t_validate_s = time.perf_counter() - t_validate_start
        timing_key = (
            "timing/validation/overlapped"
            if overlapped
            else "timing/validation/blocking"
        )
        validation_metrics.append(m.Metric(timing_key, m.NoReduce(t_validate_s)))
        return validation_metrics

    async def run(self) -> None:
        """Start every async loop and run until training completes or a stage crashes.

        Producers (_data_input_loop, _rollout_loop[N], _batcher_loop) loop forever, and
        _generator_loop waits for the generators' engine loops, which run until close; _trainer_loop is
        the only finite loop -- it runs num_training_steps, then returns, which drives shutdown.
        _generator_loop watches the training loop only: an engine loop that dies during pre- or
        post-training validation shows up as failed validation rollouts.

        Shutdown (healthy):  _trainer_loop finishes N steps -> run() finally ->
          group_buffer.close()  (wakes _data_input_loop / _rollout_loop / _batcher_loop blocked on the buffer)
          -> task.cancel()      (wakes anything blocked on training_batch_queue.put/get; close does NOT wake these)
          -> gather(..., return_exceptions=True)
        Shutdown (crash):    any loop raises -> appears in `done` -> run() re-raises -> same finally.
        """
        # Setup objects (modules, configs, dataset, actor handles) live until exit. Freeze them
        # so full GCs skip them: ~1M objects, ~0.5 s per full GC (Python 3.12, x86).
        GarbageCollector.freeze(
            "Full GC before freezing the controller's setup objects"
        )

        async_loop = self.config.async_loop
        num_training_steps = async_loop.num_training_steps
        logger.info(
            f"Running pre-training validation; then {num_training_steps} steps of async RL training"
        )

        sl.log_trace_instant("validation_start")
        pre_validation_task: asyncio.Task[list[m.Metric]] | None = None
        if async_loop.validation.overlap_training:
            # Generators already hold policy `start_step` (setup_async), so training can start right away.
            pre_validation_task = self._start_validation(step=self.start_step)
            pre_validation = {}  # read from the task after training
        else:
            pre_validation = await self._validate_and_log(step=self.start_step)
        # Extra validation sets start beside training from the first step.
        for extra_index in range(len(async_loop.extra_validations)):
            self._start_extra_validation(extra_index=extra_index, step=self.start_step)
        sl.log_trace_instant("training_start")

        # Trainer policy version, seeded from the resumed step; advances at each optimizer step.
        self._trainer_policy_version = self.start_step

        if isinstance(async_loop.group_buffer, AdaptiveRolloutGroupWorkBuffer.Config):
            logger.info(f"Adaptive rollout buffer: {async_loop.group_buffer}")
            num_rollout_workers = self._max_concurrent_rollout_groups
            # The capacity resolved in setup_async, where the generator layout is known.
            self._group_buffer = replace(
                async_loop.group_buffer,
                generation_capacity=self._max_concurrent_rollout_groups,
            ).build(
                num_prompts_per_train_step=async_loop.num_prompts_per_train_step,
                policy_version=self.start_step,
            )
        else:
            # Depth (S + 1) * P targets the mean policy age; the window caps the max age.
            max_active_rollout_groups = async_loop.max_active_rollout_groups
            window_size = async_loop.window_size
            num_rollout_workers = max_active_rollout_groups
            logger.info(
                f"max_active_rollout_groups={max_active_rollout_groups}, "
                f"target_offpolicy_steps={async_loop.target_offpolicy_steps}, "
                f"windowed_fifo_batches={async_loop.windowed_fifo_batches}, "
                f"max_offpolicy_steps={async_loop.max_offpolicy_steps}"
            )

            self._group_buffer = async_loop.group_buffer.build(
                max_active_rollout_groups=max_active_rollout_groups,
                window_size=window_size,
            )

        # training_batch_queue
        training_batch_queue: asyncio.Queue[TrainerStepBatch | None] = asyncio.Queue(
            maxsize=1
        )

        # rollout_loop
        client = GeneratorClient(router=self.generator_router)

        # One rollout worker per group that may generate at once: lets generation fill every active slot,
        # including the cold start (step 0 fills every active slot, not just num_prompts_per_train_step per wave).
        # TODO: support warm start
        rollout_tasks = [
            asyncio.create_task(
                self._rollout_loop(
                    group_buffer=self._group_buffer,
                    generate_fn=client,
                    release_session_fn=client.release_session,
                ),
                name=f"rollout_worker_{group_worker_id}",
            )
            for group_worker_id in range(num_rollout_workers)
        ]

        # data_input_loop
        data_input_task = asyncio.create_task(
            self._data_input_loop(self._group_buffer), name="data_input"
        )

        # training_sample_batcher_loop
        batcher_task = asyncio.create_task(
            self._batcher_loop(training_batch_queue=training_batch_queue),
            name="batcher",
        )

        # trainer_loop
        trainer_task = asyncio.create_task(
            self._trainer_loop(
                training_batch_queue, num_training_steps=num_training_steps
            ),
            name="trainer",
        )

        # Engine loops run until the generators close, so this ends only if one stops mid-run.
        # Otherwise a crashed generator would only show up as failed rollouts.
        generator_task = asyncio.create_task(
            self._generator_loop(), name="generator_wait_engine_loop"
        )

        # run everything until trainer finishes its number of steps
        # or some other loop breaks
        background_tasks = [
            data_input_task,
            *rollout_tasks,
            batcher_task,
            generator_task,
        ]
        try:
            done, _ = await asyncio.wait(
                [trainer_task, *background_tasks], return_when=asyncio.FIRST_COMPLETED
            )
            # The trainer is the finite clock: it runs num_training_steps then returns -> training is done.
            # Producers loop forever, so a producer in `done` means it crashed (await re-raises) or wrongly
            # returned cleanly (the RuntimeError). Check producers even when the trainer also finished this
            # wakeup, so a simultaneous producer crash isn't hidden behind the finished trainer.
            for task in done:
                if task is trainer_task:
                    continue
                await task  # raises if task crashed; returns if task ended cleanly
                raise RuntimeError(f"{task.get_name()} exited unexpectedly")
            if trainer_task in done:
                await trainer_task
        finally:
            # Graceful first: buffer.close() (awaited) wakes loops blocked on the buffer so they return.
            # Then cancel covers anything blocked on the queue (which close does not wake); gather awaits all.
            await self._group_buffer.close()
            for task in (*background_tasks, trainer_task):
                task.cancel()
            await asyncio.gather(
                *background_tasks, trainer_task, return_exceptions=True
            )

        final_pass_step = num_training_steps
        # Wait for a pass still running instead of cancelling it: its result gets logged, and the
        # generators would keep serving its requests anyway. A failed pass doesn't raise here.
        if self._validation_task is not None:
            logger.info(
                f"Training done; waiting for {self._validation_task.get_name()} before the final pass"
            )
            await asyncio.wait([self._validation_task])
            self._log_finished_validation(step=num_training_steps)
            # At the same step, W&B would keep only the final pass's values for keys both passes log.
            final_pass_step += 1
        # Await, not cancel, extra passes still running; they log at the last step (their keys
        # differ from the main set's, so they can share its step).
        if self._extra_validation_tasks:
            logger.info(
                "Training done; waiting for "
                f"{sorted(task.get_name() for task in self._extra_validation_tasks.values())}"
            )
            await asyncio.gather(
                *self._extra_validation_tasks.values(), return_exceptions=True
            )
            self._log_finished_extra_validations(step=num_training_steps, wait=True)
        if pre_validation_task is not None and pre_validation_task.exception() is None:
            pre_validation = m.MetricsProcessor._aggregate_metrics(
                pre_validation_task.result()
            )

        # Post-training validation (held-out eval after the final step).
        post_validation = await self._validate_and_log(step=final_pass_step)
        # Push the last row now; close() would push it only after teardown.
        self.metrics_processor.commit()
        self._log_reward_delta(pre_validation, post_validation)

    async def _generator_loop(self) -> None:
        """Wait until every generator's engine loop exits; raise as soon as one crashes."""
        await self.generator_router.wait_engine_loop.call_one()

    async def _validate_and_log(self, *, step: int) -> dict[str, float]:
        """Run one validation pass, log it, and return its aggregated values for the pre/post delta."""
        metrics = await self.validate(step=step)
        self.metrics_processor.log(step=step, metrics=metrics, is_validation=True)
        return m.MetricsProcessor._aggregate_metrics(metrics)

    def _start_validation(self, *, step: int) -> asyncio.Task[list[m.Metric]]:
        """Start a validation pass beside training; `_log_finished_validation` logs it once it ends.

        The caller makes sure the generators hold at least policy `step` and no pass is running.
        """
        # TODO: this pass's rollouts compete with training rollouts for env-server sandboxes, which
        # have no request priority; the generators already serve training first (`vllm_priority`).
        logger.info(f"Starting validation at step {step}, beside training")
        self._validation_task = asyncio.create_task(
            self.validate(step=step, overlapped=True),
            name=f"validation_step_{step}",
        )
        return self._validation_task

    def _log_finished_validation(self, *, step: int) -> None:
        """If the pass started by `_start_validation` has ended, log it at `step`. A failed pass logs
        its error and `validation/pass_failures` instead of raising.

        Logged at the step it ends, not the one it started at: metric backends need steps that never go back.
        """
        task = self._validation_task
        if task is None or not task.done():
            return
        self._validation_task = None
        error = task.exception()
        if error is not None:
            logger.error(
                f"{task.get_name()} failed; logging validation/pass_failures at step {step}",
                exc_info=error,
            )
            metrics = [m.Metric("validation/pass_failures", m.Sum(1.0))]
        else:
            logger.info(f"{task.get_name()} ended; logging it at step {step}")
            metrics = task.result()
        self.metrics_processor.log(step=step, metrics=metrics, is_validation=True)

    @sl.log_trace_span("validate_extra")
    async def validate_extra(self, *, extra_index: int, step: int) -> list[m.Metric]:
        """Run one pass of extra validation set `extra_validations[extra_index]` beside training.

        Returns its rollout metrics (`validation/{name}/...`) and its wall time
        (`timing/validation/{name}`).
        """
        extra = self.config.async_loop.extra_validations[extra_index]
        t_start = time.perf_counter()
        # In a thread: building the dataset would otherwise block the training loops.
        samples = await asyncio.to_thread(
            self._rollouter.get_validation_samples, extra.steps, extra.name
        )
        # Validation rollouts are never trained, so skip the top-k transport.
        sampling = replace(
            self._sampling, temperature=extra.temperature, num_topk_logprobs=0
        )
        rollout_groups, metrics = await self._collect_validation_rollouts(
            samples=samples,
            sampling=sampling,
            step=step,
            extra=extra,
            extra_index=extra_index,
        )
        self.rollout_recorder.record(is_validation=True, rollout_groups=rollout_groups)
        metrics.append(
            m.Metric(
                f"timing/validation/{extra.name}",
                m.NoReduce(time.perf_counter() - t_start),
            )
        )
        return metrics

    def _start_extra_validation(self, *, extra_index: int, step: int) -> None:
        """Start a pass of an extra validation set beside training; the caller makes sure none of
        that set is running and the generators hold at least policy `step`."""
        name = self.config.async_loop.extra_validations[extra_index].name
        logger.info(f"Starting validation set {name!r} at step {step}, beside training")
        self._extra_validation_tasks[name] = asyncio.create_task(
            self.validate_extra(extra_index=extra_index, step=step),
            name=f"validation_{name}_step_{step}",
        )

    def _log_finished_extra_validations(self, *, step: int, wait: bool = False) -> None:
        """Log every extra validation pass that has ended (all of them with `wait`, which the
        caller must have awaited) at `step`. A failed pass is logged as
        `validation/{name}/pass_errors` and does not stop training; the set's next pass starts on
        schedule."""
        for name, task in list(self._extra_validation_tasks.items()):
            if not task.done():
                if wait:
                    raise RuntimeError(f"{task.get_name()} has not ended")
                continue
            del self._extra_validation_tasks[name]
            error = task.exception()
            if error is not None:
                logger.error(
                    f"Validation set {name!r} failed ({task.get_name()}); training continues",
                    exc_info=error,
                )
                metrics = [m.Metric(f"validation/{name}/pass_errors", m.Sum(1))]
            else:
                logger.info(f"{task.get_name()} ended; logging it at step {step}")
                metrics = task.result()
            self.metrics_processor.log(step=step, metrics=metrics, is_validation=True)

    async def _step_extra_validations(
        self, *, step: int, num_training_steps: int, pull_task: asyncio.Task[float]
    ) -> None:
        """At the end of train step `step`: log extra passes that ended, mark sets that are due,
        and start each due set whose previous pass has ended."""
        extras = self.config.async_loop.extra_validations
        if not extras or step >= num_training_steps:
            # run() awaits and logs passes still running at the end.
            return
        self._log_finished_extra_validations(step=step)
        for extra in extras:
            if step % extra.freq == 0:
                self._extra_validation_requested.add(extra.name)
        for extra_index, extra in enumerate(extras):
            if (
                extra.name in self._extra_validation_requested
                and extra.name not in self._extra_validation_tasks
            ):
                # Wait only for this step's weight pull, so the pass starts on at least policy `step`.
                await pull_task
                self._start_extra_validation(extra_index=extra_index, step=step)
                self._extra_validation_requested.discard(extra.name)

    def _log_reward_delta(self, pre: dict[str, float], post: dict[str, float]) -> None:
        """Console pre/post reward summary, visible without scrolling back through the loop."""
        reward_keys = sorted(key for key in set(pre) | set(post) if "reward" in key)
        logger.info("=" * 60)
        logger.info("Validation reward (pre / post):")
        # With `overlap_training`, "pre" is the first pass, which ran beside training.
        oldest_pre_policy = pre.get("validation/min_policy_version/min", 0)
        newest_pre_policy = pre.get("validation/max_policy_version/max", 0)
        if newest_pre_policy > oldest_pre_policy:
            logger.info(
                f"  pre sampled policies {oldest_pre_policy:.0f} to {newest_pre_policy:.0f}: "
                "not a clean pre-training score"
            )
        for key in reward_keys:
            logger.info(
                f"  {key}:  {pre.get(key, float('nan')):+.3f}  /  {post.get(key, float('nan')):+.3f}"
            )
        logger.info("=" * 60)

    async def _data_input_loop(self, group_buffer: RolloutGroupWorkBuffer) -> None:
        """produces a RolloutGroupWork into group_buffer.
        waits for:    a free active slot (group_buffer.wait_for_slot)
        unblocked by: _trainer_loop release_active_groups(num_prompts_per_train_step, "trained")
            after the pull (and _batcher_loop release_active_groups(1,"untrainable_group"))

        Separate from `_rollout_loop`, so slow data prep (e.g. on-the-fly question generation) overlaps
        generation instead of serializing in front of it.
        """
        # TODO(perf): Slots are current released in batches, while this loop is a single producer.
        # we could a) increase the number of threads; b) revisit how we release slots and see if
        # we can release them on the batcher while still preserving max offpolicy steps.
        # finally, c) we need to check how will this data input loop truly overlaps with the rollout loop.
        while await group_buffer.wait_for_slot():
            with sl.log_trace_span("get_training_sample"):
                # to_thread: Dont block on dataset reads
                index, sample = await asyncio.to_thread(
                    self._rollouter.get_training_sample
                )
            await group_buffer.add_work(
                RolloutGroupWork(
                    group_id=index,
                    sample=sample,
                )
            )
        logger.info("Buffer closed; data input loop stopping")

    async def _rollout_loop(
        self,
        *,
        group_buffer: RolloutGroupWorkBuffer,
        generate_fn: GenerateFn,
        release_session_fn: ReleaseSessionFn,
    ) -> None:
        """Generate + score one group at a time; a failed group becomes an empty group + a failure metric.

        Staleness is bounded by the buffer's active-slot budget. Raw rollouts are recorded before any drop,
        so dropped groups stay inspectable on disk.

        consumes: a WAITING RolloutGroupWork (group_buffer.claim_next)
            waits for:    a claimable WAITING entry
            unblocked by: _data_input_loop group_buffer.add_work()
        produces: RolloutGroup (group_buffer.finalize_work)
            waits for:    nothing (admits its own claimed slot)
            unblocked by: n/a
        """
        while True:
            work = await group_buffer.claim_next()
            if work is None:  # group_buffer closed/shutdown signal
                logger.info("Buffer closed; rollout worker stopping")
                return
            # A task of its own, so the buffer can cancel this group (too old to ever train) without
            # cancelling this worker.
            generation_task = asyncio.create_task(
                self._rollouter.run_group_rollouts(
                    generate_fn=generate_fn,
                    release_session_fn=release_session_fn,
                    sample=work.sample,
                    group_id=work.group_id,
                    group_size=self.config.async_loop.num_samples_per_prompt,
                    sampling=self._sampling,
                ),
                name=f"rollout_group_{work.group_id}",
            )
            group_buffer.track_generation(work, generation_task)
            group: RolloutGroup | None = None
            try:
                with sl.log_trace_span("rollout_group"):
                    group = await generation_task
                # Extend: the group arrives with its rollout worker's GC metrics.
                group.metrics += compute_rollout_metrics(
                    prefix="rollout", rollouts=group.rollouts
                )
                self._rollouter.record_group_outcome(group)

                # save rollout for inspection
                self.rollout_recorder.record(
                    is_validation=False,
                    rollout_groups=[group],
                )
            except asyncio.CancelledError:
                if asyncio.current_task().cancelling():
                    # Shutdown cancelled this worker: stop the group too, then exit.
                    generation_task.cancel()
                    await asyncio.wait([generation_task])
                    raise
                if not work.cancelled:
                    # Not the buffer's cancel: propagate, as before groups ran in their own task.
                    raise
                # The buffer cancelled the group: it became too old to train while generating.
                logger.info(
                    f"rollout group {work.group_id} cancelled: past max_offpolicy_steps"
                )
            except Exception:
                logger.exception(f"rollout group {work.group_id} failed; dropping")
                group = RolloutGroup(
                    group_id=work.group_id,
                    rollouts=[],
                    metrics=[m.Metric("rollout/group_failures", m.Sum(1.0))],
                )
            finally:
                # The group makes no more generation calls, so its cache salts can go. Also on a
                # cancel: post-training validation reuses the generators.
                await self.generator_router.release_groups.call_one([work.group_id])
            if group is None:
                await group_buffer.release_cancelled(work.group_id)
            else:
                await group_buffer.finalize_work(group)

            # Drop both refs (work.rollout_group is the group too): held groups made every full GC slow.
            del group, work, generation_task

    async def _batcher_loop(
        self,
        *,
        training_batch_queue: "asyncio.Queue[TrainerStepBatch | None]",
    ) -> None:
        """Send finalized groups to the batcher actor and queue each training batch it returns.

        On a clean close/shutdown the group_buffer drains and returns None; we forward a `None` sentinel
        so the trainer stops.

        consumes: the oldest FINALIZED group inside the window (group_buffer.take_finalized)
            waits for:    a group inside the window becoming FINALIZED (any group when windowed_fifo_batches is None)
            unblocked by: _rollout_loop[N] group_buffer.finalize_work()
        produces: TrainerStepBatch (training_batch_queue.put)
            waits for:    a free training_batch_queue slot (maxsize=1)
            unblocked by: _trainer_loop training_batch_queue.get()
        """
        # Policy version that will train the batch being assembled: batches train in order, one per version.
        consuming_policy_version = self.start_step
        # Same filters as the batcher's builder: tells which rollouts it will leave out.
        sample_builder = self.config.async_loop.batcher.training_sample_builder.build()
        replay_routed_experts = self.config.generator.engine.return_routed_experts
        while True:
            rollout_group = await self._group_buffer.take_finalized(
                consuming_policy_version=consuming_policy_version
            )
            if rollout_group is None:  # closed and drained
                logger.info("Buffer drained; batcher loop stopping")
                break
            if replay_routed_experts:
                # The batcher won't train these rollouts, so don't send their router-replay expert
                # ids: ~320 bytes per token, most of a group's message.
                for rollout in sample_builder.rollouts_left_out(rollout_group):
                    for rollout_turn in rollout.turns:
                        rollout_turn.routed_expert_ids = None
            # We put a group in. We may get a batch back
            # if there are enough accumulated trainable groups to return one.
            with sl.log_trace_span("batcher_add_group"):
                (
                    maybe_training_batch,
                    group_is_trainable,
                ) = await self.batcher.add_group.call_one(rollout_group=rollout_group)
            if not group_is_trainable:
                await self._group_buffer.release_active_groups(
                    1, reason="untrainable_group"
                )
            if maybe_training_batch is not None:
                await training_batch_queue.put(maybe_training_batch)
                consuming_policy_version += 1
        await training_batch_queue.put(None)
        # TODO(async-rl): if finite datasets are supported, drain a final partial batch here.

    async def _trainer_loop(
        self,
        training_batch_queue: "asyncio.Queue[TrainerStepBatch | None]",
        *,
        num_training_steps: int,
    ) -> None:
        """Run num_training_steps optimizer steps: train one packed batch, push the trainer's model
        state dict, then pull it into the generators, log metrics.

        The push and pull of policy N overlap forward/backward N+1:

            Trainer:    forward/backward N
            Controller: await push N-1     (optimizer N must not mutate weights push N-1 stages)
            Trainer:    optimizer N -> policy N
            Controller: await pull N-1     (push N must not overwrite the TorchStore key mid-pull)
            Controller: start push N -> pull N -> release groups, in the background
                          |-> Trainer: stage policy N, TorchStore put_state_dict
                          |-> Generators: prefetch policy N into pinned CPU memory, then load it
                          |-> Group buffer: admit the next rollout groups
            Trainer:    forward/backward N+1 (does not mutate persistent model state)

        See `Trainer.push_model_state_dict` for why staging can overlap forward/backward, and
        `Generator.load_model_state_dict` for how a generator applies the load between steps.

        Impact on off-policiness: no sample is born stale, because the buffer admits new rollout
        groups only after the generators load the new policy.

        consumes: a TrainerStepBatch (training_batch_queue.get)
            waits for:    a TrainerStepBatch in the queue
            unblocked by: _batcher_loop training_batch_queue.put()
        """
        # The previous step's push and pull, each returning its wall time; None before the first.
        push_task: asyncio.Task[float] | None = None
        pull_task: asyncio.Task[float] | None = None
        step_timer = MetricsTimer()
        gc_timer = m.GCTimer("controller", per_generation=True)
        loop_lag = m.EventLoopLag("controller")
        step_start = time.perf_counter()
        # With `validation.overlap_training`: a pass is due, but waits for the running one to end.
        validation_requested = False
        for step in range(self.start_step + 1, num_training_steps + 1):
            # Push the previous step; the last step stays open for post-training validation.
            self.metrics_processor.commit()
            sl.set_step(step)  # propagate the step counter to the actors
            with (
                sl.log_trace_span("sync_log_step"),
                step_timer.record("timing/step/sync_log_step"),
            ):
                await self.trainer.all_ranks.sync_log_step.call(step)
                await self.generator_router.sync_log_step.call_one(step)
                # Not awaited: it would wait behind an in-flight add_group.
                self.batcher.sync_log_step.broadcast(step)
                await self._rollouter.sync_log_step(step)

            with sl.log_trace_span("train_step"):
                # The adaptive buffer sizes its demand from what is ready at this moment.
                await self._group_buffer.record_step_start(
                    trainer_policy_version=self._trainer_policy_version
                )
                # Waits for a TrainerStepBatch to be ready (or None on shutdown).
                with (
                    sl.log_trace_span("wait_for_training_batch"),
                    step_timer.record("timing/step/wait_for_training_batch"),
                ):
                    packed = await training_batch_queue.get()

                if packed is None:
                    logger.info("Batcher closed and drained; stopping training")
                    break

                # Policy age is computed HERE, at consumption time, against the live trainer version, so it is
                # faithful to what this step trains on -- not the version when the batch was packed.
                policy_age_panel = compute_policy_age_metrics(
                    trainer_policy_version=self._trainer_policy_version,
                    min_policy_versions=packed.min_policy_versions,
                    target_offpolicy_steps=(
                        self.config.async_loop.target_offpolicy_steps
                    ),
                    max_offpolicy_steps=self.config.async_loop.max_offpolicy_steps,
                )

                # TODO(async): can't stream microbatches (interleave pack->train) -- the loss is normalized by
                #   global counts over ALL microbatches, needed before any fwd/bwd. To
                #   support streaming, accumulate raw loss/token counts across microbatches and scale before optimizer.
                with (
                    sl.log_trace_span("forward_backward"),
                    step_timer.record("timing/step/forward_backward"),
                ):
                    # TODO: each rank's first read is never hidden (~0.5 GB/s per rank, 4 ranks over TCP).
                    # Sending the RDMAMicrobatches during the previous optim_step would hide it.
                    try:
                        # Send each DP rank only its column of the grid.
                        fwd_bwd_results = await asyncio.gather(
                            *(
                                trainer_at_dp_rank.forward_backward.call(
                                    [
                                        [row[dp_rank] for row in group]
                                        for group in packed.microbatch_groups
                                    ],
                                    packed.global_loss_token_counts,
                                    packed.global_routing_token_counts,
                                )
                                for dp_rank, trainer_at_dp_rank in enumerate(
                                    self.trainer.by_dp_rank
                                )
                            )
                        )
                        # Under PP, only trainer ranks holding the last stage
                        # compute the loss; the others return None.
                        fwd_bwd_metrics = next(
                            rank_metrics
                            for rank_metrics in fwd_bwd_results[0].values()
                            if rank_metrics is not None
                        )
                    finally:
                        # With RDMA, every rank has read its column (or the step failed).
                        # Not awaited: it would wait behind an in-flight add_group.
                        self.batcher.maybe_release_oldest_batch.broadcast()

                    if not math.isfinite(fwd_bwd_metrics["loss/mean"]):
                        logger.error("Loss is NaN/Inf; training diverged")
                        break

                # Await the trainer model state dict push before the optimizer mutates the weights.
                with (
                    sl.log_trace_span("blocking_trainer_push_model_state_dict"),
                    step_timer.record(
                        "timing/step/blocking_trainer_push_model_state_dict"
                    ),
                ):
                    push_s = await push_task if push_task is not None else 0.0

                with (
                    sl.log_trace_span("optim_step"),
                    step_timer.record("timing/step/optimizer"),
                ):
                    # Acknowledge only at the checkpoint transaction boundary:
                    # doing it earlier could lose prompts if the process exits
                    # before the corresponding optimizer step is durable. The
                    # batcher includes every consumed group ID here, including
                    # metric-only zero-std groups excluded from model inputs, plus
                    # the groups the buffer dropped as too old.
                    # Off the event loop: both wait for an in-flight sample read to finish.
                    await asyncio.to_thread(
                        self._rollouter.acknowledge_training_sample_ids,
                        [
                            *packed.group_ids,
                            *self._group_buffer.pop_dropped_group_ids(),
                        ],
                    )
                    controller_state = {
                        DATALOADER_STATE_KEY: await asyncio.to_thread(
                            self._rollouter.state_dict
                        )
                    }
                    optimizer_result = self._get_rank_0_value(
                        await self.trainer.all_ranks.optim_step.call(
                            controller_state=controller_state
                        )
                    )
                self._trainer_policy_version = optimizer_result.policy_version

                # Its own phase, so `timing/step/optimizer` excludes the save.
                with (
                    sl.log_trace_span("save_checkpoint"),
                    step_timer.record("timing/step/save_checkpoint"),
                ):
                    await self.trainer.all_ranks.save_checkpoint.call(
                        last_step=(step == num_training_steps)
                    )

                # Await the generator model state dict pull to finish before the trainer's next push.
                with (
                    sl.log_trace_span("blocking_generator_pull_model_state_dict"),
                    step_timer.record(
                        "timing/step/blocking_generator_pull_model_state_dict"
                    ),
                ):
                    pull_s = await pull_task if pull_task is not None else 0.0

                # Overlap this step's push -> pull -> group release with the next step's fwd/bwd.
                push_task = asyncio.create_task(self._push_model_state_dict())
                pull_task = asyncio.create_task(
                    self._pull_model_state_dict_and_release_groups(
                        optimizer_result.policy_version, after=push_task
                    )
                )

            # TODO(metrics): See if metrics are being computed at the right place. E.g. should we put all
            # rollout related metrics here, or move all of them to the rollouter.
            # `timing/step/total` runs from the previous step's metrics log to here, so totals cover the loop.
            step_end = time.perf_counter()
            time_metrics = [
                *step_timer.flush(),
                m.Metric(
                    "timing/step/total", m.Mean.from_list([step_end - step_start])
                ),
            ]
            step_start = step_end

            # Recorded after the flush: this log's time is the next step's `timing/step/metrics_log`.
            with (
                sl.log_trace_span("metrics_log"),
                step_timer.record("timing/step/metrics_log"),
            ):
                # In a thread: ~60 ms of /proc reads per step at 620 processes, and reading a
                # process's environ blocks while that process changes its memory mappings.
                host_memory_metrics = await asyncio.to_thread(
                    compute_host_memory_metrics
                )
                # to_thread: the loader's metrics wait for an in-flight sample read; don't block the loop on it.
                data_metrics = await asyncio.to_thread(
                    self._rollouter.training_data_metrics
                )
                session_kv_metrics = (
                    await self.generator_router.get_session_kv_metrics.call_one()
                )
                self.metrics_processor.log(
                    step=step,
                    is_validation=False,
                    metrics=[
                        *packed.metrics,
                        *[
                            m.Metric(key, m.NoReduce(value))
                            for key, value in fwd_bwd_metrics.items()
                        ],
                        *[
                            m.Metric(key, m.NoReduce(value))
                            for key, value in optimizer_result.metrics.items()
                        ],
                        *self._group_buffer.metrics(),
                        *data_metrics,
                        *session_kv_metrics,
                        *time_metrics,
                        *gc_timer.flush(),
                        *loop_lag.flush(),
                        *policy_age_panel,
                        # Background push/pull work time; the trainer's wait for it is timing/step/blocking_*.
                        m.Metric(
                            "timing/weight_sync/trainer_push_model_state_dict",
                            m.NoReduce(push_s),
                        ),
                        m.Metric(
                            "timing/weight_sync/generator_pull_model_state_dict",
                            m.NoReduce(pull_s),
                        ),
                        *compute_perf_ratio_metrics(
                            num_global_valid_tokens=int(
                                packed.global_loss_token_counts[0]
                            ),
                            num_global_tokens=int(
                                packed.global_routing_token_counts[0]
                            ),
                            time_metrics=time_metrics,
                        ),
                        *host_memory_metrics,
                    ],
                )

            validation = self.config.async_loop.validation
            is_validation_step = (
                validation.freq is not None
                and step % validation.freq == 0
                and step < num_training_steps
            )
            if validation.overlap_training:
                # Leave a pass that ends in the last step to run(), which logs it before the final pass.
                if step < num_training_steps:
                    self._log_finished_validation(step=step)
                if is_validation_step:
                    validation_requested = True
                    if self._validation_task is not None:
                        logger.info(
                            f"Validation step {step}: {self._validation_task.get_name()} is still "
                            "running; the next pass starts when it ends"
                        )
                # No pass starts at the last step: the final pass after training covers it.
                if (
                    validation_requested
                    and self._validation_task is None
                    and step < num_training_steps
                ):
                    # Wait only for this step's weight pull, so the pass starts on at least policy `step`.
                    await pull_task
                    self._start_validation(step=step)
                    validation_requested = False
            elif is_validation_step:
                # Pause training until the pass ends. Only this loop syncs weights, so every
                # validation rollout samples this step's policy; training rollouts keep generating.
                await pull_task
                await self._validate_and_log(step=step)
            await self._step_extra_validations(
                step=step, num_training_steps=num_training_steps, pull_task=pull_task
            )
        gc_timer.close()
        loop_lag.close()

        # Finish the last push and pull so generators hold the final weights for post-validation.
        if push_task is not None:
            await push_task
            await pull_task

    async def _push_model_state_dict(self) -> float:
        """Push the trainer's model state dict to TorchStore; returns its wall time in seconds."""
        with sl.log_trace_span("trainer_push_model_state_dict"):
            start = time.perf_counter()
            await self.trainer.all_ranks.push_model_state_dict.call()
            return time.perf_counter() - start

    async def _pull_model_state_dict_and_release_groups(
        self, version: int, *, after: asyncio.Task[float]
    ) -> float:
        """Once the push ``after`` lands, load policy ``version`` into every generator, then admit
        the next rollout groups; returns the pull's wall time in seconds."""
        await after
        with sl.log_trace_span("generator_pull_model_state_dict"):
            start = time.perf_counter()
            await self.generator_router.pull_model_state_dict.call_one(version)
            pull_s = time.perf_counter() - start
        # TODO(perf): pull_model_state_dict awaits ALL generators before we release any buffer slots,
        #   so a generator that finishes its pull early idles until the slowest one. Investigate
        #   per-generator release (router surfaces each pull's completion -> release that generator's
        #   share / resume it early); needs the born-fresh invariant to hold per-generator, not globally.

        # Born-fresh: admit the next groups only now that the generators are on `version`, so a new
        # rollout starts at the current version (keeps policy_age within the derived freshness bound).
        await self._group_buffer.release_active_groups(
            self.config.async_loop.num_prompts_per_train_step, reason="trained"
        )
        return pull_s
