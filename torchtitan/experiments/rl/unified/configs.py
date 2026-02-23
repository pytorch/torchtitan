# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Top-level RL training configuration.

This module defines:
- Leaf data configs: VLLMSamplingConfig, PolicyOptimizationConfig (plain dataclasses, no build())
- RLTrainer: top-level Configurable that composes a canonical JobConfig
  (as a sub-component for model/optimizer/parallelism) with RL-specific fields.

Config hierarchy::

    RLTrainer.Config
    ├── trainer: JobConfig              # canonical trainer sub-component
    │   ├── model_spec, training, parallelism, optimizer, lr_scheduler,
    │   │   checkpoint, compile, activation_checkpoint, comm, debug, ...
    ├── dump_folder: str                     # root output folder
    ├── batch_invariant_mode: bool           # policy trainer setting
    ├── policy_optimization: PolicyOptimizationConfig
    │   ├── beta, group_size, use_stable
    └── generator: Generator.Config        # rollout generator actor
        ├── vllm_engine: VLLMEngine.Config
        │   ├── dtype, gpu_memory_utilization, enforce_eager, seed
        │   ├── parallelism: ParallelismConfig
        │   └── sampling: VLLMSamplingConfig
        └── vllm_attention_backend: str
"""

from __future__ import annotations

from dataclasses import dataclass, field

# TODO: Replace with ``from torchtitan.trainer import Trainer``
# once the config branch lands.  For now we use the existing JobConfig
# as the trainer config type.
from torchtitan.config.job_config import JobConfig

# TODO: Replace with ``from torchtitan.config import Configurable``
# once the config branch lands.
from torchtitan.experiments.rl.unified.configurable import Configurable


@dataclass(kw_only=True, slots=True)
class VLLMSamplingConfig:
    """Sampling parameters passed to vLLM's SamplingParams."""

    temperature: float = 0.8
    """Sampling temperature. 0.0 = greedy, higher = more random."""

    top_p: float = 0.95
    """Nucleus sampling threshold."""

    max_tokens: int = 100
    """Maximum number of tokens to generate per completion."""


@dataclass(kw_only=True, slots=True)
class PolicyOptimizationConfig:
    """Hyperparameters for Group Relative Policy Optimization."""

    beta: float = 0.1
    """Temperature for GRPO exponential advantage weighting."""

    group_size: int = 8
    """Number of completions per prompt for group-relative ranking."""

    use_stable: bool = False
    """Use stable mean-centering GRPO instead of exponential weighting."""


class RLTrainer(Configurable):
    """Top-level RL training orchestrator.

    Initialises a canonical ``Trainer`` (for model construction, parallelism,
    optimizer, checkpoint, etc.) as a sub-component, then creates an
    Generator Monarch actor on a process mesh and runs the
    generate → train loop.

    Policy trainer settings (GRPO hyperparameters, batch invariance) live
    directly on this config because the RLTrainer *is* the policy trainer.

    Constructed via ``config.build()`` which calls ``RLTrainer(config=...)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        """Master config for RL training.

        ``trainer`` holds the canonical ``JobConfig`` that handles model
        construction, parallelism, optimizer, LR scheduler, checkpoint, etc.
        The remaining fields are RL-specific.
        """

        # -- Canonical trainer as a sub-component --

        trainer: JobConfig = field(default_factory=JobConfig)
        """Canonical TorchTitan trainer config.  Controls model_spec, training,
        parallelism, optimizer, lr_scheduler, checkpoint, compile,
        activation_checkpoint, comm, debug, and all other standard fields."""

        # -- Top-level RL settings --

        dump_folder: str = "outputs/rl"
        """Root output folder for RL artifacts (temp weights, logs, etc.)."""

        batch_invariant_mode: bool = True
        """Enable batch-invariant mode for deterministic NCCL collective
        operations and bitwise-reproducible forward/backward passes."""

        policy_optimization: PolicyOptimizationConfig = field(
            default_factory=PolicyOptimizationConfig
        )
        """Policy optimization hyperparameters (GRPO)."""

        # -- Generator actor config --
        # Lazy default factory breaks the circular import:
        #   configs.py  ↔  actors/generator.py

        generator: Generator.Config = field(  # type: ignore[name-defined]  # noqa: F821
            default_factory=lambda: _default_rl_generator_config(),
        )
        """Generator actor configuration (vLLM engine, sampling)."""

    def __init__(self, config: Config):
        self.config = config

        # TODO: Once the config branch lands and ``trainer`` becomes a
        # ``Trainer.Config`` (a Configurable), replace the line below with:
        #     self.trainer = config.trainer.build()
        # That will handle distributed init, model construction,
        # parallelisation, optimizer/LR-scheduler creation, checkpoint
        # loading, etc.


def _default_rl_generator_config():
    from torchtitan.experiments.rl.unified.actors.generator import Generator

    return Generator.Config()
