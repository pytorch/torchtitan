# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Extended JobConfig for RL/Generation workloads.

This module extends TorchTitan's base JobConfig with generation-specific
configurations needed for vLLM integration.
"""

from dataclasses import dataclass, field

from torchtitan.config.job_config import JobConfig as BaseJobConfig, Parallelism


@dataclass
class VLLMSamplingParams:
    """
    Sampling configuration for vLLM generation.

    This dataclass contains sampling parameters used during text generation.
    These map directly to vLLM's SamplingParams.
    """

    temperature: float = 0.8
    """
    Temperature for sampling. Controls randomness in generation.
    - 0.0: Deterministic (greedy decoding)
    - Higher values: More random outputs
    """

    top_p: float = 0.95
    """
    Top-p (nucleus) sampling threshold.
    Only tokens with cumulative probability up to top_p are considered.
    """

    max_tokens: int = 100
    """Maximum number of tokens to generate"""


@dataclass
class Generation:
    """
    Generation configuration for vLLM engine.

    This dataclass contains essential vLLM-specific settings for generation.
    """

    dtype: str = "bfloat16"
    """Data type for model weights (auto, float16, bfloat16, float32)"""

    gpu_memory_utilization: float = 0.5
    """Fraction of GPU memory to use for generation engine (0.0 to 1.0)"""

    enforce_eager: bool = True
    """Whether to enforce eager execution (disable CUDA graphs)"""

    distributed_executor_backend: str = "mp"
    """Distributed executor backend for vLLM (mp, ray, or external)"""

    seed: int = 42
    """Random seed for reproducible generation"""

    parallelism: Parallelism = field(default_factory=Parallelism)
    """Parallelism configuration for generation"""

    sampling: VLLMSamplingParams = field(default_factory=VLLMSamplingParams)
    """Sampling configuration for generation"""


@dataclass
class PolicyOptimization:
    """Policy optimization configuration for GRPO training."""

    grpo_beta: int = 0.1
    """Beta parameter for GRPO (Group Relative Policy Optimization) algorithm"""

    use_stable_grpo: bool = False
    """Whether to use stable version of GRPO algorithm"""

    grpo_group_size: int = 8
    """Number of samples in each GRPO group for policy optimization"""

    vllm_batch_invariant: bool = True
    """Enable vLLM batch invariant mode for deterministic backward pass"""

    vllm_attention_backend: str = "FLASH_ATTN"
    """vLLM attention backend to use (e.g., FLASH_ATTN, XFORMERS)"""


@dataclass
class JobConfig(BaseJobConfig):
    """
    Extended JobConfig with generation support.

    This extends TorchTitan's base JobConfig by adding `generation` field
    for vLLM-specific generation configurations.
    """

    generation: Generation = field(default_factory=Generation)
    """Generation configuration for vLLM engine"""
    policy_optimization: PolicyOptimization = field(default_factory=PolicyOptimization)
