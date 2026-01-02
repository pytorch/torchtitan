# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Extended JobConfig for RL/Inference workloads.

This module extends TorchTitan's base JobConfig with inference-specific
configurations needed for vLLM integration.
"""

from dataclasses import dataclass, field

from torchtitan.config.job_config import JobConfig as BaseJobConfig, Parallelism


@dataclass
class Sampling:
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
class Inference:
    """
    Inference configuration for vLLM engine.

    This dataclass contains essential vLLM-specific settings for inference.
    """

    dtype: str = "bfloat16"
    """Data type for model weights (auto, float16, bfloat16, float32)"""

    gpu_memory_utilization: float = 0.5
    """Fraction of GPU memory to use for Inference engine (0.0 to 1.0)"""

    distributed_executor_backend: str = "external_launcher"
    """
    Backend for distributed execution.
    'external_launcher' means vLLM does not spawn processes (use torchrun/external launcher)
    """

    seed: int = 42
    """Random seed for sampling"""

    enforce_eager: bool = True
    """Whether to enforce eager execution (disable CUDA graphs)"""

    parallelism: Parallelism = field(default_factory=Parallelism)
    """Parallelism configuration for inference"""

    sampling: Sampling = field(default_factory=Sampling)
    """Sampling configuration for inference"""


@dataclass
class JobConfig(BaseJobConfig):
    """
    Extended JobConfig with inference support.

    This extends TorchTitan's base JobConfig by adding `inference` and `sampling` fields
    for vLLM-specific inference and generation configurations.
    """

    inference: Inference = field(default_factory=Inference)
    """Inference configuration for vLLM engine"""
