# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Leaf data configs for RL training.

These are plain dataclasses (not ``Configurable``) since they are just data,
not buildable components.
"""

from dataclasses import dataclass


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

    use_stable_grpo: bool = False
    """Use stable mean-centering GRPO instead of exponential weighting."""
