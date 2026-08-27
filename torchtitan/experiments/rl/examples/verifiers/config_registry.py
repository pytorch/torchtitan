# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""DAPO math recipes using Verifiers for rollout orchestration."""

from __future__ import annotations

from dataclasses import replace

from torchtitan.experiments.rl.controller import Controller
from torchtitan.experiments.rl.examples.dapo_math.config_registry import (
    rl_dapo_qwen3_4b_math_32k,
    rl_dapo_qwen3_4b_math_8k,
)
from torchtitan.experiments.rl.examples.verifiers.rollouter import (
    VerifiersMathRollouter,
)


def _with_verifiers(
    config: Controller.Config,
    *,
    max_model_len: int,
    dump_folder: str,
) -> Controller.Config:
    return replace(
        config,
        dump_folder=dump_folder,
        rollouter=VerifiersMathRollouter.Config(max_model_len=max_model_len),
    )


def rl_dapo_qwen3_4b_verifiers_8k() -> Controller.Config:
    """Run the DAPO 8K recipe with Verifiers managing math episodes."""
    return _with_verifiers(
        rl_dapo_qwen3_4b_math_8k(),
        max_model_len=10240,
        dump_folder="outputs/rl/qwen3_4b_verifiers_8k",
    )


def rl_dapo_qwen3_4b_verifiers_32k() -> Controller.Config:
    """Run the DAPO 32K recipe with Verifiers managing math episodes."""
    return _with_verifiers(
        rl_dapo_qwen3_4b_math_32k(),
        max_model_len=34816,
        dump_folder="outputs/rl/qwen3_4b_verifiers_32k",
    )
