# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Environment contract + concrete envs for the RL rollout loop.

The contract lives in ``envs.types`` (``MessageEnv``, ``EnvReset``,
``EnvStep``, ``EnvBuilder``, ``EnvDataset``, ``EnvExample``). Concrete
envs are subpackages (``envs.sum_digits``, ``envs.alphabet_sort``);
each ships an env class plus a builder and dataset.

The token-level adapter that owns parse/length/context termination
lives in ``envs.token_env``.
"""

from torchtitan.experiments.rl.envs.types import (
    EnvBuilder,
    EnvDataset,
    EnvExample,
    EnvReset,
    EnvStep,
    MessageEnv,
)

__all__ = [
    "EnvBuilder",
    "EnvDataset",
    "EnvExample",
    "EnvReset",
    "EnvStep",
    "MessageEnv",
]
