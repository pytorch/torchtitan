# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os

from dataclasses import dataclass, field

from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.search_r1.data import SearchR1Dataset
from torchtitan.experiments.rl.examples.search_r1.env import SearchR1Env
from torchtitan.experiments.rl.examples.search_r1.rubric import (
    RewardAnswerEM,
    RewardSearchR1,
)

from torchtitan.experiments.rl.rollout.rollouter import Rollouter

from torchtitan.experiments.rl.rubrics import Rubric


# Default Search-R1 NQ/HotpotQA parquet locations (prepared via Search-R1's
# data_process scripts). Override `train_dataset`/`validation_dataset` data_path
# in a config to point elsewhere. See README.md for data + retriever setup.
#
# The defaults are dev-server paths; set ``SEARCH_R1_TRAIN_PARQUET`` /
# ``SEARCH_R1_TEST_PARQUET`` to relocate the data without a config change. This
# is how the MAST launcher (``mast_rl/run.sh``) points the job at the parquet
# staged on the mounted Manifold bucket — overriding the deeply-nested
# ``rollouter.*.data_path`` fields through the base-typed ``rollouter`` config
# is not reliable via tyro CLI flags.
DEFAULT_TRAIN_PARQUET = os.environ.get(
    "SEARCH_R1_TRAIN_PARQUET",
    "/home/yichuan/Search-R1/data/nq_hotpotqa_train/train.parquet",
)
DEFAULT_TEST_PARQUET = os.environ.get(
    "SEARCH_R1_TEST_PARQUET",
    "/home/yichuan/Search-R1/data/nq_hotpotqa_train/test.parquet",
)


class SearchR1Rollouter(Rollouter):
    """The Search-R1 rollouter: NQ/HotpotQA train/val datasets, a multi-turn
    search env, and an EM + format rubric. Pure config — all behavior
    (`make_env_group`, `sample_*`, `score_group`) is inherited from `Rollouter`.

    The per-rollout turn budget lives in `SearchR1Env.max_assistant_turns`;
    `token_env.max_rollout_tokens` additionally bounds total prompt length.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: SearchR1Dataset.Config = field(
            default_factory=lambda: SearchR1Dataset.Config(
                data_path=DEFAULT_TRAIN_PARQUET, seed=42
            )
        )
        validation_dataset: SearchR1Dataset.Config = field(
            default_factory=lambda: SearchR1Dataset.Config(
                # Evaluate on the NQ test split only (in-distribution with the
                # NQ+HotpotQA train set), like slime's per-benchmark eval.
                data_path=DEFAULT_TEST_PARQUET,
                seed=99,
                data_source="nq",
                # File order (no shuffle) so the first num_validation_samples rows
                # match slime's nq_test test.parquet@[0:N] exactly -> EM directly
                # comparable to slime.
                shuffle=False,
            )
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[
                    # Search-R1 reward. Default = slime's pure-EM 0/1 (correct -> 1.0,
                    # else 0). To put search on the gradient (anti closed-book reward
                    # hacking), enable the graded levers, e.g.:
                    #   RewardSearchR1.Config(weight=1.0, structure_format_score=0.2,
                    #                         retrieval_score=0.1, final_format_score=0.1)
                    RewardSearchR1.Config(weight=1.0),
                    # Metric-only (weight 0): keeps the pure-EM number in the reward
                    # breakdown comparable across runs without affecting the gradient
                    # (redundant with the default 0/1 reward, useful once graded).
                    RewardAnswerEM.Config(weight=0.0),
                ],
                # No <answer> on a truncated rollout -> no reward, no learning signal.
                truncation_reward=0.0,
            )
        )
        message_env: SearchR1Env.Config = field(default_factory=SearchR1Env.Config)
        token_env: TokenEnv.Config = field(
            default_factory=lambda: TokenEnv.Config(max_rollout_tokens=3072)
        )
