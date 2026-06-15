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
from torchtitan.experiments.rl.examples.search_r1.rubric import RewardExactMatch

from torchtitan.experiments.rl.rollout.rollouter import Rollouter

from torchtitan.experiments.rl.rubrics import Rubric


# QA parquet locations. Set SEARCH_R1_TRAIN_PARQUET / SEARCH_R1_TEST_PARQUET to
# point at the data without editing the config; see README.md for how to prepare it.
DEFAULT_TRAIN_PARQUET = os.environ.get(
    "SEARCH_R1_TRAIN_PARQUET", "data/nq_hotpotqa_train/train.parquet"
)
DEFAULT_TEST_PARQUET = os.environ.get(
    "SEARCH_R1_TEST_PARQUET", "data/nq_hotpotqa_train/test.parquet"
)


class SearchR1Rollouter(Rollouter):
    """Search-R1 rollouter: the QA datasets, the multi-turn search env, and the
    EM/format rubric wired together. All behavior is inherited from ``Rollouter``;
    this only supplies the default configs.
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
                # Evaluate on the NQ split only; fixed file order for a stable eval set.
                data_path=DEFAULT_TEST_PARQUET,
                seed=99,
                data_source="nq",
                shuffle=False,
            )
        )
        rubric: Rubric.Config = field(
            default_factory=lambda: Rubric.Config(
                reward_fns=[
                    # Default = pure-EM 0/1. Put search on the gradient (anti
                    # closed-book reward hacking) via the levers, e.g.
                    #   RewardExactMatch.Config(weight=1.0, no_search_penalty=0.2,
                    #                           retrieval_score=0.1)
                    RewardExactMatch.Config(weight=1.0),
                ],
                # No <answer> on a truncated rollout -> no reward, no learning signal.
                truncation_reward=0.0,
            )
        )
        message_env: SearchR1Env.Config = field(default_factory=SearchR1Env.Config)
        token_env: TokenEnv.Config = field(
            default_factory=lambda: TokenEnv.Config(
                max_rollout_tokens=3072, max_num_turns=4
            )
        )
