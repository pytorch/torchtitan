# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field

from torchtitan.experiments.rl.environment import TokenEnv
from torchtitan.experiments.rl.examples.search_r1.data import SearchR1Dataset
from torchtitan.experiments.rl.examples.search_r1.env import SearchR1Env
from torchtitan.experiments.rl.examples.search_r1.rubric import RewardExactMatch

from torchtitan.experiments.rl.rollout.rollouter import Rollouter

from torchtitan.experiments.rl.rubrics import Rubric


class SearchR1Rollouter(Rollouter):
    """Search-R1 rollouter: the QA datasets, the multi-turn search env, and the
    exact-match rubric wired together. All behavior is inherited from ``Rollouter``;
    this only supplies the default configs.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Rollouter.Config):
        train_dataset: SearchR1Dataset.Config = field(
            default_factory=lambda: SearchR1Dataset.Config(
                filename="train.parquet", seed=42
            )
        )
        validation_dataset: SearchR1Dataset.Config = field(
            default_factory=lambda: SearchR1Dataset.Config(
                # NQ split only; deterministic order (no shuffle) so every validation
                # pass draws the same held-out samples.
                filename="test.parquet",
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
                # A truncated rollout has no final answer -> no reward / learning signal.
                truncation_reward=0.0,
            )
        )
        message_env: SearchR1Env.Config = field(default_factory=SearchR1Env.Config)
        token_env: TokenEnv.Config = field(
            default_factory=lambda: TokenEnv.Config(
                max_context_length=3072, max_num_turns=4
            )
        )
