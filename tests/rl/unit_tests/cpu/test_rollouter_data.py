# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from torchtitan.rl.examples.alphabet_sort.data import AlphabetSortDataset
from torchtitan.rl.rollout.data import IterableRolloutDataLoader, RLDataset
from torchtitan.rl.rollout.message_env import MessageEnv
from torchtitan.rl.rollout.rollouter import MessageEnvRolloutWorker, Rollouter
from torchtitan.rl.rollout.rubric import Rubric
from torchtitan.rl.rollout.token_env import TokenEnv


@dataclass(kw_only=True)
class _ValidationConfig:
    """3 prompts, repeated unless `repeat=False`, like the RL datasets. They repeat once, not
    forever, so a test that reads too far fails instead of hanging."""

    repeat: bool = True

    def build(self):
        prompts = (1, 2, 3)
        return iter(prompts * 2 if self.repeat else prompts)


def _rollouter(validation_dataset: _ValidationConfig) -> Rollouter:
    rollouter = Rollouter.__new__(Rollouter)
    rollouter._config = SimpleNamespace(validation_dataset=validation_dataset)
    return rollouter


@pytest.mark.parametrize(
    "steps, expected", [(2, [1, 2]), (-1, [1, 2, 3])], ids=["first_2", "all"]
)
def test_validation_reads_a_fresh_pass(steps: int, expected: list[int]) -> None:
    rollouter = _rollouter(_ValidationConfig())

    # Each pass builds the dataset again, so it starts over.
    assert rollouter.get_validation_samples(steps) == expected
    assert rollouter.get_validation_samples(steps) == expected
    # Past the end it raises instead of repeating prompts.
    with pytest.raises(ValueError, match="fewer than ValidationConfig.steps=5"):
        rollouter.get_validation_samples(5)


@pytest.mark.parametrize(
    "training_dataset, validation_dataset, match",
    [
        (
            RLDataset.Config(repeat=False),
            RLDataset.Config(),
            "training_dataloader.dataset.repeat=False",
        ),
        # Validation builds it with repeat=False, and an endless dataset needs a size for that.
        (RLDataset.Config(), AlphabetSortDataset.Config(), "needs num_samples"),
    ],
)
def test_a_dataset_that_cannot_make_its_pass_is_refused_at_config_time(
    training_dataset, validation_dataset, match
) -> None:
    with pytest.raises(ValueError, match=match):
        Rollouter.Config(
            training_dataloader=IterableRolloutDataLoader.Config(
                dataset=training_dataset
            ),
            validation_dataset=validation_dataset,
            worker=MessageEnvRolloutWorker.Config(
                rubric=Rubric.Config(reward_fns=[]),
                message_env=MessageEnv.Config(),
                token_env=TokenEnv.Config(max_rollout_tokens=1024),
            ),
        )
