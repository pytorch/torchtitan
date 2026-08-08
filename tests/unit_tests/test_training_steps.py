# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest

from torchtitan.trainer import _resolve_training_steps


def test_resolve_explicit_training_steps():
    assert (
        _resolve_training_steps(
            15,
            dataset_num_tokens=None,
            tokens_per_step=65_536,
        )
        == 15
    )


def test_resolve_one_dataset_pass():
    assert (
        _resolve_training_steps(
            -1,
            dataset_num_tokens=5_343_229_233_857,
            tokens_per_step=65_536,
        )
        == 81_531_207
    )


@pytest.mark.parametrize("configured_steps", [0, -2])
def test_reject_invalid_training_steps(configured_steps):
    with pytest.raises(ValueError, match="positive or -1"):
        _resolve_training_steps(
            configured_steps,
            dataset_num_tokens=65_536,
            tokens_per_step=65_536,
        )


def test_auto_steps_requires_dataset_token_count():
    with pytest.raises(ValueError, match="exposes its total number of tokens"):
        _resolve_training_steps(
            -1,
            dataset_num_tokens=None,
            tokens_per_step=65_536,
        )


def test_auto_steps_requires_one_full_step():
    with pytest.raises(ValueError, match="required for one training step"):
        _resolve_training_steps(
            -1,
            dataset_num_tokens=65_535,
            tokens_per_step=65_536,
        )
