# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Frozen CPU random streams and independent DCP shard identities."""

import copy
import random

import numpy as np
import pytest
import torch
import torch.distributed.checkpoint as dcp
from torchtitan.last_words.rng import RNG_STATE, RNGCheckpointState
from torchtitan.last_words.validation import validate_rng_cursor


def sample():
    return (
        random.random(),
        random.gauss(0, 1),
        np.random.standard_normal(3),
        torch.rand(3),
    )


def assert_sample_equal(left, right):
    assert left[:2] == right[:2]
    assert np.array_equal(left[2], right[2])
    assert torch.equal(left[3], right[3])


@pytest.fixture(autouse=True)
def preserve_global_streams():
    state = RNGCheckpointState(torch.device("cpu"), 0)
    state.capture()
    yield
    state.restore()


def test_complete_boundary_restores_python_numpy_and_torch():
    random.seed(11)
    np.random.seed(22)
    torch.manual_seed(33)
    random.gauss(0, 1)
    np.random.standard_normal()
    state = RNGCheckpointState(torch.device("cpu"), 0)
    state.capture()
    expected = sample()
    for _ in range(7):
        sample()
    state.restore()
    assert_sample_equal(sample(), expected)


def test_export_is_frozen_and_does_not_read_mutating_global_rng():
    state = RNGCheckpointState(torch.device("cpu"), 0)
    state.capture()
    exported = state.state_dict()
    expected = sample()
    other = RNGCheckpointState(torch.device("cpu"), 0)
    other.load_state_dict(state.state_dict())
    # Mutating an exported tensor cannot corrupt the owned complete snapshot.
    exported["rank_0"]["cpu"].zero_()
    other.restore()
    assert_sample_equal(sample(), expected)
    state.restore()
    assert_sample_equal(sample(), expected)


def test_real_dcp_keeps_distinct_streams_for_each_shard(tmp_path):
    states = []
    expected = []
    for rank in range(2):
        random.seed(100 + rank)
        np.random.seed(200 + rank)
        torch.manual_seed(300 + rank)
        state = RNGCheckpointState(torch.device("cpu"), rank)
        state.capture()
        states.append(state)
        expected.append(sample())
    payload = {
        RNG_STATE: {
            key: value for state in states for key, value in state.state_dict().items()
        }
    }
    dcp.save(payload, checkpoint_id=tmp_path / "cursor")
    marker = dict(rng_schema=1, rng_shards=2)
    validate_rng_cursor(tmp_path / "cursor", marker, 2)
    for rank in range(2):
        state = RNGCheckpointState(torch.device("cpu"), rank)
        state.capture()
        dcp.load({RNG_STATE: state}, checkpoint_id=tmp_path / "cursor", no_dist=True)
        state.restore()
        assert_sample_equal(sample(), expected[rank])


def test_missing_shard_cannot_qualify_as_a_complete_cursor(tmp_path):
    state = RNGCheckpointState(torch.device("cpu"), 0)
    state.capture()
    dcp.save({RNG_STATE: state}, checkpoint_id=tmp_path / "cursor")
    with pytest.raises(ValueError, match="shard 1"):
        validate_rng_cursor(tmp_path / "cursor", dict(rng_schema=1, rng_shards=2), 2)


@pytest.mark.parametrize("field,value", [("schema", 9), ("device_type", "cuda")])
def test_incompatible_rng_state_is_rejected(field, value):
    state = RNGCheckpointState(torch.device("cpu"), 0)
    state.capture()
    checkpoint = copy.deepcopy(state.state_dict())
    checkpoint["rank_0"][field] = value
    with pytest.raises(ValueError, match="schema or device type"):
        state.load_state_dict(checkpoint)
