# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This source code is licensed under the BSD-style license found in LICENSE.

"""Synthetic accelerator metadata is checked without initializing a device."""

import pytest
import torch
import torch.distributed.checkpoint as dcp
from torchtitan.last_words.rng import RNG_STATE, RNGCheckpointState
from torchtitan.last_words.validation import validate_rng_cursor


def accelerator_payload():
    state = RNGCheckpointState(torch.device("cpu"), 0)
    state.capture()
    value = state.state_dict()
    value["rank_0"]["device_type"] = "cuda"
    return value


def test_preflight_rejects_missing_accelerator_stream(tmp_path):
    dcp.save({RNG_STATE: accelerator_payload()}, checkpoint_id=tmp_path / "cursor")
    with pytest.raises(ValueError, match="accelerator"):
        validate_rng_cursor(tmp_path / "cursor", dict(rng_schema=1, rng_shards=1), 1)


def test_load_rejects_missing_accelerator_before_replacing_complete_snapshot():
    state = RNGCheckpointState(torch.device("cuda"), 0)
    with pytest.raises(ValueError, match="accelerator"):
        state.load_state_dict(accelerator_payload())
    with pytest.raises(RuntimeError, match="must be captured"):
        state.state_dict()


def test_synthetic_accelerator_layout_can_be_inspected_on_cpu(tmp_path):
    value = accelerator_payload()
    # This is a serialization-layout test, not a CUDA/NPU RNG round-trip.
    value["rank_0"]["accelerator"] = torch.tensor([1, 2, 3, 4], dtype=torch.uint8)
    dcp.save({RNG_STATE: value}, checkpoint_id=tmp_path / "cursor")
    validate_rng_cursor(tmp_path / "cursor", dict(rng_schema=1, rng_shards=1), 1)


@pytest.mark.parametrize(
    "payload",
    [
        torch.empty(0, dtype=torch.uint8),
        torch.ones(4),
        torch.ones((2, 2), dtype=torch.uint8),
    ],
)
def test_load_rejects_invalid_accelerator_tensor(payload):
    value = accelerator_payload()
    value["rank_0"]["accelerator"] = payload
    state = RNGCheckpointState(torch.device("cuda"), 0)
    with pytest.raises(ValueError, match="accelerator"):
        state.load_state_dict(value)


def test_preflight_checks_saved_schema_value(tmp_path):
    value = accelerator_payload()
    value["rank_0"]["device_type"] = "cpu"
    value["rank_0"]["schema"] = 7
    dcp.save({RNG_STATE: value}, checkpoint_id=tmp_path / "cursor")
    with pytest.raises(ValueError, match="schema"):
        validate_rng_cursor(tmp_path / "cursor", dict(rng_schema=1, rng_shards=1), 1)
