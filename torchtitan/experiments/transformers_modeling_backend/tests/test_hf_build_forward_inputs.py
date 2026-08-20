# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.experiments.transformers_modeling_backend.model import (
    HFTransformerModel,
)


class _CP:
    cp_enabled = True


def test_hf_builds_cp_mask_under_cp(monkeypatch):
    monkeypatch.setattr(
        HFTransformerModel, "get_attention_masks", lambda self, positions: "MASK"
    )
    m = object.__new__(HFTransformerModel)  # no heavy __init__
    input_dict, sharding = m._build_forward_inputs(
        {"input": 0, "labels": "labels", "positions": 1}, parallel_dims=_CP()
    )
    assert input_dict["attention_masks"] == "MASK"
    assert sharding is None  # HF inherits base None sharding


# The mask is now built in `_build_forward_inputs` for all cases (CP and
# non-CP); `parallel_dims` is not read, so one CP case covers both.
def test_hf_no_mask_when_get_attention_masks_returns_none(monkeypatch):
    monkeypatch.setattr(
        HFTransformerModel, "get_attention_masks", lambda self, positions: None
    )
    m = object.__new__(HFTransformerModel)
    input_dict, sharding = m._build_forward_inputs(
        {"input": 0, "labels": "labels", "positions": 1}, parallel_dims=_CP()
    )
    assert "attention_masks" not in input_dict
    assert sharding is None
