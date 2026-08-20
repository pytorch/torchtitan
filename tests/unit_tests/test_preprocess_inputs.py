# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.protocols.model import BaseModel


class _DummyModel(BaseModel):
    """No self.config — mimics HFTransformerModel/FluxModel."""

    def __init__(self):
        super().__init__()

    def init_weights(self, **kwargs):
        pass


class _FakeParallelDims:
    cp_enabled = False


class _FakeParallelism:
    spmd_backend = "none"
    context_parallel_load_balancer = None
    context_parallel_ptrr_mask_key = None


def test_build_forward_inputs_splits_and_returns_none_sharding():
    model = _DummyModel()
    input_dict = {"input": torch.zeros(2, 4), "positions": torch.arange(4).repeat(2, 1)}
    labels = torch.ones(2, 4)
    inputs, out_labels, extra, sharding = model._build_forward_inputs(
        input_dict, labels, parallel_dims=_FakeParallelDims()
    )
    assert torch.equal(inputs, input_dict["input"])
    assert torch.equal(out_labels, labels)
    assert set(extra) == {"positions"}
    assert "attention_masks" not in extra
    assert sharding is None


def test_preprocess_inputs_returns_local_ntokens_cp_disabled():
    model = _DummyModel()
    input_dict = {"input": torch.zeros(2, 4)}
    labels = torch.ones(2, 4)
    inputs, out_labels, extra, ntok = model.preprocess_inputs(
        input_dict,
        labels,
        parallel_dims=_FakeParallelDims(),
        device=torch.device("cpu"),
        parallelism=_FakeParallelism(),
    )
    assert torch.equal(inputs, input_dict["input"])
    assert ntok == labels.numel()  # 8
    assert extra == {}


def test_local_ntokens_counted_after_cp_shard(monkeypatch):
    import torchtitan.distributed.context_parallel.api as cp_api

    class _CPDims:
        cp_enabled = True

        def get_mesh(self, name):
            return None

    def _fake_cp(inputs, labels, extra, mesh, device, lb, ptrr, *, input_sharding):
        return inputs, labels[:, :2], extra

    monkeypatch.setattr(cp_api, "prepare_context_parallel_input", _fake_cp)

    model = _DummyModel()
    input_dict = {"input": torch.zeros(2, 4)}
    labels = torch.ones(2, 4)
    _, _, _, ntok = model.preprocess_inputs(
        input_dict,
        labels,
        parallel_dims=_CPDims(),
        device=torch.device("cpu"),
        parallelism=_FakeParallelism(),
    )
    assert ntok == 4
