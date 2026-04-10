# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch

from torchtitan.model_setup import materialize_model
from torchtitan.models.common.linear import Linear
from torchtitan.protocols import BaseModel


class MetaModel(BaseModel):
    """Minimal BaseModel that records whether init_weights was called."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        def update_from_config(self, **kwargs):
            pass

        def get_nparams_and_flops(self, model, seq_len):
            return (0, 0)

    def __init__(self):
        super().__init__()
        self.linear = Linear.Config().build(
            in_features=4,
            out_features=4,
        )
        self.init_weights_called = False
        self.init_buffer_device = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def init_weights(self, buffer_device=None, **kwargs):
        self.init_weights_called = True
        self.init_buffer_device = buffer_device
        torch.nn.init.ones_(self.linear.weight)


def _build_on_meta() -> MetaModel:
    with torch.device("meta"):
        return MetaModel()


class TestMaterializeModel:
    def test_moves_from_meta_to_cpu(self):
        model = _build_on_meta()
        assert next(model.parameters()).device.type == "meta"

        materialize_model(model, init_device="cpu", buffer_device=None)

        param = next(model.parameters())
        assert param.device.type == "cpu"
        assert param.requires_grad

    def test_calls_init_weights_with_buffer_device(self):
        model = _build_on_meta()
        buf_dev = torch.device("cpu")

        materialize_model(model, init_device="cpu", buffer_device=buf_dev)

        assert model.init_weights_called
        assert model.init_buffer_device == buf_dev

    def test_weights_are_initialized(self):
        model = _build_on_meta()

        materialize_model(model, init_device="cpu", buffer_device=None)

        # init_weights sets weight to all ones; verify it's not garbage
        assert torch.all(model.linear.weight == 1.0)

    def test_model_in_train_mode(self):
        model = _build_on_meta()
        model.eval()

        materialize_model(model, init_device="cpu", buffer_device=None)

        assert model.training

    def test_works_in_loop_for_pp_pattern(self):
        """Simulates the PP path where materialize_model is called per stage."""
        models = [_build_on_meta() for _ in range(3)]

        for m in models:
            materialize_model(m, init_device="cpu", buffer_device=None)

        for m in models:
            assert next(m.parameters()).device.type == "cpu"
            assert m.init_weights_called
            assert m.training
