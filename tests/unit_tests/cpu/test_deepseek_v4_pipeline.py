# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from torchtitan.config import ParallelismConfig
from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    _split_module,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.deepseek_v4 import pipeline
from torchtitan.models.deepseek_v4.mhc import HcHead
from torchtitan.models.deepseek_v4.model import DeepSeekV4Model


class TokenDependentBlock(nn.Module):
    """Exercise the stage's hidden-state and original-token inputs on CPU."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(4, 4)
        self.token_embedding = nn.Embedding(16, 4)

    def forward(self, hidden, input_ids_T, attention_masks, positions, **kwargs):
        assert input_ids_T.dtype == torch.int64
        assert input_ids_T.ndim == 1
        return (
            hidden
            + self.projection(hidden)
            + self.token_embedding(input_ids_T).unsqueeze(1)
        )


def make_model():
    torch.manual_seed(42)
    model = DeepSeekV4Model.__new__(DeepSeekV4Model)
    nn.Module.__init__(model)
    model.hc_mult = 2
    model.n_main_layers = 6
    model._skip_lm_head = False
    model.tok_embeddings = nn.Embedding(16, 4)
    model.layers = nn.ModuleDict({str(i): TokenDependentBlock() for i in range(6)})
    model.hc_head = HcHead(HcHead.Config(hc_mult=2, dim=4))
    with torch.no_grad():
        for parameter in model.hc_head.parameters():
            parameter.uniform_(-0.1, 0.1)
    model.norm = nn.LayerNorm(4)
    model.lm_head = nn.Linear(4, 16)
    model.mtp_layers = nn.ModuleList()
    return model


@pytest.mark.parametrize("num_stages", [2, 4])
@pytest.mark.parametrize("skip_lm_head", [False, True])
def test_stage_outputs_gradients_and_checkpoint_keys(num_stages, skip_lm_head):
    model = make_model()
    model._skip_lm_head = skip_lm_head
    module_fqns = _generate_llm_fqn_per_model_part(num_stages, 6)
    module_fqns[-1].insert(module_fqns[-1].index("norm"), "hc_head")
    stages = [_split_module(model, names) for names in module_fqns]
    tokens = torch.tensor([1, 3, 5, 7, 9, 11])
    expected = model(tokens)
    actual = tokens
    for i, stage in enumerate(stages):
        actual = stage(actual, input_ids_T=tokens)
        if i < num_stages - 1:
            assert actual.shape == (len(tokens), 2, 4)
            assert stage.hc_head is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.square().mean().backward()
    actual.square().mean().backward()

    expected_parameters = dict(model.named_parameters())
    stage_parameters = dict(
        item for stage in stages for item in stage.named_parameters()
    )
    assert stage_parameters.keys() == expected_parameters.keys()
    assert sum(len(dict(stage.named_parameters())) for stage in stages) == len(
        expected_parameters
    )
    for name, parameter in stage_parameters.items():
        expected_grad = expected_parameters[name].grad
        if expected_grad is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, expected_grad, rtol=0, atol=0)

    checkpoint = model.state_dict()
    for stage in stages:
        stage.load_state_dict({key: checkpoint[key] for key in stage.state_dict()})
    assert set().union(*(stage.state_dict().keys() for stage in stages)) == set(
        checkpoint
    )


def test_later_stage_requires_original_token_ids():
    stage = _split_module(make_model(), ["layers.2"])
    with pytest.raises(ValueError, match="input_ids_T"):
        stage(torch.randn(6, 2, 4))


def test_preprocess_keeps_token_ids_in_microbatch_kwargs(monkeypatch):
    tokens = torch.tensor([2, 4, 6])
    labels = torch.tensor([4, 6, 8])
    positions = torch.arange(3)
    monkeypatch.setattr(
        Decoder,
        "preprocess_inputs",
        lambda self, input_dict, **kwargs: (tokens, labels, {"positions": positions}),
    )
    inputs, targets, kwargs = make_model().preprocess_inputs({})
    assert inputs is tokens
    assert targets is labels
    assert kwargs["input_ids_T"] is tokens
    assert kwargs["positions"] is positions


def test_default_pipeline_split_keeps_hc_head_with_norm(monkeypatch):
    captured = {}

    def capture_pipeline(model, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "pipeline_llm", capture_pipeline)
    parallelism = ParallelismConfig(pipeline_parallel_degree=2)
    pipeline.pipeline_deepseek_v4(
        make_model(),
        parallel_dims=SimpleNamespace(pp=2),
        parallelism=parallelism,
        model_config=SimpleNamespace(layers=[None] * 6),
    )
    splits = captured["parallelism"].module_fqns_per_model_part
    assert splits[-1][-3:] == ["hc_head", "norm", "lm_head"]
    assert all("hc_head" not in names for names in splits[:-1])
    assert parallelism.module_fqns_per_model_part is None


@pytest.mark.parametrize("hc_stage", [None, 0, 1])
def test_explicit_pipeline_split_validates_hc_head(monkeypatch, hc_stage):
    splits = [["tok_embeddings", "layers.0"], ["layers.1", "norm", "lm_head"]]
    if hc_stage is not None:
        splits[hc_stage].append("hc_head")
    parallelism = ParallelismConfig(module_fqns_per_model_part=splits)
    monkeypatch.setattr(pipeline, "pipeline_llm", lambda model, **kwargs: kwargs)
    arguments = dict(
        parallel_dims=SimpleNamespace(pp=2),
        parallelism=parallelism,
        model_config=SimpleNamespace(layers=[None] * 2),
    )
    if hc_stage != 1:
        with pytest.raises(ValueError, match="hc_head and norm"):
            pipeline.pipeline_deepseek_v4(make_model(), **arguments)
    else:
        result = pipeline.pipeline_deepseek_v4(make_model(), **arguments)
        assert result["parallelism"] is parallelism
