# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.pipeline_parallel import (
    _split_module,
    get_module_fqns_per_model_part,
)
from torchtitan.models.common.aux_loss import AuxLoss
from torchtitan_recipes.tests.models import deepseek_v4 as config_registry


@pytest.fixture(autouse=True)
def _isolate_aux_loss_state():
    with (
        patch.dict(AuxLoss._group_counts, clear=True),
        patch.dict(AuxLoss.group_acc, clear=True),
    ):
        yield


@pytest.mark.parametrize("num_stages", [2, 4])
def test_stage_outputs_gradients_and_checkpoint_keys(num_stages):
    """Chaining the PP stages of the debug model reproduces the unsplit model.

    With four stages, one stage runs hash-routed layer 1 without
    tok_embeddings and the last stage owns only hc_head, norm and lm_head.
    """
    torch.manual_seed(0)
    num_tokens = 128
    model_config = config_registry.deepseek_v4_debugmodel(seq_len=num_tokens).model
    with torch.device("meta"):
        model = model_config.build().double()
    model.to_empty(device="cpu")
    model.init_states(buffer_device=torch.device("cpu"))

    module_fqns = get_module_fqns_per_model_part(
        model,
        first_stage_module_fqns=model.pipeline_first_stage_module_fqns,
        last_stage_module_fqns=model.pipeline_last_stage_module_fqns,
        parallelism_context=SimpleNamespace(pp=num_stages),
        parallelism=ParallelismConfig(
            pipeline_parallel_degree=num_stages,
            pipeline_parallel_schedule="1F1B",
        ),
        model_config=model_config,
    )
    stages = [_split_module(model, names) for names in module_fqns]

    tokens = torch.randint(0, model_config.vocab_size, (num_tokens,))
    positions = torch.arange(num_tokens)
    attention_metadata = model._get_attention_metadata(positions)
    aux_loss_denominators = torch.tensor([num_tokens], dtype=torch.float64)
    expected = model(tokens, positions, attention_metadata, None, aux_loss_denominators)
    # Like the PP schedule: stage 0 gets tokens, later stages get the previous
    # stage's output, and every stage gets the token IDs as a kwarg.
    actual = tokens
    for stage in stages:
        actual = stage(
            actual,
            positions,
            attention_metadata,
            None,
            aux_loss_denominators,
            input_ids_T=tokens,
        )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    expected.square().mean().backward()
    actual.square().mean().backward()
    expected_parameters = dict(model.named_parameters())
    stage_parameters = dict(
        item for stage in stages for item in stage.named_parameters()
    )
    assert stage_parameters.keys() == expected_parameters.keys()
    for name, parameter in stage_parameters.items():
        expected_grad = expected_parameters[name].grad
        if expected_grad is None:
            assert parameter.grad is None, name
        else:
            torch.testing.assert_close(
                parameter.grad, expected_grad, rtol=0, atol=0, msg=name
            )

    assert set().union(*(stage.state_dict().keys() for stage in stages)) == set(
        model.state_dict()
    )
