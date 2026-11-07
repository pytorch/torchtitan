# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch_remat as remat

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.glm5 import build_model_config
from torchtitan.models.glm5.dsa import DsaMetadata


def test_mtp_attention_boundary_accepts_omitted_keyword_indices():
    config = build_model_config(
        "glm5_2_debugmodel_mtp", parallelism=ParallelismConfig(), seq_len=16
    )
    attention = config.mtp_layers[0].attention.build()
    attention._sharding_config = config.mtp_layers[0].attention.sharding_config
    positions = torch.arange(16)
    args = (
        torch.zeros(16, config.dim),
        DsaMetadata.from_positions(positions),
        positions,
    )
    actual_args, actual_kwargs = attention._redistribute_inputs(args, {})
    assert len(actual_args) == 3
    assert not actual_kwargs


@pytest.mark.parametrize("save", [[], ["*"], ["*.inner_attention"]])
def test_shared_mla_and_mtp_checkpoint_outputs_and_gradients(save):
    torch.manual_seed(42)
    config = build_model_config(
        "glm5_2_debugmodel_mtp", parallelism=ParallelismConfig(), seq_len=16
    )
    config.layers = config.layers[:3]
    config.index_sources = (0, 0, 2)
    model = config.build()
    model.init_states()
    model._skip_lm_head = True
    model.configure_remat_regions(save, [])
    tokens = torch.arange(16)
    positions = torch.arange(16)
    masks = model.get_attention_masks(positions)
    kwargs = dict(
        positions=positions,
        attention_masks=masks,
        mtp_input_valid_masks=(torch.ones(16, dtype=torch.bool),),
        mtp_attention_masks=(masks,),
    )

    def run(checkpoint):
        model.zero_grad(set_to_none=True)
        forward = remat.checkpoint()(model) if checkpoint else model
        outputs = forward((tokens, tokens), **kwargs)
        sum(output.square().mean() for output in outputs).backward()
        gradients = {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        assert gradients
        assert all(torch.isfinite(grad).all() for grad in gradients.values())
        return tuple(output.detach() for output in outputs), gradients

    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        expected = run(False)
        actual = run(True)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
