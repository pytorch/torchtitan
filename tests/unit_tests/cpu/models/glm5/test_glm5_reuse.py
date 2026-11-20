# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import (
    ActivationRematConfig,
    DefaultActivationRematConfig,
)
from torchtitan.models.deepseek_v3.mtp import roll_mtp_sequence
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


@pytest.mark.parametrize("save", ["default", [], ["*"], ["*.inner_attention"]])
@pytest.mark.parametrize("mtp_only", [False, True])
def test_shared_mla_and_mtp_checkpoint_outputs_and_gradients(save, mtp_only):
    torch.manual_seed(42)
    config = build_model_config(
        "glm5_2_debugmodel_mtp", parallelism=ParallelismConfig(), seq_len=16
    )
    config.layers = config.layers[:4]
    config.index_sources = (0, 1, 2, 2)
    model = config.build()
    model.init_states()
    model._skip_lm_head = True
    tokens = torch.arange(16)
    positions = torch.cat((torch.arange(12), torch.arange(4)))
    padding = torch.zeros(16, dtype=torch.bool)
    padding[11] = True
    shifted, valid = roll_mtp_sequence(
        tokens,
        shift=1,
        positions=positions,
        padding_mask=padding,
        fill_value=0,
        return_valid_mask=True,
    )
    masks = model.get_attention_masks(positions, padding)
    kwargs = dict(
        positions=positions,
        attention_masks=masks,
        padding_mask=padding,
        mtp_input_valid_masks=(valid,),
        mtp_attention_masks=(model.get_attention_masks(positions, ~valid | padding),),
    )

    def run(checkpoint):
        model.zero_grad(set_to_none=True)
        if checkpoint:
            # Policies own one checkpoint/name scope per decoder block.
            # One checkpoint around the whole model aliases routing regions.
            policy = (
                DefaultActivationRematConfig()
                if save == "default"
                else ActivationRematConfig(save_regions=save)
            )
            policy.apply(model)
        outputs = model((tokens, shifted), **kwargs)
        objectives = outputs[1:] if mtp_only else outputs
        sum(output.square().mean() for output in objectives).backward()
        gradients = {
            name: parameter.grad.clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        assert gradients
        assert all(torch.isfinite(grad).all() for grad in gradients.values())
        if mtp_only:
            # The auxiliary branch must reach the main decoder without clones.
            assert any(
                name.startswith("layers.0.") and grad.abs().sum() > 0
                for name, grad in gradients.items()
            )
        return tuple(output.detach() for output in outputs), gradients

    deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        expected = run(False)
        actual = run(True)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
