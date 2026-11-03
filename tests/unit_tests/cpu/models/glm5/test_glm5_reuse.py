# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch
import torch_remat as remat

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.glm5 import build_model_config


def _cpu_attention(self, q, k, v, *, attention_metadata, scale=None, **kwargs):
    # FlexAttention has no CPU backward. Keep the real DSA BlockMask and use
    # differentiable PyTorch attention to exercise the MLA/MTP remat wiring.
    query = torch.arange(q.shape[0])[:, None]
    key = torch.arange(k.shape[0])[None, :]
    allowed = attention_metadata.mask_mod(0, 0, query, key)
    scores = torch.einsum("qhd,khd->hqk", q, k) * scale
    probabilities = scores.masked_fill(~allowed, float("-inf")).softmax(-1)
    return torch.einsum("hqk,khv->qhv", probabilities, v)


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

    with patch.object(FlexInnerAttention, "forward", _cpu_attention):
        expected = run(False)
        actual = run(True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
