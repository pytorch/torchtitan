# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.local_compile import LocalCompileConfig
from torchtitan.models.qwen3 import build_model_config


class _ValueAttention(torch.nn.Module):
    """CPU stand-in for the attention kernel: each query head returns its KV head's values."""

    def forward(self, q_THK, k_THK, v_THV, **kwargs):
        return v_THV.repeat_interleave(q_THK.shape[1] // v_THV.shape[1], dim=1)


@pytest.mark.parametrize("flavor", ["debugmodel", "debugmodel_moe"])
def test_carried_residual_matches_summed_blocks(flavor: str) -> None:
    """``Qwen3Model.forward`` hands each block's residual to the next block unsummed.
    Its output must equal running every block on the summed stream.

    Example, 2 layers:
        carried: out, r = layers[0](h); out, r = layers[1](out, residual=r); norm(r + out)
        summed:  out, r = layers[0](h); out, r = layers[1](r + out);        norm(r + out)
    """
    LocalCompileConfig().apply()
    config = build_model_config(flavor, parallelism=ParallelismConfig())
    torch.manual_seed(0)
    model = config.build()
    model.init_states(buffer_device=torch.device("cpu"))
    for layer in model.layers.values():
        layer.attention.inner_attention = _ValueAttention()
    num_tokens = 16
    tokens = torch.randint(0, config.vocab_size, (num_tokens,))
    positions = torch.arange(num_tokens)
    aux_loss_denominators = torch.tensor([float(num_tokens)])

    carried = model(tokens, positions, aux_loss_denominators=aux_loss_denominators)

    h = model.tok_embeddings(tokens)
    for layer in model.layers.values():
        out, residual = layer(
            h, None, positions, aux_loss_denominator=aux_loss_denominators[0]
        )
        h = residual + out
    summed = model.lm_head(model.norm(h))

    torch.testing.assert_close(carried, summed, rtol=0, atol=0)
