# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

import pytest
import torch

from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.gpt_oss import (
    _make_gptoss_attn_config,
    gptoss_configs,
    model_registry,
)
from torchtitan.models.gpt_oss.config_registry import gpt_oss_debugmodel


@pytest.mark.parametrize("flavor", gptoss_configs)
def test_default_attention_supports_sink_gradients(flavor):
    config = model_registry(flavor, seq_len=16).model
    assert all(
        isinstance(layer.attention.inner_attention, FlexInnerAttention.Config)
        for layer in config.layers
    )


def test_default_training_config_supports_sink_gradients():
    config = gpt_oss_debugmodel(seq_len=16)
    assert all(
        isinstance(layer.attention.inner_attention, FlexInnerAttention.Config)
        for layer in config.model_spec.model.layers
    )


def _varlen_attention():
    attention = _make_gptoss_attn_config(
        dim=16,
        layer_id=0,
        n_heads=2,
        n_kv_heads=1,
        head_dim=8,
        attn_backend="varlen",
        rope=CosSinRoPE.Config(dim=8, max_context_length=16),
    ).build()
    attention.init_states()
    return attention


@pytest.mark.parametrize("training", [True, False])
def test_varlen_rejects_sink_qk_gradients(training):
    attention = _varlen_attention().train(training)
    # eval() does not disable autograd. Neither mode may silently lose gradients.
    with patch.object(attention.inner_attention, "forward") as kernel:
        with pytest.raises(RuntimeError, match="Use attn_backend='flex'"):
            attention(torch.randn(2, 16), attention_masks=None)
    kernel.assert_not_called()


def test_varlen_rejects_input_gradients_with_frozen_parameters():
    attention = _varlen_attention().requires_grad_(False)
    with pytest.raises(RuntimeError, match="LSE gradients"):
        attention(torch.randn(2, 16, requires_grad=True), attention_masks=None)


@pytest.mark.parametrize("grad_mode", [torch.no_grad, torch.inference_mode])
def test_varlen_remains_available_without_gradients(grad_mode):
    attention = _varlen_attention()
    with grad_mode(), patch.object(
        attention.inner_attention, "forward", return_value=torch.zeros(2, 2, 8)
    ) as kernel:
        output = attention(torch.randn(2, 16), attention_masks=None)
    kernel.assert_called_once()
    assert output.shape == (2, 16)
