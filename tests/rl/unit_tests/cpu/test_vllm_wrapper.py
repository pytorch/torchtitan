# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.linear import Linear
from torchtitan.models.kimi_k3 import build_model_config as build_kimi_k3_config
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.model.attention import VLLMInnerAttention, VLLMMLAAttention
from torchtitan.rl.model.vllm_wrapper import _replace_vllm_layer_configs


def test_vllm_replacement_preserves_attention_sharding() -> None:
    """Pass the trainer attention's sharding config through unchanged."""
    model_config = build_model_config("debugmodel", attn_backend="flex")
    model_config.set_sharding_(
        ParallelismConfig(tensor_parallel_degree=2, enable_sequence_parallel=True)
    )
    model_config.layers = [
        layer for layer in model_config.layers if layer.attention is not None
    ]

    vllm_config = _replace_vllm_layer_configs(model_config)

    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        assert model_layer.attention is not None
        assert vllm_layer.attention is not None
        model_sharding = model_layer.attention.inner_attention.sharding_config
        vllm_sharding = vllm_layer.attention.inner_attention.sharding_config
        assert model_sharding is not None
        assert vllm_sharding is not None
        assert vllm_sharding.in_src_shardings is model_sharding.in_src_shardings
        assert vllm_sharding.in_dst_shardings is model_sharding.in_dst_shardings
        assert vllm_sharding.out_src_shardings is model_sharding.out_src_shardings
        assert vllm_sharding.out_dst_shardings is model_sharding.out_dst_shardings
        assert vllm_sharding.local_spmd is model_sharding.local_spmd
        for name, layout in model_sharding.state_shardings.items():
            assert vllm_sharding.state_shardings[name] is layout


def test_vllm_replaces_mla_with_compact_input_adapter():
    model_config = build_kimi_k3_config(
        "debugmodel", attn_backend="varlen", seq_len=128
    )

    vllm_config = _replace_vllm_layer_configs(model_config)

    for layer in vllm_config.layers:
        if layer.attention is not None:
            assert isinstance(
                layer.attention.mla_attention,
                VLLMMLAAttention.Config,
            )
            assert isinstance(
                layer.attention.mla_attention.inner_attention,
                VLLMInnerAttention.Config,
            )


def test_vllm_mla_adapter_builds_wkv_b_from_config():
    config = VLLMMLAAttention.Config(
        wkv_b=Linear.Config(in_features=5, out_features=28),
        packed_kv_head_dim=7,
        inner_attention=VLLMInnerAttention.Config(
            attention_metadata_key=VLLMInnerAttention,
            hidden_size=24,
            num_heads=4,
            num_kv_heads=4,
            head_dim=6,
            value_head_dim=3,
        ),
    )

    def initialize_module(attention, config):
        torch.nn.Module.__init__(attention)
        attention.attention_metadata_key = config.attention_metadata_key

    with mock.patch.object(
        VLLMInnerAttention,
        "__init__",
        autospec=True,
        side_effect=initialize_module,
    ):
        attention = VLLMMLAAttention(config)

    assert attention.wkv_b.weight.shape == (28, 5)
    assert attention.packed_kv_head_dim == 7


def test_vllm_mla_adapter_materializes_kv():
    q_THK = torch.randn(8, 4, 6)
    kv_c_normed_TL = torch.randn(8, 5)
    kv_THP = torch.randn(8, 4, 7)
    k_shared_TR = torch.randn(8, 2)
    wkv_b = mock.Mock(return_value=kv_THP.flatten(-2))
    expected_k_THK = torch.cat(
        (kv_THP[..., :4], k_shared_TR.unsqueeze(1).expand(-1, 4, -1)), dim=-1
    )
    expected_v_THV = kv_THP[..., 4:]
    out_THV = torch.randn(8, 4, 3)
    attention = VLLMMLAAttention.__new__(VLLMMLAAttention)
    torch.nn.Module.__init__(attention)
    object.__setattr__(attention, "wkv_b", wkv_b)
    attention.packed_kv_head_dim = 7
    qkv_attention = VLLMInnerAttention.__new__(VLLMInnerAttention)
    torch.nn.Module.__init__(qkv_attention)
    attention.inner_attention = qkv_attention

    with mock.patch.object(
        VLLMInnerAttention,
        "forward",
        autospec=True,
        return_value=out_THV,
    ) as vllm_forward:
        result = attention.forward(
            q_THK,
            kv_c_normed_TL,
            k_shared_TR,
        )

    assert result is out_THV
    wkv_b.assert_called_once_with(kv_c_normed_TL)
    args = vllm_forward.call_args.args
    assert args[0] is qkv_attention
    assert args[1] is q_THK
    torch.testing.assert_close(args[2], expected_k_THK)
    torch.testing.assert_close(args[3], expected_v_THV)
