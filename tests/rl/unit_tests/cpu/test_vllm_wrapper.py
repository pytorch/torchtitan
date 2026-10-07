# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.kimi_k3 import build_model_config as build_kimi_k3_config
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.model.attention import VLLMInnerAttention, VLLMMLAInnerAttention
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
                layer.attention.inner_attention,
                VLLMMLAInnerAttention.Config,
            )


def test_vllm_mla_adapter_materializes_kv():
    q_THK = torch.randn(8, 4, 6)
    kv_THP = torch.randn(8, 4, 7)
    k_shared_TR = torch.randn(8, 2)
    expected_k_THK = torch.cat(
        (kv_THP[..., :4], k_shared_TR.unsqueeze(1).expand(-1, 4, -1)), dim=-1
    )
    expected_v_THV = kv_THP[..., 4:]
    out_THV = torch.randn(8, 4, 3)
    attention = VLLMMLAInnerAttention.__new__(VLLMMLAInnerAttention)
    torch.nn.Module.__init__(attention)

    with mock.patch.object(
        VLLMInnerAttention,
        "forward",
        autospec=True,
        return_value=out_THV,
    ) as vllm_forward:
        result = attention.forward(q_THK, kv_THP, k_shared_TR)

    assert result is out_THV
    args = vllm_forward.call_args.args
    assert args[0] is attention
    assert args[1] is q_THK
    torch.testing.assert_close(args[2], expected_k_THK)
    torch.testing.assert_close(args[3], expected_v_THV)
