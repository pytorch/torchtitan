# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import (
    LinearAttention,
    SlidingWindowVarlenInnerAttention,
)
from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.deepseek_v3 import build_model_config as build_dsv3_config
from torchtitan.models.kimi_k3 import build_model_config as build_kimi_k3_config
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.generator.parallelism import InferenceParallelismConfig
from torchtitan.rl.generator.vllm import gdn, kda, registry
from torchtitan.rl.generator.vllm.registry import model_config_to_hf_config_dict

from transformers import GenerationConfig, PretrainedConfig
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
)


def test_hf_config_adds_no_stop_token():
    # vLLM derives the generation config from the HF config when the checkpoint
    # has no generation_config.json; any eos_token_id there stops every request.
    hf_config = PretrainedConfig(
        **model_config_to_hf_config_dict(
            build_model_config(
                "0.8B",
                seq_len=256,
                attn_backend="varlen",
                parallelism=ParallelismConfig(),
            )
        )
    )
    assert GenerationConfig.from_model_config(hf_config).eos_token_id is None


def test_hf_config_uses_full_attention_after_sliding_attention(monkeypatch):
    from torchtitan.models.llama3 import build_model_config as build_llama3_config
    from torchtitan.rl.generator.vllm import attention

    config = build_llama3_config("debugmodel", parallelism=ParallelismConfig())
    config.layers[
        0
    ].attention.inner_attention = SlidingWindowVarlenInnerAttention.Config(
        sliding_window_size=128
    )
    get_dimensions = Mock(wraps=attention.get_attention_dimensions)
    monkeypatch.setattr(attention, "get_attention_dimensions", get_dimensions)

    model_config_to_hf_config_dict(config)

    get_dimensions.assert_called_once_with(config.layers[1].attention, config.dim)


@pytest.mark.parametrize("attention_type", ["sliding", "linear"])
def test_hf_config_requires_full_attention(attention_type):
    config = build_model_config("debugmodel", parallelism=ParallelismConfig())
    if attention_type == "linear":
        config.layers = [
            layer
            for layer in config.layers
            if isinstance(layer.attention, LinearAttention.Config)
        ]
    else:
        for layer in config.layers:
            if not isinstance(layer.attention, LinearAttention.Config):
                layer.attention.inner_attention = (
                    SlidingWindowVarlenInnerAttention.Config(sliding_window_size=128)
                )

    with pytest.raises(ValueError, match="no full attention layer"):
        model_config_to_hf_config_dict(config)


def test_gdn_hybrid_model_registers_state_copy_funcs(monkeypatch):
    copy_funcs = (object(), object())

    class FakeStateCopyFuncCalculator:
        @staticmethod
        def gated_delta_net_state_copy_func():
            return copy_funcs

    monkeypatch.setattr(
        gdn, "MambaStateCopyFuncCalculator", FakeStateCopyFuncCalculator
    )

    gdn_config = Mock(
        spec=GDN.Config,
        in_proj_q=SimpleNamespace(out_features=8),
        in_proj_v=SimpleNamespace(out_features=12),
        key_head_dim=4,
        value_head_dim=6,
        conv_kernel_size=4,
    )
    model_config = SimpleNamespace(layers=[SimpleNamespace(attention=gdn_config)])

    class Model:
        pass

    gdn.maybe_configure_gdn_hybrid_model(Model, model_config)

    gdn_type = object()
    short_conv_type = object()
    assert Model.get_mamba_state_copy_func() is copy_funcs
    assert Model.get_mamba_state_copy_funcs({gdn_type, short_conv_type}) == {
        gdn_type: copy_funcs,
        short_conv_type: copy_funcs,
    }


def test_kda_hybrid_model_registers_state_copy_funcs(monkeypatch):
    copy_funcs = (object(), object())

    class FakeStateCopyFuncCalculator:
        @staticmethod
        def kda_state_copy_func():
            return copy_funcs

    monkeypatch.setattr(
        kda, "MambaStateCopyFuncCalculator", FakeStateCopyFuncCalculator
    )

    kda_config = Mock(
        spec=KDA.Config,
        n_heads=8,
        head_dim=128,
        conv_kernel_size=4,
    )
    model_config = SimpleNamespace(layers=[SimpleNamespace(attention=kda_config)])

    class Model:
        pass

    kda.maybe_configure_kda_hybrid_model(Model, model_config)

    kda_type = object()
    short_conv_type = object()
    assert Model.get_mamba_state_copy_func() is copy_funcs
    assert Model.get_mamba_state_copy_funcs({kda_type, short_conv_type}) == {
        kda_type: copy_funcs,
        short_conv_type: copy_funcs,
    }


def test_batch_invariant_kda_registers_attention_gym_replay_state(monkeypatch):
    conv_copy = object()
    temporal_copy = object()
    base_shapes = ((384, 4), (4, 128, 128))
    base_dtypes = (torch.bfloat16, torch.float32)

    class FakeStateCopyFuncCalculator:
        @staticmethod
        def kda_state_copy_func():
            return conv_copy, temporal_copy

    class FakeStateDtypeCalculator:
        @staticmethod
        def kda_state_dtype(*args, **kwargs):
            return base_dtypes

    class FakeStateShapeCalculator:
        @staticmethod
        def kda_state_shape(*args, **kwargs):
            return base_shapes

    monkeypatch.setattr(
        kda, "MambaStateCopyFuncCalculator", FakeStateCopyFuncCalculator
    )
    monkeypatch.setattr(kda, "MambaStateDtypeCalculator", FakeStateDtypeCalculator)
    monkeypatch.setattr(kda, "MambaStateShapeCalculator", FakeStateShapeCalculator)
    monkeypatch.setattr(kda, "is_in_batch_invariant_mode", lambda: True)

    kda_config = Mock(
        spec=KDA.Config,
        n_heads=8,
        head_dim=128,
        conv_kernel_size=4,
    )
    model_config = SimpleNamespace(layers=[SimpleNamespace(attention=kda_config)])

    class Model:
        pass

    kda.maybe_configure_kda_hybrid_model(Model, model_config)
    vllm_config = SimpleNamespace(
        speculative_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=2),
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        cache_config=SimpleNamespace(mamba_cache_dtype="auto"),
    )

    assert Model.get_mamba_state_shape_from_config(vllm_config) == (
        *base_shapes,
        (64, 4, 128),
        (64, 4, 128),
        (64, 4, 128),
        (64, 4, 128),
        (64, 4),
        (1,),
    )
    assert Model.get_mamba_state_dtype_from_config(vllm_config) == (
        *base_dtypes,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.float32,
        torch.float32,
        torch.int32,
    )
    assert Model.get_mamba_state_copy_func() == (
        conv_copy,
        temporal_copy,
        *(temporal_copy for _ in range(6)),
    )


@pytest.mark.parametrize(
    "build_config", [build_dsv3_config, build_kimi_k3_config], ids=["dsv3", "kimi_k3"]
)
def test_compressed_mla_configures_vllm_native_mla(monkeypatch, build_config):
    config = build_config(
        "debugmodel", parallelism=ParallelismConfig(), attn_backend="varlen"
    )
    assert registry.vllm_attention_backend(config) is None
    hf = model_config_to_hf_config_dict(config)
    mla_attention = config.first_full_attention
    for name in ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
        assert hf[name] == getattr(mla_attention, name)

    monkeypatch.setenv("VLLM_SSM_CONV_STATE_LAYOUT", "SD")
    monkeypatch.delitem(
        MODEL_ARCH_CONFIG_CONVERTORS, registry.TORCHTITAN_MODEL_TYPE, raising=False
    )
    registry.register_to_vllm(
        config, parallelism=InferenceParallelismConfig(), checkpointer_config=None
    )
    hf_config = PretrainedConfig.from_dict(hf)
    assert MODEL_ARCH_CONFIG_CONVERTORS[hf_config.model_type](
        hf_config, hf_config
    ).is_deepseek_mla()
