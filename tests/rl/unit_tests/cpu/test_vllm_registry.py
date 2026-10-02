# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import ModuleType, SimpleNamespace

from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.model.vllm_registry import (
    _configure_gdn_hybrid_model,
    model_config_to_hf_config_dict,
)

from transformers import GenerationConfig, PretrainedConfig


def test_hf_config_adds_no_stop_token():
    # vLLM derives the generation config from the HF config when the checkpoint
    # has no generation_config.json; any eos_token_id there stops every request.
    hf_config = PretrainedConfig(
        **model_config_to_hf_config_dict(
            build_model_config("0.8B", seq_len=256, attn_backend="varlen")
        )
    )
    assert GenerationConfig.from_model_config(hf_config).eos_token_id is None


def test_gdn_hybrid_model_registers_state_copy_funcs(monkeypatch):
    copy_funcs = (object(), object())

    class FakeStateCopyFuncCalculator:
        @staticmethod
        def gated_delta_net_state_copy_func():
            return copy_funcs

    mamba_utils = ModuleType("vllm.model_executor.layers.mamba.mamba_utils")
    mamba_utils.MambaStateCopyFuncCalculator = FakeStateCopyFuncCalculator
    mamba_utils.MambaStateDtypeCalculator = object()
    mamba_utils.MambaStateShapeCalculator = object()
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.mamba.mamba_utils",
        mamba_utils,
    )

    gdn_config = SimpleNamespace(
        in_proj_q=SimpleNamespace(out_features=8),
        in_proj_v=SimpleNamespace(out_features=12),
        key_head_dim=4,
        value_head_dim=6,
        conv_kernel_size=4,
    )
    model_config = SimpleNamespace(layers=[SimpleNamespace(delta_net=gdn_config)])

    class Model:
        pass

    _configure_gdn_hybrid_model(Model, model_config)

    gdn_type = object()
    short_conv_type = object()
    assert Model.get_mamba_state_copy_func() is copy_funcs
    assert Model.get_mamba_state_copy_funcs({gdn_type, short_conv_type}) == {
        gdn_type: copy_funcs,
        short_conv_type: copy_funcs,
    }
