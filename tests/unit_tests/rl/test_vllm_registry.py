# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from types import ModuleType, SimpleNamespace

import torch

from torchtitan.rl.model import vllm_registry
from torchtitan.rl.model.vllm_registry import (
    _configure_gdn_hybrid_model,
    _configure_kda_hybrid_model,
)


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
    model_spec = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(delta_net=gdn_config)])
    )

    class Model:
        pass

    _configure_gdn_hybrid_model(Model, model_spec)

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

    mamba_utils = ModuleType("vllm.model_executor.layers.mamba.mamba_utils")
    mamba_utils.MambaStateCopyFuncCalculator = FakeStateCopyFuncCalculator
    mamba_utils.MambaStateDtypeCalculator = object()
    mamba_utils.MambaStateShapeCalculator = object()
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.mamba.mamba_utils",
        mamba_utils,
    )

    kda_config = SimpleNamespace(
        num_heads=8,
        head_dim=128,
        conv_kernel_size=4,
    )
    model_spec = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(delta_attention=kda_config)])
    )

    class Model:
        pass

    _configure_kda_hybrid_model(Model, model_spec)

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

    mamba_utils = ModuleType("vllm.model_executor.layers.mamba.mamba_utils")
    mamba_utils.MambaStateCopyFuncCalculator = FakeStateCopyFuncCalculator
    mamba_utils.MambaStateDtypeCalculator = FakeStateDtypeCalculator
    mamba_utils.MambaStateShapeCalculator = FakeStateShapeCalculator
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.layers.mamba.mamba_utils",
        mamba_utils,
    )
    monkeypatch.setattr(vllm_registry, "is_in_batch_invariant_mode", lambda: True)

    kda_config = SimpleNamespace(
        num_heads=8,
        head_dim=128,
        conv_kernel_size=4,
    )
    model_spec = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(delta_attention=kda_config)])
    )

    class Model:
        pass

    _configure_kda_hybrid_model(Model, model_spec)
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
