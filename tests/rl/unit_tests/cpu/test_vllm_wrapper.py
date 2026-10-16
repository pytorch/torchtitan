# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import LinearAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.kimi_k3 import build_model_config as build_kimi_k3_config
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.generator.vllm import model_wrapper
from torchtitan.rl.generator.vllm.attention import VLLMInnerAttention, VLLMMLAAttention
from torchtitan.rl.generator.vllm.model_wrapper import (
    _replace_vllm_layer_configs,
    VLLMModelWrapper,
)


@pytest.mark.parametrize(
    "build_config", [build_model_config, build_kimi_k3_config], ids=["gdn", "kda"]
)
def test_vllm_replacement_keeps_first_full_attention(build_config) -> None:
    """Find the full-attention layer after the vLLM swap, past linear-attention layers."""
    model_config = build_config(
        "debugmodel", attn_backend="varlen", parallelism=ParallelismConfig()
    )
    vllm_config = _replace_vllm_layer_configs(model_config)

    model_index = model_config.layers.index(
        next(
            layer
            for layer in model_config.layers
            if layer.attention is model_config.first_full_attention
        )
    )
    vllm_attention = vllm_config.first_full_attention
    assert vllm_attention is vllm_config.layers[model_index].attention
    assert not isinstance(vllm_attention, LinearAttention.Config)


def test_vllm_inner_attention_sliding_window_is_not_full_attention() -> None:
    config = VLLMInnerAttention.Config(
        hidden_size=8, num_heads=2, num_kv_heads=2, head_dim=4
    )
    assert config.is_full_attention
    assert not dataclasses.replace(config, sliding_window_size=16).is_full_attention


@pytest.mark.parametrize(
    "build_config", [build_model_config, build_kimi_k3_config], ids=["gdn", "kda"]
)
def test_vllm_replacement_preserves_attention_sharding(build_config) -> None:
    """Pass the trainer attention's sharding config through unchanged."""
    model_config = build_config(
        "debugmodel",
        attn_backend="flex",
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2, enable_sequence_parallel=True
        ),
    )
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


class _Wrapper:
    """Stand-in `self` for VLLMModelWrapper's real model state dict load methods (no vLLM)."""

    load_state_dict = VLLMModelWrapper.load_state_dict

    def __init__(self, model: nn.Module):
        self.model = model


def test_load_state_dict_allocates_one_fsdp_group_at_a_time(
    tmp_path, monkeypatch
) -> None:
    """Load new weights with at most two FSDP groups' sharded buffers allocated at once.

    Guards that each group's sharded buffers are allocated before its copy, even when
    FSDP's own load pre-hook already resharded the group (tok_embeddings here).
    """
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        # The generator's layout: a tied [tok_embeddings, norm, lm_head] group, one per block.
        model = nn.Module()
        model.tok_embeddings = nn.Embedding(32, 16)
        model.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)) for _ in range(3)
        )
        model.norm = nn.LayerNorm(16)
        model.lm_head = nn.Linear(16, 32, bias=False)
        model.lm_head.weight = model.tok_embeddings.weight
        mesh = init_device_mesh("cpu", (1,))
        fully_shard([model.tok_embeddings, model.norm, model.lm_head], mesh=mesh)
        for block in model.layers:
            fully_shard(block, mesh=mesh)
        fully_shard(model, mesh=mesh)
        model.set_keep_unsharded_storage(True)
        new_state_dict = {k: v.detach() + 1.0 for k, v in model.state_dict().items()}
        groups = [
            group
            for module in (model.tok_embeddings, *model.layers)
            for group in module._get_fsdp_state()._fsdp_param_groups
        ]

        def is_allocated(group) -> bool:
            return all(
                p._sharded_param_data.untyped_storage().size() > 0
                for p in group.fsdp_params
            )

        wrapper = _Wrapper(model)
        max_allocated_groups = 0

        restore_sharded_params = model_wrapper._restore_sharded_params

        def checked_restore_sharded_params(module) -> None:
            nonlocal max_allocated_groups
            restore_sharded_params(module)
            # The load copies into these buffers next, so freed ones would be overrun.
            assert all(map(is_allocated, module._get_fsdp_state()._fsdp_param_groups))
            max_allocated_groups = max(
                max_allocated_groups, sum(map(is_allocated, groups))
            )

        monkeypatch.setattr(
            model_wrapper, "_restore_sharded_params", checked_restore_sharded_params
        )
        for module in model.modules():  # as after the initial load
            if isinstance(module, FSDPModule):
                model_wrapper._unshard_and_free_sharded_params(module)
        wrapper.load_state_dict(new_state_dict)

        assert max_allocated_groups <= 2
        assert all(
            p._sharded_param_data.untyped_storage().size() == 0
            for group in groups
            for p in group.fsdp_params
        )
        for name, param in model.named_parameters(remove_duplicate=False):
            assert param.equal(new_state_dict[name].to_local()), name
    finally:
        dist.destroy_process_group()


def test_vllm_replaces_mla_with_compact_input_adapter():
    model_config = build_kimi_k3_config(
        "debugmodel",
        parallelism=ParallelismConfig(),
        attn_backend="varlen",
        seq_len=128,
    )

    vllm_config = _replace_vllm_layer_configs(model_config)

    for layer in vllm_config.layers:
        if not isinstance(layer.attention, LinearAttention.Config):
            assert isinstance(
                layer.attention.inner_attention,
                VLLMMLAAttention.Config,
            )
            assert isinstance(
                layer.attention.inner_attention.inner_attention,
                VLLMInnerAttention.Config,
            )


def test_vllm_mla_adapter_builds_wkv_b_from_config():
    config = VLLMMLAAttention.Config(
        wkv_b=Linear.Config(in_features=5, out_features=28),
        packed_kv_head_dim=7,
        inner_attention=VLLMInnerAttention.Config(
            hidden_size=24,
            num_heads=4,
            num_kv_heads=4,
            head_dim=6,
            value_head_dim=3,
        ),
    )

    def initialize_module(attention, config):
        torch.nn.Module.__init__(attention)

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
