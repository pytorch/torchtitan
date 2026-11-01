# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import importlib
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import InnerAttention, MLAAttention
from torchtitan.models.common.linear import Linear
from torchtitan.models.deepseek_v3 import build_model_config as build_deepseek_config
from torchtitan.models.kimi_k3 import build_model_config as build_kimi_k3_config
from torchtitan.models.qwen3 import build_model_config as build_qwen3_config
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.generator.vllm import model_wrapper, registry
from torchtitan.rl.generator.vllm.attention import VLLMInnerAttention

from torchtitan.rl.generator.vllm.mla import (
    _VLLMKVBProj,
    _VLLMNativeMLAAttention,
    VLLMMLAAttention,
    VLLMMLAInnerAttention,
)
from torchtitan.rl.generator.vllm.model_wrapper import (
    _replace_vllm_layer_configs,
    VLLMModelWrapper,
)
from vllm.model_executor.layers.attention import MLAAttention as VLLMNativeMLAAttention


@pytest.mark.parametrize(
    "model_name, expected_scale",
    [
        ("gpt_oss", 0.22665755425493697),
        ("qwen3", None),
    ],
)
def test_vllm_replacement_preserves_attention_scale(model_name, expected_scale):
    """Preserve YaRN scaling as well as the backend's default scale."""
    model_config = importlib.import_module(
        f"torchtitan.models.{model_name}"
    ).build_model_config("debugmodel", seq_len=128, parallelism=ParallelismConfig())
    vllm_config = _replace_vllm_layer_configs(model_config)

    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        attention_config = model_layer.attention
        scale = vllm_layer.attention.inner_attention.scale
        if expected_scale is None:
            assert scale is None
        else:
            assert scale == pytest.approx(expected_scale)
            with torch.device("meta"):
                attention = attention_config.build()
            assert scale == attention.softmax_scale


@pytest.mark.parametrize(
    "build_config",
    [build_qwen3_config, build_model_config],
    ids=["qwen3", "qwen3_5"],
)
def test_vllm_replacement_preserves_full_attention_selection(build_config) -> None:
    model_config = build_config("debugmodel", parallelism=ParallelismConfig())
    full_attention_index = next(
        i
        for i, layer in enumerate(model_config.layers)
        if layer.attention is model_config.first_full_attention
    )

    vllm_config = _replace_vllm_layer_configs(model_config)

    assert (
        vllm_config.first_full_attention
        is vllm_config.layers[full_attention_index].attention
    )
    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        inner_attention = vllm_layer.attention.inner_attention
        assert isinstance(inner_attention, InnerAttention.Config)
        assert inner_attention._owner is not None
        assert issubclass(inner_attention._owner, InnerAttention)
        assert (
            inner_attention.is_full_attention
            == model_layer.attention.inner_attention.is_full_attention
        )
        assert inner_attention.build_attention_metadata(torch.arange(8)) is None


@pytest.mark.parametrize("sliding_window_size", [None, 128])
def test_vllm_attention_classifies_sliding_window(sliding_window_size) -> None:
    config = VLLMInnerAttention.Config(
        hidden_size=256,
        num_heads=4,
        num_kv_heads=2,
        head_dim=64,
        sliding_window_size=sliding_window_size,
    )
    assert config.is_full_attention is (sliding_window_size is None)


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
    # Check only Kimi K3's KDA layers; test_vllm_replaces_mla_layer_configs covers MLA.
    model_config = dataclasses.replace(
        model_config,
        layers=[
            layer
            for layer in model_config.layers
            if not isinstance(layer.attention.inner_attention, MLAAttention.Config)
        ],
    )
    assert model_config.layers
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


@pytest.mark.parametrize(
    "compressed", [True, False], ids=["compressed", "batch_invariant"]
)
def test_vllm_replaces_mla_layer_configs(monkeypatch, compressed) -> None:
    """MLA uses the compressed cache unless batch invariance keeps materialized K/V."""
    monkeypatch.setattr(registry, "is_in_batch_invariant_mode", lambda: not compressed)
    model_config = build_deepseek_config(
        "debugmodel",
        parallelism=ParallelismConfig(),
        attn_backend="varlen",
        seq_len=128,
    )

    vllm_config = _replace_vllm_layer_configs(model_config)

    num_mla_layers = 0
    for model_layer, vllm_layer in zip(
        model_config.layers, vllm_config.layers, strict=True
    ):
        if not isinstance(model_layer.attention.inner_attention, MLAAttention.Config):
            continue
        num_mla_layers += 1
        attention = model_layer.attention
        vllm_mla_config = vllm_layer.attention.inner_attention
        inner_config = vllm_mla_config.inner_attention
        assert vllm_mla_config.wkv_b is attention.inner_attention.wkv_b
        assert inner_config.is_full_attention
        assert inner_config.scale == attention.softmax_scale
        if compressed:
            assert type(vllm_mla_config) is VLLMMLAAttention.Config
            assert type(inner_config) is VLLMMLAInnerAttention.Config
            assert inner_config.num_heads == attention.n_heads
            assert inner_config.kv_lora_rank == attention.kv_lora_rank
            assert inner_config.qk_nope_head_dim == attention.qk_nope_head_dim
            assert inner_config.qk_rope_head_dim == attention.qk_rope_head_dim
            assert inner_config.v_head_dim == attention.v_head_dim
        else:
            assert type(vllm_mla_config) is MLAAttention.Config
            assert type(inner_config) is VLLMInnerAttention.Config
    assert num_mla_layers > 0


def test_vllm_compressed_mla_owns_wkv_b():
    config = VLLMMLAAttention.Config(
        wkv_b=Linear.Config(in_features=5, out_features=28),
        packed_kv_head_dim=7,
        inner_attention=VLLMMLAInnerAttention.Config(
            num_heads=4,
            kv_lora_rank=5,
            qk_nope_head_dim=4,
            qk_rope_head_dim=2,
            v_head_dim=3,
            scale=0.5,
        ),
    )
    with mock.patch.object(
        VLLMMLAInnerAttention.Config,
        "build",
        autospec=True,
        return_value=nn.Identity(),
    ) as build:
        attention = VLLMMLAAttention(config)

    build.assert_called_once_with(config.inner_attention, kv_b_proj=attention.wkv_b)
    assert "wkv_b.weight" in attention.state_dict()


def test_vllm_mla_inner_attention_matches_vllm_mla_shapes():
    attention = VLLMMLAInnerAttention.__new__(VLLMMLAInnerAttention)
    torch.nn.Module.__init__(attention)
    attention.num_heads = 4
    attention.v_head_dim = 3
    attention.vllm_mla = vllm_mla = mock.Mock(
        scale=0.5, return_value=torch.randn(8, 12)
    )
    q_THK = torch.randn(8, 4, 6)
    kv_c_normed_TL = torch.randn(8, 5)
    k_shared_TR = torch.randn(8, 2)

    out_THV = attention(q_THK, kv_c_normed_TL, k_shared_TR, scale=0.5)

    args, kwargs = vllm_mla.call_args
    assert args[0] is q_THK
    assert args[1] is kv_c_normed_TL
    assert torch.equal(args[2], k_shared_TR.unsqueeze(1))
    assert kwargs == {"output_shape": (8, 12)}
    assert torch.equal(out_THV, vllm_mla.return_value.view(8, 4, 3))


@pytest.mark.parametrize("fsdp", [False, True], ids=["plain", "fully_shard"])
def test_vllm_native_mla_decode_weights_follow_weight_sync(tmp_path, fsdp) -> None:
    """vLLM's decode weights are views of wkv_b, so weight sync updates them."""
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        torch.manual_seed(0)
        block = nn.Module()
        block.wkv_b = Linear.Config(in_features=5, out_features=2 * (3 + 4)).build()
        block.to(torch.bfloat16)
        torch.nn.init.normal_(block.wkv_b.weight)
        model = nn.Module()
        model.layers = nn.ModuleList([block])
        if fsdp:
            mesh = init_device_mesh("cpu", (1,))
            fully_shard(block, mesh=mesh)
            fully_shard(model, mesh=mesh)
            model.set_keep_unsharded_storage(True)
        new_state_dict = {k: 3.0 * v.detach() for k, v in model.state_dict().items()}
        for module in model.modules():  # as after the initial load
            if isinstance(module, FSDPModule):
                model_wrapper._unshard_and_free_sharded_params(module)

        attention = _VLLMNativeMLAAttention.__new__(_VLLMNativeMLAAttention)
        torch.nn.Module.__init__(attention)
        attention.impl = mock.Mock()
        attention.is_amx_bmm_enabled = False
        attention.is_aiter_triton_fp4_bmm_enabled = False
        attention.is_aiter_triton_fp8_bmm_enabled = False
        attention.dcp_q_replicate = False
        attention.quant_config = None
        attention.kv_lora_rank = 5
        attention.num_heads = 2
        attention.qk_nope_head_dim = 3
        attention.v_head_dim = 4
        attention.kv_b_proj = _VLLMKVBProj(block.wkv_b)
        for name in VLLMInnerAttention.vllm_attn_scale_buffer_names:
            attention.register_buffer(name, torch.tensor(0.0), persistent=False)
        block.vllm_mla = attention

        attention.process_weights_after_loading(torch.bfloat16)

        assert not attention.state_dict()
        assert {"W_UK_T", "W_UV"} <= {name for name, _ in attention.named_buffers()}
        _Wrapper(model).load_state_dict(new_state_dict)

        new_weight = new_state_dict["layers.0.wkv_b.weight"]
        if fsdp:
            new_weight = new_weight.to_local()
        w_uk, w_uv = new_weight.T.view(5, 2, 7).split([3, 4], dim=-1)
        assert torch.equal(attention.W_UK_T, w_uk.permute(1, 2, 0))
        assert torch.equal(attention.W_UV, w_uv.transpose(0, 1))
    finally:
        dist.destroy_process_group()


def test_vllm_native_mla_rejects_copied_decode_weights() -> None:
    """A decode weight that does not share wkv_b's storage would go stale silently."""
    wkv_b = Linear.Config(in_features=5, out_features=2 * (3 + 4)).build()
    attention = _VLLMNativeMLAAttention.__new__(_VLLMNativeMLAAttention)
    torch.nn.Module.__init__(attention)
    attention.kv_b_proj = _VLLMKVBProj(wkv_b)

    def copy_decode_weights(self, act_dtype) -> None:
        self.W_UK_T = nn.Parameter(torch.zeros(2, 3, 5), requires_grad=False)
        self.W_UV = nn.Parameter(torch.zeros(2, 5, 4), requires_grad=False)

    with mock.patch.object(
        VLLMNativeMLAAttention, "process_weights_after_loading", copy_decode_weights
    ), pytest.raises(AssertionError, match="must be a view of wkv_b.weight"):
        attention.process_weights_after_loading(torch.bfloat16)
