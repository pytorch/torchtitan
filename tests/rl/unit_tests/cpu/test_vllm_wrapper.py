# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import dataclasses
import importlib

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import InnerAttention, MLAAttention
from torchtitan.models.deepseek_v3 import build_model_config as build_deepseek_config
from torchtitan.models.kimi_k2_7 import build_model_config as build_kimi_k2_7_config
from torchtitan.models.kimi_k3 import build_model_config as build_kimi_k3_config
from torchtitan.models.qwen3 import build_model_config as build_qwen3_config
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.rl.generator.vllm import model_wrapper
from torchtitan.rl.generator.vllm.attention import VLLMInnerAttention
from torchtitan.rl.generator.vllm.model_wrapper import (
    _replace_vllm_layer_configs,
    VLLMModelWrapper,
)


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
    # Exercise KDA independently of the unsupported MLA layers in Kimi K3.
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
    load_state_dict_direct = VLLMModelWrapper.load_state_dict_direct

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


def test_load_state_dict_direct_fetches_one_fsdp_group_at_a_time(tmp_path) -> None:
    """Fetch new weights into the sharded parameters with one FSDP group allocated at a time.

    Guards that a group left mid-forward (the tied group after a standalone lm_head
    call) still receives the new weights.
    """
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        model = nn.Module()
        model.tok_embeddings = nn.Embedding(32, 16)
        model.layers = nn.ModuleList(nn.Linear(16, 16) for _ in range(3))
        model.lm_head = nn.Linear(16, 32, bias=False)
        model.lm_head.weight = model.tok_embeddings.weight
        model.register_buffer("scale", torch.zeros(4))
        mesh = init_device_mesh("cpu", (1,))
        # The generator never reshards after forward.
        fully_shard(
            [model.tok_embeddings, model.lm_head],
            mesh=mesh,
            reshard_after_forward=False,
        )
        for block in model.layers:
            fully_shard(block, mesh=mesh, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        model.set_keep_unsharded_storage(True)
        new_state_dict = {k: v.detach() + 1.0 for k, v in model.state_dict().items()}
        wrapper = _Wrapper(model)
        (
            wrapper._direct_load_groups,
            wrapper._direct_load_other_keys,
        ) = model_wrapper._direct_load_groups(model)
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

        for module in model.modules():  # as after the initial load
            if isinstance(module, FSDPModule):
                model_wrapper._unshard_and_free_sharded_params(module)
        # As in compute_logits: calling lm_head alone leaves the tied group's forward
        # incomplete, so module.reshard() would skip it.
        model._get_fsdp_state()._lazy_init()  # the root's forward would do this
        with torch.no_grad():
            model.lm_head(torch.zeros(1, 16))

        fetched_keys = []

        async def fetch(state_dict: dict[str, torch.Tensor]) -> None:
            fetched_keys.append(sorted(state_dict))
            # Only the fetched group may hold sharded buffers.
            assert sum(map(is_allocated, groups)) <= 1
            with torch.no_grad():
                for key, tensor in state_dict.items():
                    tensor.copy_(new_state_dict[key])

        asyncio.run(wrapper.load_state_dict_direct(fetch))

        assert fetched_keys == [
            ["lm_head.weight", "tok_embeddings.weight"],
            ["layers.0.bias", "layers.0.weight"],
            ["layers.1.bias", "layers.1.weight"],
            ["layers.2.bias", "layers.2.weight"],
            ["scale"],
        ]
        assert not any(map(is_allocated, groups))
        for name, param in model.named_parameters(remove_duplicate=False):
            assert param.equal(new_state_dict[name].to_local()), name
        assert model.scale.equal(new_state_dict["scale"])
    finally:
        dist.destroy_process_group()


class TestLoadStateDictDirectTwoRanks(DTensorTestBase):
    @property
    def world_size(self) -> int:
        return 2

    @property
    def backend(self) -> str:
        return "gloo"

    @property
    def device_type(self) -> str:
        return "cpu"

    @with_comms
    def test_fetches_shards_into_the_unsharded_params(self) -> None:
        """Load twice; each rank fetches its own (uneven) shards and every rank ends
        with the full new weights, including the tied group."""
        model = nn.Module()
        model.tok_embeddings = nn.Embedding(15, 16)
        model.layers = nn.ModuleList(nn.Linear(16, 15) for _ in range(2))
        model.lm_head = nn.Linear(16, 15, bias=False)
        model.lm_head.weight = model.tok_embeddings.weight
        model.register_buffer("scale", torch.zeros(4))
        mesh = init_device_mesh("cpu", (self.world_size,))
        fully_shard(
            [model.tok_embeddings, model.lm_head],
            mesh=mesh,
            reshard_after_forward=False,
        )
        for block in model.layers:
            fully_shard(block, mesh=mesh, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        model.set_keep_unsharded_storage(True)
        initial_state_dict = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }
        wrapper = _Wrapper(model)
        (
            wrapper._direct_load_groups,
            wrapper._direct_load_other_keys,
        ) = model_wrapper._direct_load_groups(model)
        for module in model.modules():  # as after the initial load
            if isinstance(module, FSDPModule):
                model_wrapper._unshard_and_free_sharded_params(module)

        for version in (1, 2):
            new_state_dict = {k: v + version for k, v in initial_state_dict.items()}

            async def fetch(state_dict: dict[str, torch.Tensor]) -> None:
                with torch.no_grad():
                    for key, tensor in state_dict.items():
                        tensor.copy_(new_state_dict[key])

            asyncio.run(wrapper.load_state_dict_direct(fetch))

            for name, param in model.named_parameters(remove_duplicate=False):
                self.assertEqual(param, new_state_dict[name].full_tensor(), msg=name)
            self.assertEqual(model.scale, new_state_dict["scale"])

    @with_comms
    def test_unshards_every_group_after_fetch_error(self) -> None:
        """Unshard every group after a failed fetch on one rank, then raise.

        unshard() all-gathers across the FSDP group, so if rank 1 stopped after
        its fetch failed, rank 0 would hang in the next group's all-gather.
        """
        model = nn.Module()
        model.layers = nn.ModuleList(nn.Linear(16, 16) for _ in range(3))
        model.register_buffer("scale", torch.zeros(4))
        mesh = init_device_mesh("cpu", (self.world_size,))
        for block in model.layers:
            fully_shard(block, mesh=mesh, reshard_after_forward=False)
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        model.set_keep_unsharded_storage(True)
        wrapper = _Wrapper(model)
        (
            wrapper._direct_load_groups,
            wrapper._direct_load_other_keys,
        ) = model_wrapper._direct_load_groups(model)
        for module in model.modules():  # as after the initial load
            if isinstance(module, FSDPModule):
                model_wrapper._unshard_and_free_sharded_params(module)
        fetched_keys = []

        async def fetch(state_dict: dict[str, torch.Tensor]) -> None:
            fetched_keys.append(sorted(state_dict))
            if self.rank == 1:
                raise RuntimeError("fetch failed")

        if self.rank == 1:
            with self.assertRaisesRegex(RuntimeError, "fetch failed"):
                asyncio.run(wrapper.load_state_dict_direct(fetch))
            self.assertEqual(fetched_keys, [["layers.0.bias", "layers.0.weight"]])
        else:
            asyncio.run(wrapper.load_state_dict_direct(fetch))
            self.assertEqual(len(fetched_keys), 4)
        self.assertTrue(
            all(
                p._sharded_param_data.untyped_storage().size() == 0
                for block in model.layers
                for group in block._get_fsdp_state()._fsdp_param_groups
                for p in group.fsdp_params
            )
        )


def test_direct_load_groups_rejects_quantized_params(tmp_path, monkeypatch) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    try:
        model = nn.Sequential(nn.Linear(16, 16))
        fully_shard(model, mesh=init_device_mesh("cpu", (1,)))
        # Treat every sharded parameter as quantized.
        monkeypatch.setattr(model_wrapper, "_ShardedFSDPTensor", torch.Tensor)
        with pytest.raises(ValueError, match="does not support quantized parameters"):
            model_wrapper._direct_load_groups(model)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("build_config", "attn_backend"),
    [
        pytest.param(build_deepseek_config, "flex", id="deepseek_v3-flex"),
        pytest.param(build_deepseek_config, "varlen", id="deepseek_v3-varlen"),
        pytest.param(build_kimi_k2_7_config, "flex", id="kimi_k2_7-flex"),
        pytest.param(build_kimi_k3_config, "flex", id="kimi_k3-flex"),
        pytest.param(build_kimi_k3_config, "varlen", id="kimi_k3-varlen"),
    ],
)
def test_vllm_rejects_mla(build_config, attn_backend) -> None:
    model_config = build_config(
        "debugmodel",
        parallelism=ParallelismConfig(),
        attn_backend=attn_backend,
        seq_len=128,
    )

    with pytest.raises(ValueError, match="MLA attention is not supported"):
        _replace_vllm_layer_configs(model_config)
