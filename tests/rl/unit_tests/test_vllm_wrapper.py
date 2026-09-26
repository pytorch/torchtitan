# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta
from pathlib import Path

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.attention import QKVLinear
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.models.qwen3_5 import model_registry
from torchtitan.models.qwen3_5.model import Qwen35Model
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
from torchtitan.protocols.sharding import ShardingConfig

from torchtitan.rl.model.vllm_wrapper import (
    PlainToDTensorStateDictAdapter,
    VLLMModelWrapper,
)


def test_state_dict_layouts_include_native_feed_forward_weight():
    """Verify the fused dense FFN layout uses its native w13 state-dict key."""
    colwise = dense_param_placement(tp=spmd.S(1))
    rowwise = dense_param_placement(tp=spmd.S(1))
    config = FeedForward.Config(
        w13=Linear.Config(
            in_features=16,
            out_features=32,
            num_linears=2,
            sharding_config=ShardingConfig(state_shardings={"weight": colwise}),
        ),
        w2=Linear.Config(
            in_features=32,
            out_features=16,
            sharding_config=ShardingConfig(state_shardings={"weight": rowwise}),
        ),
    )
    model = torch.nn.Module()
    model.feed_forward = config.build()
    wrapper = VLLMModelWrapper.__new__(VLLMModelWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model

    layouts = wrapper.get_state_dict_layouts()

    assert layouts["feed_forward.w13.weight"] is colwise
    assert "feed_forward.w1.weight" not in layouts
    assert "feed_forward.w3.weight" not in layouts
    assert layouts["feed_forward.w2.weight"] is rowwise


def test_state_dict_layouts_include_native_qkv_weight():
    """Verify QKV layout lookup uses the native packed state-dict key."""
    colwise = dense_param_placement(tp=spmd.S(0))
    config = QKVLinear.Config(
        head_dim=8,
        n_heads=4,
        n_kv_heads=2,
        wqkv=Linear.Config(
            in_features=16,
            out_features=64,
            sharding_config=ShardingConfig(state_shardings={"weight": colwise}),
        ),
    )
    model = torch.nn.Module()
    model.qkv_linear = config.build()
    wrapper = VLLMModelWrapper.__new__(VLLMModelWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model

    layouts = wrapper.get_state_dict_layouts()

    assert layouts["qkv_linear.wqkv.weight"] is colwise
    assert "qkv_linear.wq.weight" not in layouts
    assert "qkv_linear.wk.weight" not in layouts
    assert "qkv_linear.wv.weight" not in layouts


def test_state_dict_layouts_include_native_grouped_linear_weights():
    """Verify routed expert layouts use native grouped-linear state keys."""
    physical_colwise = dense_param_placement(tp=spmd.S(2))
    rowwise = dense_param_placement(tp=spmd.S(1))
    w13_config = GroupedLinear.Config(
        group_size=4,
        in_features=16,
        out_features=32,
        num_linears=2,
        sharding_config=ShardingConfig(state_shardings={"weight": physical_colwise}),
    )
    w2_config = GroupedLinear.Config(
        group_size=4,
        in_features=32,
        out_features=16,
        sharding_config=ShardingConfig(state_shardings={"weight": rowwise}),
    )
    model = torch.nn.Module()
    model.experts = torch.nn.Module()
    model.experts.w13 = w13_config.build()
    model.experts.w2 = w2_config.build()
    wrapper = VLLMModelWrapper.__new__(VLLMModelWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = model

    layouts = wrapper.get_state_dict_layouts()

    assert layouts["experts.w13.weight"] is physical_colwise
    assert layouts["experts.w2.weight"] is rowwise


def _check_hf_adapter_restores_local_shards(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("tp",))
        model_config = model_registry("0.8B", seq_len=256, attn_backend="varlen")
        assert isinstance(model_config, Qwen35Model.Config)
        model_adapter = Qwen35StateDictAdapter(model_config, hf_assets_path=None)
        state_dict, expected, layouts = {}, {}, {}
        for pattern, shape in (
            ("layers.0.attn.in_proj_{}.weight", (2048, 1024)),
            ("layers.0.attn.conv_{}.weight", (2048, 1, 4)),
            ("vision_encoder.layers.0.attn.w{}.weight", (768, 768)),
            ("vision_encoder.layers.0.attn.w{}.bias", (768,)),
        ):
            for index, part in enumerate(("q", "k", "v")):
                key = pattern.format(part)
                full = torch.arange(torch.Size(shape).numel()).reshape(shape).float()
                full += index * 100
                state_dict[key] = distribute_tensor(full, mesh, [Shard(0)])
                expected[key] = full.chunk(2, dim=0)[rank].clone()
                layouts[key] = dense_param_placement(tp=spmd.S(0))
        # Check unchanged row-sharded, replicated, and plain values as well.
        for key, placement, shape in (
            ("layers.3.attn.wo.weight", Shard(1), (8, 8)),
            ("norm.weight", Replicate(), (8,)),
        ):
            full = torch.arange(torch.Size(shape).numel()).reshape(shape).float()
            state_dict[key] = distribute_tensor(full, mesh, [placement])
            expected[key] = (
                full.chunk(2, dim=1)[rank].clone()
                if isinstance(placement, Shard)
                else full
            )
            layouts[key] = dense_param_placement(
                tp=spmd.S(1) if isinstance(placement, Shard) else spmd.R
            )
        state_dict["lm_head.weight"] = expected["lm_head.weight"] = torch.ones(1)
        adapter = PlainToDTensorStateDictAdapter(
            model_adapter,
            layouts,
            ParallelDims(
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                tp=2,
                pp=1,
                ep=1,
                world_size=2,
                enable_sequence_parallel=False,
            ),
        )
        restored = adapter.from_hf(model_adapter.to_hf(state_dict))
        assert restored.keys() == expected.keys()
        for key in expected:
            assert type(restored[key]) is torch.Tensor
            torch.testing.assert_close(restored[key], expected[key], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def test_hf_adapter_restores_local_shards(tmp_path: Path) -> None:
    mp.spawn(
        _check_hf_adapter_restores_local_shards,
        args=(f"file://{tmp_path / 'rendezvous'}",),
        nprocs=2,
        join=True,
    )
