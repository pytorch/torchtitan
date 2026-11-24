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
from torch.distributed.tensor import Shard

from torchtitan.config import PrecisionConfig
from torchtitan.config.parallelism import FSDPSymmMemScope, ParallelismConfig
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    enable_fsdp_symm_mem,
    linear_param_shard_placements,
    maybe_store_frozen_params_in_param_dtype_,
)
from torchtitan.distributed.parallelism_context import ParallelismContext
from torchtitan.models.base.module import Module
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE
from torchtitan.models.llama3.flavors import build_model_config


class _FSDPModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.force_sum_enabled = False
        self.symm_mem_enabled = False

    def set_force_sum_reduction_for_comms(self, enable: bool) -> None:
        self.force_sum_enabled = enable

    def set_symm_mem_for_comm(self) -> None:
        self.symm_mem_enabled = True


class _FSDPTransformerBlock(_FSDPModule, TransformerBlock):
    pass


class _FSDPOutOfTreeBlock(_FSDPModule):
    pass


@pytest.mark.parametrize("preserve", [True, False])
@mock.patch("torchtitan.distributed.parallelism_context.device_type", "cpu")
def test_fsdp_unshards_preserve_dtype_params_in_their_dtype(preserve: bool) -> None:
    dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1)
    try:
        parallelism = ParallelismConfig()
        config = build_model_config("debugmodel", parallelism=parallelism)
        if preserve:
            config.norm = dataclasses.replace(
                config.norm, preserve_dtype_params=("weight",)
            )
        model = config.build()
        model._apply_fsdp(
            parallelism_context=ParallelismContext(
                dp_replicate=1,
                dp_shard=1,
                cp=1,
                tp=1,
                pp=1,
                ep=1,
                world_size=1,
                enable_sequence_parallel=False,
            ),
            precision=PrecisionConfig(),
            parallelism=parallelism,
        )
        model.norm.unshard()
        assert model.norm.weight.dtype == (
            torch.float32 if preserve else torch.bfloat16
        )
        assert model.lm_head.weight.dtype == torch.bfloat16
    finally:
        dist.destroy_process_group()


def test_apply_fsdp_to_decoder_shards_an_moe_held_as_mlp() -> None:
    model = Module()
    model.enable_weight_tying = False
    model.tok_embeddings = None
    model.norm = None
    model.lm_head = None
    moe = MoE.__new__(MoE)
    nn.Module.__init__(moe)
    moe.num_experts = 2
    moe.routed_experts = nn.Linear(4, 2, bias=False)
    block = nn.Module()
    block.mlp = moe
    model.layers = nn.ModuleDict({"0": block})
    dp_mesh = mock.MagicMock(mesh_dim_names=("dp_shard",))
    dp_mesh["dp_shard"].size.return_value = 2

    with mock.patch("torchtitan.distributed.fsdp.fully_shard") as fully_shard:
        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
        )

    block_call = next(c for c in fully_shard.call_args_list if c.args[0] is block)
    shard_placement_fn = block_call.kwargs["shard_placement_fn"]
    assert shard_placement_fn(moe.routed_experts.weight) == Shard(0)


def test_stacked_linear_shard_placements_use_num_linears() -> None:
    module = nn.Module()
    stacked = Linear.Config(
        in_features=4,
        out_features=3,
        num_linears=2,
        bias=True,
    ).build()
    shape_only = nn.Linear(4, 6, bias=False)
    shape_only.weight = nn.Parameter(shape_only.weight.unflatten(0, (2, 3)))
    module.add_module("stacked", stacked)
    module.add_module("shape_only", shape_only)

    placements = linear_param_shard_placements(module)

    assert stacked.bias is not None
    assert placements == {
        stacked.weight: Shard(1),
        stacked.bias: Shard(1),
    }


@pytest.mark.parametrize("block_cls", [_FSDPTransformerBlock, _FSDPOutOfTreeBlock])
@pytest.mark.parametrize(
    ("scope", "dense_enabled", "sparse_enabled"),
    [
        (None, False, False),
        ("all", True, True),
        ("dense", True, False),
    ],
)
def test_enable_fsdp_symm_mem_scope(
    scope: FSDPSymmMemScope, dense_enabled: bool, sparse_enabled: bool, block_cls: type
) -> None:
    model = nn.Module()
    dense = block_cls()
    dense.feed_forward = DenseFeedForward.__new__(DenseFeedForward)
    nn.Module.__init__(dense.feed_forward)
    sparse = block_cls()
    moe = MoE.__new__(MoE)
    nn.Module.__init__(moe)
    sparse.feed_forward = moe
    model.add_module("dense", dense)
    model.add_module("sparse", sparse)
    other = _FSDPModule()
    model.add_module("other", other)

    with mock.patch("torchtitan.distributed.fsdp.FSDPModule", _FSDPModule):
        enable_fsdp_symm_mem(model, scope)

    assert dense.force_sum_enabled == dense_enabled
    assert dense.symm_mem_enabled == dense_enabled
    assert sparse.force_sum_enabled == sparse_enabled
    assert sparse.symm_mem_enabled == sparse_enabled
    assert other.force_sum_enabled == dense_enabled
    assert other.symm_mem_enabled == dense_enabled


class _MixedFrozenModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable = nn.Linear(4, 4)
        self.frozen = nn.Linear(4, 4).requires_grad_(False)
        self.tied = nn.Linear(4, 4, bias=False)
        self.tied.weight = self.frozen.weight
        self.preserved = nn.Parameter(torch.ones(4), requires_grad=False)
        self._preserve_dtype_params = ("preserved",)


def test_maybe_store_frozen_params_in_param_dtype() -> None:
    # param_dtype is not narrower than dtype: nothing changes.
    model = _MixedFrozenModel()
    maybe_store_frozen_params_in_param_dtype_(
        model, PrecisionConfig(dtype="float32", param_dtype="float32")
    )
    assert model.frozen.weight.dtype == torch.float32

    model = _MixedFrozenModel()
    frozen_weight = model.frozen.weight
    frozen_weight.custom_attr = "kept"
    expected = frozen_weight.detach().to(torch.bfloat16)

    maybe_store_frozen_params_in_param_dtype_(
        model, PrecisionConfig(dtype="float32", param_dtype="bfloat16")
    )

    assert model.frozen.weight is frozen_weight
    assert model.tied.weight is frozen_weight
    assert frozen_weight.dtype == torch.bfloat16
    assert not frozen_weight.requires_grad
    assert frozen_weight.custom_attr == "kept"
    torch.testing.assert_close(frozen_weight, expected, rtol=0, atol=0)
    assert model.frozen.bias.dtype == torch.bfloat16
    assert model.trainable.weight.dtype == torch.float32
    assert model.trainable.bias.dtype == torch.float32
    assert model.preserved.dtype == torch.float32
