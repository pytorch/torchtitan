# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import mock

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Shard

from torchtitan.config.parallelism import FSDPSymmMemScope
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    enable_fsdp_symm_mem,
    linear_param_shard_placements,
)
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE


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


def test_apply_fsdp_to_decoder_forwards_param_dtype_override_fn() -> None:
    model = nn.Module()
    model.enable_weight_tying = False
    model.tok_embeddings = None
    model.norm = None
    model.lm_head = None
    model.layers = nn.ModuleDict()
    dp_mesh = mock.Mock(mesh_dim_names=("dp_shard",))

    def param_dtype_override_fn(param: nn.Parameter) -> torch.dtype | None:
        return torch.float32

    with mock.patch("torchtitan.distributed.fsdp.fully_shard") as fully_shard:
        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=False,
            param_dtype_override_fn=param_dtype_override_fn,
        )

    mp_policy = fully_shard.call_args.kwargs["mp_policy"]
    assert mp_policy.param_dtype_override_fn is param_dtype_override_fn


@pytest.mark.parametrize("enable_backward_prefetch", [True, False])
def test_apply_fsdp_to_decoder_configures_backward_prefetch_policy(
    enable_backward_prefetch: bool,
) -> None:
    """EP models may trade backward communication overlap for peak memory."""
    model = _FSDPModule()
    model.enable_weight_tying = False
    model.tok_embeddings = _FSDPModule()
    model.norm = _FSDPModule()
    model.lm_head = _FSDPModule()
    layer = _FSDPModule()
    layer.feed_forward = nn.Identity()
    layer.feed_forward.set_modules_to_backward_prefetch = mock.Mock()
    model.layers = nn.ModuleDict({"0": layer})
    fsdp_modules = (model, model.tok_embeddings, model.norm, model.lm_head, layer)
    for module in fsdp_modules:
        module.set_gradient_divide_factor = mock.Mock()
        module.set_modules_to_forward_prefetch = mock.Mock()
        module.set_modules_to_backward_prefetch = mock.Mock()
    dp_mesh = mock.Mock(mesh_dim_names=("dp_shard",))

    with (
        mock.patch("torchtitan.distributed.fsdp.fully_shard"),
        mock.patch("torchtitan.distributed.fsdp.FSDPModule", _FSDPModule),
        mock.patch(
            "torchtitan.distributed.fsdp._fsdp_unit_holds_only_embedding_weights",
            return_value=False,
        ),
    ):
        apply_fsdp_to_decoder(
            model,
            dp_mesh,
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            pp_enabled=True,
            ep_degree=2,
            enable_explicit_backward_prefetch=enable_backward_prefetch,
        )

    model.tok_embeddings.set_modules_to_forward_prefetch.assert_called_once_with(
        [layer]
    )
    layer.feed_forward.set_modules_to_backward_prefetch.assert_not_called()
    if enable_backward_prefetch:
        model.lm_head.set_modules_to_backward_prefetch.assert_called_once_with([layer])
        layer.set_modules_to_backward_prefetch.assert_called_once_with(
            [model.tok_embeddings]
        )
        model.tok_embeddings.set_modules_to_backward_prefetch.assert_not_called()
        model.norm.set_modules_to_backward_prefetch.assert_not_called()
    else:
        for module in fsdp_modules:
            module.set_modules_to_backward_prefetch.assert_called_once_with([module])


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


@pytest.mark.parametrize(
    ("scope", "dense_enabled", "sparse_enabled"),
    [
        (None, False, False),
        ("all", True, True),
        ("dense", True, False),
    ],
)
def test_enable_fsdp_symm_mem_scope(
    scope: FSDPSymmMemScope, dense_enabled: bool, sparse_enabled: bool
) -> None:
    model = nn.Module()
    dense = _FSDPTransformerBlock()
    dense.feed_forward = DenseFeedForward.__new__(DenseFeedForward)
    nn.Module.__init__(dense.feed_forward)
    sparse = _FSDPTransformerBlock()
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
