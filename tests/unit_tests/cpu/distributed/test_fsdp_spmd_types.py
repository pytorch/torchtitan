# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace
from unittest.mock import Mock

import spmd_types as spmd
import torch

# Install TorchTitan's type rule before exercising the FSDP autograd function.
import torchtitan.distributed.spmd_types  # noqa: F401
from spmd_types.checker import typecheck
from torch.distributed.fsdp._fully_shard._fsdp_param_group import (
    RegisterPostBackwardFunction,
)
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig


def test_fsdp_identity_views_preserve_each_input_layout_and_gradient():
    residual = torch.randn(8, 4, requires_grad=True)
    shared_kv = torch.randn(8, 4, requires_grad=True)
    group = SimpleNamespace(post_backward=Mock())
    axis = spmd.MeshAxis.of(2, 1)
    with spmd.set_current_mesh({"tp": axis}), typecheck():
        spmd.assert_type(residual, {"tp": spmd.S(0)})
        spmd.assert_type(shared_kv, {"tp": spmd.R})
        outputs = RegisterPostBackwardFunction.apply(group, residual, shared_kv)
        assert spmd.get_local_type(outputs[0]) == spmd.get_local_type(residual)
        assert spmd.get_local_type(outputs[1]) == spmd.get_local_type(shared_kv)
    torch.testing.assert_close(outputs[0], residual, rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], shared_kv, rtol=0, atol=0)
    (2 * outputs[0].sum() + 3 * outputs[1].sum()).backward()
    torch.testing.assert_close(
        residual.grad, torch.full_like(residual, 2), rtol=0, atol=0
    )
    torch.testing.assert_close(
        shared_kv.grad, torch.full_like(shared_kv, 3), rtol=0, atol=0
    )
    group.post_backward.assert_called_once()


def test_tuple_source_layouts_annotate_opaque_outputs():
    module = Module()
    module._sharding_config = ShardingConfig(
        out_src_shardings=(
            spmd.SpmdType({"tp": spmd.S(0)}),
            spmd.SpmdType({"tp": spmd.R}),
        )
    )
    values = (torch.randn(8, 4), torch.randn(8, 4))
    axis = spmd.MeshAxis.of(2, 1)
    with spmd.set_current_mesh({"tp": axis}), typecheck():
        outputs = module._redistribute_outputs(values)
        assert outputs is values
        assert spmd.get_local_type(outputs[0])[axis] == spmd.V
        assert spmd.get_partition_spec(outputs[0]) == spmd.PartitionSpec(axis, None)
        assert spmd.get_local_type(outputs[1])[axis] == spmd.R
