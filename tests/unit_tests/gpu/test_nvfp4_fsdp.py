# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy


pytest.importorskip("torchao")
pytest.importorskip("torchao.prototype.moe_training.nvfp4_training")

import torchtitan.quantization.nvfp4.tensor as nvfp4_tensor  # noqa: E402
from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor  # noqa: E402
from torchtitan.quantization.nvfp4 import NVFP4Linear  # noqa: E402
from torchtitan.quantization.nvfp4.tensor import (  # noqa: E402
    _LinearShardedTensorWithNVFP4Compute,
)


pytestmark = [
    pytest.mark.multi_gpu,
    pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
        reason="NVFP4 requires SM100 or later",
    ),
]


def _get_weight_param(linear):
    state = fully_shard.state(linear)
    param_group = state._fsdp_param_group
    assert param_group is not None
    return next(
        param
        for param in param_group.fsdp_params
        if param._module_info.param_name == "weight"
    )


def _run_nvfp4_fsdp_lifecycle(
    rank: int,
    world_size: int,
    port: int,
    reshard_after_forward: bool,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    original_quantize_weight = nvfp4_tensor._quantize_nvfp4_weight
    num_quantize_calls = 0

    def counted_quantize_weight(*args, **kwargs):
        nonlocal num_quantize_calls
        num_quantize_calls += 1
        return original_quantize_weight(*args, **kwargs)

    nvfp4_tensor._quantize_nvfp4_weight = counted_quantize_weight
    try:
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))
        linear = (
            NVFP4Linear.Config(in_features=128, out_features=128, bias=False)
            .build()
            .cuda()
            .bfloat16()
        )
        linear._init_self_buffers(buffer_device=torch.device("cuda"))
        fully_shard(
            linear,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            ),
            reshard_after_forward=reshard_after_forward,
        )
        input_MK = torch.randn(
            128,
            128,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        if reshard_after_forward:
            output_MN = linear(input_MK)
            weight_param = _get_weight_param(linear)
            inner_tensor_ids = tuple(map(id, weight_param._unsharded_inner_tensors))
            assert num_quantize_calls == 1
            assert isinstance(
                linear.weight.to_local(), _LinearShardedTensorWithNVFP4Compute
            )
            assert all(
                tensor.untyped_storage().size() == 0
                for tensor in weight_param._unsharded_inner_tensors
            )
            output_MN.sum().backward()
            assert num_quantize_calls == 2
            assert (
                tuple(map(id, weight_param._unsharded_inner_tensors))
                == inner_tensor_ids
            )
        else:
            linear.set_is_last_backward(False)
            linear.set_reshard_after_backward(False)
            linear.set_requires_gradient_sync(False)
            outputs = [linear(input_MK), linear(input_MK)]
            weight_param = _get_weight_param(linear)
            assert num_quantize_calls == 1
            assert isinstance(linear.weight, _UnshardedFSDPTensor)
            assert all(
                tensor.untyped_storage().size() > 0
                for tensor in weight_param._unsharded_inner_tensors
            )
            outputs[0].sum().backward(retain_graph=True)
            assert num_quantize_calls == 1
            linear.set_is_last_backward(True)
            linear.set_reshard_after_backward(True)
            linear.set_requires_gradient_sync(True)
            outputs[1].sum().backward()
            assert num_quantize_calls == 1
            assert isinstance(
                linear.weight.to_local(), _LinearShardedTensorWithNVFP4Compute
            )
    finally:
        nvfp4_tensor._quantize_nvfp4_weight = original_quantize_weight
        dist.destroy_process_group()


@pytest.mark.parametrize("reshard_after_forward", [True, False])
def test_nvfp4_fsdp_tensor_lifecycle(reshard_after_forward):
    mp.spawn(
        _run_nvfp4_fsdp_lifecycle,
        args=(2, get_free_port(), reshard_after_forward),
        nprocs=2,
        join=True,
    )
