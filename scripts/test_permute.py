#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, Shard
from torchtitan.tools.logging import logger
from torch.distributed.tensor.placement_types import _StridedShard

def _permute(w, n_heads_arg, dim1=None, dim2=None):
    """Copy of the permute function from Llama3StateDictAdapter"""
    if dim1 is None:
        dim1 = w.shape[0]
    if dim2 is None:
        dim2 = w.shape[1]
    
    if hasattr(tensor, 'placements'):
        w0_full = w.full_tensor()
        w1 = w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
        print(f"w1 tensor shape: {w1.shape}, placements: {w1.placements}, device_mesh: {w1.device_mesh}")
        w1_full = w1.full_tensor()
        print(f"Are w0 and w1 tensors equal? {torch.allclose(w0_full.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2), w1_full, rtol=1e-5, atol=1e-5)}")
        w2 = w1.transpose(1, 2)
        w2_full = w2.full_tensor()
        print(f"Are w1 and w2 tensors equal? {torch.allclose(w1_full.transpose(1, 2), w2_full, rtol=1e-5, atol=1e-5)}")
        w3 = w2.reshape(dim1, dim2)
        w3_full = w3.full_tensor()
        print(f"Are w2 and w3 tensors equal? {torch.allclose(w2_full.reshape(dim1, dim2), w3_full, rtol=1e-5, atol=1e-5)}")
        w4 = w3.clone()
        w4_full = w4.full_tensor()
        print(f"Are w2 and w3 tensors equal? {torch.allclose(w3_full.clone(), w4_full, rtol=1e-5, atol=1e-5)}")
        return w4
    else:
        # To permute a plain tensor
        return (
            w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

def _view_test(w, n_heads_arg, dim1=None, dim2=None):
    if dim1 is None:
        dim1 = w.shape[0]
    if dim2 is None:
        dim2 = w.shape[1]
    return (
        w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
    )

def _reverse_permute(w, n_heads_arg, dim1=None, dim2=None):
    """Copy of the reverse_permute function from Llama3StateDictAdapter"""
    if dim1 is None:
        dim1 = w.shape[0]
    if dim2 is None:
        dim2 = w.shape[1]
    return (
        w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
        .transpose(1, 2)
        .reshape(dim1, dim2)
    )

def test_permute_functions():
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    # Create device mesh with ['dp_shard', 'tp'], [2, 2]
    mesh_shape = [2, 2]  # [dp_shard, tp]
    device_mesh = DeviceMesh("cuda", torch.arange(4).reshape(mesh_shape))
    
    # Define model dimensions
    dim = 16
    n_heads = 4  # TP degree <= n_heads
    
    # Test 1: Query projection matrix (dim, dim)
    # Create a test tensor for query projection
    tensor_shape = [dim, dim]
    torch.manual_seed(0)
    local_tensor = torch.randn(tensor_shape, device="cuda")
    print(f"Original tensor: {local_tensor}")
    
    # Create distributed tensor with specified sharding: _StridedShard(dim=0, sf=2), Shard(dim=0)
    placements = [_StridedShard(dim=0, split_factor=2), Shard(dim=0)]
    dist_tensor = torch.distributed._tensor.distribute_tensor(
        local_tensor, device_mesh, placements
    )
    
    print(f"Original tensor shape: {dist_tensor.shape}, placements: {dist_tensor.placements}, device_mesh: {dist_tensor.device_mesh}")
    # Test _permute function
    permuted_tensor = _permute(dist_tensor, n_heads)
    
    print(f"After _permute:")
    print(f"Permuted tensor shape: {permuted_tensor.shape}, placements: {permuted_tensor.placements}, device_mesh: {permuted_tensor.device_mesh}")

    
    # Verify that _reverse_permute is the inverse of _permute
    original_full_permuted = _permute(local_tensor, n_heads)
    permuted_full = permuted_tensor.full_tensor()
    print(f"Original full tensor after permutation: {original_full_permuted}")
    print(f"Permuted Dtensor after full_tensor: {permuted_full}")
    
    are_tensors_equal = torch.allclose(
        original_full_permuted, 
        permuted_full,
        rtol=1e-5, 
        atol=1e-5
    )
    
    print(f"Are original and reverse-permuted tensors equal? {are_tensors_equal}")


def test_view_functions():
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    # Create device mesh with ['dp_shard', 'tp'], [2, 2]
    mesh_shape = [2, 2]  # [dp_shard, tp]
    device_mesh = DeviceMesh("cuda", torch.arange(4).reshape(mesh_shape))
    
    # Define model dimensions
    dim = 16
    n_heads = 4  # TP degree <= n_heads
    
    # Test 1: Query projection matrix (dim, dim)
    # Create a test tensor for query projection
    tensor_shape = [dim, dim]
    torch.manual_seed(0)
    local_tensor = torch.randn(tensor_shape, device="cuda")
    print(f"Original tensor: {local_tensor}")
    
    # Create distributed tensor with specified sharding: _StridedShard(dim=0, sf=2), Shard(dim=0)
    placements = [_StridedShard(dim=0, split_factor=2), Shard(dim=0)]
    dist_tensor = torch.distributed._tensor.distribute_tensor(
        local_tensor, device_mesh, placements
    )
    
    print(f"Original tensor shape: {dist_tensor.shape}, placements: {dist_tensor.placements}, device_mesh: {dist_tensor.device_mesh}")
    # Test _view_test function
    permuted_tensor = _view_test(dist_tensor, n_heads)
    
    print(f"After _view_test:")
    print(f"After _view_test tensor shape: {permuted_tensor.shape}, placements: {permuted_tensor.placements}, device_mesh: {permuted_tensor.device_mesh}")

    
    # Verify that _reverse_permute is the inverse of _permute
    original_full_permuted = _view_test(local_tensor, n_heads)
    permuted_full = permuted_tensor.full_tensor()
    print(f"Original full tensor after _view_test: {original_full_permuted}")
    print(f"Viewed Dtensor after full_tensor: {permuted_full}")
    
    are_tensors_equal = torch.allclose(
        original_full_permuted, 
        permuted_full,
        rtol=1e-5, 
        atol=1e-5
    )
    
    print(f"Are original and viewed tensors equal? {are_tensors_equal}")
    
if __name__ == "__main__":
    # test_permute_functions()
    test_view_functions()
