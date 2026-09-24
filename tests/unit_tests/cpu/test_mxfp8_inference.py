# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("torchao")

from torchtitan.quantization.mxfp8.inference import (  # noqa: E402
    _MXFP8StorageTensor,
    _swizzle_scale,
    _unswizzle_scale,
)


def _storage(shape, *, offset=0):
    data = (torch.arange(torch.tensor(shape).prod()).reshape(shape) % 16 - 8).float()
    tile_shape = (*shape[:-2], shape[-2] // 32, shape[-1] // 32)
    scales = (
        (
            torch.arange(torch.tensor(tile_shape).prod()).reshape(tile_shape) % 7
            + 120
            + offset
        )
        .to(torch.uint8)
        .repeat_interleave(32, dim=-2)
    )
    return _MXFP8StorageTensor(
        data.to(torch.float8_e4m3fn), scales.view(torch.float8_e8m0fnu)
    )


@pytest.mark.parametrize("shape", [(128, 128), (160, 96), (2, 128, 64)])
def test_mxfp8_storage_allocates_only_data_and_scales(shape):
    with torch.device("meta"):
        weight = _MXFP8StorageTensor.from_bf16(torch.empty(shape, dtype=torch.bfloat16))
        module = torch.nn.Module()
        module.register_parameter(
            "weight", torch.nn.Parameter(weight, requires_grad=False)
        )
    module.to_empty(device="cpu")
    weight = module.weight
    assert isinstance(weight, _MXFP8StorageTensor)
    assert weight.dtype == torch.bfloat16
    assert weight._qdata.dtype == torch.float8_e4m3fn
    assert weight._scale.dtype == torch.float8_e8m0fnu
    assert weight._qdata.numel() + weight._scale.numel() == weight.numel() * 33 // 32
    assert not weight.requires_grad


@pytest.mark.parametrize("rows,cols", [(32, 1), (128, 4), (160, 5), (256, 12)])
def test_mxfp8_scale_layout_matches_scaled_mm(rows, cols):
    scale = (torch.arange(rows * cols).reshape(rows, cols) % 255).byte()
    packed = _swizzle_scale(scale).view(torch.uint8).flatten()
    row = torch.arange(rows)[:, None]
    col = torch.arange(cols)[None, :]
    index = (
        ((row // 128) * ((cols + 3) // 4) + col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + col % 4
    )
    torch.testing.assert_close(packed[index], scale, rtol=0, atol=0)
    torch.testing.assert_close(
        _unswizzle_scale(packed, rows, cols).view(torch.uint8), scale, rtol=0, atol=0
    )


@pytest.mark.parametrize("shape,shard_dim", [((128, 96), 0), ((2, 128, 64), 1)])
def test_mxfp8_all_gather_only_transports_bytes_and_preserves_compute_buffers(
    shape, shard_dim
):
    weight = _storage(shape)
    mesh = SimpleNamespace(size=lambda: 2)
    policy = SimpleNamespace(param_dtype=torch.bfloat16)
    shards = weight.chunk(2, dim=shard_dim)
    inputs = [
        shard.fsdp_pre_all_gather(mesh, weight.shape, weight.stride(), None, policy)
        for shard in shards
    ]
    assert all(t.dtype == torch.uint8 for tensors, _ in inputs for t in tensors)
    gathered = (torch.cat([entry[0][0] for entry in inputs], dim=0),)
    metadata = inputs[0][1]
    unsharded, buffers = shards[0].fsdp_post_all_gather(
        gathered, metadata, torch.bfloat16
    )
    assert unsharded.shape == weight.shape
    assert unsharded.dtype == torch.bfloat16
    operands = unsharded.operands
    torch.testing.assert_close(
        operands.weight_qdata_dgrad_NK.view(torch.uint8),
        weight._qdata.flatten(0, -2).view(torch.uint8),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        _unswizzle_scale(
            operands.weight_scale_fprop_swizzled,
            weight.numel() // shape[-1],
            shape[-1] // 32,
        ).view(torch.uint8),
        weight._scale.flatten(0, -2).view(torch.uint8),
        rtol=0,
        atol=0,
    )
    identities = [(id(t), t.data_ptr()) for t in buffers]
    changed = _storage(shape, offset=3)
    updated_inputs = [
        shard.fsdp_pre_all_gather(mesh, weight.shape, weight.stride(), None, policy)[0]
        for shard in changed.chunk(2, dim=shard_dim)
    ]
    updated = (torch.cat([entry[0] for entry in updated_inputs], dim=0),)
    shards[0].fsdp_post_all_gather(updated, metadata, torch.bfloat16, out=unsharded)
    assert [(id(t), t.data_ptr()) for t in buffers] == identities
    torch.testing.assert_close(
        _unswizzle_scale(
            operands.weight_scale_fprop_swizzled,
            weight.numel() // shape[-1],
            shape[-1] // 32,
        ).view(torch.uint8),
        changed._scale.flatten(0, -2).view(torch.uint8),
        rtol=0,
        atol=0,
    )


def test_mxfp8_fsdp_storage_views_and_copies_preserve_scale_pairing():
    weight = _storage((2, 128, 64))
    shard = weight.chunk(2, dim=1)[1]
    target = weight.new_zeros(shard.shape)
    target.copy_(shard)
    flat = target.view(-1)
    restored = flat.as_strided(target.shape, target.stride())
    for result in (target, restored, target.clone()):
        torch.testing.assert_close(
            result.dequantize(), shard.dequantize(), rtol=0, atol=0
        )


def test_mxfp8_storage_rejects_incomplete_scale_groups():
    weight = _storage((128, 64))
    with pytest.raises(ValueError, match="32-element scale groups"):
        weight[:, :16]
    with pytest.raises(ValueError, match="even sharding"):
        weight[:64].fsdp_pre_all_gather(
            SimpleNamespace(size=lambda: 3),
            weight.shape,
            weight.stride(),
            None,
            SimpleNamespace(param_dtype=torch.bfloat16),
        )
