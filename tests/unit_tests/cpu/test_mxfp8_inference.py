# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import pytest
import torch


pytest.importorskip("torchao")

from torchtitan.quantization._fsdp_tensor import _unsharded_inner_tensors  # noqa: E402
from torchtitan.quantization.mxfp8.inference import (  # noqa: E402
    _MXFP8StorageTensor,
    _swizzle_scale,
    _unswizzle_scale,
    MXFP8InferenceLinear,
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
    scales = scales.flatten(0, -2)
    return _MXFP8StorageTensor(
        data.flatten(0, -2).to(torch.float8_e4m3fn),
        _swizzle_scale(scales),
        _swizzle_scale(scales[::32].t().repeat_interleave(32, 0)),
        shape=data.shape,
        stride=data.stride(),
    )


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


@pytest.mark.parametrize("dp_shard,cp", [(2, 1), (1, 2)])
def test_mxfp8_inference_rejects_multiple_fsdp_ranks(dp_shard, cp):
    with torch.device("meta"):
        linear = MXFP8InferenceLinear.Config(in_features=64, out_features=128).build()
    with pytest.raises(ValueError, match="requires FSDP degree 1"):
        linear._parallelize(SimpleNamespace(dp_shard=dp_shard, cp=cp))
    weight = _storage((128, 64))
    with pytest.raises(ValueError, match="requires FSDP degree 1"):
        weight.fsdp_get_unsharded_view(
            SimpleNamespace(size=lambda: dp_shard * cp),
            None,
            SimpleNamespace(param_dtype=torch.bfloat16),
        )


@pytest.mark.parametrize("shape", [(128, 64), (160, 96), (2, 128, 64)])
def test_mxfp8_storage_materializes_and_shares_compute_views(shape):
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
    assert not weight.requires_grad
    assert weight._qdata.numel() == weight.numel()
    buffers = _unsharded_inner_tensors(weight.operands)
    pointers = tuple(t.data_ptr() for t in buffers)
    for view in (weight.detach(), weight.view(-1).as_strided(shape, weight.stride())):
        assert (
            tuple(t.data_ptr() for t in _unsharded_inner_tensors(view.operands))
            == pointers
        )
    compute = weight.fsdp_get_unsharded_view(
        SimpleNamespace(size=lambda: 1),
        None,
        SimpleNamespace(param_dtype=torch.bfloat16),
    )
    assert compute.shape == weight.shape
    assert (
        tuple(t.data_ptr() for t in _unsharded_inner_tensors(compute.operands))
        == pointers
    )
    assert [t.dtype for t in buffers] == [
        torch.float8_e4m3fn,
        torch.float8_e8m0fnu,
        torch.float8_e8m0fnu,
    ]
    assert all(
        t.data_ptr() not in pointers
        for t in _unsharded_inner_tensors(weight.clone().operands)
    )
    with pytest.raises(ValueError, match="complete TP-local weight"):
        weight[..., :32, :]


def test_mxfp8_bf16_load_updates_existing_compute_buffers(monkeypatch):
    import torchtitan.quantization.mxfp8.inference as inference

    with torch.device("meta"):
        weight = _MXFP8StorageTensor.from_bf16(
            torch.empty((128, 64), dtype=torch.bfloat16)
        )
    weight = torch.empty_like(weight, device="cpu")
    compute = weight.fsdp_get_unsharded_view(
        SimpleNamespace(size=lambda: 1),
        None,
        SimpleNamespace(param_dtype=torch.bfloat16),
    )
    buffers = _unsharded_inner_tensors(compute.operands)
    pointers = tuple(t.data_ptr() for t in buffers)
    for offset in (1, 3):
        source = _storage((128, 64), offset=offset)
        expected = source.operands
        monkeypatch.setattr(inference, "_quantize_mxfp8_weight", lambda _: expected)
        weight.copy_(source.dequantize())
        for actual, reference in zip(
            buffers, _unsharded_inner_tensors(expected), strict=True
        ):
            torch.testing.assert_close(
                actual.view(torch.uint8), reference.view(torch.uint8), rtol=0, atol=0
            )
        assert tuple(t.data_ptr() for t in buffers) == pointers
        torch.testing.assert_close(
            weight.dequantize(), source.dequantize(), rtol=0, atol=0
        )


def test_mxfp8_storage_copies_and_full_views_preserve_data_and_scales():
    weight = _storage((2, 128, 64))
    target = weight.new_zeros(weight.shape)
    target.copy_(weight)
    restored = target.view(-1).as_strided(target.shape, target.stride())
    for result in (target, restored, target.clone()):
        torch.testing.assert_close(
            result.dequantize(), weight.dequantize(), rtol=0, atol=0
        )
