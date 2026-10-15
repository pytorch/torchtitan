# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""vLLM's MLA reads a quantized wkv_b as the weight its quantized GEMM uses."""

import pytest
import torch

pytest.importorskip("torchao")
if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (10, 0):
    pytest.skip("MXFP8 and NVFP4 linears require SM100+", allow_module_level=True)

from torchao.prototype.mx_formats.kernels import f4_unpacked_to_f32, unpack_uint4

from torchtitan.quantization.mxfp8.linear import MXFP8Linear
from torchtitan.quantization.mxfp8.tensor import _quantize_mxfp8_weight
from torchtitan.quantization.nvfp4.linear import NVFP4Linear
from torchtitan.quantization.nvfp4.tensor import _quantize_nvfp4_weight
from torchtitan.rl.generator.vllm.mla import _VLLMKVBProj

# A DeepSeek-V3-shaped wkv_b shard: kv_lora_rank -> 16 heads x (128 + 128).
IN_FEATURES = 512
OUT_FEATURES = 16 * 256


@pytest.mark.parametrize("std, num_outliers", [(0.02, 0), (30.0, 0), (0.001, 512)])
def test_mxfp8_kv_b_proj_weight_is_the_rounded_weight(std, num_outliers):
    torch.manual_seed(0)
    with torch.device("cuda"):
        torch.set_default_dtype(torch.bfloat16)
        wkv_b = MXFP8Linear.Config(
            in_features=IN_FEATURES, out_features=OUT_FEATURES
        ).build()
        torch.set_default_dtype(torch.float32)
        weight = torch.randn(OUT_FEATURES, IN_FEATURES) * std
        weight.view(-1)[torch.randint(0, weight.numel(), (num_outliers,))] *= 1000.0
    weight = weight.to(torch.bfloat16)
    with torch.no_grad():
        wkv_b.weight._tensor.copy_(weight)

    # MXFP8 weights use one power-of-two (E8M0, rounded up) scale per 32x32 tile.
    tiles = weight.float().reshape(OUT_FEATURES // 32, 32, IN_FEATURES // 32, 32)
    tile_amax = tiles.abs().amax(dim=(1, 3), keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(tile_amax / 448.0)))
    qdata = (tiles / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    torch.testing.assert_close(
        qdata.reshape(OUT_FEATURES, IN_FEATURES).float(),
        _quantize_mxfp8_weight(weight).weight_qdata_dgrad_NK.float(),
        rtol=0,
        atol=0,
    )
    rounded_weight = (qdata.float() * scale).reshape(OUT_FEATURES, IN_FEATURES)

    kv_b_proj = _VLLMKVBProj(wkv_b)
    kv_b_proj.refresh_weight(torch.bfloat16)
    torch.testing.assert_close(
        kv_b_proj.weight, rounded_weight.to(torch.bfloat16), rtol=0, atol=0
    )
    x = torch.randn(64, IN_FEATURES, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        torch.testing.assert_close(kv_b_proj(x)[0], wkv_b(x), rtol=0, atol=0)


@pytest.mark.parametrize("std, num_outliers", [(0.02, 0), (30.0, 0), (0.001, 512)])
def test_nvfp4_kv_b_proj_weight_is_the_rounded_weight(std, num_outliers):
    torch.manual_seed(0)
    with torch.device("cuda"):
        torch.set_default_dtype(torch.bfloat16)
        wkv_b = NVFP4Linear.Config(
            in_features=IN_FEATURES, out_features=OUT_FEATURES
        ).build()
        torch.set_default_dtype(torch.float32)
        weight = torch.randn(OUT_FEATURES, IN_FEATURES) * std
        weight.view(-1)[torch.randint(0, weight.numel(), (num_outliers,))] *= 1000.0
    weight = weight.to(torch.bfloat16)
    with torch.no_grad():
        wkv_b.weight._tensor.copy_(weight)
        wkv_b._init_self_buffers(buffer_device=torch.device("cuda"))

    # NVFP4 weights use an FP8 scale per 16x16 tile times one tensor-wide scale,
    # with values rounded to nearest-even on the E2M1 grid.
    amax = weight.float().abs().max()
    encode = (448.0 * 6.0) / amax
    tiles = weight.float().reshape(OUT_FEATURES // 16, 16, IN_FEATURES // 16, 16)
    tile_amax = tiles.abs().amax(dim=(1, 3), keepdim=True)
    block_scale = torch.clamp(tile_amax * (encode * (1.0 / 6.0)), max=448.0)
    block_scale = block_scale.to(torch.float8_e4m3fn).float()
    scaled = (tiles / (block_scale * (1.0 / encode))).clamp(-6.0, 6.0)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device="cuda")
    distance = (scaled.abs().unsqueeze(-1) - grid).abs()
    index = distance.argmin(dim=-1)
    is_tie = (distance == distance.gather(-1, index.unsqueeze(-1))).sum(-1) > 1
    index = torch.where(is_tie & (index % 2 == 1), index + 1, index)
    codes = torch.sign(scaled) * grid[index]
    torch.testing.assert_close(
        codes.reshape(OUT_FEATURES, IN_FEATURES),
        f4_unpacked_to_f32(
            unpack_uint4(
                _quantize_nvfp4_weight(weight).weight_qdata_fprop.view(torch.uint8)
            )
        ).reshape(OUT_FEATURES, IN_FEATURES),
        rtol=0,
        atol=0,
    )
    rounded_weight = (codes * block_scale * (amax / (448.0 * 6.0))).reshape(
        OUT_FEATURES, IN_FEATURES
    )

    kv_b_proj = _VLLMKVBProj(wkv_b)
    kv_b_proj.refresh_weight(torch.bfloat16)
    torch.testing.assert_close(
        kv_b_proj.weight, rounded_weight.to(torch.bfloat16), rtol=0, atol=0
    )
    x = torch.randn(128, IN_FEATURES, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        torch.testing.assert_close(kv_b_proj(x)[0], wkv_b(x), rtol=0, atol=0)
