# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GatedDeltaNet packed-sequence correctness.

When several samples are packed into one row (``positions`` restart at 0 per
sample), GatedDeltaNet is given ``cu_seqlens`` marking the boundaries so its
recurrent state and causal conv reset per sample. These tests check that the
boundaries are derived correctly and that a packed row processed with
``cu_seqlens`` matches processing each sample on its own.

The end-to-end test needs a GPU: the FLA gated-delta kernels are Triton/CUDA only.
"""

from typing import Literal

import pytest
import torch

from torchtitan.models.common.attention import create_varlen_metadata_for_document
from torchtitan.models.qwen3_5 import _qwen35_deltanet_config


def _max_abs_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    return (x - y).abs().max().item()


def test_packed_documents_cu_seqlens():
    # Two documents packed in one row (positions restart at 0 at the boundary);
    # these boundaries are what GatedDeltaNet resets its state/conv on.
    positions = torch.cat([torch.arange(30), torch.arange(40)]).unsqueeze(0)
    meta = create_varlen_metadata_for_document(positions)
    assert meta.cu_seq_q.tolist() == [0, 30, 70]
    assert meta.max_q == 40

    # One sample per batch row is not packing and stays on the batched path.
    positions = torch.arange(30).expand(2, -1)
    meta = create_varlen_metadata_for_document(positions)
    assert meta.cu_seq_q.tolist() == [0, 30, 60]
    assert meta.cu_seq_q.numel() == positions.shape[0] + 1


@pytest.mark.parametrize(
    ("backend", "device", "dtype"),
    [
        pytest.param(
            "fla_chunked",
            "cuda",
            torch.bfloat16,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="FLA conv needs CUDA"
            ),
            id="fla_chunked",
        ),
        pytest.param("torch_native", "cpu", torch.float32, id="torch_native"),
    ],
)
def test_causal_conv1d_packing_matches_separate(
    backend: Literal["fla_chunked", "torch_native"],
    device: Literal["cpu", "cuda"],
    dtype: torch.dtype,
):
    torch.manual_seed(0)
    dim = 256
    cfg = _qwen35_deltanet_config(
        dim=dim,
        n_key_heads=2,
        n_value_heads=4,
        key_head_dim=64,
        value_head_dim=64,
        layer_id=0,
        fla_backend=backend,
    )
    gdn = cfg.build().to(device=device, dtype=dtype)
    with torch.no_grad():
        gdn.conv_q.weight.normal_(0, 0.02)

    len_a, len_b = 30, 40
    channels = gdn.conv_q.in_channels
    x_a = torch.randn(1, len_a, channels, device=device, dtype=dtype)
    x_b = torch.randn(1, len_b, channels, device=device, dtype=dtype)
    x_packed = torch.cat([x_a, x_b], dim=1)
    x_a.requires_grad_()
    x_b.requires_grad_()
    x_packed.requires_grad_()
    positions = torch.cat(
        [torch.arange(len_a, device=device), torch.arange(len_b, device=device)]
    ).unsqueeze(0)
    cu_seqlens = create_varlen_metadata_for_document(positions).cu_seq_q

    out_a = gdn._causal_conv1d(x_a, gdn.conv_q)
    out_b = gdn._causal_conv1d(x_b, gdn.conv_q)
    out_packed = gdn._causal_conv1d(x_packed, gdn.conv_q, cu_seqlens)

    torch.testing.assert_close(out_packed[:, :len_a], out_a, rtol=0, atol=0)
    torch.testing.assert_close(out_packed[:, len_a:], out_b, rtol=0, atol=0)

    grad_a = torch.randn_like(out_a)
    grad_b = torch.randn_like(out_b)
    grad_packed = torch.cat([grad_a, grad_b], dim=1)
    grad_x_packed, grad_w_packed = torch.autograd.grad(
        out_packed, (x_packed, gdn.conv_q.weight), grad_packed
    )
    grad_x_a, grad_w_a = torch.autograd.grad(out_a, (x_a, gdn.conv_q.weight), grad_a)
    grad_x_b, grad_w_b = torch.autograd.grad(out_b, (x_b, gdn.conv_q.weight), grad_b)
    tol = torch.finfo(dtype).eps
    torch.testing.assert_close(grad_x_packed[:, :len_a], grad_x_a, rtol=tol, atol=tol)
    torch.testing.assert_close(grad_x_packed[:, len_a:], grad_x_b, rtol=tol, atol=tol)
    weight_tol = 16 * tol
    torch.testing.assert_close(
        grad_w_packed, grad_w_a + grad_w_b, rtol=weight_tol, atol=weight_tol
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA GDN kernels need CUDA")
def test_gated_delta_net_packing_matches_separate():
    torch.manual_seed(0)
    dev, dt, dim = "cuda", torch.bfloat16, 256
    cfg = _qwen35_deltanet_config(
        dim=dim,
        n_key_heads=2,
        n_value_heads=4,
        key_head_dim=64,
        value_head_dim=64,
        layer_id=0,
    )
    gdn = cfg.build().to(dev)
    with torch.no_grad():
        for name, param in gdn.named_parameters():
            if "A_log" in name or "dt_bias" in name:
                param.zero_()
            elif "norm" in name and "weight" in name:
                param.fill_(1.0)
            else:
                param.normal_(0, 0.02)
    gdn = gdn.to(dt)

    len_a, len_b = 30, 40
    x_a = torch.randn(1, len_a, dim, device=dev, dtype=dt)
    x_b = torch.randn(1, len_b, dim, device=dev, dtype=dt)
    x_packed = torch.cat([x_a, x_b], dim=1)
    # Packed positions restart at 0 for the second sample.
    positions = torch.cat(
        [torch.arange(len_a, device=dev), torch.arange(len_b, device=dev)]
    ).unsqueeze(0)
    cu_seqlens = create_varlen_metadata_for_document(positions).cu_seq_q

    with torch.no_grad():
        out_a = gdn(x_a)  # each sample on its own (no packing -> cu_seqlens None)
        out_b = gdn(x_b)
        out_packed = gdn(x_packed, cu_seqlens)

    # Each packed sample matches its standalone output (to bf16 tolerance).
    assert _max_abs_diff(out_packed[:, :len_a], out_a) < 1e-3
    assert _max_abs_diff(out_packed[:, len_a:], out_b) < 1e-3
