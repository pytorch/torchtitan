# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.nn.attention.flex_attention import and_masks, create_block_mask

from torchtitan.models.common.attention import (
    get_causal_mask_mod,
    get_efficient_causal_mask_mod_for_packed_document,
    get_sliding_window_mask_mod,
)
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.gpt_oss import _make_gptoss_attn_config


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _attention(n_heads, n_kv_heads):
    # Exercise the default GPT-OSS backend and its actual sink epilogue.
    return (
        _make_gptoss_attn_config(
            dim=16,
            layer_id=0,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=64,
            rope=CosSinRoPE.Config(dim=64, max_context_length=128),
        )
        .build()
        .cuda()
    )


def _explicit_sink(q, k, v, sinks, lengths, window):
    outputs = []
    n_rep = q.shape[1] // k.shape[1]
    for qi, ki, vi in zip(q.split(lengths), k.split(lengths), v.split(lengths)):
        qi = qi.transpose(0, 1)
        ki = ki.repeat_interleave(n_rep, dim=1).transpose(0, 1)
        vi = vi.repeat_interleave(n_rep, dim=1).transpose(0, 1)
        scores = (qi @ ki.transpose(-1, -2)) / q.shape[-1] ** 0.5
        pos = torch.arange(qi.shape[1], device=q.device)
        allowed = pos[:, None] >= pos[None, :]
        if window is not None:
            allowed &= pos[:, None] - pos[None, :] < window
        scores = scores.masked_fill(~allowed, -torch.inf)
        scores = torch.cat(
            (scores, sinks[:, None, None].expand(-1, qi.shape[1], 1)), dim=-1
        )
        # The additional sink has value zero and is always visible.
        outputs.append((scores.softmax(dim=-1)[..., :-1] @ vi).transpose(0, 1))
    return torch.cat(outputs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("window", [None, 4])
@pytest.mark.parametrize("first_tokens_only", [False, True])
def test_packed_sink_output_and_all_gradients(dtype, window, first_tokens_only):
    torch.manual_seed(42)
    lengths = (1, 5, 122)
    positions = torch.cat([torch.arange(n, device="cuda") for n in lengths])
    attention = _attention(n_heads=4, n_kv_heads=2)
    with torch.no_grad():
        attention.sinks.copy_(torch.tensor([-0.5, 0.0, 0.5, 1.0], device="cuda"))
    q = torch.randn(128, 4, 64, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(128, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(128, 2, 64, device="cuda", dtype=dtype, requires_grad=True)
    mask_mods = [
        get_causal_mask_mod(),
        get_efficient_causal_mask_mod_for_packed_document(positions),
    ]
    if window is not None:
        mask_mods.append(get_sliding_window_mask_mod(window))
    mask = create_block_mask(and_masks(*mask_mods), 1, None, 128, 128, device="cuda")
    actual = attention.inner_attention(
        q,
        k,
        v,
        attention_masks=mask,
        enable_gqa=True,
        out_transform=attention._apply_sinks,
    )
    inputs = (q, k, v, attention.sinks)
    reference_inputs = [t.detach().double().requires_grad_() for t in inputs]
    expected = _explicit_sink(*reference_inputs, lengths, window)
    grad = torch.randn_like(actual)
    if first_tokens_only:
        # Isolate each document start: its entire Q/K gradient comes from LSE.
        grad = grad * (positions == 0)[:, None, None]
    actual_grads = torch.autograd.grad(actual, inputs, grad)
    expected_grads = torch.autograd.grad(expected, reference_inputs, grad.double())

    atol, rtol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (2e-5, 2e-5)
    torch.testing.assert_close(actual.double(), expected, atol=atol, rtol=rtol)
    for name, got, want in zip(("Q", "K", "V", "sink"), actual_grads, expected_grads):
        torch.testing.assert_close(
            got.double(), want, atol=atol, rtol=rtol, msg=lambda msg: f"{name}: {msg}"
        )
    # A loose BF16 tolerance must not hide a completely missing first-token term.
    if first_tokens_only:
        for got, want in zip(actual_grads[:2], expected_grads[:2]):
            relative_error = (got.double() - want).norm() / want.norm()
            assert relative_error < 0.02
