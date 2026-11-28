# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.experiments.inkling import model


def _old_eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    rel: torch.Tensor,
    rel_proj: torch.Tensor,
    sinks: torch.Tensor,
    *,
    window: int | None,
) -> torch.Tensor:
    batch, tokens, heads, width = q.shape
    group = heads // k.size(2)
    scores = torch.einsum(
        "btngd,bsnd->bngts", q.unflatten(2, (k.size(2), group)), k
    ).reshape(batch, heads, tokens, tokens)
    scores = scores * width**-0.5
    offsets = torch.arange(tokens, device=q.device)
    offsets = offsets[:, None] - offsets
    extent = rel_proj.size(-1)
    table = (rel @ rel_proj).transpose(1, 2)
    bias = table[
        :,
        :,
        torch.arange(tokens, device=q.device)[:, None],
        offsets.clamp(0, extent - 1),
    ]
    scores = scores + bias.masked_fill((offsets < 0) | (offsets >= extent), 0)
    valid = offsets >= 0
    if window is not None:
        valid = valid & (offsets < window)
    scores = scores.masked_fill(~valid, -torch.inf)
    # Represent the sink literally as one extra key with zero value.
    scores = torch.cat((scores, sinks[..., None]), dim=-1)
    probs = scores.to(torch.float64 if q.dtype == torch.float64 else torch.float32)
    probs = probs.softmax(-1).to(q.dtype).unflatten(1, (k.size(2), group))
    v = torch.cat((v, torch.zeros_like(v[:, :1])), dim=1)
    return torch.einsum("bngts,bsnd->bngtd", probs, v).flatten(1, 2)


@pytest.mark.parametrize("window", [None, 37, 128, 193])
def test_inkling_block_mask_has_exact_visibility(window: int | None) -> None:
    tokens = 259
    mask = model._inkling_block_mask(tokens, window, torch.device("cpu"))
    block_size = mask.BLOCK_SIZE[0]
    pos = torch.arange(tokens)
    q_pos, kv_pos = pos[:, None], pos[None, :]
    row, col = q_pos // block_size, kv_pos // block_size
    present = mask.to_dense()[0, 0].bool()[row, col]
    full_cols = mask.full_kv_indices[0, 0]
    full_counts = mask.full_kv_num_blocks[0, 0]
    block_ids = torch.arange(full_cols.size(-1))
    full = (
        (full_cols[..., None] == block_ids)
        & (block_ids < full_counts[:, None])[..., None]
    ).any(dim=1)[row, col]
    visible = present & (full | mask.mask_mod(0, 0, q_pos, kv_pos))
    expected = q_pos >= kv_pos
    if window is not None:
        expected &= (q_pos - kv_pos) < window
    torch.testing.assert_close(visible, expected)


@pytest.mark.parametrize(
    ("window", "width"),
    [(None, 128), (47, 128), (None, 8)],
    ids=["full", "local", "debug"],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inkling_attention_matches_old_eager_forward_and_backward(
    device: str, window: int | None, width: int
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the compiled FlexAttention path")
    generator = torch.Generator().manual_seed(42)
    batch, tokens, heads, kv_heads, rel_width = 2, 137, 6, 2, 4
    extent = 73 if window is None else window
    if device == "cpu":
        # The CPU oracle also exercises slicing unused projection columns.
        tokens = 13

    def sample(*shape: int, scale: float) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * scale).bfloat16().to(device)

    q, k, v, rel = (
        sample(batch, tokens, heads, width, scale=0.6),
        sample(batch, tokens, kv_heads, width, scale=0.6),
        sample(batch, tokens, kv_heads, width, scale=0.7),
        sample(batch, tokens, heads, rel_width, scale=0.35),
    )
    proj = sample(rel_width, extent, scale=0.3)
    # Keep the sink relevant in early and late rows, with distinct head logits.
    sinks = (
        (
            sample(batch, tokens, heads, scale=0.8).float()
            + torch.linspace(-1, 5, tokens).to(device)[:, None]
        )
        .bfloat16()
        .transpose(1, 2)
    )
    grad = sample(batch, heads, tokens, width, scale=0.5)

    def evaluate(attention, dtype):
        inputs = [x.to(dtype).detach().requires_grad_() for x in (q, k, v, rel, sinks)]
        qry, key, value, relative, sink = inputs
        context = attention(
            qry, key, value, relative, proj.to(dtype), sink, window=window
        )
        return (context.detach(), *torch.autograd.grad(context, inputs, grad.to(dtype)))

    if device == "cuda":
        model._inkling_block_mask.cache_clear()
        with torch.inference_mode():
            model._inkling_attention(q, k, v, rel, proj, sinks, window=window)
    actual = evaluate(model._inkling_attention, torch.bfloat16)
    eager = evaluate(_old_eager_attention, torch.bfloat16)
    exact = evaluate(_old_eager_attention, torch.float64)
    # Allow a few extra BF16 rounding steps for the reordered softmax, sink
    # scaling and its backward. The RMS row scale supplies an absolute floor
    # when an otherwise large row nearly cancels in the softmax backward.
    rounding = 3 * torch.finfo(torch.bfloat16).eps
    diagnostics = []
    for name, result, baseline, expected in zip(
        (
            "context",
            "query grad",
            "key grad",
            "value grad",
            "relative grad",
            "sink grad",
        ),
        actual,
        eager,
        exact,
    ):
        result_error = (result.double() - expected).abs().amax(dim=-1)
        eager_error = (baseline.double() - expected).abs().amax(dim=-1)
        row_scale = expected.abs().amax(dim=-1)
        allowed = eager_error + rounding * (
            row_scale + row_scale.square().mean().sqrt()
        )
        if not (result_error <= allowed).all():
            diagnostics.append(
                f"{name}: max error {result_error.max().item():.6g}, "
                f"max excess {(result_error - allowed).max().item():.6g}"
            )
    assert not diagnostics, "\n".join(diagnostics)
    # With only one real key, the first query and relative bias gradients can
    # only flow through the LSE term in the sink rescaling.
    assert actual[1][:, 0].count_nonzero()
    assert actual[-2][:, 0].count_nonzero() and actual[-1].count_nonzero()
