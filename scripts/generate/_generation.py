# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.nn.attention import sdpa_kernel, SDPBackend


def multinomial_sample_one(
    probs: torch.Tensor, rng: torch.Generator | None = None
) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1, generator=rng)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def generate_next_token(
    model,
    x: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    logits = model(x)  # (B, T, vocab_size)
    probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
    next_token = multinomial_sample_one(probs, rng=rng)
    return next_token


@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    seed: int | None = None,
) -> torch.Tensor:
    # ensure batch dimension (T,) --> (B, T)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    rng = None
    if seed is not None:
        rng = torch.Generator(input_ids.device).manual_seed(seed)

    generated_tokens = input_ids.clone()

    # Inference can hit shapes that ``FLASH_ATTENTION`` / ``CUDNN_ATTENTION``
    # refuse (e.g., debug-model ``head_dim=16`` is outside flash's supported
    # set), and the trainer-side ``ScaledDotProductAttention`` defaults
    # intentionally exclude ``MATH``. Allow ``MATH`` as a last-resort fallback
    # here so generation works on more hardware/configs without affecting
    # training-side strictness.
    backends = [
        SDPBackend.CUDNN_ATTENTION,
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ]
    with sdpa_kernel(backends, set_priority=True):
        for _ in range(max_new_tokens):
            next_token = generate_next_token(
                model,
                x=generated_tokens,
                temperature=temperature,
                top_k=top_k,
                rng=rng,
            )

            generated_tokens = torch.cat([generated_tokens, next_token], dim=1)

    return generated_tokens
