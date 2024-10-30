# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch


def multinomial_sample_one(probs: torch.Tensor) -> torch.Tensor:
    q = torch.empty_like(probs).exponential_(1)
    return torch.argmax(probs / q, dim=-1, keepdim=True).to(dtype=torch.long)


def logits_to_probs(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:

    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))  # (k,)
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)  # (1,)
        logits = torch.where(logits < pivot, -float("Inf"), logits)  # (vocab_size, )

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def generate_next_token(
    model,
    x: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    logits = model(x)  # (B, T, vocab_size)
    probs = logits_to_probs(logits[:, -1, :], temperature, top_k)
    next_token = multinomial_sample_one(probs)
    return next_token, probs


@torch.no_grad()
def generate(
    model,
    prompt: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    if prompt.ndim == 1:
        prompt = prompt.unsqueeze(0)

    generated_tokens = prompt.clone()

    for i in range(max_new_tokens):

        tokens, logits = generate_next_token(
            model,
            x=generated_tokens.clone(),
            temperature=temperature,
            top_k=top_k,
        )

        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    return generated_tokens, logits
