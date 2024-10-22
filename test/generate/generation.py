# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional, Tuple

import torch


def sample(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """Sample from a probability distribution

    Args:
        logits (torch.Tensor): logits from which to sample (vocab_size,)
        temperature (float): value to scale logits by, default 1.0.
        top_k (Optional[int]): if specified, prune sampling to only tokens within the top_k probs.

    Returns:
        torch.Tensor: sampled token id
    """

    # scale
    logits = logits / max(temperature, 1e-5)

    # top-k
    if top_k is not None:
        v, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))  # (k,)
        # select last value from top_k above as the pivot
        pivot = v.select(dim=-1, index=-1).unsqueeze(-1)  # (1,)
        # mask values smaller than pivot to -inf since these should be pruned
        logits = torch.where(logits < pivot, -float("Inf"), logits)  # (vocab_size, )

    # normalize
    probs = torch.nn.functional.softmax(logits, dim=-1)

    return torch.multinomial(probs, num_samples=1).to(dtype=torch.int)


def generate_next_token(
    model,
    x: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = model(x)  # (B, T, vocab_size)
    return (
        sample(
            logits[0, -1, :].clone(), temperature=temperature, top_k=top_k
        ).unsqueeze(-1),
        logits,
    )


@torch.inference_mode()
def generate(
    model,
    prompt: torch.Tensor,
    *,
    max_generated_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    custom_generate_next_token: Optional[Callable] = None,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ """

    if seed is not None:
        torch.manual_seed(seed)

    prompt = prompt.view(1, -1) if prompt.ndim == 1 else prompt

    if custom_generate_next_token is None:
        _generate_next_token = generate_next_token
    else:
        _generate_next_token = custom_generate_next_token

    generated_tokens = prompt.clone()

    tokens, generated_logits = generate_next_token(
        model,
        x=prompt,
        temperature=temperature,
        top_k=top_k,
    )

    generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)

    for _ in range(max_generated_tokens - 1):
        tokens = generated_tokens.clone()
        tokens, logits = _generate_next_token(
            model,
            x=tokens.clone(),
            temperature=temperature,
            top_k=top_k,
        )

        generated_tokens = torch.cat([generated_tokens, tokens], dim=-1)
        generated_logits = logits

    return generated_tokens, generated_logits
