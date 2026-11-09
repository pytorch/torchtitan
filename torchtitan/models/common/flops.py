# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Parameter counting and FLOP formulas shared by model ``get_nparams_and_flops``."""

import logging
from collections.abc import Iterable
from fractions import Fraction

import torch.nn as nn

from torchtitan.models.common.moe import MoE


logger = logging.getLogger(__name__)


def quadratic_attention_flops_per_token(
    *,
    num_heads: int,
    qk_head_dim: int,
    v_head_dim: int,
    seq_len: int,
    sliding_window_size: int | None = None,
) -> int:
    """Training FLOPs per token for quadratic or windowed attention.

    Reasoning behind the factor of 6 for the self-attention part of the formula:
    1. each self-attention has 2 matmul in the forward and 4 (counted as 2)
       in the backward                                                      (3)
       The 2 matmuls per token are:
       a. tmp = q @ K^T: [1, qk_head_dim] @ [qk_head_dim, seq_len]
       b. tmp @ V: [1, seq_len] @ [seq_len, v_head_dim]
       so we get
       seq_len * qk_head_dim + seq_len * v_head_dim = seq_len * (qk_head_dim + v_head_dim)
    2. the flash attention does 1 more matmul recomputation in the backward
       but recomputation should not be counted in calculating MFU           (+0)
    3. each matmul performs 1 multiplication and 1 addition                 (*2)
    4. we follow the convention and do not account for sparsity in causal attention

    ``qk_head_dim`` and ``v_head_dim`` describe the two attention
    contractions. The factor of 6 accounts for multiply-adds in forward and
    backward. As in the existing MFU convention, causal sparsity and backward
    recomputation are not counted.
    """
    attended_tokens = (
        seq_len if sliding_window_size is None else min(seq_len, sliding_window_size)
    )
    return 6 * num_heads * (qk_head_dim + v_head_dim) * attended_tokens


def delta_rule_flops_per_token(
    *,
    num_heads: int,
    key_head_dim: int,
    v_head_dim: int,
) -> int:
    """Training FLOPs per token for a recurrent delta-rule state update.

    Omitting batch dimensions,
    ``state``: ``[num_heads, key_head_dim, v_head_dim]``
    ``key`` and ``query``: ``[num_heads, key_head_dim]``
    ``value`` and ``delta``: ``[num_heads, v_head_dim]``

    For each token, the recurrence performs:
    1. Decay the state: ``decayed_state = exp(decay) * state``.
    2. Read the stored value: ``memory = decayed_state.T @ key``.
    3. Form the gated correction: ``delta = beta * (value - memory)``.
    4. Update the state: ``state = decayed_state + key[:, None] * delta[None, :]``.
    5. Read the output: ``output = state.T @ query``.

    Steps 2, 4, and 5 each scale as ``key_head_dim * v_head_dim``, producing the
    factor of 3. The factor of 6 accounts for multiply-adds in forward and
    backward. Gate-producing linear projections are covered by the model's
    ``6 * active_nparams`` term. The elementwise work in steps 1 and 3, output
    gating, normalization, nonlinearities, and backward recomputation are not
    counted.
    """
    return 6 * 3 * num_heads * key_head_dim * v_head_dim


def get_nparams_and_active_nparams(
    model: nn.Module,
    *,
    modules_excluded_from_active_params: Iterable[nn.Module | None] = (),
) -> tuple[int, int]:
    """Count total and matmul-active parameters for a native decoder.

    Routed-expert parameters are weighted by the owning MoE module's active
    expert ratio. Embedding tables are excluded unless their parameter is shared
    with the output head. Explicitly excluded subtrees are also assigned zero
    per-token parameter cost.

    Args:
        model: Built model whose parameters are counted.
        modules_excluded_from_active_params: Module subtrees whose cost does not
            scale per text token, such as a vision encoder.

    Returns:
        Total parameter count and effective parameter count for the conventional
        ``6 * active_parameters`` training FLOP estimate.
    """
    named_parameters = list(model.named_parameters())
    nparams = sum(param.numel() for _, param in named_parameters)
    parameter_weights = {id(param): Fraction(1) for _, param in named_parameters}

    for module in model.modules():
        if isinstance(module, MoE):
            active_expert_ratio = Fraction(
                module.router.top_k, module.router.num_experts
            )
            for param in module.routed_experts.parameters():
                parameter_weights[id(param)] = active_expert_ratio

    lm_head = getattr(model, "lm_head", None)
    lm_head_parameter_ids = (
        {id(param) for param in lm_head.parameters()}
        if isinstance(lm_head, nn.Module)
        else set()
    )
    for module in model.modules():
        if isinstance(module, nn.Embedding):
            for param in module.parameters(recurse=False):
                if id(param) not in lm_head_parameter_ids:
                    parameter_weights[id(param)] = Fraction(0)

    for module_excluded_from_active_params in modules_excluded_from_active_params:
        if module_excluded_from_active_params is None:
            continue
        for param in module_excluded_from_active_params.parameters():
            parameter_weights[id(param)] = Fraction(0)

    nparams_for_matmul = sum(
        param.numel() * parameter_weights[id(param)] for _, param in named_parameters
    )
    assert nparams_for_matmul.denominator == 1
    active_nparams = nparams_for_matmul.numerator

    logger.info(
        f"Total parameter count: {nparams:,}, active parameters: {active_nparams:,}"
    )
    return nparams, active_nparams
