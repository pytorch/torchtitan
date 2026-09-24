# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
from fractions import Fraction

from torch import nn

from torchtitan.models.common.moe import MoE


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


def _get_active_parameter_weights(
    model: nn.Module,
    parameters: Iterable[nn.Parameter],
) -> dict[int, Fraction]:
    parameter_weights = {id(param): Fraction(1) for param in parameters}

    for module in model.modules():
        if isinstance(module, MoE):
            active_expert_ratio = Fraction(
                module.router.top_k, module.router.num_experts
            )
            for param in module.routed_experts.parameters():
                parameter_weights[id(param)] = active_expert_ratio

    return parameter_weights


def get_parameter_counts(model: nn.Module) -> tuple[int, int]:
    """Return total and architecturally active parameter counts.

    Every unique parameter is included in the total count. Routed-expert
    parameters are weighted by the owning MoE module's active expert ratio for
    the active count, while all other parameters, including embedding tables,
    are counted in full. PyTorch's parameter iterator deduplicates parameters
    shared through weight tying.

    Args:
        model: Built model whose parameters are counted.

    Returns:
        The total and architecturally active parameter counts.
    """
    parameters = list(model.parameters())
    parameter_weights = _get_active_parameter_weights(model, parameters)
    num_parameters = sum(param.numel() for param in parameters)
    usage_weighted_nparams = sum(
        param.numel() * parameter_weights[id(param)] for param in parameters
    )
    assert usage_weighted_nparams.denominator == 1, (
        f"Active parameter count must be integral, got {usage_weighted_nparams}"
    )
    return num_parameters, usage_weighted_nparams.numerator


def active_parameter_flops_per_unit(
    model: nn.Module,
    *,
    excluded_modules: Iterable[nn.Module | None] = (),
) -> int:
    """Return parameter-based training FLOPs per workload unit.

    Embedding tables are excluded unless their parameter is shared with the
    output head. Explicitly excluded subtrees are also assigned zero cost.
    """
    parameters = list(model.parameters())
    parameter_weights = _get_active_parameter_weights(model, parameters)

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

    for excluded_module in excluded_modules:
        if excluded_module is None:
            continue
        for param in excluded_module.parameters():
            parameter_weights[id(param)] = Fraction(0)

    usage_weighted_nparams = sum(
        param.numel() * parameter_weights[id(param)] for param in parameters
    )
    assert usage_weighted_nparams.denominator == 1, (
        f"Active parameter count must be integral, got {usage_weighted_nparams}"
    )
    return 6 * usage_weighted_nparams.numerator
