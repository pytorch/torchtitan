# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Batch-invariance tests for FLA's recurrent GDN kernel."""

from dataclasses import dataclass, replace
from itertools import accumulate, pairwise

import pytest
import torch
import torch.nn.functional as F


fla_gated_delta_rule = pytest.importorskip("fla.ops.gated_delta_rule")
fused_recurrent_gated_delta_rule = fla_gated_delta_rule.fused_recurrent_gated_delta_rule
fla_convolution = pytest.importorskip("fla.modules.convolution")
causal_conv1d = fla_convolution.causal_conv1d
causal_conv1d_update = fla_convolution.causal_conv1d_update

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA GPU"
)


# Shape legend:
# N: sequences, T: tokens, H: query/key heads, J: value heads,
# K: query/key head dim, V: value head dim, C: convolution channels,
# W: convolution width.


@dataclass(frozen=True)
class _SequenceInputs:
    q_THK: torch.Tensor  # noqa: N815
    k_THK: torch.Tensor  # noqa: N815
    v_TJV: torch.Tensor  # noqa: N815
    g_TJ: torch.Tensor  # noqa: N815
    beta_TJ: torch.Tensor  # noqa: N815
    initial_state_JKV: torch.Tensor  # noqa: N815

    @property
    def num_tokens(self) -> int:
        return self.q_THK.shape[0]


def _make_sequence(num_tokens: int, seed: int) -> _SequenceInputs:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    num_qk_heads, num_value_heads, head_dim = 2, 4, 128

    def randn(*shape: int, dtype: torch.dtype) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(device="cuda", dtype=dtype)

    return _SequenceInputs(
        q_THK=randn(num_tokens, num_qk_heads, head_dim, dtype=torch.bfloat16),
        k_THK=randn(num_tokens, num_qk_heads, head_dim, dtype=torch.bfloat16),
        v_TJV=randn(num_tokens, num_value_heads, head_dim, dtype=torch.bfloat16),
        g_TJ=-torch.rand(
            num_tokens, num_value_heads, generator=generator, device="cpu"
        ).to(device="cuda", dtype=torch.float32),
        beta_TJ=torch.sigmoid(randn(num_tokens, num_value_heads, dtype=torch.bfloat16)),
        initial_state_JKV=randn(
            num_value_heads, head_dim, head_dim, dtype=torch.float32
        ),
    )


def _run_packed_varlen(
    sequences: list[_SequenceInputs],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    lengths = [sequence.num_tokens for sequence in sequences]
    cu_seqlens_Np1 = torch.tensor(
        [0, *accumulate(lengths)], device="cuda", dtype=torch.int32
    )
    q_1THK = torch.cat([sequence.q_THK for sequence in sequences]).unsqueeze(0)
    k_1THK = torch.cat([sequence.k_THK for sequence in sequences]).unsqueeze(0)
    v_1TJV = torch.cat([sequence.v_TJV for sequence in sequences]).unsqueeze(0)
    g_1TJ = torch.cat([sequence.g_TJ for sequence in sequences]).unsqueeze(0)
    beta_1TJ = torch.cat([sequence.beta_TJ for sequence in sequences]).unsqueeze(0)
    initial_state_NJKV = torch.stack(
        [sequence.initial_state_JKV for sequence in sequences]
    )

    with torch.no_grad():
        output_1TJV, final_state_NJKV = fused_recurrent_gated_delta_rule(
            q_1THK,
            k_1THK,
            v_1TJV,
            g_1TJ,
            beta=beta_1TJ,
            initial_state=initial_state_NJKV,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens_Np1,
        )

    assert output_1TJV.dtype == torch.bfloat16
    assert final_state_NJKV.dtype == torch.float32
    return list(output_1TJV[0].split(lengths)), final_state_NJKV


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor, context: str) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=context)


def test_recurrent_gdn_is_invariant_to_ragged_batch_composition() -> None:
    """Packed neighbors and request order must not change a request's result."""
    target = _make_sequence(129, seed=0)
    fillers = [
        _make_sequence(length, seed) for length, seed in [(1, 1), (17, 2), (64, 3)]
    ]
    baseline_outputs, baseline_states = _run_packed_varlen([target])

    scenarios = [
        (0, [target, *fillers]),
        (1, [fillers[0], target, *fillers[1:]]),
        (len(fillers), [*fillers, target]),
    ]
    for target_index, sequences in scenarios:
        outputs, states = _run_packed_varlen(sequences)
        _assert_exact(
            outputs[target_index],
            baseline_outputs[0],
            f"output with target at batch index {target_index}",
        )
        _assert_exact(
            states[target_index],
            baseline_states[0],
            f"final state with target at batch index {target_index}",
        )


def test_recurrent_gdn_is_invariant_to_call_boundaries() -> None:
    """Restoring the FP32 SSM state must match one uninterrupted call."""
    sequence = _make_sequence(257, seed=4)
    baseline_outputs, baseline_states = _run_packed_varlen([sequence])

    state_JKV = sequence.initial_state_JKV
    split_outputs = []
    split_points = [0, 1, 64, 65, 128, 257]
    for start, end in pairwise(split_points):
        piece = replace(
            sequence,
            q_THK=sequence.q_THK[start:end],
            k_THK=sequence.k_THK[start:end],
            v_TJV=sequence.v_TJV[start:end],
            g_TJ=sequence.g_TJ[start:end],
            beta_TJ=sequence.beta_TJ[start:end],
            initial_state_JKV=state_JKV,
        )
        outputs, states = _run_packed_varlen([piece])
        split_outputs.append(outputs[0])
        state_JKV = states[0]

    _assert_exact(
        torch.cat(split_outputs),
        baseline_outputs[0],
        "output across recurrence call boundaries",
    )
    _assert_exact(
        state_JKV,
        baseline_states[0],
        "final state across recurrence call boundaries",
    )


def test_short_conv_bf16_cache_matches_prefill_and_decode() -> None:
    """The rolling convolution cache stores BF16 inputs without rounding drift."""
    generator = torch.Generator(device="cuda").manual_seed(5)
    num_tokens, conv_dim, width = 129, 512, 4
    x_1TC = torch.randn(
        1,
        num_tokens,
        conv_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight_CW = torch.randn(
        conv_dim,
        width,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    bias_C = torch.randn(
        conv_dim,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    cu_seqlens_Np1 = torch.tensor([0, num_tokens], device="cuda", dtype=torch.int32)

    prefill_output_1TC, prefill_state_NCW = causal_conv1d(
        x_1TC,
        weight=weight_CW,
        bias=bias_C,
        activation="silu",
        initial_state=x_1TC.new_zeros(1, conv_dim, width),
        output_final_state=True,
        cu_seqlens=cu_seqlens_Np1,
    )

    conv_state_NCWm1 = x_1TC.new_zeros(1, conv_dim, width - 1)
    decode_outputs = []
    for token_index in range(num_tokens):
        conv_cache_NCW = F.pad(conv_state_NCWm1, (1, 0))
        decode_output_NC, conv_cache_NCW = causal_conv1d_update(
            x_1TC[:, token_index],
            conv_cache_NCW,
            weight=weight_CW,
            bias=bias_C,
            activation="silu",
        )
        decode_outputs.append(decode_output_NC.unsqueeze(1))
        conv_state_NCWm1 = conv_cache_NCW[..., 1:]

    _assert_exact(
        torch.cat(decode_outputs, dim=1),
        prefill_output_1TC,
        "short-conv prefill versus token-by-token decode output",
    )
    _assert_exact(
        conv_state_NCWm1,
        prefill_state_NCW[..., 1:],
        "short-conv prefill versus token-by-token decode state",
    )
