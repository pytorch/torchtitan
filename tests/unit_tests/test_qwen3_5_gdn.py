# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Adversarial batch-invariance tests for FLA's recurrent GDN kernel."""

import importlib.util
import math
from dataclasses import dataclass

import pytest
import torch


if importlib.util.find_spec("fla") is None:
    pytest.skip("flash-linear-attention is not installed", allow_module_level=True)

fla_gated_delta_rule = importlib.import_module("fla.ops.gated_delta_rule")
fused_recurrent_gated_delta_rule = fla_gated_delta_rule.fused_recurrent_gated_delta_rule
fla_convolution = importlib.import_module("fla.modules.convolution")
causal_conv1d = fla_convolution.causal_conv1d
causal_conv1d_update = fla_convolution.causal_conv1d_update


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


def _make_sequence(
    *,
    num_tokens: int,
    seed: int,
    num_qk_heads: int,
    num_value_heads: int,
    qk_head_dim: int,
    value_head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> _SequenceInputs:
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # A wide dynamic range and alternating signs amplify any change in the
    # reduction order without making the recurrence overflow.
    qk_scale_K = torch.logspace(-3, 3, qk_head_dim, base=2.0)
    value_sign_V = torch.where(
        torch.arange(value_head_dim) % 2 == 0,
        torch.tensor(1.0),
        torch.tensor(-1.0),
    )
    q_THK = (
        torch.randn(num_tokens, num_qk_heads, qk_head_dim, generator=generator)
        * qk_scale_K
    )
    k_THK = torch.randn(
        num_tokens, num_qk_heads, qk_head_dim, generator=generator
    ) * qk_scale_K.flip(0)
    v_TJV = (
        torch.randn(num_tokens, num_value_heads, value_head_dim, generator=generator)
        * value_sign_V
        * 3.0
    )
    g_TJ = -(1e-4 + 0.2 * torch.rand(num_tokens, num_value_heads, generator=generator))
    beta_TJ = torch.sigmoid(
        5.0 * torch.randn(num_tokens, num_value_heads, generator=generator)
    )
    initial_state_JKV = 0.5 * torch.randn(
        num_value_heads,
        qk_head_dim,
        value_head_dim,
        generator=generator,
    )

    return _SequenceInputs(
        q_THK=q_THK.to(device=device, dtype=dtype),
        k_THK=k_THK.to(device=device, dtype=dtype),
        v_TJV=v_TJV.to(device=device, dtype=dtype),
        g_TJ=g_TJ.to(device=device, dtype=torch.float32),
        beta_TJ=beta_TJ.to(device=device, dtype=dtype),
        initial_state_JKV=initial_state_JKV.to(device=device, dtype=torch.float32),
    )


def _make_poison_sequence(source: _SequenceInputs) -> _SequenceInputs:
    q_THK = source.q_THK.clone()
    k_THK = source.k_THK.clone()
    v_TJV = source.v_TJV.clone()
    g_TJ = source.g_TJ.clone()
    beta_TJ = source.beta_TJ.clone()
    initial_state_JKV = source.initial_state_JKV.clone()

    q_THK.fill_(math.nan)
    k_THK.fill_(math.inf)
    v_TJV.fill_(-math.inf)
    g_TJ.fill_(-math.inf)
    beta_TJ.fill_(math.nan)
    initial_state_JKV.fill_(math.nan)
    return _SequenceInputs(
        q_THK=q_THK,
        k_THK=k_THK,
        v_TJV=v_TJV,
        g_TJ=g_TJ,
        beta_TJ=beta_TJ,
        initial_state_JKV=initial_state_JKV,
    )


def _run_packed_varlen(
    sequences: list[_SequenceInputs],
) -> tuple[list[torch.Tensor], torch.Tensor]:
    lengths = [sequence.num_tokens for sequence in sequences]
    cu_seqlens_Np1 = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device=sequences[0].q_THK.device,
        dtype=torch.int32,
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
    assert output_1TJV.dtype == sequences[0].v_TJV.dtype
    assert final_state_NJKV.dtype == torch.float32

    output_sequences = list(output_1TJV[0].split(lengths))
    return output_sequences, final_state_NJKV


def _assert_bitwise_equal(
    actual: torch.Tensor, expected: torch.Tensor, *, context: str
) -> None:
    if torch.equal(actual, expected):
        return

    finite = torch.isfinite(actual) & torch.isfinite(expected)
    max_delta = (
        (actual[finite].float() - expected[finite].float()).abs().max().item()
        if finite.any()
        else math.nan
    )
    num_different = torch.count_nonzero(actual != expected).item()
    pytest.fail(
        f"{context} is not bitwise equal: "
        f"num_different={num_different}/{actual.numel()}, max_delta={max_delta}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize(
    ("qk_head_dim", "value_head_dim"),
    [(128, 128), (100, 129)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fla_recurrent_gdn_is_bitwise_invariant_to_ragged_batch_composition(
    qk_head_dim: int,
    value_head_dim: int,
    dtype: torch.dtype,
) -> None:
    """Changing packed neighbors, order, and grid size must not change a request."""
    device = torch.device("cuda")
    common = {
        "num_qk_heads": 2,
        "num_value_heads": 4,
        "qk_head_dim": qk_head_dim,
        "value_head_dim": value_head_dim,
        "dtype": dtype,
        "device": device,
    }
    target = _make_sequence(num_tokens=257, seed=0, **common)
    fillers = [
        _make_sequence(num_tokens=length, seed=index + 1, **common)
        for index, length in enumerate([1, 2, 7, 31, 63, 64, 65, 127, 128, 129])
    ]
    poison = _make_poison_sequence(fillers[2])

    baseline_outputs, baseline_states = _run_packed_varlen([target])
    baseline_output_TJV = baseline_outputs[0]
    baseline_state_JKV = baseline_states[0]

    large_batch = [
        _make_sequence(
            num_tokens=(index % 5) + 1,
            seed=1000 + index,
            **common,
        )
        for index in range(67)
    ]
    scenarios = [
        [target, *fillers],
        [*fillers, target],
        [*fillers[:5], target, *fillers[5:]],
        [*reversed(fillers), poison, target],
        [target, poison, target],
        [*large_batch[:33], target, *large_batch[33:]],
    ]

    for scenario_index, scenario in enumerate(scenarios):
        outputs, states = _run_packed_varlen(scenario)
        for sequence_index, sequence in enumerate(scenario):
            if sequence is not target:
                continue
            context = f"scenario={scenario_index}, target_index={sequence_index}"
            _assert_bitwise_equal(
                outputs[sequence_index],
                baseline_output_TJV,
                context=f"output ({context})",
            )
            _assert_bitwise_equal(
                states[sequence_index],
                baseline_state_JKV,
                context=f"final state ({context})",
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fla_recurrent_gdn_is_bitwise_invariant_to_recurrence_call_boundaries(
    dtype: torch.dtype,
) -> None:
    """Saving and restoring the fp32 SSM state must match one uninterrupted call."""
    device = torch.device("cuda")
    sequence = _make_sequence(
        num_tokens=257,
        seed=2026,
        num_qk_heads=2,
        num_value_heads=4,
        qk_head_dim=128,
        value_head_dim=128,
        dtype=dtype,
        device=device,
    )
    baseline_outputs, baseline_states = _run_packed_varlen([sequence])

    split_points = [0, 1, 3, 63, 64, 65, 128, 256, 257]
    state_JKV = sequence.initial_state_JKV
    split_outputs = []
    for start, end in zip(split_points, split_points[1:]):
        piece = _SequenceInputs(
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

    _assert_bitwise_equal(
        torch.cat(split_outputs),
        baseline_outputs[0],
        context="output across recurrence call boundaries",
    )
    _assert_bitwise_equal(
        state_JKV,
        baseline_states[0],
        context="final state across recurrence call boundaries",
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU")
def test_fla_short_conv_bf16_cache_matches_prefill_and_decode() -> None:
    """The rolling conv cache stores BF16 input columns without accumulating."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(2027)
    num_tokens, conv_dim, width = 129, 512, 4
    x_1TC = torch.randn(
        1,
        num_tokens,
        conv_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    weight_CW = torch.randn(
        conv_dim,
        width,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    bias_C = torch.randn(
        conv_dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    cu_seqlens_Np1 = torch.tensor([0, num_tokens], device=device, dtype=torch.int32)
    initial_state_NCW = x_1TC.new_zeros(1, conv_dim, width)

    prefill_output_1TC, prefill_state_NCW = causal_conv1d(
        x_1TC,
        weight=weight_CW,
        bias=bias_C,
        activation="silu",
        initial_state=initial_state_NCW,
        output_final_state=True,
        cu_seqlens=cu_seqlens_Np1,
    )
    assert prefill_output_1TC.dtype == torch.bfloat16
    assert prefill_state_NCW.dtype == torch.bfloat16

    conv_state_NCWm1 = x_1TC.new_zeros(1, conv_dim, width - 1)
    decode_outputs = []
    for token_index in range(num_tokens):
        conv_cache_NCW = torch.cat(
            [conv_state_NCWm1.new_zeros(1, conv_dim, 1), conv_state_NCWm1],
            dim=-1,
        )
        decode_output_NC, conv_cache_NCW = causal_conv1d_update(
            x_1TC[:, token_index],
            conv_cache_NCW,
            weight=weight_CW,
            bias=bias_C,
            activation="silu",
        )
        decode_outputs.append(decode_output_NC.unsqueeze(1))
        conv_state_NCWm1 = conv_cache_NCW[..., 1:]

    assert conv_state_NCWm1.dtype == torch.bfloat16

    _assert_bitwise_equal(
        torch.cat(decode_outputs, dim=1),
        prefill_output_1TC,
        context="BF16 short-conv prefill versus token-by-token decode output",
    )
    _assert_bitwise_equal(
        conv_state_NCWm1,
        prefill_state_NCW[..., 1:],
        context="BF16 short-conv prefill versus token-by-token decode state",
    )
