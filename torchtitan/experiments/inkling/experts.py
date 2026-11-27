# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Frozen checkpoint-format Inkling experts with input gradients for internal LoRA.

B: batch, T: tokens, D: hidden size, N: flattened tokens, E: experts,
K: experts per token, R: rows for one expert, I/O: projection input/output.
"""

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F

from torchtitan.models.base.module import Module


def _e2m1(code: torch.Tensor) -> torch.Tensor:
    # E2M1 magnitudes as E4M3 bytes: 0, 48, 56, 60, 64, 68, 72, 76.
    mag = code & 7
    bits = torch.where(mag < 2, mag * 48, 48 + mag * 4) | ((code & 8) << 4)
    return bits.view(torch.float8_e4m3fn).bfloat16()


def _mx_scale(exponent: torch.Tensor) -> torch.Tensor:
    return (exponent.int() << 23).view(torch.float32).clamp_min(2.0**-127)


def _decode(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.dtype == torch.uint8:
        codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2)
        return (
            _e2m1(codes).unflatten(-1, (-1, 16)) * scale.bfloat16()[..., None]
        ).flatten(-2)
    return (
        (weight.float().unflatten(-1, (-1, 32)) * _mx_scale(scale)[..., None])
        .flatten(-2)
        .bfloat16()
    )


def _quantize_input(
    x_RI: torch.Tensor, nvfp4: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x_RI = x_RI.float()
    if not nvfp4:
        blocks = x_RI.unflatten(-1, (-1, 32))
        raw = blocks.abs().amax(-1, keepdim=True) * (1.0 / 448)
        exponent = ((raw.view(torch.int32) + 0x007FFFFF) >> 23).clamp(0, 254)
        scale = _mx_scale(torch.where(raw <= 2.0**-127, 0, exponent))
        codes = (blocks * scale.reciprocal()).clamp(-448, 448).to(torch.float8_e4m3fn)
        return (codes.float() * scale).flatten(-2).bfloat16(), None

    # Four-over-six selects the smaller absolute reconstruction error per block.
    # A row's outer scale (amax / 1536) is applied after the FP32 matmul.
    blocks = x_RI.unflatten(-1, (-1, 16))
    peak = blocks.abs().amax(-1, keepdim=True)
    amax_R = peak.amax(dim=(-2, -1))
    max_float = torch.finfo(torch.float32).max
    encoder = torch.div(1536.0, amax_R).clamp_max(max_float)[:, None, None]
    outer = (amax_R * (1.0 / 1536))[:, None, None]

    def candidate(reciprocal: float) -> tuple[torch.Tensor, torch.Tensor]:
        scale = (
            (peak * (encoder * reciprocal))
            .clamp_max(448)
            .to(torch.float8_e4m3fn)
            .float()
        )
        normalized = blocks * (scale * encoder.reciprocal()).reciprocal().clamp_max(
            max_float
        )
        # Round to nearest even in units of the local E2M1 spacing (0.5, 1 or 2).
        magnitude = normalized.abs()
        spacing = torch.where(magnitude < 2, 0.5, torch.where(magnitude < 4, 1.0, 2.0))
        rounded = ((normalized / spacing).round() * spacing).clamp(-6, 6)
        values = rounded * scale
        error = (values * outer - blocks).abs()
        # Keep the checkpoint encoder's pairwise FP32 sum: ties select six.
        while error.size(-1) > 1:
            error = error[..., 0::2] + error[..., 1::2]
        return values, error

    six, err_six = candidate(1.0 / 6)
    four, err_four = candidate(1.0 / 4)
    return torch.where(err_four < err_six, four, six).flatten(-2).bfloat16(), amax_R


def _mm_fp32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # CPU has no BF16 mm with FP32 output. The FP32 product of BF16 values is exact.
    if x.device.type == "cpu":
        return x.float() @ w.float()
    return torch.mm(x, w, out_dtype=torch.float32)


class _FrozenProjection(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        x_RI: torch.Tensor,
        weight: torch.Tensor,
        scale: torch.Tensor,
        weight_amax: torch.Tensor | None,
        denominator: torch.Tensor | None,
    ) -> torch.Tensor:
        # Save views of packed storage, not a dense matrix per expert or loop visit.
        ctx.save_for_backward(weight, scale, weight_amax, denominator)
        values_RI, amax_R = _quantize_input(x_RI, nvfp4=weight_amax is not None)
        y_RO = _mm_fp32(values_RI, _decode(weight, scale).T)
        if amax_R is not None:
            assert weight_amax is not None and denominator is not None
            y_RO = (
                y_RO
                * ((amax_R * weight_amax) * (1536 * denominator).reciprocal())[:, None]
            )
        return y_RO.bfloat16()

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx,
        grad_RO: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        # Identity STE for activation quantization (including row/block scales).
        # dX = dY @ decoded_weight; NVFP4's per-expert global factor is part of W.
        weight, scale, weight_amax, denominator = ctx.saved_tensors
        grad_RI = _mm_fp32(grad_RO, _decode(weight, scale))
        if weight_amax is not None:
            grad_RI = grad_RI * (weight_amax / denominator)
        return grad_RI.bfloat16(), None, None, None, None


def _frozen_parameter(*shape: int, dtype: torch.dtype) -> torch.nn.Parameter:
    return torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)


class _Router(Module):
    _preserve_dtype_params = ("bias",)

    def __init__(self, config: "InklingMoE.Config"):
        super().__init__()
        self.weight = _frozen_parameter(
            config.num_experts, config.hidden_size, dtype=torch.bfloat16
        )
        self.bias = _frozen_parameter(config.num_experts, dtype=torch.float32)

    def reset_parameters(self) -> None:
        bound = self.weight.shape[-1] ** -0.5
        self.weight.uniform_(-bound, bound)
        self.bias.zero_()


class _PackedExperts(Module):
    def __init__(self, config: "InklingMoE.Config"):
        super().__init__()
        self.config = config
        nvfp4 = config.expert_format == "nvfp4"
        for name, rows, columns in (
            ("w13_weight", 2 * config.intermediate_size, config.hidden_size),
            ("w2_weight", config.hidden_size, config.intermediate_size),
        ):
            shape = (config.num_experts, rows)
            self.register_parameter(
                name,
                _frozen_parameter(
                    *shape,
                    columns // (2 if nvfp4 else 1),
                    dtype=torch.uint8 if nvfp4 else torch.float8_e4m3fn,
                ),
            )
            self.register_parameter(
                name + "_scale",
                _frozen_parameter(
                    *shape,
                    columns // (16 if nvfp4 else 32),
                    dtype=torch.float8_e4m3fn if nvfp4 else torch.uint8,
                ),
            )
            if nvfp4:
                self.register_parameter(
                    name + "_amax",
                    _frozen_parameter(config.num_experts, dtype=torch.float32),
                )
                self.register_parameter(
                    name + "_outer_scale_denominator",
                    _frozen_parameter(1, dtype=torch.float32),
                )

        self._preserve_dtype_params = tuple(self._parameters)

    def reset_parameters(self) -> None:
        # A bounded, non-degenerate initial state for checkpoint-free debug recipes.
        nvfp4 = self.config.expert_format == "nvfp4"
        for name, value in self.named_parameters(recurse=False):
            if name.endswith("_scale"):
                value.fill_(1.0 if nvfp4 else 122)
            elif name.endswith("_amax"):
                value.fill_(1.0)
            elif name.endswith("_denominator"):
                value.fill_(2 * self.config.hidden_size**0.5)
            elif nvfp4:
                value.random_(0, 256)
            else:
                # Generate signed finite E4M3 encodings without a dense FP32 temporary.
                bits = value.view(torch.uint8)
                bits.random_(24, 56)
                bits[..., 1::2].bitwise_or_(128)

    def project(self, name: str, expert: int, x_RI: torch.Tensor) -> torch.Tensor:
        weight = self.get_parameter(name)[expert]
        nvfp4 = weight.dtype == torch.uint8
        return _FrozenProjection.apply(
            x_RI,
            weight,
            self.get_parameter(name + "_scale")[expert],
            self.get_parameter(name + "_amax")[expert] if nvfp4 else None,
            self.get_parameter(name + "_outer_scale_denominator") if nvfp4 else None,
        )


class InklingMoE(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        hidden_size: int
        intermediate_size: int
        num_experts: int
        experts_per_token: int
        route_scale: float
        gate_softcap: float
        up_softcap: float
        expert_format: Literal["nvfp4", "mxfp8"]

        def __post_init__(self):
            if self.expert_format not in ("nvfp4", "mxfp8"):
                raise ValueError(f"Unsupported expert format: {self.expert_format}")
            block_size = 16 if self.expert_format == "nvfp4" else 32
            if self.hidden_size % block_size or self.intermediate_size % block_size:
                raise ValueError(
                    f"Expert input dimensions must be divisible by {block_size}"
                )

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.gate = _Router(config)
        self.experts = _PackedExperts(config)

    def forward(self, x_BTD: torch.Tensor) -> torch.Tensor:
        # AMP must not downcast the FP32 router or the declared matmul output scales.
        with torch.autocast(x_BTD.device.type, enabled=False):
            x_ND = x_BTD.reshape(-1, self.config.hidden_size).bfloat16()
            scores_NE = F.linear(x_ND.float(), self.gate.weight.float()).sigmoid()
            ids_NK = (
                (scores_NE + self.gate.bias)
                .topk(self.config.experts_per_token, sorted=False)
                .indices
            )
            weights_NK = scores_NE.gather(1, ids_NK)
            weights_NK = (
                weights_NK
                / (weights_NK.sum(-1, keepdim=True) + 1e-20)
                * self.config.route_scale
            )
            out_ND = torch.zeros_like(x_ND, dtype=torch.float32)
            # One host read for all expert boundaries. Keep the per-expert gathers
            # and adds: combining them changes BF16 dX accumulation order.
            sorted_ids, order = ids_NK.flatten().sort(stable=True)
            boundaries = torch.searchsorted(
                sorted_ids, torch.arange(1, self.config.num_experts, device=x_ND.device)
            ).tolist()
            token_groups = (order // self.config.experts_per_token).tensor_split(
                boundaries
            )
            score_groups = weights_NK.flatten()[order].tensor_split(boundaries)
            for expert, (tokens_R, scores_R) in enumerate(
                zip(token_groups, score_groups)
            ):
                if tokens_R.numel() == 0:
                    continue
                gate_up = self.experts.project(
                    "w13_weight", expert, x_ND[tokens_R]
                ).float()
                # The checkpoint interleaves gate/up rows; SwiGLU caps each separately.
                gate, up = gate_up[..., 0::2], gate_up[..., 1::2]
                act = (
                    (gate / self.config.gate_softcap).tanh()
                    * self.config.gate_softcap
                    * gate.sigmoid()
                )
                act = act * (
                    (up / self.config.up_softcap).tanh() * self.config.up_softcap
                )
                act = (act * scores_R[:, None]).bfloat16()
                out_ND.index_add_(
                    0, tokens_R, self.experts.project("w2_weight", expert, act).float()
                )
            return out_ND.bfloat16().reshape_as(x_BTD)
