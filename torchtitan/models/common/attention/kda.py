# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention with Attention Gym and FLA training backends."""

# Shape suffix legend:
# T = packed tokens, D = model dimension, C = projection channels,
# N = num heads, K = head dimension, W = convolution width.

from dataclasses import dataclass
from functools import cache
from itertools import pairwise
from typing import Any, Literal, get_args

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.attention.attention import (
    AttentionMasksType,
    VarlenMetadata,
)
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d, RMSNorm
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger, warn_once

try:
    from attn_gym.linear.kda import (
        bounded_gate_cumsum,
        causal_conv1d,
        chunk_kda,
        l2norm,
    )
except ImportError:
    bounded_gate_cumsum: Any = None
    causal_conv1d: Any = None
    chunk_kda: Any = None
    l2norm: Any = None


__all__ = ["KDA", "InnerKDA", "KDABackend", "KDAKernel"]

KDABackend = Literal["auto", "fused", "fla", "reference"]


@cache
def _get_fla_chunk_kda():
    try:
        from fla.ops.kda import chunk_kda as fla_chunk_kda
    except ImportError as error:
        raise ImportError(
            "KDA requires flash-linear-attention on GPUs unsupported by the "
            "Attention Gym fused backend: pip install flash-linear-attention"
        ) from error
    return fla_chunk_kda


def _reference_l2norm(x_BLNK: torch.Tensor) -> torch.Tensor:
    input_dtype = x_BLNK.dtype
    x_float_BLNK = x_BLNK.float()
    return (
        x_float_BLNK
        * torch.rsqrt(x_float_BLNK.square().sum(dim=-1, keepdim=True) + 1e-6)
    ).to(input_dtype)


def _reference_gate_cumsum(
    raw_gate_BLNK: torch.Tensor,
    A_log_N: torch.Tensor,
    dt_bias_NK: torch.Tensor,
    *,
    lower_bound: float,
    chunk_size: int,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    gate_BLNK = lower_bound * torch.sigmoid(
        A_log_N.float().exp().view(1, 1, -1, 1)
        * (raw_gate_BLNK.float() + dt_bias_NK.float())
    )
    if cu_seqlens is None:
        spans = gate_BLNK.unbind(0)
    else:
        offsets = cu_seqlens.tolist()
        spans = [gate_BLNK[0, start:end] for start, end in pairwise(offsets)]

    chunks = [chunk for span in spans for chunk in span.split(chunk_size, dim=0)]
    cumulative_TNK = torch.cat([chunk.cumsum(dim=0) for chunk in chunks], dim=0)
    if cu_seqlens is None:
        return cumulative_TNK.reshape_as(gate_BLNK) * 1.4426950408889634
    return cumulative_TNK.unsqueeze(0) * 1.4426950408889634


class KDAKernel(Module):
    """Apply KDA preprocessing and dispatch to the configured kernel backend."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        backend: KDABackend = "auto"
        lower_bound: float = -5.0
        chunk_size: int = 64

        def __post_init__(self):
            valid = get_args(KDABackend)
            if self.backend not in valid:
                raise ValueError(
                    f"unknown KDA backend {self.backend!r}. "
                    f"Valid: {', '.join(repr(name) for name in valid)}."
                )
            if not -5.0 <= self.lower_bound < 0.0:
                raise ValueError(
                    "KDA lower_bound must be in the safe range [-5, 0), "
                    f"got {self.lower_bound}."
                )
            if self.chunk_size != 64:
                raise ValueError(
                    f"Attention Gym KDA requires chunk_size=64, got {self.chunk_size}."
                )

    def __init__(self, config: Config):
        super().__init__()
        if config.backend in ("fused", "reference") and chunk_kda is None:
            raise ImportError(
                "KDA requires Attention Gym: pip install 'attn-gym[linear]'"
            )
        self.backend = config.backend
        self.lower_bound = config.lower_bound
        self.chunk_size = config.chunk_size
        self._resolved_backend: KDABackend | None = None

    def resolve_backend(self, q_BLNK: torch.Tensor) -> KDABackend:
        """Select Attention Gym fused KDA when supported, otherwise FLA."""
        if self.backend != "auto":
            if self.backend == "fused":
                self._validate_fused_support(q_BLNK)
            return self.backend

        if self._resolved_backend is not None:
            return self._resolved_backend
        if q_BLNK.device.type != "cuda":
            raise RuntimeError(
                "KDA backend='auto' requires CUDA. Use backend='reference' "
                "for correctness testing on CPU."
            )
        if chunk_kda is not None and self._has_fused_support(q_BLNK):
            self._resolved_backend = "fused"
            return self._resolved_backend
        _get_fla_chunk_kda()
        capability = torch.cuda.get_device_capability(q_BLNK.device)
        reason = (
            "is unavailable"
            if chunk_kda is None
            else f"does not support CUDA capability {capability}"
        )
        warn_once(
            logger,
            f"Attention Gym fused KDA {reason}; falling back to FLA.",
        )
        self._resolved_backend = "fla"
        return self._resolved_backend

    @staticmethod
    def _has_fused_support(q_BLNK: torch.Tensor) -> bool:
        return (
            q_BLNK.device.type == "cuda"
            and q_BLNK.shape[-1] == 128
            and torch.cuda.get_device_capability(q_BLNK.device) in {(10, 0), (10, 3)}
        )

    @classmethod
    def _validate_fused_support(cls, q_BLNK: torch.Tensor) -> None:
        if cls._has_fused_support(q_BLNK):
            return
        capability = (
            torch.cuda.get_device_capability(q_BLNK.device)
            if q_BLNK.device.type == "cuda"
            else None
        )
        raise RuntimeError(
            "Attention Gym fused KDA requires a CUDA tensor with head_dim=128 "
            "on Blackwell SM100/SM103; got "
            f"device={q_BLNK.device}, shape={tuple(q_BLNK.shape)}, "
            f"capability={capability}. Use backend='auto' to fall back to FLA."
        )

    def forward(
        self,
        q_BLNK: torch.Tensor,
        k_BLNK: torch.Tensor,
        v_BLNK: torch.Tensor,
        raw_gate_BLNK: torch.Tensor,
        raw_beta_BLN: torch.Tensor,
        A_log_N: torch.Tensor,
        dt_bias_NK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        backend = self.resolve_backend(q_BLNK)
        if backend == "fla":
            fla_chunk_kda = _get_fla_chunk_kda()
            output_BLNK, _ = fla_chunk_kda(
                q_BLNK,
                k_BLNK,
                v_BLNK,
                raw_gate_BLNK,
                raw_beta_BLN,
                A_log=A_log_N,
                dt_bias=dt_bias_NK.reshape(-1),
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=True,
                use_beta_sigmoid_in_kernel=True,
                safe_gate=True,
                lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=(
                    cu_seqlens.to(device="cpu") if cu_seqlens is not None else None
                ),
            )
            return output_BLNK

        if backend == "fused":
            q_BLNK = l2norm(q_BLNK)
            k_BLNK = l2norm(k_BLNK)
            cumulative_gate_BLNK = bounded_gate_cumsum(
                raw_gate_BLNK.to(torch.bfloat16),
                A_log_N.float(),
                dt_bias_NK.float(),
                chunk_size=self.chunk_size,
                lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
            )
        else:
            q_BLNK = _reference_l2norm(q_BLNK)
            k_BLNK = _reference_l2norm(k_BLNK)
            cumulative_gate_BLNK = _reference_gate_cumsum(
                raw_gate_BLNK,
                A_log_N,
                dt_bias_NK,
                lower_bound=self.lower_bound,
                chunk_size=self.chunk_size,
                cu_seqlens=cu_seqlens,
            )

        output_BLNK, _ = chunk_kda(
            q_BLNK,
            k_BLNK,
            v_BLNK,
            cumulative_gate_BLNK,
            raw_beta_BLN.float().sigmoid(),
            cu_seqlens=cu_seqlens,
            impl=backend,
        )
        return output_BLNK


class InnerKDA(Module):
    """Run short convolution and KDA behind the vLLM replacement boundary."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        kernel: KDAKernel.Config

        def __post_init__(self):
            if self.head_dim < 1:
                raise ValueError(f"head_dim must be positive, got {self.head_dim}")
            if self.kernel.backend == "fused" and self.head_dim != 128:
                raise ValueError(
                    "Attention Gym's fused KDA backend requires head_dim=128, "
                    f"got {self.head_dim}."
                )

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.kernel = config.kernel.build()

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_TNK: torch.Tensor,
        raw_beta_TN: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_N: torch.Tensor,
        dt_bias_NK: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        kernel_cu_seqlens = cu_seqlens if cu_seqlens.numel() > 2 else None
        mixed_qkv_BTC = torch.cat(
            (query_TC, key_TC, value_TC),
            dim=-1,
        ).unsqueeze(0)
        conv_weight_C1W = torch.cat(
            (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W),
            dim=0,
        )
        backend = self.kernel.resolve_backend(raw_gate_TNK.unsqueeze(0))
        if backend == "fused":
            conv_output_BTC = causal_conv1d(
                mixed_qkv_BTC,
                conv_weight_C1W[:, 0],
                activation="silu",
                cu_seqlens=kernel_cu_seqlens,
            )
        else:
            channels = mixed_qkv_BTC.shape[-1]

            def convolve(x_BTC: torch.Tensor) -> torch.Tensor:
                conv_input_BCT = F.pad(
                    x_BTC.transpose(1, 2),
                    (conv_weight_C1W.shape[-1] - 1, 0),
                )
                return F.silu(
                    F.conv1d(
                        conv_input_BCT,
                        conv_weight_C1W,
                        groups=channels,
                    ).transpose(1, 2)
                )

            if kernel_cu_seqlens is None:
                conv_output_BTC = convolve(mixed_qkv_BTC)
            else:
                offsets = kernel_cu_seqlens.tolist()
                conv_output_BTC = torch.cat(
                    [
                        convolve(mixed_qkv_BTC[:, start:end])
                        for start, end in pairwise(offsets)
                    ],
                    dim=1,
                )

        projection_dim = conv_output_BTC.shape[-1] // 3
        q_BTC, k_BTC, v_BTC = conv_output_BTC.split(projection_dim, dim=-1)
        q_BTNK, k_BTNK, v_BTNK = (
            tensor.unflatten(-1, (-1, self.head_dim))
            for tensor in (q_BTC, k_BTC, v_BTC)
        )
        output_BTNK = self.kernel(
            q_BTNK,
            k_BTNK,
            v_BTNK,
            raw_gate_TNK.unsqueeze(0),
            raw_beta_TN.unsqueeze(0),
            A_log_N,
            dt_bias_NK,
            cu_seqlens=kernel_cu_seqlens,
        )
        return output_BTNK.squeeze(0)


class KDA(Module):
    """Kimi Delta Attention with checkpoint-compatible Kimi K3 parameters."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_heads: int
        head_dim: int
        conv_kernel_size: int
        q_proj: Linear.Config
        k_proj: Linear.Config
        v_proj: Linear.Config
        q_conv: Conv1d.Config
        k_conv: Conv1d.Config
        v_conv: Conv1d.Config
        forget_a: Linear.Config
        forget_b: Linear.Config
        beta: Linear.Config
        output_gate: Linear.Config
        inner_kda: Module.Config
        output_norm: RMSNorm.Config
        output_proj: Linear.Config

        def __post_init__(self):
            if self.num_heads < 1:
                raise ValueError(f"num_heads must be positive, got {self.num_heads}")
            if self.head_dim < 1:
                raise ValueError(f"head_dim must be positive, got {self.head_dim}")
            if self.conv_kernel_size < 1:
                raise ValueError(
                    f"conv_kernel_size must be positive, got {self.conv_kernel_size}"
                )

    def __init__(self, config: Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.conv_kernel_size = config.conv_kernel_size

        self.q_proj = config.q_proj.build()
        self.k_proj = config.k_proj.build()
        self.v_proj = config.v_proj.build()
        self.q_conv = config.q_conv.build()
        self.k_conv = config.k_conv.build()
        self.v_conv = config.v_conv.build()
        self.forget_a = config.forget_a.build()
        self.forget_b = config.forget_b.build()
        self.beta = config.beta.build()
        self.output_gate = config.output_gate.build()
        self.inner_kda = config.inner_kda.build()
        self.output_norm = config.output_norm.build()
        self.output_proj = config.output_proj.build()

        self.A_log = nn.Parameter(torch.empty(config.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(config.num_heads, config.head_dim))

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if x_TD.ndim != 2:
            raise ValueError(
                f"KDA input must have shape [T, D], got {tuple(x_TD.shape)}."
            )

        if attention_masks is not None and not isinstance(
            attention_masks, VarlenMetadata
        ):
            raise ValueError(
                "KDA attention_masks must be VarlenMetadata or None, "
                f"got {type(attention_masks).__name__}."
            )
        num_tokens = x_TD.shape[0]
        cu_seqlens = (
            attention_masks.cu_seq_q
            if isinstance(attention_masks, VarlenMetadata)
            else torch.tensor(
                [0, num_tokens],
                dtype=torch.int32,
                device=x_TD.device,
            )
        )
        raw_gate_TNK = self.forget_b(self.forget_a(x_TD)).reshape(
            num_tokens, self.num_heads, self.head_dim
        )
        raw_beta_TN = self.beta(x_TD).reshape(num_tokens, self.num_heads)
        out_TNK = self.inner_kda(
            self.q_proj(x_TD),
            self.k_proj(x_TD),
            self.v_proj(x_TD),
            raw_gate_TNK,
            raw_beta_TN,
            self.q_conv.weight,
            self.k_conv.weight,
            self.v_conv.weight,
            self.A_log,
            self.dt_bias,
            cu_seqlens,
        )

        out_float_TNK = out_TNK.float()
        assert self.output_norm.eps is not None
        out_float_TNK = out_float_TNK * torch.rsqrt(
            out_float_TNK.square().mean(dim=-1, keepdim=True) + self.output_norm.eps
        )
        out_float_TNK = out_float_TNK * self.output_norm.weight.float()
        output_gate_TNK = self.output_gate(x_TD).view_as(out_TNK).float()
        out_TD = self.output_proj(
            (out_float_TNK * output_gate_TNK.sigmoid()).to(out_TNK.dtype).flatten(-2)
        )
        return out_TD
