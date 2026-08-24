# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention with Attention Gym and FLA training backends."""

import math
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.attention import (
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

__all__ = [
    "KDA",
    "InnerKDA",
    "KDABackend",
    "KDAKernel",
    "KimiDeltaAttention",
    "KimiKDAKernel",
]


class KDABackend(StrEnum):
    AUTO = "auto"
    FLA = "fla"
    ATTN_GYM = "attn_gym"
    NAIVE = "naive"


def _naive_l2norm(x_BLNK: torch.Tensor) -> torch.Tensor:
    input_dtype = x_BLNK.dtype
    x_float_BLNK = x_BLNK.float()
    return (
        x_float_BLNK
        * torch.rsqrt(x_float_BLNK.square().sum(dim=-1, keepdim=True) + 1e-6)
    ).to(input_dtype)


def _naive_gate_cumsum(
    raw_gate_BLNK: torch.Tensor,
    A_log_N: torch.Tensor,
    dt_bias_NK: torch.Tensor,
    *,
    lower_bound: float,
    chunk_size: int,
    cu_seqlens_host: tuple[int, ...] | None,
) -> torch.Tensor:
    gate_BLNK = lower_bound * torch.sigmoid(
        A_log_N.float().exp().view(1, 1, -1, 1)
        * (raw_gate_BLNK.float() + dt_bias_NK.float())
    )
    if cu_seqlens_host is None:
        spans = gate_BLNK.unbind(0)
    else:
        spans = [gate_BLNK[0, start:end] for start, end in pairwise(cu_seqlens_host)]

    chunks = [chunk for span in spans for chunk in span.split(chunk_size, dim=0)]
    cumulative_TNK = torch.cat([chunk.cumsum(dim=0) for chunk in chunks], dim=0)
    if cu_seqlens_host is None:
        return cumulative_TNK.reshape_as(gate_BLNK) * math.log2(math.e)
    return cumulative_TNK.unsqueeze(0) * math.log2(math.e)


class KDAKernel(Module):
    """Apply KDA preprocessing and dispatch to the configured kernel backend."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        backend: KDABackend = KDABackend.AUTO
        lower_bound: float = -5.0
        chunk_size: int = 64

        def __post_init__(self):
            try:
                self.backend = KDABackend(self.backend)
            except ValueError as error:
                valid = ", ".join(backend.value for backend in KDABackend)
                raise ValueError(
                    f"Unknown KDA backend {self.backend!r}; expected one of {valid}."
                ) from error
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
        if (
            config.backend in (KDABackend.ATTN_GYM, KDABackend.NAIVE)
            and chunk_kda is None
        ):
            raise ImportError(
                "KDA requires Attention Gym: pip install 'attn-gym[linear]'"
            )
        self.backend = config.backend
        self.lower_bound = config.lower_bound
        self.chunk_size = config.chunk_size

    def resolve_backend(self, q_BLNK: torch.Tensor) -> KDABackend:
        """Select Attention Gym fused KDA when supported, otherwise FLA."""
        if self.backend is KDABackend.NAIVE:
            return KDABackend.NAIVE

        is_cuda = q_BLNK.device.type == "cuda"
        capability = (
            torch.cuda.get_device_capability(q_BLNK.device) if is_cuda else None
        )
        fused_supported = (
            is_cuda and q_BLNK.shape[-1] == 128 and capability in {(10, 0), (10, 3)}
        )

        if self.backend is KDABackend.AUTO and not is_cuda:
            raise RuntimeError(
                "KDA backend AUTO requires CUDA. Use KDABackend.NAIVE for "
                "correctness testing on CPU."
            )
        if self.backend is KDABackend.ATTN_GYM:
            if not fused_supported:
                raise RuntimeError(
                    "Attention Gym KDA requires a CUDA tensor with head_dim=128 "
                    "on Blackwell SM100/SM103; got "
                    f"device={q_BLNK.device}, shape={tuple(q_BLNK.shape)}, "
                    f"capability={capability}. Use KDABackend.AUTO to fall back "
                    "to FLA."
                )
            return KDABackend.ATTN_GYM
        elif (
            self.backend is KDABackend.AUTO
            and fused_supported
            and chunk_kda is not None
        ):
            return KDABackend.ATTN_GYM
        else:
            try:
                from fla.ops.kda import chunk_kda as fla_chunk_kda  # noqa: F401
            except ImportError as error:
                raise ImportError(
                    "KDA requires flash-linear-attention when Attention Gym is "
                    "unsupported: pip install flash-linear-attention"
                ) from error

        if self.backend is KDABackend.AUTO:
            reason = (
                "is unavailable"
                if chunk_kda is None
                else "does not support "
                f"head_dim={q_BLNK.shape[-1]} on CUDA capability {capability}"
            )
            warn_once(
                logger,
                f"Attention Gym KDA {reason}; falling back to FLA.",
            )
        return KDABackend.FLA

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
        cu_seqlens_host: tuple[int, ...] | None = None,
        backend: KDABackend | None = None,
    ) -> torch.Tensor:
        if backend is None:
            backend = self.resolve_backend(q_BLNK)
        if (
            cu_seqlens is not None
            and cu_seqlens_host is None
            and backend is not KDABackend.ATTN_GYM
        ):
            raise ValueError(
                "FLA and naive KDA varlen execution require host sequence offsets."
            )
        if backend is KDABackend.FLA:
            from fla.ops.kda import chunk_kda as fla_chunk_kda

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
                    torch.tensor(
                        cu_seqlens_host,
                        dtype=torch.int32,
                        device="cpu",
                    )
                    if cu_seqlens_host is not None
                    else None
                ),
            )
            return output_BLNK

        if backend is KDABackend.ATTN_GYM:
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
            q_BLNK = _naive_l2norm(q_BLNK)
            k_BLNK = _naive_l2norm(k_BLNK)
            cumulative_gate_BLNK = _naive_gate_cumsum(
                raw_gate_BLNK,
                A_log_N,
                dt_bias_NK,
                lower_bound=self.lower_bound,
                chunk_size=self.chunk_size,
                cu_seqlens_host=cu_seqlens_host,
            )

        output_BLNK, _ = chunk_kda(
            q_BLNK,
            k_BLNK,
            v_BLNK,
            cumulative_gate_BLNK,
            raw_beta_BLN.float().sigmoid(),
            cu_seqlens=cu_seqlens,
            impl="fused" if backend is KDABackend.ATTN_GYM else "reference",
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
            if self.kernel.backend is KDABackend.ATTN_GYM and self.head_dim != 128:
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
        *,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        kernel_cu_seqlens = cu_seqlens if cu_seqlens.numel() > 2 else None
        kernel_cu_seqlens_host = (
            cu_seqlens_host if kernel_cu_seqlens is not None else None
        )
        raw_gate_BLNK = raw_gate_TNK.unsqueeze(0)
        raw_beta_BLN = raw_beta_TN.unsqueeze(0)
        mixed_qkv_BTC = torch.cat(
            (query_TC, key_TC, value_TC),
            dim=-1,
        ).unsqueeze(0)
        conv_weight_C1W = torch.cat(
            (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W),
            dim=0,
        )
        backend = self.kernel.resolve_backend(raw_gate_BLNK)
        if backend is KDABackend.ATTN_GYM:
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
                if kernel_cu_seqlens_host is None:
                    raise ValueError(
                        "KDA varlen convolution requires host sequence offsets."
                    )
                conv_output_BTC = torch.cat(
                    [
                        convolve(mixed_qkv_BTC[:, start:end])
                        for start, end in pairwise(kernel_cu_seqlens_host)
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
            raw_gate_BLNK,
            raw_beta_BLN,
            A_log_N,
            dt_bias_NK,
            cu_seqlens=kernel_cu_seqlens,
            cu_seqlens_host=kernel_cu_seqlens_host,
            backend=backend,
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

        if attention_masks is None:
            cu_seqlens = torch.tensor(
                [0, x_TD.shape[0]],
                dtype=torch.int32,
                device=x_TD.device,
            )
            cu_seqlens_host = None
        elif isinstance(attention_masks, VarlenMetadata):
            cu_seqlens = attention_masks.cu_seq_q
            cu_seqlens_host = attention_masks.cu_seq_q_host
        else:
            raise ValueError(
                "KDA attention_masks must be VarlenMetadata or None, "
                f"got {type(attention_masks).__name__}."
            )
        num_tokens = x_TD.shape[0]
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
            cu_seqlens_host=cu_seqlens_host,
        )

        out_float_TNK = F.rms_norm(
            out_TNK.float(),
            (self.head_dim,),
            self.output_norm.weight.float(),
            self.output_norm.eps,
        )
        output_gate_TNK = self.output_gate(x_TD).view_as(out_TNK).float()
        out_TD = self.output_proj(
            (out_float_TNK * output_gate_TNK.sigmoid()).to(out_TNK.dtype).flatten(-2)
        )
        return out_TD


KimiDeltaAttention = KDA
KimiKDAKernel = KDAKernel
