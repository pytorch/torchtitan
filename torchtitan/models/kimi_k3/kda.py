# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention modules for Kimi K3."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from fla.ops.kda import chunk_kda
from torch import nn

from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.protocols.module import Module

# Shape suffixes:
# T = packed tokens, D = model dimension, H = heads,
# K = key head dimension, V = value head dimension, C = projection channels.


class KimiRMSNormGated(Module):
    """Per-head RMSNorm followed by a sigmoid output gate."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-5

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_float = x_float * torch.rsqrt(variance + self.eps)
        x_float = self.weight.float() * x_float
        return (x_float * torch.sigmoid(gate.float())).to(input_dtype)


class KimiKDAKernel(Module):
    """Stateless dispatch to FLA's chunked KDA kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        lower_bound: float | None = -5.0

    def __init__(self, config: Config):
        super().__init__()
        self.lower_bound = config.lower_bound
        if self.lower_bound is not None and not (-5.0 <= self.lower_bound < 0.0):
            raise ValueError("KDA lower_bound must be in the safe range [-5, 0).")

    def forward(
        self,
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        v_BLHV: torch.Tensor,
        gate_BLHK: torch.Tensor,
        beta_BLH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
    ) -> torch.Tensor:
        out_BLHV, _ = chunk_kda(
            q_BLHK,
            k_BLHK,
            v_BLHV,
            gate_BLHK,
            beta_BLH,
            A_log=A_log_H,
            dt_bias=dt_bias_HK.reshape(-1),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self.lower_bound is not None,
            lower_bound=self.lower_bound,
        )
        return out_BLHV


class KimiDeltaAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
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
        kernel: Module.Config
        output_norm: KimiRMSNormGated.Config
        output_proj: Linear.Config

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
        self.kernel = config.kernel.build()
        self.output_norm = config.output_norm.build()
        self.output_proj = config.output_proj.build()

        self.A_log = nn.Parameter(torch.empty(config.num_heads))
        self.dt_bias = nn.Parameter(torch.empty(config.num_heads, config.head_dim))

    def _causal_conv(self, x_TC: torch.Tensor, conv: Conv1d) -> torch.Tensor:
        x_1CT = F.pad(x_TC.T.unsqueeze(0), (self.conv_kernel_size - 1, 0))
        return F.silu(conv(x_1CT)).squeeze(0).T

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        if attention_masks is not None:
            raise NotImplementedError(
                "Kimi K3 reference KDA does not support packed-document masks."
            )

        num_tokens = x_TD.shape[0]
        q_THK = self._causal_conv(self.q_proj(x_TD), self.q_conv).view(
            num_tokens, self.num_heads, self.head_dim
        )
        k_THK = self._causal_conv(self.k_proj(x_TD), self.k_conv).view(
            num_tokens, self.num_heads, self.head_dim
        )
        v_THV = self._causal_conv(self.v_proj(x_TD), self.v_conv).view(
            num_tokens, self.num_heads, self.head_dim
        )
        forget_THK = self.forget_b(self.forget_a(x_TD)).view(
            num_tokens, self.num_heads, self.head_dim
        )
        beta_TH = self.beta(x_TD).float()

        out_THV = self.kernel(
            q_THK.unsqueeze(0),
            k_THK.unsqueeze(0),
            v_THV.unsqueeze(0),
            forget_THK.unsqueeze(0),
            beta_TH.unsqueeze(0),
            self.A_log,
            self.dt_bias,
        ).squeeze(0)
        output_gate_THV = self.output_gate(x_TD).view(
            num_tokens, self.num_heads, self.head_dim
        )
        out_THV = self.output_norm(out_THV, output_gate_THV)
        return self.output_proj(out_THV.reshape(num_tokens, -1))
