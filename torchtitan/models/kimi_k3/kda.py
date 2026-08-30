# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention modules for Kimi K3."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
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
        cp_context=None,
    ) -> torch.Tensor:
        # cp_context turns the scan into fla's prefix-scan over rank-local
        # fragments. output_final_state is unsupported there, and unneeded in
        # training: the final state only matters for decoding.
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
            **(
                {"cp_context": cp_context, "cu_seqlens": cp_context.cu_seqlens}
                if cp_context is not None
                else {}
            ),
        )
        return out_BLHV


def conv_with_halo(conv, x_local, cp_context, activation: str | None = None):
    """Depthwise causal conv on a sequence-sharded input, exactly: the CP op
    exchanges the previous rank's tail as a fixed-size halo. ``activation``
    defaults to ``conv.activation`` (fla's ``ShortConvolution`` carries one;
    a plain ``nn.Conv1d`` does not)."""
    from attn_gym.linear.kda.fla_cp import causal_conv1d_cp

    return causal_conv1d_cp(
        x_local,
        conv.weight,
        conv.bias,
        cp_context,
        activation=getattr(conv, "activation", None)
        if activation is None
        else activation,
    )


def build_kcp_context(
    seq_len_local: int,
    group,
    device,
    conv1d_kernel_size: int | None = None,
    cu_seqlens=None,
):
    """CP context for one evenly split sequence. ``cu_seqlens`` must be
    GLOBAL boundaries of the packed sequence; the default is one document
    spanning the whole sequence, matching the non-CP call sites, which also
    pass no boundaries."""
    from attn_gym.linear.kda.fla_cp import build_fla_cp_context

    return build_fla_cp_context(
        seq_len_local,
        group,
        device,
        conv1d_kernel_size=conv1d_kernel_size,
        cu_seqlens=cu_seqlens,
    )


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

    # Set by apply_cp_kimi_k3; None means the layer runs without CP.
    _cp_group = None

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

        cp_group = self._cp_group
        if cp_group is not None and dist.get_world_size(cp_group) > 1:
            return self._forward_kcp(x_TD, cp_group)

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

    def _forward_kcp(self, x_TD: torch.Tensor, cp_group) -> torch.Tensor:
        """KCP forward: the sequence stays sharded (report sec 5.1.2).

        No rank holds the full sequence. The two cross-rank dependencies have
        different structure and are handled separately: the causal convolutions
        need only the previous rank's tail, one fixed-size halo; the delta-rule
        recurrence needs the true incoming state, which does not decompose by
        summation, so fla's cp_context prefix-scans over (cumulative transition,
        zero-started state) fragments.

        The folded token stream is already one packed sequence, which is exactly
        what fla's CP ops assume, so this path has no batch loop.
        """
        t_loc = x_TD.shape[0]
        # One context serves both the conv halo and the recurrence; the conv
        # needs the kernel width, the recurrence ignores it.
        ctx = build_kcp_context(
            t_loc,
            cp_group,
            x_TD.device,
            conv1d_kernel_size=self.conv_kernel_size,
        )

        def conv(proj, conv_module) -> torch.Tensor:
            # fla's CP conv wants [1, T, C] and applies the activation itself;
            # the reference model applies SiLU outside its Conv1d, so the name
            # is passed explicitly.
            y_1TC = conv_with_halo(
                conv_module, proj(x_TD).unsqueeze(0), ctx, activation="silu"
            )
            return y_1TC.squeeze(0)

        q_THK = conv(self.q_proj, self.q_conv).view(
            t_loc, self.num_heads, self.head_dim
        )
        k_THK = conv(self.k_proj, self.k_conv).view(
            t_loc, self.num_heads, self.head_dim
        )
        v_THV = conv(self.v_proj, self.v_conv).view(
            t_loc, self.num_heads, self.head_dim
        )
        forget_THK = self.forget_b(self.forget_a(x_TD)).view(
            t_loc, self.num_heads, self.head_dim
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
            cp_context=ctx,
        ).squeeze(0)
        output_gate_THV = self.output_gate(x_TD).view(
            t_loc, self.num_heads, self.head_dim
        )
        out_THV = self.output_norm(out_THV, output_gate_THV)
        return self.output_proj(out_THV.reshape(t_loc, -1))
