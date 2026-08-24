# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention using Attention Gym kernels."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from torchtitan.models.common.attention import AttentionMasksType, VarlenMetadata
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import Conv1d, RMSNorm
from torchtitan.protocols.module import Module

try:
    import attn_gym.linear.kda as _attn_gym_kda
except ImportError:
    _attn_gym_kda = None


def _require_attention_gym():
    if _attn_gym_kda is None:
        raise ImportError("KDA requires Attention Gym: pip install 'attn-gym[linear]'")
    return _attn_gym_kda


class KDAKernel(Module):
    """Apply KDA preprocessing and the Attention Gym kernel."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        lower_bound: float = -5.0

        def __post_init__(self):
            if not -5.0 <= self.lower_bound < 0.0:
                raise ValueError(
                    "KDA lower_bound must be in the safe range [-5, 0), "
                    f"got {self.lower_bound}."
                )

    def __init__(self, config: Config):
        super().__init__()
        self.lower_bound = config.lower_bound

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
        attention_gym_kda = _require_attention_gym()
        capability = (
            torch.cuda.get_device_capability(q_BLNK.device) if q_BLNK.is_cuda else None
        )
        if (
            not q_BLNK.is_cuda
            or q_BLNK.shape[-1] != 128
            or capability not in {(10, 0), (10, 3)}
        ):
            raise RuntimeError(
                "Attention Gym KDA requires a CUDA tensor with head_dim=128 on "
                "Blackwell SM100/SM103; got "
                f"device={q_BLNK.device}, shape={tuple(q_BLNK.shape)}, "
                f"capability={capability}."
            )

        output_BLNK, _ = attention_gym_kda.chunk_kda(
            attention_gym_kda.l2norm(q_BLNK),
            attention_gym_kda.l2norm(k_BLNK),
            v_BLNK,
            attention_gym_kda.bounded_gate_cumsum(
                raw_gate_BLNK.to(torch.bfloat16),
                A_log_N.float(),
                dt_bias_NK.float(),
                chunk_size=64,
                lower_bound=self.lower_bound,
                cu_seqlens=cu_seqlens,
            ),
            raw_beta_BLN.float().sigmoid(),
            cu_seqlens=cu_seqlens,
        )
        return output_BLNK


class InnerKDA(Module):
    """Run short convolution and KDA behind the vLLM replacement boundary."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        head_dim: int
        kernel: KDAKernel.Config

        def __post_init__(self):
            if self.head_dim != 128:
                raise ValueError(
                    "Attention Gym KDA requires head_dim=128, " f"got {self.head_dim}."
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
        conv_output_BTC = _require_attention_gym().causal_conv1d(
            mixed_qkv_BTC,
            conv_weight_C1W[:, 0],
            activation="silu",
            cu_seqlens=kernel_cu_seqlens,
        )

        q_BTC, k_BTC, v_BTC = conv_output_BTC.chunk(3, dim=-1)
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
        elif isinstance(attention_masks, VarlenMetadata):
            cu_seqlens = attention_masks.cu_seq_q
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
        )

        out_float_TNK = F.rms_norm(
            out_TNK.float(),
            (self.head_dim,),
            self.output_norm.weight.float(),
            self.output_norm.eps,
        )
        output_gate_TNK = self.output_gate(x_TD).view_as(out_TNK).float()
        return self.output_proj(
            (out_float_TNK * output_gate_TNK.sigmoid()).to(out_TNK.dtype).flatten(-2)
        )


KimiDeltaAttention = KDA
