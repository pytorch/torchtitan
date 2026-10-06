# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Kimi Delta Attention using Attention Gym kernels."""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch_remat as remat
from attn_gym.linear._delta_rule.gate import _FusedGate
from attn_gym.linear.context_parallel import ContextParallelRouting
from attn_gym.linear.kda import bound_gate, chunk_kda
from attn_gym.linear.kda.fwd.triton.l2norm_fwd import _L2Norm, l2norm
from attn_gym.linear.kda.impl.cudnn import ChunkKdaCudnn
from attn_gym.linear.kda.impl.fused import _ChunkKDA
from attn_gym.linear.kda.masking import _MaskRows
from attn_gym.linear.short_conv import causal_conv1d
from attn_gym.linear.short_conv.cute import _ConfiguredShortConv, _ShortConv
from torch import nn

from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.nn_modules import Conv1d
from torchtitan.models.common.norm import GatedRMSNorm
from torchtitan.protocols.module import Module

from .attention import (
    create_varlen_metadata_for_document,
    InnerAttention,
    local_head_split,
    VarlenAttentionMetadata,
)

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels,
# H = attention heads, K = query/key head dimension, V = value head dimension,
# W = convolution kernel width.


@dataclass(frozen=True, slots=True)
class KDAAttentionMetadata:
    """Per-batch sequence metadata consumed by KDA."""

    varlen: VarlenAttentionMetadata | None
    num_conv_history_tokens: int

    def annotate_spmd_types(self) -> None:
        """Annotate sequence offsets."""
        if self.varlen is not None:
            self.varlen.annotate_spmd_types()


# The Attention Gym kernels run on rank-local heads inside InnerKDA's local
# SPMD region.
for _kernel_function in (
    _ShortConv,
    _ConfiguredShortConv,
    _FusedGate,
    _L2Norm,
    _ChunkKDA,
    ChunkKdaCudnn,
    _MaskRows,
):
    spmd.register_local_autograd_function(_kernel_function)


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
        q_1THK: torch.Tensor,
        k_1THK: torch.Tensor,
        v_1THV: torch.Tensor,
        raw_gate_1THK: torch.Tensor,
        raw_beta_1TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_1THK, k_1THK, gate_1THK, beta_1TH = self.prepare_inputs(
            q_1THK,
            k_1THK,
            raw_gate_1THK,
            raw_beta_1TH,
            A_log_H,
            dt_bias_HK,
        )
        output_1THV, _ = chunk_kda(
            q_1THK,
            k_1THK,
            v_1THV,
            gate_1THK,
            beta_1TH,
            cu_seqlens=cu_seqlens,
        )
        return output_1THV

    def prepare_inputs(
        self,
        q_1THK: torch.Tensor,
        k_1THK: torch.Tensor,
        raw_gate_1THK: torch.Tensor,
        raw_beta_1TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply the preprocessing shared by local and CP KDA cores."""
        if not q_1THK.is_cuda:
            raise RuntimeError("Attention Gym KDA requires CUDA tensors.")

        gate_1THK = bound_gate(
            raw_gate_1THK,
            # TODO: The long-term solution is to specify mixed precision per FQN
            # instead of per layer. https://github.com/pytorch/pytorch/issues/156784
            A_log_H.float(),
            dt_bias_HK.float(),
            lower_bound=self.lower_bound,
            impl="fused",
        )
        return (
            l2norm(q_1THK),
            l2norm(k_1THK),
            gate_1THK,
            raw_beta_1TH.float().sigmoid(),
        )


class InnerKDA(InnerAttention):
    """Run short convolution and KDA behind the vLLM replacement boundary."""

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        head_dim: int
        conv_kernel_size: int
        kernel: KDAKernel.Config

        def __post_init__(self):
            if self.head_dim != 128:
                raise ValueError(
                    "Attention Gym KDA requires head_dim=128, " f"got {self.head_dim}."
                )

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> KDAAttentionMetadata:
            """Build packed-sequence metadata consumed by KDA."""
            return KDAAttentionMetadata(
                varlen=create_varlen_metadata_for_document(
                    positions,
                    padding_mask=padding_mask,
                    max_num_documents=max_num_documents,
                    max_context_length=max_context_length,
                ),
                num_conv_history_tokens=self.conv_kernel_size - 1,
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
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        attention_metadata: KDAAttentionMetadata | None,
    ) -> torch.Tensor:
        varlen = attention_metadata.varlen if attention_metadata is not None else None
        cu_seqlens = varlen.cu_seq_q if varlen is not None else None
        return self.run_stages(
            query_TC,
            key_TC,
            value_TC,
            raw_gate_THK,
            raw_beta_TH,
            conv_q_weight_C1W,
            conv_k_weight_C1W,
            conv_v_weight_C1W,
            A_log_H,
            dt_bias_HK,
            cu_seqlens=cu_seqlens,
            routing=None,
        )

    def run_stages(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        """Run the stages shared by local and context-parallel KDA."""
        raw_gate_1THK = raw_gate_THK.unsqueeze(0)
        raw_beta_1TH = raw_beta_TH.unsqueeze(0)
        mixed_qkv_1TC = torch.cat(
            (query_TC, key_TC, value_TC),
            dim=-1,
        ).unsqueeze(0)
        conv_weight_C1W = torch.cat(
            (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W),
            dim=0,
        )
        conv_output_1TC = self.short_convolution(
            mixed_qkv_1TC,
            conv_weight_C1W,
            cu_seqlens=cu_seqlens,
            routing=routing,
        )

        q_1TC, k_1TC, v_1TC = conv_output_1TC.chunk(3, dim=-1)
        q_1THK, k_1THK, v_1THV = (
            tensor.unflatten(-1, (-1, self.head_dim))
            for tensor in (q_1TC, k_1TC, v_1TC)
        )
        output_1THV = self.kda_core(
            q_1THK,
            k_1THK,
            v_1THV,
            raw_gate_1THK,
            raw_beta_1TH,
            A_log_H,
            dt_bias_HK,
            cu_seqlens=cu_seqlens,
            routing=routing,
        )
        return output_1THV.squeeze(0)

    def short_convolution(
        self,
        qkv_1TC: torch.Tensor,
        conv_weight_C1W: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        del routing
        output_1TC = causal_conv1d(
            qkv_1TC,
            conv_weight_C1W[:, 0],
            activation="silu",
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
        )
        assert isinstance(output_1TC, torch.Tensor)
        return output_1TC

    def kda_core(
        self,
        q_1THK: torch.Tensor,
        k_1THK: torch.Tensor,
        v_1THV: torch.Tensor,
        raw_gate_1THK: torch.Tensor,
        raw_beta_1TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        del routing
        return self.kernel(
            q_1THK,
            k_1THK,
            v_1THV,
            raw_gate_1THK,
            raw_beta_1TH,
            A_log_H,
            dt_bias_HK,
            cu_seqlens=cu_seqlens,
        )


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
        output_norm: GatedRMSNorm.Config
        output_proj: Linear.Config

        def __post_init__(self):
            if self.num_heads < 1:
                raise ValueError(f"num_heads must be positive, got {self.num_heads}")
            if self.head_dim != 128:
                raise ValueError(
                    "Attention Gym KDA requires head_dim=128, " f"got {self.head_dim}."
                )
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
        attention_metadata: KDAAttentionMetadata | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del positions
        # All KDA input projections consume x, so gather once at their common
        # module boundary.
        x_TD = maybe_gather_tp_input(self, x_TD)

        if x_TD.ndim != 2:
            raise ValueError(
                f"KDA input must have shape [T, D], got {tuple(x_TD.shape)}."
            )

        raw_gate_THK = local_head_split(
            self.forget_b(self.forget_a(x_TD)),
            self.head_dim,
            cp_shard_dim=0,
        )
        raw_beta_TH = self.beta(x_TD)
        out_THV = remat.region(
            self.inner_kda,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            self.q_proj(x_TD),
            self.k_proj(x_TD),
            self.v_proj(x_TD),
            raw_gate_THK,
            raw_beta_TH,
            self.q_conv.weight,
            self.k_conv.weight,
            self.v_conv.weight,
            self.A_log,
            self.dt_bias,
            attention_metadata=attention_metadata,
        )

        output_gate_THV = local_head_split(
            self.output_gate(x_TD), self.head_dim, cp_shard_dim=0
        )
        # The gated output norm reads the inner_attention and output_gate projection
        # outputs with bare ops.
        remat.recompute_needs_tensor(out_THV, output_gate_THV)
        return self.output_proj(self.output_norm(out_THV, output_gate_THV).flatten(-2))
