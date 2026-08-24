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
from torch.distributed.tensor import DTensor
from fla.ops.kda import chunk_kda
from torch import nn

from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import AttentionMasksType
from torchtitan.models.kimi_k3.dtensor_ops import to_local_if_dtensor
from torchtitan.models.kimi_k3.sharding import (
    contract_for_mode,
    cp_all_to_all_headseq,
    ULYSSES,
)
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
        cp_mode: str = "kcp"

    # Set by apply_cp_kimi_k3; None means the layer runs without CP.
    _cp_group = None

    def __init__(self, config: Config):
        super().__init__()
        self.cp_mode = config.cp_mode
        # Validate against the declared contracts rather than restating the
        # accepted spellings here.
        contract_for_mode(self.cp_mode)
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
            return (
                self._forward_kcp(x_TD, cp_group)
                if self.cp_mode == "kcp"
                else self._forward_ulysses(x_TD, cp_group)
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

        # The kernel is fla triton and does not dispatch through DTensor.
        # Under TP these arrive wrapped, and handing a DTensor to it produces
        # an illegal memory access rather than anything legible, so the unwrap
        # happens at the call site and the result is re-wrapped for the
        # module's declared output layout.
        out_THV = self.kernel(
            to_local_if_dtensor(q_THK).unsqueeze(0),
            to_local_if_dtensor(k_THK).unsqueeze(0),
            to_local_if_dtensor(v_THV).unsqueeze(0),
            to_local_if_dtensor(forget_THK).unsqueeze(0),
            to_local_if_dtensor(beta_TH).unsqueeze(0),
            to_local_if_dtensor(self.A_log),
            to_local_if_dtensor(self.dt_bias),
        ).squeeze(0)
        if isinstance(q_THK, DTensor):
            out_THV = DTensor.from_local(
                out_THV, q_THK.device_mesh, q_THK.placements, run_check=False
            )
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
        from torchtitan.models.kimi_k3.kcp import build_kcp_context, conv_with_halo

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

    def _forward_ulysses(self, x_TD: torch.Tensor, cp_group) -> torch.Tensor:
        """Ulysses CP forward: trade the sharded axis, sequence for heads.

        Projections run sequence-local, one fused all-to-all moves everything to
        full-sequence head-subset layout, and the convolutions then run on the
        full sequence for this rank's heads -- so no halo is needed here, unlike
        KCP. The convolutions are depthwise, so restricting them to a contiguous
        head subset is a contiguous channel slice of the weight and is exact.

        Shape suffixes beyond the file legend: L local sequence (T/cp), G this
        rank's head count (H/cp), W the packed per-head channel width.
        """
        # Head divisibility is checked at wiring time, against tp*cp rather
        # than cp -- under TP the head axis is already split once. Repeating a
        # cp-only version of that test here would reject configurations that
        # run.
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)
        t_loc = x_TD.shape[0]
        num_heads, head_dim = self.num_heads, self.head_dim
        h_cp = num_heads // cp_size
        h0 = cp_rank * h_cp

        def heads(t_LC: torch.Tensor) -> torch.Tensor:
            return t_LC.view(t_loc, num_heads, head_dim)

        # 1) Sequence-local projections, pre-convolution.
        q_LHK = heads(self.q_proj(x_TD))
        k_LHK = heads(self.k_proj(x_TD))
        v_LHV = heads(self.v_proj(x_TD))
        forget_LHK = heads(self.forget_b(self.forget_a(x_TD)))
        gate_LHV = heads(self.output_gate(x_TD))
        beta_LH1 = self.beta(x_TD).unsqueeze(-1)

        # 2) One fused all-to-all instead of six.
        packed_LHW = torch.cat(
            [q_LHK, k_LHK, v_LHV, forget_LHK, gate_LHV, beta_LH1], dim=-1
        )
        src_dim, dst_dim = ULYSSES.in_dims()
        packed_TGW = cp_all_to_all_headseq(
            packed_LHW, cp_group, src_dim=src_dim, dst_dim=dst_dim
        )
        q_TGK, k_TGK, v_TGV, forget_TGK, gate_TGV, beta_TG1 = torch.split(
            packed_TGW, [head_dim] * 5 + [1], dim=-1
        )
        t_full = t_loc * cp_size

        # 3) Causal convolution on the full sequence, channels sliced to this
        # rank's heads.
        def conv_subset(conv: Conv1d, x_TGK: torch.Tensor) -> torch.Tensor:
            lo, hi = h0 * head_dim, (h0 + h_cp) * head_dim
            w_C1W = to_local_if_dtensor(conv.weight)[lo:hi]
            b_C = (
                to_local_if_dtensor(conv.bias)[lo:hi] if conv.bias is not None else None
            )
            x_1CT = F.pad(
                x_TGK.reshape(t_full, h_cp * head_dim).T.unsqueeze(0),
                (self.conv_kernel_size - 1, 0),
            )
            y_1CT = F.conv1d(x_1CT, w_C1W, b_C, groups=h_cp * head_dim)
            return F.silu(y_1CT).squeeze(0).T.view(t_full, h_cp, head_dim)

        q_TGK = conv_subset(self.q_conv, q_TGK)
        k_TGK = conv_subset(self.k_conv, k_TGK)
        v_TGV = conv_subset(self.v_conv, v_TGV)

        # 4) The scan runs on this rank's heads over the full sequence, so
        # A_log and dt_bias are sliced to the same subset.
        out_TGV = self.kernel(
            q_TGK.unsqueeze(0),
            k_TGK.unsqueeze(0),
            v_TGV.unsqueeze(0),
            forget_TGK.unsqueeze(0),
            beta_TG1.squeeze(-1).float().unsqueeze(0),
            to_local_if_dtensor(self.A_log)[h0 : h0 + h_cp],
            to_local_if_dtensor(self.dt_bias)[h0 : h0 + h_cp],
        ).squeeze(0)
        out_TGV = self.output_norm(out_TGV, gate_TGV)

        # 5) Back to sequence-sharded full-head layout.
        out_src_dim, out_dst_dim = ULYSSES.out_dims()
        out_LHV = cp_all_to_all_headseq(
            out_TGV, cp_group, src_dim=out_src_dim, dst_dim=out_dst_dim
        )
        return self.output_proj(out_LHV.reshape(t_loc, num_heads * head_dim))
