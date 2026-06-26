# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule,
)
from torch import nn
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from torchtitan.models.common import Conv1d, FeedForward, Linear
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_varlen_metadata_for_document,
    VarlenAttention,
    VarlenMetadata,
)
from torchtitan.models.common.decoder import Decoder
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.module import Module

from .rope import MRoPE
from .sharding import set_qwen35_sharding_config
from .vision_encoder import Qwen35VisionEncoder


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 norm using rsqrt(sum(x²) + eps), not x/max(norm, eps) like F.normalize, to match FLA kernel."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _cu_seqlens_host_to_cpu_tensor(
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: tuple[int, ...],
) -> torch.Tensor:
    return torch.tensor(cu_seqlens_host, dtype=cu_seqlens.dtype, device="cpu")


def _cu_seqlens_to_list(cu_seqlens_host: tuple[int, ...]) -> list[int]:
    return list(cu_seqlens_host)


def _require_cu_seqlens_host(
    cu_seqlens_host: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if cu_seqlens_host is None:
        raise ValueError(
            "Qwen3.5 GatedDeltaNet varlen requires host-side cu_seqlens "
            "metadata. Build VarlenMetadata with include_host_offsets=True."
        )
    return cu_seqlens_host


def _torch_native_gated_delta(
    q_BLNK: torch.Tensor,
    k_BLNK: torch.Tensor,
    v_BLNV: torch.Tensor,
    g_BLN: torch.Tensor,
    beta_BLN: torch.Tensor,
) -> torch.Tensor:
    """Standalone math reference for the gated delta rule recurrence.

    Sequential O(seqlen) loop — use FLA kernels for GPU efficiency.

    Args:
        q_BLNK, k_BLNK: (batch, seq, n_heads, key_head_dim)
        v_BLNV: (batch, seq, n_heads, value_head_dim)
        g_BLN: (batch, seq, n_heads) -- log-space decay, always negative
        beta_BLN: (batch, seq, n_heads) -- update gate in (0, 1)

    Returns:
        output: (batch, seq, n_heads, value_head_dim)
    """
    B, L, N, K = q_BLNK.shape
    V = v_BLNV.shape[-1]
    dtype = q_BLNK.dtype

    # Upcast to float32 — recurrence accumulates over seqlen steps
    q_BLNK = _l2norm(q_BLNK.float(), dim=-1) * (K**-0.5)
    k_BLNK = _l2norm(k_BLNK.float(), dim=-1)
    v_BLNV, g_BLN, beta_BLN = v_BLNV.float(), g_BLN.float(), beta_BLN.float()

    out_BLNV = torch.zeros(B, L, N, V, dtype=torch.float32, device=q_BLNK.device)
    state_BNKV = torch.zeros(B, N, K, V, dtype=torch.float32, device=q_BLNK.device)

    for t in range(L):
        q_BNK = q_BLNK[:, t]
        k_BNK = k_BLNK[:, t]
        v_BNV = v_BLNV[:, t]
        g_BN11 = g_BLN[:, t].exp().unsqueeze(-1).unsqueeze(-1)
        beta_BN1 = beta_BLN[:, t].unsqueeze(-1)

        state_BNKV = state_BNKV * g_BN11
        kv_mem_BNV = torch.einsum("bnkv,bnk->bnv", state_BNKV, k_BNK)
        delta_BNV = (v_BNV - kv_mem_BNV) * beta_BN1
        state_BNKV = state_BNKV + torch.einsum("bnk,bnv->bnkv", k_BNK, delta_BNV)
        out_BLNV[:, t] = torch.einsum("bnkv,bnk->bnv", state_BNKV, q_BNK)

    return out_BLNV.to(dtype)


def _torch_native_gated_delta_varlen(
    q_BLNK: torch.Tensor,
    k_BLNK: torch.Tensor,
    v_BLNV: torch.Tensor,
    g_BLN: torch.Tensor,
    beta_BLN: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_host: tuple[int, ...],
) -> torch.Tensor:
    out_segments_BLNV: list[torch.Tensor] = []
    cu_seqlens_list = _cu_seqlens_to_list(cu_seqlens_host)
    for start, end in zip(cu_seqlens_list[:-1], cu_seqlens_list[1:], strict=False):
        out_segments_BLNV.append(
            _torch_native_gated_delta(
                q_BLNK[:, start:end],
                k_BLNK[:, start:end],
                v_BLNV[:, start:end],
                g_BLN[:, start:end],
                beta_BLN[:, start:end],
            )
        )
    return torch.cat(out_segments_BLNV, dim=1)


def _torch_native_causal_conv1d_varlen(
    x_BTD: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    conv_kernel_size: int,
    cu_seqlens_host: tuple[int, ...],
) -> torch.Tensor:
    out_segments_BTD: list[torch.Tensor] = []
    cu_seqlens_list = _cu_seqlens_to_list(cu_seqlens_host)
    for start, end in zip(cu_seqlens_list[:-1], cu_seqlens_list[1:], strict=False):
        x_segment_BDT = F.pad(
            x_BTD[:, start:end].transpose(1, 2),
            [conv_kernel_size - 1, 0],
        )
        out_segment_BTD = F.conv1d(
            x_segment_BDT,
            weight,
            None,
            groups=weight.size(0),
        ).transpose(1, 2)
        out_segments_BTD.append(out_segment_BTD)
    return F.silu(torch.cat(out_segments_BTD, dim=1))


def _causal_conv1d_varlen(
    x_BTD: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    conv_kernel_size: int,
    cu_seqlens_host: tuple[int, ...],
    cu_seqlens_cpu: torch.Tensor | None = None,
) -> torch.Tensor:
    if x_BTD.is_cuda:
        from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d

        if cu_seqlens_cpu is None:
            raise ValueError(
                "Qwen3.5 FLA varlen conv requires a CPU cu_seqlens tensor."
            )
        out_BTD, _ = _fla_causal_conv1d(
            x=x_BTD,
            weight=weight.squeeze(1),
            bias=None,
            activation="silu",
            backend="triton",
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )
        return out_BTD
    return _torch_native_causal_conv1d_varlen(
        x_BTD,
        weight,
        cu_seqlens,
        conv_kernel_size,
        cu_seqlens_host,
    )


class SharedExperts(FeedForward):
    """Qwen3.5 shared expert: SwiGLU FFN with a per-token sigmoid gate.

    The output is ``sigmoid(gate(x)) * ffn(x)``. Inherits ``w1/w2/w3`` from
    FeedForward so weight FQNs are unchanged. This gate is specific to
    Qwen3.5; other models use a plain ``FeedForward`` shared expert.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(FeedForward.Config):
        gate: Linear.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.gate = config.gate.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        return torch.sigmoid(self.gate(x)) * out


class OffsetRMSNorm(Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``.

    Weight is zero-initialized so the norm starts as identity-scaled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability in pow/rsqrt
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)


class RMSNormGated(Module):
    """Gated RMSNorm: ``silu(gate) * weight * norm(x)``.

    Takes ``(x, gate)`` separately. Weight is ones-initialized.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability in pow/rsqrt
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        x = (self.weight.float() * x).to(input_dtype)
        x = x * F.silu(gate.float())
        return x.to(input_dtype)


class GatedDeltaKernel(Module):
    """Stateless dispatch to FLA kernel or pure-torch fallback.

    Provides a module boundary for the sharding code to wrap forward with
    DTensor→local conversion — same pattern as FlexAttention. Handles Q/K
    head expansion for grouped linear attention internally so that
    repeat_interleave runs on local tensors under TP.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # "fla_chunked": parallel within chunks, fast for training (default)
        # "fla_fused_recurrent": token-by-token, lower memory for long sequences
        # "torch_native": pure-Python reference, for numerical testing only
        backend: Literal[
            "fla_chunked", "fla_fused_recurrent", "torch_native"
        ] = "fla_chunked"

    def __init__(self, config: Config):
        super().__init__()
        self.backend = config.backend

    def forward(
        self,
        xq_BLNK: torch.Tensor,
        xk_BLNK: torch.Tensor,
        xv_BLNV: torch.Tensor,
        g_BLN: torch.Tensor,
        beta_BLN: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_host: tuple[int, ...] | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Expand Q/K heads to match V when n_value_heads > n_key_heads
        if xq_BLNK.shape[2] != xv_BLNV.shape[2]:
            assert xv_BLNV.shape[2] % xq_BLNK.shape[2] == 0
            repeat = xv_BLNV.shape[2] // xq_BLNK.shape[2]
            xq_BLNK = xq_BLNK.repeat_interleave(repeat, dim=2)
            xk_BLNK = xk_BLNK.repeat_interleave(repeat, dim=2)

        if self.backend == "torch_native":
            if cu_seqlens is None:
                return _torch_native_gated_delta(
                    xq_BLNK,
                    xk_BLNK,
                    xv_BLNV,
                    g_BLN,
                    beta_BLN,
                )
            cu_seqlens_host = _require_cu_seqlens_host(cu_seqlens_host)
            return _torch_native_gated_delta_varlen(
                xq_BLNK,
                xk_BLNK,
                xv_BLNV,
                g_BLN,
                beta_BLN,
                cu_seqlens,
                cu_seqlens_host,
            )

        if cu_seqlens is not None and xq_BLNK.shape[0] != 1:
            raise ValueError(
                f"Gated DeltaNet varlen kernels require flattened inputs with "
                f"batch size 1, got batch size {xq_BLNK.shape[0]}."
            )

        if self.backend == "fla_chunked":
            if cu_seqlens is not None and cu_seqlens_cpu is None:
                raise ValueError(
                    "Qwen3.5 FLA varlen DeltaNet requires a CPU cu_seqlens tensor."
                )
            result = _fla_chunk_gated_delta_rule(
                xq_BLNK,
                xk_BLNK,
                xv_BLNV,
                g_BLN,
                beta_BLN,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
        elif self.backend == "fla_fused_recurrent":
            result = _fla_fused_recurrent_gated_delta_rule(
                xq_BLNK,
                xk_BLNK,
                xv_BLNV,
                g_BLN,
                beta=beta_BLN,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise ValueError(
                f"Unknown fla_backend '{self.backend}'. "
                "Valid: 'fla_chunked', 'fla_fused_recurrent', 'torch_native'."
            )

        # FLA kernels return (output, final_state); we only need output
        return result[0]


class GatedDeltaNet(Module):
    """Gated DeltaNet linear attention.

    Uses recurrent state + gated delta rule instead of softmax attention.
    No RoPE, different head structure from standard attention. When varlen
    metadata (``VarlenMetadata``) is provided -- i.e. under the ``varlen``
    attention backend -- conv and recurrent state are reset at document
    boundaries. Under other backends (e.g. ``flex``, which passes a
    ``BlockMask``) no reset occurs and the packed sequence is processed as a
    single continuous stream.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        key_head_dim: int
        value_head_dim: int
        conv_kernel_size: int = 4

        # Sub-module configs
        in_proj_q: Linear.Config
        in_proj_k: Linear.Config
        in_proj_v: Linear.Config
        in_proj_z: Linear.Config
        in_proj_a: Linear.Config
        in_proj_b: Linear.Config
        conv_q: Conv1d.Config
        conv_k: Conv1d.Config
        conv_v: Conv1d.Config
        kernel: GatedDeltaKernel.Config
        norm: RMSNormGated.Config
        out_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.key_head_dim = config.key_head_dim
        self.value_head_dim = config.value_head_dim
        self.conv_kernel_size = config.conv_kernel_size

        value_dim = config.in_proj_v.out_features

        self.in_proj_q = config.in_proj_q.build()
        self.in_proj_k = config.in_proj_k.build()
        self.in_proj_v = config.in_proj_v.build()
        self.in_proj_z = config.in_proj_z.build()
        self.in_proj_a = config.in_proj_a.build()
        self.in_proj_b = config.in_proj_b.build()

        self.conv_q = config.conv_q.build()
        self.conv_k = config.conv_k.build()
        self.conv_v = config.conv_v.build()

        n_value_heads = value_dim // config.value_head_dim
        self.A_log = nn.Parameter(torch.empty(n_value_heads))
        self.dt_bias = nn.Parameter(torch.empty(n_value_heads))

        self.kernel = config.kernel.build()
        self.norm = config.norm.build()
        self.out_proj = config.out_proj.build()

    @staticmethod
    def _local_map_conv(
        x: DTensor,
        conv: Conv1d,
        conv_fn: Callable[..., torch.Tensor],
        *extra_args: torch.Tensor,
    ) -> torch.Tensor:
        """Run a depthwise, channel-sharded conv on local shards via local_map.

        ``conv_fn`` receives the local (x, weight, *extra_args) tensors. Trailing
        ``extra_args`` (e.g. ``cu_seqlens``) are plain replicated tensors passed
        through with a ``None`` placement (unmapped). Input is channel-sharded
        and the weight is ``Shard(0)``; DTensor-ness and gradient placements are
        restored explicitly.

        TODO: Remove once the DTensor Conv1d dispatch fix for sharded groups
        lands in a released torch.
        """
        x_plc = x.placements
        w = conv.weight
        w_plc = w.placements  # pyrefly: ignore [missing-attribute]
        extra_plc = (None,) * len(extra_args)
        conv_dt = local_map(
            conv_fn,
            out_placements=(x_plc,),
            in_placements=(x_plc, w_plc, *extra_plc),
            in_grad_placements=(x_plc, w_plc, *extra_plc),
            device_mesh=x.device_mesh,
        )
        return conv_dt(x, w, *extra_args)  # pyrefly: ignore

    def _causal_conv(
        self,
        x_BLD: torch.Tensor,
        conv: Conv1d,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_host: tuple[int, ...] | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cu_seqlens is not None:
            cu_seqlens_host = _require_cu_seqlens_host(cu_seqlens_host)
            if isinstance(x_BLD, DTensor):

                def _conv_varlen(
                    x_local_BLD: torch.Tensor,
                    w_local: torch.Tensor,
                    cu_seqlens_local: torch.Tensor,
                ) -> torch.Tensor:
                    return _causal_conv1d_varlen(
                        x_local_BLD,
                        w_local,
                        cu_seqlens_local,
                        self.conv_kernel_size,
                        cu_seqlens_host,
                        cu_seqlens_cpu,
                    )

                return self._local_map_conv(x_BLD, conv, _conv_varlen, cu_seqlens)
            return _causal_conv1d_varlen(
                x_BLD,
                conv.weight,
                cu_seqlens,
                self.conv_kernel_size,
                cu_seqlens_host,
                cu_seqlens_cpu,
            )

        x_BDL = F.pad(x_BLD.transpose(1, 2), [self.conv_kernel_size - 1, 0])
        if isinstance(x_BDL, DTensor):

            def _conv(x_local_BDL: torch.Tensor, w_local: torch.Tensor) -> torch.Tensor:
                # groups == local out-channels (depthwise, channel-sharded)
                return F.conv1d(
                    x_local_BDL,
                    w_local,
                    None,
                    conv.stride,
                    conv.padding,
                    conv.dilation,
                    w_local.size(0),
                )

            x_BDL = self._local_map_conv(x_BDL, conv, _conv)
        else:
            x_BDL = conv(x_BDL)
        return F.silu(x_BDL).transpose(1, 2)

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        B, L, _ = x_BLD.shape
        cu_seqlens = None
        cu_seqlens_host = None
        cu_seqlens_cpu = None
        if isinstance(attention_masks, VarlenMetadata):
            cu_seqlens = attention_masks.cu_seq_q
            cu_seqlens_host = _require_cu_seqlens_host(attention_masks.cu_seq_q_host)
            if x_BLD.is_cuda:
                cu_seqlens_cpu = _cu_seqlens_host_to_cpu_tensor(
                    cu_seqlens,
                    cu_seqlens_host,
                )

        if cu_seqlens is None:
            kernel_B, kernel_L = B, L
        else:
            kernel_B, kernel_L = 1, B * L

        def _maybe_flatten(tensor: torch.Tensor) -> torch.Tensor:
            if cu_seqlens is None:
                return tensor
            return tensor.reshape(1, B * L, *tensor.shape[2:])

        # Shapes:
        #   xq_BLNK, xk_BLNK: (B, L, n_key_heads, key_head_dim)
        #   xv_BLNV, xz_BLNV: (B, L, n_value_heads, value_head_dim)
        #   xa_BLN, xb_BLN: (B, L, n_value_heads)
        xq_BLNK = self._causal_conv(
            _maybe_flatten(self.in_proj_q(x_BLD)),
            self.conv_q,
            cu_seqlens,
            cu_seqlens_host,
            cu_seqlens_cpu,
        ).view(kernel_B, kernel_L, -1, self.key_head_dim)
        xk_BLNK = self._causal_conv(
            _maybe_flatten(self.in_proj_k(x_BLD)),
            self.conv_k,
            cu_seqlens,
            cu_seqlens_host,
            cu_seqlens_cpu,
        ).view(kernel_B, kernel_L, -1, self.key_head_dim)
        xv_BLNV = self._causal_conv(
            _maybe_flatten(self.in_proj_v(x_BLD)),
            self.conv_v,
            cu_seqlens,
            cu_seqlens_host,
            cu_seqlens_cpu,
        ).view(kernel_B, kernel_L, -1, self.value_head_dim)
        xz_BLNV = _maybe_flatten(self.in_proj_z(x_BLD)).view(
            kernel_B, kernel_L, -1, self.value_head_dim
        )
        xa_BLN = _maybe_flatten(self.in_proj_a(x_BLD))
        xb_BLN = _maybe_flatten(self.in_proj_b(x_BLD))

        # Gating signals have shape (B, L, n_value_heads):
        #   g_BLN:    decay rate per head, always negative
        #   beta_BLN: update gate in (0, 1)
        g_BLN = -torch.exp(self.A_log.float()) * F.softplus(
            xa_BLN.float() + self.dt_bias
        )
        beta_BLN = torch.sigmoid(xb_BLN)

        out_BLNV = self.kernel(
            xq_BLNK,
            xk_BLNK,
            xv_BLNV,
            g_BLN,
            beta_BLN,
            cu_seqlens=cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )

        out_BLNV = self.norm(out_BLNV, xz_BLNV)

        # Merge value heads and restore (B, L); under varlen the kernel ran on a
        # flattened (1, B*L) layout, so this also unpacks the batch.
        out_BLD = out_BLNV.reshape(B, L, -1)
        return self.out_proj(out_BLD)


class Qwen35Attention(BaseAttention):
    """Full attention with output gating and partial RoPE for Qwen3.5.

    Differences from GQAttention:
    - wq is 2x wider: produces both query and sigmoid gate
    - Partial RoPE: only first ``rotary_dim`` elements get RoPE
    - Output gating: ``attn_output * sigmoid(gate)`` before ``wo``
    - QK norm uses OffsetRMSNorm

    Uses separate ``wq``/``wk``/``wv`` instead of the common fused ``qkv_linear``
    (so this subclasses ``BaseAttention``, not ``GQAttention``): the 2x-wide,
    gated ``wq`` doesn't fit a fused QKV projection that TP-shards by head.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        n_kv_heads: int
        head_dim: int
        rotary_dim: int
        rope: MRoPE.Config
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        wo: Linear.Config
        q_norm: OffsetRMSNorm.Config
        k_norm: OffsetRMSNorm.Config
        inner_attention: Module.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.wo = config.wo.build()

        self.rope = config.rope.build()

        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()

        self.scaling = self.head_dim**-0.5

        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x_BLD.shape

        # wq is 2x wider: produces query + gate
        xq_gate_BLN2H = self.wq(x_BLD).view(B, L, -1, self.head_dim * 2)
        xq_BLNH, gate_BLNH = xq_gate_BLN2H.chunk(2, dim=-1)
        xk_BLNH = self.wk(x_BLD).view(B, L, -1, self.head_dim)
        xv_BLNH = self.wv(x_BLD).view(B, L, -1, self.head_dim)

        # QK norm (before RoPE)
        xq_BLNH = self.q_norm(xq_BLNH)
        xk_BLNH = self.k_norm(xk_BLNH)

        # Partial RoPE: only first rotary_dim elements get positional encoding
        assert self.rotary_dim <= self.head_dim
        xq_BLNR, xq_BLNP = (
            xq_BLNH[..., : self.rotary_dim],
            xq_BLNH[..., self.rotary_dim :],
        )
        xk_BLNR, xk_BLNP = (
            xk_BLNH[..., : self.rotary_dim],
            xk_BLNH[..., self.rotary_dim :],
        )
        xq_BLNR, xk_BLNR = self.rope(xq_BLNR, xk_BLNR, positions)
        xq_BLNH = torch.cat([xq_BLNR, xq_BLNP], dim=-1)
        xk_BLNH = torch.cat([xk_BLNR, xk_BLNP], dim=-1)

        out_BLNH = self.inner_attention(
            xq_BLNH,
            xk_BLNH,
            xv_BLNH,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()

        # Output gating
        out_BLNH = out_BLNH * torch.sigmoid(gate_BLNH)
        out_BLD = out_BLNH.view(B, L, -1)
        return self.wo(out_BLD)


class Qwen35TransformerBlock(Module):
    """Hybrid transformer block for Qwen3.5.

    Each layer uses either full attention (Qwen35Attention) or linear
    attention (GatedDeltaNet), determined by which config is provided.
    Both types share the same FFN/MoE structure.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: Qwen35Attention.Config | None = None
        delta_net: GatedDeltaNet.Config | None = None
        feed_forward: Module.Config | None = None
        moe: Module.Config | None = None
        attention_norm: OffsetRMSNorm.Config
        ffn_norm: OffsetRMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.full_attn = config.attention is not None

        if self.full_attn:
            self.attn = config.attention.build()  # pyrefly: ignore [missing-attribute]
        else:
            assert config.delta_net is not None
            self.attn = config.delta_net.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            # pyrefly: ignore [missing-attribute]
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_BLD = self.attention_norm(x_BLD)
        if self.full_attn:
            h_BLD = self.attn(h_BLD, attention_masks, positions)
        else:
            h_BLD = self.attn(h_BLD, attention_masks)
        x_BLD = x_BLD + h_BLD

        h_BLD = self.ffn_norm(x_BLD)
        if self.moe_enabled:
            x_BLD = x_BLD + self.moe(h_BLD)
        else:
            x_BLD = x_BLD + self.feed_forward(h_BLD)
        return x_BLD


class Qwen35Model(Decoder):
    """Qwen3.5: Multimodal model with hybrid attention.

    Combines a hybrid decoder (GatedDeltaNet linear attention + full
    attention with output gating and partial RoPE) with a Vision
    Transformer encoder for multimodal understanding.

    Key architectural features:
    - Hybrid attention: 75% GatedDeltaNet (linear) + 25% full attention
    - Output gating on full attention: ``attn_out * sigmoid(gate)``
    - Partial RoPE: only first ``rotary_dim`` elements get positional encoding
    - OffsetRMSNorm: ``(1 + weight) * norm(x)`` with zero-init weight
    - MRoPE: 3D (temporal/height/width) position IDs for multimodal batches;
      text batches use the plain 1D positions
    - MoE variant: routed experts + shared expert with sigmoid gate

    MRoPE positions (``mrope_positions``, shape ``(batch, seq, 3)``) are built by
    the dataloader and forwarded to every pipeline stage, so RoPE stays consistent
    across stages even though the raw vision inputs (``pixel_values``/``grid_thw``)
    only reach the first stage. Text batches carry no ``mrope_positions`` and use
    the 2D ``positions`` instead.

    Forward pass flow::

        forward(tokens, pixel_values, grid_thw, mrope_positions, ...)
          │
          ├─ _prepare_multimodal_embeds
          │    ├─ tok_embeddings(tokens)              → text embeddings
          │    ├─ _get_vision_embeds(pixel_values)     → vision embeddings
          │    │    └─ vision_encoder(pixel_values)     → merge patches
          │    ├─ _get_vision_positions             → locate vision regions
          │    └─ _scatter_vision_embeds                → scatter into text sequence
          │
          └─ transformer layers (hybrid), each given (mrope_positions or positions)
               └─ for each layer:
                    ├─ full attention (every Nth):  QK-norm → partial RoPE → SDPA → gate
                    │    (the layer's MRoPE builds the cos/sin cache from positions)
                    └─ GatedDeltaNet (others):      Conv1d → gated delta rule → gated norm
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        vision_encoder: Qwen35VisionEncoder.Config

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                dn_cfg = next(
                    (
                        layer_cfg.delta_net
                        for layer_cfg in self.layers
                        if layer_cfg.delta_net is not None
                    ),
                    None,
                )
                if dn_cfg is not None:
                    n_key_heads = dn_cfg.in_proj_q.out_features // dn_cfg.key_head_dim
                    n_value_heads = (
                        dn_cfg.in_proj_v.out_features // dn_cfg.value_head_dim
                    )
                    if n_key_heads % tp != 0 or n_value_heads % tp != 0:
                        raise ValueError(
                            f"tensor_parallel_degree ({tp}) must divide "
                            f"n_key_heads ({n_key_heads}) and "
                            f"n_value_heads ({n_value_heads})."
                        )

            set_qwen35_sharding_config(
                self,
                enable_ep=parallelism.expert_parallel_degree > 1,
            )

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            attn_cfg = self.first_attention
            # pyrefly: ignore [missing-attribute]
            n_heads = attn_cfg.n_heads
            # pyrefly: ignore [missing-attribute]
            head_dim = attn_cfg.head_dim
            return get_moe_model_nparams_and_flops(
                self,
                model,
                n_heads,
                2 * head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__(config)

        self.vision_encoder = config.vision_encoder.build()
        self.spatial_merge_size = config.vision_encoder.spatial_merge_size

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> AttentionMasksType | None:
        attn_config = self.config.first_attention
        if attn_config is not None and isinstance(
            attn_config.inner_attention, VarlenAttention.Config
        ):
            return create_varlen_metadata_for_document(
                positions,
                include_host_offsets=True,
            )
        return super().get_attention_masks(positions)

    def _get_vision_positions(
        self,
        tokens: torch.Tensor,
        num_tokens_per_item: torch.Tensor,
        vision_token_id: int,
    ) -> list[tuple[int, int, int, int]]:
        """Compute (item_idx, sample_idx, vision_start, n_tokens) for each vision item.

        Finds where each contiguous run of vision placeholder tokens starts
        in the text sequence.

        Args:
            tokens: Token IDs (batch, seq_len)
            num_tokens_per_item: (num_items,) actual tokens per vision item
            vision_token_id: Placeholder token ID

        Returns:
            List of (item_idx, sample_idx, vision_start, n_tokens) tuples
        """
        vision_mask = tokens == vision_token_id
        flat_mask = vision_mask.view(-1)
        prev_mask = torch.cat(
            [torch.zeros(1, dtype=torch.bool, device=flat_mask.device), flat_mask[:-1]]
        )
        region_starts = torch.where(flat_mask & ~prev_mask)[0]
        seq_len = tokens.shape[1]

        positions = []
        for i in range(num_tokens_per_item.shape[0]):
            start = int(region_starts[i].item())
            n_tokens = int(num_tokens_per_item[i].item())
            positions.append((i, start // seq_len, start % seq_len, n_tokens))
        return positions

    def _get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run vision encoder and return padded embeddings with token counts.

        Args:
            pixel_values: Padded patches (num_items, max_num_patch, patch_dim)
            grid_thw: Grid dimensions (num_items, 3) for [t, h, w]

        Returns:
            merged_embeds: (num_items, max_tokens, dim) padded vision embeddings
            num_tokens_per_item: (num_items,) actual token count per item
        """
        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        merged_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)

        merge_unit = self.vision_encoder.spatial_merge_unit
        num_tokens_per_item = grid_thw.prod(-1) // merge_unit

        return merged_embeds, num_tokens_per_item

    def _scatter_vision_embeds(
        self,
        inputs_embeds: torch.Tensor,
        *,
        merged_embeds: torch.Tensor,
        vision_positions: list[tuple[int, int, int, int]],
    ) -> torch.Tensor:
        """Scatter vision embeddings into text embeddings at placeholder positions.

        Copies directly from the padded vision encoder output into the text
        sequence.

        Args:
            inputs_embeds: Text embeddings (batch, seq_len, dim)
            merged_embeds: Padded vision embeddings (num_items, max_tokens, dim)
            vision_positions: List of (item_idx, sample_idx, vision_start, n_tokens)

        Returns:
            Updated embeddings
        """
        for item_idx, sample_idx, vision_start, n_tokens in vision_positions:
            inputs_embeds[
                sample_idx, vision_start : vision_start + n_tokens, :
            ] = merged_embeds[item_idx, :n_tokens, :]
        return inputs_embeds

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int],
    ) -> torch.Tensor:
        """Embed tokens, run vision encoder, scatter vision into text.

        Args:
            tokens: Input token IDs (batch_size, seq_len)
            pixel_values: Image patches or None
            pixel_values_videos: Video patches or None
            grid_thw: Grid dimensions for images or None
            grid_thw_videos: Grid dimensions for videos or None
            special_tokens: Special token definitions

        Returns:
            (batch, seq_len, dim) embeddings with vision tokens scattered in
        """
        image_token_id = special_tokens["image_id"]
        video_token_id = special_tokens["video_id"]

        inputs_embeds = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )

        if pixel_values is not None and grid_thw is not None:
            merged_embeds, num_tokens = self._get_vision_embeds(
                pixel_values, grid_thw=grid_thw
            )
            image_positions = self._get_vision_positions(
                tokens, num_tokens, image_token_id
            )
            if image_positions:
                inputs_embeds = self._scatter_vision_embeds(
                    inputs_embeds,
                    merged_embeds=merged_embeds,
                    vision_positions=image_positions,
                )

        if pixel_values_videos is not None and grid_thw_videos is not None:
            merged_embeds, num_tokens = self._get_vision_embeds(
                pixel_values_videos, grid_thw=grid_thw_videos
            )
            video_positions = self._get_vision_positions(
                tokens, num_tokens, video_token_id
            )
            if video_positions:
                inputs_embeds = self._scatter_vision_embeds(
                    inputs_embeds,
                    merged_embeds=merged_embeds,
                    vision_positions=video_positions,
                )

        return inputs_embeds

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
        mrope_positions: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
    ):
        if self.tok_embeddings is not None:
            x = self._prepare_multimodal_embeds(
                tokens,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                grid_thw=grid_thw,
                grid_thw_videos=grid_thw_videos,
                special_tokens=special_tokens,  # pyrefly: ignore [bad-argument-type]
            )
        else:
            x = tokens

        # 3D MRoPE positions for multimodal batches, else 2D text positions.
        rope_positions = mrope_positions if mrope_positions is not None else positions
        assert rope_positions is not None
        for layer in self.layers.values():
            x = layer(x, attention_masks, rope_positions)

        x = self.norm(x) if self.norm is not None else x
        if self._skip_lm_head:
            return x
        return self.lm_head(x) if self.lm_head is not None else x
