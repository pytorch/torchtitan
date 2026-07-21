# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from fla.modules.convolution import causal_conv1d as _fla_causal_conv1d
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule,
)
from torch import nn
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.experimental import local_map

from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_varlen_metadata_for_document,
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


def _torch_native_gated_delta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Standalone math reference for the gated delta rule recurrence.

    Sequential O(seqlen) loop — use FLA kernels for GPU efficiency.

    Args:
        q, k: (bs, seqlen, n_heads, key_head_dim)
        v: (bs, seqlen, n_heads, value_head_dim)
        g: (bs, seqlen, n_heads) — log-space decay, always negative
        beta: (bs, seqlen, n_heads) — update gate ∈ (0, 1)

    Returns:
        output: (bs, seqlen, n_heads, value_head_dim)
    """
    B, L, H, D_k = q.shape
    D_v = v.shape[-1]
    dtype = q.dtype

    # Upcast to float32 — recurrence accumulates over seqlen steps
    q = _l2norm(q.float(), dim=-1) * (D_k**-0.5)
    k = _l2norm(k.float(), dim=-1)
    v, g, beta = v.float(), g.float(), beta.float()

    output = torch.zeros(B, L, H, D_v, dtype=torch.float32, device=q.device)
    state = torch.zeros(B, H, D_k, D_v, dtype=torch.float32, device=q.device)

    for t in range(L):
        q_t = q[:, t]
        k_t = k[:, t]
        v_t = v[:, t]
        g_t = g[:, t].exp().unsqueeze(-1).unsqueeze(-1)
        b_t = beta[:, t].unsqueeze(-1)

        state = state * g_t
        kv_mem = torch.einsum("bhkv,bhk->bhv", state, k_t)
        delta = (v_t - kv_mem) * b_t
        state = state + torch.einsum("bhk,bhv->bhkv", k_t, delta)
        output[:, t] = torch.einsum("bhkv,bhk->bhv", state, q_t)

    return output.to(dtype)


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


class _RecurrentFwdChunkBwd(torch.autograd.Function):
    """Batch-invariant GDN: fla RECURRENT kernel for the forward, fla CHUNK for backward.

    The vLLM generator must use the recurrent kernel for decode (decode is inherently a
    per-token recurrence). For the trainer forward to be bitwise-identical to the
    generator it must use that SAME recurrent kernel, in fp32 and with a materialized
    zero initial state, and with cu_seqlens (varlen) -- so the USE_INITIAL_STATE and
    IS_VARLEN triton constexprs select the exact compiled kernel + fp reduction the
    generator hits. A pure-recurrent backward is O(seqlen) sequential and slow, so the
    backward recomputes the fla CHUNK kernel for efficient parallel gradients (chunk and
    recurrent compute the same function; only the forward value is swapped to recurrent).

    Inputs are the flattened [1, T, ...] varlen layout; cu_seqlens marks the packed
    per-sample boundaries so the recurrence resets per sample.
    """

    @staticmethod
    def forward(ctx, q, k, v, g, beta, cu_seqlens):  # pyrefly: ignore[bad-override]
        ctx.save_for_backward(q, k, v, g, beta)
        ctx.cu_seqlens = cu_seqlens
        n_seq = int(cu_seqlens.numel()) - 1
        h0 = q.new_zeros(n_seq, q.shape[2], q.shape[3], v.shape[3], dtype=torch.float32)
        with torch.no_grad():
            out, _ = _fla_fused_recurrent_gated_delta_rule(
                q.float(),
                k.float(),
                v.float(),
                g.float(),
                beta=beta.float(),
                initial_state=h0,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        return out.to(q.dtype)

    @staticmethod
    def backward(ctx, grad_out):  # pyrefly: ignore[bad-override]
        q, k, v, g, beta = ctx.saved_tensors
        # Recompute the chunk kernel with grad enabled and backprop through it.
        with torch.enable_grad():
            ins = [t.detach().requires_grad_(True) for t in (q, k, v, g, beta)]
            out_chunk = _fla_chunk_gated_delta_rule(
                ins[0],
                ins[1],
                ins[2],
                ins[3],
                ins[4],
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=ctx.cu_seqlens,
            )[0]
            grads = torch.autograd.grad(out_chunk, ins, grad_out)
        return grads[0], grads[1], grads[2], grads[3], grads[4], None


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
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Expand Q/K heads to match V when n_value_heads > n_key_heads
        if q.shape[2] != v.shape[2]:
            assert v.shape[2] % q.shape[2] == 0
            repeat = v.shape[2] // q.shape[2]
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)

        # Batch-invariant: recurrent-everywhere so the trainer forward matches the
        # vLLM generator's recurrent kernel bitwise (recurrent fwd, chunk bwd). The
        # BI caller passes the flattened [1, T] varlen layout + cu_seqlens.
        if is_in_batch_invariant_mode() and cu_seqlens is not None:
            return _RecurrentFwdChunkBwd.apply(q, k, v, g, beta, cu_seqlens)

        if self.backend == "torch_native":
            return _torch_native_gated_delta(q, k, v, g, beta)

        if self.backend == "fla_chunked":
            result = _fla_chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        elif self.backend == "fla_fused_recurrent":
            result = _fla_fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta=beta,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            raise ValueError(
                f"Unknown fla_backend '{self.backend}'. "
                "Valid: 'fla_chunked', 'fla_fused_recurrent', 'torch_native'."
            )

        # FLA kernels return (output, final_state); we only need output
        return result[0]


class GatedDeltaCore(Module):
    """Dense training GDN core: fused conv + gate + gated-delta recurrence.

    Parameterless; shares the flattened ``[T, ...]`` + ``cu_seqlens`` call
    signature with the paged ``VLLMGatedDeltaNetCore`` so ``GatedDeltaNet.core``
    can be swapped for it inside vLLM. Bitwise vs the pre-core dense path: non-BI
    uses a fused depthwise ``F.conv1d`` (exact vs separate convs) + fla chunk
    kernel; BI uses the fla fused conv + recurrent kernel (matching the generator).

    Legend: T = flattened tokens, C = 2*key_dim + value_dim, Hv = value heads,
    Dk = head_k_dim, Dv = head_v_dim.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        key_head_dim: int
        value_head_dim: int
        key_dim: int
        value_dim: int
        conv_kernel_size: int = 4
        activation: str = "silu"
        kernel: GatedDeltaKernel.Config

    def __init__(self, config: Config):
        super().__init__()
        self.key_head_dim = config.key_head_dim
        self.value_head_dim = config.value_head_dim
        self.key_dim = config.key_dim
        self.value_dim = config.value_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.activation = config.activation
        self.n_value_heads = config.value_dim // config.value_head_dim
        self.kernel = config.kernel.build()

    def forward(
        self,
        mixed_qkv_TC: torch.Tensor,
        a_THv: torch.Tensor,
        b_THv: torch.Tensor,
        conv_weight: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """conv + gated-delta recurrence. 3D [B, L, ...] in/out.

        Two paths, sharing the parameterized inputs (mixed_qkv_TC [B, L, C] pre-conv,
        a/b [B, L, Hv], conv_weight [C, W], A_log/dt_bias [Hv], cu_seqlens [n_seg+1]
        Replicate). Returns [B, L, Hv, Dv]. The q/k split + value-head reshape use the
        LOCAL channel/head counts (under TP mixed_qkv holds only this rank's heads;
        the q:k:v proportion is TP-invariant). No conv bias (config bias=False).

          - non-BI (pretraining): batched [B, L] depthwise F.conv1d + chunk kernel --
            BITWISE-identical to the dense pretraining path. The batch dim resets the
            recurrent state between rows and pretraining is unpacked, so cu_seqlens is
            unused (kept off the fla conv). This is the common model.py path.
          - BI (RL): flatten [B, L] -> [1, T] and run the fla VARLEN conv + recurrent
            kernel with cu_seqlens, so the conv window and recurrent state RESET at
            packed-sample boundaries (positions restart per sample) and match the
            paged generator bitwise.

        Under TP the sharding config (sharding.py) wraps this forward in local_map, so
        every input arrives LOCAL (per-rank) and both paths run head-parallel on local
        shards. Single-GPU: inputs are already plain tensors.
        """
        bs, seqlen, c = mixed_qkv_TC.shape
        # LOCAL q/k boundary (== global self.key_dim on a single GPU).
        key_dim = c * self.key_dim // (2 * self.key_dim + self.value_dim)

        if is_in_batch_invariant_mode():
            t = bs * seqlen
            mixed = mixed_qkv_TC.reshape(t, c)
            conv_out = _fla_causal_conv1d(
                mixed.unsqueeze(0),
                weight=conv_weight,
                bias=None,
                activation=self.activation,
                cu_seqlens=cu_seqlens,
            )
            if isinstance(conv_out, tuple):
                conv_out = conv_out[0]
            conv_out = conv_out.reshape(t, c)
            xq = conv_out[:, :key_dim].reshape(1, t, -1, self.key_head_dim)
            xk = conv_out[:, key_dim : 2 * key_dim].reshape(1, t, -1, self.key_head_dim)
            xv = conv_out[:, 2 * key_dim :].reshape(1, t, -1, self.value_head_dim)
            a = a_THv.reshape(t, -1)
            b = b_THv.reshape(t, -1)
            g = (-torch.exp(A_log.float()) * F.softplus(a.float() + dt_bias)).reshape(
                1, t, -1
            )
            beta = torch.sigmoid(b).reshape(1, t, -1)
            out = self.kernel(xq, xk, xv, g, beta, cu_seqlens=cu_seqlens)
            return out.reshape(bs, seqlen, -1, self.value_head_dim)

        # non-BI (pretraining): batched depthwise F.conv1d (preserves numerics).
        x = mixed_qkv_TC.transpose(1, 2)  # [B, C, L]
        x = F.pad(x, [self.conv_kernel_size - 1, 0])  # causal left pad
        x = F.conv1d(x, conv_weight.unsqueeze(1), None, groups=c)
        conv_out = F.silu(x).transpose(1, 2)  # [B, L, C]
        xq = conv_out[..., :key_dim].reshape(bs, seqlen, -1, self.key_head_dim)
        xk = conv_out[..., key_dim : 2 * key_dim].reshape(
            bs, seqlen, -1, self.key_head_dim
        )
        xv = conv_out[..., 2 * key_dim :].reshape(bs, seqlen, -1, self.value_head_dim)
        g = -torch.exp(A_log.float()) * F.softplus(a_THv.float() + dt_bias)
        beta = torch.sigmoid(b_THv)
        out = self.kernel(xq, xk, xv, g, beta)  # chunk [B, L], no cu_seqlens
        return out.reshape(bs, seqlen, -1, self.value_head_dim)


class GatedDeltaNet(Module):
    """Gated DeltaNet linear attention.

    Uses recurrent state + gated delta rule instead of softmax attention.
    No RoPE, no attention masks, different head structure from standard
    attention.
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
        core: GatedDeltaCore.Config
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

        self.norm = config.norm.build()
        self.out_proj = config.out_proj.build()

        self.n_value_heads = n_value_heads
        self.key_dim = config.in_proj_q.out_features
        self.value_dim = value_dim
        # The stateful conv + gated-delta recurrence live in a swappable core.
        # Training: a dense GatedDeltaCore (built here). Inside vLLM the generation
        # wrapper (_attach_gdn_cores) replaces it with a paged-cache
        # VLLMGatedDeltaNetCore -- same call signature -- for the unified model path.
        self.core = config.core.build()

    def _fused_conv_weight_bias(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fuse conv_q/k/v (depthwise, weight [C_i, 1, W]) into one [C, W] weight (+
        [C] bias) in q|k|v order, matching cat([xq, xk, xv]). Depthwise -> channel-wise
        concat is exact (one fused conv == three separate depthwise convs).

        Under TP the conv weights are Shard(0) (out-channels); a plain DTensor cat
        collapses to Replicate, so fuse on LOCAL shards via local_map -- each rank's
        fused weight is [convq_r|convk_r|convv_r], aligned per-channel with the
        similarly locally-fused mixed_qkv. squeeze(1) drops the size-1 in-channel dim
        (not sharded), so the Shard(0) placement carries through unchanged."""
        wq, wk, wv = self.conv_q.weight, self.conv_k.weight, self.conv_v.weight
        if isinstance(wq, DTensor):
            plc = wq.placements
            w = local_map(
                lambda a, b, c: torch.cat([a, b, c], dim=0).squeeze(1),
                out_placements=(plc,),
                in_placements=(plc, wk.placements, wv.placements),
                in_grad_placements=(plc, wk.placements, wv.placements),
                device_mesh=wq.device_mesh,
            )(
                wq,
                wk,  # pyrefly: ignore[bad-argument-count]
                wv,
            )
        else:
            w = torch.cat([wq, wk, wv], dim=0).squeeze(1)
        biases = [self.conv_q.bias, self.conv_k.bias, self.conv_v.bias]
        b = torch.cat(biases, dim=0) if all(x is not None for x in biases) else None
        return w, b

    def forward(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Gated DeltaNet, split into three parts for piecewise-cudagraph / TP.

        ``cu_seqlens`` marks packed-sample boundaries (from ``Qwen35Model`` via
        ``positions``) so the recurrent state + causal conv reset per sample; None
        for a single unpacked row.


          1. Input projections (compilable, captured in cudagraph).
          2. Core: conv + gated-delta recurrence via ``self.core`` -- a dense
             ``GatedDeltaCore`` in training, or the paged ``VLLMGatedDeltaNetCore``
             custom op inside vLLM (the eager graph-split boundary).
          3. Output gating + projection (compilable, captured in cudagraph).

        Both cores share the flattened ``[T, ...]`` + ``cu_seqlens`` signature, so
        the slot is swapped (``_attach_gdn_cores``) without changing this forward.
        """
        bs, seqlen, _ = x.shape

        # 1. Input projections (head-sharded under TP). mixed_qkv is PRE-conv: the
        #    core owns the conv (paged conv_state in the generation path). Keep the
        #    3D [B, L, ...] layout so DP/CP/TP map to distinct dims; the core
        #    flattens [B, L] -> [T] internally for the varlen kernels.
        q, k, v = self.in_proj_q(x), self.in_proj_k(x), self.in_proj_v(x)
        if isinstance(q, DTensor):
            # A plain DTensor cat collapses head-sharding to Replicate, so fuse
            # q|k|v on LOCAL shards via local_map -- each rank's mixed_qkv is
            # [q_r|k_r|v_r], aligned with the locally-fused conv weight; cat on the
            # last dim preserves the Shard placement.
            plc = q.placements
            mixed_qkv = local_map(
                lambda a, b, c: torch.cat([a, b, c], dim=-1),
                out_placements=(plc,),
                in_placements=(plc, k.placements, v.placements),
                in_grad_placements=(plc, k.placements, v.placements),
                device_mesh=q.device_mesh,
            )(
                q,
                k,  # pyrefly: ignore[bad-argument-count]
                v,
            )
        else:
            mixed_qkv = torch.cat([q, k, v], dim=-1)
        xz = self.in_proj_z(x)
        a = self.in_proj_a(x)
        b = self.in_proj_b(x)
        conv_weight, _ = self._fused_conv_weight_bias()
        # cu_seqlens marks packed-sample boundaries over the flattened [B*L] tokens.
        # None (mrope-only path) -> fall back to row boundaries (each rectangular
        # row is one segment); this also keeps a real tensor so batch-invariant mode
        # takes its recurrent route and TP has a shardable arg. Under TP pass it as a
        # Replicate DTensor so it carries a shard layout through the core's local_map
        # boundary. The paged generation core ignores it (reads vLLM attn_metadata).
        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                0, (bs + 1) * seqlen, seqlen, device=x.device, dtype=torch.int32
            )
        if isinstance(mixed_qkv, DTensor):
            cu_seqlens = DTensor.from_local(
                cu_seqlens,
                mixed_qkv.device_mesh,
                [Replicate()] * mixed_qkv.device_mesh.ndim,
            )

        # 2. Core: conv + gated-delta recurrence -> [B, L, n_value_heads, value_head_dim].
        # (The depthwise conv has no bias.)
        core_attn_out = self.core(
            mixed_qkv,
            a,
            b,
            conv_weight,
            self.A_log,
            self.dt_bias,
            cu_seqlens,
        )

        # 3. Output gating (RMSNormGated with z) + projection.
        xz = xz.reshape(bs, seqlen, -1, self.value_head_dim)
        output = self.norm(core_attn_out, xz)
        output = output.reshape(bs, seqlen, -1)
        return self.out_proj(output)


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
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        # wq is 2x wider: produces query + gate
        xq_gate = self.wq(x).view(bs, seqlen, -1, self.head_dim * 2)
        xq, gate = xq_gate.chunk(2, dim=-1)
        xk = self.wk(x).view(bs, seqlen, -1, self.head_dim)
        xv = self.wv(x).view(bs, seqlen, -1, self.head_dim)

        # QK norm (before RoPE)
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Partial RoPE: only first rotary_dim elements get positional encoding
        assert self.rotary_dim <= self.head_dim
        xq_rot, xq_pass = xq[..., : self.rotary_dim], xq[..., self.rotary_dim :]
        xk_rot, xk_pass = xk[..., : self.rotary_dim], xk[..., self.rotary_dim :]
        xq_rot, xk_rot = self.rope(xq_rot, xk_rot, positions)
        xq = torch.cat([xq_rot, xq_pass], dim=-1)
        xk = torch.cat([xk_rot, xk_pass], dim=-1)

        output = self.inner_attention(
            xq,
            xk,
            xv,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()

        # Output gating
        output = output * torch.sigmoid(gate)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


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
        x: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.attention_norm(x)
        if self.full_attn:
            h = self.attn(h, attention_masks, positions)
        else:
            # GatedDeltaNet: cu_seqlens marks packed-sample boundaries so its
            # recurrent state + causal conv reset per sample (full attention uses
            # the block-diagonal mask in attention_masks for the same purpose).
            h = self.attn(h, cu_seqlens)
        x = x + h

        h = self.ffn_norm(x)
        if self.moe_enabled:
            x = x + self.moe(h)
        else:
            x = x + self.feed_forward(h)
        return x


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
                    (l.delta_net for l in self.layers if l.delta_net is not None),
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
                enable_sp=parallelism.enable_sequence_parallel,
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
        inputs_embeds = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )

        if pixel_values is not None and grid_thw is not None:
            image_token_id = special_tokens["image_id"]
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
            video_token_id = special_tokens["video_id"]
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
        # cu_seqlens marks packed-sample boundaries (the RL trainer packs several
        # samples per row, restarting positions at 0). Computed once from positions
        # (reusing the document-varlen helper the full-attention masks use) and
        # shared across GatedDeltaNet layers so their recurrent state + causal conv
        # reset per sample. None for a single unpacked sequence. Not CP-aware (CP is
        # unsupported for GatedDeltaNet).
        cu_seqlens = None
        if positions is not None:
            cu_seqlens = create_varlen_metadata_for_document(positions).cu_seq_q
            if cu_seqlens.numel() <= 2:  # single unpacked sample -> non-varlen path
                cu_seqlens = None
        for layer in self.layers.values():
            x = layer(x, attention_masks, rope_positions, cu_seqlens)

        x = self.norm(x) if self.norm is not None else x
        if self._skip_lm_head:
            return x
        return self.lm_head(x) if self.lm_head is not None else x
