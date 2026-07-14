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
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.experimental import local_map

from torchtitan.models.common import Conv1d, FeedForward, Linear
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
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Standalone math reference for the gated delta rule recurrence.

    Sequential O(seqlen) loop -- use FLA kernels for GPU efficiency.

    Args:
        q, k: (bs, seqlen, n_heads, key_head_dim)
        v: (bs, seqlen, n_heads, value_head_dim)
        g: (bs, seqlen, n_heads) -- log-space decay, always negative
        beta: (bs, seqlen, n_heads) -- update gate in (0, 1)
        cu_seqlens: optional packed-sample boundaries over the flattened
            [1, bs*seqlen] timeline; the state resets at each one.

    Returns:
        output: (bs, seqlen, n_heads, value_head_dim)
    """
    B, L, H, D_k = q.shape
    D_v = v.shape[-1]
    dtype = q.dtype

    # Upcast to float32 -- the recurrence accumulates over seqlen steps.
    q = _l2norm(q.float(), dim=-1) * (D_k**-0.5)
    k = _l2norm(k.float(), dim=-1)
    v, g, beta = v.float(), g.float(), beta.float()

    # Flatten [B, L] into one [1, B*L] timeline and reset the state at each sample
    # start, so packed samples stay independent. cu_seqlens gives the boundaries;
    # without it each row start is one (reproducing the plain batched result).
    q, k, v, g, beta = (t.reshape(1, B * L, *t.shape[2:]) for t in (q, k, v, g, beta))
    starts = (
        set(cu_seqlens[:-1].tolist())
        if cu_seqlens is not None
        else {b * L for b in range(B)}
    )

    output = torch.zeros(1, B * L, H, D_v, dtype=torch.float32, device=q.device)
    state = torch.zeros(1, H, D_k, D_v, dtype=torch.float32, device=q.device)

    for t in range(B * L):
        if t in starts:  # packed-sample boundary -> fresh recurrent state
            state = torch.zeros_like(state)
        g_t = g[:, t].exp().unsqueeze(-1).unsqueeze(-1)
        b_t = beta[:, t].unsqueeze(-1)

        state = state * g_t
        kv_mem = torch.einsum("bhkv,bhk->bhv", state, k[:, t])
        delta = (v[:, t] - kv_mem) * b_t
        state = state + torch.einsum("bhk,bhv->bhkv", k[:, t], delta)
        output[:, t] = torch.einsum("bhkv,bhk->bhv", state, q[:, t])

    return output.reshape(B, L, H, D_v).to(dtype)


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
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        # Keyword-only so the TP sharding's local_map treats it as replicated
        # pass-through metadata (like attention_masks), not a sharded input.
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Expand Q/K heads to match V when n_value_heads > n_key_heads
        if q.shape[2] != v.shape[2]:
            assert v.shape[2] % q.shape[2] == 0
            repeat = v.shape[2] // q.shape[2]
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)

        if self.backend == "torch_native":
            return _torch_native_gated_delta(q, k, v, g, beta, cu_seqlens=cu_seqlens)

        # The FLA varlen path takes ONE [1, total] sequence with cu_seqlens marking
        # the sample boundaries, so the recurrent state resets per packed sample.
        # Flatten [B, L, ...] -> [1, B*L, ...]; cu_seqlens is None when not packed.
        bs, seqlen = q.shape[0], q.shape[1]
        if cu_seqlens is not None:
            q = q.reshape(1, bs * seqlen, *q.shape[2:])
            k = k.reshape(1, bs * seqlen, *k.shape[2:])
            v = v.reshape(1, bs * seqlen, *v.shape[2:])
            g = g.reshape(1, bs * seqlen, *g.shape[2:])
            beta = beta.reshape(1, bs * seqlen, *beta.shape[2:])

        if self.backend == "fla_chunked":
            result = _fla_chunk_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        elif self.backend == "fla_fused_recurrent":
            result = _fla_fused_recurrent_gated_delta_rule(
                q,
                k,
                v,
                g,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise ValueError(
                f"Unknown fla_backend '{self.backend}'. "
                "Valid: 'fla_chunked', 'fla_fused_recurrent', 'torch_native'."
            )

        # FLA kernels return (output, final_state); we only need output
        out = result[0]
        if cu_seqlens is not None:
            out = out.reshape(bs, seqlen, *out.shape[2:])
        return out


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

    def _causal_conv(
        self, x: torch.Tensor, conv: nn.Module, cu_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Depthwise causal conv, matching the GatedDeltaKernel backend.

        The causal conv window must not cross packed-sample boundaries (else the
        first ``conv_kernel_size - 1`` tokens of each sample would see the previous
        one). fla backends use ``fla.causal_conv1d``, which resets the window at
        ``cu_seqlens`` (and runs batched when unpacked). ``torch_native`` uses
        ``F.conv1d`` and, when packed, runs it per segment. The conv is depthwise
        (per-channel), so under TP it runs on channel-local shards via local_map.
        """
        if self.kernel.backend == "torch_native":
            return self._torch_causal_conv(x, conv, cu_seqlens)
        return self._fla_causal_conv(x, conv, cu_seqlens)

    def _fla_causal_conv(
        self, x: torch.Tensor, conv: nn.Module, cu_seqlens: torch.Tensor | None
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        def _conv(x_local: torch.Tensor, w_local: torch.Tensor) -> torch.Tensor:
            # Packed: fla wants one [1, total, C] sequence and cu_seqlens spans the
            # flattened [bs*seqlen] layout (offsets already include batch rows).
            # Unpacked: a plain batched [bs, seqlen, C] conv. fla applies the silu.
            packed = cu_seqlens is not None
            xin = x_local.reshape(1, bs * seqlen, -1) if packed else x_local
            # fla's causal_conv1d is untyped (pyrefly reads it as a Tensor).
            y, _ = _fla_causal_conv1d(
                xin,
                weight=w_local.squeeze(1),  # [C, 1, k] -> [C, k]
                bias=None,  # GDN conv is bias-free
                activation="silu",
                cu_seqlens=cu_seqlens,
            )
            return y.reshape(bs, seqlen, -1) if packed else y

        if isinstance(x, DTensor):
            # Channel-sharded depthwise conv: run per shard, restore DTensor-ness.
            x_plc = x.placements
            w_plc = conv.weight.placements  # pyrefly: ignore [missing-attribute]
            conv_dt = local_map(
                _conv,
                out_placements=(x_plc,),
                in_placements=(x_plc, w_plc),
                in_grad_placements=(x_plc, w_plc),
                device_mesh=x.device_mesh,
            )
            return conv_dt(x, conv.weight)  # pyrefly: ignore
        return _conv(x, conv.weight)  # pyrefly: ignore [bad-argument-type]

    def _torch_causal_conv(
        self, x: torch.Tensor, conv: nn.Module, cu_seqlens: torch.Tensor | None
    ) -> torch.Tensor:
        if cu_seqlens is None:
            return self._dense_causal_conv(x, conv)
        # Packed: run F.conv1d per segment so the window never crosses a boundary.
        bs, seqlen = x.shape[0], x.shape[1]
        xf = x.reshape(1, bs * seqlen, *x.shape[2:])
        bounds = cu_seqlens.tolist()
        segs = [
            self._dense_causal_conv(xf[:, s:e], conv)
            for s, e in zip(bounds[:-1], bounds[1:])
        ]
        return torch.cat(segs, dim=1).reshape(bs, seqlen, *x.shape[2:])

    def _dense_causal_conv(self, x: torch.Tensor, conv: nn.Module) -> torch.Tensor:
        # Plain depthwise causal conv via F.conv1d (left-pad k-1, then silu).
        x = F.pad(x.transpose(1, 2), [self.conv_kernel_size - 1, 0])
        if isinstance(x, DTensor):
            # TODO: Remove once the DTensor Conv1d dispatch fix for sharded
            # groups lands in a released torch. local_map runs the conv on
            # local shards (channel-sharded input + Shard(0) weight) and
            # restores DTensor-ness, with explicit gradient placements.
            x_plc = x.placements
            w = conv.weight
            w_plc = w.placements  # pyrefly: ignore [missing-attribute]

            def _conv(x_local: torch.Tensor, w_local: torch.Tensor) -> torch.Tensor:
                # groups == local out-channels (depthwise, channel-sharded)
                # pyrefly: ignore [no-matching-overload]
                return F.conv1d(
                    x_local,
                    w_local,
                    None,
                    conv.stride,
                    conv.padding,
                    conv.dilation,
                    w_local.size(0),
                )

            conv_dt = local_map(
                _conv,
                out_placements=(x_plc,),
                in_placements=(x_plc, w_plc),
                in_grad_placements=(x_plc, w_plc),
                device_mesh=x.device_mesh,
            )
            x = conv_dt(x, w)  # pyrefly: ignore
        else:
            x = conv(x)
        return F.silu(x).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, cu_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        # Shapes:
        #   xq, xk: (bs, seqlen, n_key_heads * key_head_dim)
        #   xv, xz: (bs, seqlen, n_value_heads * value_head_dim)
        #   xa, xb: (bs, seqlen, n_value_heads)
        xq = self._causal_conv(self.in_proj_q(x), self.conv_q, cu_seqlens)
        xk = self._causal_conv(self.in_proj_k(x), self.conv_k, cu_seqlens)
        xv = self._causal_conv(self.in_proj_v(x), self.conv_v, cu_seqlens)
        xz = self.in_proj_z(x)
        xa = self.in_proj_a(x)
        xb = self.in_proj_b(x)

        xq = xq.view(bs, seqlen, -1, self.key_head_dim)
        xk = xk.view(bs, seqlen, -1, self.key_head_dim)
        xv = xv.view(bs, seqlen, -1, self.value_head_dim)

        # Gating signals, shape (bs, seqlen, n_value_heads):
        #   g:    decay rate per head, always negative
        #   beta: update gate ∈ (0, 1)
        g = -torch.exp(self.A_log.float()) * F.softplus(xa.float() + self.dt_bias)
        beta = torch.sigmoid(xb)

        output = self.kernel(xq, xk, xv, g, beta, cu_seqlens=cu_seqlens)

        xz = xz.view(bs, seqlen, -1, self.value_head_dim)
        output = self.norm(output, xz)

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
        # cu_seqlens holds the packed sample boundaries, None for a single unpacked
        # sequence. Computed once and shared across GatedDeltaNet layers so their
        # recurrent state and causal conv reset per sample; reuses the same
        # document-varlen metadata built for the full-attention masks.
        # NOTE: not context-parallel aware; CP is unsupported for GatedDeltaNet.
        cu_seqlens = None
        if positions is not None:  # None on the mrope-only path (no text positions)
            cu_seqlens = create_varlen_metadata_for_document(positions).cu_seq_q
            if cu_seqlens.numel() <= 2:  # single unpacked sample -> non-varlen path
                cu_seqlens = None
        for layer in self.layers.values():
            x = layer(x, attention_masks, rope_positions, cu_seqlens)

        x = self.norm(x) if self.norm is not None else x
        if self._skip_lm_head:
            return x
        return self.lm_head(x) if self.lm_head is not None else x
