# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_varlen_metadata_for_document,
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
    VarlenAttentionWrapper,
    VarlenMetadata,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.moe.moe import MoE
from torchtitan.models.common.rope import apply_rotary_emb_cos_sin, RoPE
from torchtitan.protocols import Module
from torchtitan.protocols.model import BaseModel

try:
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule,
    )

    _HAS_FLA = True
except ImportError:
    _HAS_FLA = False


# ---------------------------------------------------------------------------
# Utility modules
# ---------------------------------------------------------------------------


class OffsetRMSNorm(nn.Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``, weight init to zeros."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)

    def reset_parameters(self):
        nn.init.zeros_(self.weight)


class RMSNormGated(nn.Module):
    """Gated RMSNorm: ``silu(gate) * weight * norm(x)``, weight init to ones.

    Used inside GatedDeltaNet. Takes ``(hidden_states, gate)`` separately.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()
        # Norm before gate (matching transformers Qwen3_5MoeRMSNormGated)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = (self.weight * hidden_states).to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.float())
        return hidden_states.to(input_dtype)

    def reset_parameters(self):
        nn.init.ones_(self.weight)


# ---------------------------------------------------------------------------
# Gated Delta Rule — pure-torch fallback
# ---------------------------------------------------------------------------


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization matching the FLA library implementation."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _torch_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    """Pure-torch reference implementation of the gated delta rule.

    Matches the transformers ``torch_recurrent_gated_delta_rule`` semantics:
    L2-normalizes Q/K, scales Q by ``1/sqrt(d_k)``, uses ``exp(g)`` as decay,
    and applies the delta-rule error-correction update.

    Uses ``(B, L, H, D)`` layout matching the FLA kernel convention.

    Args:
        q: (B, L, H, D_k)
        k: (B, L, H, D_k)
        v: (B, L, H, D_v)
        g: (B, L, H) — log-space decay (negative values)
        beta: (B, L, H) — update weight

    Returns:
        output: (B, L, H, D_v)
    """
    B, L, H, D_k = q.shape
    D_v = v.shape[-1]
    dtype = q.dtype

    # L2 normalize Q and K (matching use_qk_l2norm_in_kernel=True)
    q = _l2norm(q.float(), dim=-1)
    k = _l2norm(k.float(), dim=-1)

    # Scale query
    scale = D_k**-0.5
    q = q * scale

    v = v.float()
    g, beta = g.float(), beta.float()

    output = torch.zeros(B, L, H, D_v, dtype=torch.float32, device=q.device)
    state = torch.zeros(B, H, D_k, D_v, dtype=torch.float32, device=q.device)

    for t in range(L):
        q_t = q[:, t, :, :]  # (B, H, D_k)
        k_t = k[:, t, :, :]  # (B, H, D_k)
        v_t = v[:, t, :, :]  # (B, H, D_v)
        g_t = (
            g[:, t, :].exp().unsqueeze(-1).unsqueeze(-1)
        )  # (B, H, 1, 1) — exp(log-decay)
        b_t = beta[:, t, :].unsqueeze(-1)  # (B, H, 1)

        # Decay state
        state = state * g_t
        # Delta-rule error correction: retrieve, compute delta, update
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)  # (B, H, D_v)
        delta = (v_t - kv_mem) * b_t  # (B, H, D_v)
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)  # (B, H, D_k, D_v)
        # Query against state
        output[:, t, :, :] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    return output.to(dtype)


def _gated_delta_rule_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    backend: str,
) -> torch.Tensor:
    """Dispatch gated delta rule to the selected backend.

    Args:
        q: (B, L, H, D_k)
        k: (B, L, H, D_k)
        v: (B, L, H, D_v)
        g: (B, L, H) — log-space decay
        beta: (B, L, H) — update weight
        backend: One of ``"fla_chunked"``, ``"fla_fused_recurrent"``,
            ``"torch_naive"``.

    Returns:
        output: (B, L, H, D_v)
    """
    _VALID_BACKENDS = {"fla_chunked", "fla_fused_recurrent", "torch_naive"}
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"Unknown fla_backend '{backend}'. Valid options: "
            "'fla_chunked', 'fla_fused_recurrent', 'torch_naive'."
        )

    if backend == "torch_naive":
        return _torch_chunk_gated_delta_rule(q, k, v, g, beta)

    if not _HAS_FLA:
        raise RuntimeError(
            f"Backend '{backend}' requires the `fla` package, but it is not installed."
        )

    if backend == "fla_chunked":
        result = _fla_chunk_gated_delta_rule(
            q, k, v, g, beta, use_qk_l2norm_in_kernel=True
        )
    elif backend == "fla_fused_recurrent":
        result = _fla_fused_recurrent_gated_delta_rule(
            q, k, v, g, beta=beta, use_qk_l2norm_in_kernel=True
        )

    if isinstance(result, tuple):
        return result[0]
    return result


# ---------------------------------------------------------------------------
# GatedDeltaNet — linear attention module
# ---------------------------------------------------------------------------


class GatedDeltaNet(Module):
    """Gated DeltaNet linear attention.

    Completely different from standard attention: no RoPE, no attention masks,
    different head structure. Uses recurrent state + gated delta rule.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_key_heads: int
        n_value_heads: int
        key_head_dim: int
        value_head_dim: int
        conv_kernel_size: int = 4
        norm_eps: float = 1e-6
        fla_backend: str = "fla_chunked"

    def __init__(self, config: Config, *, dim: int, **kwargs):
        super().__init__()
        self.n_key_heads = config.n_key_heads
        self.n_value_heads = config.n_value_heads
        self.key_head_dim = config.key_head_dim
        self.value_head_dim = config.value_head_dim
        self.conv_kernel_size = config.conv_kernel_size
        self.fla_backend = config.fla_backend

        key_dim = config.n_key_heads * config.key_head_dim
        value_dim = config.n_value_heads * config.value_head_dim
        conv_dim = key_dim * 2 + value_dim

        self.in_proj_qkv = nn.Linear(dim, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(dim, value_dim, bias=False)
        self.in_proj_a = nn.Linear(dim, config.n_value_heads, bias=False)
        self.in_proj_b = nn.Linear(dim, config.n_value_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=False,
            kernel_size=config.conv_kernel_size,
            groups=conv_dim,  # depthwise
            padding=0,  # causal padding applied manually in forward
        )

        self.A_log = nn.Parameter(torch.zeros(config.n_value_heads))
        self.dt_bias = nn.Parameter(torch.ones(config.n_value_heads))

        self.norm = RMSNormGated(config.value_head_dim, eps=config.norm_eps)
        self.out_proj = nn.Linear(value_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape

        # Projections
        qkv = self.in_proj_qkv(x)  # (B, L, conv_dim)
        z = self.in_proj_z(x)  # (B, L, value_dim)
        a = self.in_proj_a(x)  # (B, L, n_value_heads)
        b = self.in_proj_b(x)  # (B, L, n_value_heads)

        # Causal Conv1d + SiLU
        qkv = F.pad(qkv.transpose(1, 2), (self.conv_kernel_size - 1, 0))
        qkv = F.silu(self.conv1d(qkv).transpose(1, 2))  # (B, L, conv_dim)

        # Split into q, k, v
        key_dim = self.n_key_heads * self.key_head_dim
        value_dim = self.n_value_heads * self.value_head_dim
        q, k, v = qkv.split([key_dim, key_dim, value_dim], dim=-1)

        # Reshape to heads — stay in (B, L, H, D) layout for FLA kernel
        q = q.view(B, L, self.n_key_heads, self.key_head_dim)
        k = k.view(B, L, self.n_key_heads, self.key_head_dim)
        v = v.view(B, L, self.n_value_heads, self.value_head_dim)

        # Repeat q, k if n_value_heads > n_key_heads (grouped heads)
        if self.n_value_heads > self.n_key_heads:
            repeat = self.n_value_heads // self.n_key_heads
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)

        # Compute log-decay (g) and update weight (beta) — (B, L, H_v) layout
        # g is in log-space: always negative, exp(g) ∈ (0, 1) is the actual decay
        g = -torch.exp(self.A_log.float()) * F.softplus(
            a.float() + self.dt_bias
        )  # (B, L, H_v)
        beta = torch.sigmoid(b)  # (B, L, H_v)

        # Gated delta rule — all tensors in (B, L, H, D) layout
        output = _gated_delta_rule_dispatch(
            q, k, v, g, beta, self.fla_backend
        )  # (B, L, H_v, D_v)

        # Apply gated norm (output already in (B, L, H_v, D_v))
        z = z.view(B, L, self.n_value_heads, self.value_head_dim)
        output = self.norm(output, z)

        # Project output
        output = output.reshape(B, L, -1)
        return self.out_proj(output)

    def init_weights(self, init_std: float = 0.02, **kwargs) -> None:
        for linear in (
            self.in_proj_qkv,
            self.in_proj_z,
            self.in_proj_a,
            self.in_proj_b,
        ):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=init_std)
        # A_log: log-uniform init for decay values
        with torch.no_grad():
            self.A_log.copy_(
                torch.log(torch.empty_like(self.A_log).uniform_(1e-6, 16.0))
            )
        nn.init.ones_(self.dt_bias)
        self.norm.reset_parameters()


# ---------------------------------------------------------------------------
# Attention — full attention with output gating + partial RoPE
# ---------------------------------------------------------------------------


class Attention(BaseAttention):
    """Full attention with output gating and partial RoPE for Qwen3.5 MoE.

    Key differences from GQAttention:
    - ``wq`` is 2x wider: produces both query and sigmoid gate
    - Partial RoPE: only first ``rotary_dim`` elements get RoPE
    - Output gating: ``attn_output * sigmoid(gate)`` before ``wo``
    - QK norm uses ``OffsetRMSNorm``
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        n_kv_heads: int | None = None
        head_dim: int | None = None
        rotary_dim: int | None = None
        qk_norm: bool = True
        norm_eps: float = 1e-6
        bias: bool = False
        attn_backend: str = "sdpa"
        attn_mask_type: str = "causal"
        rope_backend: str = "cos_sin"

    def __init__(self, config: Config, *, dim: int, **kwargs):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = (
            config.n_heads if config.n_kv_heads is None else config.n_kv_heads
        )
        self.head_dim = (
            config.head_dim if config.head_dim is not None else dim // config.n_heads
        )
        self.rotary_dim = config.rotary_dim
        self.enable_gqa = self.n_heads > self.n_kv_heads

        # QK norm uses OffsetRMSNorm (not nn.RMSNorm)
        self.q_norm: OffsetRMSNorm | None = None
        self.k_norm: OffsetRMSNorm | None = None
        if config.qk_norm:
            self.q_norm = OffsetRMSNorm(self.head_dim, eps=config.norm_eps)
            self.k_norm = OffsetRMSNorm(self.head_dim, eps=config.norm_eps)

        # Scaling factor (explicit when head_dim differs from dim // n_heads)
        self.scaling = self.head_dim**-0.5 if config.head_dim is not None else None

        # wq is 2x wider: produces query + gate
        self.wq = nn.Linear(dim, self.n_heads * self.head_dim * 2, bias=config.bias)
        self.wk = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.wv = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=config.bias)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=config.bias)

        self.attn_backend = config.attn_backend
        self.inner_attention: nn.Module
        match self.attn_backend:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "varlen":
                self.inner_attention = VarlenAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_backend}")

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape

        # Project Q (2x wider for query + gate), K, V
        xq_gate = self.wq(x).view(bs, seqlen, -1, self.head_dim * 2)
        xq, gate = xq_gate.chunk(2, dim=-1)  # each (bs, seqlen, n_heads, head_dim)
        xk = self.wk(x).view(bs, seqlen, -1, self.head_dim)
        xv = self.wv(x).view(bs, seqlen, -1, self.head_dim)

        # QK norm (before RoPE)
        if self.q_norm is not None:
            xq = self.q_norm(xq)
        if self.k_norm is not None:
            xk = self.k_norm(xk)

        # Partial RoPE: only first rotary_dim elements get RoPE
        if self.rotary_dim is not None and self.rotary_dim < self.head_dim:
            xq_rot, xq_pass = xq[..., : self.rotary_dim], xq[..., self.rotary_dim :]
            xk_rot, xk_pass = xk[..., : self.rotary_dim], xk[..., self.rotary_dim :]
            xq_rot, xk_rot = apply_rotary_emb_cos_sin(
                xq_rot, xk_rot, rope_cache, positions
            )
            xq = torch.cat([xq_rot, xq_pass], dim=-1)
            xk = torch.cat([xk_rot, xk_pass], dim=-1)
        else:
            xq, xk = apply_rotary_emb_cos_sin(xq, xk, rope_cache, positions)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scale_kwargs = {"scale": self.scaling} if self.scaling is not None else {}

        match self.attn_backend:
            case "flex":
                if isinstance(attention_masks, dict):
                    block_mask = attention_masks["rope"]
                else:
                    assert isinstance(attention_masks, BlockMask), attention_masks
                    block_mask = attention_masks
                output = (
                    self.inner_attention(
                        xq,
                        xk,
                        xv,
                        block_mask=block_mask,
                        enable_gqa=self.enable_gqa,
                        **scale_kwargs,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )  # (bs, seqlen, n_heads, head_dim)
            case "varlen":
                assert isinstance(attention_masks, VarlenMetadata), attention_masks
                output = self.inner_attention(
                    xq, xk, xv, attention_masks, **scale_kwargs
                )
                # VarlenAttention returns (bs * seqlen, n_heads, head_dim)
                output = output.view(bs, seqlen, -1, self.head_dim)
            case "sdpa":
                assert attention_masks is None
                output = (
                    self.inner_attention(
                        xq,
                        xk,
                        xv,
                        enable_gqa=self.enable_gqa,
                        **scale_kwargs,
                    )
                    .transpose(1, 2)
                    .contiguous()
                )  # (bs, seqlen, n_heads, head_dim)
            case _:
                raise ValueError(f"Unknown attention type: {self.attn_backend}")

        # Output gating: attn_output * sigmoid(gate) before wo
        output = output * torch.sigmoid(gate)
        output = output.view(bs, seqlen, -1)
        return self.wo(output)

    def init_weights(self, init_std: float = 0.02, **kwargs) -> None:
        nn.init.trunc_normal_(self.wq.weight, mean=0.0, std=0.02)
        if self.wq.bias is not None:
            nn.init.trunc_normal_(self.wq.bias, mean=0.0, std=0.02)
        for linear in (self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            if linear.bias is not None:
                nn.init.trunc_normal_(linear.bias, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.wo.bias is not None:
            nn.init.trunc_normal_(self.wo.bias, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()


# ---------------------------------------------------------------------------
# TransformerBlock — hybrid decoder layer
# ---------------------------------------------------------------------------


class TransformerBlock(Module):
    """Transformer block for Qwen3.5 MoE hybrid decoder.

    Each layer uses either full attention (``Attention``) or linear attention
    (``GatedDeltaNet``), determined at build time by ``layer_type``. Both types
    share the same MoE + gated shared expert FFN structure.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm_eps: float = 1e-6
        attention: Attention.Config
        deltanet: GatedDeltaNet.Config
        moe: MoE.Config
        feed_forward: FeedForward.Config  # shared expert

    def __init__(
        self, config: Config, *, dim: int, layer_type: str, layer_id: int, n_layers: int
    ):
        super().__init__()
        self.layer_type = layer_type
        self.layer_id = layer_id
        self.n_layers = n_layers

        # Attention: full or DeltaNet
        if layer_type == "full_attention":
            self.attn = config.attention.build(dim=dim)
        else:
            self.attn = config.deltanet.build(dim=dim)

        # MoE (routed experts only, num_shared_experts=0)
        # NOTE: Weight layout difference vs transformers —
        # transformers fuses gate_proj and up_proj into a single gate_up_proj
        # tensor of shape (num_experts, 2*intermediate_size, hidden_size),
        # while we keep them as separate w1 (gate) and w3 (up) in GroupedExperts.
        # Checkpoint conversion must split gate_up_proj along dim=1 into w1/w3.
        self.moe = config.moe.build(dim=dim)

        # Shared expert: FeedForward + sigmoid gate
        self.shared_ffn = config.feed_forward.build(dim=dim)
        self.shared_gate = nn.Linear(dim, 1, bias=False)

        # Norms (OffsetRMSNorm)
        self.attention_norm = OffsetRMSNorm(dim, eps=config.norm_eps)
        self.ffn_norm = OffsetRMSNorm(dim, eps=config.norm_eps)

    @property
    def moe_enabled(self) -> bool:
        return hasattr(self, "moe") and self.moe is not None

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention block
        h = self.attention_norm(x)
        if self.layer_type == "full_attention":
            h = self.attn(h, rope_cache, attention_masks, positions)
        else:
            h = self.attn(h)  # DeltaNet ignores rope/masks
        x = x + h

        # FFN block: MoE + gated shared expert
        h = self.ffn_norm(x)
        moe_out = self.moe(h)
        shared_out = torch.sigmoid(self.shared_gate(h)) * self.shared_ffn(h)
        x = x + moe_out + shared_out
        return x

    def init_weights(self, **kwargs) -> None:
        buffer_device = kwargs.get("buffer_device")
        weight_init_std = 0.02 / (2 * (self.layer_id + 1)) ** 0.5

        self.attn.init_weights(init_std=weight_init_std)
        self.moe.init_weights(
            init_std=weight_init_std,
            buffer_device=buffer_device or torch.device("cpu"),
        )
        self.shared_ffn.init_weights(init_std=weight_init_std)
        nn.init.trunc_normal_(self.shared_gate.weight, mean=0.0, std=0.02)
        self.attention_norm.reset_parameters()
        self.ffn_norm.reset_parameters()


# ---------------------------------------------------------------------------
# Model — top-level hybrid decoder
# ---------------------------------------------------------------------------


class Model(BaseModel):
    """Qwen3.5 MoE hybrid decoder model.

    Alternates between GatedDeltaNet (linear attention) and full attention
    layers, controlled by ``full_attention_interval``. Every Nth layer uses
    full attention; the rest use GatedDeltaNet.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int
        n_layers: int
        vocab_size: int
        norm_eps: float = 1e-6
        rope: RoPE.Config
        layer: TransformerBlock.Config
        full_attention_interval: int = 4

        def update_from_config(
            self,
            *,
            trainer_config,
            **kwargs,
        ) -> None:
            import dataclasses as _dc
            import logging

            logger = logging.getLogger(__name__)

            training = trainer_config.training
            parallelism = trainer_config.parallelism
            debug = trainer_config.debug
            seq_len = training.seq_len
            if seq_len > self.rope.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.rope.max_seq_len}."
                )
            # Sync rope max_seq_len
            self.rope = _dc.replace(self.rope, max_seq_len=seq_len)

            if self.layer.moe is not None:
                self.layer.moe._debug_force_load_balance = debug.moe_force_load_balance

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            from torchtitan.models.utils import get_moe_model_nparams_and_flops

            assert isinstance(self.layer.attention, Attention.Config)
            assert self.layer.attention.head_dim is not None
            return get_moe_model_nparams_and_flops(
                self,
                model,
                self.layer.attention.n_heads,
                2 * self.layer.attention.head_dim,
                seq_len,
            )

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        self.rope = config.rope.build()
        self.register_buffer("freqs_cis", self.rope.cache, persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.n_layers):
            is_full = (layer_id + 1) % config.full_attention_interval == 0
            layer_type = "full_attention" if is_full else "linear_attention"
            self.layers[str(layer_id)] = config.layer.build(
                dim=config.dim,
                layer_type=layer_type,
                layer_id=layer_id,
                n_layers=config.n_layers,
            )

        self.norm = OffsetRMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    @property
    def attn_config(self):
        """Convenience accessor for the attention config from layer."""
        return self.config.layer.attention

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        match self.attn_config.attn_backend:
            case "varlen":
                if self.attn_config.attn_mask_type != "block_causal":
                    raise ValueError(
                        f"varlen attention is only supported with block_causal "
                        f"attention mask type, got {self.attn_config.attn_mask_type}"
                    )
                assert tokenizer.eos_id is not None
                return create_varlen_metadata_for_document(
                    input_batch, tokenizer.eos_id
                )
            case _:
                raise TypeError(
                    f"get_attention_masks not supported for "
                    f"attn_backend='{self.attn_config.attn_backend}'"
                )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, self.freqs_cis, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output

    def init_weights(self, **kwargs) -> None:
        buffer_device: torch.device | None = kwargs.get("buffer_device")
        buffer_device = buffer_device or self.freqs_cis.device

        if self.rope is not None:
            self.rope.init_weights(buffer_device=buffer_device)
            self.freqs_cis = self.rope.cache
        else:
            # PP case: rope module was pruned, rebuild to get freqs_cis
            rope = self.config.rope.build()
            rope.init_weights(buffer_device=buffer_device)
            self.freqs_cis = rope.cache

        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            # pyrefly: ignore [not-callable]
            layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()
        final_out_std = self.config.dim**-0.5
        cutoff_factor = 3
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )
