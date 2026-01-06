# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import and_masks

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.attention import (
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from torchtitan.models.moe import MoE
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

from .args import KimiLinearModelArgs

try:
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda

    HAS_FLA = True
except ImportError:
    HAS_FLA = False
    FusedRMSNormGated = None
    ShortConvolution = None
    chunk_kda = None
    fused_recurrent_kda = None


def kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
) -> torch.Tensor:
    """
    Manual KDA gate computation using PyTorch.

    This replaces fla's fused_kda_gate which has Triton kernel issues.
    Computes: g = -exp(A_log) * softplus(g + dt_bias)

    Args:
        g: Gate tensor of shape [batch, seq, num_heads, head_dim]
        A_log: Log decay factor, shape [1, 1, num_heads, 1]
        dt_bias: Bias tensor, shape [num_heads * head_dim]

    Returns:
        Gated output of shape [batch, seq, num_heads, head_dim] in float32
        (matching fla's fused_kda_gate default output_dtype)
    """
    num_heads, head_dim = g.shape[-2], g.shape[-1]

    # Add bias: dt_bias is [num_heads * head_dim], reshape to [num_heads, head_dim]
    g = g.float() + dt_bias.view(num_heads, head_dim)

    # Apply gate: -exp(A_log) * softplus(g)
    # A_log is [1, 1, num_heads, 1], broadcasts across batch, seq, head_dim
    # Note: A_log needs to be flattened and reshaped like fla expects
    g = -A_log.view(-1).view(num_heads, 1).float().exp() * F.softplus(g)

    # Return float32 to match fla's fused_kda_gate default behavior
    return g


def precompute_rope_cache(
    dim: int, max_seq_len: int, base: float = 10000.0
) -> torch.Tensor:
    """Precompute RoPE cache for positional embeddings."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)
    idx_theta = torch.outer(t, freqs).float()
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    rope_cache: torch.Tensor,
    position_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to query and key tensors."""
    if position_ids is not None:
        rope_cache = rope_cache[position_ids]
        rope_cache = rope_cache.unsqueeze(2)  # [batch_size, seqlen, 1, head_dim * 2]
    else:
        # reshape for broadcast
        _, seqlen, _, head_dim = q.shape
        rope_cache = rope_cache[0:seqlen]
        rope_cache = rope_cache.view(-1, seqlen, 1, head_dim * 2)

    head_dim = q.shape[-1]
    cos = rope_cache[..., :head_dim].to(dtype=q.dtype, device=q.device)
    sin = rope_cache[..., head_dim : head_dim * 2].to(dtype=q.dtype, device=q.device)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot.type_as(q), k_rot.type_as(k)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat key/value heads for multi-head attention."""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        torch.unsqueeze(x, dim=3)
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale parameter."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        return (output * self.weight).type_as(x)


class MLAAttention(nn.Module):
    """
    Multi-Latent Attention adapted from DeepSeek-V3.

    Uses low-rank projections for KV compression.
    """

    def __init__(self, model_args: KimiLinearModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = model_args.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.v_head_dim = model_args.v_head_dim
        self.q_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.scaling = self.q_head_dim ** (-0.5)
        self.use_nope = model_args.mla_use_nope

        # Query projection (no LoRA for queries in this config)
        self.q_proj = nn.Linear(
            model_args.dim, self.n_heads * self.q_head_dim, bias=False
        )

        # KV projection with LoRA
        self.kv_a_proj_with_mqa = nn.Linear(
            model_args.dim,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        # Output projection
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, model_args.dim, bias=False)

        self.use_flex_attn = model_args.use_flex_attn
        if self.use_flex_attn:
            self.inner_attention = FlexAttentionWrapper()
        else:
            self.inner_attention = ScaledDotProductAttentionWrapper()

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.kv_a_proj_with_mqa.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.kv_b_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        position_ids: torch.Tensor | None = None,
    ):
        bs, seqlen, _ = x.shape

        # Query projection
        q = self.q_proj(x)
        q = q.view(bs, seqlen, self.n_heads, self.q_head_dim)
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )

        # KV projection with LoRA compression
        compressed_kv = self.kv_a_proj_with_mqa(x)
        kv_pass, k_rope = torch.split(
            compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )

        # Decompress KV
        kv_pass = self.kv_b_proj(self.kv_a_layernorm(kv_pass))
        kv_pass = kv_pass.view(
            bs, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = torch.split(
            kv_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )

        # Expand k_rope for all heads
        k_rope = k_rope.view(bs, seqlen, 1, self.qk_rope_head_dim)
        k_rope = k_rope.expand(bs, seqlen, self.n_heads, self.qk_rope_head_dim)

        # Apply RoPE to rope parts only
        q_rope, k_rope = apply_rotary_emb(q_rope, k_rope, rope_cache, position_ids)

        # Combine nope and rope parts
        q = torch.cat([q_nope, q_rope], dim=-1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Transpose for attention
        q = q.transpose(1, 2)  # [bs, n_heads, seqlen, q_head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply attention
        if self.use_flex_attn:
            output = self.inner_attention(
                q, k, v, block_mask=attention_masks["flex_attn"], scale=self.scaling
            )
        else:
            output = self.inner_attention(q, k, v, scale=self.scaling)

        output = output.transpose(1, 2).contiguous().view(bs, seqlen, -1)
        return self.wo(output)


class KDAAttention(nn.Module):
    """
    Key-Delta Attention (KDA) for linear attention.

    Uses the fla library for efficient chunk-wise computation.
    """

    def __init__(self, model_args: KimiLinearModelArgs):
        super().__init__()
        self.hidden_size = model_args.dim
        self.head_dim = model_args.linear_attn_head_dim
        self.num_heads = model_args.linear_attn_num_heads
        self.conv_size = model_args.linear_attn_conv_kernel_size

        projection_size = self.head_dim * self.num_heads

        # Input projections
        self.q_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, projection_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, projection_size, bias=False)

        # Short convolutions for Q, K, V
        self.q_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.k_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation="silu",
        )
        self.v_conv1d = ShortConvolution(
            hidden_size=projection_size,
            kernel_size=self.conv_size,
            activation="silu",
        )

        # Gating parameters
        self.A_log = nn.Parameter(
            torch.log(
                torch.empty(self.num_heads, dtype=torch.float32).uniform_(1, 16)
            ).view(1, 1, -1, 1)
        )

        # Forget gate projections (low-rank)
        self.f_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.f_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)
        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))

        # Beta (write gate) projection
        self.b_proj = nn.Linear(self.hidden_size, self.num_heads, bias=False)

        # Output gate projections (low-rank)
        self.g_a_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.g_b_proj = nn.Linear(self.head_dim, projection_size, bias=False)

        # Output normalization and projection
        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=model_args.norm_eps, activation="sigmoid"
        )
        self.o_proj = nn.Linear(projection_size, self.hidden_size, bias=False)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.q_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.k_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.v_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.f_a_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.f_b_proj.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.b_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.g_a_proj.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.g_b_proj.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.o_proj.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.dt_bias)

        # Initialize conv weights
        # fla's ShortConvolution inherits from nn.Conv1d, so weight is directly on module
        for conv in [self.q_conv1d, self.k_conv1d, self.v_conv1d]:
            if hasattr(conv, "weight"):
                nn.init.kaiming_uniform_(conv.weight, a=math.sqrt(5))

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        position_ids: torch.Tensor | None = None,
    ):
        batch_size, seq_len, _ = x.shape

        # Get cu_seqlens for variable length sequences
        cu_seqlens = None
        use_packed = attention_masks is not None and "cu_seqlens" in attention_masks
        if use_packed:
            cu_seqlens = attention_masks["cu_seqlens"]

        # Project and apply convolutions
        q, _ = self.q_conv1d(x=self.q_proj(x), cache=None, output_final_state=False)
        k, _ = self.k_conv1d(x=self.k_proj(x), cache=None, output_final_state=False)
        v, _ = self.v_conv1d(x=self.v_proj(x), cache=None, output_final_state=False)

        # Compute forget gate
        # g needs to be reshaped to [batch, seq, num_heads, head_dim] for kda_gate
        g = self.f_b_proj(self.f_a_proj(x))
        g = g.view(batch_size, seq_len, self.num_heads, self.head_dim)
        g = kda_gate(g, self.A_log, self.dt_bias)

        # Compute beta (write gate)
        beta = self.b_proj(x).float().sigmoid()

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Apply KDA attention
        # Use fused_recurrent for inference (short sequences), chunk for training
        # When using cu_seqlens, batch must be flattened to size 1
        if use_packed:
            total_len = batch_size * seq_len
            q = q.reshape(1, total_len, self.num_heads, self.head_dim)
            k = k.reshape(1, total_len, self.num_heads, self.head_dim)
            v = v.reshape(1, total_len, self.num_heads, self.head_dim)
            g = g.reshape(1, total_len, self.num_heads, self.head_dim)
            beta = beta.reshape(1, total_len, self.num_heads)

            o, _ = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )

            # Reshape back to original batch/seq dims
            o = o.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        else:
            # Always use chunk_kda to match HF behavior
            o, _ = chunk_kda(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
            )

        # Output gating
        out_gate = self.g_b_proj(self.g_a_proj(x))
        out_gate = out_gate.view(batch_size, seq_len, self.num_heads, self.head_dim)
        o = self.o_norm(o, out_gate)

        # Output projection
        o = o.view(batch_size, seq_len, -1)
        return self.o_proj(o)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)


class TransformerBlock(nn.Module):
    """Transformer block with either MLA or KDA attention."""

    def __init__(self, layer_id: int, model_args: KimiLinearModelArgs):
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim
        self.n_layers = model_args.n_layers
        self.is_linear_attn = model_args.is_kda_layer(layer_id)

        # Choose attention type
        if self.is_linear_attn:
            self.attention = KDAAttention(model_args)
        else:
            self.attention = MLAAttention(model_args)

        # MoE or dense FFN
        self.moe_enabled = model_args.is_moe_layer(layer_id)
        if self.moe_enabled:
            self.moe = MoE(
                model_args.moe_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = FeedForward(
                dim=model_args.dim, hidden_dim=model_args.hidden_dim
            )

        # Layer norms
        self.attention_norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)

        # Weight initialization std
        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        position_ids: torch.Tensor | None = None,
    ):
        # Attention with residual
        x = x + self.attention(
            self.attention_norm(x),
            rope_cache,
            attention_masks,
            position_ids=position_ids,
        )

        # FFN with residual
        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))

        return x

    def init_weights(self, buffer_device: torch.device):
        self.attention_norm.weight.data.fill_(1.0)
        self.ffn_norm.weight.data.fill_(1.0)
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device, self.n_layers)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class KimiLinearModel(nn.Module, ModelProtocol):
    """
    Kimi Linear Model with hybrid MLA and KDA attention.

    Features:
    - Multi-Latent Attention (MLA) for specified layers
    - Key-Delta Attention (KDA) for linear attention layers
    - Optional MoE FFN layers
    """

    def __init__(self, model_args: KimiLinearModelArgs):
        super().__init__()

        have_linear_attention = any(
            lt == "linear_attention" for lt in model_args.layer_types
        )
        if have_linear_attention and not HAS_FLA:
            raise ImportError(
                "The 'fla' package is required for models with 'linear_attention' layers. "
                "Please install it: `pip install fla-core`"
            )

        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        # RoPE cache for MLA attention
        self.register_buffer(
            "rope_cache", self._precompute_rope_cache(), persistent=False
        )

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlock(layer_id, model_args)

        self.norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

    def init_weights(self, buffer_device: torch.device | None = None):
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        nn.init.normal_(self.tok_embeddings.weight, mean=0.0, std=0.02)
        for layer in self.layers.values():
            layer.init_weights(buffer_device)
        self.norm.weight.data.fill_(1.0)
        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3
        nn.init.trunc_normal_(
            self.output.weight,
            mean=0.0,
            std=final_out_std,
            a=-cutoff_factor * final_out_std,
            b=cutoff_factor * final_out_std,
        )

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.qk_rope_head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        ret = {}
        mask_mods = [get_causal_mask_mod()]

        match self.model_args.attn_mask_type:
            case "causal":
                return create_attention_mask(
                    and_masks(*mask_mods),
                    1,
                    None,
                    input_batch.shape[1],
                    input_batch.shape[1],
                )
            case "block_causal":
                B, T = input_batch.shape
                mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))

                device = input_batch.device
                reset = torch.zeros(B, T, dtype=torch.bool, device=device)
                reset[:, 0] = True
                prev = torch.cat(
                    [
                        torch.full(
                            (B, 1),
                            tokenizer.eos_id,
                            dtype=input_batch.dtype,
                            device=device,
                        ),
                        input_batch[:, :-1],
                    ],
                    dim=1,
                )
                reset |= prev == tokenizer.eos_id
                seg = (
                    torch.cumsum(reset.to(torch.int32), dim=1, dtype=torch.int32) - 1
                ).contiguous()
                ret["seq_idx"] = seg

                # Compute cu_seqlens for variable length sequences
                cu_seqlens = [torch.tensor([0], dtype=torch.int32, device=device)]
                for b in range(B):
                    if T == 0:
                        continue
                    _, segment_counts = torch.unique(seg[b], return_counts=True)
                    segment_lengths = segment_counts.to(torch.int32)
                    cumulative_lengths = torch.cumsum(segment_lengths, dim=0)
                    offset_cumulative = cu_seqlens[-1][-1] + cumulative_lengths
                    cu_seqlens.append(offset_cumulative)
                cu_seqlens = torch.cat(cu_seqlens)
                ret["cu_seqlens"] = cu_seqlens

            case _:
                raise ValueError(
                    f"Unknown attention mask type: {self.model_args.attn_mask_type}"
                )

        ret["flex_attn"] = create_attention_mask(
            and_masks(*mask_mods), B, None, input_batch.shape[1], input_batch.shape[1]
        )
        return ret

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        position_ids: torch.Tensor | None = None,
    ):
        h = self.tok_embeddings(tokens)
        for layer in self.layers.values():
            h = layer(
                h,
                self.rope_cache,
                attention_masks=attention_masks,
                position_ids=position_ids,
            )
        h = self.norm(h) if self.norm else h
        output = self.output(h) if self.output else h
        return output
