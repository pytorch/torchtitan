# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Adapted from HuggingFace NemotronH implementation.
# Copyright 2024 HuggingFace Inc. team.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Nemotron3 (NemotronH) Hybrid Model Implementation.

This module implements the Nemotron3 hybrid architecture which combines:
- Mamba2 layers (M): State space model layers for efficient sequence modeling
- Attention layers (*): Standard multi-head attention with GQA support
- MLP layers (-): Simple feed-forward layers
- MoE layers (E/O or any other char): Mixture of Experts

The layer pattern is defined by the `hybrid_override_pattern` configuration.
"""

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import and_masks, BlockMask

from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.models.common.attention import (
    AttentionMasksType,
    FlexAttentionWrapper,
    ScaledDotProductAttentionWrapper,
    create_attention_mask,
    get_causal_mask_mod,
    get_document_mask_mod,
)
from torchtitan.models.common.moe.moe_nemotron import NemotronMoE
from torchtitan.protocols.model import BaseModel
from torchtitan.tools.logging import logger


@dataclass(kw_only=True, slots=True)
class Nemotron3Config(BaseModel.Config):
    # Core model architecture
    vocab_size: int = 131072
    dim: int = 4096  # hidden_size
    hidden_dim: int = 21504  # intermediate_size
    n_layers: int = 52  # num_hidden_layers

    # Hybrid layer pattern: M=Mamba2, *=Attention, -=MLP, other (E/O)=MoE
    hybrid_override_pattern: str = (
        "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-"
    )

    # Attention configuration
    n_heads: int = 32  # num_attention_heads
    head_dim: int = 128
    n_kv_heads: int = 8  # num_key_value_heads (GQA)
    max_seq_len: int = 4096  # max_position_embeddings

    # Activation and biases
    mlp_hidden_act: str = "relu2"
    attn_bias: bool = False  # attention_bias
    mlp_bias: bool = False

    # Initialization and normalization
    initializer_range: float = 0.02
    norm_eps: float = 1e-5  # layer_norm_epsilon
    residual_in_fp32: bool = False

    # Weight tying
    enable_weight_tying: bool = False  # tie_word_embeddings

    # Mamba2 configuration
    ssm_state_size: int = 128  # mamba_state_size
    mamba_num_heads: int = 128
    mamba_n_groups: int = 8
    mamba_head_dim: int = 64
    mamba_d_conv: int = 4
    mamba_hidden_act: str = "silu"
    mamba_dt_limit: tuple[float, float] = field(
        default_factory=lambda: (0.0, float("inf"))
    )
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False
    mamba_chunk_size: int = 128

    # MoE (Mixture of Experts) configuration
    n_routed_experts: int = 8
    moe_intermediate_size: int = 7688
    moe_shared_expert_intermediate_size: int = 7688
    num_experts_per_tok: int = 2
    routed_scaling_factor: float = 1.0
    n_group: int = 1
    topk_group: int = 1
    norm_topk_prob: bool = True

    # Attention backend for attention blocks in the hybrid stack.
    # Supported: "sdpa", "flex".
    attn_type: str = "sdpa"
    # Attention mask type for attention blocks.
    # Supported: "causal", "block_causal".
    attn_mask_type: str = "causal"

    # Block initialization strategy
    depth_init: bool = True

    def __post_init__(self) -> None:
        assert len(self.hybrid_override_pattern) == self.n_layers, (
            f"hybrid_override_pattern length ({len(self.hybrid_override_pattern)}) "
            f"must match n_layers ({self.n_layers})"
        )
        assert re.match(r"^[*M\-EO]+$", self.hybrid_override_pattern), (
            "hybrid_override_pattern must only contain characters 'M' (Mamba2), "
            "'*' (Attention), '-' (MLP), or 'E'/'O' (MoE)"
        )

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        if self.attn_backend not in {"sdpa", "varlen", "flex"}:
            raise ValueError(
                f"attn_type must be one of ['sdpa', 'varlen', 'flex'], got {self.attn_type}"
            )
        if self.attn_backend == "varlen":
            raise ValueError("Varlen attention is not supported with Nemotron3.")
        if self.attn_mask_type not in {"causal", "block_causal"}:
            raise ValueError(
                f"attn_mask_type must be one of ['causal', 'block_causal'], got {self.attn_mask_type}"
            )

    @property
    def attn_backend(self) -> str:
        return self.attn_type

    @attn_backend.setter
    def attn_backend(self, value: str) -> None:
        self.attn_type = value

    @property
    def layers_block_type(self):
        return [
            "mamba"
            if self.hybrid_override_pattern[i] == "M"
            else "attention"
            if self.hybrid_override_pattern[i] == "*"
            else "mlp"
            if self.hybrid_override_pattern[i] == "-"
            else "moe"
            for i in range(self.n_layers)
        ]

    def update_from_config(self, *, trainer_config: Any, **kwargs) -> None:
        seq_len = trainer_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if (
            trainer_config.parallelism.context_parallel_degree > 1
            and self.attn_backend != "sdpa"
        ):
            raise NotImplementedError(
                "CP support for non-SDPA attention backends is still in progress."
            )

    def get_nparams_and_flops(
        self, model: nn.Module, seq_len: int
    ) -> tuple[int, int]:
        nparams_embedding = 0
        nparams_moe_router = 0
        nparams_shared_experts = 0
        nparams_experts = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "tok_embeddings" in name:
                nparams_embedding += p.numel()
                nparams_dense += p.numel()
            elif ".mixer.shared_experts." in name:
                nparams_shared_experts += p.numel()
            elif ".mixer.router." in name:
                nparams_moe_router += p.numel()
            elif ".mixer.experts." in name:
                nparams_experts += p.numel()
            else:
                nparams_dense += p.numel()

        nparams_sparse = nparams_moe_router + nparams_shared_experts + nparams_experts
        nparams = nparams_dense + nparams_sparse

        if self.n_routed_experts > 0:
            nparams_sparse_active = (
                nparams_moe_router
                + nparams_shared_experts
                + nparams_experts * self.num_experts_per_tok // self.n_routed_experts
            )
        else:
            nparams_sparse_active = nparams_moe_router + nparams_shared_experts
        active_nparams = nparams_dense + nparams_sparse_active

        attention_layers = self.layers_block_type.count("attention")
        head_dims = 2 * self.head_dim
        num_flops_per_token = (
            6 * (active_nparams - nparams_embedding)
            + 6 * attention_layers * self.n_heads * head_dims * seq_len
        )

        logger.info(
            f"Nemotron hybrid parameter count: dense {nparams_dense:,}, "
            f"sparse {nparams_sparse:,}, active {active_nparams:,}, "
            f"attention_layers {attention_layers}"
        )

        if self.enable_weight_tying:
            nparams = nparams - nparams_embedding

        return nparams, num_flops_per_token


# =============================================================================
# Helper Functions
# =============================================================================


def pad_tensor_by_size(input_tensor: torch.Tensor, pad_size: int) -> torch.Tensor:
    """
    Pad tensor with `pad_size` on the seq_len dim (dim=1).
    Assumes tensors of either size 4 or 3.
    """
    if pad_size == 0:
        return input_tensor
    pad_shape = (
        (0, 0, 0, 0, 0, pad_size, 0, 0)
        if len(input_tensor.shape) == 4
        else (0, 0, 0, pad_size, 0, 0)
    )
    return F.pad(input_tensor, pad_shape, mode="constant", value=0)


def reshape_into_chunks(
    input_tensor: torch.Tensor, pad_size: int, chunk_size: int
) -> torch.Tensor:
    """
    Pad input_tensor and split into chunk sequences.
    """
    input_tensor = pad_tensor_by_size(input_tensor, pad_size)

    if len(input_tensor.shape) == 3:
        # [bsz, seq_len, num_heads] -> [bsz, -1, chunk_size, num_heads]
        return input_tensor.reshape(
            input_tensor.shape[0], -1, chunk_size, input_tensor.shape[2]
        )
    else:
        # [bsz, seq_len, num_heads, head_dim] -> [bsz, -1, chunk_size, num_heads, head_dim]
        return input_tensor.reshape(
            input_tensor.shape[0],
            -1,
            chunk_size,
            input_tensor.shape[2],
            input_tensor.shape[3],
        )


def segment_sum(input_tensor: torch.Tensor) -> torch.Tensor:
    """
    Stable segment sum calculation using cumulative sums and masking.

    Uses a large negative value instead of -inf to avoid NaN when combined
    with exp() in downstream operations (0 * inf = NaN).
    """
    chunk_size = input_tensor.size(-1)
    # Expand to [..., chunk_size, chunk_size]
    input_tensor = input_tensor[..., None].expand(*input_tensor.size(), chunk_size)

    # Lower triangular mask (diagonal=-1 to zero out elements above diag)
    mask = torch.tril(
        torch.ones(
            chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
        ),
        diagonal=-1,
    )
    input_tensor = input_tensor.masked_fill(~mask, 0)

    # Cumulative sum
    tensor_segsum = torch.cumsum(input_tensor, dim=-2)

    # Keep only lower triangular (including diagonal)
    # Use a large negative value instead of -inf to avoid NaN in exp() operations
    # -1e9 is small enough that exp(-1e9) ≈ 0 but avoids inf * 0 = NaN issues
    mask = torch.tril(
        torch.ones(
            chunk_size, chunk_size, device=input_tensor.device, dtype=torch.bool
        ),
        diagonal=0,
    )

    tensor_segsum = tensor_segsum.masked_fill(~mask, -torch.inf)
    return tensor_segsum


def apply_mask_to_padding_states(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor | None
) -> torch.Tensor:
    """
    Zero out hidden states for padding tokens.
    See: https://github.com/state-spaces/mamba/issues/66
    """
    if (
        attention_mask is not None
        and attention_mask.shape[1] > 1
        and attention_mask.shape[0] > 1
    ):
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


# =============================================================================
# Activation Functions
# =============================================================================
def relu2(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


ACT2FN: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": F.relu,
    "relu2": relu2,
    "gelu": F.gelu,
    "silu": F.silu,
    "swish": F.silu,
    "tanh": torch.tanh,
}


# =============================================================================
# RoPE (Rotary Position Embedding)
# =============================================================================
def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> torch.Tensor:
    """
    Precompute the frequency tensor for complex exponentials (cis).

    Args:
        dim: Dimension of the frequency tensor (head_dim).
        max_seq_len: Maximum sequence length.
        theta: Scaling factor for frequency computation.

    Returns:
        Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Reshape frequency tensor for broadcasting with target tensor."""
    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    freqs_cis = freqs_cis[0:seqlen]
    assert freqs_cis.shape == (seqlen, x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to query and key tensors."""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads if n_kv_heads < n_heads (GQA).
    Shape: (batch, num_kv_heads, seqlen, head_dim) -> (batch, num_heads, seqlen, head_dim)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


# =============================================================================
# Normalization Layers
# =============================================================================
class RMSNorm(nn.Module):
    """RMS Normalization layer (equivalent to LlamaRMSNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight.to(torch.float32) * hidden_states).to(input_dtype)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


class MambaRMSNormGated(nn.RMSNorm):
    """Gated RMS Normalization for Mamba2 layers."""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__(hidden_size, eps=eps)

    def forward(
        self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None
    ) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = super().forward(hidden_states.to(torch.float32))

        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.to(torch.float32))

        return hidden_states.to(input_dtype)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


# =============================================================================
# Mamba2 Mixer (State Space Model)
# =============================================================================
class Mamba2Mixer(nn.Module):
    """
    Mamba2 SSM Mixer layer.

    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """

    def __init__(self, model_args: Nemotron3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = model_args.dim
        self.num_heads = model_args.mamba_num_heads
        self.head_dim = model_args.mamba_head_dim
        self.ssm_state_size = model_args.ssm_state_size
        self.conv_kernel_size = model_args.mamba_d_conv
        self.intermediate_size = self.num_heads * self.head_dim
        self.n_groups = model_args.mamba_n_groups
        self.chunk_size = model_args.mamba_chunk_size
        self.time_step_limit = model_args.mamba_dt_limit
        # Hard-coded Mamba-2 dt initialization range.
        self.time_step_min = 1e-3
        self.time_step_max = 1e-1
        self.dt_init_floor = 1e-4
        self.A_init_range = (1.0, 16.0)
        self.use_conv_bias = model_args.mamba_conv_bias
        self.use_bias = model_args.mamba_proj_bias

        self.activation = model_args.mamba_hidden_act
        self.act = ACT2FN[model_args.mamba_hidden_act]

        # Conv dimension
        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size

        # 1D convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        # Input projection: projects to gate, hidden_states_B_C, and dt
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=self.use_bias)

        # Time step projection bias (inverse-softplus parameterization).
        self.dt_bias = nn.Parameter(
            self._build_dt_bias(
                num_heads=self.num_heads,
                time_step_min=self.time_step_min,
                time_step_max=self.time_step_max,
                dt_init_floor=self.dt_init_floor,
                device=self.in_proj.weight.device,
            )
        )
        self.dt_bias._no_weight_decay = True

        # S4D real parameter for A (initialized in init_weights).
        self.A_log = nn.Parameter(
            torch.empty(
                self.num_heads,
                dtype=self.in_proj.weight.dtype,
                device=self.in_proj.weight.device,
            )
        )
        self.A_log._no_weight_decay = True

        # Gated normalization
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=model_args.norm_eps)

        # D skip connection (initialized in init_weights).
        self.D = nn.Parameter(
            torch.empty(
                self.num_heads,
                dtype=self.in_proj.weight.dtype,
                device=self.in_proj.weight.device,
            )
        )
        self.D._no_weight_decay = True

        # Output projection
        self.out_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=self.use_bias
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass using naive torch implementation (SSD algorithm).

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Optional attention mask

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        batch_size, seq_len, _ = hidden_states.shape
        dtype = hidden_states.dtype

        # Apply mask to padding states
        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        # 1. Project input
        projected_states = self.in_proj(hidden_states)

        # Split projection into gate, hidden_states_B_C, and dt
        d_mlp = (
            projected_states.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.n_groups * self.ssm_state_size
            - self.num_heads
        ) // 2

        _, _, gate, hidden_states_B_C, dt = projected_states.split(
            [d_mlp, d_mlp, self.intermediate_size, self.conv_dim, self.num_heads],
            dim=-1,
        )

        # 2. Convolution
        hidden_states_B_C = self.act(
            self.conv1d(hidden_states_B_C.transpose(1, 2))[..., :seq_len].transpose(
                1, 2
            )
        )
        hidden_states_B_C = apply_mask_to_padding_states(
            hidden_states_B_C, attention_mask
        )

        # Split into hidden_states, B, C
        groups_time_state_size = self.n_groups * self.ssm_state_size
        hidden_states_inner, B, C = torch.split(
            hidden_states_B_C,
            [self.intermediate_size, groups_time_state_size, groups_time_state_size],
            dim=-1,
        )

        # 3. SSM (State Space Model) computation
        A = -torch.exp(self.A_log.float())

        # Softplus and clamp dt
        dt = F.softplus(dt + self.dt_bias)
        dt = torch.clamp(dt, self.time_step_limit[0], self.time_step_limit[1])

        # Reshape for computation
        hidden_states_inner = hidden_states_inner.reshape(
            batch_size, seq_len, -1, self.head_dim
        ).float()
        B = B.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()
        C = C.reshape(batch_size, seq_len, -1, self.ssm_state_size).float()

        # Repeat B and C for heads
        B = B.repeat_interleave(
            self.num_heads // self.n_groups, dim=2, output_size=self.num_heads
        )
        C = C.repeat_interleave(
            self.num_heads // self.n_groups, dim=2, output_size=self.num_heads
        )

        # Padding for chunking
        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size

        # D residual connection
        D_residual = self.D[..., None] * pad_tensor_by_size(
            hidden_states_inner, pad_size
        )

        # Discretize x and A
        hidden_states_inner = hidden_states_inner * dt[..., None]
        A = A.to(hidden_states_inner.dtype) * dt

        # Reshape into chunks
        hidden_states_inner, A, B, C = [
            reshape_into_chunks(t, pad_size, self.chunk_size)
            for t in (hidden_states_inner, A, B, C)
        ]

        # [bsz, -1, chunk_size, num_heads] -> [bsz, num_heads, -1, chunk_size]
        A = A.permute(0, 3, 1, 2)
        A_cumsum = torch.cumsum(A, dim=-1)

        # 1. Compute intra-chunk (diagonal blocks)
        L = torch.exp(segment_sum(A))

        # G = attention-weights like
        G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]
        G = G_intermediate.sum(dim=-1)

        # M = apply attention mask to weights
        M_intermediate = G[..., None] * L.permute(0, 2, 3, 4, 1)[..., None]
        M = M_intermediate.sum(dim=-1)

        # Y_diag (apply to values)
        Y_diag = (M[..., None] * hidden_states_inner[:, :, None]).sum(dim=3)

        # 2. Compute state for each intra-chunk
        decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        B_decay = B * decay_states.permute(0, -2, -1, 1)[..., None]
        states = (B_decay[..., None, :] * hidden_states_inner[..., None]).sum(dim=2)

        # 3. Inter-chunk SSM recurrence
        previous_states = torch.zeros_like(states[:, :1])
        states = torch.cat([previous_states, states], dim=1)
        decay_chunk = torch.exp(segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        decay_chunk = decay_chunk.transpose(1, 3)
        new_states = (decay_chunk[..., None, None] * states[:, :, None, ...]).sum(dim=1)
        states = new_states[:, :-1]

        # 4. State -> output conversion
        state_decay_out = torch.exp(A_cumsum)
        C_times_states = C[..., None, :] * states[:, :, None, ...]
        state_decay_out_permuted = state_decay_out.permute(0, 2, 3, 1)
        Y_off = C_times_states.sum(-1) * state_decay_out_permuted[..., None]

        # Combine diagonal and off-diagonal
        y = Y_diag + Y_off
        y = y.reshape(batch_size, -1, self.num_heads, self.head_dim)
        y = y + D_residual

        # Cut off padding
        if pad_size > 0:
            y = y[:, :seq_len, :, :]

        y = y.reshape(batch_size, seq_len, -1)

        # Apply gated normalization
        scan_output = self.norm(y, gate)

        # Output projection
        return self.out_proj(scan_output.to(dtype))

    def init_weights(self, init_std: float) -> None:
        """Initialize weights for Mamba2Mixer."""
        with torch.no_grad():
            # S4D real initialization for A.
            assert self.A_init_range[0] > 0 and self.A_init_range[1] >= self.A_init_range[0]
            A = torch.empty(
                self.num_heads, dtype=torch.float32, device=self.A_log.device
            ).uniform_(*self.A_init_range)
            self.A_log.copy_(torch.log(A).to(dtype=self.A_log.dtype))

            # D skip connection.
            self.D.fill_(1.0)

            # =================================================================
            # FIX: NaN Loss Bug - dt_bias initialization produces -inf in bf16
            # =================================================================
            #
            # PROBLEM:
            # When training with bf16, the inverse softplus
            # computation `log(1 - exp(-x))` produces -inf values, which then
            # propagate through the forward pass causing NaN loss.
            #
            # ROOT CAUSE:
            # After clamping, dt_bias has small values like 0.0001. When computing
            # `exp(-0.0001)` in bf16, the result is ~0.9999, but bf16's limited
            # precision can round this to exactly 1.0.
            # Then `1 - exp(-x) = 0`, and `log(0) = -inf`.
            #
            # Example failure path:
            #   dt_bias = 0.0001 (after clamp)
            #   neg_exp = exp(-0.0001) = 0.99990... -> rounds to 1.0 in bf16
            #   log(1 - 1.0) = log(0) = -inf
            #   dt + dt_bias = some_value + (-inf) = -inf
            #   softplus(-inf) = 0, but gradients become NaN
            #
            # SOLUTION:
            # 1. Perform all initialization math in float32
            # 2. Use log1p(-x) which is more numerically stable than log(1-x)
            # 3. Clamp neg_exp to ensure 1-neg_exp never becomes exactly 0
            # 4. Copy the result back to the original dtype
            # =================================================================

            dt_bias_f32 = self.dt_bias.float()

            # Generate uniform values in log space, then transform
            dt_bias_f32.uniform_(math.log(0.001), math.log(0.1))
            dt_bias_f32.exp_()
            dt_bias_f32.clamp_(min=1e-4)

            # Apply inverse softplus: inv_softplus(x) = x + log(1 - exp(-x))
            # Use log1p for numerical stability: log1p(-y) = log(1 - y)
            neg_exp = torch.exp(-dt_bias_f32)
            # Clamp to ensure (1 - neg_exp) > 0, preventing log(0) = -inf
            log_term = torch.log1p(-neg_exp.clamp(max=1.0 - 1e-7))
            dt_bias_f32.add_(log_term)

            # Copy back to original dtype (e.g., bf16)
            self.dt_bias.copy_(dt_bias_f32)

        # Initialize projections
        nn.init.trunc_normal_(self.in_proj.weight, mean=0.0, std=0.02)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)

        nn.init.trunc_normal_(self.out_proj.weight, mean=0.0, std=init_std)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # Initialize conv1d
        nn.init.trunc_normal_(self.conv1d.weight, mean=0.0, std=0.02)
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

        # Initialize norm
        self.norm.reset_parameters()

    @staticmethod
    def _build_dt_bias(
        *,
        num_heads: int,
        time_step_min: float,
        time_step_max: float,
        dt_init_floor: float,
        device: torch.device,
    ) -> torch.Tensor:
        dt = torch.exp(
            torch.rand(num_heads, device=device, dtype=torch.float32)
            * (math.log(time_step_max) - math.log(time_step_min))
            + math.log(time_step_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus.
        return dt + torch.log(-torch.expm1(-dt))


# =============================================================================
# Attention Layer
# =============================================================================
class Attention(nn.Module):
    """Multi-headed attention with GQA support."""

    def __init__(self, model_args: Nemotron3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = model_args.dim
        self.num_heads = model_args.n_heads
        self.head_dim = model_args.head_dim
        self.num_kv_heads = model_args.n_kv_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.attn_backend = model_args.attn_backend
        self.attn_mask_type = model_args.attn_mask_type

        self.wq = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=model_args.attn_bias
        )
        self.wk = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=model_args.attn_bias,
        )
        self.wv = nn.Linear(
            self.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=model_args.attn_bias,
        )
        self.wo = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=model_args.attn_bias
        )
        self.inner_attention: nn.Module
        match self.attn_backend:
            case "flex":
                self.inner_attention = FlexAttentionWrapper()
            case "sdpa":
                self.inner_attention = ScaledDotProductAttentionWrapper()
            case "varlen":
                raise ValueError("Varlen attention is not supported with Nemotron3.")
            case _:
                raise ValueError(f"Unknown attention backend: {self.attn_backend}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of attention.

        Args:
            hidden_states: Input tensor of shape (batch_size, seq_len, hidden_size)
            freqs_cis: Precomputed RoPE frequencies

        Returns:
            Output tensor of shape (batch_size, seq_len, hidden_size)
        """
        bsz, q_len, _ = hidden_states.size()

        # Project to Q, K, V
        query_states = self.wq(hidden_states)
        key_states = self.wk(hidden_states)
        value_states = self.wv(hidden_states)

        # Reshape
        query_states = query_states.view(
            bsz, q_len, -1, self.head_dim
        )  # [batch_size, sequence_length, computed_num_heads, head_dim]
        key_states = key_states.view(
            bsz, q_len, -1, self.head_dim
        )  # [batch_size, sequence_length, computed_num_heads, head_dim]
        value_states = value_states.view(
            bsz, q_len, -1, self.head_dim
        )  # [batch_size, sequence_length, computed_num_heads, head_dim]

        # Apply RoPE
        query_states, key_states = apply_rotary_emb(query_states, key_states, freqs_cis)

        # Transpose for attention: (bsz, num_heads, seq_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # Repeat K/V for GQA once before backend dispatch.
        key_states = repeat_kv(key_states, self.num_kv_groups)
        value_states = repeat_kv(value_states, self.num_kv_groups)

        attn_kwargs = {}
        match self.attn_backend:
            case "sdpa":
                attn_kwargs = {
                    "is_causal": True,
                }
            case "varlen":
                raise ValueError("Varlen attention is not supported with Nemotron3.")
            case "flex":
                if attention_masks is None:
                    if self.attn_mask_type != "causal":
                        raise TypeError(
                            "flex attention with block_causal mask requires "
                            "attention_masks to be provided."
                        )
                    attention_masks = create_attention_mask(
                        and_masks(get_causal_mask_mod()),
                        1,
                        None,
                        q_len,
                        q_len,
                    )
                if not isinstance(attention_masks, BlockMask):
                    raise TypeError(
                        f"flex attention expects BlockMask, got {type(attention_masks)}"
                    )
                attn_kwargs = {"block_mask": attention_masks}

        attn_output = self.inner_attention(
            query_states, key_states, value_states, **attn_kwargs
        )

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        return self.wo(attn_output)

    def init_weights(self, init_std: float) -> None:
        """Initialize attention weights."""
        for linear in (self.wq, self.wk, self.wv):
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.wo.bias is not None:
            nn.init.zeros_(self.wo.bias)


# =============================================================================
# MLP Layer
# =============================================================================


class MLP(nn.Module):
    """Simple MLP with configurable activation."""

    def __init__(self, model_args: Nemotron3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = model_args.dim
        self.intermediate_size = model_args.hidden_dim

        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=model_args.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=model_args.mlp_bias
        )
        self.act_fn = ACT2FN[model_args.mlp_hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(x)))

    def init_weights(self, init_std: float) -> None:
        """Initialize MLP weights."""
        nn.init.trunc_normal_(self.up_proj.weight, mean=0.0, std=0.02)
        if self.up_proj.bias is not None:
            nn.init.zeros_(self.up_proj.bias)
        nn.init.trunc_normal_(self.down_proj.weight, mean=0.0, std=init_std)
        if self.down_proj.bias is not None:
            nn.init.zeros_(self.down_proj.bias)


# =============================================================================
# Nemotron3 Block (Hybrid Layer)
# =============================================================================
class Nemotron3Block(nn.Module):
    """
    Hybrid block that can be Mamba2, Attention, MLP, or MoE.

    Block type is determined by the `layers_block_type` from model args.
    """

    def __init__(self, layer_idx: int, model_args: Nemotron3Config):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = model_args.layers_block_type[layer_idx]
        self.residual_in_fp32 = model_args.residual_in_fp32

        # Pre-normalization
        self.norm = RMSNorm(model_args.dim, eps=model_args.norm_eps)

        # Create the appropriate mixer based on block type
        if self.block_type == "mamba":
            self.mixer = Mamba2Mixer(model_args, layer_idx=layer_idx)
        elif self.block_type == "attention":
            self.mixer = Attention(model_args, layer_idx=layer_idx)
        elif self.block_type == "mlp":
            self.mixer = MLP(model_args, layer_idx=layer_idx)
        elif self.block_type == "moe":
            moe_config = NemotronMoE.Config(
                num_experts=model_args.n_routed_experts,
                num_shared_experts=1,
                score_func="sigmoid",
                route_norm=model_args.norm_topk_prob,
                route_scale=model_args.routed_scaling_factor,
                top_k=model_args.num_experts_per_tok,
                num_expert_groups=model_args.n_group if model_args.n_group > 1 else None,
                num_limited_groups=model_args.topk_group
                if model_args.n_group > 1
                else None,
                use_grouped_mm=False,
                load_balance_coeff=None,
                hidden_dim=model_args.moe_intermediate_size,
                activation=model_args.mlp_hidden_act,
                bias=model_args.mlp_bias,
                shared_hidden_dim=model_args.moe_shared_expert_intermediate_size,
            )
            self.mixer = moe_config.build(dim=model_args.dim)
        else:
            raise ValueError(f"Invalid block type: {self.block_type}")

        # Weight initialization std (depth-scaled)
        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of hybrid block.

        Args:
            hidden_states: Input tensor
            freqs_cis: RoPE frequencies (for attention layers)
            attention_mask: Attention mask (for Mamba layers)

        Returns:
            Output tensor after residual connection
        """
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        if self.block_type == "mamba":
            hidden_states = self.mixer(hidden_states, attention_mask=attention_mask)
        elif self.block_type == "attention":
            hidden_states = self.mixer(
                hidden_states, freqs_cis=freqs_cis, attention_masks=attention_masks
            )
        elif self.block_type in ["mlp", "moe"]:
            hidden_states = self.mixer(hidden_states)

        return residual + hidden_states

    def init_weights(self) -> None:
        """Initialize block weights."""
        self.norm.reset_parameters()
        self.mixer.init_weights(init_std=self.weight_init_std)


# =============================================================================
# Nemotron3 Model
# =============================================================================
class Nemotron3Model(BaseModel):
    """
    Nemotron3 (NemotronH) Hybrid Model.

    A hybrid architecture combining Mamba2, Attention, and MLP/MoE layers.
    The layer pattern is defined by the `hybrid_override_pattern` configuration.
    """

    Config = Nemotron3Config

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.model_args = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        # Token embeddings
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # Precompute RoPE frequencies for attention layers
        self.register_buffer(
            "freqs_cis", self._precompute_freqs_cis(), persistent=False
        )

        # Build layers
        self.layers = nn.ModuleDict()
        for layer_idx in range(config.n_layers):
            self.layers[str(layer_idx)] = Nemotron3Block(layer_idx, config)

        # Final normalization
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Output projection
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying
        if config.enable_weight_tying:
            self.output.weight = self.tok_embeddings.weight

        # Log model configuration
        self._log_config()

    def _log_config(self) -> None:
        """Log model configuration."""
        logger.info("=" * 60)
        logger.info("Nemotron3Model initialized with:")
        logger.info(f"  vocab_size: {self.model_args.vocab_size}")
        logger.info(f"  dim: {self.model_args.dim}")
        logger.info(f"  hidden_dim: {self.model_args.hidden_dim}")
        logger.info(f"  n_layers: {self.model_args.n_layers}")
        logger.info(f"  n_heads: {self.model_args.n_heads}")
        logger.info(f"  n_kv_heads: {self.model_args.n_kv_heads}")
        logger.info(f"  head_dim: {self.model_args.head_dim}")
        logger.info(f"  max_seq_len: {self.model_args.max_seq_len}")
        logger.info(
            f"  hybrid_override_pattern: {self.model_args.hybrid_override_pattern}"
        )

        # Count layer types
        block_types = self.model_args.layers_block_type
        type_counts = {}
        for bt in block_types:
            type_counts[bt] = type_counts.get(bt, 0) + 1
        logger.info(f"  layer_type_counts: {type_counts}")

    def _precompute_freqs_cis(self) -> torch.Tensor:
        """Precompute RoPE frequencies."""
        return precompute_freqs_cis(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
        )

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        """
        Initialize model weights.

        Args:
            buffer_device: Device to place buffers on during initialization.
        """
        buffer_device = buffer_device or self.freqs_cis.device
        with torch.device(buffer_device):
            self.freqs_cis = self._precompute_freqs_cis()

        # Initialize embeddings
        if self.tok_embeddings is not None:
            nn.init.normal_(
                self.tok_embeddings.weight, std=self.model_args.initializer_range
            )

        # Initialize layers
        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights()

        # Initialize final norm
        if self.norm is not None:
            self.norm.reset_parameters()

        # Initialize output projection
        if self.output is not None and not self.model_args.enable_weight_tying:
            final_out_std = self.model_args.dim**-0.5
            cutoff_factor = 3
            nn.init.trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of Nemotron3 model.

        Args:
            tokens: Input token ids of shape (batch_size, seq_len)
            attention_mask: Optional attention mask for Mamba layers

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size)
        """
        # Token embeddings
        h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

        # Apply layers
        for layer in self.layers.values():
            h = layer(
                h,
                freqs_cis=self.freqs_cis,
                attention_mask=attention_mask,
                attention_masks=attention_masks,
            )

        # Final normalization
        h = self.norm(h) if self.norm else h

        # Output projection
        output = self.output(h) if self.output else h

        return output

    def get_attention_masks(
        self,
        input_batch: torch.Tensor,
        tokenizer: BaseTokenizer,
        extra_inputs: dict[str, torch.Tensor] | None = None,
    ) -> AttentionMasksType:
        attn_backend = self.model_args.attn_backend
        attn_mask_type = self.model_args.attn_mask_type

        if attn_backend == "varlen":
            raise ValueError("Varlen attention is not supported with Nemotron3.")

        if attn_backend == "flex":
            mask_mods = [get_causal_mask_mod()]
            match attn_mask_type:
                case "causal":
                    B = 1
                case "block_causal":
                    B = input_batch.shape[0]
                    assert tokenizer.eos_id is not None
                    mask_mods.append(get_document_mask_mod(input_batch, tokenizer.eos_id))
                case _:
                    raise ValueError(f"Unknown attention mask type: {attn_mask_type}")
            seqlen = input_batch.shape[1]
            return create_attention_mask(
                and_masks(*mask_mods), B, None, seqlen, seqlen
            )

        raise TypeError("Only flex backend uses explicit attention masks.")
