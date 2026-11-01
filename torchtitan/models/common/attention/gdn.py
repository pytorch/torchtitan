# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated DeltaNet linear attention backed by Attention Gym."""

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels,
# H = attention heads, K = query/key head dimension, V = value head dimension,
# S = state slots, W = convolution kernel width, G = key-head groups.

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
import torch_remat as remat
from torch import nn

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode
from torchtitan.distributed.local_compile import local_compile
from torchtitan.models.base.module import Module

from torchtitan.models.common.attn_gym import (
    _ChunkGDN,
    causal_conv1d,
    chunk_gdn,
    ChunkGdnCudnnPacked,
    ContextParallelRouting,
    is_missing_kernel,
    l2norm,
    recurrent_gdn,
    require_attn_gym,
)
from torchtitan.models.common.linear import Linear, maybe_gather_tp_input
from torchtitan.models.common.nn_modules import Conv1d
from torchtitan.models.common.norm import GatedRMSNorm
from .attention import create_varlen_metadata_for_document, InnerAttention
from .linear_attention import (
    chunk_kernel_options,
    LinearAttention,
    LinearAttentionMetadata,
)

# The Attention Gym kernels run on rank-local heads inside local SPMD regions.
# ContextParallelInnerGDN handles cross-rank sequence dependencies
# before invoking them.
for _kernel_function in (_ChunkGDN, ChunkGdnCudnnPacked):
    if not is_missing_kernel(_kernel_function):
        spmd.register_local_autograd_function(_kernel_function)


@spmd.local_map(
    in_types=(
        {"dp": spmd.S(0), "tp": spmd.S(1)},
        {"dp": spmd.R, "tp": spmd.S(0)},
        {"dp": spmd.V, "tp": spmd.R},
    ),
    out_types={"dp": spmd.S(0), "tp": spmd.S(1)},
)
def _causal_conv1d_varlen(
    x_TD: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Depthwise causal conv with per-document resets (CUDA-only).

    A pure-torch per-document reference lives in
    ``tests/unit_tests/gpu/models/qwen3_5/test_qwen3_5_deltanet.py``.
    """
    out_BTD = causal_conv1d(
        x_TD.unsqueeze(0),
        weight.squeeze(1),
        activation="silu",
        cu_seqlens=cu_seqlens,
    )
    assert isinstance(out_BTD, torch.Tensor)
    return out_BTD.squeeze(0)


@torch.library.custom_op(
    "torchtitan::recurrent_gdn_fwd", mutates_args=(), device_types="cuda"
)
def _recurrent_gdn_fwd(
    q_BTHK: torch.Tensor,
    k_BTHK: torch.Tensor,
    v_BTHV: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Run the batch-invariant GDN recurrent forward kernel.

    The vLLM generator uses Attention Gym's paging-aware recurrent kernel for
    per-token decode. The trainer uses the same recurrence with a materialized
    float32 initial state and varlen metadata so its forward is bitwise identical
    to generation.
    """
    num_sequences = int(cu_seqlens.numel()) - 1
    # state_cache_SHVK: [num_sequences + 1, H, V, K].
    state_cache_SHVK = q_BTHK.new_empty(
        num_sequences + 1,
        v_BTHV.shape[2],
        v_BTHV.shape[3],
        q_BTHK.shape[3],
        dtype=torch.float32,
    )
    state_indices = torch.arange(
        1,
        num_sequences + 1,
        dtype=torch.int32,
        device=q_BTHK.device,
    )
    has_initial_state = torch.zeros(
        num_sequences,
        dtype=torch.bool,
        device=q_BTHK.device,
    )
    # The recurrent operator consumes normalized Q/K.
    normalized_q_BTHK = l2norm(q_BTHK, cu_seqlens=cu_seqlens)
    normalized_k_BTHK = l2norm(k_BTHK, cu_seqlens=cu_seqlens)
    out_BTHV, _ = recurrent_gdn(
        normalized_q_BTHK,
        normalized_k_BTHK,
        v_BTHV,
        g,
        beta,
        state_cache_SHVK,
        cu_seqlens=cu_seqlens,
        scale=q_BTHK.shape[-1] ** -0.5,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        autotune=False,
    )
    return out_BTHV.to(q_BTHK.dtype)


@_recurrent_gdn_fwd.register_fake
def _recurrent_gdn_fwd_fake(
    q_BTHK: torch.Tensor,
    k_BTHK: torch.Tensor,
    v_BTHV: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(v_BTHV, dtype=q_BTHK.dtype)


def _chunk_gdn_gradients(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute the parallel GDN chunk kernel and return its gradients."""
    with torch.enable_grad():
        inputs = tuple(
            tensor.detach().requires_grad_(True) for tensor in (q, k, v, g, beta)
        )
        normalized_q = l2norm(inputs[0], cu_seqlens=cu_seqlens)
        normalized_k = l2norm(inputs[1], cu_seqlens=cu_seqlens)
        output, _ = chunk_gdn(
            normalized_q,
            normalized_k,
            inputs[2],
            inputs[3],
            inputs[4],
            cu_seqlens=cu_seqlens,
            scale=inputs[0].shape[-1] ** -0.5,
            impl="fused",
        )
        grad_q, grad_k, grad_v, grad_g, grad_beta = torch.autograd.grad(
            output, inputs, grad_output
        )
        return grad_q, grad_k, grad_v, grad_g, grad_beta


def _recurrent_gdn_setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs)


def _recurrent_gdn_backward(ctx, grad_output):
    q, k, v, g, beta, cu_seqlens = ctx.saved_tensors
    grads = _chunk_gdn_gradients(
        grad_output,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
    )
    return (*grads, None)


_recurrent_gdn_fwd.register_autograd(
    _recurrent_gdn_backward, setup_context=_recurrent_gdn_setup_context
)


@spmd.local_map(
    out_types=(
        # Nominal tp S(1): each rank holds its own [q | k | v]; never redistribute or gather it.
        spmd.PartitionSpec(("dp", "cp"), "tp"),  # mixed_qkv_TC
        spmd.PartitionSpec(("dp", "cp"), "tp"),  # a_TH
        spmd.PartitionSpec(("dp", "cp"), "tp"),  # b_TH
        spmd.PartitionSpec(("dp", "cp"), "tp", None),  # gate_THV
    ),
)
@local_compile("gdn_split_in_proj", batch_invariant=True)
def _split_in_proj(
    projected_TC: torch.Tensor,
    split_sizes: list[int],
    value_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split ``GDN.in_proj``'s output into ``(mixed_qkv_TC, a_TH, b_TH, gate_THV)``.

    ``in_proj`` rows are grouped by key head, so ``q|k|v`` is copied into one
    contiguous block for the convolution. Example, 27B at TP4 (4 local key heads,
    3 value heads per key head, head dims 128):

        projected_TC [T, 4 * 1030]   per key head [q 128 | k 128 | v 384 | z 384 | a 3 | b 3]
        mixed_qkv_TC [T, 2560]       [q 512 | k 512 | v 1536]
        a_TH, b_TH   [T, 12]
        gate_THV     [T, 12, 128]
    """
    num_tokens = projected_TC.shape[0]
    grouped_TGC = projected_TC.unflatten(-1, (-1, sum(split_sizes)))
    query, key, value, gate, a, b = grouped_TGC.split(split_sizes, dim=-1)
    mixed_qkv_TC = torch.cat(
        [query.flatten(1), key.flatten(1), value.flatten(1)], dim=-1
    )
    gate_THV = gate.reshape(num_tokens, -1, value_head_dim)
    # The GDN kernels need contiguous heads. With one value head per key head (0.8B, 2B),
    # flatten and reshape return strided views instead of copies.
    return (
        mixed_qkv_TC,
        a.flatten(1).contiguous(),
        b.flatten(1).contiguous(),
        gate_THV.contiguous(),
    )


class GDNKernel(Module):
    """Run GDN on rank-local tensors.

    This module provides a local SPMD boundary for the sharding code. A
    pure-torch reference implementation lives in
    ``tests/unit_tests/gpu/models/qwen3_5/test_qwen3_5_deltanet.py``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
        # Every attn_gym export this module calls; keep the list in sync.
        require_attn_gym(chunk_gdn, recurrent_gdn, l2norm)
        super().__init__()

    def forward(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        xv_THV: torch.Tensor,
        g_TH: torch.Tensor,
        beta_TH: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        xq_BTHK = xq_THK.unsqueeze(0)
        xk_BTHK = xk_THK.unsqueeze(0)
        xv_BTHV = xv_THV.unsqueeze(0)
        g_BTH = g_TH.unsqueeze(0)
        beta_BTH = beta_TH.unsqueeze(0)

        if is_in_batch_invariant_mode() and cu_seqlens is not None:
            return _recurrent_gdn_fwd(
                xq_BTHK,
                xk_BTHK,
                xv_BTHV,
                g_BTH,
                beta_BTH,
                cu_seqlens,
            ).squeeze(0)

        normalized_q = l2norm(xq_BTHK, cu_seqlens=cu_seqlens)
        normalized_k = l2norm(xk_BTHK, cu_seqlens=cu_seqlens)
        output, _ = chunk_gdn(
            normalized_q,
            normalized_k,
            xv_BTHV,
            g_BTH,
            beta_BTH,
            cu_seqlens=cu_seqlens,
            scale=xq_BTHK.shape[-1] ** -0.5,
            impl="fused",
            kernel_options=chunk_kernel_options(xq_BTHK),
        )
        return output.squeeze(0)


class InnerGDN(InnerAttention):
    """Dense GDN computation behind the vLLM replacement boundary.

    ``mixed_qkv_TC`` holds each rank's ``[q | k | v]`` columns. This dense path
    convolves the three blocks separately; the vLLM replacement convolves them
    in one paged call.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        conv_kernel_size: int
        kernel: GDNKernel.Config

        def build_attention_metadata(
            self,
            positions: torch.Tensor,
            *,
            padding_mask: torch.Tensor | None = None,
            max_num_documents: int | None = None,
            max_context_length: int | None = None,
        ) -> LinearAttentionMetadata:
            """Build packed-sequence metadata consumed by Gated DeltaNet."""
            return LinearAttentionMetadata(
                varlen=create_varlen_metadata_for_document(
                    positions,
                    padding_mask=padding_mask,
                    max_num_documents=max_num_documents,
                    max_context_length=max_context_length,
                ),
                num_conv_history_tokens=self.conv_kernel_size - 1,
            )

    def __init__(self, config: Config):
        # Every attn_gym export this module calls; keep the list in sync.
        require_attn_gym(causal_conv1d)
        super().__init__()
        self.kernel = config.kernel.build()

    def forward(
        self,
        mixed_qkv_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        attention_metadata: LinearAttentionMetadata | None,
    ) -> torch.Tensor:
        query_TC, key_TC, value_TC = mixed_qkv_TC.split(
            [
                conv_q_weight_C1W.shape[0],
                conv_k_weight_C1W.shape[0],
                conv_v_weight_C1W.shape[0],
            ],
            dim=-1,
        )
        varlen = attention_metadata.varlen if attention_metadata is not None else None
        cu_seqlens = varlen.cu_seq_q if varlen is not None else None
        if cu_seqlens is None and is_in_batch_invariant_mode():
            cu_seqlens = torch.tensor(
                [0, query_TC.shape[0]],
                dtype=torch.int32,
                device=query_TC.device,
            )
        return self.run_stages(
            query_TC,
            key_TC,
            value_TC,
            a_TH,
            b_TH,
            conv_q_weight_C1W,
            conv_k_weight_C1W,
            conv_v_weight_C1W,
            A_log_H,
            dt_bias_H,
            cu_seqlens=cu_seqlens,
            key_head_dim=key_head_dim,
            value_head_dim=value_head_dim,
            routing=None,
        )

    def run_stages(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        a_TH: torch.Tensor,
        b_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_H: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        key_head_dim: int,
        value_head_dim: int,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        """Run the stages shared by local and context-parallel GDN."""
        num_tokens = query_TC.shape[0]
        use_varlen_kernels = cu_seqlens is not None and (
            cu_seqlens.numel() > 2 or is_in_batch_invariant_mode()
        )
        convolution_offsets = cu_seqlens if use_varlen_kernels else None
        xq_THK = self.short_convolution(
            query_TC,
            conv_q_weight_C1W,
            cu_seqlens=convolution_offsets,
            routing=routing,
        ).reshape(num_tokens, -1, key_head_dim)
        xk_THK = self.short_convolution(
            key_TC,
            conv_k_weight_C1W,
            cu_seqlens=convolution_offsets,
            routing=routing,
        ).reshape(num_tokens, -1, key_head_dim)
        xv_THV = self.short_convolution(
            value_TC,
            conv_v_weight_C1W,
            cu_seqlens=convolution_offsets,
            routing=routing,
        ).reshape(num_tokens, -1, value_head_dim)
        g_TH = -torch.exp(A_log_H.float()) * F.softplus(a_TH.float() + dt_bias_H)
        beta_TH = torch.sigmoid(b_TH)
        return self.gdn_core(
            xq_THK,
            xk_THK,
            xv_THV,
            g_TH,
            beta_TH,
            cu_seqlens=cu_seqlens if use_varlen_kernels else None,
            routing=routing,
        )

    def short_convolution(
        self,
        x_TC: torch.Tensor,
        weight_C1W: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        del routing
        if initial_state is not None:
            output_1TC = causal_conv1d(
                x_TC.unsqueeze(0),
                weight_C1W[:, 0],
                activation="silu",
                cu_seqlens=cu_seqlens,
                initial_state=initial_state,
            )
            assert isinstance(output_1TC, torch.Tensor)
            return output_1TC.squeeze(0)
        if cu_seqlens is not None:
            return _causal_conv1d_varlen(x_TC, weight_C1W, cu_seqlens)

        x_1CT = F.pad(
            x_TC.transpose(0, 1).unsqueeze(0),
            [weight_C1W.shape[-1] - 1, 0],
        )
        return (
            F.silu(
                F.conv1d(
                    x_1CT,
                    weight_C1W,
                    None,
                    groups=weight_C1W.shape[0],
                )
            )
            .squeeze(0)
            .transpose(0, 1)
        )

    def gdn_core(
        self,
        xq_THK: torch.Tensor,
        xk_THK: torch.Tensor,
        xv_THV: torch.Tensor,
        g_TH: torch.Tensor,
        beta_TH: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
        routing: ContextParallelRouting | None,
    ) -> torch.Tensor:
        del routing
        return self.kernel(
            xq_THK,
            xk_THK,
            xv_THV,
            g_TH,
            beta_TH,
            cu_seqlens=cu_seqlens,
        )


class GDN(LinearAttention):
    """Gated DeltaNet linear attention.

    Uses recurrent state + gated delta rule instead of softmax attention.
    No RoPE, different head structure from standard attention. Conv and
    recurrent state are reset at document boundaries whenever document
    offsets (``LinearAttentionMetadata``) are provided. With no offsets
    (``None``), the packed sequence is processed as a single continuous stream.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(LinearAttention.Config):
        key_head_dim: int
        value_head_dim: int
        # Query/key heads; n_heads counts value heads.
        n_key_heads: int
        conv_kernel_size: int = 4

        # Sub-module configs
        # One projection for q, k, v, z, a and b, with rows grouped by key head
        # (r = n_heads // n_key_heads value heads per key head):
        #   [q key_head_dim | k key_head_dim | v r*value_head_dim | z r*value_head_dim | a r | b r]
        # so a TP shard holds whole heads of all six, like QKVLinear's wqkv per KV group.
        in_proj: Linear.Config
        conv_q: Conv1d.Config
        conv_k: Conv1d.Config
        conv_v: Conv1d.Config
        norm: GatedRMSNorm.Config
        out_proj: Linear.Config

        @property
        def in_proj_split_sizes(self) -> list[int]:
            """Rows of q, k, v, z, a, b per key head; 27B: [128, 128, 384, 384, 3, 3]."""
            heads_per_key = self.n_heads // self.n_key_heads
            value_dim = heads_per_key * self.value_head_dim
            return [
                self.key_head_dim,
                self.key_head_dim,
                value_dim,
                value_dim,
                heads_per_key,
                heads_per_key,
            ]

    def __init__(self, config: Config):
        super().__init__()
        self.key_head_dim = config.key_head_dim
        self.value_head_dim = config.value_head_dim
        if config.n_heads % config.n_key_heads != 0:
            raise ValueError(
                f"n_heads ({config.n_heads}) must be divisible by "
                f"n_key_heads ({config.n_key_heads})"
            )
        self.in_proj_split_sizes = config.in_proj_split_sizes

        self.in_proj = config.in_proj.build()

        self.conv_q = config.conv_q.build()
        self.conv_k = config.conv_k.build()
        self.conv_v = config.conv_v.build()

        self.A_log = nn.Parameter(torch.empty(config.n_heads))
        self.dt_bias = nn.Parameter(torch.empty(config.n_heads))

        self.norm = config.norm.build()
        self.out_proj = config.out_proj.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_metadata: LinearAttentionMetadata | None = None,
    ) -> torch.Tensor:
        x_TD = maybe_gather_tp_input(self, x_TD)

        num_tokens = x_TD.shape[0]
        projected_TC = self.in_proj(x_TD)
        # The split copies the in_proj output with bare ops.
        remat.recompute_needs_tensor(projected_TC)
        mixed_qkv_TC, a_TH, b_TH, gate_THV = _split_in_proj(
            projected_TC, self.in_proj_split_sizes, self.value_head_dim
        )

        output_THV = remat.region(
            self.inner_attention,
            self.remat_region_name("inner_attention"),
            recompute=self.remat_should_recompute("inner_attention"),
        )(
            mixed_qkv_TC,
            a_TH,
            b_TH,
            self.conv_q.weight,
            self.conv_k.weight,
            self.conv_v.weight,
            self.A_log,
            self.dt_bias,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            attention_metadata=attention_metadata,
        )
        # The gated norm reads the inner_attention and split outputs with bare ops.
        remat.recompute_needs_tensor(output_THV, gate_THV)
        output_THV = self.norm(output_THV, gate_THV)
        out_TD = output_THV.reshape(num_tokens, -1)
        return self.out_proj(out_TD)
