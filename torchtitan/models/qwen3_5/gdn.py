# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated DeltaNet modules for Qwen3.5."""

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels,
# H = attention heads, K = query/key head dimension, V = value head dimension,
# S = state slots, W = convolution kernel width.

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from attn_gym.linear import causal_conv1d, chunk_gdn, l2norm, recurrent_gdn
from torch import nn

from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.protocols.module import Module


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
    ``tests/unit_tests/gpu/test_qwen3_5_deltanet.py``.
    """
    out_BTD = causal_conv1d(
        x_TD.unsqueeze(0),
        weight.squeeze(1),
        activation="silu",
        cu_seqlens=cu_seqlens,
    )
    assert isinstance(out_BTD, torch.Tensor)
    return out_BTD.squeeze(0)


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


class GatedDeltaKernel(Module):
    """Run GDN on rank-local tensors.

    This module provides the boundary that sharding wraps with DTensor-to-local
    conversion. A pure-torch reference implementation lives in
    ``tests/unit_tests/gpu/test_qwen3_5_deltanet.py``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

    def __init__(self, config: Config):
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
        )
        return output.squeeze(0)


class InnerGatedDeltaNet(Module):
    """Dense GDN computation behind the vLLM replacement boundary.

    The trainer keeps Q, K, and V separate, matching the main-branch GDN flow.
    The vLLM replacement may fuse them internally for its paged convolution
    cache without changing this dense path.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        kernel: GatedDeltaKernel.Config

    def __init__(self, config: Config):
        super().__init__()
        self.kernel = config.kernel.build()

    def forward(
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
        cu_seqlens: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
    ) -> torch.Tensor:
        """Run separate Q/K/V convolutions and recurrence on local heads."""
        num_tokens = query_TC.shape[0]
        use_varlen_kernels = cu_seqlens.numel() > 2 or is_in_batch_invariant_mode()

        def causal_conv(
            x_TC: torch.Tensor,
            weight_C1W: torch.Tensor,
        ) -> torch.Tensor:
            if use_varlen_kernels:
                return _causal_conv1d_varlen(
                    x_TC,
                    weight_C1W,
                    cu_seqlens,
                )

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

        xq_THK = causal_conv(query_TC, conv_q_weight_C1W).reshape(
            num_tokens, -1, key_head_dim
        )
        xk_THK = causal_conv(key_TC, conv_k_weight_C1W).reshape(
            num_tokens, -1, key_head_dim
        )
        xv_THV = causal_conv(value_TC, conv_v_weight_C1W).reshape(
            num_tokens, -1, value_head_dim
        )
        g_TH = -torch.exp(A_log_H.float()) * F.softplus(a_TH.float() + dt_bias_H)
        beta_TH = torch.sigmoid(b_TH)
        return self.kernel(
            xq_THK,
            xk_THK,
            xv_THV,
            g_TH,
            beta_TH,
            cu_seqlens=cu_seqlens if use_varlen_kernels else None,
        )


class GatedDeltaNet(Module):
    """Gated DeltaNet linear attention.

    Uses recurrent state + gated delta rule instead of softmax attention.
    No RoPE, different head structure from standard attention. Conv and
    recurrent state are reset at document boundaries whenever document
    offsets (``VarlenMetadata``) are provided -- the transformer block picks
    them out of the model's attention-mask dict under the ``"deltanet"`` key
    (both attention backends). With no offsets (``None``) the packed sequence
    is processed as a single continuous stream.
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
        inner_gated_delta_net: Module.Config
        norm: RMSNormGated.Config
        out_proj: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.key_head_dim = config.key_head_dim
        self.value_head_dim = config.value_head_dim
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
        self.inner_gated_delta_net = config.inner_gated_delta_net.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: VarlenMetadata | None = None,
    ) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        if attention_masks is not None:
            cu_seqlens = attention_masks.cu_seq_q
        else:
            cu_seqlens = torch.arange(
                0,
                num_tokens + 1,
                num_tokens,
                dtype=torch.int32,
                device=x_TD.device,
            )

        query_TC = self.in_proj_q(x_TD)
        key_TC = self.in_proj_k(x_TD)
        value_TC = self.in_proj_v(x_TD)
        gate_TC = self.in_proj_z(x_TD)
        a_TH = self.in_proj_a(x_TD)
        b_TH = self.in_proj_b(x_TD)

        output_THV = self.inner_gated_delta_net(
            query_TC,
            key_TC,
            value_TC,
            a_TH,
            b_TH,
            self.conv_q.weight,
            self.conv_k.weight,
            self.conv_v.weight,
            self.A_log,
            self.dt_bias,
            cu_seqlens,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
        )
        gate_THV = gate_TC.view(num_tokens, -1, self.value_head_dim)
        output_THV = self.norm(output_THV, gate_THV)
        out_TD = output_THV.reshape(num_tokens, -1)
        return self.out_proj(out_TD)
