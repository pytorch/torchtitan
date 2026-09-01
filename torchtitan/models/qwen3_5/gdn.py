# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gated DeltaNet modules for Qwen3.5."""

from dataclasses import dataclass
from typing import Literal

import spmd_types as spmd
import torch
import torch.nn.functional as F
from fla.modules.conv.triton.ops import CausalConv1dFunction
from fla.ops.gated_delta_rule import (
    chunk_gated_delta_rule as _fla_chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule as _fla_fused_recurrent_gated_delta_rule,
)
from fla.ops.gated_delta_rule.chunk import ChunkGatedDeltaRuleFunction
from fla.ops.gated_delta_rule.fused_recurrent import FusedRecurrentFunction
from torch import nn

from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import VarlenMetadata
from torchtitan.protocols.module import Module

GatedDeltaBackend = Literal["fla_chunked", "fla_fused_recurrent"]

spmd.register_local_autograd_function(ChunkGatedDeltaRuleFunction)
spmd.register_local_autograd_function(FusedRecurrentFunction)
spmd.register_local_autograd_function(CausalConv1dFunction)


@spmd.local_map(
    in_types=(
        {"dp": spmd.S(0), "tp": spmd.S(1)},
        {"dp": spmd.R, "tp": spmd.S(0)},
        {"dp": spmd.V, "tp": spmd.R},
        {"dp": spmd.V, "tp": spmd.R},
    ),
    out_types={"dp": spmd.S(0), "tp": spmd.S(1)},
)
def _causal_conv1d_varlen(
    x_TD: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None,
) -> torch.Tensor:
    """FLA depthwise causal conv with per-document resets (CUDA-only).

    A pure-torch per-document reference lives in
    ``tests/unit_tests/gpu/test_qwen3_5_deltanet.py``.
    """
    if cu_seqlens_cpu is None:
        raise ValueError(
            "Qwen3.5 FLA varlen conv requires a CPU cu_seqlens tensor. "
            "Build VarlenMetadata with include_host_offsets=True."
        )

    from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d

    out_BTD, _ = _fla_causal_conv1d(
        x=x_TD.unsqueeze(0),
        weight=weight.squeeze(1),
        bias=None,
        activation="silu",
        backend="triton",
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
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
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
) -> torch.Tensor:
    """Run the batch-invariant GDN recurrent forward kernel.

    The vLLM generator must use the recurrent kernel for per-token decode. The
    trainer uses the same kernel with a materialized float32 initial state and
    varlen metadata so its forward is bitwise identical to generation.
    """
    num_sequences = int(cu_seqlens.numel()) - 1
    initial_state = q.new_zeros(
        num_sequences,
        q.shape[2],
        q.shape[3],
        v.shape[3],
        dtype=torch.float32,
    )
    output, _ = _fla_fused_recurrent_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    return output.to(q.dtype)


@_recurrent_gdn_fwd.register_fake
def _recurrent_gdn_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(v, dtype=q.dtype)


@torch.library.custom_op(
    "torchtitan::chunk_gdn_bwd", mutates_args=(), device_types="cuda"
)
def _chunk_gdn_bwd(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute the parallel GDN chunk kernel and return its gradients."""
    with torch.enable_grad():
        inputs = tuple(
            tensor.detach().requires_grad_(True) for tensor in (q, k, v, g, beta)
        )
        output = _fla_chunk_gated_delta_rule(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
        )[0]
        grad_q, grad_k, grad_v, grad_g, grad_beta = torch.autograd.grad(
            output, inputs, grad_output
        )
        return grad_q, grad_k, grad_v, grad_g, grad_beta


@_chunk_gdn_bwd.register_fake
def _chunk_gdn_bwd_fake(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(g),
        torch.empty_like(beta),
    )


def _recurrent_gdn_setup_context(ctx, inputs, output) -> None:
    ctx.save_for_backward(*inputs)


def _recurrent_gdn_backward(ctx, grad_output):
    q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu = ctx.saved_tensors
    grads = _chunk_gdn_bwd(
        grad_output,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        cu_seqlens_cpu,
    )
    return (*grads, None, None)


_recurrent_gdn_fwd.register_autograd(
    _recurrent_gdn_backward, setup_context=_recurrent_gdn_setup_context
)


class GatedDeltaKernel(Module):
    """Stateless dispatch to the configured FLA gated delta kernel.

    Provides a module boundary for the sharding code to wrap forward with
    DTensor-to-local conversion -- same pattern as FlexAttention. Handles Q/K
    head expansion for grouped linear attention internally so that
    repeat_interleave runs on local tensors under TP. A pure-torch reference
    implementation lives in ``tests/unit_tests/gpu/test_qwen3_5_deltanet.py``;
    it is far too slow for training use.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        # "fla_chunked": parallel within chunks for training (default)
        # "fla_fused_recurrent": for inference only in rl, no backward
        backend: GatedDeltaBackend = "fla_chunked"

    def __init__(self, config: Config):
        super().__init__()
        self.backend = config.backend

    def forward(
        self,
        xq_TNK: torch.Tensor,
        xk_TNK: torch.Tensor,
        xv_TNV: torch.Tensor,
        g_TN: torch.Tensor,
        beta_TN: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Expand Q/K heads to match V when n_value_heads > n_key_heads
        if xq_TNK.shape[1] != xv_TNV.shape[1]:
            assert xv_TNV.shape[1] % xq_TNK.shape[1] == 0
            repeat = xv_TNV.shape[1] // xq_TNK.shape[1]
            xq_TNK = xq_TNK.repeat_interleave(repeat, dim=1)
            xk_TNK = xk_TNK.repeat_interleave(repeat, dim=1)

        xq_BTNK = xq_TNK.unsqueeze(0)
        xk_BTNK = xk_TNK.unsqueeze(0)
        xv_BTNV = xv_TNV.unsqueeze(0)
        g_BTN = g_TN.unsqueeze(0)
        beta_BTN = beta_TN.unsqueeze(0)

        if is_in_batch_invariant_mode() and cu_seqlens is not None:
            if cu_seqlens_cpu is None:
                raise ValueError(
                    "Batch-invariant Gated DeltaNet requires CPU cu_seqlens."
                )
            return _recurrent_gdn_fwd(
                xq_BTNK,
                xk_BTNK,
                xv_BTNV,
                g_BTN,
                beta_BTN,
                cu_seqlens,
                cu_seqlens_cpu,
            ).squeeze(0)

        if self.backend == "fla_chunked":
            if cu_seqlens is not None and cu_seqlens_cpu is None:
                raise ValueError(
                    "Qwen3.5 FLA varlen DeltaNet requires a CPU cu_seqlens tensor."
                )
            result = _fla_chunk_gated_delta_rule(
                xq_BTNK,
                xk_BTNK,
                xv_BTNV,
                g_BTN,
                beta_BTN,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
        elif self.backend == "fla_fused_recurrent":
            result = _fla_fused_recurrent_gated_delta_rule(
                xq_BTNK,
                xk_BTNK,
                xv_BTNV,
                g_BTN,
                beta=beta_BTN,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu_seqlens,
            )
        else:
            raise ValueError(
                f"Unknown fla_backend '{self.backend}'. "
                "Valid: 'fla_chunked', 'fla_fused_recurrent'."
            )

        # FLA kernels return (output, final_state); we only need output
        return result[0].squeeze(0)


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
        a_TN: torch.Tensor,
        b_TN: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_N: torch.Tensor,
        dt_bias_N: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        key_head_dim: int,
        value_head_dim: int,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Run separate Q/K/V convolutions and recurrence on local heads."""
        num_tokens = query_TC.shape[0]

        if cu_seqlens_host is not None:
            cu_seqlens_cpu = torch.tensor(
                cu_seqlens_host,
                dtype=cu_seqlens.dtype,
                device="cpu",
            )
        else:
            cu_seqlens_cpu = None

        def causal_conv(
            x_TC: torch.Tensor,
            weight_C1W: torch.Tensor,
        ) -> torch.Tensor:
            if cu_seqlens_host is not None:
                return _causal_conv1d_varlen(
                    x_TC,
                    weight_C1W,
                    cu_seqlens,
                    cu_seqlens_cpu,
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

        xq_TNK = causal_conv(query_TC, conv_q_weight_C1W).reshape(
            num_tokens, -1, key_head_dim
        )
        xk_TNK = causal_conv(key_TC, conv_k_weight_C1W).reshape(
            num_tokens, -1, key_head_dim
        )
        xv_TNV = causal_conv(value_TC, conv_v_weight_C1W).reshape(
            num_tokens, -1, value_head_dim
        )
        g_TN = -torch.exp(A_log_N.float()) * F.softplus(a_TN.float() + dt_bias_N)
        beta_TN = torch.sigmoid(b_TN)
        return self.kernel(
            xq_TNK,
            xk_TNK,
            xv_TNV,
            g_TN,
            beta_TN,
            cu_seqlens=cu_seqlens if cu_seqlens_host is not None else None,
            cu_seqlens_cpu=cu_seqlens_cpu,
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
        cu_seqlens_host = None
        if attention_masks is not None:
            # FLA caches varlen index helpers by tensor identity. A fresh
            # tensor ensures forward and activation-checkpoint recompute both
            # execute the helpers instead of taking different cache paths.
            with spmd.local():
                cu_seqlens = attention_masks.cu_seq_q.clone()
            cu_seqlens_host = attention_masks.cu_seq_q_host
            if cu_seqlens_host is None:
                raise ValueError(
                    "Qwen3.5 GatedDeltaNet varlen requires CPU cu_seqlens "
                    "metadata. Build VarlenMetadata with include_host_offsets=True."
                )
        else:
            cu_seqlens = torch.arange(
                0,
                num_tokens + 1,
                num_tokens,
                dtype=torch.int32,
                device=x_TD.device,
            )
            if is_in_batch_invariant_mode():
                cu_seqlens_host = (0, num_tokens)

        query_TC = self.in_proj_q(x_TD)
        key_TC = self.in_proj_k(x_TD)
        value_TC = self.in_proj_v(x_TD)
        gate_TC = self.in_proj_z(x_TD)
        a_TN = self.in_proj_a(x_TD)
        b_TN = self.in_proj_b(x_TD)

        output_TNV = self.inner_gated_delta_net(
            query_TC,
            key_TC,
            value_TC,
            a_TN,
            b_TN,
            self.conv_q.weight,
            self.conv_k.weight,
            self.conv_v.weight,
            self.A_log,
            self.dt_bias,
            cu_seqlens,
            key_head_dim=self.key_head_dim,
            value_head_dim=self.value_head_dim,
            cu_seqlens_host=cu_seqlens_host,
        )
        gate_TNV = gate_TC.view(num_tokens, -1, self.value_head_dim)
        output_TNV = self.norm(output_TNV, gate_TNV)
        out_TD = output_TNV.reshape(num_tokens, -1)
        return self.out_proj(out_TD)
