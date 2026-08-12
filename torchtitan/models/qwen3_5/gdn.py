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
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor.experimental import local_map

from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common import Conv1d, Linear
from torchtitan.models.common.attention import AttentionMasksType, VarlenMetadata
from torchtitan.protocols.module import Module

GatedDeltaBackend = Literal["fla_chunked", "fla_fused_recurrent"]

spmd.register_local_autograd_function(ChunkGatedDeltaRuleFunction)
spmd.register_local_autograd_function(FusedRecurrentFunction)
spmd.register_local_autograd_function(CausalConv1dFunction)


@spmd.local_map(
    in_types=(
        {"dp": spmd.S(1), "tp": spmd.S(2)},
        {"dp": spmd.R, "tp": spmd.S(0)},
        {"dp": spmd.V, "tp": spmd.R},
        {"dp": spmd.V, "tp": spmd.R},
    ),
    out_types={"dp": spmd.S(1), "tp": spmd.S(2)},
)
def _causal_conv1d_varlen(
    x_BTD: torch.Tensor,
    weight: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None,
) -> torch.Tensor:
    """FLA depthwise causal conv with per-document resets (CUDA-only).

    A pure-torch per-document reference lives in
    ``tests/unit_tests/test_qwen3_5_deltanet.py``.
    """
    if cu_seqlens_cpu is None:
        raise ValueError(
            "Qwen3.5 FLA varlen conv requires a CPU cu_seqlens tensor. "
            "Build VarlenMetadata with include_host_offsets=True."
        )

    from fla.modules.conv.causal_conv1d import causal_conv1d as _fla_causal_conv1d

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
    implementation lives in ``tests/unit_tests/test_qwen3_5_deltanet.py``;
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
        xq_BLNK: torch.Tensor,
        xk_BLNK: torch.Tensor,
        xv_BLNV: torch.Tensor,
        g_BLN: torch.Tensor,
        beta_BLN: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Expand Q/K heads to match V when n_value_heads > n_key_heads
        if xq_BLNK.shape[2] != xv_BLNV.shape[2]:
            assert xv_BLNV.shape[2] % xq_BLNK.shape[2] == 0
            repeat = xv_BLNV.shape[2] // xq_BLNK.shape[2]
            xq_BLNK = xq_BLNK.repeat_interleave(repeat, dim=2)
            xk_BLNK = xk_BLNK.repeat_interleave(repeat, dim=2)

        if cu_seqlens is not None and xq_BLNK.shape[0] != 1:
            raise ValueError(
                f"Gated DeltaNet varlen kernels require flattened inputs with "
                f"batch size 1, got batch size {xq_BLNK.shape[0]}."
            )

        if is_in_batch_invariant_mode() and cu_seqlens is not None:
            if cu_seqlens_cpu is None:
                raise ValueError(
                    "Batch-invariant Gated DeltaNet requires CPU cu_seqlens."
                )
            return _recurrent_gdn_fwd(
                xq_BLNK,
                xk_BLNK,
                xv_BLNV,
                g_BLN,
                beta_BLN,
                cu_seqlens,
                cu_seqlens_cpu,
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
                "Valid: 'fla_chunked', 'fla_fused_recurrent'."
            )

        # FLA kernels return (output, final_state); we only need output
        return result[0]


class InnerGatedDeltaNet(Module):
    """Dense inner GDN implementation shared with the paged vLLM boundary.

    This module owns the depthwise convolution and gated-delta recurrence. Varlen
    execution follows PR #3801 metadata, while batch-invariant execution selects
    the recurrent-forward/chunk-backward kernel used by the generator.
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
        self.kernel = config.kernel.build()

    def forward(
        self,
        mixed_qkv_BLC: torch.Tensor,
        a_BLN: torch.Tensor,
        b_BLN: torch.Tensor,
        conv_weight_CW: torch.Tensor,
        A_log_N: torch.Tensor,
        dt_bias_N: torch.Tensor,
        cu_seqlens: torch.Tensor,
        *,
        cu_seqlens_host: tuple[int, ...] | None = None,
    ) -> torch.Tensor:
        """Run convolution and recurrence on rank-local head shards."""
        batch_size, seq_len, conv_dim = mixed_qkv_BLC.shape
        local_key_dim = conv_dim * self.key_dim // (2 * self.key_dim + self.value_dim)

        if cu_seqlens_host is not None:
            num_tokens = batch_size * seq_len
            cu_seqlens_cpu = torch.tensor(
                cu_seqlens_host,
                dtype=cu_seqlens.dtype,
                device="cpu",
            )
            mixed_qkv_1TC = mixed_qkv_BLC.reshape(1, num_tokens, conv_dim)
            conv_output_1TC = _causal_conv1d_varlen(
                mixed_qkv_1TC,
                conv_weight_CW.unsqueeze(1),
                cu_seqlens,
                cu_seqlens_cpu,
            )
            query_1TNK = conv_output_1TC[..., :local_key_dim].reshape(
                1, num_tokens, -1, self.key_head_dim
            )
            key_1TNK = conv_output_1TC[..., local_key_dim : 2 * local_key_dim].reshape(
                1, num_tokens, -1, self.key_head_dim
            )
            value_1TNV = conv_output_1TC[..., 2 * local_key_dim :].reshape(
                1, num_tokens, -1, self.value_head_dim
            )
            decay_1TN = (
                -torch.exp(A_log_N.float())
                * F.softplus(a_BLN.reshape(num_tokens, -1).float() + dt_bias_N)
            ).unsqueeze(0)
            update_gate_1TN = torch.sigmoid(b_BLN.reshape(num_tokens, -1)).unsqueeze(0)
            output_1TNV = self.kernel(
                query_1TNK,
                key_1TNK,
                value_1TNV,
                decay_1TN,
                update_gate_1TN,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
            )
            return output_1TNV.reshape(batch_size, seq_len, -1, self.value_head_dim)

        # Dense unpacked training path. Each batch row owns an independent state.
        conv_input_BCL = F.pad(
            mixed_qkv_BLC.transpose(1, 2),
            [self.conv_kernel_size - 1, 0],
        )
        conv_output_BLC = F.silu(
            F.conv1d(
                conv_input_BCL,
                conv_weight_CW.unsqueeze(1),
                None,
                groups=conv_dim,
            )
        ).transpose(1, 2)
        query_BLNK = conv_output_BLC[..., :local_key_dim].reshape(
            batch_size, seq_len, -1, self.key_head_dim
        )
        key_BLNK = conv_output_BLC[..., local_key_dim : 2 * local_key_dim].reshape(
            batch_size, seq_len, -1, self.key_head_dim
        )
        value_BLNV = conv_output_BLC[..., 2 * local_key_dim :].reshape(
            batch_size, seq_len, -1, self.value_head_dim
        )
        decay_BLN = -torch.exp(A_log_N.float()) * F.softplus(a_BLN.float() + dt_bias_N)
        update_gate_BLN = torch.sigmoid(b_BLN)
        output_BLNV = self.kernel(
            query_BLNK,
            key_BLNK,
            value_BLNV,
            decay_BLN,
            update_gate_BLN,
        )
        return output_BLNV.reshape(batch_size, seq_len, -1, self.value_head_dim)


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
        inner_gated_delta_net: Module.Config
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
        # The dense training implementation is replaced by a paged-cache
        # implementation inside vLLM. Both share this call signature.
        self.inner_gated_delta_net = config.inner_gated_delta_net.build()

    def _fused_conv_weight_bias(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Fuse the depthwise Q/K/V convolution parameters by channel."""
        weight_q_C1W = self.conv_q.weight
        weight_k_C1W = self.conv_k.weight
        weight_v_C1W = self.conv_v.weight
        if isinstance(weight_q_C1W, DTensor):
            placements = weight_q_C1W.placements
            weight_CW = local_map(
                lambda q_C1W, k_C1W, v_C1W: torch.cat(
                    [q_C1W, k_C1W, v_C1W], dim=0
                ).squeeze(1),
                out_placements=(placements,),
                in_placements=(
                    placements,
                    weight_k_C1W.placements,
                    weight_v_C1W.placements,
                ),
                in_grad_placements=(
                    placements,
                    weight_k_C1W.placements,
                    weight_v_C1W.placements,
                ),
                device_mesh=weight_q_C1W.device_mesh,
            )(
                weight_q_C1W,
                weight_k_C1W,  # pyrefly: ignore[bad-argument-count]
                weight_v_C1W,
            )
        else:
            weight_CW = torch.cat(
                [weight_q_C1W, weight_k_C1W, weight_v_C1W], dim=0
            ).squeeze(1)

        biases = [self.conv_q.bias, self.conv_k.bias, self.conv_v.bias]
        bias_C = (
            torch.cat(biases, dim=0)
            if all(bias is not None for bias in biases)
            else None
        )
        return weight_CW, bias_C

    def forward(
        self,
        x_BLD: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x_BLD.shape
        cu_seqlens_host = None
        if isinstance(attention_masks, VarlenMetadata):
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
                (batch_size + 1) * seq_len,
                seq_len,
                dtype=torch.int32,
                device=x_BLD.device,
            )
            if is_in_batch_invariant_mode():
                cu_seqlens_host = tuple(range(0, (batch_size + 1) * seq_len, seq_len))

        query_BLC = self.in_proj_q(x_BLD)
        key_BLC = self.in_proj_k(x_BLD)
        value_BLC = self.in_proj_v(x_BLD)
        if isinstance(query_BLC, DTensor):
            placements = query_BLC.placements
            mixed_qkv_BLC = local_map(
                lambda query_LC, key_LC, value_LC: torch.cat(
                    [query_LC, key_LC, value_LC], dim=-1
                ),
                out_placements=(placements,),
                in_placements=(
                    placements,
                    key_BLC.placements,
                    value_BLC.placements,
                ),
                in_grad_placements=(
                    placements,
                    key_BLC.placements,
                    value_BLC.placements,
                ),
                device_mesh=query_BLC.device_mesh,
            )(
                query_BLC,
                key_BLC,  # pyrefly: ignore[bad-argument-count]
                value_BLC,
            )
        else:
            mixed_qkv_BLC = torch.cat([query_BLC, key_BLC, value_BLC], dim=-1)

        gate_BLC = self.in_proj_z(x_BLD)
        a_BLN = self.in_proj_a(x_BLD)
        b_BLN = self.in_proj_b(x_BLD)
        conv_weight_CW, _ = self._fused_conv_weight_bias()
        if isinstance(mixed_qkv_BLC, DTensor):
            cu_seqlens = DTensor.from_local(
                cu_seqlens,
                mixed_qkv_BLC.device_mesh,
                [Replicate()] * mixed_qkv_BLC.device_mesh.ndim,
            )

        output_BLNV = self.inner_gated_delta_net(
            mixed_qkv_BLC,
            a_BLN,
            b_BLN,
            conv_weight_CW,
            self.A_log,
            self.dt_bias,
            cu_seqlens,
            cu_seqlens_host=cu_seqlens_host,
        )
        gate_BLNV = gate_BLC.reshape(batch_size, seq_len, -1, self.value_head_dim)
        output_BLNV = self.norm(output_BLNV, gate_BLNV)
        out_BLD = output_BLNV.reshape(batch_size, seq_len, -1)
        return self.out_proj(out_BLD)
