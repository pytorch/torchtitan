# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

import torch
from fla.ops.kda import chunk_kda

from torchtitan.models.kimi_k3.kda import KimiKDAKernel


@torch.library.custom_op(
    "torchtitan::graph_trainer_kda_fwd",
    mutates_args=(),
    device_types="cuda",
)
def _kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    output, _ = chunk_kda(
        q,
        k,
        v,
        gate,
        beta,
        A_log=A_log,
        dt_bias=dt_bias.reshape(-1),
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        safe_gate=True,
        lower_bound=lower_bound,
    )
    return output


@_kda_fwd.register_fake
def _kda_fwd_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> torch.Tensor:
    return torch.empty_like(v, memory_format=torch.contiguous_format)


@torch.library.custom_op(
    "torchtitan::graph_trainer_kda_bwd",
    mutates_args=(),
    device_types="cuda",
)
def _kda_bwd(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    with torch.enable_grad():
        inputs = tuple(
            tensor.detach().requires_grad_(True)
            for tensor in (q, k, v, gate, beta, A_log)
        )
        dt_bias_flat = dt_bias.detach().reshape(-1).requires_grad_(True)
        output, _ = chunk_kda(
            inputs[0],
            inputs[1],
            inputs[2],
            inputs[3],
            inputs[4],
            A_log=inputs[5],
            dt_bias=dt_bias_flat,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=True,
            lower_bound=lower_bound,
        )
        (
            grad_q,
            grad_k,
            grad_v,
            grad_gate,
            grad_beta,
            grad_A_log,
            grad_dt_bias,
        ) = torch.autograd.grad(output, (*inputs, dt_bias_flat), grad_output)
        return (
            grad_q,
            grad_k,
            grad_v,
            grad_gate,
            grad_beta,
            grad_A_log,
            grad_dt_bias.reshape_as(dt_bias),
        )


@_kda_bwd.register_fake
def _kda_bwd_fake(
    grad_output: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
        torch.empty_like(v, memory_format=torch.contiguous_format),
        torch.empty_like(gate, memory_format=torch.contiguous_format),
        torch.empty_like(beta, memory_format=torch.contiguous_format),
        torch.empty_like(A_log, memory_format=torch.contiguous_format),
        torch.empty_like(dt_bias, memory_format=torch.contiguous_format),
    )


def _kda_setup_context(ctx, inputs, output) -> None:
    *tensors, lower_bound = inputs
    ctx.save_for_backward(*tensors)
    ctx.lower_bound = lower_bound


def _kda_backward(ctx, grad_output):
    return (
        *_kda_bwd(
            grad_output,
            *ctx.saved_tensors,
            ctx.lower_bound,
        ),
        None,
    )


_kda_fwd.register_autograd(_kda_backward, setup_context=_kda_setup_context)


class GraphTrainerKDAKernel(KimiKDAKernel):
    @dataclass(kw_only=True, slots=True)
    class Config(KimiKDAKernel.Config):
        pass

    def forward(
        self,
        q_BLHK: torch.Tensor,
        k_BLHK: torch.Tensor,
        v_BLHV: torch.Tensor,
        gate_BLHK: torch.Tensor,
        beta_BLH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
    ) -> torch.Tensor:
        if self.lower_bound is None:
            raise NotImplementedError(
                "GraphTrainer KDA currently requires a finite lower_bound."
            )
        return _kda_fwd(
            q_BLHK,
            k_BLHK,
            v_BLHV,
            gate_BLHK,
            beta_BLH,
            A_log_H,
            dt_bias_HK,
            self.lower_bound,
        )
