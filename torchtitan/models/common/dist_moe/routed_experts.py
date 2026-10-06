# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan module boundary for Dist-MoE routed experts.

Shape suffixes use ``T`` for local input tokens, ``K`` for selected experts,
``E`` for local experts, ``F`` for expert intermediate dimension, and ``D``
for model dimension.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
import torch_remat as remat

from torchtitan.models.common.linear import GroupedLinear
from torchtitan.protocols.module import Module

from . import _dist_moe as dist_moe
from .runtime import DistMoeRuntime


class DistMoeRoutedExperts(Module):
    """BF16 routed experts executed by the standalone Dist-MoE backend.

    W13 and W2 remain ordinary TorchTitan modules so parameter, FSDP,
    optimizer, and checkpoint ownership does not change. Their module
    ``forward`` methods are not called: Dist-MoE consumes the weights directly
    and owns dispatch, SwiGLU, expert GEMMs, and combine.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        """Configure BF16-specific Dist-MoE execution.

        Args:
            w13: Grouped gate/up projection with stored shape ``(E, 2, F, D)``.
            w2: Grouped down projection with stored shape ``(E, D, F)``.
            top_k: Number of experts selected for each local input token.
            output_postprocess: Optional module translated into the annex's
                typed or eager post-expert processing policy.
            inplace_wgrad_accum: Whether Dist-MoE writes W13/W2 gradients
                directly into existing standard ``parameter.grad`` buffers.
                This is enabled by default and remains visible to graph tracing.
            bf16_grouped_gemm_preset: Optional expert override for the annex's
                BF16 FPROP/DGRAD grouped-GEMM schedule. ``None`` selects the
                shape-aware production defaults.
            activation: Expert activation implemented by the Dist-MoE kernels.
            swiglu_alpha: Sigmoid multiplier when ``activation`` is clamped
                SwiGLU; otherwise ``None``.
            swiglu_limit: Gate and up-projection bound when ``activation`` is
                clamped SwiGLU; otherwise ``None``.
        """

        w13: GroupedLinear.Config
        w2: GroupedLinear.Config
        top_k: int
        output_postprocess: Module.Config | None = None
        inplace_wgrad_accum: bool = True
        bf16_grouped_gemm_preset: dist_moe.Bf16GroupedGemmPreset | None = None
        activation: Literal["swiglu", "swiglu_clamped"] = "swiglu"
        swiglu_alpha: float | None = None
        swiglu_limit: float | None = None

        def __post_init__(self) -> None:
            """Validate expert dimensions consumed by the annex kernels."""
            if self.w13.group_size != self.w2.group_size:
                raise ValueError("w13 and w2 must contain the same number of experts")
            if self.w13.in_features != self.w2.out_features:
                raise ValueError("w13 input and w2 output dimensions must match")
            if self.w13.num_linears != 2:
                raise ValueError("w13 output must contain gate and up projections")
            if self.w13.out_features != self.w2.in_features:
                raise ValueError("w13 output and w2 input dimensions must match")
            if self.w2.num_linears != 1:
                raise ValueError("w2 must contain one down projection")
            if self.top_k <= 0:
                raise ValueError("top_k must be positive")
            if self.activation == "swiglu":
                if self.swiglu_alpha is not None or self.swiglu_limit is not None:
                    raise ValueError("plain SwiGLU must not define clamp parameters")
            elif self.activation == "swiglu_clamped":
                if self.swiglu_alpha is None or self.swiglu_limit is None:
                    raise ValueError("clamped SwiGLU requires alpha and limit")
            else:
                raise ValueError(f"unsupported activation {self.activation!r}")
            postprocess_config = self.output_postprocess
            owner = None if postprocess_config is None else postprocess_config._owner
            if postprocess_config is not None and not callable(
                getattr(owner, "to_dist_moe_postprocess", None)
            ):
                raise TypeError(
                    f"{type(postprocess_config).__qualname__} cannot execute "
                    "inside Dist-MoE"
                )

    def __init__(self, config: Config):
        Module.__init__(self)
        self.w13 = config.w13.build()
        self.w2 = config.w2.build()
        self.output_postprocess = (
            config.output_postprocess.build()
            if config.output_postprocess is not None
            else None
        )
        self.hidden_dim = config.w13.in_features
        self.intermediate_dim = config.w2.in_features
        self.num_experts = config.w13.group_size
        self.top_k = config.top_k
        self.inplace_wgrad_accum = config.inplace_wgrad_accum
        self.bf16_grouped_gemm_preset = config.bf16_grouped_gemm_preset
        self.block_scaled_config: dist_moe.BlockScaledConfig | None = None
        self.activation = config.activation
        self.swiglu_alpha = config.swiglu_alpha
        self.swiglu_limit = config.swiglu_limit
        self._runtime: DistMoeRuntime | None = None

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Leave communication and activation storage to the shared runtime."""
        del buffer_device

    def _weight_operands(
        self,
    ) -> tuple[
        torch.Tensor | dist_moe.PreparedWeight,
        torch.Tensor | dist_moe.PreparedWeight,
    ]:
        """Return W13 and W2 operands for the annex invocation."""
        w13_E2FD = self.w13.weight
        w2_EDF = self.w2.weight
        w13_EFD = w13_E2FD.flatten(1, 2)
        return w13_EFD, w2_EDF

    def _output_postprocess(
        self,
    ) -> dist_moe.RMSNormPostprocess | Callable[[torch.Tensor], torch.Tensor] | None:
        """Bind the current TorchTitan postprocess parameters to the annex."""
        module = self.output_postprocess
        if module is None:
            return None
        factory = getattr(module, "to_dist_moe_postprocess", None)
        if not callable(factory):
            raise TypeError(
                f"{type(module).__qualname__} cannot execute inside Dist-MoE"
            )
        postprocess = factory()
        if not (
            isinstance(postprocess, dist_moe.RMSNormPostprocess)
            or callable(postprocess)
        ):
            raise TypeError(
                "to_dist_moe_postprocess() must return an RMSNormPostprocess "
                "or callable"
            )
        return postprocess

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run distributed dispatch, expert computation, and combine.

        Args:
            x_TD: Local input tokens with model dimension ``D``.
            topk_scores_TK: Selected routing weights for ``K`` experts.
            topk_expert_ids_TK: Selected global expert IDs.
            num_local_tokens_per_expert_E: Router statistics retained by the
                surrounding MoE module; Dist-MoE derives dispatch metadata from
                the selected IDs.
            padding_mask_T: Optional bool ``(T,)``, true for rows the caller
                padded. Their routes get expert ID ``-1`` and score ``0``, which
                Dist-MoE never dispatches, and their output rows are zero.

        Returns:
            Combined local expert output with shape ``(T, D)``.
        """
        del num_local_tokens_per_expert_E
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Dist-MoE context is not initialized")
        if padding_mask_T is not None:
            padding_mask_TK = padding_mask_T.unsqueeze(-1)
            topk_expert_ids_TK = topk_expert_ids_TK.masked_fill(padding_mask_TK, -1)
            topk_scores_TK = topk_scores_TK.masked_fill(padding_mask_TK, 0.0)
        # Every EP rank must pass the annex the same row count, and local batches
        # can differ across ranks. Pad to the call's count with rows routed to
        # expert -1, then slice the output.
        num_tokens = x_TD.shape[0]
        num_call_tokens = runtime.num_local_input_tokens_for_call(num_tokens)
        num_padded_tokens = num_call_tokens - num_tokens
        if num_padded_tokens < 0:
            raise ValueError(
                f"{num_tokens} local tokens exceed the Dist-MoE call's {num_call_tokens}"
            )
        if num_padded_tokens > 0:
            # Dist-MoE requires score 0 on -1 routes. The zero input rows are
            # never dispatched.
            x_TD = F.pad(x_TD, (0, 0, 0, num_padded_tokens))
            topk_scores_TK = F.pad(topk_scores_TK, (0, 0, 0, num_padded_tokens))
            topk_expert_ids_TK = F.pad(
                topk_expert_ids_TK, (0, 0, 0, num_padded_tokens), value=-1
            )
        w13_operand, w2_operand = self._weight_operands()
        postprocess = self._output_postprocess()
        execution_options = dist_moe.ExecutionOptions(
            inplace_wgrad_accum=self.inplace_wgrad_accum,
            experts_output_postprocess=postprocess,
            # A Python callback otherwise reads unspecified storage in the rows
            # of -1 routes, and their stale values can reach its gradients.
            zero_out_padded_callback_inputs=(
                callable(postprocess)
                and (padding_mask_T is not None or num_padded_tokens > 0)
            ),
        )
        out_TD = remat.region(
            dist_moe.routed_experts,
            self.remat_region_name("dist_moe"),
            recompute=False,
        )(
            x_TD.contiguous(),
            topk_expert_ids_TK.contiguous(),
            topk_scores_TK.contiguous(),
            w13_operand,
            w2_operand,
            runtime.context,
            options=execution_options,
        )
        remat.recompute_needs_tensor(out_TD)
        return out_TD[:num_tokens] if num_padded_tokens > 0 else out_TD
