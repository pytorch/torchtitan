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

from dataclasses import dataclass

import dist_moe
import torch
import torch_remat as remat

from torchtitan.models.common.linear import GroupedLinear
from torchtitan.protocols.module import Module

from .runtime import DistMoeRuntime


_DistMoeWeightOperand = torch.Tensor | dist_moe.PreparedWeight


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
                fused post-expert processing descriptor.
            inplace_wgrad_accum: Whether Dist-MoE writes W13/W2 gradients
                directly into existing standard ``parameter.grad`` buffers.
                The default functional path returns WGrad to autograd.
            bf16_grouped_gemm_preset: Optional expert override for the annex's
                BF16 FPROP/DGRAD grouped-GEMM schedule. ``None`` selects the
                shape-aware production defaults.
        """

        w13: GroupedLinear.Config
        w2: GroupedLinear.Config
        top_k: int
        output_postprocess: Module.Config | None = None
        inplace_wgrad_accum: bool = False
        bf16_grouped_gemm_preset: dist_moe.Bf16GroupedGemmPreset | None = None

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
        self._runtime: DistMoeRuntime | None = None

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        """Leave communication and activation storage to the shared runtime."""
        del buffer_device

    def _weight_operands(
        self,
    ) -> tuple[_DistMoeWeightOperand, _DistMoeWeightOperand]:
        """Return W13 and W2 operands for the annex invocation."""
        w13_E2FD = self.w13.weight
        w2_EDF = self.w2.weight
        w13_EFD = w13_E2FD.flatten(1, 2)
        return w13_EFD, w2_EDF

    def _output_postprocess(self) -> dist_moe.RMSNormPostprocess | None:
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
        if not isinstance(postprocess, dist_moe.RMSNormPostprocess):
            raise TypeError(
                "to_dist_moe_postprocess() must return dist_moe.RMSNormPostprocess"
            )
        return postprocess

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
    ) -> torch.Tensor:
        """Run distributed dispatch, expert computation, and combine.

        Args:
            x_TD: Local input tokens with model dimension ``D``.
            topk_scores_TK: Selected routing weights for ``K`` experts.
            topk_expert_ids_TK: Selected global expert IDs.
            num_local_tokens_per_expert_E: Router statistics retained by the
                surrounding MoE module; Dist-MoE derives dispatch metadata from
                the selected IDs.

        Returns:
            Combined local expert output with shape ``(T, D)``.
        """
        del num_local_tokens_per_expert_E
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("Dist-MoE context is not initialized")
        w13_operand, w2_operand = self._weight_operands()
        execution_options = dist_moe.ExecutionOptions(
            inplace_wgrad_accum=self.inplace_wgrad_accum,
            experts_output_postprocess=self._output_postprocess(),
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
        return out_TD
