# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""LoRA linear modules.

Shape suffixes: ``X`` arbitrary leading dimensions, ``R`` routed rows,
``E`` experts, ``D`` model dimension, ``F`` expert intermediate dimension,
``I`` input features, ``L`` LoRA rank, ``O`` output features.
"""

import functools
import math
from dataclasses import dataclass
from typing import cast

import spmd_types as spmd

import torch
import torch.nn as nn
from spmd_types import SpmdType

from torchtitan.distributed.local_compile import compile
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.models.common.moe.sharding import expert_param_placement_sparse

TP = MeshAxisName.TP

__all__ = [
    "get_lora_dist_moe_routed_experts",
    "get_lora_grouped_linear",
    "get_lora_linear",
]


# TODO: Support checkpoint interoperability for LoRA adapter state. Native DCP
# base checkpoints cannot initialize LoRA models because lora_a/lora_b keys are
# absent. HF export raises in BaseCheckpointManager._validate_hf_export until a
# PEFT adapter export exists.
class _LoRAMixin:
    _lora_scaling: float

    def __init__(self, config) -> None:
        super().__init__(config)  # type: ignore[misc]
        for param in nn.Module.parameters(self):  # type: ignore[arg-type]
            param.requires_grad_(False)
        self._lora_scaling = config.alpha / config.rank


class _LoRALinearMixin(_LoRAMixin):
    """Add a LoRA update to a Linear's local computation."""

    num_linears: int
    lora_a: Linear
    lora_b: Linear

    def __init__(self, config) -> None:
        super().__init__(config)
        lora_a_sharding, lora_b_sharding = self._adapter_sharding(
            config.sharding_config, num_linears=config.num_linears
        )
        self.lora_a = Linear.Config(
            in_features=config.in_features,
            out_features=config.rank,
            bias=False,
            # Inherit the option, so turning it off for tracing covers the adapters.
            inplace_wgrad_accum=config.inplace_wgrad_accum,
            sharding_config=lora_a_sharding,
            param_init={
                "weight": lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5)),
            },
        ).build()
        self.lora_b = Linear.Config(
            in_features=config.rank,
            out_features=config.out_features,
            num_linears=config.num_linears,
            bias=False,
            inplace_wgrad_accum=config.inplace_wgrad_accum,
            sharding_config=lora_b_sharding,
            param_init={"weight": nn.init.zeros_},
        ).build()

    def _linear(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        base_out_XO = super()._linear(input, weight, bias)  # type: ignore[misc]
        # The adapters call their local compute directly, so they run inside this
        # projection's remat region instead of declaring their own: a region
        # nested in a saved region cannot be recomputed.
        lora_out_XO = self.lora_b._linear(
            self.lora_a._linear(input, self.lora_a.weight, None),
            self.lora_b.weight,
            None,
        )
        return base_out_XO + self._lora_scaling * lora_out_XO

    @staticmethod
    def _adapter_sharding(
        base_sharding: ShardingConfig | None,
        *,
        num_linears: int,
    ) -> tuple[ShardingConfig | None, ShardingConfig | None]:
        """Derive adapter weight sharding from the base weight, changing only TP.

        A column-wise base gets a TP-replicated A and a B sharded like the base.
        A row-wise base gets an A sharded like the base and a TP-replicated B.
        A TP-replicated or TP-invariant base is copied to both adapters; other TP
        types raise ValueError. The adapters inherit only parameter sharding; the
        base projection owns the TP collectives.
        """
        base_weight = (
            base_sharding.state_shardings.get("weight") if base_sharding else None
        )
        if base_weight is None:
            return None, None

        # Parameter shardings use S(i), not a PartitionSpec.
        assert base_weight.partition_spec is None
        original_sharding = ShardingConfig(state_shardings={"weight": base_weight})
        local_type = base_weight.local_type

        def with_tp(tp: spmd.PerMeshAxisSpmdType) -> ShardingConfig:
            weight = SpmdType({**local_type, TP: tp})
            return ShardingConfig(state_shardings={"weight": weight})

        # The base weight is [O, I], or [num_linears, O, I] when stacked. A stacked
        # base shares one A [L, I] and stacks B [num_linears, O, L].
        output_dim = 0 if num_linears == 1 else 1
        base_tp = local_type.get(TP)
        if base_tp == spmd.S(output_dim):
            return with_tp(spmd.R), original_sharding
        if base_tp == spmd.S(output_dim + 1):
            return with_tp(spmd.S(1)), with_tp(spmd.R)
        if base_tp not in (None, spmd.R, spmd.I):
            raise ValueError(
                f"LoRA does not support base weight TP type {base_tp} with "
                f"num_linears={num_linears}."
            )
        return original_sharding, original_sharding


def _grouped_lora_adapter_sharding(
    base_sharding: ShardingConfig | None,
) -> tuple[ShardingConfig | None, ShardingConfig | None]:
    """Apply the base projection's expert placement to both adapters."""
    base_weight_sharding = (
        base_sharding.state_shardings.get("weight") if base_sharding else None
    )
    if base_weight_sharding is None:
        return None, None
    if base_weight_sharding not in (
        expert_param_placement_sparse(),
        dense_param_placement(tp=spmd.R),
    ):
        raise ValueError(
            "Grouped LoRA supports only expert-axis or replicated parameter "
            f"sharding, got {base_weight_sharding}."
        )

    return (
        ShardingConfig(state_shardings={"weight": base_weight_sharding}),
        ShardingConfig(state_shardings={"weight": base_weight_sharding}),
    )


def _build_grouped_lora_adapters(
    config: GroupedLinear.Config,
    *,
    rank: int,
) -> tuple[GroupedLinear, GroupedLinear]:
    """Build A ``[E, L, I]`` and B ``[E, O, L]`` or ``[E, num_linears, O, L]``."""
    lora_a_sharding, lora_b_sharding = _grouped_lora_adapter_sharding(
        config.sharding_config
    )
    lora_a = GroupedLinear.Config(
        group_size=config.group_size,
        in_features=config.in_features,
        out_features=rank,
        sharding_config=lora_a_sharding,
        param_init={
            "weight": lambda w: nn.init.kaiming_uniform_(
                w.flatten(0, -2), a=math.sqrt(5)
            ),
        },
    ).build()
    lora_b = GroupedLinear.Config(
        group_size=config.group_size,
        in_features=rank,
        out_features=config.out_features,
        num_linears=config.num_linears,
        sharding_config=lora_b_sharding,
        param_init={"weight": nn.init.zeros_},
    ).build()
    return lora_a, lora_b


class _LoRAGroupedLinearMixin(_LoRAMixin):
    """Add an expert-specific LoRA update to a GroupedLinear."""

    num_linears: int
    lora_a: GroupedLinear
    lora_b: GroupedLinear

    def __init__(self, config) -> None:
        super().__init__(config)
        self.lora_a, self.lora_b = _build_grouped_lora_adapters(
            config,
            rank=config.rank,
        )

    def _grouped_mm(
        self,
        *,
        input_RI: torch.Tensor,
        weight_EOI: torch.Tensor,
        offsets_E: torch.Tensor,
    ) -> torch.Tensor:
        base_out_RO = super()._grouped_mm(  # type: ignore[misc]
            input_RI=input_RI,
            weight_EOI=weight_EOI,
            offsets_E=offsets_E,
        )
        # The adapters call their grouped matmul directly, so they run inside this
        # projection's remat region instead of declaring their own (see
        # _LoRALinearMixin._linear).
        lora_hidden_RL = self.lora_a._grouped_mm(
            input_RI=input_RI,
            weight_EOI=self.lora_a.weight.flatten(1, -2),
            offsets_E=offsets_E,
        )
        lora_out_RO = self.lora_b._grouped_mm(
            input_RI=lora_hidden_RL,
            weight_EOI=self.lora_b.weight.flatten(1, -2),
            offsets_E=offsets_E,
        )
        return base_out_RO + self._lora_scaling * lora_out_RO


def _merge_expert_lora(
    weight_EOI: torch.Tensor,
    lora_b_EOL: torch.Tensor,
    lora_a_ELI: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Return ``W + scale * (B @ A)`` without copying the full weight W.

    Out-of-place ``torch.baddbmm(W, B, A)`` makes two passes over W-sized memory:
    a copy kernel reads W and writes the output buffer, then the GEMM reads that
    buffer as C and writes the result back into it. Here the GEMM writes only
    ``scale * (B @ A)`` (``beta=0`` ignores the empty buffer), and the compiled
    add reads W once and writes the result once. cuBLAS baddbmm also rounds
    ``scale * (B @ A)`` to the weight dtype before adding W, so the rounding
    matches ``torch.baddbmm``.

    TODO: Call ``torch.baddbmm(W, B, A)`` directly once out-of-place baddbmm
    passes self to cuBLASLt as C and writes D without copying self first.
    """
    lora_delta_EOI = torch.baddbmm(
        torch.empty_like(weight_EOI), lora_b_EOL, lora_a_ELI, beta=0, alpha=scale
    )
    return weight_EOI + lora_delta_EOI


# GEMM autotuning can fuse the add into a Triton GEMM epilogue, so W is read
# once and the result written once. It benchmarks kernels in each process, so
# the chosen kernel can differ between runs.
_autotuned_merge_expert_lora = compile(
    _merge_expert_lora,
    fullgraph=True,
    dynamic=False,
    options={"max_autotune_gemm": True},
)
# Deterministic mode skips benchmarking: a cuBLAS GEMM, then an in-place add.
_deterministic_merge_expert_lora = compile(
    _merge_expert_lora,
    fullgraph=True,
    dynamic=False,
    options={"deterministic": True},
)


class _MergeExpertLoRA(torch.autograd.Function):
    """Compile only the merge forward.

    The backward stays eager and uses the ``torch.baddbmm`` derivative formulas,
    so the gradients do not depend on kernel choices.
    """

    @staticmethod
    def forward(  # pyrefly: ignore [bad-override]
        ctx,
        weight_EOI: torch.Tensor,
        lora_b_EOL: torch.Tensor,
        lora_a_ELI: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        ctx.save_for_backward(lora_b_EOL, lora_a_ELI)
        ctx.scale = scale
        # NOTE: Only Dist-MoE routed expert weights reach this merge. They are
        # never spmd.I, so ranks may pick different autotuned kernels.
        # TODO(fegin): Always use the deterministic merge once it is as fast as
        # the autotuned one.
        merge = (
            _deterministic_merge_expert_lora
            if torch.are_deterministic_algorithms_enabled()
            else _autotuned_merge_expert_lora
        )
        return merge(weight_EOI, lora_b_EOL, lora_a_ELI, scale)

    @staticmethod
    def backward(  # pyrefly: ignore [bad-override]
        ctx, grad_EOI: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, None]:
        lora_b_EOL, lora_a_ELI = ctx.saved_tensors
        grad_weight_EOI = grad_EOI if ctx.needs_input_grad[0] else None
        grad_b_EOL = grad_EOI.bmm(lora_a_ELI.transpose(1, 2)) * ctx.scale
        grad_a_ELI = lora_b_EOL.transpose(1, 2).bmm(grad_EOI) * ctx.scale
        return grad_weight_EOI, grad_b_EOL, grad_a_ELI, None


class _LoRADistMoeRoutedExpertsMixin:
    """Materialize child GroupedLinear LoRA updates into weight operands.

    Linear and GroupedLinear LoRA apply A and B to input rows. Their extra
    compute scales with the number of rows and the LoRA rank, and autograd
    retains the input and rank-sized activation needed by the adapter backward.
    They never construct a dense ``B @ A`` update.

    Dist-MoE instead consumes complete expert-weight operands and never calls
    W13 or W2 forward. The child GroupedLinear modules still own their adapters
    and scaling; this mixin only detects adapted children and materializes a
    full-sized ``W + scale * (B @ A)`` tensor for each one. Dist-MoE saves each
    effective weight until its backward and, with in-place WGRAD accumulation
    disabled, returns a fresh full-sized gradient for it. The merge backward
    then contracts that gradient into A and B gradients; only after that can the
    effective weight and its gradient be released.

    This avoids a separate fused Dist-MoE LoRA kernel, but its temporary memory
    and adapter-composition work scale with total expert-weight volume rather
    than routed-row count. It also cannot use Dist-MoE's direct-to-parameter
    WGRAD accumulation because the effective weights are transient. A fused
    implementation could apply the low-rank path to routed activations and
    produce adapter gradients without full-sized effective weights or WGRADs.
    """

    w13: GroupedLinear
    w2: GroupedLinear

    def __init__(self, config) -> None:
        super().__init__(config)  # type: ignore[misc]
        # Freeze parameters owned by the wrapper without freezing adapters
        # owned by its w13 and w2 children.
        for param in nn.Module.parameters(self, recurse=False):  # type: ignore[arg-type]
            param.requires_grad_(False)

    def _weight_operands(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_w13_EFD, base_w2_EDF = cast(
            tuple[torch.Tensor, torch.Tensor],
            super()._weight_operands(),  # type: ignore[misc]
        )
        w13_EFD = base_w13_EFD
        w13 = self.w13
        if isinstance(w13, _LoRAGroupedLinearMixin):
            w13_EFD = _MergeExpertLoRA.apply(
                base_w13_EFD,
                w13.lora_b.weight.flatten(1, -2),
                w13.lora_a.weight,
                w13._lora_scaling,
            )
        w2_EDF = base_w2_EDF
        w2 = self.w2
        if isinstance(w2, _LoRAGroupedLinearMixin):
            w2_EDF = _MergeExpertLoRA.apply(
                base_w2_EDF,
                w2.lora_b.weight,
                w2.lora_a.weight,
                w2._lora_scaling,
            )
        return w13_EFD, w2_EDF


def _create_lora_class(
    parent_cls: type[Module],
    mixin_cls: type[_LoRAMixin],
) -> type[Module]:
    parent_config_cls = parent_cls.Config

    class LoRAProjection(mixin_cls, parent_cls):  # type: ignore[misc, valid-type]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            rank: int
            alpha: float

    LoRAProjection.__name__ = f"LoRA{parent_cls.__name__}"
    LoRAProjection.__qualname__ = f"LoRA{parent_cls.__name__}"
    return LoRAProjection


@functools.cache
def get_lora_linear(parent_cls: type[Module]) -> type[Module]:
    """Get a cached LoRA version of a linear module class."""
    return _create_lora_class(parent_cls, _LoRALinearMixin)


@functools.cache
def get_lora_grouped_linear(parent_cls: type[Module]) -> type[Module]:
    """Get a cached LoRA version of a grouped-linear module class."""
    return _create_lora_class(parent_cls, _LoRAGroupedLinearMixin)


@functools.cache
def get_lora_dist_moe_routed_experts() -> type[DistMoeRoutedExperts]:
    """Get the cached BF16 Dist-MoE wrapper for LoRA grouped projections."""

    class LoRADistMoeRoutedExperts(
        _LoRADistMoeRoutedExpertsMixin, DistMoeRoutedExperts
    ):
        @dataclass(kw_only=True, slots=True)
        class Config(DistMoeRoutedExperts.Config):
            pass

    return LoRADistMoeRoutedExperts
