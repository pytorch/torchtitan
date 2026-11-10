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
from typing import cast, Literal

import spmd_types as spmd

import torch
import torch.nn as nn
from spmd_types import SpmdType

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.models.common.moe.sharding import (
    expert_param_placement_sparse,
    replicated_param_placement_sparse,
)

TP = MeshAxisName.TP

__all__ = [
    "get_lora_dist_moe_routed_experts",
    "get_lora_grouped_linear",
    "get_lora_linear",
]


# TODO: Support checkpoint interoperability for LoRA adapter state. Native DCP
# base checkpoints cannot initialize LoRA models because lora_a/lora_b keys are
# absent, and Hugging Face state-dict adapters omit those keys during export.
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


def _shared_lora_adapter_sharding(
    base_sharding: ShardingConfig | None,
) -> ShardingConfig | None:
    """Placement of an adapter factor shared across the experts of a projection.

    The factor is replicated where the base weight is split by expert, so the
    sparse-family replicated placement keeps its EP replicas in step.
    """
    base_weight_sharding = (
        base_sharding.state_shardings.get("weight") if base_sharding else None
    )
    if base_weight_sharding is None:
        return None
    if base_weight_sharding == expert_param_placement_sparse():
        return ShardingConfig(
            state_shardings={"weight": replicated_param_placement_sparse()}
        )
    return ShardingConfig(state_shardings={"weight": base_weight_sharding})


def _build_grouped_lora_adapters(
    config: GroupedLinear.Config,
    *,
    rank: int,
    shared_factor: Literal["lora_a", "lora_b"] | None,
) -> tuple[Linear | GroupedLinear, Linear | GroupedLinear]:
    """Build the adapters of a grouped projection.

    Per expert, A is ``[E, L, I]`` and B ``[E, O, L]`` or ``[E, num_linears, O, L]``.
    A shared factor drops the expert axis: A ``[L, I]`` when ``shared_factor`` is
    ``"lora_a"``, B ``[O, L]`` or ``[num_linears, O, L]`` when it is ``"lora_b"``.
    """
    lora_a_sharding, lora_b_sharding = _grouped_lora_adapter_sharding(
        config.sharding_config
    )
    shared_sharding = _shared_lora_adapter_sharding(config.sharding_config)
    lora_a_init = {
        "weight": lambda w: nn.init.kaiming_uniform_(w.flatten(0, -2), a=math.sqrt(5)),
    }
    lora_a: Module.Config
    if shared_factor == "lora_a":
        lora_a = Linear.Config(
            in_features=config.in_features,
            out_features=rank,
            bias=False,
            # Applied by _adapter_matmul, never through Linear._linear.
            inplace_wgrad_accum=False,
            sharding_config=shared_sharding,
            param_init=lora_a_init,
        )
    else:
        lora_a = GroupedLinear.Config(
            group_size=config.group_size,
            in_features=config.in_features,
            out_features=rank,
            sharding_config=lora_a_sharding,
            param_init=lora_a_init,
        )
    lora_b: Module.Config
    if shared_factor == "lora_b":
        lora_b = Linear.Config(
            in_features=rank,
            out_features=config.out_features,
            num_linears=config.num_linears,
            bias=False,
            inplace_wgrad_accum=False,
            sharding_config=shared_sharding,
            param_init={"weight": nn.init.zeros_},
        )
    else:
        lora_b = GroupedLinear.Config(
            group_size=config.group_size,
            in_features=rank,
            out_features=config.out_features,
            num_linears=config.num_linears,
            sharding_config=lora_b_sharding,
            param_init={"weight": nn.init.zeros_},
        )
    return lora_a.build(), lora_b.build()


def _adapter_matmul(
    adapter: Linear | GroupedLinear,
    input_RI: torch.Tensor,
    offsets_E: torch.Tensor,
) -> torch.Tensor:
    """Apply one adapter factor to the routed rows: grouped by expert, or shared."""
    if isinstance(adapter, GroupedLinear):
        return adapter._grouped_mm(
            input_RI=input_RI,
            weight_EOI=adapter.weight.flatten(1, -2),
            offsets_E=offsets_E,
        )
    # Respect the same row boundaries as the per-expert path: padded tail rows
    # must not contribute to the shared weight gradient. Expansion is zero-copy.
    weight_1OI = adapter.weight.bfloat16().flatten(0, -2).unsqueeze(0)
    weight_EOI = weight_1OI.expand(offsets_E.shape[0], -1, -1)
    return torch._grouped_mm(input_RI, weight_EOI.transpose(-2, -1), offs=offsets_E)


def _adapter_operands(
    adapter_a: Linear | GroupedLinear, adapter_b: Linear | GroupedLinear
) -> tuple[torch.Tensor, torch.Tensor]:
    """``B [E, O, L]`` and ``A [E, L, I]`` per expert, a shared factor as a
    stride-0 expanded view."""
    if isinstance(adapter_a, Linear):
        weight_b_EOL = adapter_b.weight.flatten(1, -2)
        weight_a_ELI = adapter_a.weight.unsqueeze(0).expand(
            weight_b_EOL.shape[0], -1, -1
        )
    elif isinstance(adapter_b, Linear):
        weight_a_ELI = adapter_a.weight
        weight_b_EOL = (
            adapter_b.weight.flatten(0, -2)
            .unsqueeze(0)
            .expand(weight_a_ELI.shape[0], -1, -1)
        )
    else:
        weight_a_ELI = adapter_a.weight
        weight_b_EOL = adapter_b.weight.flatten(1, -2)
    return weight_b_EOL, weight_a_ELI


class _LoRAGroupedLinearMixin(_LoRAMixin):
    """Add a LoRA update to a GroupedLinear.

    Both factors are per expert unless ``config.shared_factor`` names one that
    all experts share (see ``LoRATransform.expert_adapter_layout``).
    """

    num_linears: int
    lora_a: Linear | GroupedLinear
    lora_b: Linear | GroupedLinear

    def __init__(self, config) -> None:
        super().__init__(config)
        self.lora_a, self.lora_b = _build_grouped_lora_adapters(
            config,
            rank=config.rank,
            shared_factor=config.shared_factor,
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
        # The adapters call their local matmul directly, so they run inside this
        # projection's remat region instead of declaring their own (see
        # _LoRALinearMixin._linear).
        lora_hidden_RL = _adapter_matmul(self.lora_a, input_RI, offsets_E)
        lora_out_RO = _adapter_matmul(self.lora_b, lora_hidden_RL, offsets_E)
        return base_out_RO + self._lora_scaling * lora_out_RO


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
    disabled, returns a fresh full-sized gradient for it. The baddbmm backward
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
            w13_EFD = torch.baddbmm(
                base_w13_EFD,
                *_adapter_operands(w13.lora_a, w13.lora_b),
                beta=1,
                alpha=w13._lora_scaling,
            )
        w2_EDF = base_w2_EDF
        w2 = self.w2
        if isinstance(w2, _LoRAGroupedLinearMixin):
            w2_EDF = torch.baddbmm(
                base_w2_EDF,
                *_adapter_operands(w2.lora_a, w2.lora_b),
                beta=1,
                alpha=w2._lora_scaling,
            )
        return w13_EFD, w2_EDF


def _name_lora_class(lora_cls: type[Module], parent_cls: type[Module]) -> type[Module]:
    lora_cls.__name__ = f"LoRA{parent_cls.__name__}"
    lora_cls.__qualname__ = f"LoRA{parent_cls.__name__}"
    return lora_cls


@functools.cache
def get_lora_linear(parent_cls: type[Module]) -> type[Module]:
    """Get a cached LoRA version of a linear module class."""
    parent_config_cls = parent_cls.Config

    class LoRAProjection(_LoRALinearMixin, parent_cls):  # type: ignore[misc, valid-type]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            rank: int
            alpha: float

    return _name_lora_class(LoRAProjection, parent_cls)


@functools.cache
def get_lora_grouped_linear(parent_cls: type[Module]) -> type[Module]:
    """Get a cached LoRA version of a grouped-linear module class."""
    parent_config_cls = parent_cls.Config

    class LoRAProjection(_LoRAGroupedLinearMixin, parent_cls):  # type: ignore[misc, valid-type]
        @dataclass(kw_only=True, slots=True)
        class Config(parent_config_cls):  # type: ignore[misc]
            rank: int
            alpha: float
            shared_factor: Literal["lora_a", "lora_b"] | None
            """Adapter factor every expert shares; ``None`` keeps both per expert."""

    return _name_lora_class(LoRAProjection, parent_cls)


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
