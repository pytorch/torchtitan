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
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.base.module import Module
from torchtitan.models.base.sharding import ShardingConfig

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.models.common.moe_sharding import expert_param_placement_sparse

__all__ = [
    "get_lora_dist_moe_routed_experts",
    "get_lora_grouped_linear",
    "get_lora_linear",
    "merge_lora_state_dict",
    "trainable_state_dict",
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
        if config.num_linears > 1:
            # A stacked base projection shares one A matrix across its logical
            # linears and stacks their B matrices along the same output axis as
            # the base weight. The adapters inherit only parameter sharding;
            # the base projection remains responsible for TP collectives.
            replicated_weight = ShardingConfig(
                state_shardings={"weight": dense_param_placement(tp=spmd.R)},
            )
            lora_a_sharding = (
                replicated_weight if config.sharding_config is not None else None
            )
            lora_b_sharding = (
                ShardingConfig(
                    state_shardings=dict(config.sharding_config.state_shardings),
                )
                if config.sharding_config is not None
                else None
            )
        else:
            lora_a_sharding, lora_b_sharding = self._adapter_sharding(
                config.sharding_config
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
    ) -> tuple[ShardingConfig | None, ShardingConfig | None]:
        """Derive adapter sharding from the base linear's TP sharding."""
        base_weight_sharding = (
            base_sharding.state_shardings.get("weight") if base_sharding else None
        )
        if base_weight_sharding is None:
            return None, None

        replicated_weight = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        )
        if base_weight_sharding == dense_param_placement(tp=spmd.S(0)):
            lora_b_sharding = ShardingConfig(
                state_shardings={"weight": base_weight_sharding},
            )
            return replicated_weight, lora_b_sharding

        assert base_weight_sharding == dense_param_placement(tp=spmd.S(1))
        lora_a_sharding = ShardingConfig(
            state_shardings={"weight": dense_param_placement(tp=spmd.S(1))},
        )
        return lora_a_sharding, replicated_weight


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
                w13.lora_b.weight.flatten(1, -2),
                w13.lora_a.weight,
                beta=1,
                alpha=w13._lora_scaling,
            )
        w2_EDF = base_w2_EDF
        w2 = self.w2
        if isinstance(w2, _LoRAGroupedLinearMixin):
            w2_EDF = torch.baddbmm(
                base_w2_EDF,
                w2.lora_b.weight,
                w2.lora_a.weight,
                beta=1,
                alpha=w2._lora_scaling,
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


_WRAPPER_SEGMENTS = frozenset(
    {"_checkpoint_wrapped_module", "_fsdp_wrapped_module", "_orig_mod"}
)


def _strip_wrappers(name: str) -> str:
    return ".".join(part for part in name.split(".") if part not in _WRAPPER_SEGMENTS)


def _key(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _state_dict_prefix(module_name: str, state_dict: dict[str, torch.Tensor]) -> str:
    for candidate in (_strip_wrappers(module_name), module_name):
        if _key(candidate, "lora_a.weight") in state_dict:
            return candidate
    raise KeyError(
        f"LoRA module {module_name!r} has no adapter in the state dict; "
        "an unknown module wrapper is in its path."
    )


def _merged_weight(module: _LoRAMixin, base: torch.Tensor) -> torch.Tensor:
    module = cast(_LoRALinearMixin | _LoRAGroupedLinearMixin, module)
    grouped = isinstance(module, _LoRAGroupedLinearMixin)
    lora_a, lora_b, weight = module.lora_a.weight, module.lora_b.weight, base
    if isinstance(base, DTensor):
        # B shares the base's output sharding; A is only cut on its rank dim, by FSDP.
        rank_dim = 1 if grouped else 0
        lora_a = lora_a.redistribute(
            placements=[
                Replicate() if placement.is_shard(rank_dim) else placement
                for placement in lora_a.placements
            ]
        ).to_local()
        lora_b, weight = lora_b.to_local(), base.to_local()
    lora_a, lora_b = lora_a.float(), lora_b.float()
    if grouped:
        delta = torch.bmm(lora_b.flatten(1, -2), lora_a).view(weight.shape)
    else:
        delta = torch.matmul(lora_b, lora_a)
    merged = (weight.float() + module._lora_scaling * delta).to(base.dtype)
    if isinstance(base, DTensor):
        return DTensor.from_local(
            merged,
            base.device_mesh,
            base.placements,
            run_check=False,
            shape=base.shape,
            stride=base.stride(),
        )
    return merged


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """The parameters left trainable, keyed as in ``state_dict()``: under LoRA, the adapters."""
    return {
        _strip_wrappers(name): param
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def merge_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """``model.state_dict()`` with each base weight replaced by ``W + (alpha / rank) * B @ A`` and no adapter keys."""
    state_dict = dict(model.state_dict())
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, _LoRAMixin):
                continue
            prefix = _state_dict_prefix(name, state_dict)
            key = _key(prefix, "weight")
            state_dict[key] = _merged_weight(module, state_dict[key])
            del state_dict[_key(prefix, "lora_a.weight")]
            del state_dict[_key(prefix, "lora_b.weight")]
    return state_dict
