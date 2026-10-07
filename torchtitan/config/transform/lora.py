# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, fields
from typing import Any, cast

from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.models.common.lora import (
    get_lora_dist_moe_routed_experts,
    get_lora_grouped_linear,
    get_lora_linear,
)
from torchtitan.protocols.module import Module

from .base import ModelConfigTransform, ModelConfigTransformContext


logger = logging.getLogger(__name__)


_frozen_config_class_cache: dict[type, type] = {}


def _matches_fqn(fqn: str, target: str) -> bool:
    return fqn == target or fqn.endswith(f".{target}")


def _validate_shared_projection_target_consistency(
    configs: list[tuple[str, Module.Config, object | None, str | int | None]],
    target_module_names: set[str] | None,
) -> None:
    projection_occurrences: dict[int, tuple[Module.Config, list[str], list[str]]] = {}
    for fqn, cfg, _parent, _attr in configs:
        if not isinstance(cfg, (Linear.Config, GroupedLinear.Config)):
            continue
        is_target = target_module_names is None or any(
            _matches_fqn(fqn, target) for target in target_module_names
        )
        _, lora_fqns, frozen_fqns = projection_occurrences.setdefault(
            id(cfg), (cfg, [], [])
        )
        (lora_fqns if is_target else frozen_fqns).append(fqn)

    for cfg, lora_fqns, frozen_fqns in projection_occurrences.values():
        if lora_fqns and frozen_fqns:
            raise ValueError(
                f"Shared {type(cfg).__qualname__} has inconsistent LoRA "
                f"decisions: LoRA at {sorted(lora_fqns)}, frozen at "
                f"{sorted(frozen_fqns)}. Create separate config objects to "
                "target these paths independently."
            )


def _validate_weight_tied_lm_head_target(
    configs: list[tuple[str, Module.Config, object | None, str | int | None]],
    target_module_names: set[str] | None,
) -> None:
    for fqn, cfg, parent, attr in configs:
        if not isinstance(cfg, Linear.Config) or attr != "lm_head":
            continue
        if not getattr(parent, "enable_weight_tying", False):
            continue

        is_target = target_module_names is None or any(
            _matches_fqn(fqn, target) for target in target_module_names
        )
        if is_target:
            raise ValueError(
                f"LoRA cannot target {fqn!r} while weight tying is "
                "enabled because tok_embeddings and lm_head share their base "
                "weight, but only lm_head would receive the adapter. Disable "
                "weight tying or exclude lm_head from target_modules."
            )


def _get_frozen_config_cls(
    config_cls: type[Module.Config],
) -> type[Module.Config]:
    """Get or create a config subclass that freezes direct build parameters."""
    if config_cls in _frozen_config_class_cache:
        return _frozen_config_class_cache[config_cls]

    class FrozenConfig(config_cls):  # type: ignore[valid-type, misc]
        def build(self, **kwargs):
            instance = config_cls.build(self, **kwargs)
            for param in instance.parameters(recurse=False):
                param.requires_grad_(False)
            return instance

    FrozenConfig.__name__ = f"Frozen{config_cls.__name__}"
    FrozenConfig.__qualname__ = f"Frozen{config_cls.__qualname__}"
    _frozen_config_class_cache[config_cls] = FrozenConfig
    return FrozenConfig


def _make_frozen_config(cfg: Module.Config) -> Module.Config:
    """Create a frozen config that still passes checks for the original type."""
    frozen_cls = _get_frozen_config_cls(type(cfg))
    return frozen_cls(**{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init})


def _make_linear_lora_config(
    cfg: Linear.Config,
    *,
    rank: int,
    alpha: float,
) -> Module.Config:
    assert cfg._owner is not None
    lora_cls = get_lora_linear(cast(type[Module], cfg._owner))
    lora_config_cls = cast(Any, lora_cls.Config)
    return lora_config_cls(
        **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
        rank=rank,
        alpha=alpha,
    )


def _make_grouped_linear_lora_config(
    cfg: GroupedLinear.Config,
    *,
    rank: int,
    alpha: float,
) -> Module.Config:
    if rank % 8:
        raise ValueError(f"Grouped LoRA rank must be divisible by 8, got {rank}")
    assert cfg._owner is not None
    lora_cls = get_lora_grouped_linear(cast(type[Module], cfg._owner))
    lora_config_cls = cast(Any, lora_cls.Config)
    return lora_config_cls(
        **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
        rank=rank,
        alpha=alpha,
    )


def _make_dist_moe_lora_config(
    cfg: DistMoeRoutedExperts.Config,
) -> Module.Config:
    """Wrap Dist-MoE so it consumes LoRA-enabled grouped projections."""
    owner = cfg._owner
    if owner is not DistMoeRoutedExperts:
        owner_name = owner.__qualname__ if owner is not None else "None"
        raise ValueError(
            "Dist-MoE LoRA supports only DistMoeRoutedExperts configs, got "
            f"{owner_name}."
        )
    if cfg.inplace_wgrad_accum:
        raise ValueError(
            "Dist-MoE LoRA requires inplace_wgrad_accum=False because its "
            "effective weights are transient tensors."
        )
    lora_grouped_linear = get_lora_grouped_linear(GroupedLinear)
    supported_owners = (GroupedLinear, lora_grouped_linear)
    if cfg.w13._owner not in supported_owners or cfg.w2._owner not in supported_owners:
        w13_owner = cfg.w13._owner
        w2_owner = cfg.w2._owner
        w13_owner_name = w13_owner.__qualname__ if w13_owner is not None else "None"
        w2_owner_name = w2_owner.__qualname__ if w2_owner is not None else "None"
        raise ValueError(
            "Dist-MoE LoRA subtree conflict: W13 and W2 must be stock or LoRA "
            "GroupedLinear configs, got "
            f"w13={w13_owner_name} and w2={w2_owner_name}."
        )

    return get_lora_dist_moe_routed_experts().Config(
        **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
    )


@dataclass(kw_only=True, slots=True)
class LoRATransform(ModelConfigTransform):
    """Apply LoRA adapters to supported projection layers in a model.

    Built-in dispatch selects the correct LoRA implementation for each target.
    Non-target configs are frozen so only adapter parameters are trainable.

    A shared projection config must receive the same LoRA decision at every FQN.
    For example, if ``layers.0.attention.wo`` and ``layers.1.attention.wo``
    reference the same ``Linear.Config``, targeting only the first is rejected.
    Target both occurrences or construct separate config objects instead.

    Targeting ``lm_head`` is rejected when weight tying is enabled because the
    tied token embedding would share only its base weight, not the adapter.

    Dist-MoE W13 and W2 projections are selected independently by their logical
    FQNs. Each selected child owns its grouped-linear adapters and scaling. The
    Dist-MoE parent is wrapped only to materialize effective weights because its
    fused execution bypasses the child forwards.

    This transform conflicts with itself because every application freezes all
    non-target configs. Applying multiple LoRA transforms would make freezing
    and adapter configuration depend on their order.
    """

    rank: int = 8
    """Rank of the LoRA matrices."""

    alpha: float = 16.0
    """Scaling factor. Output is scaled by alpha/rank."""

    target_modules: list[str] | None = None
    """Module FQNs or dot-delimited FQN suffixes to adapt.

    Examples:

    - ``"layers.0.moe.routed_experts.w2"``: one exact projection.
    - ``"routed_experts.w2"``: every matching FQN suffix.
    - ``"w2"``: every supported projection named w2.
    - ``"*.w2"``: no matches; glob patterns are not supported.

    ``None`` means all supported projection layers. An empty list means no
    layers.
    """

    def __post_init__(self) -> None:
        if self.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.rank}")
        if self.target_modules is None:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha} "
                f"(all supported projection layers)"
            )
        else:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha}, "
                f"target_modules={sorted(self.target_modules)}"
            )

    def transform(
        self,
        model: Module.Config,
        *,
        context: ModelConfigTransformContext | None = None,
    ) -> Module.Config:
        """Apply LoRA while walking the config tree from leaves to root.

        This lets Dist-MoE parents observe whether their projection children
        were converted to LoRA. All other configs are frozen so only adapters
        are trainable.
        """
        del context
        transformed_root = model
        target_module_names = (
            set(self.target_modules) if self.target_modules is not None else None
        )

        matched_targets = set()
        configs = list(model.traverse(Module.Config, recurse=True))
        _validate_shared_projection_target_consistency(
            configs,
            target_module_names,
        )
        _validate_weight_tied_lm_head_target(configs, target_module_names)

        # Walk the traversal backward so Dist-MoE parents observe converted
        # projection children when selecting their wrapper.
        for fqn, cfg, parent, attr in reversed(configs):
            is_target = target_module_names is None
            if isinstance(cfg, (Linear.Config, GroupedLinear.Config)):
                for target in target_module_names or ():
                    if _matches_fqn(fqn, target):
                        matched_targets.add(target)
                        is_target = True

            if isinstance(cfg, DistMoeRoutedExperts.Config):
                # The fused path bypasses child forwards. Wrap the parent only
                # when it must materialize an adapted child's effective weight.
                lora_grouped_linear = get_lora_grouped_linear(GroupedLinear)
                if (
                    cfg.w13._owner is lora_grouped_linear
                    or cfg.w2._owner is lora_grouped_linear
                ):
                    new_cfg = _make_dist_moe_lora_config(cfg)
                else:
                    new_cfg = _make_frozen_config(cfg)
            elif isinstance(cfg, GroupedLinear.Config) and is_target:
                # Keep adapters on the projection even when Dist-MoE consumes
                # its weight directly instead of calling its forward.
                new_cfg = _make_grouped_linear_lora_config(
                    cfg,
                    rank=self.rank,
                    alpha=self.alpha,
                )
            elif isinstance(cfg, Linear.Config) and is_target:
                new_cfg = _make_linear_lora_config(
                    cfg,
                    rank=self.rank,
                    alpha=self.alpha,
                )
            else:
                new_cfg = _make_frozen_config(cfg)

            if parent is None:
                transformed_root = new_cfg
            elif isinstance(parent, list):
                assert isinstance(attr, int)
                parent[attr] = new_cfg
            else:
                assert isinstance(attr, str)
                setattr(parent, attr, new_cfg)

        unmatched_targets = (target_module_names or set()) - matched_targets
        if unmatched_targets:
            logger.warning(
                f"LoRA target_modules {sorted(unmatched_targets)} did not match any "
                f"supported projection config in the model config tree."
            )
        return transformed_root
