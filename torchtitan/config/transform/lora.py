# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# TODO: consider not exposing handlers and let the user target
# fqns directly, and transform handles applying the correct LoRA.

import logging
from dataclasses import dataclass, fields
from typing import Any, cast, ClassVar, Protocol

from torchtitan.models.common.dist_moe import DistMoeRoutedExperts
from torchtitan.models.common.linear import GroupedLinear, Linear
from torchtitan.models.common.lora import (
    get_lora_dist_moe_routed_experts,
    get_lora_grouped_linear,
    get_lora_linear,
)
from torchtitan.protocols.module import Module

from .base import ModelConfigTransform, ModelConfigTransformContext
from .context_parallel import ContextParallelTransform
from .dist_moe import DistMoeTransform


logger = logging.getLogger(__name__)


_frozen_config_class_cache: dict[type, type] = {}


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


class _LoRAHandler(Protocol):
    @property
    def config_type(self) -> type[Module.Config]: ...

    def make_config(
        self,
        cfg: Module.Config,
        *,
        parent: Module.Config | list[Any] | None,
        fqn: str,
        rank: int,
        alpha: float,
    ) -> Module.Config: ...


class LinearLoRAHandler:
    """Convert ``Linear.Config`` instances to LoRA-enabled configs."""

    config_type = Linear.Config

    def make_config(
        self,
        cfg: Module.Config,
        *,
        parent: Module.Config | list[Any] | None,
        fqn: str,
        rank: int,
        alpha: float,
    ) -> Module.Config:
        del parent, fqn
        assert cfg._owner is not None
        lora_cls = get_lora_linear(cast(type[Module], cfg._owner))
        lora_config_cls = cast(Any, lora_cls.Config)
        return lora_config_cls(
            **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
            rank=rank,
            alpha=alpha,
        )


class GroupedLinearLoRAHandler:
    """Convert ``GroupedLinear.Config`` instances to LoRA-enabled configs."""

    config_type = GroupedLinear.Config

    def make_config(
        self,
        cfg: Module.Config,
        *,
        parent: Module.Config | list[Any] | None,
        fqn: str,
        rank: int,
        alpha: float,
    ) -> Module.Config:
        if isinstance(parent, DistMoeRoutedExperts.Config):
            raise ValueError(
                f"GroupedLinearLoRAHandler cannot target {fqn!r} under "
                f"{type(parent).__qualname__}: Dist-MoE reads the projection's "
                "weight directly and does not call GroupedLinear.forward(), so "
                "its LoRA adapter would be ignored. Target the routed-experts "
                "parent with DistMoeLoRAHandler instead; Dist-MoE LoRA currently "
                "supports BF16 only."
            )
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


class DistMoeLoRAHandler:
    """Convert BF16 ``DistMoeRoutedExperts.Config`` instances to LoRA."""

    config_type = DistMoeRoutedExperts.Config

    def make_config(
        self,
        cfg: Module.Config,
        *,
        parent: Module.Config | list[Any] | None,
        fqn: str,
        rank: int,
        alpha: float,
    ) -> Module.Config:
        del parent, fqn
        owner = cfg._owner
        if owner is not DistMoeRoutedExperts:
            owner_name = owner.__qualname__ if owner is not None else "None"
            raise ValueError(
                "Dist-MoE LoRA supports only DistMoeRoutedExperts configs, got "
                f"{owner_name}."
            )

        dist_moe_cfg = cast(DistMoeRoutedExperts.Config, cfg)
        if dist_moe_cfg.inplace_wgrad_accum:
            raise ValueError(
                "Dist-MoE LoRA requires inplace_wgrad_accum=False because its "
                "effective weights are transient tensors."
            )
        if (
            dist_moe_cfg.w13._owner is not GroupedLinear
            or dist_moe_cfg.w2._owner is not GroupedLinear
        ):
            w13_owner = dist_moe_cfg.w13._owner
            w2_owner = dist_moe_cfg.w2._owner
            w13_owner_name = w13_owner.__qualname__ if w13_owner is not None else "None"
            w2_owner_name = w2_owner.__qualname__ if w2_owner is not None else "None"
            raise ValueError(
                "Dist-MoE LoRA subtree conflict: W13 and W2 must retain stock "
                "GroupedLinear owners, got "
                f"w13={w13_owner_name} and w2={w2_owner_name}."
            )

        lora_cls = get_lora_dist_moe_routed_experts(DistMoeRoutedExperts)
        lora_config_cls = cast(Any, lora_cls.Config)
        return lora_config_cls(
            **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
            rank=rank,
            alpha=alpha,
        )


@dataclass(kw_only=True, slots=True)
class LoRATransform(ModelConfigTransform):
    """Apply LoRA adapters to supported projection layers in a model.

    ``handlers`` defines the projection config types supported by this
    transform. Include ``LinearLoRAHandler``, ``GroupedLinearLoRAHandler``,
    and ``DistMoeLoRAHandler` to adapt their respective projection configs.
    Non-target modules are replaced with dynamic frozen config subclasses
    that freeze direct parameters at build time.

    When ``target_modules`` is None (default), every supported projection is
    converted. When specified, only configs whose FQN's last segment matches
    one of the entries are converted (e.g. ``["wq", "wv"]``). The
    ``"routed_experts"`` parent target adapts its W13 and W2 projections
    together when using ``DistMoeLoRAHandler``.

    This transform conflicts with itself because every application freezes all
    non-target configs. Applying multiple LoRA transforms would make freezing
    and adapter configuration depend on their order.
    """

    run_after: ClassVar[tuple[type[ModelConfigTransform], ...]] = (
        ContextParallelTransform,
        DistMoeTransform,
    )

    handlers: tuple[_LoRAHandler, ...]
    """Handlers for the projection config types that support LoRA."""

    rank: int = 8
    """Rank of the LoRA matrices."""

    alpha: float = 16.0
    """Scaling factor. Output is scaled by alpha/rank."""

    target_modules: list[str] | None = None
    """Module names to adapt, matched against the last FQN segment.

    ``None`` means all supported projection layers. An empty list means no
    layers.
    """

    def __post_init__(self) -> None:
        for index, handler in enumerate(self.handlers):
            for earlier in self.handlers[:index]:
                if issubclass(handler.config_type, earlier.config_type):
                    raise ValueError(
                        f"{type(handler).__qualname__} for "
                        f"{handler.config_type.__qualname__} is shadowed by earlier "
                        f"handler {type(earlier).__qualname__} for "
                        f"{earlier.config_type.__qualname__}."
                    )
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
        """Walk the module config tree from leaves to root.

        Target projection modules get their config replaced with an adapter
        config. All other module configs become frozen config subclasses so
        LoRA training updates only adapter parameters.
        """
        del context
        transformed_root = model
        matched = set()
        configs = list(model.traverse(Module.Config, recurse=True))
        target_module_names = (
            set(self.target_modules) if self.target_modules is not None else None
        )

        for fqn, cfg, parent, attr in reversed(configs):
            last_segment = fqn.rsplit(".", 1)[-1]
            handler = next(
                (
                    handler
                    for handler in self.handlers
                    if isinstance(cfg, handler.config_type)
                ),
                None,
            )
            is_target = handler is not None and (
                target_module_names is None or last_segment in target_module_names
            )

            if is_target:
                assert handler is not None
                new_cfg = handler.make_config(
                    cfg,
                    parent=cast(Module.Config | list[Any] | None, parent),
                    fqn=fqn,
                    rank=self.rank,
                    alpha=self.alpha,
                )
                matched.add(last_segment)
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

        unmatched = (target_module_names or set()) - matched
        if unmatched:
            logger.warning(
                f"LoRA target_modules {sorted(unmatched)} did not match any "
                f"supported projection config in the model config tree."
            )
        return transformed_root


LoRATransform.conflicts_with = (LoRATransform,)
