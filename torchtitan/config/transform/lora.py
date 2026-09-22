# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, fields
from typing import Any, cast, ClassVar, Protocol

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.lora import get_lora_linear
from torchtitan.protocols.module import Module

from .base import ModelConfigTransform
from .context_parallel import ContextParallelTransform


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
    def config_type(self) -> type[Module.Config]:
        ...

    def make_config(
        self,
        cfg: Module.Config,
        *,
        rank: int,
        alpha: float,
    ) -> Module.Config:
        ...


class LinearLoRAHandler:
    """Convert ``Linear.Config`` instances to LoRA-enabled configs."""

    config_type = Linear.Config

    def make_config(
        self,
        cfg: Module.Config,
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


@dataclass(kw_only=True, slots=True)
class LoRATransform(ModelConfigTransform):
    """Apply LoRA adapters to supported projection layers in a model.

    ``handlers`` defines the projection config types supported by this
    transform. Include ``LinearLoRAHandler`` to adapt ``Linear.Config``
    instances. Non-target modules are replaced with dynamic frozen config
    subclasses that freeze direct parameters at build time.

    When ``target_modules`` is None (default), every supported projection is
    converted. When specified, only configs whose FQN's last segment matches
    one of the entries are converted (e.g. ``["wq", "wv"]``).

    This transform conflicts with itself because every application freezes all
    non-target configs. Applying multiple LoRA transforms would make freezing
    and adapter configuration depend on their order.
    """

    # TODO: Add quantization transforms here after they migrate from
    # ModelConfigConverter so LoRA always wraps an already quantized linear.
    run_after: ClassVar[tuple[type[ModelConfigTransform], ...]] = (
        ContextParallelTransform,
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

    def transform(self, model: Module.Config) -> Module.Config:
        """Walk the module config tree from leaves to root.

        Target projection modules get their config replaced with an adapter
        config. All other module configs become frozen config subclasses so
        LoRA training updates only adapter parameters.
        """
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
