# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, replace
from typing import ClassVar, Protocol

from torchtitan.models.common.linear import Linear
from torchtitan.models.common.lora import LinearLoRADecorator
from torchtitan.protocols.module import Module

from .base import ModelConfigTransform
from .context_parallel import ContextParallelTransform


logger = logging.getLogger(__name__)


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
        assert isinstance(cfg, Linear.Config)
        return replace(
            cfg,
            _module_decorators=(
                *cfg._module_decorators,
                LinearLoRADecorator(rank=rank, alpha=alpha),
            ),
            _freeze_direct_parameters=True,
        )


@dataclass(kw_only=True, slots=True)
class LoRATransform(ModelConfigTransform):
    """Apply LoRA adapters to supported projection layers in a model.

    ``handlers`` defines the projection config types supported by this
    transform. Include ``LinearLoRAHandler`` to decorate ``Linear.Config``
    instances at build time. Direct parameters on all existing modules are
    frozen so only adapter parameters remain trainable.

    When ``target_modules`` is None (default), every supported projection is
    converted. When specified, only configs whose FQN's last segment matches
    one of the entries are converted (e.g. ``["wq", "wv"]``).

    This transform conflicts with itself because every application freezes all
    non-target configs. Applying multiple LoRA transforms would make freezing
    and adapter configuration depend on their order.
    """

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

        Target projection modules record a build-time adapter decoration. All
        existing module configs freeze their direct parameters so LoRA training
        updates only adapter parameters.
        """
        transformed_root = model
        matched = set()
        configs = list(model.traverse(Module.Config, recurse=True))
        target_module_names = (
            set(self.target_modules) if self.target_modules is not None else None
        )

        for fqn, cfg, parent, attr in reversed(configs):
            assert isinstance(cfg, Module.Config)
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
                new_cfg = cfg
                new_cfg._freeze_direct_parameters = True

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
