# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, fields
from typing import ClassVar, Protocol

import spmd_types as spmd

import torch
import torch.nn as nn

from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.module import Module
from torchtitan.protocols.sharding import ShardingConfig

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


class _LinearLoRAHandler:
    config_type = Linear.Config

    def __init__(self) -> None:
        self._class_cache: dict[type, type] = {}

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
        else:
            assert base_weight_sharding == dense_param_placement(tp=spmd.S(1))
            lora_a_sharding = ShardingConfig(
                state_shardings={"weight": dense_param_placement(tp=spmd.S(1))},
            )
            return lora_a_sharding, replicated_weight

    def _get_lora_cls(self, parent_cls: type) -> type:
        """Get or create a LoRA subclass for a linear implementation."""
        if parent_cls in self._class_cache:
            return self._class_cache[parent_cls]

        parent_config_cls = parent_cls.Config  # pyrefly: ignore [missing-attribute]
        adapter_sharding = type(self)._adapter_sharding

        class LoRALinear(parent_cls):  # type: ignore[valid-type, misc]
            @dataclass(kw_only=True, slots=True)
            class Config(parent_config_cls):  # type: ignore[misc]
                rank: int
                alpha: float

            def __init__(self, config: Config) -> None:
                super().__init__(config)
                for param in nn.Module.parameters(self):
                    param.requires_grad_(False)
                self._lora_scaling = config.alpha / config.rank
                lora_a_sharding, lora_b_sharding = adapter_sharding(
                    config.sharding_config
                )
                self.lora_a = Linear.Config(
                    in_features=config.in_features,
                    out_features=config.rank,
                    bias=False,
                    sharding_config=lora_a_sharding,
                    param_init={
                        "weight": lambda w: nn.init.kaiming_uniform_(w, a=math.sqrt(5)),
                    },
                ).build()
                self.lora_b = Linear.Config(
                    in_features=config.rank,
                    out_features=config.out_features,
                    bias=False,
                    sharding_config=lora_b_sharding,
                    param_init={"weight": nn.init.zeros_},
                ).build()

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                base_out = super().forward(input)
                lora_out = self.lora_b(self.lora_a(input))
                return base_out + self._lora_scaling * lora_out

        LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
        LoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
        self._class_cache[parent_cls] = LoRALinear
        return LoRALinear

    def make_config(
        self,
        cfg: Module.Config,
        *,
        rank: int,
        alpha: float,
    ) -> Module.Config:
        assert cfg._owner is not None
        lora_cls = self._get_lora_cls(cfg._owner)
        return lora_cls.Config(  # pyrefly: ignore [missing-attribute]
            **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
            rank=rank,
            alpha=alpha,
        )


@dataclass(kw_only=True, slots=True)
class LoRATransform(ModelConfigTransform):
    """Apply LoRA adapters to supported projection layers in a model.

    The base transform supports ``Linear.Config``. Subclasses may extend
    ``handlers`` for other projection types. Non-target modules are replaced
    with dynamic frozen config subclasses that freeze direct parameters at
    build time.

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
    handlers: ClassVar[tuple[_LoRAHandler, ...]] = (_LinearLoRAHandler(),)

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
                assert not issubclass(handler.config_type, earlier.config_type), (
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
