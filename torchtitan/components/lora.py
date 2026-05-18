# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn

from torch.distributed.tensor import Replicate

from torchtitan.config import Configurable
from torchtitan.models.common.decoder_sharding import dense_param_placement
from torchtitan.models.common.linear import Linear
from torchtitan.protocols.model import ModelConfigConverter
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.logging import logger


def _lora_adapter_sharding(
    base_sharding: ShardingConfig | None,
) -> tuple[ShardingConfig | None, ShardingConfig | None]:
    """Derive LoRA adapter sharding from the base linear's TP sharding.

    ``lora_b`` mirrors the base weight's TP shard so its output matches
    ``base_out``'s placement (Shard(-1) for colwise, Partial for rowwise).

    ``lora_a`` weight is Replicate (small rank dim, no benefit from TP
    sharding).
    """
    base_weight_sharding = (
        base_sharding.state_shardings.get("weight") if base_sharding else None
    )
    if base_weight_sharding is None:
        return None, None

    replicate_weight = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=Replicate())},
    )
    # lora_b: same TP shard as base weight → output matches base_out
    lora_b_sharding = ShardingConfig(
        state_shardings={"weight": base_weight_sharding},
    )
    return replicate_weight, lora_b_sharding


_lora_class_cache: dict[type, type] = {}


def _get_lora_cls(parent_cls: type) -> type:
    """Get or create a LoRA subclass for *parent_cls* (e.g. Linear, Float8Linear).

    The returned class has a proper ``Config`` that extends the parent's Config
    with ``rank`` and ``alpha``.  Adapters are built in ``__init__`` from the
    base config's dimensions and sharding.
    """
    if parent_cls in _lora_class_cache:
        return _lora_class_cache[parent_cls]

    parent_config_cls = parent_cls.Config  # pyrefly: ignore [missing-attribute]

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
            lora_a_sharding, lora_b_sharding = _lora_adapter_sharding(
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
    _lora_class_cache[parent_cls] = LoRALinear
    return LoRALinear


@dataclass(kw_only=True, slots=True)
class FrozenConfig(Configurable.Config):
    """Config wrapper that freezes all parameters at build time.

    Works with any ``Module.Config`` — used by ``LoRAConverter`` to freeze
    non-target Linear modules.
    """

    inner: Configurable.Config

    def __getattr__(self, name: str):
        try:
            inner = object.__getattribute__(self, "inner")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(inner, name)

    def __setattr__(self, name: str, value) -> None:
        if name in type(self).__slots__:
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def build(self, **kwargs):
        instance = self.inner.build(**kwargs)
        instance.requires_grad_(False)
        return instance


class LoRAConverter(ModelConfigConverter):
    """Apply LoRA adapters to Linear layers in a model.

    Operates on the model config tree: target Linear configs are replaced
    with ``LoRALinear.Config`` (which builds a LoRA subclass with frozen base
    and trainable adapters). Non-target Linear modules are wrapped with
    ``FrozenConfig``.

    When ``target_modules`` is None (default), every ``Linear.Config`` is
    converted.  When specified, only configs whose FQN's last segment matches
    one of the entries are converted (e.g. ``["wq", "wv"]``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ModelConfigConverter.Config):
        rank: int = 8
        """Rank of the LoRA matrices."""

        alpha: float = 16.0
        """Scaling factor. Output is scaled by alpha/rank."""

        target_modules: list[str] | None = None
        """Module names to apply LoRA to (matched against the last segment of the FQN).
        None means all Linear layers. An empty list means no layers."""

    def __init__(self, config: Config, **kwargs):
        if config.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {config.rank}")
        self.config = config
        self.rank = config.rank
        self.alpha = config.alpha
        self.target_modules = (
            set(config.target_modules) if config.target_modules is not None else None
        )
        if self.target_modules is None:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha} "
                f"(all Linear layers)"
            )
        else:
            logger.info(
                f"LoRA training active with rank={self.rank}, alpha={self.alpha}, "
                f"target_modules={sorted(self.target_modules)}"
            )

    def _make_lora_config(self, cfg: Linear.Config):
        """Create a LoRALinear.Config from a base Linear.Config."""
        assert cfg._owner is not None
        lora_cls = _get_lora_cls(cfg._owner)
        return lora_cls.Config(  # pyrefly: ignore [missing-attribute]
            **{f.name: getattr(cfg, f.name) for f in fields(cfg)},
            rank=self.rank,
            alpha=self.alpha,
        )

    def convert(self, model_config) -> None:
        """Walk the model config tree for Linear modules.

        Target Linear modules get their config replaced with
        ``LoRALinear.Config``.  Non-target Linear modules are wrapped
        with ``FrozenConfig``.
        """
        matched = set()

        # First pass: replace target Linears with LoRA, freeze non-targets
        for fqn, cfg, parent, attr in model_config.traverse(Linear.Config):
            last_segment = fqn.rsplit(".", 1)[-1]
            is_target = (
                self.target_modules is None or last_segment in self.target_modules
            )
            if is_target:
                new_cfg = self._make_lora_config(cfg)
                matched.add(last_segment)
            else:
                new_cfg = FrozenConfig(inner=cfg)

            if isinstance(parent, list):
                parent[attr] = new_cfg
            else:
                setattr(parent, attr, new_cfg)

        unmatched = (self.target_modules or set()) - matched
        if unmatched:
            logger.warning(
                f"LoRA target_modules {sorted(unmatched)} did not match any "
                f"Linear.Config in the model config tree."
            )
