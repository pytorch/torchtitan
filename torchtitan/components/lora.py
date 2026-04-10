# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from torchtitan.config import Configurable
from torchtitan.models.common.linear import Linear
from torchtitan.tools.logging import logger

# Cache for dynamically created LoRA classes
_lora_class_cache: dict[type, type] = {}


def apply_lora(linear: nn.Linear, rank: int, alpha: float) -> nn.Linear:
    """Apply LoRA adapters to a Linear module via dynamic class inheritance.

    Mutates ``linear`` in-place by swapping its ``__class__`` to a dynamically
    created LoRA subclass, then attaches ``lora_a`` and ``lora_b`` child modules.
    Returns the same object for convenience.
    """
    parent_cls = type(linear)
    if not issubclass(parent_cls, nn.Linear):
        raise ValueError(f"apply_lora expects an nn.Linear subclass, got {parent_cls}")

    if parent_cls not in _lora_class_cache:

        class LoRALinear(parent_cls):  # type: ignore[valid-type, misc]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("LoRALinear should not be instantiated directly.")

            @classmethod
            def from_linear(
                cls, linear: nn.Linear, rank: int, alpha: float
            ) -> "LoRALinear":
                linear.__class__ = cls
                linear._init_lora(rank, alpha)  # type: ignore[attr-defined]
                return linear  # type: ignore[return-value]

            def _init_lora(self, rank: int, alpha: float) -> None:
                self._lora_scaling = alpha / rank
                device = self.weight.device
                dtype = self.weight.dtype
                self.lora_a = (
                    Linear.Config(
                        in_features=self.in_features,
                        out_features=rank,
                        bias=False,
                    )
                    .build()
                    .to(device=device, dtype=dtype)
                )
                self.lora_b = (
                    Linear.Config(
                        in_features=rank,
                        out_features=self.out_features,
                        bias=False,
                    )
                    .build()
                    .to(device=device, dtype=dtype)
                )

            def init_weights(self, **kwargs) -> None:
                super().init_weights(**kwargs)  # pyrefly: ignore [not-callable]
                nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_b.weight)

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                base_out = super().forward(input)
                lora_out = self.lora_b(self.lora_a(input))
                return base_out + self._lora_scaling * lora_out

        LoRALinear.__name__ = f"LoRA{parent_cls.__name__}"
        LoRALinear.__qualname__ = f"LoRA{parent_cls.__name__}"
        _lora_class_cache[parent_cls] = LoRALinear

    # pyrefly: ignore [missing-attribute]
    return _lora_class_cache[parent_cls].from_linear(linear, rank, alpha)


class LoRAConverter(Configurable):
    """Apply LoRA adapters to Linear layers in a model.

    When ``target_modules`` is None (default), every ``nn.Linear`` receives a
    LoRA adapter.  When specified, only modules whose attribute name matches one
    of the entries are converted (e.g. ``["wq", "wv"]`` targets the query and
    value projections).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        rank: int = 8
        """Rank of the LoRA matrices (lora_a: in_features x rank, lora_b: rank x out_features)."""

        alpha: float = 16.0
        """Scaling factor. Output is scaled by alpha/rank."""

        target_modules: list[str] | None = None
        """Attribute names of Linear modules to apply LoRA to (e.g. ["wq", "wv"]).
        Matches on the direct attribute name (last segment of the FQN).
        None means all nn.Linear layers. An empty list means no layers."""

    def __init__(self, config: Config, **kwargs):
        if config.rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {config.rank}")
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

    def convert(self, model: nn.Module) -> None:
        model.requires_grad_(False)
        self._replace_linears_with_lora(model)

    def _replace_linears_with_lora(self, module: nn.Module) -> None:
        matched = set()
        visited: set[int] = set()
        for _, parent in list(module.named_modules()):
            for attr_name, child in list(parent.named_children()):
                if id(child) in visited or not isinstance(child, nn.Linear):
                    continue
                visited.add(id(child))
                if (
                    self.target_modules is not None
                    and attr_name not in self.target_modules
                ):
                    continue
                apply_lora(child, self.rank, self.alpha)
                matched.add(attr_name)
        unmatched = (self.target_modules or set()) - matched
        if unmatched:
            logger.warning(
                f"LoRA target_modules {sorted(unmatched)} did not match any "
                f"nn.Linear in the model. Check module attribute names."
            )

    def post_optimizer_hook(self, model: nn.Module | list[nn.Module]) -> None:
        pass
