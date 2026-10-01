# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base class and helpers for model transforms."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import ClassVar, TYPE_CHECKING

from torchtitan.protocols.module import Module

if TYPE_CHECKING:
    from torchtitan.config.configs import TrainingConfig
    from torchtitan.config.parallelism import ParallelismConfig

__all__ = [
    "ModelConfigTransform",
    "ModelConfigTransformContext",
    "convert_config_type",
]


@dataclass(frozen=True, kw_only=True)
class ModelConfigTransformContext:
    """Training settings available while rewriting a model config."""

    training: "TrainingConfig"
    parallelism: "ParallelismConfig"


class ModelConfigTransform(ABC):
    """A feature that rewrites a completed model config tree.

    ``run_after`` declares ordering. ``conflicts_with`` declares incompatible
    transforms. Validation belongs in ``Trainer.Config.__post_init__``.
    """

    run_after: ClassVar[tuple[type["ModelConfigTransform"], ...]] = ()
    conflicts_with: ClassVar[tuple[type["ModelConfigTransform"], ...]] = ()

    @abstractmethod
    def transform(
        self,
        model: Module.Config,
        *,
        context: ModelConfigTransformContext | None = None,
    ) -> Module.Config:
        """Rewrite ``model`` and return its root.

        Rewrite configs in place. Return a different config to replace the root
        itself, as a transform that wraps the whole model does.
        """


def convert_config_type(
    existing: Module.Config, replacement: type[Module]
) -> Module.Config:
    """Build ``replacement``'s config from ``existing``, keeping its fields.

    Requiring inheritance preserves wrappers added by earlier transforms.
    """
    if not issubclass(replacement.Config, type(existing)):
        raise ValueError(
            f"{replacement.__qualname__}.Config must inherit "
            f"{type(existing).__qualname__}."
        )
    return replacement.Config(
        **{f.name: getattr(existing, f.name) for f in fields(existing)}
    )
