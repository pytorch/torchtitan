# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Base class and helpers for model transforms."""

from abc import ABC, abstractmethod
from dataclasses import fields
from typing import ClassVar

from torchtitan.protocols.module import Module

__all__ = ["ModelTransform", "retype_node"]


class ModelTransform(ABC):
    """A feature that rewrites a completed model config tree.

    ``run_after`` declares ordering. ``conflicts_with`` declares incompatible
    transforms. Validation belongs in ``Trainer.Config.__post_init__``.
    """

    run_after: ClassVar[tuple[type["ModelTransform"], ...]] = ()
    conflicts_with: ClassVar[tuple[type["ModelTransform"], ...]] = ()

    @abstractmethod
    def transform(self, model: Module.Config) -> Module.Config:
        """Rewrite ``model`` and return its root.

        Rewrite nodes in place. Return a different config to replace the root
        itself, as a transform that wraps the whole model does.
        """


def retype_node(
    existing: Module.Config,
    replacement: type[Module],
    **overrides: object,
) -> Module.Config:
    """Build ``replacement``'s config from ``existing``, keeping its fields.

    Overrides set fields defined by the replacement config. Requiring
    inheritance preserves wrappers added by earlier transforms.
    """
    if not issubclass(replacement.Config, type(existing)):
        raise ValueError(
            f"{replacement.__qualname__}.Config must inherit "
            f"{type(existing).__qualname__}."
        )
    values = {f.name: getattr(existing, f.name) for f in fields(existing)}
    values.update(overrides)
    return replacement.Config(**values)
