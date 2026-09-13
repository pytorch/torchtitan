# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ordering and application of model transforms."""

import copy
from typing import cast, TYPE_CHECKING

from torchtitan.protocols.module import Module

from .base import ModelConfigTransform

if TYPE_CHECKING:
    from torchtitan.protocols.model import BaseModel
    from torchtitan.trainer import Trainer

__all__ = ["apply_transforms", "transform_model_config_"]


def _ordered(
    transforms: list[ModelConfigTransform],
) -> list[ModelConfigTransform]:
    """Stable-sort transforms by their ``run_after`` declarations."""

    # Not the best performance but simple enough. Given that there are
    # not many transforms, this is acceptable. We can improve it later.
    ordered: list[ModelConfigTransform] = []
    remaining = list(transforms)
    while remaining:
        for i, candidate in enumerate(remaining):
            blockers = [
                other
                for other in remaining
                if other is not candidate
                and isinstance(other, tuple(candidate.run_after) or ())
            ]
            if not blockers:
                ordered.append(remaining.pop(i))
                break
        else:
            cycle = ", ".join(type(t).__qualname__ for t in remaining)
            raise ValueError(f"run_after declarations form a cycle: {cycle}.")
    return ordered


def _reject_conflicts(transforms: list[ModelConfigTransform]) -> None:
    for transform in transforms:
        for other in transforms:
            if other is transform:
                continue
            if isinstance(other, transform.conflicts_with):
                raise ValueError(
                    f"{type(transform).__qualname__} and "
                    f"{type(other).__qualname__} cannot be combined."
                )


def transform_model_config_(
    model: Module.Config, transforms: list[ModelConfigTransform]
) -> Module.Config:
    """Apply every transform to ``model`` and return the rewritten root.

    Rewrites in place, so copy ``model`` first to keep the original. Use this
    where there is no ``Trainer.Config``, such as a bare ``ModelSpec``.
    Validation is the caller's job.
    """
    _reject_conflicts(transforms)
    for transform in _ordered(transforms):
        model = transform.transform(model)
    return model


def apply_transforms(
    config: "Trainer.Config", transforms: list[ModelConfigTransform]
) -> "Trainer.Config":
    """Apply every transform to a copy of ``config`` and return it.

    Set all training options before calling this function. It orders the
    transforms, applies them, and validates the result.
    """
    working = copy.deepcopy(config)
    assert working.model_spec is not None, "model_spec must be set before transforms."
    working.model_spec.model = cast(
        "BaseModel.Config",
        transform_model_config_(working.model_spec.model, transforms),
    )
    working.__post_init__()
    return working
