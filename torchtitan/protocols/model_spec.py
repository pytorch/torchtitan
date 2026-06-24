# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

if TYPE_CHECKING:
    from torchtitan.config import Configurable

# Type aliases for ModelSpec callables
ParallelizeFunction: TypeAlias = Callable[..., nn.Module]
PipeliningFunction: TypeAlias = Callable[
    ..., tuple[_PipelineSchedule, list[nn.Module], bool, bool]
]
FragmentFunction: TypeAlias = Callable[..., list[nn.Module]]
PostOptimizerBuildFn: TypeAlias = Callable[..., None]


# TODO: deprecate ModelSpec, move fields to model config or trainer config
@dataclass
class ModelSpec:
    """Per-model bundle. Contains already-selected arch config + callables."""

    name: str
    flavor: str
    model: BaseModel.Config
    # NOTE: Callable fields use bare ``Callable`` instead of the parameterised
    # TypeAliases (e.g. ``ParallelizeFunction``) because tyro's type-parameter
    # resolver does not handle ``Callable[..., X]`` (Ellipsis as param spec).
    # The detailed TypeAliases above are still available for use in function
    # signatures elsewhere in the codebase.
    parallelize_fn: Callable
    pipelining_fn: Callable | None
    post_optimizer_build_fn: Callable | None
    state_dict_adapter: type[BaseStateDictAdapter] | None

    def traverse(
        self, config_cls: type, *, _prefix: str = ""
    ) -> Iterator[tuple[str, "Configurable.Config", object, str | int]]:
        """Expose the nested model config to ``Configurable.Config.traverse``.

        ``ModelSpec`` is a plain dataclass, not a ``Configurable.Config``, so a
        traversal of ``Trainer.Config`` would otherwise stop here. Implementing
        ``traverse`` lets the override mechanism reach the model config and its
        components. Only ``self.model`` is exposed; the callable fields
        (``parallelize_fn`` etc.) are intentionally not traversable.
        """
        model_fqn = f"{_prefix}.model" if _prefix else "model"
        if isinstance(self.model, config_cls):
            # The model config itself matches — yield it; mirror the generic
            # traverse, which does not descend into a matched node.
            yield model_fqn, self.model, self, "model"
        else:
            yield from self.model.traverse(config_cls, _prefix=model_fqn)
