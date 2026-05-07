# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

# Type aliases for ModelSpec callables
ParallelizeFunction: TypeAlias = Callable[..., nn.Module]
PipeliningFunction: TypeAlias = Callable[
    ..., tuple[_PipelineSchedule, list[nn.Module], bool, bool]
]
FragmentFunction: TypeAlias = Callable[..., list[nn.Module]]
PostOptimizerBuildFn: TypeAlias = Callable[..., None]


def validate_converter_order(converters: list) -> None:
    """Validate that quantization/QAT converters precede LoRA.

    Raises ``ValueError`` if a quantization converter appears after a LoRA
    converter in the list.
    """
    from torchtitan.components.lora import LoRAConverter
    from torchtitan.components.quantization import QuantizationConverter

    _BEFORE_LORA = (QuantizationConverter,)

    seen_lora = False
    for converter in converters:
        if isinstance(converter, LoRAConverter):
            seen_lora = True
        elif seen_lora and isinstance(converter, _BEFORE_LORA):
            raise ValueError(
                f"{type(converter).__name__} must be applied before "
                f"LoRAConverter. Reorder the converters list."
            )


@dataclass
class ModelSpec:
    """Per-model bundle. Contains already-selected arch config + callables."""

    name: str
    flavor: str
    model: BaseModel.Config
    # TODO: improve the serializability of ModelSpec by refactoring the following
    #       fields, e.g. by having their own classes, or hard-coding into trainer
    # NOTE: Callable fields use bare ``Callable`` instead of the parameterised
    # TypeAliases (e.g. ``ParallelizeFunction``) because tyro's type-parameter
    # resolver does not handle ``Callable[..., X]`` (Ellipsis as param spec).
    # The detailed TypeAliases above are still available for use in function
    # signatures elsewhere in the codebase.
    parallelize_fn: Callable
    pipelining_fn: Callable | None
    post_optimizer_build_fn: Callable | None
    state_dict_adapter: type[BaseStateDictAdapter] | None
    converters: list[Any] = field(default_factory=list)
