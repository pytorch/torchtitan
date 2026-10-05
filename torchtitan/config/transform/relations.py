# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Built-in model transform ordering and conflict relations."""

from typing import TypeAlias

from .async_tensor_parallel import AsyncTensorParallelTransform
from .base import ModelConfigTransform
from .context_parallel import ContextParallelTransform
from .dist_moe import DistMoeTransform
from .lora import LoRATransform
from .token_dispatcher import TokenDispatcherTransform

TransformType: TypeAlias = type[ModelConfigTransform]
PrecedenceRelation: TypeAlias = tuple[TransformType, TransformType]
ConflictRelation: TypeAlias = tuple[TransformType, TransformType]

__all__ = ["TransformRelations"]


# Each pair is (before, after).
_BUILTIN_PRECEDES: tuple[PrecedenceRelation, ...] = (
    # LoRA freezes configs by creating dynamic subclasses. Context parallelism
    # must replace attention configs first because convert_config_type requires
    # its replacement config to inherit the current config type.
    (ContextParallelTransform, LoRATransform),
    # Dist-MoE must create DistMoeRoutedExperts.Config before LoRA can adapt it.
    (DistMoeTransform, LoRATransform),
)

# Each pair is unordered.
_BUILTIN_CONFLICTS: tuple[ConflictRelation, ...] = (
    # Async kernels invoke fused autograd functions directly instead of the
    # projection's _linear method, which would silently omit LoRA computation.
    # TODO: Add quantization transforms when they migrate from
    # ModelConfigConverter to ModelConfigTransform.
    (AsyncTensorParallelTransform, LoRATransform),
    # LoRA freezes every non-target config. Applying it more than once would
    # make freezing and adapter configuration depend on transform order.
    (LoRATransform, LoRATransform),
    # Dist-MoE owns token dispatch and combine instead of calling the selected
    # token dispatcher's runtime implementation.
    (DistMoeTransform, TokenDispatcherTransform),
)


class TransformRelations:
    """Mutable relations initialized with TorchTitan's built-in relations."""

    def __init__(self) -> None:
        self._precedes = _BUILTIN_PRECEDES
        self._conflicts = _BUILTIN_CONFLICTS

    @property
    def precedes(self) -> tuple[PrecedenceRelation, ...]:
        return self._precedes

    @property
    def conflicts(self) -> tuple[ConflictRelation, ...]:
        return self._conflicts

    def add_precedence(
        self,
        *,
        before: TransformType,
        after: TransformType,
    ) -> None:
        """Require ``before`` to run before ``after``."""
        edge = (before, after)
        if edge not in self._precedes:
            self._precedes = (*self._precedes, edge)

    def add_conflict(
        self,
        left: TransformType,
        right: TransformType,
    ) -> None:
        """Reject using ``left`` and ``right`` together."""
        edge = (left, right)
        if edge not in self._conflicts and edge[::-1] not in self._conflicts:
            self._conflicts = (*self._conflicts, edge)
