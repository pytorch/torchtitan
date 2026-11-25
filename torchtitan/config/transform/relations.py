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
from .lm_head_fp32 import LMHeadFP32OutputTransform
from .lora import LoRATransform
from .quantization import (
    MXFP8LinearTransform,
    NVFP4LinearTransform,
    QuantizationTransform,
)
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
    # LoRA replaces each target projection config with an adapter and every
    # other config with a frozen dynamic subclass. Quantization and the fp32
    # lm_head convert only stock projection configs, so they run first.
    (QuantizationTransform, LoRATransform),
    (LMHeadFP32OutputTransform, LoRATransform),
    # Quantization and the fp32 lm_head must target different Linears. Running
    # quantization first lets the lm_head transform reject a quantized lm_head.
    (QuantizationTransform, LMHeadFP32OutputTransform),
)

# Each pair is unordered.
_BUILTIN_CONFLICTS: tuple[ConflictRelation, ...] = (
    # Async kernels invoke fused autograd functions directly instead of the
    # projection's _linear method, which would silently omit LoRA computation.
    (AsyncTensorParallelTransform, LoRATransform),
    # Async TP converts only stock ColumnParallelLinear and RowParallelLinear
    # owners, so it would silently leave quantized projections synchronous, and
    # quantization does not support the async projection classes. Grouped
    # quantization touches only routed experts, which async TP never converts.
    (AsyncTensorParallelTransform, MXFP8LinearTransform),
    (AsyncTensorParallelTransform, NVFP4LinearTransform),
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
