# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

from .sharding_metadata import (
    _DSTORAGE_ATTR,
    _DSTORAGES_ATTR,
    _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR,
    _EAGER_COMM_CONTEXTS_ATTR,
)
from .utils import (
    _is_graph_capture_active,
    _raise_graph_capture_unsupported,
    _raise_missing_eager_batched_unshard,
)

if TYPE_CHECKING:
    from .storage import DStorage


class _MixedPrecisionCast(torch.autograd.Function):
    """Cast with decoupled forward/backward dtype control."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        param_dtype: torch.dtype | None,
        reduce_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        ctx.reduce_dtype = reduce_dtype
        if param_dtype is not None and x.dtype != param_dtype:
            return x.to(param_dtype)
        return x

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        if ctx.reduce_dtype is not None and grad.dtype != ctx.reduce_dtype:
            return grad.to(ctx.reduce_dtype), None, None
        return grad, None, None


@dataclass
class EagerParamAccessState:
    """Mutable state for eager-only FlexShard parameter access."""

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    compute_device: torch.device | None = None
    _pre_gathered: torch.Tensor | None = None
    _unsharded_for_reduce: torch.Tensor | None = None


class FlexShardModule:
    """Mixin added to modules after flex_shard()."""

    @property
    def dstorage(self) -> DStorage:
        """First (or only) DStorage. For multi-bucket, use .dstorages."""
        return getattr(self, _DSTORAGE_ATTR)

    @property
    def dstorages(self) -> list:
        """All DStorage instances (one per bucket)."""
        return getattr(self, _DSTORAGES_ATTR)


def _check_not_already_flex_sharded(module: nn.Module) -> None:
    """Raise if FlexShard was already applied to this module."""
    if getattr(module, _DSTORAGES_ATTR, None) is not None:
        raise ValueError(
            f"Module {type(module).__name__} already has DStorage. "
            "Cannot apply flex_shard twice to the same module."
        )


def _attach_flex_shard_module_state(
    module: nn.Module,
    storages: list[DStorage],
) -> None:
    """Attach FlexShard state and mixin accessors to a module."""
    setattr(module, _DSTORAGES_ATTR, storages)
    setattr(module, _DSTORAGE_ATTR, storages[0] if storages else None)
    setattr(module, _EAGER_COMM_CONTEXTS_ATTR, {})

    cls = type(module)
    if not issubclass(cls, FlexShardModule):
        module.__class__ = type(cls.__name__, (cls, FlexShardModule), {})


_parametrized_module_class_counter = 0


def _register_param_accessors(
    module: nn.Module,
    param_states: dict[str, EagerParamAccessState],
) -> None:
    """Register per-parameter property getters that read eager access state.

    Uses dynamic subclass creation (not nn.utils.parametrize) to avoid
    state_dict key mangling. state_dict reads self._parameters directly,
    bypassing property getters.

    Args:
        module: The leaf module owning the parameters.
        param_states: Maps parameter name to its eager access state.
    """
    global _parametrized_module_class_counter
    _parametrized_module_class_counter += 1

    def _make_flex_shard_param_getter(
        param_name: str,
        param_state: EagerParamAccessState,
    ):
        def get_flex_shard_param(self):
            # In eager mode, _pre_gathered is set by the batched all-gather
            # pre-forward hook. This PR intentionally has no per-parameter
            # all-gather fallback. TODO: later compile support should add a
            # separate graph-safe parametrization or graph path, with tests
            # proving graph capture behavior.
            pre = param_state._pre_gathered
            if pre is not None:
                param_state._pre_gathered = None
                param_dtype = param_state.param_dtype
                reduce_dtype = param_state.reduce_dtype
                if getattr(param_state, _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR, False):
                    if param_dtype is not None or reduce_dtype is not None:
                        pre = _MixedPrecisionCast.apply(pre, param_dtype, reduce_dtype)
                    return pre

                if param_dtype is not None and pre.dtype != param_dtype:
                    pre = pre.to(param_dtype)
                unsharded = pre.detach().requires_grad_(True)
                if (
                    torch.is_grad_enabled()
                    and param_state._unsharded_for_reduce is None
                ):
                    param_state._unsharded_for_reduce = unsharded
                return unsharded
            if _is_graph_capture_active():
                _raise_graph_capture_unsupported()
            _raise_missing_eager_batched_unshard(param_state)

        return get_flex_shard_param

    param_name_to_property = {
        param_name: property(_make_flex_shard_param_getter(param_name, state))
        for param_name, state in param_states.items()
    }
    module_cls = type(
        f"FlexShard{module.__class__.__name__}_{_parametrized_module_class_counter}",
        (module.__class__,),
        param_name_to_property,
    )
    module.__class__ = module_cls
    sys.modules[module_cls.__module__].__dict__[module_cls.__name__] = module_cls


def _register_module_param_accessors(
    module_param_map: dict[nn.Module, dict[str, EagerParamAccessState]],
) -> None:
    """Register property-based parameter accessors grouped by owning module."""
    for module, param_map in module_param_map.items():
        _register_param_accessors(module, param_map)
