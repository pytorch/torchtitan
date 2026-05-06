# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from .state import (
    _DSTORAGE_ATTR,
    _DSTORAGES_ATTR,
    _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR,
    _EAGER_BATCHED_HOOK_REGISTERED_ATTR,
    _EAGER_COMM_CONTEXTS_ATTR,
    _REQUIRES_EAGER_BATCHED_UNSHARD_ATTR,
)
from .placements import _MixedPrecisionCast
from .utils import (
    _is_graph_capture_active,
    _raise_graph_capture_unsupported,
    _raise_missing_eager_batched_unshard,
)

if TYPE_CHECKING:
    from .eager_runtime import EagerAllGatherContext
    from .storage import DStorage


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

    @property
    def eager_comm_contexts(self) -> dict[torch.device, EagerAllGatherContext]:
        """Root-owned eager communication contexts keyed by CUDA device."""
        return getattr(self, _EAGER_COMM_CONTEXTS_ATTR)


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


def _register_parametrization(
    module: nn.Module,
    param_parametrizations: dict[str, nn.Module],
) -> None:
    """Register per-parameter property getters that call parametrization forward.

    Uses dynamic subclass creation (not nn.utils.parametrize) to avoid
    state_dict key mangling. state_dict reads self._parameters directly,
    bypassing property getters.

    Args:
        module: The leaf module owning the parameters.
        param_parametrizations: Maps parameter name to its parametrization module.
    """
    global _parametrized_module_class_counter
    _parametrized_module_class_counter += 1

    def _make_flex_shard_param_getter(param_name, parametrization):
        def get_flex_shard_param(self):
            # In eager batched mode, _pre_gathered is set on the
            # parametrization by the batched all-gather pre_forward hook.
            pre = getattr(parametrization, "_pre_gathered", None)
            if pre is not None:
                parametrization._pre_gathered = None
                param_dtype = getattr(parametrization, "param_dtype", None)
                reduce_dtype = getattr(parametrization, "reduce_dtype", None)
                if getattr(parametrization, _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR, False):
                    if param_dtype is not None or reduce_dtype is not None:
                        pre = _MixedPrecisionCast.apply(pre, param_dtype, reduce_dtype)
                    return pre

                if param_dtype is not None and pre.dtype != param_dtype:
                    pre = pre.to(param_dtype)
                unsharded = pre.detach().requires_grad_(True)
                if (
                    torch.is_grad_enabled()
                    and getattr(parametrization, "_unsharded_for_reduce", None) is None
                ):
                    parametrization._unsharded_for_reduce = unsharded
                return unsharded
            if _is_graph_capture_active():
                _raise_graph_capture_unsupported()
            if getattr(
                parametrization, _REQUIRES_EAGER_BATCHED_UNSHARD_ATTR, False
            ) and not getattr(
                parametrization, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, False
            ):
                _raise_missing_eager_batched_unshard(parametrization)
            return parametrization(self._parameters[param_name])

        return get_flex_shard_param

    param_name_to_property = {
        param_name: property(_make_flex_shard_param_getter(param_name, param))
        for param_name, param in param_parametrizations.items()
    }
    module_cls = type(
        f"FlexShard{module.__class__.__name__}_{_parametrized_module_class_counter}",
        (module.__class__,),
        param_name_to_property,
    )
    module.__class__ = module_cls
    sys.modules[module_cls.__module__].__dict__[module_cls.__name__] = module_cls


def _register_module_parametrizations(
    module_param_map: dict[nn.Module, dict[str, nn.Module]],
) -> None:
    """Register property-based parametrizations grouped by owning module."""
    for module, param_map in module_param_map.items():
        _register_parametrization(module, param_map)
