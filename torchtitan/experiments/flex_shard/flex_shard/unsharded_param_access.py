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

from .sharded_param_metadata import is_flex_shard_param

if TYPE_CHECKING:
    from .bucket_storage import DStorage


# Module attribute names for storing DStorage
_DSTORAGE_ATTR = "_dstorage"
_DSTORAGES_ATTR = "_dstorages"

_EAGER_BUCKET_UNSHARD_HOOK_REGISTERED_ATTR = (
    "_flex_shard_eager_bucket_unshard_hook_registered"
)
_EAGER_COMM_CONTEXTS_ATTR = "_flex_shard_eager_comm_contexts"
_PARAM_FQN_ATTR = "_flex_shard_param_fqn"
_BUCKET_FQN_ATTR = "_flex_shard_bucket_fqn"


def _is_graph_capture_active() -> bool:
    """Return whether unsupported graph capture is active."""
    if torch.compiler.is_compiling():
        return True
    try:
        return torch._guards.TracingContext.try_get() is not None
    except AttributeError:
        return False


def _raise_graph_capture_unsupported() -> None:
    raise ValueError(
        "FlexShard currently supports eager execution only; torch.compile and "
        "graph capture are not supported yet."
    )


def _raise_missing_eager_bucket_unshard(param_state: Any) -> None:
    param_fqn = getattr(param_state, _PARAM_FQN_ATTR, "<unknown>")
    bucket_fqn = getattr(param_state, _BUCKET_FQN_ATTR, None)
    hook_registered = getattr(
        param_state, _EAGER_BUCKET_UNSHARD_HOOK_REGISTERED_ATTR, False
    )
    bucket_msg = f" in bucket {bucket_fqn!r}" if bucket_fqn else ""
    hook_msg = (
        " The bucket hook was registered but did not run before parameter access."
        if hook_registered
        else " No bucket hook was registered for this parameter."
    )
    raise RuntimeError(
        "FlexShard eager mode requires full parameter data from a "
        f"bucket unshard hook for parameter {param_fqn!r}{bucket_msg}."
        f"{hook_msg} This usually means the parameter was accessed outside "
        "the hooked module forward, or the BucketSpec boundary does not match "
        "the module hook/checkpoint execution unit. Split the bucket to match "
        "forward module boundaries."
    )


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
class ParamModuleInfo:
    """Owning module and local parameter name for a managed parameter."""

    module: nn.Module
    param_name: str

    @classmethod
    def resolve(cls, root_module: nn.Module, fqn: str) -> ParamModuleInfo:
        """Resolve a parameter FQN from root, unwrapping checkpoint wrappers."""
        parts = fqn.split(".")
        leaf_module = root_module
        for part in parts[:-1]:
            child = getattr(leaf_module, part, None)
            if child is None:
                wrapped = getattr(leaf_module, "_checkpoint_wrapped_module", None)
                if wrapped is not None:
                    leaf_module = getattr(wrapped, part)
                else:
                    leaf_module = getattr(leaf_module, part)
            else:
                leaf_module = child
        if hasattr(leaf_module, "_checkpoint_wrapped_module"):
            leaf_module = leaf_module._checkpoint_wrapped_module
        return cls(leaf_module, parts[-1])


@dataclass
class ParamAccessorState:
    """Mutable per-parameter state used by FlexShard's property getter.

    Bucket pre-forward hooks write the current full parameter tensor into this
    state. The dynamically installed parameter property reads it when module
    code accesses ``self.weight``. This lets eager module code see full params
    while persistent module storage remains sharded.
    """

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    compute_device: torch.device | None = None
    _pre_unsharded: torch.Tensor | None = None

    def consume_pre_unsharded_param(self) -> torch.Tensor:
        """Return the hook-provided full param or raise for unsupported access."""
        # _pre_unsharded is set by the bucket unshard pre-forward hook so
        # parameter reads preserve bucketed collectives.
        pre = self._pre_unsharded
        if pre is not None:
            # TODO: Keep this cache valid for the whole forward. Clearing it
            # here breaks legal modules that read the same parameter more than
            # once, and the post-forward hook already owns cleanup.
            self._pre_unsharded = None
            if self.param_dtype is not None or self.reduce_dtype is not None:
                pre = _MixedPrecisionCast.apply(
                    pre,
                    self.param_dtype,
                    self.reduce_dtype,
                )
            return pre

        if _is_graph_capture_active():
            _raise_graph_capture_unsupported()
        _raise_missing_eager_bucket_unshard(self)


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
    """Raise if applying FlexShard would create nested ownership."""
    if getattr(module, _DSTORAGES_ATTR, None) is not None:
        raise ValueError(
            f"Module {type(module).__name__} already has DStorage. "
            "Cannot apply flex_shard twice to the same module."
        )
    for child_fqn, child in module.named_modules():
        if child_fqn and getattr(child, _DSTORAGES_ATTR, None) is not None:
            raise ValueError(
                "Nested flex_shard wrapping is not supported. "
                f"Child module {child_fqn!r} is already FlexSharded. "
                "Apply flex_shard once at the root module and express bucket "
                "boundaries with BucketSpec FQN patterns."
            )
    for param_fqn, param in module.named_parameters(remove_duplicate=False):
        if is_flex_shard_param(param):
            raise ValueError(
                "Nested flex_shard wrapping is not supported. "
                f"Parameter {param_fqn!r} is already managed by FlexShard. "
                "Apply flex_shard once at the root module and express bucket "
                "boundaries with BucketSpec FQN patterns."
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
    param_states: dict[str, ParamAccessorState],
) -> None:
    """Register per-parameter property getters that read accessor state.

    Uses dynamic subclass creation (not nn.utils.parametrize) to avoid
    state_dict key mangling. state_dict reads self._parameters directly,
    bypassing property getters.

    Args:
        module: The leaf module owning the parameters.
        param_states: Maps parameter name to its accessor state.
    """
    global _parametrized_module_class_counter
    _parametrized_module_class_counter += 1

    def _make_flex_shard_param_getter(param_state: ParamAccessorState):
        def get_flex_shard_param(self):
            return param_state.consume_pre_unsharded_param()

        return get_flex_shard_param

    param_name_to_property = {
        param_name: property(_make_flex_shard_param_getter(state))
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
    module_param_map: dict[nn.Module, dict[str, ParamAccessorState]],
) -> None:
    """Register property-based parameter accessors grouped by owning module."""
    for module, param_map in module_param_map.items():
        _register_param_accessors(module, param_map)
