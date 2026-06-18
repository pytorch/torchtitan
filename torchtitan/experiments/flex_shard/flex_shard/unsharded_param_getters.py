# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import inspect
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class _UnshardedParamFrame:
    """Forward-scoped parameter view exposed by a bucket unshard hook."""

    unsharded_param: torch.Tensor
    saved_tensor_handle: Any | None = None
    exposed_param: torch.Tensor | None = None


@dataclass(frozen=True)
class _RafSavedTensorView:
    """A view of a saved RAF full parameter."""

    base_handle: Any
    size: torch.Size
    stride: tuple[int, ...]
    storage_offset_delta: int

    def unpack_raf_saved_tensor(self) -> torch.Tensor:
        base = self.base_handle.unpack_raf_saved_tensor()
        return torch.as_strided(
            base,
            size=self.size,
            stride=self.stride,
            storage_offset=base.storage_offset() + self.storage_offset_delta,
        )


class _RafSavedTensorContext:
    """Per-forward registry for replacing saved full params with handles."""

    def __init__(self) -> None:
        self.tensor_handles: dict[int, Any] = {}

    def register(self, tensor: torch.Tensor, handle: Any) -> None:
        self.tensor_handles[id(tensor)] = handle

    def pack(self, tensor: torch.Tensor) -> Any:
        handle = self.tensor_handles.get(id(tensor))
        if handle is not None:
            return handle

        base = getattr(tensor, "_base", None)
        if base is not None:
            base_handle = self.tensor_handles.get(id(base))
            if base_handle is not None:
                return _RafSavedTensorView(
                    base_handle=base_handle,
                    size=tensor.size(),
                    stride=tensor.stride(),
                    storage_offset_delta=tensor.storage_offset()
                    - base.storage_offset(),
                )

        return tensor

    def unpack(self, packed: Any) -> torch.Tensor:
        unpack = getattr(packed, "unpack_raf_saved_tensor", None)
        if callable(unpack):
            return unpack()
        return packed


_ACTIVE_RAF_SAVED_TENSOR_CONTEXT: ContextVar[
    _RafSavedTensorContext | None
] = ContextVar("_flex_shard_active_raf_saved_tensor_context", default=None)


@contextmanager
def flex_shard_raf_saved_tensors():
    context = _RafSavedTensorContext()
    token = _ACTIVE_RAF_SAVED_TENSOR_CONTEXT.set(context)
    try:
        with torch.autograd.graph.saved_tensors_hooks(context.pack, context.unpack):
            yield
    finally:
        _ACTIVE_RAF_SAVED_TENSOR_CONTEXT.reset(token)


def _register_raf_saved_tensor(tensor: torch.Tensor, handle: Any | None) -> None:
    if handle is None:
        return
    context = _ACTIVE_RAF_SAVED_TENSOR_CONTEXT.get()
    if context is None:
        return
    context.register(tensor, handle)


@dataclass
class UnshardedParamSlot:
    """Per-parameter handoff slot for the current unsharded parameter.

    FlexShard replaces managed module parameters with dynamic properties.
    A bucket pre-forward hook writes the full parameter tensor produced by
    bucket unshard into this slot. The dynamic property getter returns that
    tensor when module code reads ``self.weight``. The post-forward hook clears
    the forward frame.

    The slot also carries per-parameter dtype/device policy for the getter; it
    does not own the unsharded parameter tensor's storage.
    """

    param_fqn: str
    bucket_fqn: str | None
    bucket_unshard_hook_registered: bool = False
    param_dtype: torch.dtype | None = None
    compute_device: torch.device | None = None
    _unsharded_param_stack: list[_UnshardedParamFrame] = field(default_factory=list)

    def push_unsharded_param(
        self,
        unsharded_param: torch.Tensor,
        saved_tensor_handle: Any | None = None,
    ) -> None:
        """Push the hook-provided full parameter for a module forward."""
        self._unsharded_param_stack.append(
            _UnshardedParamFrame(
                unsharded_param,
                saved_tensor_handle=saved_tensor_handle,
            )
        )

    def pop_unsharded_param(self) -> None:
        """Pop the current module forward's full parameter frame."""
        if not self._unsharded_param_stack:
            raise AssertionError(
                "Attempted to clear a FlexShard unsharded parameter slot "
                f"for {self.param_fqn!r}, but no slot frame is active."
            )
        self._unsharded_param_stack.pop()

    def apply_unsharded_param_policy(
        self, unsharded_param: torch.Tensor
    ) -> torch.Tensor:
        current = unsharded_param
        if self.param_dtype is not None and current.dtype != self.param_dtype:
            current = _UnshardedParamCast.apply(current, self.param_dtype)
        return current

    def get_unsharded_param(self) -> torch.Tensor:
        """Return the hook-provided unsharded param or raise if absent."""
        # Filled from bucket unshard output by the pre-forward hook so parameter
        # reads preserve bucketed collectives. The value is intentionally stable
        # for the whole hooked forward because legal modules may read the same
        # parameter attribute more than once.
        if self._unsharded_param_stack:
            frame = self._unsharded_param_stack[-1]
            if frame.exposed_param is not None:
                return frame.exposed_param

            current = self.apply_unsharded_param_policy(frame.unsharded_param)
            _register_raf_saved_tensor(current, frame.saved_tensor_handle)
            frame.exposed_param = current
            return current

        _raise_missing_eager_bucket_unshard(self)


_unsharded_param_getter_class_counter = 0


def _install_unsharded_param_getters(
    module_param_slots: dict[nn.Module, dict[str, UnshardedParamSlot]],
) -> None:
    """Install unsharded parameter getters grouped by owning module."""
    for module, param_slots in module_param_slots.items():
        _install_module_unsharded_param_getters(module, param_slots)


def _install_module_unsharded_param_getters(
    module: nn.Module,
    unsharded_param_slots: dict[str, UnshardedParamSlot],
) -> None:
    """Install unsharded parameter getters on one leaf module.

    The generated properties intercept module forward reads like ``self.weight``.
    ``state_dict()`` still reads ``self._parameters`` directly, so checkpoint keys
    and stored sharded tensors stay unchanged.

    Args:
        module: The leaf module owning the parameters.
        unsharded_param_slots: Maps parameter name to its unsharded slot.
    """
    global _unsharded_param_getter_class_counter
    _unsharded_param_getter_class_counter += 1

    def _make_flex_shard_param_getter(
        param_name: str,
        unsharded_param_slot: UnshardedParamSlot,
    ):
        def get_flex_shard_param(self):
            try:
                return unsharded_param_slot.get_unsharded_param()
            except RuntimeError:
                if _is_state_dict_introspection():
                    return self._parameters[param_name]
                raise

        return get_flex_shard_param

    param_name_to_property = {
        param_name: property(_make_flex_shard_param_getter(param_name, state))
        for param_name, state in unsharded_param_slots.items()
    }
    module_cls = type(
        f"FlexShard{module.__class__.__name__}_{_unsharded_param_getter_class_counter}",
        (module.__class__,),
        param_name_to_property,
    )
    module.__class__ = module_cls
    sys.modules[module_cls.__module__].__dict__[module_cls.__name__] = module_cls


def _is_state_dict_introspection() -> bool:
    """Return whether a missing unsharded param read is state-dict metadata lookup."""
    for frame_info in inspect.stack(context=0)[2:]:
        module_name = frame_info.frame.f_globals.get("__name__", "")
        if module_name == "torch.distributed.checkpoint.state_dict":
            return True
    return False


def _raise_missing_eager_bucket_unshard(slot: UnshardedParamSlot) -> None:
    bucket_msg = f" in bucket {slot.bucket_fqn!r}" if slot.bucket_fqn else ""
    hook_msg = (
        " The bucket hook was registered but did not run before parameter access."
        if slot.bucket_unshard_hook_registered
        else " No bucket hook was registered for this parameter."
    )
    raise RuntimeError(
        "FlexShard eager mode requires full parameter data from a "
        f"bucket unshard hook for parameter {slot.param_fqn!r}{bucket_msg}."
        f"{hook_msg} This usually means the parameter was accessed outside "
        "the hooked module forward, or the BucketSpec boundary does not match "
        "the module hook/checkpoint execution unit. Split the bucket to match "
        "forward module boundaries."
    )


class _UnshardedParamCast(torch.autograd.Function):
    """Forward param cast with identity gradient dtype semantics."""

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        param_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        if param_dtype is not None and x.dtype != param_dtype:
            return x.to(param_dtype)
        return x

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad, None
