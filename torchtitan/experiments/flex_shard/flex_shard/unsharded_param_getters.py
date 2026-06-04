# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class UnshardedParamSlot:
    """Per-parameter handoff slot for the current unsharded parameter.

    FlexShard replaces managed module parameters with dynamic properties.
    A bucket pre-forward hook writes the full parameter tensor produced by
    bucket unshard into this slot. The dynamic property getter consumes that
    tensor when module code reads ``self.weight``. The post-forward hook clears
    any remaining value.

    The slot also carries per-parameter dtype/device policy for the getter; it
    does not own the unsharded parameter tensor's storage.
    """

    param_fqn: str
    bucket_fqn: str | None
    bucket_unshard_hook_registered: bool = False
    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None
    compute_device: torch.device | None = None
    _current_unsharded_param: torch.Tensor | None = None

    def consume_unsharded_param(self) -> torch.Tensor:
        """Return the hook-provided unsharded param or raise if absent."""
        # Filled from bucket unshard output by the pre-forward hook so parameter
        # reads preserve bucketed collectives.
        current = self._current_unsharded_param
        if current is not None:
            # TODO: Keep this cache valid for the whole forward. Clearing it
            # here breaks legal modules that read the same parameter more than
            # once, and the post-forward hook already owns cleanup.
            self._current_unsharded_param = None
            if self.param_dtype is not None or self.reduce_dtype is not None:
                current = _UnshardedParamMixedPrecisionCast.apply(
                    current,
                    self.param_dtype,
                    self.reduce_dtype,
                )
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

    def _make_flex_shard_param_getter(unsharded_param_slot: UnshardedParamSlot):
        def get_flex_shard_param(self):
            return unsharded_param_slot.consume_unsharded_param()

        return get_flex_shard_param

    param_name_to_property = {
        param_name: property(_make_flex_shard_param_getter(state))
        for param_name, state in unsharded_param_slots.items()
    }
    module_cls = type(
        f"FlexShard{module.__class__.__name__}_{_unsharded_param_getter_class_counter}",
        (module.__class__,),
        param_name_to_property,
    )
    module.__class__ = module_cls
    sys.modules[module_cls.__module__].__dict__[module_cls.__name__] = module_cls


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


class _UnshardedParamMixedPrecisionCast(torch.autograd.Function):
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
