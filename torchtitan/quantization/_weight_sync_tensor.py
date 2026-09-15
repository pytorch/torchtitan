# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tensor wrapper that builds format-specific operands, such as those for
quantization, on weight sync, and stores them instead of the original
synced weight, which are released immediately.

These operands remain until the next weight sync, such that we only
need to build them once per weight sync lifecycle.

Lifecycle::

    initial model load       high-precision parameter
                                      |
                                      | install
                                      v
                              _WeightSyncTensor
                                      |
                                      +-- owns format-specific operands
                                      |
    each weight sync         copy_(high-precision tensor)
                                      |
                                      | refill operands in place
                                      v
    each model forward       consume the stored operands
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import torch
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import return_and_correct_aliasing


_WEIGHT_SYNC_VIEW_OPS = {
    torch.ops.aten.alias.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten.detach.default,
    torch.ops.aten.view.default,
}

_WEIGHT_SYNC_FACTORY_OPS = {
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.zeros_like.default,
}


def _operand_names(operands_cls: type) -> tuple[str, ...]:
    if not is_dataclass(operands_cls):
        raise TypeError(
            f"Weight-sync operands must be a dataclass; got {operands_cls.__name__}."
        )
    return tuple(field.name for field in fields(operands_cls))


# TODO: refactor duplicate code with _UnshardedFSDPTensor
class _WeightSyncTensor(torch.Tensor):
    """Storage-free weight whose format operands are refilled after weight sync."""

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, *args: Any, **kwargs: Any):
        del args
        return torch.Tensor._make_wrapper_subclass(
            cls,
            kwargs.get("_logical_size", tensor.size()),
            strides=kwargs.get("_logical_stride", tensor.stride()),
            storage_offset=kwargs.get(
                "_logical_storage_offset", tensor.storage_offset()
            ),
            dtype=kwargs.get("_logical_dtype", tensor.dtype),
            layout=tensor.layout,
            device=kwargs.get("_logical_device", tensor.device),
            pin_memory=tensor.is_pinned(),
            requires_grad=kwargs.get("_logical_requires_grad", tensor.requires_grad),
        )

    def __init__(
        self,
        metadata_source: torch.Tensor,
        *,
        _operands: Any | None = None,
        **logical_metadata: Any,
    ) -> None:
        del logical_metadata
        self._operands = (
            self._build_operands(metadata_source)
            if _operands is None
            else _operands
        )
        for name in _operand_names(type(self._operands)):
            setattr(self, f"_{name}", getattr(self._operands, name))

    def _build_operands(self, logical_tensor: torch.Tensor, out: Any | None = None):
        raise NotImplementedError

    def refill_from_tensor(self, logical_tensor: torch.Tensor) -> None:
        """Refill this weight's existing operands from a synchronized tensor."""
        with torch.no_grad():
            self._operands = self._build_operands(
                logical_tensor,
                out=self._operands,
            )

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"{type(self).__name__}(shape={tuple(self.shape)}, "
            f"dtype={self.dtype}, device={self.device}, "
            f"operands={type(self._operands).__name__})"
        )

    def __tensor_flatten__(self):
        operands_cls = type(self._operands)
        names = [f"_{name}" for name in _operand_names(operands_cls)]
        return names, (operands_cls, self.dtype)

    @classmethod
    def __tensor_unflatten__(
        cls, inner_tensors, metadata, outer_size, outer_stride
    ):
        operands_cls, dtype = metadata
        operand_tensors = [
            inner_tensors[f"_{name}"] for name in _operand_names(operands_cls)
        ]
        return cls(
            operand_tensors[0],
            _operands=operands_cls(*operand_tensors),
            _logical_size=outer_size,
            _logical_stride=outer_stride,
            _logical_dtype=dtype,
        )

    @classmethod
    # pyrefly: ignore [bad-param-name-override]
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        del types
        if func is torch.ops.aten.copy_.default:
            weight_sync_tensor, source = args[:2]
            if not isinstance(weight_sync_tensor, _WeightSyncTensor):
                raise RuntimeError("weight sync copy has an invalid destination")
            if isinstance(source, _WeightSyncTensor):
                if source._operands is not weight_sync_tensor._operands:
                    raise RuntimeError("copy mixed weight sync tensor operands")
                return weight_sync_tensor
            if source.device != weight_sync_tensor.device:
                source = source.to(weight_sync_tensor.device)
            weight_sync_tensor.refill_from_tensor(source)
            return weight_sync_tensor

        template = None

        def unwrap(tensor: _WeightSyncTensor) -> torch.Tensor:
            nonlocal template
            if template is None:
                template = tensor
            elif tensor._operands is not template._operands:
                raise RuntimeError("operation mixed weight sync tensor operands")
            return torch.empty_strided(
                tensor.size(),
                tensor.stride(),
                dtype=tensor.dtype,
                device="meta",
                requires_grad=tensor.requires_grad,
            )

        def wrap_view(tensor: torch.Tensor):
            assert template is not None
            operands = template._operands
            layout_source = getattr(operands, _operand_names(type(operands))[0])
            return type(template)(
                layout_source,
                _operands=operands,
                _logical_size=tensor.size(),
                _logical_stride=tensor.stride(),
                _logical_storage_offset=tensor.storage_offset(),
                _logical_dtype=template.dtype,
                _logical_device=template.device,
                _logical_requires_grad=tensor.requires_grad,
            )

        original_args, original_kwargs = args, kwargs or {}
        args, kwargs = pytree.tree_map_only(
            cls, unwrap, (original_args, original_kwargs)
        )
        assert template is not None
        if func in _WEIGHT_SYNC_FACTORY_OPS:
            kwargs.setdefault("device", template.device)
            return func(*args, **kwargs)
        if func not in _WEIGHT_SYNC_VIEW_OPS:
            raise RuntimeError(
                f"{func} attempted to read a storage-free weight sync tensor"
            )
        wrapped = pytree.tree_map_only(torch.Tensor, wrap_view, func(*args, **kwargs))
        return return_and_correct_aliasing(
            func, original_args, original_kwargs, wrapped
        )

    @property
    def operands(self) -> Any:
        return self._operands
