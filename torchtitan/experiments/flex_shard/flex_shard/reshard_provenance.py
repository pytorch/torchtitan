# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager, ExitStack
from contextvars import ContextVar
from typing import Any

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


_FLEX_SHARD_RECOMPUTE_ATTR = "_flex_shard_reshard_after_forward_recompute"
_flex_shard_all_gather_depth: ContextVar[int] = ContextVar(
    "_flex_shard_all_gather_depth",
    default=0,
)


def _iter_tensors(obj: Any) -> Generator[torch.Tensor, None, None]:
    leaves, _ = tree_flatten(obj)
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            yield leaf


def _mark_flex_shard_recompute_tensors(obj: Any) -> None:
    """Mark tensors whose storage should be managed by FlexShard recompute."""
    for tensor in _iter_tensors(obj):
        setattr(tensor, _FLEX_SHARD_RECOMPUTE_ATTR, True)


def _is_flex_shard_recompute_tensor(tensor: torch.Tensor) -> bool:
    return bool(getattr(tensor, _FLEX_SHARD_RECOMPUTE_ATTR, False))


def _has_flex_shard_recompute_tensor(*objs: Any) -> bool:
    return any(
        _is_flex_shard_recompute_tensor(tensor)
        for obj in objs
        for tensor in _iter_tensors(obj)
    )


def _op_returns_alias(func: Any) -> bool:
    schema = getattr(func, "_schema", None)
    if schema is None:
        return False
    return any(ret.alias_info is not None for ret in schema.returns)


def _is_flex_shard_alias_op(func: Any, args: Any, kwargs: Any) -> bool:
    return _op_returns_alias(func) and _has_flex_shard_recompute_tensor(args, kwargs)


def _in_flex_shard_all_gather() -> bool:
    if torch.compiler.is_compiling():
        return False
    return _flex_shard_all_gather_depth.get() > 0


@contextmanager
def _flex_shard_all_gather_region() -> Generator[None, None, None]:
    if torch.compiler.is_compiling():
        yield
        return
    token = _flex_shard_all_gather_depth.set(_flex_shard_all_gather_depth.get() + 1)
    try:
        yield
    finally:
        _flex_shard_all_gather_depth.reset(token)


def _maybe_mark_policy_outputs(ctx: Any, func: Any, args: Any, kwargs: Any) -> bool:
    """Mark FlexShard-derived outputs and return whether they must recompute."""
    if _in_flex_shard_all_gather():
        output = getattr(ctx, "op_output", None)
        _mark_flex_shard_recompute_tensors(output)
        return True

    if _is_flex_shard_alias_op(func, args, kwargs):
        output = getattr(ctx, "op_output", None)
        _mark_flex_shard_recompute_tensors(output)
        return True

    return False


class _FlexShardProvenanceTorchDispatchMode(TorchDispatchMode):
    """Propagate FlexShard provenance through recomputed alias/view ops."""

    @classmethod
    def ignore_compile_internals(cls):
        return True

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        output = func(*args, **kwargs)
        if _in_flex_shard_all_gather() or _is_flex_shard_alias_op(func, args, kwargs):
            _mark_flex_shard_recompute_tensors(output)
        return output


class _FlexShardProvenanceContext:
    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx
        self._stack: ExitStack | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ctx, name)

    def __enter__(self) -> Any:
        # Enter provenance first so SAC's mode is above it. When SAC recomputes
        # an op, provenance still sees the actual execution and can mark aliases.
        stack = ExitStack()
        self._stack = stack
        stack.enter_context(_FlexShardProvenanceTorchDispatchMode())
        return stack.enter_context(self._ctx)

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        if self._stack is None:
            return False
        return self._stack.__exit__(exc_type, exc_value, traceback)


def _with_flex_shard_provenance(ctx: Any) -> _FlexShardProvenanceContext:
    return _FlexShardProvenanceContext(ctx)
