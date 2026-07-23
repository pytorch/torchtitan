# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module-level profiler instrumentation."""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable, Iterable
from typing import Any, cast

import torch
import torch.autograd.profiler as autograd_profiler
import torch.nn as nn

from torchtitan.tools.logging import logger


_WRAPPED_ATTR = "_torchtitan_module_profiler_wrapped"
_LABEL_ATTR = "_torchtitan_module_profiler_label"
_KIND_ATTR = "_torchtitan_module_profiler_kind"

MarkKernelsFactory = Callable[[dict[str, Any]], contextlib.AbstractContextManager]

_MARK_KERNELS_UNSET = object()
_mark_kernels_factory: MarkKernelsFactory | object | None = _MARK_KERNELS_UNSET
_mark_kernels_warned = False


def apply_module_profiler(
    model_parts: Iterable[nn.Module],
) -> int:
    """Wrap coarse model modules with profiler annotations.

    The default target policy is intentionally coarse:
    ``layers.<N>`` blocks, attention modules, dense feed-forward modules, and
    MoE modules. Wrapped modules emit ``record_function`` contexts while the
    PyTorch profiler is active and best-effort CUDA graph kernel annotations.

    Returns:
        Number of modules newly wrapped.
    """
    num_wrapped = 0
    for model_part_idx, model_part in enumerate(model_parts):
        for module_fqn, module in model_part.named_modules():
            if module_fqn == "":
                continue
            kind = coarse_module_kind(module_fqn)
            if kind is None:
                continue
            if wrap_module_forward(
                module,
                module_fqn=module_fqn,
                module_kind=kind,
                model_part_idx=model_part_idx,
            ):
                num_wrapped += 1

    logger.info(f"Applied module profiler instrumentation to {num_wrapped} module(s)")
    return num_wrapped


def coarse_module_kind(module_fqn: str) -> str | None:
    parts = module_fqn.split(".")
    name = parts[-1]

    if len(parts) >= 2 and parts[-2] == "layers" and name.isdigit():
        return "block"
    if name in {"attention", "attn"} or name.endswith("_attn"):
        return "attention"
    if name == "feed_forward":
        return "ffn"
    if name == "moe":
        return "moe"
    return None


def wrap_module_forward(
    module: nn.Module,
    *,
    module_fqn: str,
    module_kind: str,
    model_part_idx: int,
) -> bool:
    if getattr(module, _WRAPPED_ATTR, False):
        return False

    original_forward = module.forward
    label = f"{module_kind}::{module_fqn}"
    annotation: dict[str, Any] = {
        "name": label,
        "module_fqn": module_fqn,
        "module_kind": module_kind,
        "model_part_idx": model_part_idx,
    }

    @functools.wraps(original_forward)
    def wrapped_forward(*args, **kwargs):
        record_active = is_profiler_active()

        with contextlib.ExitStack() as stack:
            if record_active:
                stack.enter_context(torch.profiler.record_function(label))
            stack.enter_context(mark_kernels_context(annotation))
            return original_forward(*args, **kwargs)

    module.forward = wrapped_forward
    setattr(module, _WRAPPED_ATTR, True)
    setattr(module, _LABEL_ATTR, label)
    setattr(module, _KIND_ATTR, module_kind)
    return True


def is_profiler_active() -> bool:
    return bool(getattr(autograd_profiler, "_is_profiler_enabled", False))


def get_mark_kernels() -> MarkKernelsFactory | None:
    global _mark_kernels_factory
    if _mark_kernels_factory is not _MARK_KERNELS_UNSET:
        return cast(MarkKernelsFactory | None, _mark_kernels_factory)

    try:
        from torch.cuda._graph_annotations import mark_kernels
    except (AttributeError, ImportError) as exc:
        warn_mark_kernels_unavailable(exc)
        _mark_kernels_factory = None
    else:
        _mark_kernels_factory = mark_kernels
    return cast(MarkKernelsFactory | None, _mark_kernels_factory)


def warn_mark_kernels_unavailable(exc: Exception) -> None:
    global _mark_kernels_warned
    if _mark_kernels_warned:
        return
    _mark_kernels_warned = True
    logger.warning(
        "Module profiler was enabled, but CUDA graph kernel annotations are "
        "unavailable because importing torch.cuda._graph_annotations.mark_kernels "
        "failed: "
        f"{type(exc).__name__}: {exc}"
    )


def mark_kernels_context(
    annotation: dict[str, Any],
) -> contextlib.AbstractContextManager:
    mark_kernels = get_mark_kernels()
    if mark_kernels is None:
        return contextlib.nullcontext()
    return mark_kernels(annotation)
