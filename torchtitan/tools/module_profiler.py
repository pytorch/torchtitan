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

import torch.autograd.profiler as autograd_profiler
import torch.nn as nn
from torch._C import _profiler as torch_profiler_c

from torchtitan.tools.logging import logger


_WRAPPED_ATTR = "_torchtitan_module_profiler_wrapped"

MarkKernelsFactory = Callable[[dict[str, Any]], contextlib.AbstractContextManager]


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

    label = f"{module_kind}::{module_fqn}"
    annotation: dict[str, Any] = {
        "name": label,
        "module_fqn": module_fqn,
        "module_kind": module_kind,
        "model_part_idx": model_part_idx,
    }
    stacks: list[contextlib.ExitStack] = []

    def pre_hook(
        _module: nn.Module, _args: tuple[Any, ...], _kwargs: dict[str, Any]
    ) -> None:
        stack = contextlib.ExitStack()
        try:
            record_active = is_profiler_active()
            if record_active:
                stack.enter_context(record_function_context(label))
            stack.enter_context(mark_kernels_context(annotation))
        except Exception:
            stack.close()
            raise

        stacks.append(stack)

    def post_hook(
        _module: nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        _output: Any,
    ) -> None:
        if stacks:
            stacks.pop().close()

    module.register_forward_pre_hook(pre_hook, prepend=True, with_kwargs=True)
    module.register_forward_hook(post_hook, with_kwargs=True, always_call=True)
    setattr(module, _WRAPPED_ATTR, True)
    return True


def is_profiler_active() -> bool:
    return bool(getattr(autograd_profiler, "_is_profiler_enabled", False))


def record_function_context(label: str) -> contextlib.AbstractContextManager:
    # The public torch.profiler.record_function enters and exits through
    # dispatcher ops. Selective activation checkpointing installs a
    # TorchDispatchMode, so those profiler ops also show up as PythonDispatchMode
    # events in raw traces. Those PythonDispatchMode events can start before a
    # module range exits and finish after it, which makes Perfetto display the
    # module ranges as crossing siblings instead of cleanly nested scopes.
    #
    # _RecordFunctionFast opens the same user annotation directly in the C++
    # profiler path without dispatching an op, so the raw trace keeps the module
    # ranges properly nested under TorchDispatchMode.
    return torch_profiler_c._RecordFunctionFast(
        label,
        keyword_values={"scope": "user_scope"},
    )


@functools.cache
def get_mark_kernels() -> MarkKernelsFactory | None:
    try:
        from torch.cuda._graph_annotations import mark_kernels
    except (AttributeError, ImportError) as exc:
        logger.warning(
            "Module profiler was enabled, but CUDA graph kernel annotations are "
            "unavailable because importing "
            "torch.cuda._graph_annotations.mark_kernels failed: "
            f"{type(exc).__name__}: {exc}"
        )
        return None
    return cast(MarkKernelsFactory, mark_kernels)


def mark_kernels_context(
    annotation: dict[str, Any],
) -> contextlib.AbstractContextManager:
    mark_kernels = get_mark_kernels()
    if mark_kernels is None:
        return contextlib.nullcontext()
    return mark_kernels(annotation)
