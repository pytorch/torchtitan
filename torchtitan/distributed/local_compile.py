# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Function-scoped torch.compile registration and configuration."""

import functools
from collections.abc import Callable
from typing import Any

import torch

from torchtitan.config.configs import LocalCompileConfig
from torchtitan.distributed.utils import is_in_batch_invariant_mode


_LOCAL_COMPILE_CALLBACKS: dict[
    str, list[Callable[[LocalCompileConfig | None], None]]
] = {}


def local_compile(
    name: str,
    *,
    batch_invariant: bool,
    **compile_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function that can be compiled independently.

    Args:
        name: Name used to enable the function in ``LocalCompileConfig.regions``.
        batch_invariant: Whether the compiled function preserves batch invariance.
        **compile_kwargs: ``torch.compile`` arguments such as ``backend``,
            ``dynamic``, and ``options``.
    """

    def decorate(reference: Callable[..., Any]) -> Callable[..., Any]:
        fn = reference

        def bind_local_compile(
            compile_config: LocalCompileConfig | None,
        ) -> None:
            nonlocal fn
            enabled = compile_config is not None and name in compile_config.regions
            batch_invariant_mode = is_in_batch_invariant_mode()
            if enabled and batch_invariant_mode and not batch_invariant:
                raise ValueError(
                    f"Local compile region {name!r} does not support "
                    "batch-invariant mode; remove it from compile.regions."
                )
            if enabled:
                fn = torch.compile(reference, fullgraph=True, **compile_kwargs)
            else:
                fn = reference

        @functools.wraps(reference)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        _LOCAL_COMPILE_CALLBACKS.setdefault(name, []).append(bind_local_compile)
        return wrapped

    return decorate


def apply_local_compile(
    compile_config: LocalCompileConfig | None,
) -> None:
    """Bind registered functions to eager or torch.compile implementations."""
    if compile_config is not None:
        unknown = [
            name
            for name in compile_config.regions
            if name not in _LOCAL_COMPILE_CALLBACKS
        ]
        if unknown:
            raise ValueError(
                f"Unknown compile.regions entries {unknown}; "
                f"registered values are {sorted(_LOCAL_COMPILE_CALLBACKS)}"
            )

    for callbacks in _LOCAL_COMPILE_CALLBACKS.values():
        for bind_local_compile_fn in callbacks:
            bind_local_compile_fn(compile_config)


__all__ = ["apply_local_compile", "local_compile"]
