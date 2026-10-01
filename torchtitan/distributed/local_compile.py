# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Function-scoped torch.compile registration and configuration."""

import functools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from torchtitan.distributed.utils import is_in_batch_invariant_mode


@dataclass(kw_only=True, slots=True)
class LocalCompileConfig:
    regions: list[str] = field(
        default_factory=lambda: [
            "gated_rmsnorm",
            "loss",
            "swiglu",
            "cos_sin_rope",
        ]
    )
    """Named regions to compile independently with ``torch.compile``.

    Gated RMSNorm, loss, SwiGLU, and cos/sin RoPE compilation are enabled by
    default.
    FlexAttention manages its own compilation and is not controlled by this list.
    """

    def apply_local_compile(self) -> None:
        """Bind registered functions to eager or compiled implementations."""
        unknown = [
            name for name in self.regions if name not in _LOCAL_COMPILE_CALLBACKS
        ]
        if unknown:
            raise ValueError(
                f"Unknown compile.regions entries {unknown}; "
                f"registered values are {sorted(_LOCAL_COMPILE_CALLBACKS)}"
            )

        for callbacks in _LOCAL_COMPILE_CALLBACKS.values():
            for bind_local_compile_fn in callbacks:
                bind_local_compile_fn(self)


_LOCAL_COMPILE_CALLBACKS: dict[str, list[Callable[[LocalCompileConfig], None]]] = {}


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
        **compile_kwargs: Additional ``torch.compile`` keyword arguments;
            ``fullgraph`` is fixed to ``True`` so each function forms one complete
            compile region.
    """
    if compile_kwargs.pop("fullgraph", True) is not True:
        raise ValueError("local_compile requires fullgraph=True.")

    def decorate(reference: Callable[..., Any]) -> Callable[..., Any]:
        fn = reference

        def bind_local_compile(
            local_compile_config: LocalCompileConfig,
        ) -> None:
            nonlocal fn
            enabled = name in local_compile_config.regions
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


__all__ = ["local_compile", "LocalCompileConfig"]
