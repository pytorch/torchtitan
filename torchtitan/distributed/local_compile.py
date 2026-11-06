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
import torch._inductor.config as inductor_config

from torchtitan.distributed.batch_invariant import is_in_batch_invariant_mode


_LOCAL_COMPILE_CALLBACKS: dict[str, list[Callable[["LocalCompileConfig"], None]]] = {}
_ALWAYS_COMPILE_CALLBACKS: list[Callable[["LocalCompileConfig"], None]] = []


@dataclass(kw_only=True, slots=True)
class LocalCompileConfig:
    """Configuration for TorchTitan-owned local ``torch.compile`` boundaries."""

    regions: list[str] = field(default_factory=list)

    @staticmethod
    def _compile(
        reference: Callable[..., Any],
        **compile_kwargs: Any,
    ) -> Callable[..., Any]:
        """Compile one boundary with Inductor.

        Out-of-tree subclasses can override this hook to select another backend.
        """
        return torch.compile(reference, **compile_kwargs)

    def apply(self) -> None:
        """Bind named regions and always-compiled helpers using this config."""
        unknown = [
            name for name in self.regions if name not in _LOCAL_COMPILE_CALLBACKS
        ]
        if unknown:
            raise ValueError(
                f"Unknown local_compile.regions entries {unknown}; "
                f"registered values are {sorted(_LOCAL_COMPILE_CALLBACKS)}"
            )

        for callbacks in _LOCAL_COMPILE_CALLBACKS.values():
            for bind_local_compile_fn in callbacks:
                bind_local_compile_fn(self)
        for bind_compile_fn in _ALWAYS_COMPILE_CALLBACKS:
            bind_compile_fn(self)


def local_compile(
    name: str,
    *,
    batch_invariant: bool,
    **compile_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function that can be compiled independently.

    Args:
        name: Name used to enable the function in a model's
            ``local_compile.regions``.
        batch_invariant: Whether the compiled function preserves batch invariance.
        **compile_kwargs: Additional ``torch.compile`` keyword arguments;
            ``fullgraph`` is fixed to ``True`` so each function forms one complete
            compile region.
    """
    if compile_kwargs.pop("fullgraph", True) is not True:
        raise ValueError("local_compile requires fullgraph=True.")

    def decorate(reference: Callable[..., Any]) -> Callable[..., Any]:
        fn = reference

        def bind_local_compile(config: LocalCompileConfig) -> None:
            nonlocal fn
            enabled = name in config.regions
            batch_invariant_mode = is_in_batch_invariant_mode()
            if enabled and batch_invariant_mode and not batch_invariant:
                raise ValueError(
                    f"Local compile region {name!r} does not support "
                    "batch-invariant mode; remove it from local_compile.regions."
                )
            if enabled:
                region_kwargs = dict(compile_kwargs)
                options = region_kwargs.get("options") or {}
                if options.get("coordinate_descent_tuning") and (
                    torch.are_deterministic_algorithms_enabled()
                    # TORCHINDUCTOR_DETERMINISTIC=1 sets only Inductor's flag.
                    or inductor_config.deterministic
                    or batch_invariant_mode
                ):
                    # Deterministic mode skips reduction tuning but keeps the option's
                    # rewrite of size-1 matmuls: Kimi K3's attention_residual ran 5.7x
                    # slower. Batch-invariant mode can't use configs picked by timing.
                    # set_determinism does the same for flex.
                    # TODO: drop the deterministic case once Inductor's size-1 bmm/mm
                    # rewrite (torch/_inductor/decomposition.py) respects that mode.
                    region_kwargs["options"] = {
                        **options,
                        "coordinate_descent_tuning": False,
                    }
                fn = config._compile(reference, fullgraph=True, **region_kwargs)
            else:
                fn = reference

        @functools.wraps(reference)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            return fn(*args, **kwargs)

        _LOCAL_COMPILE_CALLBACKS.setdefault(name, []).append(bind_local_compile)
        return wrapped

    return decorate


def compile(
    reference: Callable[..., Any],
    **compile_kwargs: Any,
) -> Callable[..., Any]:
    """Create an always-compiled helper controlled by ``LocalCompileConfig``."""
    fn = LocalCompileConfig()._compile(reference, **compile_kwargs)

    def bind_compile(config: LocalCompileConfig) -> None:
        nonlocal fn
        fn = config._compile(reference, **compile_kwargs)

    @functools.wraps(reference)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    _ALWAYS_COMPILE_CALLBACKS.append(bind_compile)
    return wrapped


__all__ = [
    "compile",
    "local_compile",
    "LocalCompileConfig",
]
