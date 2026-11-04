# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pass registries for graph_trainer.

Centralizes all registries so that ``passes.py`` and ``memory_policy.py``
can both import from here without circular dependencies.
"""

from __future__ import annotations

from collections.abc import Callable


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------


def _make_registry_decorator(registry: dict):
    """Create a decorator that registers a function into the given registry."""

    def register(key: str):
        def decorator(fn: Callable) -> Callable:
            registry[key] = fn
            return fn

        return decorator

    return register


# ---------------------------------------------------------------------------
# Registries — keyed by string name
# ---------------------------------------------------------------------------

MEMORY_POLICY_REGISTRY: dict[str, Callable] = {}
PASS_PIPELINE_REGISTRY: dict[str, Callable] = {}

register_memory_policy = _make_registry_decorator(MEMORY_POLICY_REGISTRY)
register_pass_pipeline = _make_registry_decorator(PASS_PIPELINE_REGISTRY)


GRAPH_PASS_REGISTRY: dict[str, Callable] = {}


def register_graph_pass(fn: Callable) -> Callable:
    """Register a graph pass that recipes enable through ``compile.extra_passes``.

    The pass is registered under its function name, which is also the name
    ``compile.extra_passes``, ``compile.disable_passes`` and the pass log use. It
    has the usual ``(gm, example_inputs) -> gm`` signature and runs on the
    normalized joint forward/backward graph before memory policy tagging, in
    ``compile.extra_passes`` order. This is the extension point for model- or
    shape-specialized rewrites, e.g. replacing a subgraph with a handwritten
    kernel. The registering module can live anywhere, including outside
    torchtitan; the recipe imports it before building the trainer.

    Example::

        @register_graph_pass
        def my_kernel_fusion_pass(gm, example_inputs):
            ...

        config.compile.extra_passes = [my_kernel_fusion_pass.__name__]
    """
    name = fn.__name__
    existing = GRAPH_PASS_REGISTRY.get(name)
    # Re-importing the defining module re-registers the same pass.
    if existing is not None and (existing.__module__, existing.__qualname__) != (
        fn.__module__,
        fn.__qualname__,
    ):
        raise ValueError(
            f"Graph pass {name!r} from {fn.__module__} conflicts with the "
            f"pass of the same name registered by {existing.__module__}"
        )
    GRAPH_PASS_REGISTRY[name] = fn
    return fn
