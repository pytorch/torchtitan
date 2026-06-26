# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Pass and hook registries for graph_trainer.

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
POST_INIT_HOOKS: dict[str, Callable] = {}
PRE_TRAIN_STEP_HOOKS: dict[str, Callable] = {}
TRACE_INPUT_PREPARERS: dict[str, Callable] = {}
TRACE_CALL_INPUT_PREPARERS: dict[str, Callable] = {}

register_memory_policy = _make_registry_decorator(MEMORY_POLICY_REGISTRY)
register_pass_pipeline = _make_registry_decorator(PASS_PIPELINE_REGISTRY)
register_post_init_hook = _make_registry_decorator(POST_INIT_HOOKS)
register_pre_train_step_hook = _make_registry_decorator(PRE_TRAIN_STEP_HOOKS)
register_trace_input_preparer = _make_registry_decorator(TRACE_INPUT_PREPARERS)
register_trace_call_input_preparer = _make_registry_decorator(
    TRACE_CALL_INPUT_PREPARERS
)
