# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aot_nested_region: trace a subgraph once and reuse it via invoke_subgraph.

Patches a free function or an nn.Module instance so that during make_fx tracing:

1. First call: traces the subgraph via reenter_make_fx, caches it, and calls
   invoke_subgraph which emits the HOP node into the outer graph.
2. Subsequent calls with the same hash_fn key: cache hit — calls invoke_subgraph
   with the cached GraphModule (the HOP's ProxyTorchDispatchMode impl reuses
   the cached subgraph via TracingContext.hop_dispatch_set_cache).
3. Outside of tracing: transparent passthrough to the original callable.

For nn.Module instances, parameters and buffers are automatically flattened as
explicit positional operands so invoke_subgraph can deduplicate across calls with
identical structure (e.g. repeated transformer blocks sharing one traced subgraph).

For free functions, all inputs must already be positional tensor arguments.
"""

from typing import Any, Callable

import torch
import torch.nn as nn
from torch._higher_order_ops.invoke_subgraph import (
    get_invoke_subgraph_cache,
    invoke_subgraph,
)
from torch._higher_order_ops.utils import reenter_make_fx
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.nn.utils import stateless


def _ensure_subgraph_cached(
    cache_key: str,
    subgraph_fn: Callable,
    all_operands: tuple,
) -> torch.fx.GraphModule:
    """Trace and cache the subgraph if not already cached."""
    cache = get_invoke_subgraph_cache()
    if cache is not None:
        cached = cache.get_proxy_dispatch_entry(cache_key)
        if cached is not None:
            return cached

    gm = reenter_make_fx(subgraph_fn)(*all_operands)

    if cache is not None:
        cache.add_proxy_dispatch_entry(cache_key, gm)

    return gm


def _make_module_wrapper(
    module: nn.Module,
    orig_forward: Callable,
    hash_fn: Callable[..., str],
) -> Callable:
    """Return a patched forward that auto-flattens params/buffers as operands."""
    param_names = [n for n, _ in module.named_parameters()]
    buffer_names = [n for n, _ in module.named_buffers()]
    n_params = len(param_names)
    n_buffers = len(buffer_names)

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise ValueError(
                "aot_nested_region does not support keyword arguments. "
                f"Got kwargs: {set(kwargs.keys())}."
            )

        if get_proxy_mode() is None:
            return orig_forward(*args)

        cache_key = hash_fn(*args)
        if not isinstance(cache_key, str):
            raise ValueError(
                f"hash_fn must return a str, got {type(cache_key).__name__}"
            )

        # Flatten parameters and buffers as explicit operands so invoke_subgraph
        # can deduplicate across layers with the same architecture.
        param_vals = [p for _, p in module.named_parameters()]
        buffer_vals = [b for _, b in module.named_buffers()]

        # Only tensor forward args become invoke_subgraph operands — non-tensors
        # (None, int, etc.) are specialized as constants inside the subgraph, same
        # as Dynamo's "automatic" input mode for invoke_subgraph.
        tensor_arg_indices = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
        tensor_args = tuple(args[i] for i in tensor_arg_indices)
        all_operands = (*param_vals, *buffer_vals, *tensor_args)

        def subgraph_fn(*operands: Any) -> tuple:
            params = dict(zip(param_names, operands[:n_params]))
            buffers = dict(zip(buffer_names, operands[n_params : n_params + n_buffers]))
            tensor_fwd_args = operands[n_params + n_buffers :]
            # Reconstruct the full forward args: start from the original call args
            # (non-tensors stay as-is from the outer closure), then slot the traced
            # tensor operands back into their original positions.
            full_args = list(args)
            for idx, val in zip(tensor_arg_indices, tensor_fwd_args):
                full_args[idx] = val
            with stateless._reparametrize_module(module, {**params, **buffers}):
                out = orig_forward(*full_args)
            if isinstance(out, torch.Tensor):
                return (out,)
            return tuple(out) if isinstance(out, (list, tuple)) else (out,)

        gm = _ensure_subgraph_cached(cache_key, subgraph_fn, all_operands)
        result = invoke_subgraph(gm, cache_key, *all_operands)
        if isinstance(result, (tuple, list)) and len(result) == 1:
            return result[0]
        return result

    return wrapper


def aot_nested_region(
    fn=None,
    *,
    hash_fn: Callable[..., str],
):
    """Patch a free function or nn.Module instance to emit invoke_subgraph during make_fx tracing.

    For nn.Module instances, parameters and buffers are automatically flattened
    as the leading positional operands, enabling deduplication across layers that
    share the same architecture (e.g. all transformer blocks map to one subgraph).

    For free functions, all inputs must already be positional tensor arguments.

    Usage on an nn.Module instance::

        aot_nested_region(layer, hash_fn=lambda *args: "block")

    Usage as a decorator on a free function::

        @aot_nested_region(hash_fn=lambda *args: "block")
        def block_fwd(x, w1, b1, w2, b2):
            ...

    Args:
        fn: The callable or nn.Module instance to patch. If None, returns a decorator.
        hash_fn: Returns a str cache key from the forward call arguments (not including
            the module itself). Calls with the same key share a single traced subgraph.
    """

    def patch(target):
        if isinstance(target, type):
            raise ValueError(
                f"aot_nested_region does not support classes, got {target.__name__}. "
                "Pass an nn.Module instance or a free function instead."
            )

        if isinstance(target, nn.Module):
            orig_forward = target.forward
            target.forward = _make_module_wrapper(target, orig_forward, hash_fn)
            return target

        if not callable(target):
            raise ValueError(
                f"aot_nested_region expects a callable, got {type(target).__name__}."
            )

        orig_fn = target

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if kwargs:
                raise ValueError(
                    "aot_nested_region does not support keyword arguments. "
                    f"Got kwargs: {set(kwargs.keys())}. invoke_subgraph requires "
                    "all inputs to be positional tensor operands."
                )

            if get_proxy_mode() is None:
                return orig_fn(*args)

            cache_key = hash_fn(*args)
            if not isinstance(cache_key, str):
                raise ValueError(
                    f"hash_fn must return a str, got {type(cache_key).__name__}"
                )

            def subgraph_fn(*a):
                out = orig_fn(*a)
                if isinstance(out, torch.Tensor):
                    return (out,)
                return tuple(out) if isinstance(out, (list, tuple)) else (out,)

            gm = _ensure_subgraph_cached(cache_key, subgraph_fn, args)
            result = invoke_subgraph(gm, cache_key, *args)
            if isinstance(result, (tuple, list)) and len(result) == 1:
                return result[0]
            return result

        return wrapper

    if fn is None:
        return patch
    return patch(fn)
