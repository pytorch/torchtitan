# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aot_nested_region: trace a subgraph once and reuse it via invoke_subgraph.

Patches a free function, an nn.Module instance, or a method so that during
make_fx tracing:

1. First call: traces the subgraph via reenter_make_fx, caches it, and calls
   invoke_subgraph which emits the HOP node into the outer graph.
2. Subsequent calls with the same hash_fn key: cache hit — calls invoke_subgraph
   with the cached GraphModule (the HOP's ProxyTorchDispatchMode impl reuses
   the cached subgraph via TracingContext.hop_dispatch_set_cache).
3. Outside of tracing: transparent passthrough to the original callable.

For nn.Module instances and method decorators, parameters and buffers are
automatically flattened as explicit positional operands so invoke_subgraph can
deduplicate across calls with identical structure (e.g. repeated transformer
blocks sharing one traced subgraph).

Tensor arguments (positional and keyword) become invoke_subgraph operands.
Non-tensor arguments (None, int, etc.) are specialized as constants inside the
subgraph, mirroring Dynamo's "automatic" input mode for invoke_subgraph.
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

    # Store the original callable on the gm so _trace_bw_graph_for_make_fx
    # can differentiate through the Python function rather than the traced gm,
    # ensuring correct autograd semantics in the backward.
    gm._orig_subgraph_fn = subgraph_fn

    if cache is not None:
        cache.add_proxy_dispatch_entry(cache_key, gm)

    return gm


def _extract_tensor_operands(
    args: tuple, kwargs: dict
) -> tuple[list[int], list[str], tuple, tuple]:
    """Split args/kwargs into tensor operands and non-tensor constants.

    Returns:
        tensor_arg_indices: positions in args that are tensors
        tensor_kwarg_keys: keys in kwargs whose values are tensors
        tensor_args: tensor values from args (in index order)
        tensor_kwargs: tensor values from kwargs (in key order)
    """
    tensor_arg_indices = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
    tensor_kwarg_keys = [k for k, v in kwargs.items() if isinstance(v, torch.Tensor)]
    tensor_args = tuple(args[i] for i in tensor_arg_indices)
    tensor_kwargs = tuple(kwargs[k] for k in tensor_kwarg_keys)
    return tensor_arg_indices, tensor_kwarg_keys, tensor_args, tensor_kwargs


def _reconstruct_args_kwargs(
    args: tuple,
    kwargs: dict,
    tensor_arg_indices: list[int],
    tensor_kwarg_keys: list[str],
    tensor_fwd_args: tuple,
    tensor_fwd_kwargs: tuple,
) -> tuple[list, dict]:
    """Reconstruct full args/kwargs by slotting traced tensor operands back in."""
    full_args = list(args)
    for idx, val in zip(tensor_arg_indices, tensor_fwd_args):
        full_args[idx] = val
    full_kwargs = dict(kwargs)
    for k, val in zip(tensor_kwarg_keys, tensor_fwd_kwargs):
        full_kwargs[k] = val
    return full_args, full_kwargs


def _invoke_as_module(
    module: nn.Module,
    orig_forward: Callable,
    cache_key: str,
    args: tuple,
    kwargs: dict,
) -> Any:
    """Dispatch an nn.Module forward call through invoke_subgraph.

    Flattens parameters and buffers as the leading operands so invoke_subgraph
    can deduplicate across layers with the same architecture.

    orig_forward must be the bound method (no self in args/kwargs).
    """
    param_names = [n for n, _ in module.named_parameters()]
    buffer_names = [n for n, _ in module.named_buffers()]
    n_params = len(param_names)
    n_buffers = len(buffer_names)

    # Flatten parameters and buffers as explicit operands so invoke_subgraph
    # can deduplicate across layers with the same architecture.
    param_vals = [p for _, p in module.named_parameters()]
    buffer_vals = [b for _, b in module.named_buffers()]

    # Only tensor args/kwargs become invoke_subgraph operands — non-tensors
    # (None, int, etc.) are specialized as constants inside the subgraph, same
    # as Dynamo's "automatic" input mode for invoke_subgraph.
    tensor_arg_indices, tensor_kwarg_keys, tensor_args, tensor_kwargs = (
        _extract_tensor_operands(args, kwargs)
    )
    all_operands = (*param_vals, *buffer_vals, *tensor_args, *tensor_kwargs)

    def subgraph_fn(*operands: Any) -> tuple:
        params = dict(zip(param_names, operands[:n_params]))
        buffers = dict(zip(buffer_names, operands[n_params : n_params + n_buffers]))
        n_tensor_args = len(tensor_arg_indices)
        tensor_fwd_args = operands[n_params + n_buffers : n_params + n_buffers + n_tensor_args]
        tensor_fwd_kwargs = operands[n_params + n_buffers + n_tensor_args :]
        # Reconstruct full args/kwargs: non-tensors stay from the outer closure,
        # tensor positions are filled from the traced operands.
        full_args, full_kwargs = _reconstruct_args_kwargs(
            args, kwargs,
            tensor_arg_indices, tensor_kwarg_keys,
            tensor_fwd_args, tensor_fwd_kwargs,
        )
        with stateless._reparametrize_module(module, {**params, **buffers}):
            out = orig_forward(*full_args, **full_kwargs)
        if isinstance(out, torch.Tensor):
            return (out,)
        return tuple(out) if isinstance(out, (list, tuple)) else (out,)

    gm = _ensure_subgraph_cached(cache_key, subgraph_fn, all_operands)
    result = invoke_subgraph(gm, cache_key, *all_operands)
    if isinstance(result, (tuple, list)) and len(result) == 1:
        return result[0]
    return result


def _make_module_wrapper(
    module: nn.Module,
    orig_forward: Callable,
    hash_fn: Callable[..., str],
) -> Callable:
    """Return a patched forward that auto-flattens params/buffers as operands."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if get_proxy_mode() is None:
            return orig_forward(*args, **kwargs)
        cache_key = hash_fn(*args, **kwargs)
        if not isinstance(cache_key, str):
            raise ValueError(
                f"hash_fn must return a str, got {type(cache_key).__name__}"
            )
        return _invoke_as_module(module, orig_forward, cache_key, args, kwargs)

    return wrapper


def aot_nested_region(
    fn=None,
    *,
    hash_fn: Callable[..., str],
):
    """Patch a free function, nn.Module instance, or method to emit invoke_subgraph during make_fx tracing.

    For nn.Module instances and method decorators, parameters and buffers are
    automatically flattened as leading operands for deduplication across layers.

    Tensor arguments (positional and keyword) become invoke_subgraph operands.
    Non-tensor arguments (None, int, etc.) are specialized as constants inside the
    subgraph, mirroring Dynamo's "automatic" input mode for invoke_subgraph.

    Usage on an nn.Module instance::

        aot_nested_region(layer, hash_fn=lambda *args, **kwargs: "block")

    Usage as a decorator on a method (params/buffers auto-flattened)::

        class Block(nn.Module):
            @aot_nested_region(hash_fn=lambda *args, **kwargs: "block")
            def forward(self, x):
                ...

    Usage as a decorator on a free function::

        @aot_nested_region(hash_fn=lambda *args, **kwargs: "block")
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
            if get_proxy_mode() is None:
                return orig_fn(*args, **kwargs)

            cache_key = hash_fn(*args, **kwargs)
            if not isinstance(cache_key, str):
                raise ValueError(
                    f"hash_fn must return a str, got {type(cache_key).__name__}"
                )

            # Method decorator: first arg is an nn.Module instance (self).
            # Dispatch through _invoke_as_module so params/buffers are flattened
            # as operands, identical to the nn.Module instance patching path.
            if args and isinstance(args[0], nn.Module):
                module = args[0]
                bound_method = orig_fn.__get__(module, type(module))
                return _invoke_as_module(module, bound_method, cache_key, args[1:], kwargs)

            # Free function path: only tensor args/kwargs become operands.
            tensor_arg_indices, tensor_kwarg_keys, tensor_args, tensor_kwargs = (
                _extract_tensor_operands(args, kwargs)
            )
            all_operands = (*tensor_args, *tensor_kwargs)

            def subgraph_fn(*operands: Any) -> tuple:
                n_tensor_args = len(tensor_arg_indices)
                tensor_fwd_args = operands[:n_tensor_args]
                tensor_fwd_kwargs = operands[n_tensor_args:]
                full_args, full_kwargs = _reconstruct_args_kwargs(
                    args, kwargs,
                    tensor_arg_indices, tensor_kwarg_keys,
                    tensor_fwd_args, tensor_fwd_kwargs,
                )
                out = orig_fn(*full_args, **full_kwargs)
                if isinstance(out, torch.Tensor):
                    return (out,)
                return tuple(out) if isinstance(out, (list, tuple)) else (out,)

            gm = _ensure_subgraph_cached(cache_key, subgraph_fn, all_operands)
            result = invoke_subgraph(gm, cache_key, *all_operands)
            if isinstance(result, (tuple, list)) and len(result) == 1:
                return result[0]
            return result

        return wrapper

    if fn is None:
        return patch
    return patch(fn)
