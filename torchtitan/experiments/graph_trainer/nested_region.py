# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""aot_nested_region: trace a subgraph once and reuse it via invoke_subgraph.

Follows the same flow as torch.compile's invoke_subgraph handling in Dynamo:

1. First call during make_fx tracing: traces the subgraph via reenter_make_fx,
   caches the resulting GraphModule and output metadata, emits invoke_subgraph.
2. Subsequent calls: cache hit — constructs cheap fake tensors from cached
   metadata and emits invoke_subgraph without re-tracing or re-running.
3. Outside of tracing: transparent passthrough to the original forward.
"""

from typing import Any, Callable, NamedTuple

import torch
import torch.nn as nn
from torch._higher_order_ops.invoke_subgraph import invoke_subgraph
from torch._higher_order_ops.utils import reenter_make_fx
from torch.fx.experimental.proxy_tensor import get_proxy_mode, track_tensor_tree
from torch.nn.utils import stateless


class _TensorMetadata(NamedTuple):
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device
    requires_grad: bool


class _CachedSubgraph(NamedTuple):
    gm: torch.fx.GraphModule
    output_metadata: list[_TensorMetadata]


# Cache: identifier -> _CachedSubgraph
_subgraph_cache: dict[str, _CachedSubgraph] = {}


def _extract_output_metadata_from_graph(
    gm: torch.fx.GraphModule,
) -> list[_TensorMetadata]:
    """Extract output tensor metadata from a traced graph's output node.

    The output node's args contain FX nodes whose ``meta['val']`` holds
    the fake tensors produced during tracing — no need to re-execute the graph.
    """
    for node in reversed(list(gm.graph.nodes)):
        if node.op == "output":
            # output args are ((tensor_node, ...),)
            out_nodes = node.args[0]
            if not isinstance(out_nodes, (tuple, list)):
                out_nodes = (out_nodes,)
            metadata = []
            for n in out_nodes:
                val = n.meta["val"]
                metadata.append(
                    _TensorMetadata(
                        val.shape, val.stride(), val.dtype, val.device, val.requires_grad
                    )
                )
            return metadata
    raise RuntimeError("Graph has no output node")


def _reconstruct_fake_outputs(
    metadata: list[_TensorMetadata], fake_mode: Any
) -> tuple[torch.Tensor, ...]:
    with fake_mode:
        return tuple(
            torch.empty_strided(
                m.shape, m.stride, dtype=m.dtype, device=m.device,
                requires_grad=m.requires_grad,
            )
            for m in metadata
        )


def _make_functional_module(
    module: nn.Module,
    orig_forward: Callable,
    param_names: list[str],
) -> Callable[..., tuple[torch.Tensor, ...]]:
    n_params = len(param_names)

    def functional_forward(*flat_args):
        params = flat_args[:n_params]
        user_args = flat_args[n_params:]
        params_dict = dict(zip(param_names, params))
        with stateless._reparametrize_module(module, params_dict):
            out = orig_forward(*user_args)
        if isinstance(out, torch.Tensor):
            return (out,)
        return tuple(out) if isinstance(out, (list, tuple)) else (out,)

    return functional_forward


def _emit_invoke_subgraph(
    cache_key: str,
    all_operands: tuple,
    cached: _CachedSubgraph,
) -> Any:
    """Emit an invoke_subgraph FX node into the outer graph."""
    gm = cached.gm
    proxy_mode = get_proxy_mode()
    tracer = proxy_mode.tracer
    qualname = tracer.get_fresh_qualname("repeated_subgraph")
    tracer.root.register_module(qualname, gm)

    proxy_operands = tuple(tracer.unwrap_proxy(op) for op in all_operands)

    out_proxy = tracer.create_proxy(
        "call_function",
        invoke_subgraph,
        (gm, cache_key, *proxy_operands),
        {},
    )

    from torch._guards import detect_fake_mode

    fake_mode = detect_fake_mode(all_operands)
    example_out = _reconstruct_fake_outputs(cached.output_metadata, fake_mode)

    result = track_tensor_tree(
        example_out, out_proxy, constant=None, tracer=tracer
    )

    if isinstance(result, (tuple, list)) and len(result) == 1:
        return result[0]
    return result


def _traced_forward_module(
    module: nn.Module,
    orig_forward: Callable,
    hash_fn: Callable[..., str],
    *args: Any,
) -> Any:
    """Tracing-time forward for nn.Module targets."""
    names = []
    tensors = []
    for name, param in module.named_parameters():
        names.append(name)
        tensors.append(param)
    for name, buf in module.named_buffers():
        names.append(name)
        tensors.append(buf)

    cache_key = hash_fn(module, *args)
    if not isinstance(cache_key, str):
        raise ValueError(
            f"hash_fn must return a str, got {type(cache_key).__name__}"
        )
    all_operands = (*tensors, *args)

    if cache_key not in _subgraph_cache:
        functional_fn = _make_functional_module(module, orig_forward, names)
        gm = reenter_make_fx(functional_fn)(*all_operands)
        _subgraph_cache[cache_key] = _CachedSubgraph(
            gm, _extract_output_metadata_from_graph(gm)
        )

    return _emit_invoke_subgraph(cache_key, all_operands, _subgraph_cache[cache_key])


def _traced_forward_fn(
    fn: Callable,
    hash_fn: Callable[..., str],
    *args: Any,
) -> Any:
    """Tracing-time forward for plain callable targets."""
    cache_key = hash_fn(*args)
    if not isinstance(cache_key, str):
        raise ValueError(
            f"hash_fn must return a str, got {type(cache_key).__name__}"
        )

    if cache_key not in _subgraph_cache:

        def wrapper(*a):
            out = fn(*a)
            if isinstance(out, torch.Tensor):
                return (out,)
            return tuple(out) if isinstance(out, (list, tuple)) else (out,)

        gm = reenter_make_fx(wrapper)(*args)
        _subgraph_cache[cache_key] = _CachedSubgraph(
            gm, _extract_output_metadata_from_graph(gm)
        )

    return _emit_invoke_subgraph(cache_key, args, _subgraph_cache[cache_key])


def aot_nested_region(
    fn=None,
    *,
    hash_fn: Callable[..., str],
):
    """Patch a callable to emit invoke_subgraph during make_fx tracing.

    Works on nn.Module instances, nn.Module classes, or plain callables:

        @aot_nested_region(hash_fn=lambda *args: "block")
        class TransformerBlock(nn.Module):
            ...

        aot_nested_region(layer, hash_fn=lambda *args: "block")

        @aot_nested_region(hash_fn=lambda *args: "fn")
        def my_fn(x, y):
            ...
    """

    def patch(target):
        if isinstance(target, type) and issubclass(target, nn.Module):
            orig_forward = target.forward

            def patched_forward(self, *args):
                if get_proxy_mode() is not None:
                    return _traced_forward_module(
                        self, orig_forward.__get__(self), hash_fn, *args
                    )
                return orig_forward(self, *args)

            target.forward = patched_forward
            return target

        if isinstance(target, nn.Module):
            orig_forward = target.forward

            def patched_forward(*args):
                if get_proxy_mode() is not None:
                    return _traced_forward_module(
                        target, orig_forward, hash_fn, *args
                    )
                return orig_forward(*args)

            target.forward = patched_forward
            return target

        # Plain callable
        orig_fn = target

        def patched_fn(*args):
            if get_proxy_mode() is not None:
                return _traced_forward_fn(orig_fn, hash_fn, *args)
            return orig_fn(*args)

        return patched_fn

    if fn is None:
        return patch
    return patch(fn)
