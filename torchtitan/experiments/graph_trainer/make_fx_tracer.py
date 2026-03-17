# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


@dataclass
class SubclassMeta:
    cls: type
    attrs: list[str]
    ctx: Any
    inner_metas: dict[str, tuple[int, Any]]
    outer_size: torch.Size
    outer_stride: tuple[int, ...]


@dataclass
class SubclassLayout:
    num_tensors: int
    meta: SubclassMeta | None


@dataclass
class TracedResult:
    """Holds the traced graph and metadata needed to run it."""

    gm: torch.fx.GraphModule
    params_len: int
    params_spec: pytree.TreeSpec
    input_subclass_layouts: list[SubclassLayout]
    output_subclass_layouts: list[SubclassLayout]


def _unwrap_subclass(t: torch.Tensor) -> tuple[list[torch.Tensor], SubclassMeta | None]:
    if not is_traceable_wrapper_subclass(t):
        return [t], None
    attrs, ctx = t.__tensor_flatten__()
    all_inner = []
    inner_metas = {}
    for attr in attrs:
        inner_t = getattr(t, attr)
        tensors, meta = _unwrap_subclass(inner_t)
        all_inner.extend(tensors)
        inner_metas[attr] = (len(tensors), meta)
    meta = SubclassMeta(
        cls=type(t),
        attrs=attrs,
        ctx=ctx,
        inner_metas=inner_metas,
        outer_size=t.size(),
        outer_stride=t.stride(),
    )
    return all_inner, meta


def _wrap_to_subclass(
    plain_tensors: list[torch.Tensor], meta: SubclassMeta
) -> torch.Tensor:
    inner_dict = {}
    idx = 0
    for attr in meta.attrs:
        num_inner, inner_meta = meta.inner_metas[attr]
        inner_tensors = plain_tensors[idx : idx + num_inner]
        idx += num_inner
        if inner_meta is None:
            inner_dict[attr] = inner_tensors[0]
        else:
            inner_dict[attr] = _wrap_to_subclass(list(inner_tensors), inner_meta)
    return meta.cls.__tensor_unflatten__(
        inner_dict, meta.ctx, meta.outer_size, meta.outer_stride
    )


def _wrap_to_subclasses(
    flat_tensors: tuple[torch.Tensor, ...] | list[torch.Tensor],
    layouts: list[SubclassLayout],
) -> list[torch.Tensor]:
    wrapped = []
    idx = 0
    for layout in layouts:
        tensors = flat_tensors[idx : idx + layout.num_tensors]
        idx += layout.num_tensors
        if layout.meta is None:
            wrapped.append(tensors[0])
        else:
            wrapped.append(_wrap_to_subclass(list(tensors), layout.meta))
    return wrapped


def _remove_cpu_shadow_chains(gm: torch.fx.GraphModule) -> None:
    """Remove dead CPU tensor chains left by DTensor's shadow-op bookkeeping.

    DTensor keeps CPU "shadow" copies of tensor metadata (size, stride) as
    regular aten ops.  After make_fx tracing these ops end up in the graph but
    never feed a real GPU computation, so they are pure overhead.  This pass
    finds every chain rooted at a CPU ``empty_strided`` whose outputs never
    reach a GPU node with downstream users, and erases the whole chain.

    TODO: figure out a way to avoid tracing them into graph in the first place.
    """
    to_remove: set[torch.fx.Node] = set()

    for node in gm.graph.nodes:
        if node in to_remove:
            continue

        if not (
            node.op == "call_function"
            and node.target == torch.ops.aten.empty_strided.default
        ):
            continue
        device = node.kwargs.get("device")
        if device is None or device.type != "cpu":
            continue

        chain: set[torch.fx.Node] = set()
        queue = [node]
        feeds_gpu = False

        while queue and not feeds_gpu:
            current = queue.pop()
            if current in chain:
                continue
            chain.add(current)
            for user in current.users:
                val = user.meta.get("val")
                if isinstance(val, torch.Tensor) and val.device.type != "cpu":
                    if user.users:
                        feeds_gpu = True
                        break
                    chain.add(user)
                    continue
                queue.append(user)

        if not feeds_gpu:
            to_remove |= chain

    for node in reversed(list(gm.graph.nodes)):
        if node in to_remove:
            gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()


def trace_module(
    mod: nn.Module,
    args: tuple,
) -> TracedResult:
    """Trace ``mod(*args)`` into a flat FX graph, unwrapping tensor subclasses.

    Parameters and buffers are lifted as extra graph inputs so the returned
    graph is a pure function.  Tensor subclasses (e.g. DTensor) are recursively
    unwrapped into plain tensors for tracing, and the layouts needed to rewrap
    them are recorded in the returned :class:`TracedResult`.
    """
    named_parameters = dict(mod.named_parameters(remove_duplicate=False))
    named_buffers = dict(mod.named_buffers(remove_duplicate=False))

    params_and_buffers = {**named_parameters, **named_buffers}
    params_and_buffers_flat, params_spec = pytree.tree_flatten(params_and_buffers)
    params_len = len(params_and_buffers_flat)

    def functional_call(*all_args):
        flat_params = all_args[:params_len]
        user_args = all_args[params_len:]
        params = pytree.tree_unflatten(list(flat_params), params_spec)
        with stateless._reparametrize_module(mod, params):
            return mod.forward(*user_args)

    user_args_flat, user_args_spec = pytree.tree_flatten(args)
    full_args = tuple(params_and_buffers_flat) + tuple(user_args_flat)

    unwrapped_args = []
    input_layouts: list[SubclassLayout] = []

    for arg in full_args:
        if isinstance(arg, torch.Tensor) and is_traceable_wrapper_subclass(arg):
            inner_tensors, meta = _unwrap_subclass(arg)
            unwrapped_args.extend(inner_tensors)
            input_layouts.append(SubclassLayout(len(inner_tensors), meta))
        else:
            unwrapped_args.append(arg)
            input_layouts.append(SubclassLayout(1, None))

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    def to_fake(t):
        if isinstance(t, torch.Tensor):
            return fake_mode.from_tensor(t)
        return t

    fake_args = tuple(to_fake(a) for a in unwrapped_args)

    output_layouts: list[SubclassLayout] = []

    def fn_with_subclass_handling(*plain_args):
        nonlocal output_layouts
        output_layouts = []

        wrapped_args = _wrap_to_subclasses(plain_args, input_layouts)

        params_args = wrapped_args[:params_len]
        user_args_wrapped = wrapped_args[params_len:]
        user_args_restored = pytree.tree_unflatten(
            list(user_args_wrapped), user_args_spec
        )

        outputs = functional_call(*params_args, *user_args_restored)

        flat_outputs, _ = pytree.tree_flatten(outputs)
        unwrapped_outputs = []
        for out in flat_outputs:
            if isinstance(out, torch.Tensor) and is_traceable_wrapper_subclass(out):
                inner, meta = _unwrap_subclass(out)
                unwrapped_outputs.extend(inner)
                output_layouts.append(SubclassLayout(len(inner), meta))
            else:
                unwrapped_outputs.append(out)
                output_layouts.append(SubclassLayout(1, None))

        return unwrapped_outputs

    # preserve_node_meta propagates fx.traceback.annotate metadata to traced nodes
    with fake_mode, preserve_node_meta():
        traced = make_fx(fn_with_subclass_handling, record_stack_traces=True)(
            *fake_args
        )

    _remove_cpu_shadow_chains(traced)

    return TracedResult(
        gm=traced,
        params_len=params_len,
        params_spec=params_spec,
        input_subclass_layouts=input_layouts,
        output_subclass_layouts=output_layouts,
    )


def run_traced_module(
    traced_result: TracedResult,
    params_and_buffers: dict[str, torch.Tensor],
    args: tuple,
) -> list[torch.Tensor]:
    """Execute a traced graph and rewrap outputs into their original subclass types.

    Accepts a ``params_and_buffers`` dict (from ``named_parameters`` /
    ``named_buffers``) instead of the module itself, so callers control exactly
    which parameter snapshot is used.
    """
    params_flat, _ = pytree.tree_flatten(params_and_buffers)
    user_args_flat, _ = pytree.tree_flatten(args)

    all_args = []
    for a in itertools.chain(params_flat, user_args_flat):
        if isinstance(a, torch.Tensor) and is_traceable_wrapper_subclass(a):
            inner, _ = _unwrap_subclass(a)
            all_args.extend(inner)
        else:
            all_args.append(a)

    flat_outputs = traced_result.gm(*all_args)
    return _wrap_to_subclasses(flat_outputs, traced_result.output_subclass_layouts)
