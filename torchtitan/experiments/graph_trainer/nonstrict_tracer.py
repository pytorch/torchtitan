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


def unwrap_subclass(t: torch.Tensor) -> tuple[list[torch.Tensor], SubclassMeta | None]:
    if not is_traceable_wrapper_subclass(t):
        return [t], None
    attrs, ctx = t.__tensor_flatten__()
    all_inner = []
    inner_metas = {}
    for attr in attrs:
        inner_t = getattr(t, attr)
        tensors, meta = unwrap_subclass(inner_t)
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


def wrap_to_subclass(
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
            inner_dict[attr] = wrap_to_subclass(list(inner_tensors), inner_meta)
    return meta.cls.__tensor_unflatten__(
        inner_dict, meta.ctx, meta.outer_size, meta.outer_stride
    )


def wrap_inputs_to_subclasses(
    plain_args: tuple[torch.Tensor, ...],
    subclass_metas: list[tuple[int, SubclassMeta | None]],
) -> list[torch.Tensor]:
    wrapped = []
    idx = 0
    for num_tensors, meta in subclass_metas:
        tensors = plain_args[idx : idx + num_tensors]
        idx += num_tensors
        if meta is None:
            wrapped.append(tensors[0])
        else:
            wrapped.append(wrap_to_subclass(list(tensors), meta))
    return wrapped


def rewrap_outputs(outputs, output_subclass_metas):
    wrapped_outputs = []
    idx = 0
    for num_tensors, meta in output_subclass_metas:
        output_tensors = outputs[idx : idx + num_tensors]
        idx += num_tensors
        if meta is None:
            wrapped_outputs.append(output_tensors[0])
        else:
            wrapped_outputs.append(wrap_to_subclass(list(output_tensors), meta))
    return wrapped_outputs


def _remove_cpu_shadow_chains(gm: torch.fx.GraphModule) -> None:
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
) -> torch.fx.GraphModule:
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
    subclass_metas = []

    for arg in full_args:
        if isinstance(arg, torch.Tensor) and is_traceable_wrapper_subclass(arg):
            inner_tensors, meta = unwrap_subclass(arg)
            unwrapped_args.extend(inner_tensors)
            subclass_metas.append((len(inner_tensors), meta))
        else:
            unwrapped_args.append(arg)
            subclass_metas.append((1, None))

    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)

    def to_fake(t):
        if isinstance(t, torch.Tensor):
            return fake_mode.from_tensor(t)
        return t

    fake_args = tuple(to_fake(a) for a in unwrapped_args)

    output_subclass_metas = []

    def fn_with_subclass_handling(*plain_args):
        nonlocal output_subclass_metas
        output_subclass_metas = []

        wrapped_args = wrap_inputs_to_subclasses(plain_args, subclass_metas)

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
                inner, meta = unwrap_subclass(out)
                unwrapped_outputs.extend(inner)
                output_subclass_metas.append((len(inner), meta))
            else:
                unwrapped_outputs.append(out)
                output_subclass_metas.append((1, None))

        return unwrapped_outputs

    with fake_mode, preserve_node_meta():
        traced = make_fx(fn_with_subclass_handling, record_stack_traces=True)(
            *fake_args
        )

    _remove_cpu_shadow_chains(traced)

    traced._params_len = params_len
    traced._params_spec = params_spec
    traced._input_subclass_metas = subclass_metas
    traced._output_subclass_metas = output_subclass_metas

    return traced


def run_traced_module(
    traced: torch.fx.GraphModule,
    mod: nn.Module,
    args: tuple,
):
    params_and_buffers = {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }
    params_flat, _ = pytree.tree_flatten(params_and_buffers)
    user_args_flat, _ = pytree.tree_flatten(args)

    all_args = []
    for a in itertools.chain(params_flat, user_args_flat):
        if isinstance(a, torch.Tensor) and is_traceable_wrapper_subclass(a):
            inner, _ = unwrap_subclass(a)
            all_args.extend(inner)
        else:
            all_args.append(a)

    return traced(*all_args)
