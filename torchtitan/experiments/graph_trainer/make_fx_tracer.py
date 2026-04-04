# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._functorch._aot_autograd.logging_utils import (
    setup_stacktrace_preservation_hooks,
)
from torch._guards import tracing, TracingContext
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.traceback import preserve_node_meta
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

# Tensors and make_fx-safe primitives are allowed as pytree leaves in args.
# Everything else (callables, custom objects) should be registered as pytree
# nodes/constants or captured in fn's closure.
_ALLOWED_LEAF_TYPES = (torch.Tensor, int, float, bool, str, type(None))


@contextmanager
def _skip_nested_compile() -> Generator[None, None, None]:
    """Tell dynamo to skip torch.compile calls encountered during make_fx tracing.

    make_fx cannot trace through torch.compile'd functions (e.g. compiled
    flex_attention in FlexAttentionWrapper). Setting error_on_nested_fx_trace
    to False makes dynamo silently inline the wrapped function instead of
    raising, so make_fx traces the underlying ops normally.
    """
    prev = torch._dynamo.config.error_on_nested_fx_trace
    torch._dynamo.config.error_on_nested_fx_trace = False
    try:
        yield
    finally:
        torch._dynamo.config.error_on_nested_fx_trace = prev


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


def _unwrap_subclasses(
    args: list,
) -> tuple[list, dict[int, SubclassLayout]]:
    """Unwrap tensor subclasses into plain tensors.

    Returns the flattened plain tensors and a dict mapping original arg index
    to its SubclassLayout.  Plain tensors have no entry.
    """
    flat: list = []
    layouts: dict[int, SubclassLayout] = {}
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and is_traceable_wrapper_subclass(arg):
            inner_tensors, meta = _unwrap_subclass(arg)
            layouts[i] = SubclassLayout(len(inner_tensors), meta)
            flat.extend(inner_tensors)
        else:
            flat.append(arg)
    return flat, layouts


def _wrap_subclasses(
    flat_tensors: tuple | list,
    num_args: int,
    layouts: dict[int, SubclassLayout],
) -> list:
    """Rewrap plain tensors back into their original subclass types.

    Positions not in ``layouts`` are plain tensors (taken one-to-one).
    """
    wrapped = []
    idx = 0
    for i in range(num_args):
        if i in layouts:
            layout = layouts[i]
            tensors = flat_tensors[idx : idx + layout.num_tensors]
            idx += layout.num_tensors
            wrapped.append(_wrap_to_subclass(list(tensors), layout.meta))
        else:
            wrapped.append(flat_tensors[idx])
            idx += 1
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


@contextmanager
def _patch_engine_run_backward() -> Generator[None, None, None]:
    """Patch _engine_run_backward to install stacktrace preservation hooks.

    Why this is needed:
    When make_fx traces a function that calls loss.backward(), the backward
    pass is decomposed into primitive ATen ops. Normally (in eager autograd),
    ``setup_stacktrace_preservation_hooks`` is called by the autograd engine
    to propagate ``seq_nr`` from forward ops to their corresponding backward
    ops. Under make_fx tracing, this hook setup doesn't happen automatically
    because the engine path differs, so backward FX nodes end up without
    ``seq_nr`` metadata. Without ``seq_nr``, we can't correlate backward
    nodes back to their forward counterparts (needed by
    ``_copy_fwd_metadata_to_bw_nodes``).

    This context manager patches ``_engine_run_backward`` to call
    ``setup_stacktrace_preservation_hooks`` before the autograd engine runs,
    restoring ``seq_nr`` propagation during tracing.

    We must patch the name in both modules since ``torch.autograd.__init__``
    imports it via ``from .graph import``.
    """
    import torch.autograd
    import torch.autograd.graph

    _orig_fn = torch.autograd.graph._engine_run_backward

    def _patched(t_outputs, *args, **kwargs):  # type: ignore[no-untyped-def]
        roots = [
            t.grad_fn
            for t in t_outputs
            if isinstance(t, torch.Tensor) and t.grad_fn is not None
        ]
        if roots:
            setup_stacktrace_preservation_hooks(roots)
        return _orig_fn(t_outputs, *args, **kwargs)

    torch.autograd.graph._engine_run_backward = _patched  # type: ignore[assignment]
    torch.autograd._engine_run_backward = _patched  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.autograd.graph._engine_run_backward = _orig_fn  # type: ignore[assignment]
        torch.autograd._engine_run_backward = _orig_fn  # type: ignore[assignment]


def _copy_fwd_metadata_to_bw_nodes(fx_g: torch.fx.GraphModule) -> None:
    """Copy forward node metadata (custom) to later nodes sharing the same seq_nr.

    Walks the graph in a single pass. The first node seen for each seq_nr is
    treated as the forward node.
    Subsequent nodes with the same seq_nr (typically backward nodes) receive
    the forward node's custom metadata.
    """
    seq_nr_to_fwd_node: dict[int, torch.fx.Node] = {}

    for node in fx_g.graph.nodes:
        if node.op not in ("call_function", "get_attr") or "seq_nr" not in node.meta:
            continue
        seq_nr = node.meta["seq_nr"]
        if seq_nr not in seq_nr_to_fwd_node:
            seq_nr_to_fwd_node[seq_nr] = node
        else:
            fwd_node = seq_nr_to_fwd_node[seq_nr]

            custom = fwd_node.meta.get("custom")
            if custom:
                node.meta.setdefault("custom", {}).update(custom)
            nn_module_stack = fwd_node.meta.get("nn_module_stack")
            if nn_module_stack is not None:
                node.meta["nn_module_stack"] = nn_module_stack.copy()


def _get_params_and_buffers(mod: nn.Module) -> dict[str, torch.Tensor]:
    """Return a merged dict of the module's named parameters and buffers."""
    return {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }


class TracedResult:
    """Holds the traced graph and metadata needed to execute it.

    Returned by :func:`minimal_fx_tracer`. Call the tracer returned by
    :func:`minimal_fx_tracer` with trace-time args, then execute it with
    :func:`run_traced`::

        traced = minimal_fx_tracer(train_step)(model, tokens, labels)
        result = run_traced(traced, model, tokens, labels)

    Args:
        gm: The traced FX graph (a pure function of flat tensors).
        param_fqns: Fully qualified names of the module's parameters and
            buffers, recorded at trace time for validation.
        num_params: Number of parameters + buffers (flat count).
        num_flat_inputs: Number of original args (before subclass unwrapping).
        input_subclass_layouts: Maps arg positions that are tensor subclasses
            to their unwrap/rewrap metadata.  Plain tensors have no entry.
        num_flat_outputs: Number of original outputs (before subclass unwrapping).
        output_subclass_layouts: Maps output positions that are tensor subclasses
            to their rewrap metadata.  Plain tensors have no entry.
        output_spec: Pytree spec of the original function's return value,
            used to reconstruct the output structure.
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        param_fqns: list[str],
        num_params: int,
        num_flat_inputs: int,
        input_subclass_layouts: dict[int, SubclassLayout],
        num_flat_outputs: int,
        output_subclass_layouts: dict[int, SubclassLayout],
        output_spec: pytree.TreeSpec,
    ) -> None:
        self.gm = gm
        self.param_fqns = param_fqns
        self.num_params = num_params
        self.num_flat_inputs = num_flat_inputs
        self.input_subclass_layouts = input_subclass_layouts
        self.num_flat_outputs = num_flat_outputs
        self.output_subclass_layouts = output_subclass_layouts
        self.output_spec = output_spec


def _trace_with_args(
    fn: Callable,
    args: tuple,
) -> TracedResult:
    """Trace ``fn(*args)`` into a flat FX graph, unwrapping tensor subclasses.

    ``args[0]`` must be an ``nn.Module``.  Its parameters and buffers are
    lifted as extra graph inputs so the returned graph is a pure function.
    Tensor subclasses (e.g. DTensor) are recursively unwrapped into plain
    tensors for tracing, and the layouts needed to rewrap them are recorded
    in the returned :class:`TracedResult`.

    ``fn`` must be a plain callable (not an ``nn.Module``).  This keeps the
    trace and execute calling conventions identical — the same ``args`` are
    passed at both trace time and execution time, with no hidden arg
    prepending.  Non-tensor, non-module values like ``loss_fn`` should be
    captured in ``fn``'s closure rather than passed as args.

    Execute the returned :class:`TracedResult` with :func:`run_traced`, passing
    the same positional arguments (with the live module first)::

        traced = minimal_fx_tracer(train_step)(model, tokens, labels)
        result = run_traced(traced, model, tokens, labels)

    Args:
        fn: The callable to trace.
        args: The positional arguments to trace with.  The first element must
            be an ``nn.Module`` whose parameters will be lifted.
    """
    if not isinstance(args[0], nn.Module):
        raise ValueError(
            "minimal_fx_tracer requires args[0] to be an nn.Module, "
            f"got {type(args[0]).__name__}."
        )
    if any(isinstance(a, nn.Module) for a in args[1:]):
        raise ValueError(
            "minimal_fx_tracer supports exactly one nn.Module at args[0]. "
            "Additional nn.Module instances found in args[1:]."
        )
    mod = args[0]

    # Extract params/buffers from the module.
    params_dict = _get_params_and_buffers(mod)
    param_fqns = list(params_dict.keys())
    params_flat = list(params_dict.values())
    num_params = len(params_flat)

    # User args: everything after the module.
    user_args = list(args[1:])
    user_args_flat, user_args_spec = pytree.tree_flatten(user_args)

    # Validate leaves.
    for leaf in user_args_flat:
        if not isinstance(leaf, _ALLOWED_LEAF_TYPES):
            raise ValueError(
                f"minimal_fx_tracer requires all pytree leaves in args to be tensors "
                f"or primitives (int/float/bool/str), got {type(leaf).__name__}. "
                f"Non-primitive values should either be registered as pytree "
                f"nodes (register_pytree_node) or constants "
                f"(pytree.register_constant), or captured in fn's closure."
            )

    # Combined flat input: [*params, *user_args] with subclasses unwrapped.
    full_args = params_flat + list(user_args_flat)
    num_full_args = len(full_args)
    unwrapped_args, input_layouts = _unwrap_subclasses(full_args)

    fake_mode = FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=torch.fx.experimental.symbolic_shapes.ShapeEnv(),
    )
    fake_args = tuple(
        (
            fake_mode.from_tensor(a, static_shapes=True)
            if isinstance(a, torch.Tensor)
            else a
        )
        for a in unwrapped_args
    )

    output_layouts: dict[int, SubclassLayout] = {}
    num_flat_outputs: int = 0
    output_spec: pytree.TreeSpec | None = None

    def fn_with_subclass_handling(*plain_args: Any) -> list:
        nonlocal output_layouts, output_spec, num_flat_outputs
        output_layouts = {}

        wrapped = _wrap_subclasses(plain_args, num_full_args, input_layouts)
        params_wrapped = wrapped[:num_params]
        user_flat = wrapped[num_params:]

        params_for_mod = dict(zip(param_fqns, params_wrapped, strict=True))
        user_list = pytree.tree_unflatten(list(user_flat), user_args_spec)

        # Reconstruct the original args: module at position 0 keeps the live
        # module, remaining positions get the traced user tensors.
        rebuilt = [mod] + user_list

        with stateless._reparametrize_module(mod, params_for_mod):
            with _patch_engine_run_backward():
                result = fn(*rebuilt)

        flat_outs, output_spec = pytree.tree_flatten(result)
        num_flat_outputs = len(flat_outs)
        unwrapped_outs, output_layouts = _unwrap_subclasses(flat_outs)
        return unwrapped_outs

    ctx = TracingContext(fake_mode)
    # preserve_node_meta propagates fx.traceback.annotate metadata to traced nodes
    with fake_mode, tracing(ctx), preserve_node_meta(), _skip_nested_compile():
        traced = make_fx(
            fn_with_subclass_handling,
            record_stack_traces=True,
            record_module_stack=False,  # don't need nn_module_stack for now
        )(*fake_args)

    # Copy forward annotations to backward nodes.
    # Must run before DCE so that forward nodes used for matching aren't removed.
    _copy_fwd_metadata_to_bw_nodes(traced)

    _remove_cpu_shadow_chains(traced)

    assert output_spec is not None
    return TracedResult(
        gm=traced,
        param_fqns=param_fqns,
        num_params=num_params,
        num_flat_inputs=num_full_args,
        input_subclass_layouts=input_layouts,
        num_flat_outputs=num_flat_outputs,
        output_subclass_layouts=output_layouts,
        output_spec=output_spec,
    )


def minimal_fx_tracer(fn: Callable) -> Callable[..., TracedResult]:
    """Return a tracer for ``fn`` that traces a concrete module/input invocation.

    ``fn`` must be a plain callable (not an ``nn.Module``). The returned
    callable expects the trace-time positional arguments, with the live module
    at position 0::

        traced = minimal_fx_tracer(train_step)(model, tokens, labels)
        result = run_traced(traced, model, tokens, labels)
    """

    def trace_with_args(*args: Any) -> TracedResult:
        return _trace_with_args(fn, args)

    return trace_with_args


def run_traced(traced_result: TracedResult, *args: Any) -> Any:
    """Execute a traced graph with fresh parameters read from the live module.

    This is a reference implementation of traced-graph execution. It keeps the
    parameter lookup, subclass unwrapping, and output reconstruction logic
    explicit instead of baking those semantics into ``TracedResult`` itself.

    The module must be the first argument (position 0), matching the
    convention enforced by :func:`minimal_fx_tracer`.
    """

    mod = args[0]
    params_dict = _get_params_and_buffers(mod)
    fqns = list(params_dict.keys())
    if fqns != traced_result.param_fqns:
        raise ValueError(
            f"Module at args[0] has different parameter/buffer "
            f"names than during tracing.\n"
            f"  Traced: {traced_result.param_fqns}\n"
            f"  Got:    {fqns}"
        )
    params_flat = list(params_dict.values())

    user_args = list(args[1:])
    user_args_flat, _ = pytree.tree_flatten(user_args)

    all_args = params_flat + list(user_args_flat)
    flat_inputs, _ = _unwrap_subclasses(all_args)

    flat_outputs = traced_result.gm(*flat_inputs)
    wrapped = _wrap_subclasses(
        flat_outputs,
        traced_result.num_flat_outputs,
        traced_result.output_subclass_layouts,
    )
    return pytree.tree_unflatten(wrapped, traced_result.output_spec)


run_traced_module = run_traced
