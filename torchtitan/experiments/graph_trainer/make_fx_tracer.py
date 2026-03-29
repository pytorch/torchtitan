# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
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
from torch._guards import TracingContext, tracing
from torch._higher_order_ops.invoke_subgraph import InvokeSubgraphAutogradOp
from torch._higher_order_ops.utils import reenter_make_fx
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import get_proxy_mode, make_fx
from torch.fx.traceback import preserve_node_meta
from torch.nn.utils import stateless
from torch.utils._python_dispatch import is_traceable_wrapper_subclass


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


def _trace_bw_graph_for_make_fx(
    subgraph_fn: Callable,
    num_primals: int,
    num_tangents: int,
    primals_and_tangents: tuple,
) -> torch.fx.GraphModule:
    """Trace the backward graph for an invoke_subgraph subgraph inside make_fx.

    Differentiates through the original Python callable (subgraph_fn) rather
    than a pre-traced GraphModule. The backward re-runs the forward from
    scratch (activation-checkpointing style) so no partitioner is needed.

    reenter_make_fx converts primals_and_tangents into fresh leaf fake tensors,
    preserving requires_grad=True for params and activations alike (requires_grad
    is True for both leaf params and non-leaf activation tensors output by prior
    invoke_subgraph calls). No .detach() or .requires_grad_() is needed: we never
    touch requires_grad on the original primals, and the fresh leaf fake tensors
    inside bwd_fn have the correct requires_grad for torch.autograd.grad.

    Output signature: (*primals, *tangents) → (*grads_for_all_primals, *fw_outs)
    so InvokeSubgraphAutogradOp.backward can extract grads via [:num_primals].
    """

    def bwd_fn(*args: torch.Tensor) -> tuple:
        primals = list(args[:num_primals])
        tangents = list(args[num_primals : num_primals + num_tangents])

        with torch.enable_grad():
            # Differentiate through the original Python callable, not the
            # pre-traced gm. This ensures correct autograd semantics; the
            # forward is recomputed here (activation-checkpointing style).
            fw_outs = subgraph_fn(*primals)

        if isinstance(fw_outs, torch.Tensor):
            fw_outs = (fw_outs,)
        fw_outs = tuple(fw_outs)

        differentiable_outs = [
            o for o in fw_outs if isinstance(o, torch.Tensor) and o.requires_grad
        ]
        differentiable_ins = [
            p for p in primals if isinstance(p, torch.Tensor) and p.requires_grad
        ]

        raw_grads = torch.autograd.grad(
            differentiable_outs, differentiable_ins, tangents, allow_unused=True
        )

        # Map back to all primals, zeros for non-differentiable.
        grad_iter = iter(raw_grads)
        all_grads: list[torch.Tensor] = []
        for p in primals:
            if isinstance(p, torch.Tensor) and p.requires_grad:
                g = next(grad_iter)
                all_grads.append(g if g is not None else torch.zeros_like(p))
            else:
                all_grads.append(torch.zeros_like(p) if isinstance(p, torch.Tensor) else p)

        # Signature: (*grads_for_all_primals, *fw_outs)
        return tuple(all_grads) + fw_outs

    return reenter_make_fx(bwd_fn)(*primals_and_tangents)


@contextmanager
def _patch_invoke_subgraph_backward() -> Generator[None, None, None]:
    """Patch InvokeSubgraphAutogradOp.backward to handle make_fx tracing context.

    In the normal torch.compile path, InvokeSubgraphAutogradOp.backward is
    called by AOTAutograd's joint tracer with fake tensors. It lazily traces a
    bw_graph via trace_joint_graph_as_bwd (which uses create_joint, an
    AOTAutograd utility) and emits an invoke_subgraph(bw_graph, ...) HOP node.

    When loss.backward() is called inside make_fx instead, two things break:

    1. trace_joint_graph_as_bwd / create_joint is AOTAutograd machinery that is
       not appropriate for the make_fx tracing path. A concrete symptom: it reads
       output_node.meta["val"].requires_grad to filter tangents, but snapshot_fake
       always strips requires_grad → all outputs appear non-differentiable →
       filtered_grad_outs=[] → AssertionError inside create_joint.

    2. The C++ autograd engine runs InvokeSubgraphAutogradOp.backward on a worker
       thread. threading.local() does not propagate across threads, so
       get_invoke_subgraph_cache() returns None on the worker thread even though
       a TracingContext is active on the main thread.

    This patch fixes both: it captures the TracingContext on the main thread and
    re-establishes it on the worker thread, and replaces trace_joint_graph_as_bwd
    with _trace_bw_graph_for_make_fx, which runs subgraph_fn (the original Python
    callable) directly to build a real autograd graph and uses torch.autograd.grad
    — no AOTAutograd machinery needed.

    TODO(upstream): This patch should be contributed upstream to PyTorch once the
    make_fx-based tracer moves out of torchtitan/experiments. At that point,
    InvokeSubgraphAutogradOp.backward should natively handle the make_fx tracing
    context instead of requiring this monkey-patch.
    """
    from torch._guards import TracingContext, tracing
    from torch._higher_order_ops.invoke_subgraph import (
        get_invoke_subgraph_cache,
        invoke_subgraph,
        saved_values,
    )

    # Capture the TracingContext from the main thread now. The autograd engine
    # runs InvokeSubgraphAutogradOp.backward on a worker thread, where
    # threading.local() does not propagate — so get_invoke_subgraph_cache()
    # would return None if we didn't re-establish the context inside the patch.
    captured_tracing_ctx = TracingContext.try_get()

    _orig_backward = InvokeSubgraphAutogradOp.backward

    @staticmethod  # type: ignore[misc]
    def _patched_backward(ctx, *grad_outs):  # type: ignore[no-untyped-def]
        if get_proxy_mode() is None:
            # Not in make_fx — use the original path unchanged.
            return _orig_backward(ctx, *grad_outs)

        # Re-establish the TracingContext for this worker thread so that
        # get_invoke_subgraph_cache() can find the cache.
        with tracing(captured_tracing_ctx):
            return _patched_backward_impl(ctx, *grad_outs)

    @staticmethod  # type: ignore[misc]
    def _patched_backward_impl(ctx, *grad_outs):  # type: ignore[no-untyped-def]
        # ── make_fx path ──────────────────────────────────────────────────────
        subgraph = ctx._subgraph
        identifier = ctx._identifier
        primals = saved_values(ctx)

        # We do not use output_metadata.indexes_with_no_grad here.
        # _trace_bw_graph_for_make_fx runs subgraph_fn directly and inspects
        # o.requires_grad on the real (fake) tensors to determine differentiability,
        # so static graph metadata is not needed. The C++ autograd engine already
        # passes None for outputs that don't require grad, so filtering by
        # o is not None is sufficient.
        tangents = tuple(o for o in grad_outs if o is not None)
        primals_and_tangents = primals + tangents

        # Use the original Python callable stored on the gm so we differentiate
        # through the source function rather than the pre-traced graph.
        # Falls back to the gm itself for subgraphs not created by aot_nested_region
        # (e.g. mark_compile_region), preserving the old behavior.
        subgraph_fn = getattr(subgraph, "_orig_subgraph_fn", subgraph)

        bw_graph = _trace_bw_graph_for_make_fx(
            subgraph_fn, len(primals), len(tangents), primals_and_tangents
        )

        invoke_subgraph_cache = get_invoke_subgraph_cache()
        assert invoke_subgraph_cache is not None, (
            "_patched_backward: expected TracingContext to be active on this thread "
            "(captured_tracing_ctx should have been re-established via tracing())"
        )
        existing, suffix = invoke_subgraph_cache.get_lazy_bwd_entry(identifier, ())
        if existing is None:
            suffix = invoke_subgraph_cache.add_lazy_bwd_entry(identifier, (), bw_graph)

        # bw_graph output: (*grads_for_all_primals, *fw_outs)
        # We want only the gradient tensors (first len(primals) outputs).
        grads = invoke_subgraph(
            bw_graph, f"bw_{identifier}_{suffix}", *primals_and_tangents
        )[: len(primals)]
        return None, None, None, *grads

    InvokeSubgraphAutogradOp.backward = _patched_backward
    try:
        yield
    finally:
        InvokeSubgraphAutogradOp.backward = _orig_backward


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


def trace_module(
    mod: nn.Module,
    args: tuple,
) -> TracedResult:
    """Trace ``mod(*args)`` into a flat FX graph, unwrapping tensor subclasses.

    Parameters and buffers are lifted as extra graph inputs so the returned
    graph is a pure function.  Tensor subclasses (e.g. DTensor) are recursively
    unwrapped into plain tensors for tracing, and the layouts needed to rewrap
    them are recorded in the returned :class:`TracedResult`.

    Args:
        mod: The module to trace.
        args: The user arguments to trace with.
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

    fake_mode = FakeTensorMode(
        allow_non_fake_inputs=True,
        shape_env=torch.fx.experimental.symbolic_shapes.ShapeEnv(),
    )

    def to_fake(t):
        if isinstance(t, torch.Tensor):
            return fake_mode.from_tensor(t, static_shapes=True)
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

        with _patch_engine_run_backward():
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
    # Install TracingContext so that invoke_subgraph's ProxyTorchDispatchMode impl
    # can cache traced subgraphs (its cache lives in TracingContext.hop_dispatch_set_cache).
    # Without this, invoke_subgraph re-traces the subgraph on every call.
    tracing_ctx = TracingContext(fake_mode=fake_mode)
    with fake_mode, preserve_node_meta(), _skip_nested_compile(), tracing(tracing_ctx), _patch_invoke_subgraph_backward():
        traced = make_fx(
            fn_with_subclass_handling,
            record_stack_traces=True,
            record_module_stack=False,  # don't need nn_module_stack for now
        )(*fake_args)

    # Copy forward annotations to backward nodes.
    # Must run before DCE so that forward nodes used for matching aren't removed.
    _copy_fwd_metadata_to_bw_nodes(traced)

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
