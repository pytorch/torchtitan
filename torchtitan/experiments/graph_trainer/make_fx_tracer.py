# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
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
from torch._subclasses import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
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
class _ModuleParamsMeta:
    """Per-module parameter metadata captured during tracing."""

    fqns: list[str]
    spec: pytree.TreeSpec
    length: int


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


def _collect_module_params(
    args: tuple, module_indices: list[int]
) -> tuple[list[_ModuleParamsMeta], list[torch.Tensor]]:
    """Extract flattened params/buffers for each nn.Module arg."""
    per_module: list[_ModuleParamsMeta] = []
    all_flat: list[torch.Tensor] = []
    for i in module_indices:
        mod = args[i]
        params_dict = {
            **dict(mod.named_parameters(remove_duplicate=False)),
            **dict(mod.named_buffers(remove_duplicate=False)),
        }
        fqns = list(params_dict.keys())
        flat, spec = pytree.tree_flatten(params_dict)
        per_module.append(_ModuleParamsMeta(fqns=fqns, spec=spec, length=len(flat)))
        all_flat.extend(flat)
    return per_module, all_flat


class TracedResult:
    """Holds the traced graph and metadata needed to execute it.

    Returned by :func:`aot_function`.  Call the instance directly to execute
    the traced graph with fresh parameters read from the live modules::

        traced = aot_function(train_step, (model, tokens, labels))
        result = traced(model, tokens, labels)
    """

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        module_indices: list[int],
        per_module_params: list[_ModuleParamsMeta],
        input_subclass_layouts: list[SubclassLayout],
        output_subclass_layouts: list[SubclassLayout],
    ) -> None:
        self.gm = gm
        self.module_indices = module_indices
        self.per_module_params = per_module_params
        self.input_subclass_layouts = input_subclass_layouts
        self.output_subclass_layouts = output_subclass_layouts

    def __call__(self, *args: Any) -> list[torch.Tensor]:
        """Execute the traced graph, reading fresh params from modules in ``args``."""
        module_indices_set = set(self.module_indices)

        # Read current params from live modules.
        all_params_flat: list[torch.Tensor] = []
        for i, pmp in zip(self.module_indices, self.per_module_params, strict=True):
            mod = args[i]
            params_dict = {
                **dict(mod.named_parameters(remove_duplicate=False)),
                **dict(mod.named_buffers(remove_duplicate=False)),
            }
            fqns = list(params_dict.keys())
            if fqns != pmp.fqns:
                raise ValueError(
                    f"Module at arg position {i} has different parameter/buffer "
                    f"names than during tracing.\n"
                    f"  Traced: {pmp.fqns}\n"
                    f"  Got:    {fqns}"
                )
            flat, _ = pytree.tree_flatten(params_dict)
            all_params_flat.extend(flat)

        user_args = [a for i, a in enumerate(args) if i not in module_indices_set]
        user_args_flat, _ = pytree.tree_flatten(user_args)

        all_args = all_params_flat + list(user_args_flat)
        flat_inputs: list = []
        for a in all_args:
            if isinstance(a, torch.Tensor) and is_traceable_wrapper_subclass(a):
                inner, _ = _unwrap_subclass(a)
                flat_inputs.extend(inner)
            else:
                flat_inputs.append(a)

        flat_outputs = self.gm(*flat_inputs)
        return _wrap_to_subclasses(flat_outputs, self.output_subclass_layouts)


def aot_function(
    fn: nn.Module | Callable,
    args: tuple,
) -> TracedResult:
    """Trace ``fn(*args)`` into a flat FX graph, unwrapping tensor subclasses.

    Parameters and buffers from ``nn.Module`` instances in ``args`` are lifted
    as extra graph inputs so the returned graph is a pure function.  Tensor
    subclasses (e.g. DTensor) are recursively unwrapped into plain tensors for
    tracing, and the layouts needed to rewrap them are recorded in the returned
    :class:`TracedResult`.

    ``fn`` may be an ``nn.Module`` — in which case it is treated as though the
    caller wrote ``aot_function(lambda m, *a: m(*a), (fn,) + args)``, i.e. the
    module is prepended to ``args`` and its parameters are lifted automatically.

    The returned :class:`TracedResult` is directly callable — pass the same
    positional arguments (with live modules) to execute the graph::

        traced = aot_function(train_step, (model, tokens, labels))
        result = traced(model, tokens, labels)

    Args:
        fn: The callable (or ``nn.Module``) to trace.
        args: The positional arguments to trace with.  ``nn.Module`` instances
            are detected automatically and their parameters are lifted.
    """
    # When fn is an nn.Module, treat it as the first arg so its params get
    # lifted like any other module arg.
    if isinstance(fn, nn.Module):
        args = (fn,) + args
        fn = type(fn).__call__
    module_indices = [i for i, a in enumerate(args) if isinstance(a, nn.Module)]
    module_indices_set = set(module_indices)

    per_module_params, all_params_flat = _collect_module_params(args, module_indices)
    params_len = len(all_params_flat)

    # User args: positional args that are not nn.Module instances.
    user_args = [a for i, a in enumerate(args) if i not in module_indices_set]
    user_args_flat, user_args_spec = pytree.tree_flatten(user_args)

    # Validate leaves: tensors and make_fx-safe primitives (int, float, bool,
    # str) are allowed.  Everything else (callables, custom objects) should be
    # registered as pytree nodes/constants or captured in fn's closure.
    _ALLOWED_LEAF_TYPES = (torch.Tensor, int, float, bool, str, type(None))
    for leaf in user_args_flat:
        if not isinstance(leaf, _ALLOWED_LEAF_TYPES):
            raise ValueError(
                f"aot_function requires all pytree leaves in args to be tensors "
                f"or primitives (int/float/bool/str), got {type(leaf).__name__}. "
                f"Non-primitive values should either be registered as pytree "
                f"nodes (register_pytree_node) or constants "
                f"(pytree.register_constant), or captured in fn's closure."
            )

    # Combined flat input: [*params, *user_args] with subclasses unwrapped.
    full_args = all_params_flat + list(user_args_flat)

    unwrapped_args: list = []
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
    fake_args = tuple(
        (
            fake_mode.from_tensor(a, static_shapes=True)
            if isinstance(a, torch.Tensor)
            else a
        )
        for a in unwrapped_args
    )

    output_layouts: list[SubclassLayout] = []

    def fn_with_subclass_handling(*plain_args: Any) -> list:
        nonlocal output_layouts
        output_layouts = []

        wrapped = _wrap_to_subclasses(plain_args, input_layouts)
        all_params = wrapped[:params_len]
        user_flat = wrapped[params_len:]

        # Reconstruct per-module param dicts.
        offset = 0
        module_param_dicts = []
        for pmp in per_module_params:
            flat = all_params[offset : offset + pmp.length]
            module_param_dicts.append(pytree.tree_unflatten(list(flat), pmp.spec))
            offset += pmp.length

        user_list = pytree.tree_unflatten(list(user_flat), user_args_spec)

        # Reconstruct the original args: module positions keep the live module
        # (already in args), non-module positions get the traced user tensors.
        rebuilt: list = list(args)
        user_idx = 0
        for i in range(len(args)):
            if i not in module_indices_set:
                rebuilt[i] = user_list[user_idx]
                user_idx += 1

        with contextlib.ExitStack() as stack:
            for i, pmp_dict in zip(module_indices, module_param_dicts, strict=True):
                stack.enter_context(
                    stateless._reparametrize_module(args[i], pmp_dict)
                )

            with _patch_engine_run_backward():
                result = fn(*rebuilt)

        flat_outs, _ = pytree.tree_flatten(result)
        unwrapped_outs: list = []
        for out in flat_outs:
            if isinstance(out, torch.Tensor) and is_traceable_wrapper_subclass(out):
                inner, meta = _unwrap_subclass(out)
                unwrapped_outs.extend(inner)
                output_layouts.append(SubclassLayout(len(inner), meta))
            else:
                unwrapped_outs.append(out)
                output_layouts.append(SubclassLayout(1, None))
        return unwrapped_outs

    ctx = TracingContext(fake_mode)
    with fake_mode, tracing(ctx), preserve_node_meta(), _skip_nested_compile():
        traced = make_fx(
            fn_with_subclass_handling,
            record_stack_traces=True,
            record_module_stack=False,
        )(*fake_args)

    _copy_fwd_metadata_to_bw_nodes(traced)
    _remove_cpu_shadow_chains(traced)

    return TracedResult(
        gm=traced,
        module_indices=module_indices,
        per_module_params=per_module_params,
        input_subclass_layouts=input_layouts,
        output_subclass_layouts=output_layouts,
    )
