# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.utils._pytree as pytree
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
    flex_attention in FlexAttention). Setting error_on_nested_fx_trace
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


def _copy_fwd_metadata_to_bw_nodes(fx_g: torch.fx.GraphModule) -> None:
    """Copy forward metadata to backward nodes across all nested FX subgraphs.

    Uses a two-pass approach over all submodule graphs (including HOP subgraphs
    like score_mod/mask_mod). Pass 1 collects forward nodes by seq_nr; pass 2
    copies custom/nn_module_stack/stack_trace from the matching forward node to
    each backward node. Backward nodes are identified by the autograd engine's
    ``autograd_backward`` tag on ``node.meta``.
    """

    def _is_backward(node: torch.fx.Node) -> bool:
        return node.meta.get("autograd_backward", False)

    seq_nr_to_fwd_node: dict[int, torch.fx.Node] = {}

    for submod in fx_g.modules():
        if not isinstance(submod, torch.fx.GraphModule):
            continue
        for node in submod.graph.nodes:
            if (
                node.op not in ("call_function", "get_attr")
                or "seq_nr" not in node.meta
                or _is_backward(node)
            ):
                continue
            seq_nr = node.meta["seq_nr"]
            if seq_nr not in seq_nr_to_fwd_node:
                seq_nr_to_fwd_node[seq_nr] = node

    for submod in fx_g.modules():
        if not isinstance(submod, torch.fx.GraphModule):
            continue
        for node in submod.graph.nodes:
            if (
                node.op not in ("call_function", "get_attr")
                or "seq_nr" not in node.meta
                or not _is_backward(node)
            ):
                continue
            fwd_node = seq_nr_to_fwd_node.get(node.meta["seq_nr"])
            if fwd_node is None or fwd_node is node:
                continue

            custom = fwd_node.meta.get("custom")
            if custom:
                node.meta.setdefault("custom", {}).update(copy.deepcopy(custom))
            nn_module_stack = fwd_node.meta.get("nn_module_stack")
            if nn_module_stack is not None:
                node.meta["nn_module_stack"] = nn_module_stack.copy()
            stack_trace = fwd_node.meta.get("stack_trace")
            if stack_trace is not None:
                node.meta["stack_trace"] = stack_trace


def extract_module_state(mod: nn.Module) -> dict[str, torch.Tensor]:
    """Return a merged dict of the module's named parameters and buffers."""
    return {
        **dict(mod.named_parameters(remove_duplicate=False)),
        **dict(mod.named_buffers(remove_duplicate=False)),
    }


@dataclass
class TracedResult:
    """Execution metadata returned by :func:`minimal_fx_tracer`.

    Attributes:
        gm: The traced FX graph as a pure function of flat tensors.
        example_inputs: Trace-time fake flat inputs used by downstream graph passes.
        state_fqns: Trace-time state keys.
        num_flat_inputs: Number of flat graph inputs before subclass unwrapping.
        input_subclass_layouts: Subclass unwrap/rewrap metadata for inputs.
        num_flat_outputs: Number of flat graph outputs before subclass rewrapping.
        output_subclass_layouts: Subclass unwrap/rewrap metadata for outputs.
        output_spec: Original output pytree spec used during reconstruction.
    """

    gm: torch.fx.GraphModule
    example_inputs: tuple[Any, ...]
    state_fqns: list[str]
    num_flat_inputs: int
    input_subclass_layouts: dict[int, SubclassLayout]
    num_flat_outputs: int
    output_subclass_layouts: dict[int, SubclassLayout]
    output_spec: pytree.TreeSpec
    tensor_input_indices: list[int] = field(default_factory=list)

    @property
    def num_static_inputs(self) -> int:
        """Number of leading graph inputs with stable tensor addresses.

        Parameters and buffers (the state entries) have fixed addresses across
        training steps.  Each may expand to multiple plain tensors after
        subclass unwrapping (e.g. DTensor -> inner tensors).
        """
        num_state = len(self.state_fqns)
        return sum(
            self.input_subclass_layouts[i].num_tensors
            if i in self.input_subclass_layouts
            else 1
            for i in range(num_state)
        )


def minimal_fx_tracer(fn: Callable) -> Callable[..., TracedResult]:
    """Return a tracer for a stateless ``fn`` with explicit ``state`` input.

    ``fn`` must be a plain callable (not an ``nn.Module``). The returned
    callable expects ``state`` as the first positional argument, followed by
    the traced user inputs::

        traced_result = minimal_fx_tracer(train_step)(state, tokens, labels)
        result = run_traced(traced_result, state, tokens, labels)

    The trace-time ``state`` and ``args`` must satisfy these constraints:
    - ``state`` must be a ``dict[str, Tensor]`` of parameters/buffers
    - all pytree leaves must be tensors or make_fx-safe primitives
      (``int``, ``float``, ``bool``, ``str``, ``None``)
    - there must be no ``nn.Module`` instances in ``state`` or ``args``

    Tensor subclasses (for example ``DTensor``) are recursively unwrapped into
    plain tensors for tracing, and the layouts needed to rewrap them are stored
    in the returned :class:`TracedResult`.
    """

    def _trace_with_args(state: Any, *args: Any) -> TracedResult:
        state_fqns = list(state.keys())
        state_flat = list(state.values())
        user_args = list(args)
        user_args_flat, user_args_spec = pytree.tree_flatten(user_args)

        # Validate leaves.
        for leaf in [*state_flat, *user_args_flat]:
            if isinstance(leaf, nn.Module):
                raise ValueError(
                    "minimal_fx_tracer requires explicit tensor state, not nn.Module "
                    "instances. Use trace_train_step(...) for the reference "
                    "train-step wrapper."
                )
            if not isinstance(leaf, _ALLOWED_LEAF_TYPES):
                raise ValueError(
                    "minimal_fx_tracer requires all pytree leaves in state/args to "
                    f"be tensors or primitives (int/float/bool/str), got "
                    f"{type(leaf).__name__}. Non-primitive values should either be "
                    "registered as pytree nodes (register_pytree_node) or constants "
                    f"(pytree.register_constant), or captured in fn's closure."
                )

        # Combined flat input: [*state, *user_args] with subclasses unwrapped.
        full_args = list(state_flat) + list(user_args_flat)
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
            state_wrapped = wrapped[: len(state_flat)]
            user_flat = wrapped[len(state_flat) :]

            state_for_fn = dict(zip(state_fqns, state_wrapped, strict=True))
            user_list = pytree.tree_unflatten(list(user_flat), user_args_spec)
            with torch.compiler._patch_autograd_grad():
                result = fn(state_for_fn, *user_list)
            flat_outs, output_spec = pytree.tree_flatten(result)
            num_flat_outputs = len(flat_outs)
            unwrapped_outs, output_layouts = _unwrap_subclasses(flat_outs)
            return unwrapped_outs

        ctx = TracingContext(fake_mode)
        # preserve_node_meta propagates fx.traceback.annotate metadata to traced nodes
        # Disable autograd multithreading so that backward tracing
        # runs on the calling thread. Without this, the C++ autograd
        # engine dispatches backward to a worker thread that has a
        # fresh contextvars.Context, making the compile_on_one_rank
        # ContextVar invisible and causing _sym_get_coordinate to
        # bake rank 0's concrete coordinates into the backward graph.
        # TODO: Move set_multithreading_enabled(False) to global init.
        # Forcing backward onto the main CPU thread is a good default
        # for both tracing and runtime, not just the tracing path.
        # _skip_nested_compile lets the current make_fx trace inline through
        # torch.compile'd FlexAttention kernels instead of erroring.
        # _non_strict_tracing_context is required by _patch_autograd_grad() and
        # marks this make_fx pass as the non-strict tracing flow, distinct from
        # other make_fx-based entry points such as non-strict export.
        with (
            fake_mode,
            tracing(ctx),
            preserve_node_meta(),
            _skip_nested_compile(),
            torch.autograd.set_multithreading_enabled(False),
            torch.compiler._non_strict_tracing_context(),
        ):
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
            example_inputs=fake_args,
            state_fqns=state_fqns,
            num_flat_inputs=num_full_args,
            input_subclass_layouts=input_layouts,
            num_flat_outputs=num_flat_outputs,
            output_subclass_layouts=output_layouts,
            output_spec=output_spec,
            tensor_input_indices=[
                i for i, x in enumerate(fake_args) if isinstance(x, torch.Tensor)
            ],
        )

    return _trace_with_args


def run_traced(
    traced_result: TracedResult,
    state: Any,
    *args: Any,
) -> Any:
    """Execute a traced graph with fresh parameters read from the live module.

    This is a reference implementation of traced-graph execution. It keeps the
    state handling, subclass unwrapping, and output reconstruction logic
    explicit instead of baking those semantics into ``TracedResult`` itself.
    Runs under ``torch.no_grad()`` because the graph already contains explicit
    backward ops (from ``torch.autograd.grad`` traced by make_fx). Without
    this, PyTorch would build a redundant autograd graph on top, keeping all
    forward intermediates alive via ``grad_fn`` references.
    """
    state_flat = list(state.values())
    user_args_flat, _ = pytree.tree_flatten(list(args))
    if any(isinstance(leaf, nn.Module) for leaf in [*state_flat, *user_args_flat]):
        raise ValueError(
            "run_traced requires explicit tensor state, not nn.Module instances. "
            "Use run_traced_train_step(...) for the reference train-step wrapper."
        )
    all_args = list(state_flat) + list(user_args_flat)
    flat_inputs, _ = _unwrap_subclasses(all_args)

    with torch.no_grad():
        flat_outputs = traced_result.gm(*flat_inputs)
    wrapped = _wrap_subclasses(
        flat_outputs,
        traced_result.num_flat_outputs,
        traced_result.output_subclass_layouts,
    )
    return pytree.tree_unflatten(wrapped, traced_result.output_spec)


def trace_train_step(fn: Callable) -> Callable[..., TracedResult]:
    """Reference implementation for capturing a whole train step via the core API."""

    def _trace_with_module(module: nn.Module, *args: Any) -> TracedResult:
        if not isinstance(module, nn.Module):
            raise ValueError(
                "trace_train_step requires args[0] to be an nn.Module, "
                f"got {type(module).__name__}."
            )
        if any(isinstance(arg, nn.Module) for arg in args):
            raise ValueError(
                "trace_train_step supports exactly one nn.Module at args[0]. "
                "Additional nn.Module instances found in args[1:]."
            )

        def _stateless_fn(state: dict[str, torch.Tensor], *user_args: Any) -> Any:
            with stateless._reparametrize_module(module, state):
                return fn(module, *user_args)

        return minimal_fx_tracer(_stateless_fn)(extract_module_state(module), *args)

    return _trace_with_module


def run_traced_train_step(
    traced_result: TracedResult,
    module: nn.Module,
    *args: Any,
    validate_module_fqns: bool = False,
) -> Any:
    """Reference implementation for executing a traced whole train step."""

    if not isinstance(module, nn.Module):
        raise ValueError(
            "run_traced_train_step requires args[0] to be an nn.Module, "
            f"got {type(module).__name__}."
        )
    if any(isinstance(arg, nn.Module) for arg in args):
        raise ValueError(
            "run_traced_train_step supports exactly one nn.Module at args[0]. "
            "Additional nn.Module instances found in args[1:]."
        )

    # TODO: Consider stronger state validation once the long-term state API settles.
    state = extract_module_state(module)
    if validate_module_fqns and list(state.keys()) != traced_result.state_fqns:
        raise ValueError(
            "module has different parameter/buffer names than during tracing.\n"
            f"  Traced: {traced_result.state_fqns}\n"
            f"  Got:    {list(state.keys())}"
        )
    return run_traced(traced_result, state, *args)
