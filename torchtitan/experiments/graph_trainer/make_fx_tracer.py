# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
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

from torchtitan.experiments.graph_trainer.dynamic_shapes import (
    _fakeify_input,
    _insert_runtime_asserts as _insert_runtime_asserts_pass,
    _wrapper_subclass_has_marked_dynamic_dims,
)

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
    plain_tensors: list[torch.Tensor],
    meta: SubclassMeta,
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
        inner_dict,
        meta.ctx,
        meta.outer_size,
        meta.outer_stride,
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


def extract_train_state(
    module: nn.Module | None = None,
    optimizer: "torch.optim.Optimizer | None" = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return ``(model_state, optim_state)`` for ``minimal_fx_tracer``.

    Both are dicts (empty when the corresponding object is ``None``) and are
    sampled from the live module/optimizer, so callers can reuse this helper
    to refresh state at runtime.
    """
    model_state = extract_module_state(module) if module is not None else {}
    optim_state = optimizer.state_dict() if optimizer is not None else {}
    return model_state, optim_state


def _check_optimizer_has_module(
    module: nn.Module | None,
    optimizer: "torch.optim.Optimizer | None",
) -> None:
    """Optimizer parameters align with module.named_parameters() by order, so
    requiring a module guarantees that alignment."""
    if optimizer is not None and module is None:
        raise ValueError(
            "minimal_fx_tracer: when 'optimizer' is provided, 'module' must also "
            "be provided so optimizer parameters align with the module's parameters."
        )


@contextlib.contextmanager
def _reparametrize_train_state(
    module: nn.Module | None,
    optimizer: "torch.optim.Optimizer | None",
    model_state: dict[str, torch.Tensor],
    optim_state: dict[str, Any],
):
    """Reparametrize module and optimizer with explicit tensor state for tracing."""
    with contextlib.ExitStack() as stack:
        if optimizer is not None:
            # swap_in pairs values positionally in optimizer.param_groups flat
            # order, which differs from named_parameters() order for bucketed
            # param_groups. Walk param_groups and resolve names by id() against
            # the originals; must run before _reparametrize_module rebinds them.
            id_to_name = {
                id(p): n for n, p in module.named_parameters(remove_duplicate=False)
            }
            params_for_optim = {
                id_to_name[id(p)]: model_state[id_to_name[id(p)]]
                for group in optimizer.param_groups
                for p in group["params"]
            }
            stack.enter_context(
                torch.optim.swap_in_optimizer_params_and_state(
                    optimizer, params_for_optim, optim_state
                )
            )
        if module is not None:
            stack.enter_context(stateless._reparametrize_module(module, model_state))
        yield


@dataclass
class TracedResult:
    """Execution metadata returned by :func:`minimal_fx_tracer`.

    Attributes:
        gm: The traced FX graph as a pure function of flat tensors.
        example_inputs: Trace-time fake flat inputs used by downstream graph passes.
        num_flat_inputs: Number of flat graph inputs before subclass unwrapping.
        input_subclass_layouts: Subclass unwrap/rewrap metadata for inputs.
        user_inputs_spec: Trace-time pytree spec for ``(args, kwargs)``.
        num_flat_outputs: Number of flat graph outputs before subclass rewrapping.
        output_subclass_layouts: Subclass unwrap/rewrap metadata for outputs.
        output_spec: Original output pytree spec used during reconstruction.
        state_fqns: Trace-time module parameter/buffer FQNs.
    """

    gm: torch.fx.GraphModule

    # input related
    example_inputs: tuple[Any, ...]
    num_flat_inputs: int
    input_subclass_layouts: dict[int, SubclassLayout]
    user_inputs_spec: pytree.TreeSpec
    tensor_input_indices: list[int]

    # output related
    num_flat_outputs: int
    output_subclass_layouts: dict[int, SubclassLayout]
    output_spec: pytree.TreeSpec

    # state related
    state_fqns: list[str]

    @property
    def num_static_inputs(self) -> int:
        """Number of leading graph inputs with stable tensor addresses.

        Parameters and buffers (the state entries) have fixed addresses across
        training steps. Each may expand to multiple plain tensors after
        subclass unwrapping (e.g. DTensor -> inner tensors).

        TODO: graph_trainer does not trace optimizers yet, so optimizer state
        tensors are not counted here. When optimizer tracing is enabled, they
        should be included since their addresses are also stable across steps,
        avoiding cudagraph re-copying them every step.
        """
        num_state = len(self.state_fqns)
        return sum(
            self.input_subclass_layouts[i].num_tensors
            if i in self.input_subclass_layouts
            else 1
            for i in range(num_state)
        )


def minimal_fx_tracer(
    fn: Callable,
    module: nn.Module | None = None,
    optimizer: "torch.optim.Optimizer | None" = None,
    *,
    prepare_inputs: Callable[[tuple[Any, ...], dict[str, Any]], None] | None = None,
    prepare_call_inputs: Callable[
        [tuple[Any, ...], dict[str, Any]],
        tuple[tuple[Any, ...], dict[str, Any]] | None,
    ]
    | None = None,
    _insert_runtime_asserts: bool = False,
) -> Callable[..., TracedResult]:
    """Return a tracer that captures ``fn`` with implicit module/optimizer state.

    The returned callable takes the user-facing ``*args`` and ``**kwargs`` for
    ``fn``; module parameters/buffers and optimizer state are extracted from the
    live objects and threaded through the graph as static inputs::

        # Stateless function: no module, no optimizer.
        traced = minimal_fx_tracer(fn)(*args, **kwargs)

        # Module-only: parameters/buffers extracted from `module`.
        traced = minimal_fx_tracer(fn, module=model)(*args, **kwargs)

        # Module + optimizer: optimizer state must already be initialized
        # before tracing.
        traced = minimal_fx_tracer(fn, module=model, optimizer=opt)(*args, **kwargs)

    ``fn`` should reference ``module`` and ``optimizer`` from its enclosing
    closure — passing them explicitly through ``args``/``kwargs`` is invalid
    because ``nn.Module`` and ``Optimizer`` instances are not pytree-able.

    The trace-time ``args`` and ``kwargs`` must satisfy these constraints:

    - all pytree leaves must be tensors or make_fx-safe primitives
      (``int``, ``float``, ``bool``, ``str``, ``None``)
    - there must be no ``nn.Module`` instances in ``args`` or ``kwargs``

    Tensor subclasses (for example ``DTensor``) are recursively unwrapped into
    plain tensors for tracing, and the layouts needed to rewrap them are stored
    in the returned :class:`TracedResult`.

    ``_insert_runtime_asserts`` opts into materializing the ShapeEnv's deferred
    runtime asserts (from ``mark_unbacked()`` bounds and ``torch._check()``
    calls) into the graph as ``_assert_scalar`` nodes. Off by default because
    cudagraph capture does not evaluate these nodes, and downstream graph
    passes generally don't need them.
    """
    _check_optimizer_has_module(module, optimizer)

    def _trace_with_args(*args: Any, **kwargs: Any) -> TracedResult:
        if prepare_inputs is not None:
            prepare_inputs(args, kwargs)

        model_state, optim_state = extract_train_state(module, optimizer)
        state_fqns = list(model_state.keys())

        state_tree = {"model": model_state, "optim": optim_state}
        state_flat, state_spec = pytree.tree_flatten(state_tree)
        num_state_inputs = len(state_flat)

        user_inputs_flat, user_inputs_spec = pytree.tree_flatten((args, kwargs))

        # Validate leaves.
        for leaf in [*state_flat, *user_inputs_flat]:
            if isinstance(leaf, nn.Module):
                raise ValueError(
                    "minimal_fx_tracer requires explicit tensor state, not nn.Module "
                    "instances. Capture nn.Modules in fn's closure or pass them "
                    "via the 'module' kwarg."
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
        full_args = list(state_flat) + list(user_inputs_flat)
        num_full_args = len(full_args)
        for arg in full_args:
            if not isinstance(arg, torch.Tensor):
                continue
            if _wrapper_subclass_has_marked_dynamic_dims(arg):
                raise ValueError(
                    "minimal_fx_tracer only supports marked dynamic dims on plain "
                    "tensor inputs; wrapper subclasses such as DTensor are not "
                    "supported"
                )
        unwrapped_args, input_layouts = _unwrap_subclasses(full_args)
        fake_mode = FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=torch.fx.experimental.symbolic_shapes.ShapeEnv(),
        )
        fake_args = tuple(
            _fakeify_input(fake_mode, a, input_name=f"input_{i}")
            if isinstance(a, torch.Tensor)
            else a
            for i, a in enumerate(unwrapped_args)
        )

        output_layouts: dict[int, SubclassLayout] = {}
        num_flat_outputs: int = 0
        output_spec: pytree.TreeSpec | None = None

        def fn_with_subclass_handling(*plain_args: Any) -> list:
            nonlocal output_layouts, output_spec, num_flat_outputs
            output_layouts = {}

            wrapped = _wrap_subclasses(plain_args, num_full_args, input_layouts)
            state_wrapped = wrapped[:num_state_inputs]
            user_flat = wrapped[num_state_inputs:]

            state_t = pytree.tree_unflatten(list(state_wrapped), state_spec)
            model_state_t = state_t["model"]
            optim_state_t = state_t["optim"]
            user_args, user_kwargs = pytree.tree_unflatten(
                list(user_flat), user_inputs_spec
            )

            with _reparametrize_train_state(
                module, optimizer, model_state_t, optim_state_t
            ), torch.compiler._patch_engine_backward():
                result = fn(*user_args, **user_kwargs)

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
        _copy_fwd_metadata_to_bw_nodes(traced)

        if _insert_runtime_asserts:
            _insert_runtime_asserts_pass(traced, fake_mode)

        assert output_spec is not None
        return TracedResult(
            gm=traced,
            example_inputs=fake_args,
            num_flat_inputs=num_full_args,
            input_subclass_layouts=input_layouts,
            user_inputs_spec=user_inputs_spec,
            tensor_input_indices=[
                i for i, x in enumerate(fake_args) if isinstance(x, torch.Tensor)
            ],
            num_flat_outputs=num_flat_outputs,
            output_subclass_layouts=output_layouts,
            output_spec=output_spec,
            state_fqns=state_fqns,
        )

    return _trace_with_args


def run_traced(
    traced_result: TracedResult,
    *,
    module: nn.Module | None = None,
    optimizer: "torch.optim.Optimizer | None" = None,
    _validate_runtime: bool = False,
    interpreter_cls: type | None = None,
) -> Callable[..., Any]:
    """Return a runner that executes a traced graph against live module/optimizer state.

    The returned callable takes user-facing args and kwargs::

        traced = minimal_fx_tracer(fn, module=model, optimizer=opt)(*args, **kwargs)
        outputs = run_traced(traced, module=model, optimizer=opt)(*args, **kwargs)

    Mirrors :func:`minimal_fx_tracer`'s state extraction: parameters/buffers
    are sampled from ``module`` and the optimizer state is sampled from
    ``optimizer.state_dict()``. Runs under ``torch.no_grad()`` because the
    graph already contains explicit backward ops (from ``torch.autograd.grad``
    traced by make_fx). Without this, PyTorch would build a redundant autograd
    graph on top, keeping all forward intermediates alive via ``grad_fn``
    references.

    With ``_validate_runtime=True``, runtime module parameter/buffer FQNs must match
    trace time and runtime ``(args, kwargs)`` must flatten to the same pytree
    spec as trace time; any mismatch raises. Off by default to keep the
    per-step path overhead-free; the caller must pass kwargs in trace-time
    order.

    If ``interpreter_cls`` is provided, the traced graph is executed via that
    FX interpreter instead of called directly; used by activation tracing.
    """
    _check_optimizer_has_module(module, optimizer)

    def _run(*args: Any, **kwargs: Any) -> Any:
        model_state, optim_state = extract_train_state(module, optimizer)
        if _validate_runtime and list(model_state.keys()) != traced_result.state_fqns:
            raise ValueError(
                "module has different parameter/buffer names than during tracing.\n"
                f"  Traced: {traced_result.state_fqns}\n"
                f"  Got:    {list(model_state.keys())}"
            )
        state_tree = {"model": model_state, "optim": optim_state}
        state_flat, _ = pytree.tree_flatten(state_tree)

        user_inputs_flat, runtime_spec = pytree.tree_flatten((args, kwargs))
        # TODO: pytree's dict flatten preserves insertion order, so kwargs in a
        # different order than trace produce a different spec even though they
        # describe the same logical inputs. If pytree sorted dict keys (or
        # provided a canonicalizing flatten), this check could match valid
        # reordered calls without needing an explicit reorder step here.
        if _validate_runtime and runtime_spec != traced_result.user_inputs_spec:
            raise ValueError(
                f"input spec mismatch: runtime {runtime_spec} != "
                f"trace-time {traced_result.user_inputs_spec}"
            )
        if any(
            isinstance(leaf, nn.Module) for leaf in [*state_flat, *user_inputs_flat]
        ):
            raise ValueError(
                "run_traced requires explicit tensor state, not nn.Module instances. "
                "Capture nn.Modules in fn's closure or pass them via the 'module' kwarg."
            )
        all_args = list(state_flat) + list(user_inputs_flat)
        flat_inputs, _ = _unwrap_subclasses(all_args)

        with torch.no_grad():
            if interpreter_cls is not None:
                flat_outputs = interpreter_cls(traced_result.gm).run(*flat_inputs)
            else:
                flat_outputs = traced_result.gm(*flat_inputs)
        wrapped = _wrap_subclasses(
            flat_outputs,
            traced_result.num_flat_outputs,
            traced_result.output_subclass_layouts,
        )
        return pytree.tree_unflatten(wrapped, traced_result.output_spec)

    return _run
