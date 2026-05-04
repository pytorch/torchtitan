# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
from collections import defaultdict
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


def extract_train_state(
    module: nn.Module | None = None,
    optimizer: "torch.optim.Optimizer | None" = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any] | None]:
    """Return ``(model_state, optim_state)`` for ``minimal_fx_tracer``.

    ``model_state`` is the merged parameter/buffer dict (empty if ``module``
    is ``None``). ``optim_state`` is ``optimizer.state_dict()`` (``None`` if
    ``optimizer`` is ``None``). Both are sampled from the live module/optimizer,
    so callers can reuse this helper to refresh state at runtime.
    """
    model_state = extract_module_state(module) if module is not None else {}
    optim_state = optimizer.state_dict() if optimizer is not None else None
    return model_state, optim_state


def _prepare_optimizer_reparametrization(
    optimizer: "torch.optim.Optimizer",
    parameters_and_buffers: dict[str, torch.Tensor],
    optimizer_state_dict: dict[str, Any],
):
    """Validate and normalize optimizer state for ``_reparametrize_optimizer``.

    This follows the same structural assumptions as DCP-compatible optimizers,
    but consumes the raw ``optimizer.state_dict()`` format: ``state`` is keyed
    by packed parameter ids and each param group contains the live optimizer
    group fields plus a packed ``params`` list whose order matches
    ``optimizer.param_groups``.

    TODO: remove this local copy and switch to the upstream
    ``torch.nn.utils._reparametrize_optimizer`` helper once
    https://github.com/pytorch/pytorch/pull/181643 lands.
    """
    if not optimizer.state:
        raise RuntimeError(
            "_reparametrize_optimizer requires initialized optimizer state."
        )
    if not isinstance(optimizer_state_dict, dict):
        raise RuntimeError(
            "_reparametrize_optimizer requires a DCP-style optimizer state_dict."
        )

    state = optimizer_state_dict.get("state")
    param_groups = optimizer_state_dict.get("param_groups")
    if not isinstance(state, dict) or not isinstance(param_groups, list):
        raise RuntimeError(
            "_reparametrize_optimizer requires an optimizer.state_dict()-style "
            "state_dict with 'state' and 'param_groups' entries."
        )
    if any(isinstance(name, torch.Tensor) for name in state):
        raise RuntimeError(
            "_reparametrize_optimizer requires optimizer.state_dict()-style "
            "state keyed by packed parameter ids."
        )
    if len(optimizer.param_groups) != len(param_groups):
        raise RuntimeError(
            "optimizer_state_dict has a different number of parameter groups than "
            "the live optimizer."
        )

    group_rebind_infos = []
    # Raw optimizer state_dicts address parameters by packed integer ids, so we
    # align explicit parameter tensors with optimizer.param_groups by order.
    # Example: if param_groups[*]["params"] is [[0, 1], [2]] and
    # parameters_and_buffers.values() is [fake_p0, fake_p1, fake_p2], then the
    # first optimizer group is rebound to [fake_p0, fake_p1] and the second to
    # [fake_p2].
    flat_parameters = list(parameters_and_buffers.values())
    flat_param_offset = 0
    packed_param_ids: set[int] = set()
    for idx, (group, saved_group) in enumerate(
        zip(optimizer.param_groups, param_groups, strict=True)
    ):
        if not isinstance(saved_group, dict):
            raise RuntimeError(
                "_reparametrize_optimizer requires each optimizer param group "
                "to be a dictionary."
            )
        names = saved_group.get("params")
        if not isinstance(names, list) or not all(
            isinstance(param_id, int) for param_id in names
        ):
            raise RuntimeError(
                "_reparametrize_optimizer requires optimizer.state_dict()-style "
                "param_groups[*]['params'] entries keyed by packed parameter ids."
            )
        if len(group["params"]) != len(names):
            raise RuntimeError(
                "optimizer_state_dict param group does not match the size of "
                f"live optimizer param group {idx}."
            )
        next_offset = flat_param_offset + len(names)
        if next_offset > len(flat_parameters):
            raise RuntimeError(
                "_reparametrize_optimizer requires the explicit parameter state to "
                "match optimizer.param_groups ordering."
            )
        # Slice out the explicit tensors that should back this optimizer group.
        rebind_params = flat_parameters[flat_param_offset:next_offset]
        flat_param_offset = next_offset

        for param_id in names:
            packed_param_ids.add(param_id)
            param_state = state.get(param_id, {})
            if not isinstance(param_state, dict):
                raise RuntimeError(
                    "_reparametrize_optimizer requires per-parameter optimizer "
                    "state entries to be dictionaries."
                )

        missing_group_keys = [
            key for key in saved_group if key != "params" and key not in group
        ]
        if missing_group_keys:
            raise RuntimeError(
                "_reparametrize_optimizer requires optimizer.state_dict()-style "
                "param group keys to match the live optimizer group keys. "
                f"Missing live keys for group {idx}: {missing_group_keys}"
            )

        group_rebind_infos.append(
            (
                group,  # live optimizer group to mutate
                saved_group,  # serialized group values to install temporarily
                rebind_params,  # explicit tensors that replace group["params"]
                {
                    key: group[key] for key in saved_group if key != "params"
                },  # restore data
            )
        )

    if flat_param_offset != len(flat_parameters):
        raise RuntimeError(
            "_reparametrize_optimizer requires the explicit parameter state to "
            "match optimizer.param_groups ordering."
        )

    if any(key not in packed_param_ids for key in state):
        raise RuntimeError(
            "_reparametrize_optimizer requires optimizer.state_dict()-style state "
            "to contain only per-parameter entries keyed by packed parameter ids."
        )
    return state, group_rebind_infos


@contextlib.contextmanager
def _reparametrize_optimizer(
    optimizer: "torch.optim.Optimizer",
    parameters_and_buffers: dict[str, torch.Tensor],
    optimizer_state_dict: dict[str, Any],
):
    """Temporarily rebind an optimizer to explicit parameter tensors.

    ``optimizer_state_dict`` must be in the raw ``optimizer.state_dict()``
    format. This helper assumes a DCP-compatible optimizer structure, but it
    consumes the optimizer-native packed-param-id representation rather than
    DCP's FQN-keyed exported state. Tensor values in the per-parameter
    optimizer state are shared (not cloned), so in-place ops on existing
    entries (e.g. ``exp_avg.add_(...)``) propagate back to
    ``optimizer_state_dict``. The per-parameter state dicts themselves are
    shallow-copied, so structural changes made during the trace (new keys,
    re-bound tensors) stay local and do not pollute ``optimizer_state_dict``.

    TODO: remove this local copy and switch to the upstream
    ``torch.nn.utils._reparametrize_optimizer`` helper once
    https://github.com/pytorch/pytorch/pull/181643 lands.
    """
    state, group_rebind_infos = _prepare_optimizer_reparametrization(
        optimizer, parameters_and_buffers, optimizer_state_dict
    )

    original_state = optimizer.state
    original_group_params = [group["params"] for group in optimizer.param_groups]

    try:
        rebind_state: defaultdict[torch.Tensor, Any] = defaultdict(dict)

        for group, saved_group, rebind_params, _ in group_rebind_infos:
            # Rebind the live optimizer group to the explicit tensors and saved
            # group metadata for the trace region.
            group["params"] = rebind_params
            for key, value in saved_group.items():
                if key == "params":
                    continue
                group[key] = value

            for rebind_param, param_id in zip(
                group["params"], saved_group["params"], strict=True
            ):
                # Re-key per-parameter optimizer state from packed ids to the
                # rebound parameter tensors. Shallow-copy the per-param dict so
                # tensor values are shared (in-place ops propagate) but
                # structural mutations during the trace stay local.
                rebind_state[rebind_param] = dict(state.get(param_id, {}))

        optimizer.state = rebind_state
        yield
    finally:
        # Restore the original live optimizer object exactly.
        for group, params in zip(
            optimizer.param_groups, original_group_params, strict=True
        ):
            group["params"] = params
        for group, _, _, saved_values in group_rebind_infos:
            for key, value in saved_values.items():
                group[key] = value
        optimizer.state = original_state


@dataclass
class TracedResult:
    """Execution metadata returned by :func:`minimal_fx_tracer`.

    Attributes:
        gm: The traced FX graph as a pure function of flat tensors.
        example_inputs: Trace-time fake flat inputs used by downstream graph passes.
        state_fqns: Trace-time module parameter/buffer FQNs.
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
        training steps. Each may expand to multiple plain tensors after
        subclass unwrapping (e.g. DTensor -> inner tensors). Optimizer state
        tensors (when an optimizer is passed to ``minimal_fx_tracer``) are
        also stable across steps but are not included here, so cudagraph
        copies them once per step rather than reusing stable addresses.
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
) -> Callable[..., TracedResult]:
    """Return a tracer that captures ``fn`` with implicit module/optimizer state.

    The returned callable takes only the user-facing ``*args`` for ``fn``;
    module parameters/buffers and optimizer state are extracted from the live
    objects and threaded through the graph as static inputs::

        # Stateless function: no module, no optimizer.
        traced = minimal_fx_tracer(fn)(*args)

        # Module-only: parameters/buffers extracted from `module` and
        # reparametrized via stateless._reparametrize_module while tracing.
        traced = minimal_fx_tracer(fn, module=model)(*args)

        # Module + optimizer: optimizer state must already be initialized
        # before tracing. The optimizer is reparametrized while tracing so
        # `optimizer.step()` / `zero_grad()` calls inside `fn` are captured.
        traced = minimal_fx_tracer(fn, module=model, optimizer=opt)(*args)

    ``fn`` should reference ``module`` and ``optimizer`` from its enclosing
    closure — passing them explicitly through ``args`` is invalid because
    ``nn.Module`` and ``Optimizer`` instances are not pytree-able.

    The trace-time ``args`` must satisfy these constraints:

    - all pytree leaves must be tensors or make_fx-safe primitives
      (``int``, ``float``, ``bool``, ``str``, ``None``)
    - there must be no ``nn.Module`` instances in ``args``

    Tensor subclasses (for example ``DTensor``) are recursively unwrapped into
    plain tensors for tracing, and the layouts needed to rewrap them are stored
    in the returned :class:`TracedResult`.
    """
    if optimizer is not None and module is None:
        raise ValueError(
            "minimal_fx_tracer: when 'optimizer' is provided, 'module' must "
            "also be provided so optimizer parameters align with the module's "
            "parameters."
        )

    def _trace_with_args(*args: Any) -> TracedResult:
        model_state, optim_state = extract_train_state(module, optimizer)
        state_fqns = list(model_state.keys())

        # Flatten state and user args into a single tensor list. The state
        # pytree spec lets us reconstruct the dicts inside the trace function
        # so we can hand them to the reparametrize context managers. When no
        # optimizer is provided, only flatten model_state — pytree treats
        # ``None`` as a leaf, which would otherwise emit a non-tensor graph
        # input that the cudagraph wrapper rejects.
        state_tree: Any = (
            model_state if optim_state is None else (model_state, optim_state)
        )
        state_flat, state_spec = pytree.tree_flatten(state_tree)
        num_state_inputs = len(state_flat)

        user_args = list(args)
        user_args_flat, user_args_spec = pytree.tree_flatten(user_args)

        # Validate leaves.
        for leaf in [*state_flat, *user_args_flat]:
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
            state_wrapped = wrapped[:num_state_inputs]
            user_flat = wrapped[num_state_inputs:]

            state_t = pytree.tree_unflatten(list(state_wrapped), state_spec)
            if optimizer is None:
                model_state_t, optim_state_t = state_t, None
            else:
                model_state_t, optim_state_t = state_t
            user_list = pytree.tree_unflatten(list(user_flat), user_args_spec)

            with contextlib.ExitStack() as stack:
                if module is not None:
                    stack.enter_context(
                        stateless._reparametrize_module(module, model_state_t)
                    )
                if optimizer is not None:
                    # _reparametrize_optimizer aligns parameters_and_buffers
                    # values with optimizer.param_groups by order, so pass only
                    # parameters (in module.named_parameters() order).
                    params_for_optim = {
                        name: model_state_t[name]
                        for name, _ in module.named_parameters(remove_duplicate=False)
                    }
                    stack.enter_context(
                        _reparametrize_optimizer(
                            optimizer, params_for_optim, optim_state_t
                        )
                    )
                stack.enter_context(torch.compiler._patch_engine_backward())
                result = fn(*user_list)

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
    *args: Any,
    module: nn.Module | None = None,
    optimizer: "torch.optim.Optimizer | None" = None,
    validate_module_fqns: bool = False,
) -> Any:
    """Execute a traced graph against the live module/optimizer state.

    Mirrors :func:`minimal_fx_tracer`'s state extraction: parameters/buffers
    are sampled from ``module`` and the optimizer state is sampled from
    ``optimizer.state_dict()``. Runs under ``torch.no_grad()`` because the
    graph already contains explicit backward ops (from ``torch.autograd.grad``
    traced by make_fx). Without this, PyTorch would build a redundant autograd
    graph on top, keeping all forward intermediates alive via ``grad_fn``
    references.
    """
    if optimizer is not None and module is None:
        raise ValueError(
            "run_traced: when 'optimizer' is provided, 'module' must also be "
            "provided so optimizer parameters align with the module's parameters."
        )

    model_state, optim_state = extract_train_state(module, optimizer)
    if validate_module_fqns and list(model_state.keys()) != traced_result.state_fqns:
        raise ValueError(
            "module has different parameter/buffer names than during tracing.\n"
            f"  Traced: {traced_result.state_fqns}\n"
            f"  Got:    {list(model_state.keys())}"
        )
    # Mirror the trace-time state-tree shape: skip optim_state when no
    # optimizer was used, otherwise pytree would emit None as a leaf.
    state_tree: Any = model_state if optim_state is None else (model_state, optim_state)
    state_flat, _ = pytree.tree_flatten(state_tree)

    user_args_flat, _ = pytree.tree_flatten(list(args))
    if any(isinstance(leaf, nn.Module) for leaf in [*state_flat, *user_args_flat]):
        raise ValueError(
            "run_traced requires explicit tensor state, not nn.Module instances. "
            "Capture nn.Modules in fn's closure or pass them via the 'module' kwarg."
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
