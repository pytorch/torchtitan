# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CODA-style FlexGEMM epilogue fusion passes for joint training graphs.

The pass runs after forward/backward tracing, so it can optimize forward and
backward boundaries without relying on FlexGEMM autograd support. It recognizes
the pattern families documented by the CODA investigation and rewrites the
GEMM-rooted portions as ``torch.ops.higher_order.flex_gemm`` calls. Match
eligibility depends only on graph structure and tensor metadata, never module
names or source locations.
"""

from __future__ import annotations

import contextvars

import copy
import functools
import operator
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import gcd, prod
from typing import Any

import torch
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch._higher_order_ops.flex_gemm import (
    flex_gemm_hop,
    mark_flex_gemm_body_gemm_node,
)
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.utils.fuser_utils import (
    fuse_as_graphmodule,
    fuse_by_partitions,
    validate_partition,
)

from torchtitan.experiments.graph_trainer.compile_time_benchmark import (
    apply_benchmarked_rewrites,
    BenchmarkCandidateSelection,
    make_rewrite_benchmark_region,
    RewriteBenchmarkRegion,
)
from torchtitan.tools.logging import logger


aten = torch.ops.aten

RMSNORM_FORWARD_GROUP = 64
RMSNORM_BACKWARD_GROUP = 128
_CODA_INDUCTOR_REGION = "coda_flex_gemm"
_CODA_AUTOTUNE = contextvars.ContextVar("coda_autotune", default=True)

_VIEW_TARGETS = {
    aten.alias.default,
    aten.reshape.default,
    aten.t.default,
    aten.transpose.int,
    aten.view.default,
    aten._unsafe_view.default,
}
_RESHAPE_TARGETS = {
    aten.alias.default,
    aten.reshape.default,
    aten.view.default,
    aten._unsafe_view.default,
}
_CAST_TARGET = aten._to_copy.default
_MM_TARGET = aten.mm.default
_BMM_TARGET = aten.bmm.default
_POINTWISE_TARGETS = {
    aten.add.Scalar,
    aten.add.Tensor,
    aten.expand.default,
    aten.div.Scalar,
    aten.div.Tensor,
    aten.mul.Scalar,
    aten.mul.Tensor,
    aten.neg.default,
    aten.pow.Tensor_Scalar,
    aten.rsub.Scalar,
    aten.sigmoid.default,
    aten.sigmoid_backward.default,
    aten.silu.default,
    aten.silu_backward.default,
    aten.sub.Scalar,
    aten.sub.Tensor,
    aten.tanh.default,
    aten.tanh_backward.default,
    _CAST_TARGET,
    *_VIEW_TARGETS,
}


@dataclass(frozen=True)
class _MatmulChain:
    root: Node
    nodes: tuple[Node, ...]


@dataclass(frozen=True)
class _MatmulUse:
    node: Node
    operand: int
    wrappers: tuple[Node, ...]


@dataclass(frozen=True)
class _SwiGLUMatch:
    gate: _MatmulChain
    silu: Node
    up: _MatmulChain
    output: Node


@dataclass(frozen=True)
class _SiTUMatch:
    gate: _MatmulChain
    gate_nodes: frozenset[Node]
    up: _MatmulChain
    up_nodes: frozenset[Node]
    output: Node


@dataclass(frozen=True)
class _PrimitiveRMSNorm:
    output: Node
    normalized_cast: Node
    norm_input: Node
    input_cast: Node
    rstd: Node
    weight: Node
    eps: float


CodaMatcher = Callable[
    [GraphModule, Counter[str], list[RewriteBenchmarkRegion] | None], None
]


@dataclass(frozen=True)
class CodaPattern:
    name: str
    matcher: CodaMatcher
    priority: int
    # Keys are CUDA compute-capability major versions (10 means SM100). Configs
    # follow the FlexGEMM emission order for patterns that produce multiple HOPs.
    best_configs: dict[int, tuple[dict[str, Any], ...]]
    tune_split_k: bool = False


_CODA_PATTERNS: dict[str, CodaPattern] = {}
_CODA_MATCH_SELECTION = contextvars.ContextVar[BenchmarkCandidateSelection | None](
    "coda_match_selection", default=None
)


def register_coda_pattern(
    *,
    best_configs: dict[int, tuple[dict[str, Any], ...]] | None = None,
    tune_split_k: bool = False,
) -> Callable[[CodaMatcher], CodaMatcher]:
    """Register one matcher and optional device-specific FlexGEMM configs."""

    def register(matcher: CodaMatcher) -> CodaMatcher:
        name = matcher.__name__
        if name in _CODA_PATTERNS:
            raise ValueError(f"CODA pattern {name!r} is already registered")
        _CODA_PATTERNS[name] = CodaPattern(
            name,
            matcher,
            len(_CODA_PATTERNS),
            {} if best_configs is None else best_configs,
            tune_split_k,
        )
        return matcher

    return register


def _quack_config(
    tile_m: int,
    tile_n: int,
    *,
    dynamic: bool,
    cluster_m: int = 2,
    cluster_n: int = 1,
    swap_ab: bool = False,
) -> dict[str, Any]:
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": None,
        "num_warps": None,
        "pingpong": False,
        "is_dynamic_persistent": dynamic,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "cluster_k": 1,
        "swap_ab": swap_ab,
        "max_swizzle_size": 8,
        "device_capacity": 10,
        "use_tma_gather": False,
    }


def _best_configs(pattern: str) -> tuple[dict[str, Any], ...]:
    configs = _CODA_PATTERNS[pattern].best_configs
    if not configs or not torch.cuda.is_available():
        return ()
    device_capacity = torch.cuda.get_device_capability()[0]
    if device_capacity == 11:
        device_capacity = 10
    return configs.get(device_capacity, ())


def _kernel_options(
    pattern: str,
    *,
    config_index: int = 0,
    fast_math: bool = False,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "backend": "QUACK",
        "tuned": _CODA_AUTOTUNE.get(),
    }
    if _CODA_PATTERNS[pattern].tune_split_k and options["tuned"]:
        options["tune_split_k"] = True
    configs = _best_configs(pattern)
    if configs:
        index = 0 if len(configs) == 1 else config_index
        if index < len(configs):
            options["tuned"] = False
            options["config"] = dict(configs[index])
    if fast_math:
        options["fast_math"] = True
    return options


def _is_backward(node: Node) -> bool:
    return bool(node.meta.get("autograd_backward"))


def _claim_coda_match(pattern: str, anchor: Node) -> bool:
    selection = _CODA_MATCH_SELECTION.get()
    if selection is None:
        return True
    match_id = f"{pattern}:{anchor.name}"
    if selection.accepted is not None:
        selection.selected = match_id
        return match_id in selection.accepted
    if match_id in selection.rejected:
        return False
    if selection.collect_all:
        selection.selected = match_id
        selection.candidates.append(match_id)
        return True
    if selection.selected is None:
        selection.selected = match_id
    return selection.selected == match_id


def _path_has_phase(path: _MatmulChain, *, backward: bool) -> bool:
    return all(_is_backward(node) is backward for node in path.nodes)


def _tensor_value(node: Node) -> torch.Tensor | None:
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _shape(node: Node) -> tuple[Any, ...] | None:
    value = _tensor_value(node)
    return tuple(value.shape) if value is not None else None


def _static_numel(shape: Sequence[Any]) -> int | None:
    result = 1
    for dim in shape:
        if not isinstance(dim, int):
            return None
        result *= dim
    return result


def _same_numel(lhs: Sequence[Any], rhs: Sequence[Any]) -> bool:
    lhs_static = _static_numel(lhs)
    rhs_static = _static_numel(rhs)
    if lhs_static is not None and rhs_static is not None:
        return lhs_static == rhs_static
    return statically_known_true(prod(lhs) == prod(rhs))


def _dtype(node: Node) -> torch.dtype | None:
    value = _tensor_value(node)
    return value.dtype if value is not None else None


def _single_tensor_input(node: Node) -> Node | None:
    inputs = [arg for arg in node.all_input_nodes]
    return inputs[0] if len(inputs) == 1 else None


def _alias_source(node: Node) -> Node:
    while node.target is aten.alias.default:
        source = _single_tensor_input(node)
        if source is None:
            break
        node = source
    return node


def _other_node_input(node: Node, known: Node) -> Node | None:
    inputs = list(node.all_input_nodes)
    if len(inputs) != 2 or known not in inputs:
        return None
    return inputs[1] if inputs[0] is known else inputs[0]


def _chain_to_mm(node: object, *, allow_cast: bool = True) -> _MatmulChain | None:
    if not isinstance(node, Node):
        return None
    reverse_path: list[Node] = []
    current = node
    while current.target in _VIEW_TARGETS or (
        allow_cast and current.target is _CAST_TARGET
    ):
        reverse_path.append(current)
        current = _single_tensor_input(current)
        if current is None:
            return None
    if current.target is not _MM_TARGET:
        return None
    return _MatmulChain(current, tuple([current, *reversed(reverse_path)]))


def _find_mm_path(node: Node) -> _MatmulChain | None:
    direct = _chain_to_mm(node)
    if direct is not None:
        return direct
    frontier: list[tuple[Node, tuple[Node, ...]]] = [(node, ())]
    visited: set[Node] = set()
    while frontier:
        current, outer_adds = frontier.pop(0)
        if current in visited or current.target is not aten.add.Tensor:
            continue
        visited.add(current)
        for arg in current.all_input_nodes:
            path = _chain_to_mm(arg)
            if path is not None:
                return _MatmulChain(
                    path.root,
                    (*path.nodes, current, *reversed(outer_adds)),
                )
        frontier.extend(
            (arg, (*outer_adds, current)) for arg in current.all_input_nodes
        )
    return None


def _is_reshape_only_path(path: _MatmulChain) -> bool:
    return all(
        node is path.root
        or node.target in _RESHAPE_TARGETS
        or node.target is aten.add.Tensor
        for node in path.nodes
    )


def _has_exact_add_shapes(path: _MatmulChain) -> bool:
    for node in path.nodes:
        if node.target is not aten.add.Tensor:
            continue
        inputs = [arg for arg in node.args[:2] if isinstance(arg, Node)]
        output_shape = _shape(node)
        if len(inputs) != 2 or output_shape is None:
            return False
        if any(_shape(arg) != output_shape for arg in inputs):
            return False
    return True


def _ordered_nodes(gm: GraphModule, nodes: Iterable[Node]) -> list[Node]:
    del gm
    return sorted(set(nodes))


def _coda_nodes_available(nodes: Iterable[Node]) -> bool:
    return all(not node.meta.get("coda_consumed") for node in nodes)


def _mark_coda_owned(pattern: str, nodes: Iterable[Node]) -> None:
    for node in nodes:
        owner = node.meta.get("coda_owner")
        if owner is not None and owner != pattern:
            raise AssertionError(
                f"CODA node {node.name} is already owned by pattern {owner}"
            )
        node.meta["coda_consumed"] = True
        node.meta["coda_owner"] = pattern


def _match_primitive_rmsnorm(output: Node) -> _PrimitiveRMSNorm | None:
    if output.target is not aten.mul.Tensor or _is_backward(output):
        return None
    normalized_cast = next(
        (arg for arg in output.all_input_nodes if arg.target is _CAST_TARGET),
        None,
    )
    if normalized_cast is None:
        return None
    weight = _other_node_input(output, normalized_cast)
    normalized = _single_tensor_input(normalized_cast)
    if weight is None or normalized is None or normalized.target is not aten.mul.Tensor:
        return None
    input_cast = next(
        (
            arg
            for arg in normalized.all_input_nodes
            if arg.target is _CAST_TARGET and _cast_dtype(arg) is torch.float32
        ),
        None,
    )
    if input_cast is None:
        return None
    rstd = _other_node_input(normalized, input_cast)
    norm_input = _single_tensor_input(input_cast)
    if rstd is None or rstd.target is not aten.rsqrt.default or norm_input is None:
        return None
    variance = _single_tensor_input(rstd)
    if variance is None or variance.target not in {aten.add.Scalar, aten.add.Tensor}:
        return None
    mean = next(
        (arg for arg in variance.all_input_nodes if arg.target is aten.mean.dim),
        None,
    )
    if mean is None:
        return None
    if mean.args[1] != [-1] or mean.args[2] is not True:
        return None
    if variance.kwargs.get("alpha", 1) != 1:
        return None
    squared = mean.args[0]
    if (
        not isinstance(squared, Node)
        or squared.target is not aten.pow.Tensor_Scalar
        or squared.args[0] is not input_cast
        or squared.args[1] != 2
    ):
        return None
    eps = next(
        (arg for arg in variance.args if isinstance(arg, (float, int))),
        None,
    )
    if eps is None:
        return None
    norm_shape = _shape(norm_input)
    if (
        norm_shape is None
        or _shape(output) != norm_shape
        or _shape(weight) != (norm_shape[-1],)
        or _dtype(normalized_cast) != _dtype(norm_input)
    ):
        return None
    return _PrimitiveRMSNorm(
        output=output,
        normalized_cast=normalized_cast,
        norm_input=norm_input,
        input_cast=input_cast,
        rstd=rstd,
        weight=weight,
        eps=float(eps),
    )


def _primitive_rmsnorm_path(match: _PrimitiveRMSNorm) -> _MatmulChain | None:
    path = _chain_to_mm(match.norm_input, allow_cast=False)
    if path is None and match.norm_input.target is operator.getitem:
        split = match.norm_input.args[0]
        if (
            isinstance(split, Node)
            and split.target is aten.split_with_sizes.default
            and match.norm_input.args[1] == 0
            and isinstance(split.args[0], Node)
        ):
            path = _chain_to_mm(split.args[0], allow_cast=False)
    return path if path is not None else _find_mm_path(match.norm_input)


def _single_user_with_target(node: Node, target: Any) -> Node | None:
    matches = [user for user in node.users if user.target is target]
    return matches[0] if len(matches) == 1 else None


def _canonicalize_primitive_rmsnorm_backward(
    gm: GraphModule, match: _PrimitiveRMSNorm
) -> None:
    norm_shape = _shape(match.norm_input)
    weight_shape = _shape(match.weight)
    if norm_shape is None or weight_shape is None:
        return
    for grad_weight_mul in list(match.normalized_cast.users):
        if (
            not _is_backward(grad_weight_mul)
            or grad_weight_mul.target is not aten.mul.Tensor
        ):
            continue
        grad = _other_node_input(grad_weight_mul, match.normalized_cast)
        if grad is None:
            continue
        path = _chain_to_mm(grad, allow_cast=False)
        if (
            path is None
            or not _is_reshape_only_path(path)
            or not _path_has_phase(path, backward=False)
        ):
            continue
        grad_weighted = next(
            (
                user
                for user in grad.users
                if _is_backward(user)
                and user.target is aten.mul.Tensor
                and match.weight in user.all_input_nodes
            ),
            None,
        )
        if grad_weighted is None:
            continue
        grad_float = _single_user_with_target(grad_weighted, _CAST_TARGET)
        if grad_float is None or _cast_dtype(grad_float) is not torch.float32:
            continue
        direct = next(
            (
                user
                for user in grad_float.users
                if user.target is aten.mul.Tensor and match.rstd in user.all_input_nodes
            ),
            None,
        )
        if direct is None:
            continue
        grad_add = _single_user_with_target(direct, aten.add.Tensor)
        if grad_add is None:
            continue
        old_grad_input = _single_user_with_target(grad_add, _CAST_TARGET)
        if (
            old_grad_input is None
            or _shape(old_grad_input) != norm_shape
            or _dtype(old_grad_input) != _dtype(match.norm_input)
        ):
            continue

        grad_weight_sum = _single_user_with_target(
            grad_weight_mul, aten.sum.dim_IntList
        )
        if grad_weight_sum is None:
            continue
        old_grad_weight = grad_weight_sum
        while _shape(old_grad_weight) != weight_shape:
            view_users = [
                user for user in old_grad_weight.users if user.target in _VIEW_TARGETS
            ]
            if len(view_users) != 1:
                old_grad_weight = None
                break
            old_grad_weight = view_users[0]
        if old_grad_weight is None:
            continue
        width = norm_shape[-1]
        if not isinstance(width, int):
            continue
        cursor = max((grad, match.norm_input, match.rstd, match.weight))
        with gm.graph.inserting_after(cursor):
            fused_backward = gm.graph.call_function(
                aten._fused_rms_norm_backward.default,
                args=(
                    grad,
                    match.norm_input,
                    [width],
                    match.rstd,
                    match.weight,
                    [True, True],
                ),
            )
        _copy_meta(fused_backward, old_grad_input)
        fused_backward.meta["val"] = (
            old_grad_input.meta.get("val"),
            old_grad_weight.meta.get("val"),
        )
        new_grad_input = _call_after(
            gm.graph, fused_backward, operator.getitem, (fused_backward, 0)
        )
        _copy_meta(new_grad_input, old_grad_input)
        new_grad_weight = _call_after(
            gm.graph, new_grad_input, operator.getitem, (fused_backward, 1)
        )
        _copy_meta(new_grad_weight, old_grad_weight)
        old_grad_input.replace_all_uses_with(new_grad_input)
        old_grad_weight.replace_all_uses_with(new_grad_weight)
        return


def _canonicalize_primitive_rmsnorm(gm: GraphModule) -> None:
    r"""_canonicalize_primitive_rmsnorm(gm) -> None

    Replace decomposed RMSNorm forward and backward graphs with ATen operators.

    Forward before::

        x_fp32 = aten._to_copy(x, dtype=torch.float32)
        mean_square = aten.mean(aten.pow(x_fp32, 2), [-1], True)
        rstd = aten.rsqrt(aten.add(mean_square, eps))
        normalized = aten._to_copy(aten.mul(x_fp32, rstd), dtype=x.dtype)
        output = aten.mul(normalized, weight)

    Forward after::

        output, rstd = aten._fused_rms_norm(x, [width], weight, eps)

    When the corresponding primitive backward is present, its pointwise and
    reduction graph is similarly replaced with::

        grad_input, grad_weight = aten._fused_rms_norm_backward(
            grad, x, [width], rstd, weight, [True, True]
        )

    This gives the CODA matchers one stable RMSNorm representation while
    preserving the original output and ``rstd`` users.

    Args:
        gm (GraphModule): graph to canonicalize in place.
    """
    matches = [
        match
        for node in list(gm.graph.nodes)
        if (match := _match_primitive_rmsnorm(node)) is not None
        and (path := _primitive_rmsnorm_path(match)) is not None
        and _is_reshape_only_path(path)
        and _downstream_mm(match.output, backward=False) is not None
    ]
    for match in matches:
        _canonicalize_primitive_rmsnorm_backward(gm, match)
        norm_shape = _shape(match.norm_input)
        if norm_shape is None or not isinstance(norm_shape[-1], int):
            continue
        width = norm_shape[-1]
        cursor = max((match.norm_input, match.weight))
        with gm.graph.inserting_after(cursor):
            fused = gm.graph.call_function(
                aten._fused_rms_norm.default,
                args=(match.norm_input, [width], match.weight, match.eps),
            )
        _copy_meta(fused, match.output)
        fused.meta["val"] = (
            match.output.meta.get("val"),
            match.rstd.meta.get("val"),
        )
        new_output = _call_after(gm.graph, fused, operator.getitem, (fused, 0))
        _copy_meta(new_output, match.output)
        new_rstd = _call_after(gm.graph, new_output, operator.getitem, (fused, 1))
        _copy_meta(new_rstd, match.rstd)
        match.output.replace_all_uses_with(new_output)
        match.rstd.replace_all_uses_with(new_rstd)


def _boundary_outputs(body_nodes: Sequence[Node]) -> list[Node]:
    body_set = set(body_nodes)
    return [
        node for node in body_nodes if any(user not in body_set for user in node.users)
    ]


def _unique_submodule_name(gm: GraphModule, prefix: str) -> str:
    index = 0
    while hasattr(gm, f"{prefix}_{index}"):
        index += 1
    return f"{prefix}_{index}"


def _copy_meta(dst: Node, src: Node) -> None:
    dst.meta = src.meta.copy()
    custom = dst.meta.get("custom")
    if isinstance(custom, dict):
        dst.meta["custom"] = custom.copy()


def _call_after(
    graph: torch.fx.Graph,
    cursor: Node,
    target: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> Node:
    with graph.inserting_after(cursor):
        return graph.call_function(target, args=args, kwargs=kwargs or {})


def _set_view_meta(node: Node, source: Node, shape: Sequence[Any]) -> None:
    value = _tensor_value(source)
    if value is not None:
        node.meta["val"] = aten.view.default(value, list(shape))


def _set_empty_meta(
    node: Node,
    source: Node,
    shape: Sequence[Any],
    *,
    dtype: torch.dtype,
) -> None:
    value = _tensor_value(source)
    if value is not None:
        node.meta["val"] = value.new_empty(tuple(shape), dtype=dtype)


def _view_region_input(
    gm: GraphModule,
    node: Node,
    shape: Sequence[Any],
    existing_nodes: set[Node],
) -> Node:
    node_shape = _shape(node)
    if node_shape == tuple(shape):
        return node
    value = _tensor_value(node)
    if value is None:
        raise AssertionError(f"CODA input {node.name} requires tensor metadata")
    try:
        viewed_value = aten.view.default(value, list(shape))
    except RuntimeError as error:
        raise AssertionError(
            f"CODA input {node.name} cannot be viewed as {tuple(shape)}"
        ) from error
    reshaped = _call_after(
        gm.graph,
        node,
        aten.view.default,
        (node, list(shape)),
    )
    reshaped.meta["val"] = viewed_value
    existing_nodes.add(reshaped)
    return reshaped


def _register_body(gm: GraphModule, body: GraphModule, pattern: str) -> str:
    name = _unique_submodule_name(gm, f"coda_{pattern.lower()}_body")
    gm.add_submodule(name, body)
    return name


def _propagate_body_meta(body: GraphModule, inputs: Sequence[Node]) -> None:
    values = [node.meta.get("val") for node in inputs]
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise AssertionError("CODA FlexGEMM body inputs require tensor metadata")
    fake_modes = {value.fake_mode for value in values if isinstance(value, FakeTensor)}
    if len(fake_modes) == 1 and all(isinstance(value, FakeTensor) for value in values):
        fake_mode = next(iter(fake_modes))
        FakeTensorProp(body, mode=fake_mode).propagate_dont_convert_inputs(*values)
        return
    if not fake_modes:
        FakeTensorProp(body).propagate(*values)
        return

    fake_mode = FakeTensorMode()
    normalized_values = []
    for value in values:
        meta_value = torch.empty_strided(
            value.shape,
            value.stride(),
            dtype=value.dtype,
            device="meta",
        )
        normalized_values.append(
            fake_mode.fake_tensor_converter.from_meta_and_device(
                fake_mode, meta_value, value.device
            )
        )
    FakeTensorProp(body, mode=fake_mode).propagate_dont_convert_inputs(
        *normalized_values
    )


def _body_output_values(body: GraphModule, pattern: str) -> tuple[torch.Tensor, ...]:
    output = next(node for node in body.graph.nodes if node.op == "output")
    result_nodes = output.args[0]
    if not isinstance(result_nodes, (list, tuple)):
        result_nodes = (result_nodes,)
    values = tuple(
        _tensor_value(node) if isinstance(node, Node) else None for node in result_nodes
    )
    if not all(value is not None for value in values):
        raise AssertionError(f"CODA {pattern} body outputs require tensor metadata")
    return tuple(value for value in values if isinstance(value, torch.Tensor))


def _validate_body_outputs(
    body: GraphModule,
    expected_values: Sequence[torch.Tensor | None],
    pattern: str,
) -> tuple[torch.Tensor, ...]:
    actual_values = _body_output_values(body, pattern)
    if len(actual_values) != len(expected_values):
        raise AssertionError(
            f"CODA {pattern} body returns {len(actual_values)} outputs, "
            f"expected {len(expected_values)}"
        )
    for index, (actual, expected) in enumerate(
        zip(actual_values, expected_values, strict=True)
    ):
        if expected is None:
            raise AssertionError(
                f"CODA {pattern} output {index} requires tensor metadata"
            )
        actual_spec = (
            tuple(actual.shape),
            tuple(actual.stride()),
            actual.dtype,
            actual.device,
        )
        expected_spec = (
            tuple(expected.shape),
            tuple(expected.stride()),
            expected.dtype,
            expected.device,
        )
        if actual_spec != expected_spec:
            raise AssertionError(
                f"CODA {pattern} output {index} has spec {actual_spec}, "
                f"expected {expected_spec}"
            )
    return actual_values


def _flex_gemm_body_has_aliasing(
    inputs: Sequence[Node], outputs: Sequence[torch.Tensor]
) -> bool:
    input_values = [
        value for node in inputs if (value := _tensor_value(node)) is not None
    ]
    if len(input_values) != len(inputs):
        return True
    if any(
        torch._C._is_alias_of(input_value, other)
        for index, input_value in enumerate(input_values)
        for other in input_values[index + 1 :]
    ):
        return True
    if any(
        torch._C._is_alias_of(output, input_value)
        for output in outputs
        for input_value in input_values
    ):
        return True
    return any(
        torch._C._is_alias_of(output, other)
        for index, output in enumerate(outputs)
        for other in outputs[index + 1 :]
    )


def _make_flex_gemm_benchmark_region(
    body: GraphModule,
    inputs: Sequence[Node],
    outputs: Sequence[torch.Tensor],
    options: dict[str, Any],
) -> RewriteBenchmarkRegion:
    baseline = copy.deepcopy(body)
    candidate_body = copy.deepcopy(body)
    root = torch.nn.Module()
    root.add_module("body", candidate_body)
    graph = torch.fx.Graph()
    candidate_inputs = []
    baseline_inputs = [
        node for node in baseline.graph.nodes if node.op == "placeholder"
    ]
    for baseline_input, source in zip(baseline_inputs, inputs, strict=True):
        candidate_input = graph.placeholder(baseline_input.name)
        _copy_meta(candidate_input, source)
        candidate_inputs.append(candidate_input)
    body_attr = graph.get_attr("body")
    fused = graph.call_function(
        flex_gemm_hop,
        args=(_MM_TARGET, body_attr, tuple(candidate_inputs), {}, options),
    )
    fused.meta["val"] = tuple(outputs)
    candidate_outputs = []
    for index, value in enumerate(outputs):
        output = graph.call_function(operator.getitem, args=(fused, index))
        output.meta["val"] = value
        candidate_outputs.append(output)
    graph.output(tuple(candidate_outputs))
    return make_rewrite_benchmark_region(baseline, GraphModule(root, graph))


def _mark_inductor_region(
    body: GraphModule,
    body_attr: Node,
    fused: Node,
    *,
    pattern: str,
    region: str = _CODA_INDUCTOR_REGION,
) -> None:
    group = str(body_attr.target)
    _mark_nodes_for_inductor((body_attr, fused), region=region, group=group)
    fused.meta["custom"]["coda_pattern"] = pattern
    _mark_nodes_for_inductor(body.graph.nodes, region=region, group=group)


def _mark_nodes_for_inductor(
    nodes: Iterable[Node],
    *,
    region: str = _CODA_INDUCTOR_REGION,
    group: str | None = None,
) -> None:
    annotation = {"inductor_region": region}
    for node in nodes:
        custom = node.meta.setdefault("custom", {})
        custom["compile_with_inductor"] = annotation
        if group is not None:
            custom["coda_region_group"] = group


def _coda_region_group(node: Node) -> str | None:
    custom = node.meta.get("custom", {})
    group = custom.get("coda_region_group") if isinstance(custom, dict) else None
    return group if isinstance(group, str) else None


def _assign_coda_inductor_regions(gm: GraphModule) -> None:
    groups: dict[str, list[Node]] = {}
    for node in gm.graph.nodes:
        group = _coda_region_group(node)
        if group is not None:
            groups.setdefault(group, []).append(node)

    ordered_groups = sorted(
        groups,
        key=lambda group: min(groups[group]),
    )
    assignments = {
        group: f"{_CODA_INDUCTOR_REGION}_{index}"
        for index, group in enumerate(ordered_groups)
    }

    for module in gm.modules():
        if not isinstance(module, GraphModule):
            continue
        for node in module.graph.nodes:
            group = _coda_region_group(node)
            if group not in assignments:
                continue
            custom = node.meta["custom"]
            custom["compile_with_inductor"]["inductor_region"] = assignments[group]


def _infer_new_node_meta(node: Node) -> None:
    if node.op != "call_function" or "val" in node.meta:
        return

    def load_value(arg: Node) -> Any:
        if "val" not in arg.meta:
            raise AssertionError(f"CODA input {arg.name} is missing value metadata")
        return arg.meta["val"]

    args, kwargs = torch.fx.node.map_arg((node.args, node.kwargs), load_value)
    if node.target is operator.getitem:
        node.meta["val"] = node.target(*args, **kwargs)
        return

    fake_mode = FakeTensorMode()

    def normalize(value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        device = value.fake_device if isinstance(value, FakeTensor) else value.device
        meta_value = torch.empty_strided(
            value.shape,
            value.stride(),
            dtype=value.dtype,
            device="meta",
        )
        return fake_mode.fake_tensor_converter.from_meta_and_device(
            fake_mode, meta_value, device
        )

    args, kwargs = torch.utils._pytree.tree_map(normalize, (args, kwargs))
    with fake_mode:
        node.meta["val"] = node.target(*args, **kwargs)


def _mark_new_nodes_for_inductor(
    gm: GraphModule,
    existing_nodes: set[Node],
    *,
    region: str = _CODA_INDUCTOR_REGION,
    group: str | None = None,
) -> None:
    new_nodes = [node for node in gm.graph.nodes if node not in existing_nodes]
    for node in new_nodes:
        _infer_new_node_meta(node)
        custom = node.meta.setdefault("custom", {})
        custom.setdefault(
            "compile_with_inductor",
            {"inductor_region": region},
        )
        if group is not None:
            custom.setdefault("coda_region_group", group)


def _insert_flex_gemm(
    gm: GraphModule,
    *,
    root: Node,
    body_nodes: Iterable[Node],
    pattern: str,
    config_index: int = 0,
    fast_math: bool = False,
    fused_outputs: Iterable[Node] | None = None,
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> Node | None:
    inputs = root.all_input_nodes
    if len(inputs) < 2:
        return None
    lhs_dtype = _dtype(inputs[0])
    rhs_dtype = _dtype(inputs[1])
    if lhs_dtype != rhs_dtype or lhs_dtype not in {
        torch.bfloat16,
        torch.float16,
    }:
        return None

    ordered = _ordered_nodes(gm, body_nodes)
    if not _coda_nodes_available(ordered):
        return None
    if not validate_partition(ordered):
        return None
    body_set = set(ordered)
    outputs = (
        _ordered_nodes(gm, fused_outputs)
        if fused_outputs is not None
        else _boundary_outputs(ordered)
    )
    if not outputs:
        raise AssertionError(f"CODA {pattern} has no externally visible output")

    external_inputs: list[Node] = []
    for arg in root.all_input_nodes:
        if arg not in external_inputs:
            external_inputs.append(arg)
    for node in ordered:
        for arg in node.all_input_nodes:
            if arg not in body_set and arg not in external_inputs:
                external_inputs.append(arg)

    root_shape = _shape(root)
    if root_shape is None:
        raise AssertionError(f"CODA {pattern} requires GEMM shape metadata")
    actual_inputs = external_inputs

    body_graph = torch.fx.Graph()
    env: dict[Node, Node] = {}
    for index, (external, actual) in enumerate(zip(external_inputs, actual_inputs)):
        placeholder = body_graph.placeholder(f"arg{index}")
        _copy_meta(placeholder, actual)
        env[external] = placeholder
    for node in ordered:
        copied = body_graph.node_copy(node, lambda old: env[old])
        _copy_meta(copied, node)
        env[node] = copied
    outputs.sort(key=lambda node: not _same_numel(_shape(node) or (), root_shape))
    body_results: list[Node] = []
    restore_shapes: list[tuple[Any, ...] | None] = []
    for output in outputs:
        output_shape = _shape(output)
        if (
            output_shape is not None
            and output_shape != root_shape
            and _same_numel(output_shape, root_shape)
        ):
            body_results.append(
                body_graph.call_function(
                    aten.view.default, args=(env[output], list(root_shape))
                )
            )
            restore_shapes.append(output_shape)
        else:
            body_results.append(env[output])
            restore_shapes.append(None)
    body_graph.output(tuple(body_results))
    body = GraphModule(torch.nn.Module(), body_graph)
    _propagate_body_meta(body, actual_inputs)
    physical_values = []
    for output, restore_shape in zip(outputs, restore_shapes, strict=True):
        value = _tensor_value(output)
        if value is not None and restore_shape is not None:
            value = aten.view.default(value, list(root_shape))
        physical_values.append(value)
    body_values = _validate_body_outputs(body, physical_values, pattern)
    if _flex_gemm_body_has_aliasing(actual_inputs, body_values):
        return None
    mark_flex_gemm_body_gemm_node(body, _MM_TARGET)
    options = _kernel_options(
        pattern,
        config_index=config_index,
        fast_math=fast_math,
    )
    body_name = _register_body(gm, body, pattern)
    region = _CODA_INDUCTOR_REGION

    latest_input = max(actual_inputs)
    with gm.graph.inserting_after(latest_input):
        body_attr = gm.graph.get_attr(body_name)
    with gm.graph.inserting_after(body_attr):
        fused = gm.graph.call_function(
            flex_gemm_hop,
            args=(
                _MM_TARGET,
                body_attr,
                tuple(actual_inputs),
                {},
                options,
            ),
        )
    fused.meta["val"] = body_values
    _mark_inductor_region(
        body,
        body_attr,
        fused,
        pattern=pattern,
        region=region,
    )

    replacements: dict[Node, Node] = {}
    new_region_nodes: list[Node] = []
    cursor = fused
    for index, (output, restore_shape) in enumerate(
        zip(outputs, restore_shapes, strict=True)
    ):
        with gm.graph.inserting_after(cursor):
            replacement = gm.graph.call_function(operator.getitem, args=(fused, index))
        _copy_meta(replacement, output)
        replacement.meta["val"] = physical_values[index]
        replacement.meta.setdefault("custom", {})["coda_pattern"] = pattern
        new_region_nodes.append(replacement)
        cursor = replacement
        if restore_shape is not None:
            replacement = _call_after(
                gm.graph,
                cursor,
                aten.view.default,
                (replacement, list(restore_shape)),
            )
            _copy_meta(replacement, output)
            new_region_nodes.append(replacement)
            cursor = replacement
        replacements[output] = replacement

    for old, new in replacements.items():
        for user in list(old.users):
            if user not in body_set:
                user.replace_input_with(old, new)
    _mark_nodes_for_inductor(new_region_nodes, region=region, group=body_name)
    _mark_coda_owned(pattern, ordered)
    if benchmark_regions is not None:
        benchmark_regions.append(
            _make_flex_gemm_benchmark_region(
                body,
                actual_inputs,
                body_values,
                options,
            )
        )
    return fused


def _collect_pointwise_descendants(
    root: Node,
    *,
    allow_other_mm_inputs: bool = False,
    limit: int = 32,
) -> set[Node]:
    selected = {root}
    frontier = list(root.users)
    while frontier and len(selected) < limit:
        node = frontier.pop(0)
        if (
            node in selected
            or node.target not in _POINTWISE_TARGETS
            or _is_backward(node) != _is_backward(root)
        ):
            continue
        if not any(inp in selected for inp in node.all_input_nodes):
            continue
        if node.target in _VIEW_TARGETS and any(
            user.target is _MM_TARGET for user in node.users
        ):
            continue
        external_inputs = [inp for inp in node.all_input_nodes if inp not in selected]
        if not allow_other_mm_inputs and any(
            chain is not None and chain.root is not root
            for inp in external_inputs
            if (chain := _chain_to_mm(inp)) is not None
        ):
            continue
        selected.add(node)
        frontier.extend(node.users)
    return selected


def _match_swiglu(output: Node) -> _SwiGLUMatch | None:
    if output.target is not aten.mul.Tensor or _is_backward(output):
        return None
    for gate_output, up_output in (
        (output.args[0], output.args[1]),
        (output.args[1], output.args[0]),
    ):
        if (
            not isinstance(gate_output, Node)
            or gate_output.target is not aten.silu.default
        ):
            continue
        gate = _chain_to_mm(gate_output.args[0], allow_cast=False)
        up = _chain_to_mm(up_output, allow_cast=False)
        if gate is None or up is None:
            continue
        if any(node.target not in _RESHAPE_TARGETS for node in gate.nodes[1:]):
            continue
        if any(node.target not in _RESHAPE_TARGETS for node in up.nodes[1:]):
            continue
        gate_input = gate.root.args[0]
        up_input = up.root.args[0]
        output_shape = _shape(output)
        if (
            gate_input is not up_input
            or output_shape is None
            or _shape(gate_output) != output_shape
            or _shape(up_output) != output_shape
        ):
            continue
        return _SwiGLUMatch(gate, gate_output, up, output)
    return None


def _scalar_input(node: object, target: Any, scalar: float) -> Node | None:
    if not isinstance(node, Node) or node.target is not target or len(node.args) < 2:
        return None
    lhs, rhs = node.args[:2]
    if isinstance(lhs, Node) and rhs == scalar:
        return lhs
    if isinstance(rhs, Node) and lhs == scalar:
        return rhs
    return None


def _match_situ(output: Node) -> _SiTUMatch | None:
    if output.target is not _CAST_TARGET or _cast_dtype(output) is not torch.bfloat16:
        return None
    product = _single_tensor_input(output)
    if product is None or product.target is not aten.mul.Tensor:
        return None
    for gate_output, up_output in (
        (product.args[0], product.args[1]),
        (product.args[1], product.args[0]),
    ):
        if (
            not isinstance(gate_output, Node)
            or gate_output.target is not aten.mul.Tensor
        ):
            continue
        gate_scaled, gate_sigmoid = gate_output.args[:2]
        if (
            not isinstance(gate_sigmoid, Node)
            or gate_sigmoid.target is not aten.sigmoid.default
        ):
            gate_scaled, gate_sigmoid = gate_sigmoid, gate_scaled
        gate_tanh = _scalar_input(gate_scaled, aten.mul.Tensor, 4.0)
        gate_div = _single_tensor_input(gate_tanh) if gate_tanh is not None else None
        gate_cast = _scalar_input(gate_div, aten.div.Tensor, 4.0)
        if (
            gate_tanh is None
            or gate_tanh.target is not aten.tanh.default
            or gate_cast is None
            or not isinstance(gate_sigmoid, Node)
            or gate_sigmoid.target is not aten.sigmoid.default
            or gate_sigmoid.args[0] is not gate_cast
            or gate_cast.target is not _CAST_TARGET
            or _cast_dtype(gate_cast) is not torch.float32
        ):
            continue
        gate = _chain_to_mm(gate_cast.args[0], allow_cast=False)

        up_tanh = _scalar_input(up_output, aten.mul.Tensor, 25.0)
        up_div = _single_tensor_input(up_tanh) if up_tanh is not None else None
        up_cast = _scalar_input(up_div, aten.div.Tensor, 25.0)
        if (
            up_tanh is None
            or up_tanh.target is not aten.tanh.default
            or up_cast is None
            or up_cast.target is not _CAST_TARGET
            or _cast_dtype(up_cast) is not torch.float32
        ):
            continue
        up = _chain_to_mm(up_cast.args[0], allow_cast=False)
        if gate is None or up is None or gate.root.args[0] is not up.root.args[0]:
            continue
        output_shape = _shape(product)
        if (
            output_shape is None
            or _shape(gate_output) != output_shape
            or _shape(up_output) != output_shape
        ):
            continue
        gate_nodes = frozenset(
            {
                *gate.nodes,
                gate_cast,
                gate_div,
                gate_tanh,
                gate_scaled,
                gate_sigmoid,
                gate_output,
            }
        )
        up_nodes = frozenset(
            {*up.nodes, up_cast, up_div, up_tanh, up_output, product, output}
        )
        return _SiTUMatch(gate, gate_nodes, up, up_nodes, output)
    return None


def _find_pointwise_user(
    node: Node, target: Any
) -> tuple[Node, tuple[Node, ...]] | None:
    frontier: list[tuple[Node, tuple[Node, ...]]] = [(user, ()) for user in node.users]
    visited: set[Node] = set()
    while frontier:
        current, path = frontier.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if current.target is target:
            return current, path
        if current.target in _VIEW_TARGETS:
            frontier.extend((user, (*path, current)) for user in current.users)
    return None


def _cast_dtype(node: Node) -> torch.dtype | None:
    dtype = node.kwargs.get("dtype")
    return dtype if isinstance(dtype, torch.dtype) else None


def _insert_transposed_cast_fusion(
    gm: GraphModule,
    *,
    chain: _MatmulChain,
    cast: Node,
    pattern: str,
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> bool:
    """Fuse the FP32 cast into GEMM while leaving shape-only ops outside."""
    if not _coda_nodes_available(chain.nodes):
        return False
    if not validate_partition(list(chain.nodes)):
        return False
    existing_nodes = set(gm.graph.nodes)
    root = chain.root
    external_inputs = list(root.all_input_nodes)

    body_graph = torch.fx.Graph()
    env: dict[Node, Node] = {}
    for index, external in enumerate(external_inputs):
        placeholder = body_graph.placeholder(f"arg{index}")
        _copy_meta(placeholder, external)
        env[external] = placeholder
    copied_root = body_graph.node_copy(root, lambda old: env[old])
    _copy_meta(copied_root, root)
    copied_cast = body_graph.call_function(
        _CAST_TARGET,
        args=(copied_root,),
        kwargs={"dtype": torch.float32},
    )
    root_value = _tensor_value(root)
    if root_value is not None:
        copied_cast.meta["val"] = root_value.to(dtype=torch.float32)
    body_graph.output((copied_cast,))
    body = GraphModule(torch.nn.Module(), body_graph)
    _propagate_body_meta(body, external_inputs)
    expected = root_value.to(dtype=torch.float32) if root_value is not None else None
    body_values = _validate_body_outputs(body, (expected,), pattern)
    mark_flex_gemm_body_gemm_node(body, _MM_TARGET)
    body_name = _register_body(gm, body, pattern)
    region = _CODA_INDUCTOR_REGION

    latest_input = max(external_inputs)
    with gm.graph.inserting_after(latest_input):
        body_attr = gm.graph.get_attr(body_name)
    options = _kernel_options(pattern)
    with gm.graph.inserting_after(body_attr):
        fused = gm.graph.call_function(
            flex_gemm_hop,
            args=(
                _MM_TARGET,
                body_attr,
                tuple(external_inputs),
                {},
                options,
            ),
        )
    fused.meta["val"] = body_values
    _mark_inductor_region(
        body,
        body_attr,
        fused,
        pattern=pattern,
        region=region,
    )

    with gm.graph.inserting_after(fused):
        extracted = gm.graph.call_function(operator.getitem, args=(fused, 0))
    if root_value is not None:
        extracted.meta["val"] = root_value.to(dtype=torch.float32)
    replacements = {root: extracted}
    cursor = extracted
    for old in chain.nodes[1:]:
        if old is cast:
            break
        with gm.graph.inserting_after(cursor):
            copied = gm.graph.node_copy(old, lambda arg: replacements.get(arg, arg))
        _copy_meta(copied, old)
        old_shape = _shape(old)
        if old_shape is not None:
            _set_empty_meta(copied, fused, old_shape, dtype=torch.float32)
        replacements[old] = copied
        cursor = copied
    _copy_meta(cursor, cast)
    cursor.meta.setdefault("custom", {})["coda_pattern"] = pattern
    cast.replace_all_uses_with(cursor)
    _mark_new_nodes_for_inductor(gm, existing_nodes, region=region, group=body_name)
    _mark_coda_owned(pattern, chain.nodes)
    if benchmark_regions is not None:
        benchmark_regions.append(
            _make_flex_gemm_benchmark_region(
                body,
                external_inputs,
                body_values,
                options,
            )
        )
    return True


def _rmsnorm_getitems(node: Node) -> tuple[Node, Node | None] | None:
    by_index: dict[int, Node] = {}
    for user in node.users:
        if user.target is operator.getitem and len(user.args) >= 2:
            index = user.args[1]
            if not isinstance(index, int) or index not in {0, 1} or index in by_index:
                return None
            by_index[index] = user
    if 0 not in by_index:
        return None
    return by_index[0], by_index.get(1)


def _valid_forward_rmsnorm(node: Node) -> bool:
    if node.target is not aten._fused_rms_norm.default or len(node.args) < 4:
        return False
    norm_input, normalized_shape, weight, eps = node.args[:4]
    if (
        not isinstance(norm_input, Node)
        or not isinstance(weight, Node)
        or not isinstance(eps, (float, int))
        or not isinstance(normalized_shape, (list, tuple))
        or _rmsnorm_getitems(node) is None
    ):
        return False
    input_shape = _shape(norm_input)
    return (
        input_shape is not None
        and list(normalized_shape) == [input_shape[-1]]
        and _shape(weight) == (input_shape[-1],)
    )


def _valid_backward_rmsnorm(node: Node) -> bool:
    if node.target is not aten._fused_rms_norm_backward.default or len(node.args) < 6:
        return False
    grad, norm_input, normalized_shape, rstd, weight, output_mask = node.args[:6]
    if (
        not all(isinstance(arg, Node) for arg in (grad, norm_input, rstd, weight))
        or not isinstance(normalized_shape, (list, tuple))
        or not isinstance(output_mask, (list, tuple))
        or list(output_mask) != [True, True]
    ):
        return False
    grad_shape = _shape(grad)
    input_shape = _shape(norm_input)
    outputs = _rmsnorm_getitems(node)
    return (
        grad_shape is not None
        and input_shape == grad_shape
        and list(normalized_shape) == [grad_shape[-1]]
        and _shape(weight) == (grad_shape[-1],)
        and _shape(rstd) == (*grad_shape[:-1], 1)
        and outputs is not None
        and outputs[1] is not None
    )


def _residual_rmsnorm_path(norm: Node) -> _MatmulChain | None:
    if (
        _is_backward(norm)
        or not _valid_forward_rmsnorm(norm)
        or norm.meta.get("coda_consumed")
    ):
        return None
    hidden = norm.args[0]
    if not isinstance(hidden, Node):
        return None
    path = _find_mm_path(hidden)
    if (
        path is None
        or not _is_reshape_only_path(path)
        or not _has_exact_add_shapes(path)
        or not _coda_nodes_available((*path.nodes, norm))
    ):
        return None
    path_set = set(path.nodes)
    residuals = [
        input_node
        for path_node in path.nodes
        if path_node.target is aten.add.Tensor
        for input_node in path_node.all_input_nodes
        if input_node not in path_set
    ]
    if not residuals:
        return None
    return path


def _weighted_residual_rmsnorm_path(norm: Node) -> _MatmulChain | None:
    if (
        _is_backward(norm)
        or not _valid_forward_rmsnorm(norm)
        or norm.meta.get("coda_consumed")
    ):
        return None
    hidden = norm.args[0]
    if not isinstance(hidden, Node) or hidden.target is not _CAST_TARGET:
        return None
    squeeze = _single_tensor_input(hidden)
    if (
        squeeze is None
        or squeeze.target is not aten.squeeze.dim
        or len(squeeze.args) < 2
        or squeeze.args[1] not in {1, -2}
    ):
        return None
    bmm = _single_tensor_input(squeeze)
    if bmm is None or bmm.target is not _BMM_TARGET or len(bmm.args) < 2:
        return None
    norm_outputs = _rmsnorm_getitems(norm)
    if norm_outputs is None:
        return None
    expected_norm_users = {norm_outputs[0]}
    if norm_outputs[1] is not None:
        expected_norm_users.add(norm_outputs[1])
    if set(norm.users) != expected_norm_users:
        return None
    lhs, rhs = bmm.args[:2]
    lhs_shape = _shape(lhs) if isinstance(lhs, Node) else None
    rhs_shape = _shape(rhs) if isinstance(rhs, Node) else None
    bmm_shape = _shape(bmm)
    hidden_shape = _shape(hidden)
    if (
        lhs_shape is None
        or rhs_shape is None
        or bmm_shape is None
        or hidden_shape is None
        or len(lhs_shape) != 3
        or len(rhs_shape) != 3
        or lhs_shape[1] != 1
        or lhs_shape[0] != rhs_shape[0]
        or lhs_shape[2] != rhs_shape[1]
        or bmm_shape != (lhs_shape[0], 1, rhs_shape[2])
        or hidden_shape != (bmm_shape[0], bmm_shape[2])
        or set(bmm.users) != {squeeze}
        or set(squeeze.users) != {hidden}
        or not _coda_nodes_available((bmm, squeeze, hidden, norm))
    ):
        return None
    return _MatmulChain(bmm, (bmm, squeeze, hidden))


def _downstream_mm(node: Node, *, backward: bool | None = None) -> _MatmulUse | None:
    frontier = [(user, ()) for user in node.users]
    visited: set[Node] = set()
    matches: list[_MatmulUse] = []
    while frontier:
        user, wrappers = frontier.pop(0)
        if user in visited:
            continue
        visited.add(user)
        if user.target is _MM_TARGET:
            if backward is not None and _is_backward(user) is not backward:
                continue
            operands = [index for index, arg in enumerate(user.args[:2]) if arg is node]
            if wrappers:
                operands = [
                    index
                    for index, arg in enumerate(user.args[:2])
                    if arg is wrappers[-1]
                ]
            if len(operands) == 1:
                matches.append(_MatmulUse(user, operands[0], wrappers))
            continue
        if user.target in _RESHAPE_TARGETS:
            frontier.extend(
                (downstream, (*wrappers, user)) for downstream in user.users
            )
    return matches[0] if len(matches) == 1 else None


def _valid_projection_rmsnorm_use(
    norm: Node,
    first_path: _MatmulChain,
    second_use: _MatmulUse,
    full_output: Node,
) -> bool:
    getitems = _rmsnorm_getitems(norm)
    if getitems is None or second_use.operand != 0:
        return False
    norm_input = norm.args[0]
    second_mm = second_use.node
    second_lhs, second_rhs = second_mm.args[:2]
    if not all(isinstance(node, Node) for node in (norm_input, second_lhs, second_rhs)):
        return False
    root_shape = _shape(first_path.root)
    full_shape = _shape(full_output)
    norm_shape = _shape(norm_input)
    norm_output_shape = _shape(getitems[0])
    lhs_shape = _shape(second_lhs)
    rhs_shape = _shape(second_rhs)
    output_shape = _shape(second_mm)
    if not all(
        shape is not None
        for shape in (
            root_shape,
            full_shape,
            norm_shape,
            norm_output_shape,
            lhs_shape,
            rhs_shape,
            output_shape,
        )
    ):
        return False
    if not all(
        len(shape) == 2 for shape in (root_shape, lhs_shape, rhs_shape, output_shape)
    ):
        return False
    return bool(
        _same_numel(root_shape, full_shape)
        and norm_output_shape == norm_shape
        and _same_numel(norm_shape, lhs_shape)
        and lhs_shape[-1] == rhs_shape[0]
        and output_shape == (lhs_shape[0], rhs_shape[1])
    )


def _insert_projection_rmsnorm_fusion(
    gm: GraphModule,
    *,
    norm: Node,
    first_path: _MatmulChain,
    second_use: _MatmulUse,
    full_output: Node,
    pattern: str,
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    existing_nodes = set(gm.graph.nodes)
    getitems = _rmsnorm_getitems(norm)
    if getitems is None:
        raise AssertionError(f"CODA {pattern} expected RMSNorm tuple outputs")
    old_norm_out, old_rstd = getitems
    second_mm = second_use.node
    if second_use.operand != 0:
        raise AssertionError(f"CODA {pattern} requires RMSNorm as GEMM operand 0")
    norm_input = norm.args[0]
    weight = norm.args[2]
    if not isinstance(norm_input, Node) or not isinstance(weight, Node):
        raise AssertionError(f"CODA {pattern} expected tensor RMSNorm inputs")
    norm_output_dtype = _dtype(old_norm_out)
    second_output_dtype = _dtype(second_mm)
    if norm_output_dtype is None or second_output_dtype is None:
        raise AssertionError(f"CODA {pattern} requires output dtype metadata")
    full_shape = _shape(full_output)
    norm_shape = _shape(norm_input)
    second_shape = _shape(second_mm)
    if full_shape is None or norm_shape is None or second_shape is None:
        raise AssertionError(f"CODA {pattern} requires shape metadata")
    full_width = full_shape[-1]
    norm_width = norm_shape[-1]
    if not isinstance(full_width, int) or not isinstance(norm_width, int):
        raise AssertionError(
            f"CODA {pattern} unsupported widths full={full_width}, norm={norm_width}"
        )
    reduction_group = gcd(RMSNORM_FORWARD_GROUP, full_width, norm_width)

    insertion_input = max([*first_path.root.all_input_nodes, weight])
    cursor = insertion_input
    padded_weight = weight
    if full_width != norm_width:
        padded_weight = _call_after(
            gm.graph,
            cursor,
            aten.constant_pad_nd.default,
            (weight, [0, full_width - norm_width], 1.0),
        )
        padded_value = _tensor_value(weight)
        if padded_value is not None:
            padded_weight.meta["val"] = aten.constant_pad_nd.default(
                padded_value,
                [0, full_width - norm_width],
                1.0,
            )
        cursor = padded_weight
    weight_row = _call_after(
        gm.graph,
        cursor,
        aten.view.default,
        (padded_weight, [1, full_width]),
    )
    _set_view_meta(weight_row, padded_weight, [1, full_width])
    weight_row_fp32 = _call_after(
        gm.graph,
        weight_row,
        _CAST_TARGET,
        (weight_row,),
        {"dtype": torch.float32},
    )
    _set_empty_meta(
        weight_row_fp32,
        weight_row,
        [1, full_width],
        dtype=torch.float32,
    )
    cursor = weight_row_fp32
    body_graph = torch.fx.Graph()
    external_inputs = [*first_path.root.all_input_nodes, weight_row_fp32]
    env: dict[Node, Node] = {}
    for index, external in enumerate(external_inputs):
        placeholder = body_graph.placeholder(f"arg{index}")
        _copy_meta(placeholder, external)
        env[external] = placeholder
    body_full = body_graph.call_function(
        _MM_TARGET,
        args=(env[first_path.root.args[0]], env[first_path.root.args[1]]),
    )
    body_full_fp32 = body_graph.call_function(
        _CAST_TARGET, args=(body_full,), kwargs={"dtype": torch.float32}
    )
    weighted_fp32 = body_graph.call_function(
        aten.mul.Tensor, args=(body_full_fp32, env[weight_row_fp32])
    )
    weighted = body_graph.call_function(
        _CAST_TARGET, args=(weighted_fp32,), kwargs={"dtype": norm_output_dtype}
    )
    root_shape = _shape(first_path.root)
    if root_shape is None:
        raise AssertionError(f"CODA {pattern} requires first GEMM shape metadata")
    grouped_shape = [
        root_shape[0],
        full_width // reduction_group,
        reduction_group,
    ]
    grouped = body_graph.call_function(
        aten.view.default, args=(body_full_fp32, grouped_shape)
    )
    squared = body_graph.call_function(aten.mul.Tensor, args=(grouped, grouped))
    partial = body_graph.call_function(aten.mean.dim, args=(squared, [-1], False))
    body_graph.output((body_full, weighted, partial))
    body = GraphModule(torch.nn.Module(), body_graph)
    _propagate_body_meta(body, external_inputs)
    mark_flex_gemm_body_gemm_node(body, _MM_TARGET)
    body_name = _register_body(gm, body, pattern)
    first_region = _CODA_INDUCTOR_REGION
    first_options = _kernel_options(pattern)

    with gm.graph.inserting_after(cursor):
        body_attr = gm.graph.get_attr(body_name)
    fused_first = _call_after(
        gm.graph,
        body_attr,
        flex_gemm_hop,
        (
            _MM_TARGET,
            body_attr,
            tuple(external_inputs),
            {},
            first_options,
        ),
    )
    root_value = _tensor_value(first_path.root)
    if root_value is None:
        raise AssertionError(f"CODA {pattern} requires first GEMM value metadata")
    partial_value = root_value.new_empty(
        [root_shape[0], full_width // reduction_group],
        dtype=torch.float32,
    )
    fused_first.meta["val"] = (root_value, root_value, partial_value)
    _mark_inductor_region(
        body,
        body_attr,
        fused_first,
        pattern=pattern,
        region=first_region,
    )
    new_full_2d = _call_after(gm.graph, fused_first, operator.getitem, (fused_first, 0))
    _copy_meta(new_full_2d, first_path.root)
    weighted_full_2d = _call_after(
        gm.graph, new_full_2d, operator.getitem, (fused_first, 1)
    )
    _copy_meta(weighted_full_2d, first_path.root)
    partial_out = _call_after(
        gm.graph, weighted_full_2d, operator.getitem, (fused_first, 2)
    )
    _mark_nodes_for_inductor(
        (new_full_2d, weighted_full_2d, partial_out),
        region=first_region,
        group=body_name,
    )
    _set_empty_meta(
        partial_out,
        first_path.root,
        [root_shape[0], full_width // reduction_group],
        dtype=torch.float32,
    )
    new_full = _call_after(
        gm.graph, partial_out, aten.view.default, (new_full_2d, list(full_shape))
    )
    _copy_meta(new_full, full_output)
    weighted_full = _call_after(
        gm.graph,
        new_full,
        aten.view.default,
        (weighted_full_2d, list(full_shape)),
    )
    _infer_new_node_meta(weighted_full)
    cursor = weighted_full
    active_partial = partial_out
    if full_width != norm_width:
        weighted_norm = _call_after(
            gm.graph,
            cursor,
            aten.slice.Tensor,
            (weighted_full, -1, 0, norm_width),
        )
        _infer_new_node_meta(weighted_norm)
        cursor = weighted_norm
        new_norm_input = _call_after(
            gm.graph,
            cursor,
            aten.slice.Tensor,
            (new_full, -1, 0, norm_width),
        )
        cursor = new_norm_input
        active_partial = _call_after(
            gm.graph,
            cursor,
            aten.slice.Tensor,
            (partial_out, -1, 0, norm_width // reduction_group),
        )
        _set_empty_meta(
            active_partial,
            first_path.root,
            [root_shape[0], norm_width // reduction_group],
            dtype=torch.float32,
        )
        cursor = active_partial
    else:
        weighted_norm = weighted_full
        new_norm_input = new_full
    partial_mean = _call_after(
        gm.graph, cursor, aten.mean.dim, (active_partial, [-1], True)
    )
    variance = _call_after(
        gm.graph, partial_mean, aten.add.Scalar, (partial_mean, norm.args[3])
    )
    new_rstd_2d = _call_after(gm.graph, variance, aten.rsqrt.default, (variance,))
    _set_empty_meta(
        new_rstd_2d,
        first_path.root,
        [root_shape[0], 1],
        dtype=torch.float32,
    )
    rstd_shape = _shape(old_rstd) if old_rstd is not None else (*norm_shape[:-1], 1)
    new_rstd = _call_after(
        gm.graph,
        new_rstd_2d,
        aten.view.default,
        (new_rstd_2d, list(rstd_shape)),
    )
    norm_input_fp32 = _call_after(
        gm.graph,
        new_rstd,
        _CAST_TARGET,
        (new_norm_input,),
        {"dtype": torch.float32},
    )
    weight_fp32 = _call_after(
        gm.graph,
        norm_input_fp32,
        _CAST_TARGET,
        (weight,),
        {"dtype": torch.float32},
    )
    normalized = _call_after(
        gm.graph, weight_fp32, aten.mul.Tensor, (norm_input_fp32, new_rstd)
    )
    normalized_weighted = _call_after(
        gm.graph, normalized, aten.mul.Tensor, (normalized, weight_fp32)
    )
    new_norm_out = _call_after(
        gm.graph,
        normalized_weighted,
        _CAST_TARGET,
        (normalized_weighted,),
        {"dtype": norm_output_dtype},
    )
    _copy_meta(new_norm_out, old_norm_out)
    if old_rstd is not None:
        _copy_meta(new_rstd, old_rstd)

    second_lhs = second_mm.args[0]
    second_rhs = second_mm.args[1]
    if not isinstance(second_lhs, Node) or not isinstance(second_rhs, Node):
        raise AssertionError(f"CODA {pattern} expected tensor expansion GEMM inputs")
    lhs_shape = _shape(second_lhs)
    if lhs_shape is None:
        raise AssertionError(f"CODA {pattern} requires expansion input shape")
    weighted_2d = _call_after(
        gm.graph, new_norm_out, aten.view.default, (weighted_norm, list(lhs_shape))
    )
    _infer_new_node_meta(weighted_2d)

    second_body_graph = torch.fx.Graph()
    lhs_arg = second_body_graph.placeholder("lhs")
    rhs_arg = second_body_graph.placeholder("rhs")
    rstd_arg = second_body_graph.placeholder("rstd")
    second_acc = second_body_graph.call_function(_MM_TARGET, args=(lhs_arg, rhs_arg))
    expanded_shape = [root_shape[0], second_shape[-1]]
    expanded = second_body_graph.call_function(
        aten.view.default, args=(second_acc, expanded_shape)
    )
    expanded_fp32 = second_body_graph.call_function(
        _CAST_TARGET, args=(expanded,), kwargs={"dtype": torch.float32}
    )
    scaled = second_body_graph.call_function(
        aten.mul.Tensor, args=(expanded_fp32, rstd_arg)
    )
    scaled_output = second_body_graph.call_function(
        _CAST_TARGET, args=(scaled,), kwargs={"dtype": second_output_dtype}
    )
    second_out = second_body_graph.call_function(
        aten.view.default, args=(scaled_output, list(second_shape))
    )
    second_body_graph.output((second_out,))
    second_body = GraphModule(torch.nn.Module(), second_body_graph)
    _propagate_body_meta(second_body, (weighted_2d, second_rhs, new_rstd_2d))
    mark_flex_gemm_body_gemm_node(second_body, _MM_TARGET)
    second_body_name = _register_body(gm, second_body, pattern)
    second_region = _CODA_INDUCTOR_REGION
    second_options = _kernel_options(pattern, config_index=1)
    cursor = max([weighted_2d, second_rhs, new_rstd_2d])
    with gm.graph.inserting_after(cursor):
        second_body_attr = gm.graph.get_attr(second_body_name)
    fused_second = _call_after(
        gm.graph,
        second_body_attr,
        flex_gemm_hop,
        (
            _MM_TARGET,
            second_body_attr,
            (weighted_2d, second_rhs, new_rstd_2d),
            {},
            second_options,
        ),
    )
    fused_second.meta["val"] = (second_mm.meta.get("val"),)
    _mark_inductor_region(
        second_body,
        second_body_attr,
        fused_second,
        pattern=pattern,
        region=second_region,
    )
    fused_second_out = _call_after(
        gm.graph, fused_second, operator.getitem, (fused_second, 0)
    )
    _copy_meta(fused_second_out, second_mm)
    _mark_nodes_for_inductor(
        (fused_second_out,),
        region=second_region,
        group=second_body_name,
    )

    full_output.replace_all_uses_with(new_full)
    old_norm_out.replace_all_uses_with(new_norm_out)
    if old_rstd is not None:
        old_rstd.replace_all_uses_with(new_rstd)
    second_mm.replace_all_uses_with(fused_second_out)
    _mark_new_nodes_for_inductor(
        gm, existing_nodes, region=first_region, group=body_name
    )
    _mark_coda_owned(
        pattern,
        (*first_path.nodes, norm, old_norm_out, second_mm),
    )
    if old_rstd is not None:
        _mark_coda_owned(pattern, (old_rstd,))
    if benchmark_regions is not None:
        benchmark_regions.extend(
            (
                _make_flex_gemm_benchmark_region(
                    body,
                    external_inputs,
                    _body_output_values(body, pattern),
                    first_options,
                ),
                _make_flex_gemm_benchmark_region(
                    second_body,
                    (weighted_2d, second_rhs, new_rstd_2d),
                    _body_output_values(second_body, pattern),
                    second_options,
                ),
            )
        )


def _insert_forward_rmsnorm_fusion(
    gm: GraphModule,
    *,
    norm: Node,
    path: _MatmulChain,
    pattern: str,
    flex_gemm_options: dict[str, Any] | None = None,
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    existing_nodes = set(gm.graph.nodes)
    hidden = norm.args[0]
    weight = norm.args[2]
    if not isinstance(hidden, Node) or not isinstance(weight, Node):
        raise AssertionError(f"CODA {pattern} expected tensor RMSNorm inputs")
    getitems = _rmsnorm_getitems(norm)
    if getitems is None:
        raise AssertionError(f"CODA {pattern} expected RMSNorm tuple outputs")
    norm_out, old_rstd = getitems
    baseline_region = None
    baseline_inputs: tuple[Node, ...] = ()
    baseline_outputs: tuple[Node, ...] = ()
    if benchmark_regions is not None:
        baseline_nodes = _ordered_nodes(
            gm,
            (*path.nodes, norm, norm_out, *((old_rstd,) if old_rstd else ())),
        )
        baseline_region, baseline_inputs, baseline_outputs = fuse_as_graphmodule(
            gm,
            baseline_nodes,
            f"BenchmarkBaseline_{pattern}",
            always_return_tuple=True,
        )
    norm_output_dtype = _dtype(norm_out)
    if norm_output_dtype is None:
        raise AssertionError(f"CODA {pattern} requires output dtype metadata")
    hidden_shape = _shape(hidden)
    if hidden_shape is None:
        raise AssertionError(f"CODA {pattern} requires shape metadata")
    width = hidden_shape[-1]
    if not isinstance(width, int) or width % RMSNORM_FORWARD_GROUP:
        raise AssertionError(f"CODA {pattern} unsupported RMSNorm width {width}")

    ordered = _ordered_nodes(gm, path.nodes)
    body_set = set(ordered)
    external_inputs: list[Node] = []
    for arg in path.root.all_input_nodes:
        if arg not in external_inputs:
            external_inputs.append(arg)
    for body_node in ordered:
        for arg in body_node.all_input_nodes:
            if arg not in body_set and arg not in external_inputs:
                external_inputs.append(arg)

    root_shape = _shape(path.root)
    if root_shape is None:
        raise AssertionError(f"CODA {pattern} requires GEMM shape metadata")
    root_inputs = set(path.root.all_input_nodes)
    body_input_pairs = [
        (
            external,
            external
            if external in root_inputs
            else _view_region_input(gm, external, root_shape, existing_nodes),
        )
        for external in external_inputs
    ]
    cursor = max([*(actual for _, actual in body_input_pairs), weight])

    body_graph = torch.fx.Graph()
    env: dict[Node, Node] = {}
    for index, (external, actual) in enumerate(body_input_pairs):
        placeholder = body_graph.placeholder(f"arg{index}")
        _copy_meta(placeholder, actual)
        env[external] = placeholder
    gemm_target = path.root.target
    if gemm_target not in {_MM_TARGET, _BMM_TARGET}:
        raise AssertionError(f"CODA {pattern} expected MM or BMM root")
    body_hidden = body_graph.call_function(
        gemm_target,
        args=(env[path.root.args[0]], env[path.root.args[1]]),
    )
    previous = path.root
    for body_node in ordered[1:]:
        if body_node.target in _VIEW_TARGETS or body_node.target is aten.squeeze.dim:
            previous = body_node
            continue
        if body_node.target is _CAST_TARGET:
            body_hidden = body_graph.call_function(
                _CAST_TARGET,
                args=(body_hidden,),
                kwargs=body_node.kwargs,
            )
            previous = body_node
            continue
        if body_node.target is not aten.add.Tensor:
            raise AssertionError(
                f"CODA {pattern} found unsupported residual-path op {body_node.target}"
            )
        if body_node.args[0] is previous:
            residual = body_node.args[1]
            lhs_is_accumulator = True
        elif body_node.args[1] is previous:
            residual = body_node.args[0]
            lhs_is_accumulator = False
        else:
            raise AssertionError(f"CODA {pattern} lost the residual add path")
        if not isinstance(residual, Node):
            raise AssertionError(f"CODA {pattern} expected a tensor residual")
        add_args = (
            (body_hidden, env[residual])
            if lhs_is_accumulator
            else (env[residual], body_hidden)
        )
        body_hidden = body_graph.call_function(
            aten.add.Tensor,
            args=(*add_args, *body_node.args[2:]),
            kwargs=body_node.kwargs,
        )
        previous = body_node
    hidden_fp32 = body_graph.call_function(
        _CAST_TARGET, args=(body_hidden,), kwargs={"dtype": torch.float32}
    )
    grouped_shape = [
        *root_shape[:-1],
        width // RMSNORM_FORWARD_GROUP,
        RMSNORM_FORWARD_GROUP,
    ]
    grouped = body_graph.call_function(
        aten.view.default, args=(hidden_fp32, grouped_shape)
    )
    squared = body_graph.call_function(aten.mul.Tensor, args=(grouped, grouped))
    partial = body_graph.call_function(aten.mean.dim, args=(squared, [-1], False))
    body_graph.output((body_hidden, partial))
    body = GraphModule(torch.nn.Module(), body_graph)
    _propagate_body_meta(body, tuple(actual for _, actual in body_input_pairs))
    mark_flex_gemm_body_gemm_node(body, gemm_target)
    body_name = _register_body(gm, body, pattern)
    region = _CODA_INDUCTOR_REGION
    options = (
        _kernel_options(pattern) if flex_gemm_options is None else flex_gemm_options
    )

    with gm.graph.inserting_after(cursor):
        body_attr = gm.graph.get_attr(body_name)
    with gm.graph.inserting_after(body_attr):
        fused = gm.graph.call_function(
            flex_gemm_hop,
            args=(
                gemm_target,
                body_attr,
                tuple(actual for _, actual in body_input_pairs),
                {},
                options,
            ),
        )
    root_value = _tensor_value(path.root)
    if root_value is None:
        raise AssertionError(f"CODA {pattern} requires GEMM value metadata")
    hidden_dtype = _dtype(hidden)
    if hidden_dtype is None:
        raise AssertionError(f"CODA {pattern} requires RMSNorm input dtype metadata")
    hidden_value = root_value.new_empty(root_shape, dtype=hidden_dtype)
    partial_shape = [*root_shape[:-1], width // RMSNORM_FORWARD_GROUP]
    partial_value = root_value.new_empty(
        partial_shape,
        dtype=torch.float32,
    )
    fused.meta["val"] = (hidden_value, partial_value)
    _mark_inductor_region(
        body,
        body_attr,
        fused,
        pattern=pattern,
        region=region,
    )
    with gm.graph.inserting_after(fused):
        hidden_physical = gm.graph.call_function(operator.getitem, args=(fused, 0))
    _set_empty_meta(hidden_physical, path.root, root_shape, dtype=hidden_dtype)
    partial_out = _call_after(gm.graph, hidden_physical, operator.getitem, (fused, 1))
    _mark_nodes_for_inductor(
        (hidden_physical, partial_out),
        region=region,
        group=body_name,
    )
    _set_empty_meta(
        partial_out,
        path.root,
        partial_shape,
        dtype=torch.float32,
    )
    new_hidden = _call_after(
        gm.graph,
        partial_out,
        aten.view.default,
        (hidden_physical, list(hidden_shape)),
    )
    _copy_meta(new_hidden, hidden)
    partial_mean = _call_after(
        gm.graph, new_hidden, aten.mean.dim, (partial_out, [-1], True)
    )
    variance = _call_after(
        gm.graph, partial_mean, aten.add.Scalar, (partial_mean, norm.args[3])
    )
    new_rstd_physical = _call_after(gm.graph, variance, aten.rsqrt.default, (variance,))
    _set_empty_meta(
        new_rstd_physical,
        path.root,
        [*root_shape[:-1], 1],
        dtype=torch.float32,
    )
    rstd_shape = _shape(old_rstd) if old_rstd is not None else (*hidden_shape[:-1], 1)
    new_rstd = _call_after(
        gm.graph,
        new_rstd_physical,
        aten.view.default,
        (new_rstd_physical, list(rstd_shape)),
    )
    hidden_float = _call_after(
        gm.graph,
        new_rstd,
        _CAST_TARGET,
        (new_hidden,),
        {"dtype": torch.float32},
    )
    weight_float = _call_after(
        gm.graph,
        hidden_float,
        _CAST_TARGET,
        (weight,),
        {"dtype": torch.float32},
    )
    normalized = _call_after(
        gm.graph, weight_float, aten.mul.Tensor, (hidden_float, new_rstd)
    )
    weighted = _call_after(
        gm.graph, normalized, aten.mul.Tensor, (normalized, weight_float)
    )
    new_norm_out = _call_after(
        gm.graph,
        weighted,
        _CAST_TARGET,
        (weighted,),
        {"dtype": norm_output_dtype},
    )
    _copy_meta(new_norm_out, norm_out)
    if old_rstd is not None:
        _copy_meta(new_rstd, old_rstd)

    body_nodes = set(ordered)
    for user in list(hidden.users):
        if user not in body_nodes and user is not norm:
            user.replace_input_with(hidden, new_hidden)
    norm_out.replace_all_uses_with(new_norm_out)
    if old_rstd is not None:
        old_rstd.replace_all_uses_with(new_rstd)
    _mark_new_nodes_for_inductor(gm, existing_nodes, region=region, group=body_name)
    _mark_coda_owned(pattern, (*path.nodes, norm, norm_out))
    if old_rstd is not None:
        _mark_coda_owned(pattern, (old_rstd,))
    if benchmark_regions is not None:
        if baseline_region is None:
            raise AssertionError(f"CODA {pattern} expected a baseline benchmark region")
        candidate_nodes = [
            node for node in gm.graph.nodes if node not in existing_nodes
        ]
        candidate_region, candidate_inputs, candidate_outputs = fuse_as_graphmodule(
            gm,
            candidate_nodes,
            f"BenchmarkCandidate_{pattern}",
            always_return_tuple=True,
        )
        replacements = {
            hidden: new_hidden,
            norm_out: new_norm_out,
            **({old_rstd: new_rstd} if old_rstd is not None else {}),
        }
        output_order = [
            candidate_outputs.index(replacements[node]) for node in baseline_outputs
        ]
        output = next(
            node for node in candidate_region.graph.nodes if node.op == "output"
        )
        values = output.args[0]
        if not isinstance(values, tuple):
            raise AssertionError(f"CODA {pattern} expected tuple benchmark outputs")
        output.args = (tuple(values[index] for index in output_order),)
        candidate_region.graph.lint()
        candidate_region.recompile()
        benchmark_regions.append(
            make_rewrite_benchmark_region(
                baseline_region,
                candidate_region,
            )
        )


def _insert_backward_rmsnorm_fusion(
    gm: GraphModule,
    *,
    norm_backward: Node,
    path: _MatmulChain,
    pattern: str,
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    existing_nodes = set(gm.graph.nodes)
    grad = norm_backward.args[0]
    norm_input = norm_backward.args[1]
    rstd = norm_backward.args[3]
    weight = norm_backward.args[4]
    if not all(isinstance(arg, Node) for arg in (grad, norm_input, rstd, weight)):
        raise AssertionError(f"CODA {pattern} expected tensor RMSNorm inputs")
    outputs = _rmsnorm_getitems(norm_backward)
    if outputs is None:
        raise AssertionError(f"CODA {pattern} expected RMSNorm backward outputs")
    old_grad_input, old_grad_weight = outputs
    if old_grad_weight is None:
        raise AssertionError(f"CODA {pattern} expected RMSNorm weight gradient")
    grad_input_dtype = _dtype(old_grad_input)
    grad_weight_dtype = _dtype(old_grad_weight)
    if grad_input_dtype is None or grad_weight_dtype is None:
        raise AssertionError(f"CODA {pattern} requires output dtype metadata")
    grad_shape = _shape(grad)
    if grad_shape is None:
        raise AssertionError(f"CODA {pattern} requires shape metadata")
    width = grad_shape[-1]
    if not isinstance(width, int) or width % RMSNORM_BACKWARD_GROUP:
        raise AssertionError(f"CODA {pattern} unsupported RMSNorm width {width}")

    ordered = _ordered_nodes(gm, path.nodes)
    body_set = set(ordered)
    external_inputs: list[Node] = []
    for arg in path.root.all_input_nodes:
        if arg not in external_inputs:
            external_inputs.append(arg)
    for external in (norm_input, rstd, weight):
        if external not in external_inputs:
            external_inputs.append(external)
    for body_node in ordered:
        for arg in body_node.all_input_nodes:
            if arg not in body_set and arg not in external_inputs:
                external_inputs.append(arg)

    root_shape = _shape(path.root)
    if root_shape is None:
        raise AssertionError(f"CODA {pattern} requires GEMM shape metadata")
    root_inputs = set(path.root.all_input_nodes)
    desired_shapes = {
        norm_input: root_shape,
        rstd: (root_shape[0], 1),
        weight: (1, root_shape[-1]),
    }
    body_input_pairs = [
        (
            external,
            external
            if external in root_inputs
            else _view_region_input(
                gm,
                external,
                desired_shapes.get(external, root_shape),
                existing_nodes,
            ),
        )
        for external in external_inputs
    ]
    cursor = max(actual for _, actual in body_input_pairs)

    body_graph = torch.fx.Graph()
    env: dict[Node, Node] = {}
    for index, (external, actual) in enumerate(body_input_pairs):
        placeholder = body_graph.placeholder(f"arg{index}")
        _copy_meta(placeholder, actual)
        env[external] = placeholder
    body_grad = body_graph.call_function(
        _MM_TARGET,
        args=(env[path.root.args[0]], env[path.root.args[1]]),
    )
    grad_fp32 = body_graph.call_function(
        _CAST_TARGET, args=(body_grad,), kwargs={"dtype": torch.float32}
    )
    input_fp32 = body_graph.call_function(
        _CAST_TARGET, args=(env[norm_input],), kwargs={"dtype": torch.float32}
    )
    x_hat = body_graph.call_function(aten.mul.Tensor, args=(input_fp32, env[rstd]))
    weight_fp32 = body_graph.call_function(
        _CAST_TARGET, args=(env[weight],), kwargs={"dtype": torch.float32}
    )
    grad_x_hat = body_graph.call_function(
        aten.mul.Tensor, args=(grad_fp32, weight_fp32)
    )
    dot = body_graph.call_function(aten.mul.Tensor, args=(x_hat, grad_x_hat))
    grouped_shape = [
        root_shape[0],
        width // RMSNORM_BACKWARD_GROUP,
        RMSNORM_BACKWARD_GROUP,
    ]
    grouped = body_graph.call_function(aten.view.default, args=(dot, grouped_shape))
    partial = body_graph.call_function(
        aten.sum.dim_IntList, args=(grouped, [-1], False)
    )
    body_graph.output((body_grad, partial))
    body = GraphModule(torch.nn.Module(), body_graph)
    _propagate_body_meta(body, tuple(actual for _, actual in body_input_pairs))
    mark_flex_gemm_body_gemm_node(body, _MM_TARGET)
    body_name = _register_body(gm, body, pattern)
    region = _CODA_INDUCTOR_REGION
    options = _kernel_options(pattern)

    with gm.graph.inserting_after(cursor):
        body_attr = gm.graph.get_attr(body_name)
    with gm.graph.inserting_after(body_attr):
        fused = gm.graph.call_function(
            flex_gemm_hop,
            args=(
                _MM_TARGET,
                body_attr,
                tuple(actual for _, actual in body_input_pairs),
                {},
                options,
            ),
        )
    root_value = _tensor_value(path.root)
    if root_value is None:
        raise AssertionError(f"CODA {pattern} requires GEMM value metadata")
    partial_value = root_value.new_empty(
        [root_shape[0], width // RMSNORM_BACKWARD_GROUP],
        dtype=torch.float32,
    )
    fused.meta["val"] = (root_value, partial_value)
    _mark_inductor_region(
        body,
        body_attr,
        fused,
        pattern=pattern,
        region=region,
    )
    with gm.graph.inserting_after(fused):
        grad_2d = gm.graph.call_function(operator.getitem, args=(fused, 0))
    _copy_meta(grad_2d, path.root)
    partial_out = _call_after(gm.graph, grad_2d, operator.getitem, (fused, 1))
    _mark_nodes_for_inductor(
        (grad_2d, partial_out),
        region=region,
        group=body_name,
    )
    _set_empty_meta(
        partial_out,
        path.root,
        [root_shape[0], width // RMSNORM_BACKWARD_GROUP],
        dtype=torch.float32,
    )
    new_grad = _call_after(
        gm.graph, partial_out, aten.view.default, (grad_2d, list(grad_shape))
    )
    _copy_meta(new_grad, grad)
    grad_fp32_out = _call_after(
        gm.graph,
        new_grad,
        _CAST_TARGET,
        (new_grad,),
        {"dtype": torch.float32},
    )
    input_fp32_out = _call_after(
        gm.graph,
        grad_fp32_out,
        _CAST_TARGET,
        (norm_input,),
        {"dtype": torch.float32},
    )
    x_hat_out = _call_after(
        gm.graph, input_fp32_out, aten.mul.Tensor, (input_fp32_out, rstd)
    )
    weight_fp32_out = _call_after(
        gm.graph,
        x_hat_out,
        _CAST_TARGET,
        (weight,),
        {"dtype": torch.float32},
    )
    grad_x_hat_out = _call_after(
        gm.graph, weight_fp32_out, aten.mul.Tensor, (grad_fp32_out, weight_fp32_out)
    )
    row_dot_2d = _call_after(
        gm.graph,
        grad_x_hat_out,
        aten.sum.dim_IntList,
        (partial_out, [-1], True),
    )
    rstd_shape = _shape(rstd)
    if rstd_shape is None:
        raise AssertionError(f"CODA {pattern} requires RMSNorm rstd shape metadata")
    row_dot = _call_after(
        gm.graph,
        row_dot_2d,
        aten.view.default,
        (row_dot_2d, list(rstd_shape)),
    )
    scaled_x = _call_after(gm.graph, row_dot, aten.div.Scalar, (x_hat_out, width))
    correction = _call_after(gm.graph, scaled_x, aten.mul.Tensor, (scaled_x, row_dot))
    centered = _call_after(
        gm.graph, correction, aten.sub.Tensor, (grad_x_hat_out, correction)
    )
    grad_input_fp32 = _call_after(gm.graph, centered, aten.mul.Tensor, (centered, rstd))
    new_grad_input = _call_after(
        gm.graph,
        grad_input_fp32,
        _CAST_TARGET,
        (grad_input_fp32,),
        {"dtype": grad_input_dtype},
    )
    grad_weight_terms = _call_after(
        gm.graph, new_grad_input, aten.mul.Tensor, (grad_fp32_out, x_hat_out)
    )
    reduction_dims = list(range(len(grad_shape) - 1))
    grad_weight_fp32 = _call_after(
        gm.graph,
        grad_weight_terms,
        aten.sum.dim_IntList,
        (grad_weight_terms, reduction_dims, False),
    )
    new_grad_weight = _call_after(
        gm.graph,
        grad_weight_fp32,
        _CAST_TARGET,
        (grad_weight_fp32,),
        {"dtype": grad_weight_dtype},
    )
    _copy_meta(new_grad_input, old_grad_input)
    _copy_meta(new_grad_weight, old_grad_weight)
    old_grad_input.replace_all_uses_with(new_grad_input)
    old_grad_weight.replace_all_uses_with(new_grad_weight)
    _mark_new_nodes_for_inductor(gm, existing_nodes, region=region, group=body_name)
    _mark_coda_owned(
        pattern,
        (*path.nodes, norm_backward, old_grad_input, old_grad_weight),
    )
    if benchmark_regions is not None:
        benchmark_regions.append(
            _make_flex_gemm_benchmark_region(
                body,
                tuple(actual for _, actual in body_input_pairs),
                _body_output_values(body, pattern),
                options,
            )
        )


# Function names intentionally match the canonical report pattern IDs.
@register_coda_pattern(
    best_configs={
        10: (
            _quack_config(256, 192, dynamic=False),
            _quack_config(256, 256, dynamic=True),
        )
    },
)
def F_mla_qproj_rmsnorm_expand(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse a Q projection, RMSNorm, and the following expansion projection.

    Match:   Q projection -> RMSNorm -> expansion projection
    Ops:     ``aten.mm.default -> [_RESHAPE_TARGETS] ->
             aten._fused_rms_norm.default -> aten.mm.default``.
    Rewrite: The first ``flex_gemm`` emits weighted values and partial mean
             squares; the second consumes the resulting rstd in its epilogue.
    """
    pattern = F_mla_qproj_rmsnorm_expand.__name__
    for norm in list(gm.graph.nodes):
        if (
            _is_backward(norm)
            or not _valid_forward_rmsnorm(norm)
            or norm.meta.get("coda_consumed")
        ):
            continue
        norm_input = norm.args[0]
        if not isinstance(norm_input, Node):
            continue
        path = _chain_to_mm(norm_input, allow_cast=False)
        if path is None or not _is_reshape_only_path(path):
            continue
        norm_out = _rmsnorm_getitems(norm)
        if norm_out is None:
            continue
        second_use = _downstream_mm(norm_out[0], backward=False)
        if (
            second_use is None
            or second_use.operand != 0
            or _is_backward(second_use.node)
            or not _valid_projection_rmsnorm_use(norm, path, second_use, norm_input)
            or not _coda_nodes_available((*path.nodes, norm, second_use.node))
        ):
            continue
        if not _claim_coda_match(pattern, norm):
            continue
        _insert_projection_rmsnorm_fusion(
            gm,
            norm=norm,
            first_path=path,
            second_use=second_use,
            full_output=norm_input,
            pattern=pattern,
            benchmark_regions=benchmark_regions,
        )
        counts[pattern] += 1


@register_coda_pattern(
    best_configs={
        10: (
            _quack_config(128, 192, dynamic=False, cluster_m=1),
            _quack_config(128, 256, dynamic=True),
        )
    },
)
def F_mla_kvproj_rmsnorm_expand(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse a segmented KV projection, RMSNorm, and expansion projection.

    Match:   KV projection -> split[0] -> RMSNorm -> expansion projection
    Ops:     ``aten.mm.default -> [_RESHAPE_TARGETS] ->
             aten.split_with_sizes.default -> operator.getitem(..., 0) ->
             aten._fused_rms_norm.default -> aten.mm.default``.
    Rewrite: The first ``flex_gemm`` emits weighted values and partial mean
             squares; the second consumes the resulting rstd in its epilogue.
    """
    pattern = F_mla_kvproj_rmsnorm_expand.__name__
    for norm in list(gm.graph.nodes):
        if (
            _is_backward(norm)
            or not _valid_forward_rmsnorm(norm)
            or norm.meta.get("coda_consumed")
        ):
            continue
        norm_input = norm.args[0]
        if (
            not isinstance(norm_input, Node)
            or norm_input.target is not operator.getitem
        ):
            continue
        split = norm_input.args[0]
        if (
            not isinstance(split, Node)
            or split.target is not aten.split_with_sizes.default
            or norm_input.args[1] != 0
            or not isinstance(split.args[0], Node)
        ):
            continue
        full_output = split.args[0]
        path = _chain_to_mm(full_output, allow_cast=False)
        if (
            path is None
            or not _is_reshape_only_path(path)
            or not _path_has_phase(path, backward=False)
        ):
            continue
        full_shape = _shape(full_output)
        norm_shape = _shape(norm_input)
        split_sizes = split.args[1]
        split_dim = split.args[2] if len(split.args) > 2 else split.kwargs.get("dim", 0)
        if (
            full_shape is None
            or norm_shape is None
            or not isinstance(split_sizes, (list, tuple))
            or not all(isinstance(size, int) for size in split_sizes)
            or not isinstance(split_dim, int)
        ):
            continue
        split_dim %= len(full_shape)
        if (
            split_dim != len(full_shape) - 1
            or len(split_sizes) < 2
            or split_sizes[0] != norm_shape[-1]
            or sum(split_sizes) != full_shape[-1]
            or norm_shape[:-1] != full_shape[:-1]
        ):
            continue
        norm_out = _rmsnorm_getitems(norm)
        if norm_out is None:
            continue
        second_use = _downstream_mm(norm_out[0], backward=False)
        if (
            second_use is None
            or second_use.operand != 0
            or _is_backward(second_use.node)
            or not _valid_projection_rmsnorm_use(norm, path, second_use, full_output)
            or not _coda_nodes_available((*path.nodes, norm, second_use.node))
        ):
            continue
        if not _claim_coda_match(pattern, norm):
            continue
        _insert_projection_rmsnorm_fusion(
            gm,
            norm=norm,
            first_path=path,
            second_use=second_use,
            full_output=full_output,
            pattern=pattern,
            benchmark_regions=benchmark_regions,
        )
        counts[pattern] += 1


@register_coda_pattern()
def F_weighted_residual_bmm_prenorm(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse a weighted residual BMM with its following RMSNorm.

    Match:   probabilities @ residual values -> cast -> RMSNorm
    Ops:     ``aten.bmm.default -> aten.squeeze.dim ->
             aten._to_copy.default -> aten._fused_rms_norm.default``.
    Rewrite: ``flex_gemm`` performs the BMM and emits partial mean squares;
             the rstd reduction and final normalization remain outside.
    """
    pattern = F_weighted_residual_bmm_prenorm.__name__
    for norm in list(gm.graph.nodes):
        path = _weighted_residual_rmsnorm_path(norm)
        if path is None or not _claim_coda_match(pattern, norm):
            continue
        # TODO: Use QUACK after it supports local-reduction outputs for BMM.
        _insert_forward_rmsnorm_fusion(
            gm,
            norm=norm,
            path=path,
            pattern=pattern,
            flex_gemm_options={"backend": "TRITON"},
            benchmark_regions=benchmark_regions,
        )
        counts[pattern] += 1


@register_coda_pattern()
def F_mm_residual_rmsnorm(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse a GEMM, residual add chain, and RMSNorm.

    Match:   GEMM -> residual add(s) -> RMSNorm
    Ops:     ``aten.mm.default -> [_RESHAPE_TARGETS] -> aten.add.Tensor ->
             aten._fused_rms_norm.default`` with same-shape residuals.
    Rewrite: ``flex_gemm`` performs the add and emits partial mean squares;
             the rstd reduction and final normalization remain outside.
    """
    pattern = F_mm_residual_rmsnorm.__name__
    for norm in list(gm.graph.nodes):
        path = _residual_rmsnorm_path(norm)
        if path is None or not _claim_coda_match(pattern, norm):
            continue
        _insert_forward_rmsnorm_fusion(
            gm,
            norm=norm,
            path=path,
            pattern=pattern,
            benchmark_regions=benchmark_regions,
        )
        counts[pattern] += 1


@register_coda_pattern()
def F_swiglu(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse structurally matched SwiGLU projection epilogues.

    Match:   X -> {GEMM gate, GEMM up} -> silu(gate) * up
    Ops:     ``aten.mm.default`` descendants containing ``aten.silu.default``.
    Rewrite: Each matched GEMM component becomes one ``flex_gemm`` body.
    """
    pattern = F_swiglu.__name__
    for output in list(gm.graph.nodes):
        match = _match_swiglu(output)
        if match is None or not _claim_coda_match(pattern, output):
            continue
        gate_fused = _insert_flex_gemm(
            gm,
            root=match.gate.root,
            body_nodes={*match.gate.nodes, match.silu},
            pattern=pattern,
            config_index=0,
            fast_math=True,
            benchmark_regions=benchmark_regions,
        )
        if gate_fused is None:
            continue
        up_fused = _insert_flex_gemm(
            gm,
            root=match.up.root,
            body_nodes={*match.up.nodes, match.output},
            pattern=pattern,
            config_index=1,
            fast_math=True,
            benchmark_regions=benchmark_regions,
        )
        if up_fused is not None:
            counts[pattern] += 1


@register_coda_pattern()
def F_situ(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse structurally matched SiTU projection epilogues.

    Match:   X -> {GEMM gate, GEMM up} -> situ(gate) * transform(up)
    Ops:     ``aten.mm.default`` descendants containing ``aten.tanh.default``
             and ``aten.sigmoid.default``.
    Rewrite: Each matched GEMM component becomes one ``flex_gemm`` body.
    """
    pattern = F_situ.__name__
    for output in list(gm.graph.nodes):
        match = _match_situ(output)
        if match is None or not _claim_coda_match(pattern, output):
            continue
        gate_fused = _insert_flex_gemm(
            gm,
            root=match.gate.root,
            body_nodes=match.gate_nodes,
            pattern=pattern,
            config_index=0,
            fast_math=True,
            benchmark_regions=benchmark_regions,
        )
        if gate_fused is None:
            continue
        up_fused = _insert_flex_gemm(
            gm,
            root=match.up.root,
            body_nodes=match.up_nodes,
            pattern=pattern,
            config_index=1,
            fast_math=True,
            benchmark_regions=benchmark_regions,
        )
        if up_fused is not None:
            counts[pattern] += 1


@register_coda_pattern()
def F_k3_mla_output_gate(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse the MLA output gate into its projection epilogue.

    Match:   projection -> sigmoid -> multiply activation
    Ops:     ``aten.mm.default -> [_VIEW_TARGETS] -> aten.sigmoid.default ->
             [_VIEW_TARGETS] -> aten.mul.Tensor``.
    Rewrite: One ``flex_gemm`` body returns the externally used sigmoid and
             gated output.
    """
    pattern = F_k3_mla_output_gate.__name__
    for sigmoid in list(gm.graph.nodes):
        if (
            _is_backward(sigmoid)
            or sigmoid.target is not aten.sigmoid.default
            or sigmoid.meta.get("coda_consumed")
        ):
            continue
        chain = _chain_to_mm(sigmoid.args[0], allow_cast=False)
        mul_match = _find_pointwise_user(sigmoid, aten.mul.Tensor)
        if chain is None or mul_match is None:
            continue
        mul, bridge = mul_match
        if not _claim_coda_match(pattern, sigmoid):
            continue
        fused = _insert_flex_gemm(
            gm,
            root=chain.root,
            body_nodes={*chain.nodes, sigmoid, *bridge, mul},
            pattern=pattern,
            fast_math=True,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern(
    best_configs={10: (_quack_config(256, 256, dynamic=True),)},
)
def F_router_sigmoid_bias(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse router sigmoid and expert bias into the projection epilogue.

    Match:   router projection -> sigmoid -> add expert bias
    Ops:     ``aten.mm.default -> [_VIEW_TARGETS] -> aten.sigmoid.default ->
             [_VIEW_TARGETS] -> aten.add.Tensor``.
    Rewrite: One ``flex_gemm`` body returns the externally used sigmoid and
             bias-adjusted scores.
    """
    pattern = F_router_sigmoid_bias.__name__
    for sigmoid in list(gm.graph.nodes):
        if (
            _is_backward(sigmoid)
            or sigmoid.target is not aten.sigmoid.default
            or sigmoid.meta.get("coda_consumed")
        ):
            continue
        chain = _chain_to_mm(sigmoid.args[0], allow_cast=False)
        add_match = _find_pointwise_user(sigmoid, aten.add.Tensor)
        if chain is None or add_match is None:
            continue
        add, bridge = add_match
        if not _claim_coda_match(pattern, sigmoid):
            continue
        sigmoid_output = bridge[-1] if bridge else sigmoid
        bias = add.args[1] if add.args[0] is sigmoid_output else add.args[0]
        bias_shape = _shape(bias) if isinstance(bias, Node) else None
        if bias_shape is not None and len(bias_shape) == 1:
            with gm.graph.inserting_before(add):
                bias_view = gm.graph.call_function(
                    aten.view.default, args=(bias, [1, bias_shape[0]])
                )
            _infer_new_node_meta(bias_view)
            add.replace_input_with(bias, bias_view)
        fused = _insert_flex_gemm(
            gm,
            root=chain.root,
            body_nodes={*chain.nodes, sigmoid, *bridge, add},
            pattern=pattern,
            fast_math=True,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern(
    best_configs={10: (_quack_config(256, 256, dynamic=True),)},
)
def B_reshape_bf16_to_fp32(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse a reshaped GEMM output's FP32 cast into the GEMM.

    Match:   BF16 GEMM -> reshape -> FP32 cast
    Ops:     backward ``aten.mm.default -> [_RESHAPE_TARGETS] ->
             aten._to_copy.default(dtype=torch.float32)`` with a same-numel
             shape change and no transpose.
    Rewrite: ``flex_gemm`` writes FP32 after preserving the BF16 rounding point.
    """
    pattern = B_reshape_bf16_to_fp32.__name__
    for cast in list(gm.graph.nodes):
        if (
            not _is_backward(cast)
            or cast.target is not _CAST_TARGET
            or _cast_dtype(cast) is not torch.float32
            or cast.meta.get("coda_consumed")
        ):
            continue
        chain = _chain_to_mm(cast)
        if (
            chain is None
            or not _path_has_phase(chain, backward=True)
            or _dtype(chain.root) is not torch.bfloat16
        ):
            continue
        has_transpose = any(
            node.target in {aten.t.default, aten.transpose.int} for node in chain.nodes
        )
        root_shape = _shape(chain.root)
        cast_shape = _shape(cast)
        reshaped_output = (
            root_shape is not None
            and cast_shape is not None
            and root_shape != cast_shape
            and _same_numel(root_shape, cast_shape)
        )
        if has_transpose or not reshaped_output:
            continue
        if not _claim_coda_match(pattern, cast):
            continue
        fused = _insert_flex_gemm(
            gm,
            root=chain.root,
            body_nodes=chain.nodes,
            pattern=pattern,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern(
    best_configs={10: (_quack_config(128, 128, dynamic=True, swap_ab=True),)},
)
def B_swiglu_backward_activation(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse the SwiGLU pointwise backward into its output-gradient GEMM.

    Match:   output-gradient GEMM -> SwiGLU backward -> {gate grad, up grad}
    Ops:     backward ``aten.mm.default`` pointwise descendants containing
             ``aten.silu_backward.default``.
    Rewrite: One ``flex_gemm`` emits both branch gradients.
    """
    pattern = B_swiglu_backward_activation.__name__
    for node in list(gm.graph.nodes):
        if (
            not _is_backward(node)
            or node.target is not _MM_TARGET
            or node.meta.get("coda_consumed")
        ):
            continue
        descendants = _collect_pointwise_descendants(node, allow_other_mm_inputs=True)
        if not any(
            descendant.target is aten.silu_backward.default
            for descendant in descendants
        ):
            continue
        if not _claim_coda_match(pattern, node):
            continue
        outputs = _boundary_outputs(_ordered_nodes(gm, descendants))
        fused = _insert_flex_gemm(
            gm,
            root=node,
            body_nodes=descendants,
            pattern=pattern,
            fast_math=True,
            fused_outputs=outputs,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern()
def B_parallel_mm_dx_merge(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse two parallel GEMM input-gradient branches at their final add.

    Match:   {dX GEMM, dX GEMM} -> add
    Ops:     two backward ``aten.mm.default`` chains into ``aten.add.Tensor``.
    Rewrite: One contributing GEMM and the add become one ``flex_gemm``.
    """
    pattern = B_parallel_mm_dx_merge.__name__
    for add in list(gm.graph.nodes):
        if (
            not _is_backward(add)
            or add.target is not aten.add.Tensor
            or add.meta.get("coda_consumed")
        ):
            continue
        chains = [
            chain for arg in add.args[:2] if (chain := _chain_to_mm(arg)) is not None
        ]
        add_shape = _shape(add)
        add_inputs = [arg for arg in add.args[:2] if isinstance(arg, Node)]
        if (
            len(chains) != 2
            or len(add_inputs) != 2
            or add_shape is None
            or any(_shape(arg) != add_shape for arg in add_inputs)
            or any(not _path_has_phase(chain, backward=True) for chain in chains)
            or any(_dtype(chain.root) is not torch.bfloat16 for chain in chains)
            or not _claim_coda_match(pattern, add)
        ):
            continue
        chain = chains[-1]
        fused = _insert_flex_gemm(
            gm,
            root=chain.root,
            body_nodes={*chain.nodes, add},
            pattern=pattern,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern()
def B_k3_mla_output_gate(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse MLA output-gate backward into the output-projection dX GEMM.

    Match:   output-projection dX GEMM -> multiply branches -> sigmoid backward
    Rewrite: One ``flex_gemm`` emits the attention and gate gradients.
    """
    pattern = B_k3_mla_output_gate.__name__
    for root in list(gm.graph.nodes):
        if (
            not _is_backward(root)
            or root.target is not _MM_TARGET
            or root.meta.get("coda_consumed")
        ):
            continue
        mul_users = [user for user in root.users if user.target is aten.mul.Tensor]
        if len(mul_users) != 2 or set(root.users) != set(mul_users):
            continue
        match = None
        for gate_product, attention_grad in (
            (mul_users[0], mul_users[1]),
            (mul_users[1], mul_users[0]),
        ):
            gate_grad = _single_user_with_target(
                gate_product, aten.sigmoid_backward.default
            )
            if gate_grad is None or gate_grad.args[0] is not gate_product:
                continue
            sigmoid_gate = gate_grad.args[1]
            attention_gate = _other_node_input(attention_grad, root)
            saved_attention = _other_node_input(gate_product, root)
            if (
                not isinstance(sigmoid_gate, Node)
                or attention_gate is None
                or saved_attention is None
                or _alias_source(attention_gate) is not _alias_source(sigmoid_gate)
                or _alias_source(saved_attention) is _alias_source(sigmoid_gate)
            ):
                continue
            root_shape = _shape(root)
            if root_shape is None or any(
                _shape(node) != root_shape
                for node in (gate_product, attention_grad, gate_grad, sigmoid_gate)
            ):
                continue
            match = gate_product, attention_grad, gate_grad
            break
        if match is None:
            continue
        gate_product, attention_grad, gate_grad = match
        body_nodes = {root, gate_product, attention_grad, gate_grad}
        if (
            set(_boundary_outputs(_ordered_nodes(gm, body_nodes)))
            != {attention_grad, gate_grad}
            or not _coda_nodes_available(body_nodes)
            or not _claim_coda_match(pattern, root)
        ):
            continue
        fused = _insert_flex_gemm(
            gm,
            root=root,
            body_nodes=body_nodes,
            pattern=pattern,
            fused_outputs=(attention_grad, gate_grad),
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern()
def B_k3_situ_backward_activation(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse the SiTU pointwise backward into its output-gradient GEMM.

    Match:   output-gradient GEMM -> SiTU backward -> {gate grad, up grad}
    Ops:     backward ``aten.mm.default`` pointwise descendants containing
             both sigmoid and tanh operations.
    Rewrite: One ``flex_gemm`` emits both branch gradients.
    """
    pattern = B_k3_situ_backward_activation.__name__
    sigmoid_targets = {
        aten.sigmoid.default,
        aten.sigmoid_backward.default,
    }
    tanh_targets = {
        aten.tanh.default,
        aten.tanh_backward.default,
    }
    for node in list(gm.graph.nodes):
        if (
            not _is_backward(node)
            or node.target is not _MM_TARGET
            or node.meta.get("coda_consumed")
        ):
            continue
        descendants = _collect_pointwise_descendants(node, allow_other_mm_inputs=True)
        has_silu_backward = any(
            descendant.target is aten.silu_backward.default
            for descendant in descendants
        )
        if (
            has_silu_backward
            or not any(
                descendant.target in sigmoid_targets for descendant in descendants
            )
            or not any(descendant.target in tanh_targets for descendant in descendants)
        ):
            continue
        if not _claim_coda_match(pattern, node):
            continue
        outputs = _boundary_outputs(_ordered_nodes(gm, descendants))
        fused = _insert_flex_gemm(
            gm,
            root=node,
            body_nodes=descendants,
            pattern=pattern,
            fast_math=True,
            fused_outputs=outputs,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern(
    best_configs={10: (_quack_config(256, 256, dynamic=True, cluster_n=2),)},
)
def B_mm_dx_residual_add(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse a GEMM input-gradient branch with a residual-gradient add.

    Match:   dX GEMM branch + residual gradient branch
    Ops:     backward ``aten.add.Tensor`` with exactly one input tracing through
             ``_VIEW_TARGETS`` or ``aten._to_copy.default`` to
             ``aten.mm.default``.
    Rewrite: The GEMM chain and add become one ``flex_gemm`` body.
    """
    pattern = B_mm_dx_residual_add.__name__
    for add in list(gm.graph.nodes):
        if (
            not _is_backward(add)
            or add.target is not aten.add.Tensor
            or add.meta.get("coda_consumed")
        ):
            continue
        chains = [
            chain for arg in add.args[:2] if (chain := _chain_to_mm(arg)) is not None
        ]
        add_shape = _shape(add)
        add_inputs = [arg for arg in add.args[:2] if isinstance(arg, Node)]
        if (
            len(chains) != 1
            or len(add_inputs) != 2
            or add_shape is None
            or any(_shape(arg) != add_shape for arg in add_inputs)
            or not _path_has_phase(chains[0], backward=True)
            or not _claim_coda_match(pattern, add)
        ):
            continue
        chain = chains[0]
        fused = _insert_flex_gemm(
            gm,
            root=chain.root,
            body_nodes={*chain.nodes, add},
            pattern=pattern,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern()
def B_mm_dx_rmsnorm(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse a GEMM input gradient with RMSNorm backward.

    Match:   dX GEMM -> RMSNorm backward
    Ops:     backward ``aten.mm.default -> [_RESHAPE_TARGETS] ->
             aten._fused_rms_norm_backward.default``.
    Rewrite: ``flex_gemm`` emits d_x_hat and partial row dots; the remaining
             reduction and pointwise operations produce input and weight grads.
    """
    pattern = B_mm_dx_rmsnorm.__name__
    for node in list(gm.graph.nodes):
        if (
            not _is_backward(node)
            or not _valid_backward_rmsnorm(node)
            or node.meta.get("coda_consumed")
        ):
            continue
        path = _chain_to_mm(node.args[0], allow_cast=False)
        if (
            path is None
            or not _is_reshape_only_path(path)
            or not _path_has_phase(path, backward=True)
            or not _coda_nodes_available((*path.nodes, node))
            or not _claim_coda_match(pattern, node)
        ):
            continue
        _insert_backward_rmsnorm_fusion(
            gm,
            norm_backward=node,
            path=path,
            pattern=pattern,
            benchmark_regions=benchmark_regions,
        )
        counts[pattern] += 1


@register_coda_pattern(tune_split_k=True)
def B_linear_dw_bf16_to_fp32(  # noqa: N802
    gm: GraphModule,
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    """Fuse a linear weight-gradient FP32 cast into its BF16 GEMM.

    Match:   BF16 weight-gradient GEMM -> optional view/transpose -> FP32 cast
    Ops:     backward ``aten.mm.default -> [_VIEW_TARGETS] ->
             aten._to_copy.default(dtype=torch.float32)`` excluding the
             reshape-only LM-head dX case.
    Rewrite: ``flex_gemm`` writes FP32 after preserving the BF16 rounding point;
             transpose and shape-only operations remain outside when needed.
    """
    pattern = B_linear_dw_bf16_to_fp32.__name__
    for cast in list(gm.graph.nodes):
        if (
            not _is_backward(cast)
            or cast.target is not _CAST_TARGET
            or _cast_dtype(cast) is not torch.float32
            or cast.meta.get("coda_consumed")
        ):
            continue
        chain = _chain_to_mm(cast)
        if (
            chain is None
            or not _path_has_phase(chain, backward=True)
            or _dtype(chain.root) is not torch.bfloat16
        ):
            continue
        has_transpose = any(
            node.target in {aten.t.default, aten.transpose.int} for node in chain.nodes
        )
        root_shape = _shape(chain.root)
        cast_shape = _shape(cast)
        reshaped_output = (
            root_shape is not None
            and cast_shape is not None
            and root_shape != cast_shape
            and _same_numel(root_shape, cast_shape)
        )
        if reshaped_output and not has_transpose:
            continue
        if not _claim_coda_match(pattern, cast):
            continue
        if has_transpose:
            if not _insert_transposed_cast_fusion(
                gm,
                chain=chain,
                cast=cast,
                pattern=pattern,
                benchmark_regions=benchmark_regions,
            ):
                continue
        else:
            fused = _insert_flex_gemm(
                gm,
                root=chain.root,
                body_nodes=chain.nodes,
                pattern=pattern,
                benchmark_regions=benchmark_regions,
            )
            if fused is None:
                continue
        counts[pattern] += 1


CODA_PATTERN_NAMES = tuple(_CODA_PATTERNS)


def _prepare_coda_graph(
    gm: GraphModule,
    *,
    eliminate_dead_code: bool = True,
    recompile: bool = True,
) -> GraphModule:
    _stable_topological_sort(gm.graph, {})
    if eliminate_dead_code:
        gm.graph.eliminate_dead_code()
    _assign_coda_inductor_regions(gm)
    gm.graph.lint()
    if recompile:
        gm.recompile()
    return gm


def _prepare_coda_graph_for_next_pass(gm: GraphModule) -> GraphModule:
    return _prepare_coda_graph(gm, recompile=False)


def materialize_coda_inductor_regions_pass(
    gm: GraphModule,
    example_inputs: tuple | None = None,
) -> GraphModule:
    """Fuse already-grouped CODA nodes without repeated region discovery."""
    del example_inputs
    _prepare_coda_graph(gm, recompile=False)
    groups: dict[str, dict[Node, int | None]] = {}
    for node in gm.graph.nodes:
        group = _coda_region_group(node)
        if group is not None:
            groups.setdefault(group, {})[node] = None
    if not groups:
        gm.recompile()
        return gm
    fuse_by_partitions(
        gm,
        list(groups.values()),
        prefix="__marked_inductor_submod_coda_",
        always_return_tuple=True,
    )
    gm.recompile()
    logger.info(f"Materialized {len(groups)} CODA FlexGEMM regions")
    return gm


def _finalize_coda_graph(
    gm: GraphModule,
    counts: Counter[str],
    *,
    log_matches: bool = True,
) -> GraphModule:
    _prepare_coda_graph(gm)
    gm.meta["coda_pattern_counts"] = dict(counts)
    if log_matches:
        matched = {name: count for name, count in counts.items() if count}
        logger.info(f"CODA FlexGEMM matched {sum(matched.values())} groups: {matched}")
    return gm


def _copy_graph_module_for_rewrite(gm: GraphModule) -> GraphModule:
    graph = torch.fx.Graph(
        tracer_cls=gm.graph._tracer_cls,
        tracer_extras=gm.graph._tracer_extras,
    )
    graph.set_codegen(copy.deepcopy(gm.graph._codegen))
    values: dict[Node, Node] = {}
    for node in gm.graph.nodes:
        copied = graph.node_copy(node, values.__getitem__)
        custom = copied.meta.get("custom")
        if isinstance(custom, dict):
            copied.meta["custom"] = copy.deepcopy(custom)
        values[node] = copied

    result = GraphModule.__new__(GraphModule)
    result.__dict__ = gm.__dict__.copy()
    result._modules = gm._modules.copy()
    result.__dict__["_graph"] = graph
    graph.owning_module = result
    result.meta = copy.deepcopy(gm.meta)
    return result


def _apply_coda_pattern(
    gm: GraphModule,
    pattern: CodaPattern,
    *,
    log_matches: bool = True,
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    finalize: bool = True,
) -> GraphModule:
    gm = _copy_graph_module_for_rewrite(gm)
    counts: Counter[str] = Counter({name: 0 for name in CODA_PATTERN_NAMES})
    counts.update(gm.meta.get("coda_pattern_counts", {}))
    pattern.matcher(gm, counts, benchmark_regions)
    if finalize:
        return _finalize_coda_graph(gm, counts, log_matches=log_matches)
    gm.meta["coda_pattern_counts"] = dict(counts)
    return gm


def _apply_coda_candidate(
    gm: GraphModule,
    selection: BenchmarkCandidateSelection,
    benchmark_regions: list[RewriteBenchmarkRegion],
    *,
    pattern: CodaPattern,
) -> GraphModule:
    selection_token = _CODA_MATCH_SELECTION.set(selection)
    try:
        return _apply_coda_pattern(
            gm,
            pattern,
            log_matches=False,
            benchmark_regions=(
                None if selection.accepted is not None else benchmark_regions
            ),
            finalize=not selection.defer_finalize,
        )
    finally:
        _CODA_MATCH_SELECTION.reset(selection_token)


_CODA_PATTERN_ALIASES = {
    "f2-q": ("F_mla_qproj_rmsnorm_expand",),
    "f2-kv": ("F_mla_kvproj_rmsnorm_expand",),
    "f-attnres-prenorm": ("F_weighted_residual_bmm_prenorm",),
    "F_attnout_residual_ffn_prenorm": ("F_mm_residual_rmsnorm",),
    "F_sharedE_out_residual_attn_prenorm": ("F_mm_residual_rmsnorm",),
    "F_dense_ffnout_residual_attn_prenorm": ("F_mm_residual_rmsnorm",),
    "f3-attention": ("F_mm_residual_rmsnorm",),
    "f3-shared": ("F_mm_residual_rmsnorm",),
    "f3-dense": ("F_mm_residual_rmsnorm",),
    "F_sharedE_swiglu": ("F_swiglu",),
    "F_dense_ffn_swiglu": ("F_swiglu",),
    "f4-shared": ("F_swiglu",),
    "f4-dense": ("F_swiglu",),
    "F_k3_sharedE_situ": ("F_situ",),
    "F_k3_dense_ffn_situ": ("F_situ",),
    "k3-f4-shared-situ": ("F_situ",),
    "k3-f4-dense-situ": ("F_situ",),
    "k3-mla-output-gate": ("F_k3_mla_output_gate",),
    "f6-router": ("F_router_sigmoid_bias",),
    "B_lmhead_dx_bf16_to_fp32": ("B_reshape_bf16_to_fp32",),
    "b1-lm-head-cast": ("B_reshape_bf16_to_fp32",),
    "B_swiglu_dx_merge": ("B_parallel_mm_dx_merge",),
    "B_swiglu_backward_dx_merge": (
        "B_swiglu_backward_activation",
        "B_parallel_mm_dx_merge",
    ),
    "b2-shared-swiglu": (
        "B_swiglu_backward_activation",
        "B_parallel_mm_dx_merge",
    ),
    "k3-b-output-gate": ("B_k3_mla_output_gate",),
    "B_k3_situ_dx_merge": ("B_parallel_mm_dx_merge",),
    "B_k3_situ_backward_dx_merge": (
        "B_k3_situ_backward_activation",
        "B_parallel_mm_dx_merge",
    ),
    "k3-b2-shared-situ": (
        "B_k3_situ_backward_activation",
        "B_parallel_mm_dx_merge",
    ),
    "B_router_dx_cast_expert_merge": ("B_mm_dx_residual_add",),
    "b4-router-input-grad": ("B_mm_dx_residual_add",),
    "B_routed_up_dx_rmsnorm": ("B_mm_dx_rmsnorm",),
    "B_mla_qproj_dx_rmsnorm": ("B_mm_dx_rmsnorm",),
    "B_mla_kvproj_dx_rmsnorm": ("B_mm_dx_rmsnorm",),
    "b5-routed-rmsnorm": ("B_mm_dx_rmsnorm",),
    "b5-q-rmsnorm": ("B_mm_dx_rmsnorm",),
    "b5-kv-rmsnorm": ("B_mm_dx_rmsnorm",),
    "b6-weight-grad-cast": ("B_linear_dw_bf16_to_fp32",),
    "B_mla_qkv_dx_merge": ("B_parallel_mm_dx_merge",),
    "b7-attention-grad-merge": ("B_parallel_mm_dx_merge",),
}


def _resolve_coda_patterns(patterns: Iterable[str] | None) -> tuple[str, ...]:
    requested = list(CODA_PATTERN_NAMES if not patterns else patterns)
    duplicates = sorted(name for name in set(requested) if requested.count(name) > 1)
    if duplicates:
        raise ValueError(f"Duplicate CODA pattern entries: {duplicates}")
    resolved = [
        resolved_name
        for name in requested
        for resolved_name in _CODA_PATTERN_ALIASES.get(name, (name,))
    ]
    unknown = sorted(set(resolved) - _CODA_PATTERNS.keys())
    if unknown:
        raise ValueError(
            f"Unknown CODA pattern entries: {unknown}; "
            f"supported patterns: {sorted(CODA_PATTERN_NAMES)}"
        )
    enabled = (_CODA_PATTERNS[name] for name in set(resolved))
    return tuple(
        pattern.name for pattern in sorted(enabled, key=lambda item: item.priority)
    )


def _configured_coda_pass(
    pattern: CodaPattern,
    *,
    compile_time_benchmark: bool,
    benchmark_strict: bool,
    coda_autotune: bool,
) -> Callable:
    @functools.wraps(_apply_coda_pattern)
    def apply(
        gm: GraphModule,
        example_inputs: tuple | None = None,
    ) -> GraphModule:
        configs = _best_configs(pattern.name)
        if configs:
            logger.info(f"CODA {pattern.name} using pinned FlexGEMM configs: {configs}")
        elif coda_autotune:
            logger.info(f"CODA {pattern.name} using FlexGEMM autotuning")
        else:
            logger.info(f"CODA {pattern.name} using the default FlexGEMM config")
        token = _CODA_AUTOTUNE.set(coda_autotune)
        try:
            if compile_time_benchmark:
                return apply_benchmarked_rewrites(
                    gm,
                    rewrite_name=f"CODA {pattern.name}",
                    apply_candidate=functools.partial(
                        _apply_coda_candidate,
                        pattern=pattern,
                    ),
                    namespace=("coda_flex_gemm", pattern.name, coda_autotune),
                    strict=benchmark_strict,
                    report_title=f"CODA benchmark results for {pattern.name}",
                    artifact_name=f"coda_benchmark_{pattern.name}",
                    candidate_prefix=f"{pattern.name}:",
                    candidate_label="FlexGEMM",
                    finalize=_prepare_coda_graph_for_next_pass,
                    batch_candidates=True,
                )
            return _apply_coda_pattern(gm, pattern)
        finally:
            _CODA_AUTOTUNE.reset(token)

    apply.__name__ = pattern.name
    return apply


def get_coda_pattern_passes(
    patterns: Iterable[str] | None = None,
    *,
    compile_time_benchmark: bool | None = None,
    benchmark_strict: bool = False,
    coda_autotune: bool | None = None,
) -> list[Callable]:
    """Resolve CODA names to benchmark-gated, independently logged passes."""
    benchmark = compile_time_benchmark is not False
    autotune = coda_autotune is not False
    return [
        _configured_coda_pass(
            _CODA_PATTERNS[name],
            compile_time_benchmark=benchmark,
            benchmark_strict=benchmark_strict,
            coda_autotune=autotune,
        )
        for name in _resolve_coda_patterns(patterns)
    ]


def coda_flex_gemm_pass(
    gm: GraphModule,
    example_inputs: tuple | None = None,
    *,
    patterns: Iterable[str] | None = None,
    compile_time_benchmark: bool | None = None,
    benchmark_strict: bool = False,
    coda_autotune: bool | None = None,
) -> GraphModule:
    """Apply selected CODA pattern passes to a joint training graph."""
    for pattern_pass in get_coda_pattern_passes(
        patterns,
        compile_time_benchmark=compile_time_benchmark,
        benchmark_strict=benchmark_strict,
        coda_autotune=coda_autotune,
    ):
        gm = pattern_pass(gm, example_inputs)
    return gm
