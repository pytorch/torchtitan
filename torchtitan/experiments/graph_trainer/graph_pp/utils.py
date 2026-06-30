# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Small GraphPP helpers shared by graph extraction passes and runtime.

The wrapping helpers deliberately delegate to ``make_fx_tracer``'s
``_unwrap_subclasses`` / ``_wrap_subclasses`` implementation. GraphPP keeps only
slice metadata for values that cross PP boundaries; internal saved values,
FSDP params, and raw grad leaves stay in the flat tracer calling convention.
"""

import copy
import hashlib
import inspect
import operator
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

import sympy
import torch
import torch.fx as fx
import torch.fx.node
import torch.utils._pytree as pytree
from torch._dynamo.source import ConstantSource
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.distributed.pipelining.schedules import (
    _Action,
    BACKWARD_INPUT,
    FORWARD,
    FULL_BACKWARD,
)
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _unwrap_subclasses,
    _wrap_subclasses,
    SubclassLayout,
)

T = TypeVar("T")


def trace_graph_pp_graph(name: str, gm: fx.GraphModule) -> None:
    """Emit a readable FX graph artifact for tlparse."""

    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": name,
            "encoding": "string",
        },
        payload_fn=lambda: gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
        ),
    )


def _iter_meta_leaves(value: Any):
    if isinstance(value, Mapping):
        for key, val in value.items():
            yield from _iter_meta_leaves(key)
            yield from _iter_meta_leaves(val)
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _iter_meta_leaves(item)
    else:
        yield value


def _find_fake_mode(gm: fx.GraphModule) -> FakeTensorMode | None:
    """Return the first FakeTensorMode referenced by graph node metadata."""

    for node in gm.graph.nodes:
        for value in node.meta.values():
            for leaf in _iter_meta_leaves(value):
                if isinstance(leaf, FakeTensor):
                    return leaf.fake_mode
    return None


class _MetaShapeEnvTransfer:
    """Move copied FX metadata into a destination ShapeEnv.

    Contract:
      A copied FX graph must expose metadata owned by one FakeTensorMode and one
      ShapeEnv before full Inductor sees it. Callers pick the destination graph
      whose ShapeEnv must be preserved, then copy foreign metadata into it.

      PyTorch owns FakeTensor metadata transfer. GraphPP only pre-seeds foreign
      unbacked base symbols from FakeTensor sizes/strides/offsets and raw
      SymInt/SymExpr leaves stored directly in node.meta. Derived expressions
      are copied after those bases exist in the destination ShapeEnv, preserving
      relations such as ``u0 + u1`` instead of turning the whole expression into
      one opaque unbacked symbol.

    TODO(sanketpurandare): Replace this class when PyTorch exposes a metadata
    tree transfer utility that first seeds raw foreign unbacked base symbols and
    then rewrites derived SymPy expressions while preserving bindings.
    """

    def __init__(self, dst_fake_mode: FakeTensorMode | None) -> None:
        self.dst_fake_mode = dst_fake_mode
        self.dst_shape_env = None if dst_fake_mode is None else dst_fake_mode.shape_env
        self._symbol_sources: dict[sympy.Symbol, object] = {}

    def collect(self, meta: dict[str, Any]) -> None:
        if self.dst_shape_env is None:
            return
        for leaf in _iter_meta_leaves(meta):
            if isinstance(leaf, FakeTensor):
                self._collect_fake_tensor(leaf)
            elif isinstance(leaf, torch.SymInt):
                self._collect_symint(leaf)

    def seed(self) -> None:
        if self.dst_shape_env is None:
            return
        for symbol, src_shape_env in sorted(
            self._symbol_sources.items(), key=lambda item: str(item[0])
        ):
            cache_key = (id(src_shape_env), symbol)
            if cache_key in self.dst_shape_env.foreign_unbacked_symbol_cache:
                continue
            symint = src_shape_env.create_symintnode(symbol, hint=None)
            self.dst_shape_env._transfer_foreign_expr_as_unbacked(
                symint,
                source=self._source(src_shape_env, symbol),
            )

    def copy_meta(self, meta: dict[str, Any]) -> dict[str, Any]:
        if self.dst_fake_mode is None or self.dst_shape_env is None:
            return copy.copy(meta)
        return self._copy_value(meta)

    def _source(self, src_shape_env: object, expr: sympy.Expr) -> ConstantSource:
        source_hash = hashlib.sha1(f"{id(src_shape_env)}:{expr!s}".encode()).hexdigest()
        return ConstantSource(f"graph_pp_multiplex_meta_{source_hash}")

    def _collect_fake_tensor(self, value: FakeTensor) -> None:
        if value.fake_mode is self.dst_fake_mode:
            return
        for dim in (*value.size(), *value.stride(), value.storage_offset()):
            if isinstance(dim, torch.SymInt):
                self._collect_symint(dim)

    def _collect_symint(self, value: torch.SymInt) -> None:
        src_shape_env = value.node.shape_env
        if src_shape_env is None or src_shape_env is self.dst_shape_env:
            return
        for symbol in value.node.expr.free_symbols:
            if src_shape_env.has_guarding_hint(symbol):
                continue
            existing = self._symbol_sources.get(symbol)
            if existing is not None and existing is not src_shape_env:
                raise RuntimeError(
                    "GraphPP cannot disambiguate same-named SymInt metadata "
                    "symbols from multiple foreign ShapeEnvs."
                )
            self._symbol_sources[symbol] = src_shape_env

    def _copy_value(self, value: Any) -> Any:
        if isinstance(value, FakeTensor):
            return self._copy_fake_tensor(value)
        if isinstance(value, torch.SymInt):
            return self._copy_symint(value)
        if isinstance(value, sympy.Expr):
            return self._copy_sympy_expr(value)
        if isinstance(value, Mapping):
            return {
                self._copy_value(key): self._copy_value(val)
                for key, val in value.items()
            }
        if isinstance(value, tuple):
            return tuple(self._copy_value(item) for item in value)
        if isinstance(value, list):
            return [self._copy_value(item) for item in value]
        if isinstance(value, set):
            return {self._copy_value(item) for item in value}
        if isinstance(value, frozenset):
            return frozenset(self._copy_value(item) for item in value)
        return value

    def _copy_fake_tensor(self, value: FakeTensor) -> FakeTensor:
        if value.fake_mode is self.dst_fake_mode:
            return value
        try:
            with self.dst_fake_mode:
                # Delegate tensor shape metadata transfer to PyTorch.
                return self.dst_fake_mode.from_tensor(value)
        except Exception as exc:
            raise RuntimeError(
                "GraphPP failed to transfer multiplexed graph FakeTensor "
                "metadata into the destination graph FakeTensorMode."
            ) from exc

    def _copy_symint(self, value: torch.SymInt) -> torch.SymInt:
        src_shape_env = value.node.shape_env
        if src_shape_env is None or src_shape_env is self.dst_shape_env:
            return value
        try:
            expr = self.dst_shape_env._transfer_foreign_expr_as_unbacked(
                value,
                source=self._source(src_shape_env, value.node.expr),
            )
            return self.dst_shape_env.create_symintnode(
                expr,
                hint=None,
                source=self._source(src_shape_env, value.node.expr),
            )
        except Exception as exc:
            raise RuntimeError(
                "GraphPP failed to transfer multiplexed graph SymInt metadata "
                "into the destination graph ShapeEnv."
            ) from exc

    def _copy_sympy_expr(self, value: sympy.Expr) -> sympy.Expr:
        replacements = {}
        for symbol in value.free_symbols:
            src_shape_env = self._symbol_sources.get(symbol)
            if src_shape_env is None:
                continue
            cache_key = (id(src_shape_env), symbol)
            if cache_key not in self.dst_shape_env.foreign_unbacked_symbol_cache:
                raise RuntimeError(
                    "GraphPP missing seeded ShapeEnv symbol while copying "
                    f"multiplexed metadata: {symbol}"
                )
            replacements[symbol] = self.dst_shape_env.foreign_unbacked_symbol_cache[
                cache_key
            ]
        return value.xreplace(replacements) if replacements else value


def graph_outputs(graph: fx.Graph) -> tuple[Any, ...]:
    """Return the flattened values from the graph's single output node."""

    outputs = graph.find_nodes(op="output")
    if len(outputs) != 1:
        raise ValueError(f"Expected one output node, found {len(outputs)}")
    return tuple(pytree.arg_tree_leaves(outputs[0].args[0]))


def graph_module_outputs(gm: fx.GraphModule) -> tuple[Any, ...]:
    """Return flattened output values from ``gm``."""

    return graph_outputs(gm.graph)


def _value_name(value: Any, index: int) -> str:
    if isinstance(value, fx.Node):
        return value.name
    return f"output_{index}"


def placeholder_names(gm: fx.GraphModule) -> tuple[str, ...]:
    """Return graph placeholder names in calling-convention order."""

    return tuple(node.name for node in gm.graph.find_nodes(op="placeholder"))


def output_names(gm: fx.GraphModule) -> tuple[str, ...]:
    """Return stable names for flattened graph outputs."""

    return tuple(
        _value_name(value, index)
        for index, value in enumerate(graph_module_outputs(gm))
    )


def node_closure(values: Iterable[object]) -> set[fx.Node]:
    """Return all FX nodes needed to compute ``values``.

    Graph output leaves may be FX nodes or Python literals such as ``None``.
    Literal leaves are ignored because they do not add graph dependencies.
    """

    closure: set[fx.Node] = set()
    queue = [value for value in values if isinstance(value, fx.Node)]
    while queue:
        node = queue.pop()
        if node in closure:
            continue
        closure.add(node)
        queue.extend(node.all_input_nodes)
    return closure


def placeholder_dependencies(values: Iterable[object]) -> set[fx.Node]:
    """Return placeholder nodes in the dependency closure of ``values``."""

    return {node for node in node_closure(values) if node.op == "placeholder"}


def node_order(graph: fx.Graph) -> dict[fx.Node, int]:
    """Map each node in ``graph`` to its topological position."""

    return {node: index for index, node in enumerate(graph.nodes)}


def unique_in_order(values: Iterable[T]) -> list[T]:
    """Return ``values`` without duplicates, preserving first occurrence."""

    return list(dict.fromkeys(values).keys())


def is_getitem_node(node: fx.Node) -> bool:
    return node.op == "call_function" and node.target is operator.getitem


def is_mutation_node(node: fx.Node) -> bool:
    """Return whether ``node`` writes to one of its aliased arguments."""

    if node.op != "call_function":
        return False
    # ``target`` is a Torch operator overload when this metadata exists.
    # Python callables/literals do not have schemas and cannot be mutations.
    schema = getattr(node.target, "_schema", None)
    if schema is None:
        return False
    return any(
        arg.alias_info is not None and arg.alias_info.is_write
        for arg in schema.arguments
    )


def base_tensor_for_mutation_target(value: object) -> fx.Node | None:
    """Return the placeholder/intermediate base reached through view nodes."""

    if not isinstance(value, fx.Node):
        return None
    node = value
    while (
        node.op == "call_function"
        # Operator overloads expose ``is_view``; other call targets do not.
        and hasattr(node.target, "is_view")
        and node.target.is_view
        and node.args
        and isinstance(node.args[0], fx.Node)
    ):
        node = node.args[0]
    return node


def is_fake_tensor_node(node: fx.Node) -> bool:
    return isinstance(node.meta.get("val"), torch._subclasses.FakeTensor)


def rename_placeholder(
    gm: fx.GraphModule,
    node: fx.Node,
    new_name: str,
) -> None:
    """Replace a placeholder with a new placeholder that has ``new_name``."""

    if node.op != "placeholder":
        raise ValueError(f"Can only rename placeholder nodes, got {node.op}")
    with gm.graph.inserting_before(node):
        new_node = gm.graph.placeholder(new_name)
    new_node.meta.update(node.meta)
    node.replace_all_uses_with(new_node)
    gm.graph.erase_node(node)


def example_inputs_from_placeholders(gm: fx.GraphModule) -> tuple[Any, ...]:
    """Read fake/example values from placeholder metadata."""

    example_inputs = []
    for node in gm.graph.find_nodes(op="placeholder"):
        if "val" not in node.meta:
            raise ValueError(
                "GraphPP cannot compile graph without placeholder metadata: "
                f"{node.name}"
            )
        example_inputs.append(node.meta["val"])
    return tuple(example_inputs)


@contextmanager
def allow_fx_graph_extraction_of_side_effectful_ops(exclude_vals: set[object]):
    """Temporarily allow FX graph extraction to copy selected side-effect ops."""

    original_val = torch.fx.node._side_effectful_functions.copy()
    try:
        torch.fx.node._side_effectful_functions -= exclude_vals
        yield
    finally:
        torch.fx.node._side_effectful_functions.clear()
        torch.fx.node._side_effectful_functions.update(original_val)


def overlap_fw_bw_sub_actions(action: _Action) -> tuple[_Action, _Action]:
    """Validate an ``OVERLAP_F_B`` action and return ``(fw_action, bw_action)``."""

    if action.sub_actions is None or len(action.sub_actions) != 2:
        raise ValueError(f"GraphPP OVERLAP_F_B action is malformed: {action}")
    fw_action, bw_action = action.sub_actions
    if fw_action.computation_type != FORWARD:
        raise ValueError(f"GraphPP OVERLAP_F_B first sub-action must be F: {action}")
    if bw_action.computation_type == BACKWARD_INPUT:
        raise NotImplementedError(
            "GraphPP OVERLAP_F_B with BACKWARD_INPUT is not implemented. "
            "Current multiplexed graphs support FORWARD + FULL_BACKWARD only."
        )
    if bw_action.computation_type != FULL_BACKWARD:
        raise ValueError(
            "GraphPP OVERLAP_F_B second sub-action must be FULL_BACKWARD: " f"{action}"
        )
    return fw_action, bw_action


def _subclass_layout_slice(
    layouts: dict[int, SubclassLayout],
    *,
    start: int,
    count: int,
) -> dict[int, SubclassLayout]:
    return {
        index - start: layout
        for index, layout in layouts.items()
        if start <= index < start + count
    }


def _num_unwrapped_values(
    num_values: int,
    layouts: dict[int, SubclassLayout],
) -> int:
    return sum(
        layouts[index].num_tensors if index in layouts else 1
        for index in range(num_values)
    )


@dataclass(frozen=True, slots=True)
class GraphPPValueSpec:
    """Subclass/tree metadata for one semantic slice of a flat GraphPP API.

    This is a thin view over ``TracedResult.output_subclass_layouts``. It is
    used only for values exposed to the PP/runtime boundary: forward outputs,
    input gradients, and parameter gradients before accumulation.
    """

    num_values: int
    layouts: dict[int, SubclassLayout]
    tree_spec: pytree.TreeSpec | None = None

    @property
    def num_flat_values(self) -> int:
        return _num_unwrapped_values(self.num_values, self.layouts)

    def wrap_flat_values(self, values: tuple[Any, ...] | list[Any]) -> list[Any]:
        return _wrap_unwrapped_values(
            values,
            num_values=self.num_values,
            layouts=self.layouts,
        )

    def unflatten(self, values: tuple[Any, ...] | list[Any]) -> Any:
        wrapped = self.wrap_flat_values(values)
        if self.tree_spec is None:
            return wrapped
        return pytree.tree_unflatten(wrapped, self.tree_spec)


def graph_pp_value_spec(
    layouts: dict[int, SubclassLayout],
    *,
    start: int,
    count: int,
    tree_spec: pytree.TreeSpec | None = None,
) -> GraphPPValueSpec:
    return GraphPPValueSpec(
        num_values=count,
        layouts=_subclass_layout_slice(layouts, start=start, count=count),
        tree_spec=tree_spec,
    )


def flatten_graph_values(values: list[Any]) -> list[Any]:
    flat_values, _ = _unwrap_subclasses(values)
    return flat_values


def _wrap_unwrapped_values(
    values: tuple[Any, ...] | list[Any],
    *,
    num_values: int,
    layouts: dict[int, SubclassLayout],
) -> list[Any]:
    expected = _num_unwrapped_values(num_values, layouts)
    if len(values) != expected:
        raise ValueError(
            "GraphPP subclass rewrap count mismatch: "
            f"expected {expected} raw values for {num_values} semantic values, "
            f"got {len(values)}"
        )
    return _wrap_subclasses(values, num_values, layouts)


def _graphable_split_block_mask(
    block_mask: BlockMask,
) -> BlockMask:
    """Make a PyTorch PP-split BlockMask replayable by one traced graph.

    ``_split_block_mask`` correctly creates per-microbatch tensor metadata, but
    its ``mask_mod`` closure captures the batch offset as a Python int. ``make_fx``
    specializes that int, so a graph traced on microbatch 0 would replay every
    later microbatch with offset 0. Rebuild only the closure so the offset is a
    scalar tensor leaf while preserving PyTorch PP's split tensor metadata.

    TODO(sanketpurandare): Delete this shim once upstream PP ``_split_block_mask``
    captures the microbatch batch offset as a scalar tensor closure leaf instead
    of a Python int. Then normal ``BlockMask._flatten`` will expose all
    replay-varying state to GraphPP tracing.
    """
    mask_mod = block_mask.mask_mod
    if not inspect.isfunction(mask_mod) or mask_mod.__closure__ is None:
        return block_mask

    closure_values = {
        name: cell.cell_contents
        for name, cell in zip(mask_mod.__code__.co_freevars, mask_mod.__closure__)
    }
    base_block_mask = closure_values.get("block_mask")
    batch_offset_index = closure_values.get("idx")
    if not isinstance(base_block_mask, BlockMask) or not isinstance(
        batch_offset_index, int
    ):
        return block_mask
    base_batch_size = base_block_mask.kv_num_blocks.size(0)
    if batch_offset_index < 0 or batch_offset_index >= base_batch_size:
        raise ValueError(
            "Split BlockMask batch offset is outside the base BlockMask: "
            f"offset={batch_offset_index}, batch_size={base_batch_size}"
        )
    batch_offset = torch.tensor(
        batch_offset_index,
        device=block_mask.kv_num_blocks.device,
        dtype=torch.int64,
    )
    base_mask_mod = base_block_mask.mask_mod

    def graphable_mask_mod(
        b: torch.Tensor,
        h: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        offset = batch_offset.to(device=b.device, dtype=b.dtype)
        return base_mask_mod(b + offset, h, q_idx, kv_idx)

    return BlockMask(
        kv_num_blocks=block_mask.kv_num_blocks,
        kv_indices=block_mask.kv_indices,
        full_kv_num_blocks=block_mask.full_kv_num_blocks,
        full_kv_indices=block_mask.full_kv_indices,
        q_num_blocks=block_mask.q_num_blocks,
        q_indices=block_mask.q_indices,
        full_q_num_blocks=block_mask.full_q_num_blocks,
        full_q_indices=block_mask.full_q_indices,
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
        mask_mod=graphable_mask_mod,
        seq_lengths=block_mask.seq_lengths,
        dq_write_order=block_mask.dq_write_order,
        dq_write_order_full=block_mask.dq_write_order_full,
        dq_kv_order=block_mask.dq_kv_order,
        dq_kv_order_spt=block_mask.dq_kv_order_spt,
    )


def normalize_graph_pp_microbatch_inputs(
    args_split: list[Any],
    kwargs_split: list[Any],
) -> tuple[list[Any], list[Any]]:
    """Normalize PP-split inputs so one GraphPP trace can replay all microbatches."""
    if len(args_split) != len(kwargs_split):
        raise ValueError(
            "GraphPP expected args_split and kwargs_split to have the same "
            f"microbatch count, got {len(args_split)} and {len(kwargs_split)}"
        )

    def normalize_leaf(value: Any) -> Any:
        if isinstance(value, BlockMask):
            return _graphable_split_block_mask(value)
        return value

    def normalize_tree(value: Any) -> Any:
        return pytree.tree_map(
            normalize_leaf,
            value,
            is_leaf=lambda leaf: isinstance(leaf, BlockMask),
        )

    return (
        [normalize_tree(args) for args in args_split],
        [normalize_tree(kwargs) for kwargs in kwargs_split],
    )
