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

import inspect
import operator
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

import torch
import torch.fx as fx
import torch.fx.node
import torch.utils._pytree as pytree
from torch.distributed.pipelining.microbatch import _split_block_mask
from torch.distributed.pipelining.schedules import (
    _Action,
    BACKWARD_INPUT,
    FORWARD,
    FULL_BACKWARD,
)
from torch._logging import trace_structured
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


def _resplit_block_mask_with_torch_pipelining(
    block_mask: BlockMask,
    *,
    num_microbatches: int,
) -> BlockMask:
    """Use PyTorch PP's BlockMask splitter for GraphPP microbatch inputs.

    Older split BlockMask objects can carry a closure pointing back to the
    original full-batch BlockMask plus this microbatch's index. Re-split that
    original value with PyTorch PP's splitter instead of maintaining a separate
    GraphPP BlockMask construction path.
    """
    mask_mod = block_mask.mask_mod
    if not inspect.isfunction(mask_mod) or mask_mod.__closure__ is None:
        return block_mask

    closure_values = {
        name: cell.cell_contents
        for name, cell in zip(mask_mod.__code__.co_freevars, mask_mod.__closure__)
    }
    base_block_mask = closure_values.get("block_mask")
    microbatch_index = closure_values.get("idx")
    if not isinstance(base_block_mask, BlockMask) or not isinstance(
        microbatch_index, int
    ):
        return block_mask
    if microbatch_index >= num_microbatches:
        raise ValueError(
            "Split BlockMask microbatch index is outside the schedule: "
            f"index={microbatch_index}, num_microbatches={num_microbatches}"
        )
    return _split_block_mask(base_block_mask, num_microbatches)[microbatch_index]


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
    num_microbatches = len(args_split)

    def normalize_leaf(value: Any) -> Any:
        if isinstance(value, BlockMask):
            return _resplit_block_mask_with_torch_pipelining(
                value,
                num_microbatches=num_microbatches,
            )
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
