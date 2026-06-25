# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Small GraphPP helpers shared by graph extraction passes and runtime.

The wrapping helpers deliberately delegate to ``make_fx_tracer``'s
``_unwrap_subclasses`` / ``_wrap_subclasses`` implementation. GraphPP keeps only
slice metadata for values that cross PP boundaries; internal saved values,
FSDP params, and raw grad leaves stay in the flat tracer ABI.
"""

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch import nn
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from torch.nn.attention.flex_attention import BlockMask

from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _unwrap_subclasses,
    _wrap_subclasses,
    SubclassLayout,
)


def _make_graph_module_like(gm: fx.GraphModule, graph: fx.Graph) -> fx.GraphModule:
    return torch.fx._lazy_graph_module._make_graph_module(gm, graph)


def extract_graph_with_graph_pp_abi(
    graph: fx.Graph,
    inputs: list[fx.Node],
    outputs: list[Any],
    output_descs: list[Any],
    graph_name: str | None = None,
) -> fx.Graph:
    """Extract an FX graph after GraphPP has selected an explicit flat ABI.

    GraphPP owns the stage-local partitioning policy: it chooses placeholders,
    saved values, side-effect materialization outputs, and runtime tangent
    inputs before extraction.  PyTorch's extractor remains the low-level graph
    copier, but GraphPP intentionally allows selected nodes to sit outside the
    default AOTAutograd forward/backward split.
    """

    args = (graph, inputs, outputs, output_descs)
    if graph_name is None:
        return _extract_graph_with_inputs_outputs(
            *args,
            ignore_must_be_in_fw_bw=True,
        )
    return _extract_graph_with_inputs_outputs(
        *args,
        graph_name,
        ignore_must_be_in_fw_bw=True,
    )


def _graph_outputs(graph: fx.Graph) -> tuple[Any, ...]:
    outputs = graph.find_nodes(op="output")
    if len(outputs) != 1:
        raise ValueError(f"Expected one output node, found {len(outputs)}")
    return tuple(pytree.arg_tree_leaves(outputs[0].args[0]))


def _graph_module_outputs(gm: fx.GraphModule) -> tuple[Any, ...]:
    return _graph_outputs(gm.graph)


def _value_name(value: Any, index: int) -> str:
    if isinstance(value, fx.Node):
        return value.name
    return f"output_{index}"


def _placeholder_names(gm: fx.GraphModule) -> tuple[str, ...]:
    return tuple(node.name for node in gm.graph.find_nodes(op="placeholder"))


def _output_names(gm: fx.GraphModule) -> tuple[str, ...]:
    return tuple(
        _value_name(value, index)
        for index, value in enumerate(_graph_module_outputs(gm))
    )


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
    """Subclass/tree metadata for one semantic slice of a flat GraphPP ABI.

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


@contextmanager
def preserve_module_buffer_state(module: nn.Module) -> Iterator[None]:
    """Restore module buffers after trace-time example forwards.

    Some model forwards update persistent training state in buffers. GraphPP
    graph construction runs example forwards only to infer metadata, so those
    writes must not leak into the first real PP step.
    """

    snapshots = [
        (buffer, buffer.detach().clone())
        for _, buffer in module.named_buffers(remove_duplicate=False)
    ]
    try:
        yield
    finally:
        with torch.no_grad():
            for buffer, snapshot in snapshots:
                buffer.copy_(snapshot)


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


def _graph_pp_block_mask_with_dynamic_offset(block_mask: BlockMask) -> BlockMask:
    """Convert PP-split BlockMask batch offsets from constants to tensor leaves.

    PyTorch's PP microbatch splitter creates each chunked ``BlockMask`` with a
    ``mask_mod`` closure that captures the batch offset as a Python ``int``.
    ``make_fx`` specializes primitive closure values, so tracing microbatch 0
    bakes offset 0 into a graph that GraphPP later reuses for every microbatch.
    Rebuild that closure with the same base mask function but a scalar tensor
    offset, which BlockMask's pytree flattening exposes as a normal graph input.

    TODO(sanketpurandare): requires upstream change: PP BlockMask microbatch
    splitting should represent batch offset as a tensor/pytree leaf instead of
    a Python closure constant. Fixing that split is the intended upstream fix
    for this GraphPP normalization.
    """
    mask_mod = block_mask.mask_mod
    if not inspect.isfunction(mask_mod) or mask_mod.__closure__ is None:
        return block_mask

    closure_values = {
        name: cell.cell_contents
        for name, cell in zip(mask_mod.__code__.co_freevars, mask_mod.__closure__)
    }
    base_block_mask = closure_values.get("block_mask")
    offset = closure_values.get("idx")
    if not isinstance(base_block_mask, BlockMask) or not isinstance(offset, int):
        return block_mask

    base_mask_mod = base_block_mask.mask_mod
    offset_tensor = torch.tensor(
        offset,
        device=block_mask.kv_num_blocks.device,
        dtype=torch.int64,
    )

    def graph_pp_batch_offset_mask_mod(b, h, q_idx, kv_idx):
        dynamic_offset = offset_tensor.to(device=b.device, dtype=b.dtype)
        return base_mask_mod(b + dynamic_offset, h, q_idx, kv_idx)

    return BlockMask(
        seq_lengths=block_mask.seq_lengths,
        kv_num_blocks=block_mask.kv_num_blocks,
        kv_indices=block_mask.kv_indices,
        full_kv_num_blocks=block_mask.full_kv_num_blocks,
        full_kv_indices=block_mask.full_kv_indices,
        q_num_blocks=block_mask.q_num_blocks,
        q_indices=block_mask.q_indices,
        full_q_num_blocks=block_mask.full_q_num_blocks,
        full_q_indices=block_mask.full_q_indices,
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
        mask_mod=graph_pp_batch_offset_mask_mod,
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

    def normalize_leaf(value: Any) -> Any:
        if isinstance(value, BlockMask):
            return _graph_pp_block_mask_with_dynamic_offset(value)
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
