# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Gradient accumulation specific fx passes.
"""

import copy
from typing import Any

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.common_utils import (
    node_tensor_meta as _tensor_meta,
)
from torchtitan.experiments.graph_trainer.debug_utils import tlparse_log_graph_pass
from torchtitan.experiments.graph_trainer.graph_pp.utils import graph_outputs

_GRAD_ACCUMULATOR_INPUT_META = "grad_accumulator_input"


def _new_accumulator(value: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    if value.layout != torch.strided:
        raise NotImplementedError(
            "Gradient accumulator allocation requires strided gradient outputs, "
            f"got {value.layout}"
        )
    assert value.is_contiguous(), "Gradient accumulator must be contiguous"
    shape = tuple(int(dim) for dim in value.shape)
    return torch.empty(
        shape,
        dtype=value.dtype,
        device=device,
    )


def _validate_accumulator(
    accumulator: torch.Tensor,
    value: torch.Tensor,
    *,
    index: int,
    device: torch.device,
) -> None:
    assert value.is_contiguous(), "Gradient output must be contiguous"
    assert accumulator.is_contiguous(), "Gradient accumulator must be contiguous"
    if (
        accumulator.shape != value.shape
        or accumulator.dtype != value.dtype
        or accumulator.device != device
    ):
        raise ValueError(
            "Gradient accumulator does not match backward output at index " f"{index}"
        )


def _insert_accumulator_placeholders(
    gm: fx.GraphModule,
    grad_outputs: list[Any] | tuple[Any, ...],
    accumulator_values: tuple[Any, ...],
    *,
    device: torch.device,
) -> list[fx.Node | None]:
    """Insert graph inputs for distinct gradient accumulators.

    1. Skip non-tensor parameter-gradient outputs.
    2. Validate supplied tensor accumulators against gradient metadata.
    3. Reuse a graph input when parameter-gradient outputs share an accumulator.
    4. Insert each new input before the first computation node.
    5. Assign distinct fake-tensor metadata to avoid aliasing the gradient.

    Example::

        Input::
            (x) -> grad(x)

        Output::
            (x, accumulator) -> grad(x)
    """
    first_compute = next(node for node in gm.graph.nodes if node.op != "placeholder")
    accumulator_nodes: list[fx.Node | None] = []
    node_by_accumulator: dict[tuple[str, int], tuple[fx.Node, fx.Node]] = {}
    with gm.graph.inserting_before(first_compute):
        for index, (grad, accumulator) in enumerate(
            zip(grad_outputs, accumulator_values, strict=True)
        ):
            grad_value = _tensor_meta(grad) if isinstance(grad, fx.Node) else None
            if grad_value is None:
                accumulator_nodes.append(None)
                continue
            if isinstance(accumulator, torch.Tensor):
                _validate_accumulator(
                    accumulator,
                    grad_value,
                    index=index,
                    device=device,
                )
                accumulator_key = ("tensor", id(accumulator))
            elif type(accumulator) is int:
                accumulator_key = ("output", accumulator)
            else:
                raise ValueError(
                    "Tensor gradient output has no matching accumulator "
                    f"at index {index}"
                )
            existing = node_by_accumulator.get(accumulator_key)
            if existing is not None:
                previous_grad, node = existing
                if grad is not previous_grad:
                    raise ValueError(
                        "One gradient accumulator cannot represent "
                        f"different graph outputs at index {index}"
                    )
                accumulator_nodes.append(node)
                continue
            node = gm.graph.placeholder(f"grad_accumulator_{index}")
            node.meta = copy.copy(grad.meta)
            node.meta["val"] = grad_value.new_empty(
                grad_value.shape,
                requires_grad=grad_value.requires_grad,
            )
            node.meta[_GRAD_ACCUMULATOR_INPUT_META] = True
            node_by_accumulator[accumulator_key] = (grad, node)
            accumulator_nodes.append(node)
    return accumulator_nodes


def _gradient_output_accumulator_indices(
    grad_outputs: list[Any] | tuple[Any, ...],
) -> tuple[int | None, ...]:
    """Map each tensor gradient output to its first occurrence.

    1. Assign the first output index to each distinct tensor gradient node.
    2. Reuse that index for repeated gradient nodes.
    3. Return ``None`` for non-tensor gradient outputs.
    """
    accumulator_index_by_grad: dict[fx.Node, int] = {}
    indices: list[int | None] = []
    for index, grad in enumerate(grad_outputs):
        if not isinstance(grad, fx.Node) or _tensor_meta(grad) is None:
            indices.append(None)
            continue
        indices.append(accumulator_index_by_grad.setdefault(grad, index))
    return tuple(indices)


def insert_graph_gradient_accumulation(
    gm: fx.GraphModule,
    *,
    num_param_grads: int,
    device: torch.device,
    param_grad_output_start: int = 0,
    accumulators: tuple[Any, ...] | None = None,
) -> tuple[Any, ...]:
    """Insert accumulation for parameter-gradient outputs.

    1. Select the parameter-gradient outputs.
    2. Allocate accumulators if none were supplied.
    3. Insert gradient accumulator inputs into the graph.
    4. Insert an in-place addition for each distinct tensor gradient.
    5. Replace the parameter-gradient outputs with updated accumulators.
    6. Return accumulators in parameter-gradient output order.

    If ``accumulators`` is omitted, initialize the returned tensors before
    graph execution. Supplied accumulators may be tensors or graph-output indices.

    Example::

        Input::
            (x) -> (loss, grad(x))

        Output::
            (x, accumulator) -> (loss, accumulator.add_(grad(x)))

    The example uses ``param_grad_output_start=1``.
    """
    tlparse_log_graph_pass(gm, graph_name="before_insert_graph_gradient_accumulation")
    outputs = graph_outputs(gm.graph)
    if param_grad_output_start < 0:
        raise ValueError(
            "Parameter gradient output start must be non-negative, got "
            f"{param_grad_output_start}"
        )
    param_grad_output_end = param_grad_output_start + num_param_grads
    if param_grad_output_end > len(outputs):
        raise ValueError(
            "Parameter gradient output range exceeds graph outputs: "
            f"[{param_grad_output_start}, {param_grad_output_end}) with "
            f"{len(outputs)} outputs"
        )

    grad_outputs = outputs[param_grad_output_start:param_grad_output_end]
    if accumulators is None:
        accumulator_values_list: list[Any] = []
        accumulator_by_grad: dict[fx.Node, torch.Tensor] = {}
        for index, output in enumerate(grad_outputs):
            if output is None:
                accumulator_values_list.append(None)
                continue
            if not isinstance(output, fx.Node):
                accumulator_values_list.append(output)
                continue
            value = output.meta.get("val")
            if value is None:
                raise ValueError(
                    "Parameter gradient output has no metadata at "
                    f"index {index}: {output!r}"
                )
            if not isinstance(value, torch.Tensor):
                accumulator_values_list.append(value)
                continue
            accumulator = accumulator_by_grad.get(output)
            if accumulator is None:
                accumulator = _new_accumulator(value, device=device)
                accumulator_by_grad[output] = accumulator
            accumulator_values_list.append(accumulator)
        accumulator_values = tuple(accumulator_values_list)
    else:
        if len(accumulators) != num_param_grads:
            raise ValueError(
                "Gradient accumulator count does not match graph outputs: "
                f"{len(accumulators)} != {num_param_grads}"
            )
        accumulator_values = accumulators

    accumulator_nodes = _insert_accumulator_placeholders(
        gm,
        grad_outputs,
        accumulator_values,
        device=device,
    )

    accumulated_outputs: list[Any] = []
    sink_by_accumulator: dict[fx.Node, fx.Node] = {}
    output_node = gm.graph.find_nodes(op="output")[0]
    for grad, accumulator in zip(grad_outputs, accumulator_nodes, strict=True):
        if not isinstance(accumulator, fx.Node):
            accumulated_outputs.append(grad)
            continue
        assert isinstance(grad, fx.Node)
        if accumulator in sink_by_accumulator:
            accumulated_outputs.append(sink_by_accumulator[accumulator])
            continue
        with gm.graph.inserting_before(output_node):
            sink = gm.graph.call_function(
                torch.ops.aten.add_.Tensor,
                args=(accumulator, grad),
            )
        sink.meta = copy.copy(grad.meta)
        sink_by_accumulator[accumulator] = sink
        accumulated_outputs.append(sink)

    output_node.args = (
        tuple(
            [
                *outputs[:param_grad_output_start],
                *accumulated_outputs,
                *outputs[param_grad_output_end:],
            ]
        ),
    )
    gm.graph.lint()
    gm.recompile()
    tlparse_log_graph_pass(gm, graph_name="after_insert_graph_gradient_accumulation")
    return accumulator_values


def insert_graph_gradient_accumulation_from_outputs(
    gm: fx.GraphModule,
    *,
    num_param_grads: int,
    device: torch.device,
    param_grad_output_start: int = 0,
) -> tuple[int | None, ...]:
    """Use earlier graph outputs as accumulator inputs to this graph.

    1. Map each tensor gradient output to its first output index.
    2. Insert gradient accumulator inputs for those indices.
    3. Return the indices for runtime graph-input packing.

    This path does not allocate separate accumulator buffers.

    Example::

        Input::
            first(x0) -> (loss0, grad0)
            repeat(x1) -> (loss1, grad1)

        Output::
            first(x0) -> (loss0, grad0)
            repeat(x1, grad0) -> (loss1, grad0.add_(grad1))

    The example uses ``param_grad_output_start=1`` and returns ``(0,)``.
    """
    tlparse_log_graph_pass(
        gm, graph_name="before_insert_graph_gradient_accumulation_from_outputs"
    )
    outputs = graph_outputs(gm.graph)
    param_grad_output_end = param_grad_output_start + num_param_grads
    grad_outputs = outputs[param_grad_output_start:param_grad_output_end]
    accumulator_indices = _gradient_output_accumulator_indices(grad_outputs)
    insert_graph_gradient_accumulation(
        gm,
        num_param_grads=num_param_grads,
        param_grad_output_start=param_grad_output_start,
        accumulators=accumulator_indices,
        device=device,
    )
    tlparse_log_graph_pass(
        gm, graph_name="after_insert_graph_gradient_accumulation_from_outputs"
    )
    return accumulator_indices


def insert_graph_gradient_accumulation_before_reduction(
    gm: fx.GraphModule,
    *,
    param_grad_output_names: tuple[str, ...],
    reduce_grad_input_names: tuple[str, ...],
    accumulators: tuple[Any, ...],
    device: torch.device,
) -> tuple[Any, ...]:
    """Accumulate parameter gradients before FSDP reduction.

    1. Match parameter-gradient outputs to accumulators by node name.
    2. Select the graph nodes named by ``reduce_grad_input_names``.
    3. Insert one graph input for each distinct gradient accumulator.
    4. Insert an in-place addition after each selected gradient node.
    5. Redirect each gradient node's users to the in-place addition.
    6. Return distinct accumulators in graph-input order.

    Example::

        Input::
            (x) -> (loss, reduce_grad(raw_grad(x)))

        Output::
            (x, accumulator) ->
                (loss, reduce_grad(accumulator.add_(raw_grad(x))))
    """
    tlparse_log_graph_pass(
        gm,
        graph_name="before_insert_graph_gradient_accumulation_before_reduction",
    )
    if len(param_grad_output_names) != len(accumulators):
        raise ValueError(
            "Gradient accumulator count does not match parameter gradients: "
            f"{len(accumulators)} != {len(param_grad_output_names)}"
        )
    accumulator_by_name: dict[str, Any] = {}
    for name, accumulator in zip(
        param_grad_output_names,
        accumulators,
        strict=True,
    ):
        previous = accumulator_by_name.setdefault(name, accumulator)
        if previous is not accumulator:
            raise ValueError(f"Parameter gradient {name} maps to multiple accumulators")

    nodes_by_name = {node.name: node for node in gm.graph.nodes}
    try:
        grad_inputs = [nodes_by_name[name] for name in reduce_grad_input_names]
        accumulator_values = tuple(
            accumulator_by_name[name] for name in reduce_grad_input_names
        )
    except KeyError as error:
        raise ValueError(
            f"Gradient reduction input has no accumulator: {error.args[0]}"
        ) from error

    accumulator_nodes = _insert_accumulator_placeholders(
        gm,
        grad_inputs,
        accumulator_values,
        device=device,
    )
    graph_input_accumulators: list[Any] = []
    seen_accumulators: set[tuple[str, int]] = set()
    for grad, accumulator, accumulator_value in zip(
        grad_inputs,
        accumulator_nodes,
        accumulator_values,
        strict=True,
    ):
        if accumulator is None:
            continue
        accumulator_key = (
            ("tensor", id(accumulator_value))
            if isinstance(accumulator_value, torch.Tensor)
            else ("output", accumulator_value)
        )
        if accumulator_key not in seen_accumulators:
            seen_accumulators.add(accumulator_key)
            graph_input_accumulators.append(accumulator_value)
        old_users = list(grad.users)
        with gm.graph.inserting_after(grad):
            sink = gm.graph.call_function(
                torch.ops.aten.add_.Tensor,
                args=(accumulator, grad),
            )
        sink.meta = copy.copy(grad.meta)
        for user in old_users:
            user.replace_input_with(grad, sink)

    gm.graph.lint()
    gm.recompile()
    tlparse_log_graph_pass(
        gm,
        graph_name="after_insert_graph_gradient_accumulation_before_reduction",
    )
    return tuple(graph_input_accumulators)


__all__ = [
    "insert_graph_gradient_accumulation",
    "insert_graph_gradient_accumulation_before_reduction",
    "insert_graph_gradient_accumulation_from_outputs",
]
