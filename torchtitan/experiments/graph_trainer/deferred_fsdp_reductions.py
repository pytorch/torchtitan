# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import functools
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import TensorMeta
from torch.fx._lazy_graph_module import _make_graph_module

from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

from torchtitan.experiments.graph_trainer.cudagraph import (
    cudagraph_pass,
    insert_kernel_annotations_pass,
    is_cudagraph_compatible,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    joint_transformer_block_bucketing_reordering_pass,
)
from torchtitan.experiments.graph_trainer.fsdp_patterns import (
    find_fsdp_reduce_grad_input,
    find_fsdp_unshard_outputs_by_param,
    is_all_gather_into_tensor,
    is_all_reduce,
    is_reduce_scatter_tensor,
)
from torchtitan.experiments.graph_trainer.graph_pp.split_fsdp_collectives import (
    split_forward_fsdp_collectives,
)
from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    allow_fx_graph_extraction_of_side_effectful_ops,
    example_inputs_from_placeholders,
    graph_outputs,
    placeholder_names,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    _flat_tensor_ranges,
    SubclassLayout,
    TracedResult,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    final_inductor_compile_passes,
)


_DEFERRED_GRADIENT_OUTPUT_META = "deferred_fsdp_gradient_outputs"


@dataclass(frozen=True, slots=True)
class _BoundarySpec:
    output_flat_index: int
    shape: torch.Size
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True, slots=True)
class GraphWithDeferredFSDPReductions:
    """A whole-optimizer-step minimal-FX graph with one FSDP gradient sync."""

    gm: fx.GraphModule
    traced_result: TracedResult
    num_microbatches: int
    num_all_gathers: int
    num_reduce_scatters: int
    num_all_reduces: int

    @property
    def num_gradient_collectives(self) -> int:
        return self.num_reduce_scatters + self.num_all_reduces


def _gradient_output_indices(traced_result: TracedResult) -> tuple[int, ...]:
    indices = tuple(range(1, len(traced_result.graph_state.mappings) + 1))
    if traced_result.num_flat_outputs != len(indices) + 1:
        raise ValueError(
            "Deferred FSDP gradient sync requires one loss followed by one "
            "gradient per graph-state buffer"
        )
    return indices


def tag_deferred_fsdp_gradient_outputs(
    traced_result: TracedResult,
    gm: fx.GraphModule | None = None,
) -> None:
    """Tag each flattened gradient output with its destination state buffer."""
    outputs = graph_outputs((traced_result.gm if gm is None else gm).graph)
    output_ranges = _flat_tensor_ranges(
        traced_result.num_flat_outputs,
        traced_result.output_subclass_layouts,
    )
    for mapping, output_index in zip(
        traced_result.graph_state.mappings,
        _gradient_output_indices(traced_result),
        strict=True,
    ):
        for leaf_offset, flat_index in enumerate(output_ranges[output_index]):
            leaf = outputs[flat_index]
            if not isinstance(leaf, fx.Node):
                raise ValueError(
                    f"Gradient output {output_index} for {mapping.fqn!r} "
                    "is not a tensor"
                )
            tag = (mapping.fqn, leaf_offset)
            existing_tags = leaf.meta.get(_DEFERRED_GRADIENT_OUTPUT_META, ())
            if tag not in existing_tags:
                leaf.meta[_DEFERRED_GRADIENT_OUTPUT_META] = (*existing_tags, tag)


def _validate_gradient_output_mapping(
    gm: fx.GraphModule,
    traced_result: TracedResult,
) -> None:
    outputs = graph_outputs(gm.graph)
    output_ranges = _flat_tensor_ranges(
        traced_result.num_flat_outputs,
        traced_result.output_subclass_layouts,
    )
    expected_num_outputs = sum(len(indices) for indices in output_ranges)
    if len(outputs) != expected_num_outputs:
        raise ValueError(
            "Deferred FSDP graph output count changed: "
            f"expected {expected_num_outputs}, got {len(outputs)}"
        )
    for mapping, output_index in zip(
        traced_result.graph_state.mappings,
        _gradient_output_indices(traced_result),
        strict=True,
    ):
        for leaf_offset, flat_index in enumerate(output_ranges[output_index]):
            output = outputs[flat_index]
            if not isinstance(output, fx.Node) or (
                mapping.fqn,
                leaf_offset,
            ) not in output.meta.get(_DEFERRED_GRADIENT_OUTPUT_META, ()):
                raise ValueError(
                    "Graph pass changed the explicit gradient-output mapping "
                    f"for {mapping.fqn!r}"
                )


def _subclass_context_without_strides(value: Any) -> Any:
    return pytree.tree_map(
        lambda item: item._replace(stride=()) if isinstance(item, TensorMeta) else item,
        value,
        is_leaf=lambda item: isinstance(item, TensorMeta),
    )


def _graph_state_leaf_offsets(
    fqn: str,
    buffer_layout: SubclassLayout | None,
    gradient_layout: SubclassLayout | None,
) -> tuple[tuple[int, ...], int | None]:
    if (buffer_layout is None) != (gradient_layout is None):
        raise ValueError(f"Gradient tensor subclass does not match buffer for {fqn!r}")
    if buffer_layout is None or gradient_layout is None:
        return (0,), None

    buffer_meta = buffer_layout.meta
    gradient_meta = gradient_layout.meta
    if buffer_meta is None or gradient_meta is None:
        raise ValueError(f"Missing tensor-subclass metadata for {fqn!r}")
    if (
        buffer_meta.cls is not gradient_meta.cls
        or buffer_meta.attrs != gradient_meta.attrs
        or _subclass_context_without_strides(buffer_meta.ctx)
        != _subclass_context_without_strides(gradient_meta.ctx)
        or buffer_meta.outer_size != gradient_meta.outer_size
    ):
        raise ValueError(
            "Gradient tensor subclass metadata does not match buffer for "
            f"{fqn!r}: buffer={buffer_meta!r}, gradient={gradient_meta!r}"
        )
    if buffer_layout.num_tensors != gradient_layout.num_tensors:
        raise ValueError(
            f"Gradient tensor subclass leaves do not match buffer for {fqn!r}"
        )
    if not issubclass(buffer_meta.cls, DTensor):
        raise NotImplementedError(
            "Deferred FSDP gradient sync only supports plain tensors and "
            f"DTensor subclasses, got {buffer_meta.cls.__name__} for {fqn!r}"
        )

    flat_offset = 0
    local_tensor_offset = None
    device_mesh_offset = None
    for attr in buffer_meta.attrs:
        num_tensors, inner_meta = buffer_meta.inner_metas[attr]
        gradient_num_tensors, gradient_inner_meta = gradient_meta.inner_metas[attr]
        if num_tensors != gradient_num_tensors:
            raise ValueError(
                f"Gradient tensor subclass leaves do not match buffer for {fqn!r}"
            )
        if attr == "_local_tensor":
            if (
                num_tensors != 1
                or inner_meta is not None
                or gradient_inner_meta is not None
            ):
                raise NotImplementedError(
                    "Deferred FSDP gradient sync requires plain DTensor local "
                    f"tensors for {fqn!r}"
                )
            local_tensor_offset = flat_offset
        elif attr == "device_mesh":
            if (
                num_tensors != 1
                or inner_meta is not None
                or gradient_inner_meta is not None
            ):
                raise NotImplementedError(
                    "Deferred FSDP gradient sync requires a plain DTensor device "
                    f"mesh for {fqn!r}"
                )
            device_mesh_offset = flat_offset
        else:
            raise NotImplementedError(
                "Deferred FSDP gradient sync does not support DTensor wrapper "
                f"attribute {attr!r} for {fqn!r}"
            )
        flat_offset += num_tensors
    if local_tensor_offset is None:
        raise ValueError(f"DTensor gradient state {fqn!r} has no local tensor")
    if device_mesh_offset is None:
        raise ValueError(f"DTensor gradient state {fqn!r} has no device mesh")
    return (local_tensor_offset,), device_mesh_offset


def _validate_device_mesh_leaf(
    fqn: str,
    buffer: fx.Node,
    gradient: fx.Node,
) -> None:
    buffer_mesh = buffer.meta.get("val")
    gradient_mesh = gradient.meta.get("val")
    if (
        not isinstance(buffer_mesh, DeviceMesh)
        or not isinstance(gradient_mesh, DeviceMesh)
        or buffer_mesh != gradient_mesh
    ):
        raise ValueError(f"DTensor device mesh does not match buffer for {fqn!r}")


def _validate_tensor_leaf(
    fqn: str,
    buffer: fx.Node,
    gradient: fx.Node,
) -> None:
    buffer_value = buffer.meta.get("val")
    gradient_value = gradient.meta.get("val")
    if not isinstance(buffer_value, torch.Tensor) or not isinstance(
        gradient_value, torch.Tensor
    ):
        raise ValueError(f"Missing tensor metadata for gradient state {fqn!r}")
    if (
        buffer_value.shape != gradient_value.shape
        or buffer_value.dtype != gradient_value.dtype
        or buffer_value.device != gradient_value.device
    ):
        raise ValueError(
            "Gradient shape, dtype, and device must match its buffer for "
            f"{fqn!r}; got gradient "
            f"{tuple(gradient_value.shape)}, {gradient_value.dtype}, "
            f"{gradient_value.device} and buffer "
            f"{tuple(buffer_value.shape)}, {buffer_value.dtype}, "
            f"{buffer_value.device}"
        )
    if gradient_value.layout != torch.strided:
        raise NotImplementedError(
            "Deferred FSDP gradient sync does not support "
            f"{gradient_value.layout} gradient {fqn!r}"
        )


def _insert_graph_gradient_sink(
    gm: fx.GraphModule,
    *,
    fqn: str,
    buffer: fx.Node,
    gradient: fx.Node,
) -> fx.Node:
    _validate_tensor_leaf(fqn, buffer, gradient)
    sink = gm.graph.call_function(
        torch.ops.aten.add_.Tensor,
        args=(buffer, gradient),
    )
    _copy_placeholder_meta(gradient, sink)
    if "val" in buffer.meta:
        sink.meta["val"] = buffer.meta["val"]
    sink.meta["graph_gradient_fqn"] = fqn
    return sink


def _validate_gradient_leaf_mapping(
    fqn: str,
    leaf_offset: int,
    gradient: fx.Node,
) -> None:
    if (fqn, leaf_offset) not in gradient.meta.get(_DEFERRED_GRADIENT_OUTPUT_META, ()):
        raise ValueError(
            "Graph pass changed the explicit gradient-output mapping for " f"{fqn!r}"
        )


def _gradient_and_loss_flat_indices(
    traced_result: TracedResult,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    ranges = _flat_tensor_ranges(
        traced_result.num_flat_outputs,
        traced_result.output_subclass_layouts,
    )
    gradient_indices = tuple(
        index
        for output_index in _gradient_output_indices(traced_result)
        for index in ranges[output_index]
    )
    gradient_index_set = set(gradient_indices)
    loss_indices = tuple(
        index
        for index in range(sum(len(indices) for indices in ranges))
        if index not in gradient_index_set
    )
    if len(loss_indices) != 1:
        raise ValueError(
            "Deferred FSDP gradient sync requires one flat loss output, got "
            f"{len(loss_indices)}"
        )
    return gradient_indices, loss_indices


def _boundary_signature(node: fx.Node, output_flat_index: int) -> _BoundarySpec:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor) or not (
        value.is_floating_point() or value.is_complex()
    ):
        raise ValueError(
            "Deferred FSDP gradient boundaries must be floating-point tensors; "
            f"output {output_flat_index} has {value!r}"
        )
    if value.layout != torch.strided:
        raise NotImplementedError(
            "Deferred FSDP gradient sync requires strided pre-reduction "
            f"gradients, got {value.layout}"
        )
    return _BoundarySpec(
        output_flat_index=output_flat_index,
        shape=value.shape,
        stride=value.stride(),
        dtype=value.dtype,
        device=value.device,
    )


def _gradient_boundaries(
    gm: fx.GraphModule,
    traced_result: TracedResult,
) -> tuple[list[fx.Node], tuple[_BoundarySpec, ...], int]:
    outputs = graph_outputs(gm.graph)
    gradient_indices, _ = _gradient_and_loss_flat_indices(traced_result)
    boundaries: list[fx.Node] = []
    specs: list[_BoundarySpec] = []
    seen: set[fx.Node] = set()
    collective_boundaries: set[fx.Node] = set()
    for output_index in gradient_indices:
        output = outputs[output_index]
        if not isinstance(output, fx.Node):
            continue
        boundary = find_fsdp_reduce_grad_input(
            output,
            allow_bucket_fanout=True,
        )
        if boundary is not None:
            collective_boundaries.add(boundary)
        else:
            boundary = output
        value = boundary.meta.get("val")
        if not isinstance(value, torch.Tensor) or not (
            value.is_floating_point() or value.is_complex()
        ):
            continue
        if boundary in seen:
            continue
        seen.add(boundary)
        boundaries.append(boundary)
        specs.append(_boundary_signature(boundary, output_index))
    if not boundaries:
        raise ValueError("Deferred FSDP gradient sync found no gradient tensors")
    return boundaries, tuple(specs), len(collective_boundaries)


def _assert_matching_boundaries(
    expected: tuple[_BoundarySpec, ...],
    actual: tuple[_BoundarySpec, ...],
) -> None:
    expected_metadata = tuple(
        (
            spec.output_flat_index,
            spec.shape,
            spec.stride,
            spec.dtype,
            spec.device,
        )
        for spec in expected
    )
    actual_metadata = tuple(
        (
            spec.output_flat_index,
            spec.shape,
            spec.stride,
            spec.dtype,
            spec.device,
        )
        for spec in actual
    )
    if expected_metadata != actual_metadata:
        raise ValueError(
            "Deferred FSDP gradient boundaries changed after unshard "
            f"extraction: {expected_metadata!r} != {actual_metadata!r}"
        )


def _extract_graph(
    gm: fx.GraphModule,
    selected_outputs: list[fx.Node],
    *,
    name: str,
) -> fx.GraphModule:
    placeholders = gm.graph.find_nodes(op="placeholder")
    with allow_fx_graph_extraction_of_side_effectful_ops(
        {
            torch.ops._c10d_functional.wait_tensor,
            torch.ops._c10d_functional.wait_tensor.default,
            torch.ops._c10d_functional.reduce_scatter_tensor.default,
            torch.ops._c10d_functional.all_reduce.default,
        }
    ):
        graph = _extract_graph_with_inputs_outputs(
            gm.graph,
            placeholders,
            selected_outputs,
            [None] * len(selected_outputs),  # pyrefly: ignore [bad-argument-type]
            name,
            ignore_must_be_in_fw_bw=True,
        )
    return _make_graph_module(gm, graph)


def _unsharded_parameter_outputs(
    gm: fx.GraphModule,
    *,
    num_flat_parameters: int,
) -> list[fx.Node]:
    placeholders = gm.graph.find_nodes(op="placeholder")
    parameter_placeholders = placeholders[:num_flat_parameters]
    outputs_by_param = find_fsdp_unshard_outputs_by_param(parameter_placeholders)
    outputs: list[fx.Node] = []
    found_unshard = False
    for placeholder in parameter_placeholders:
        unsharded = outputs_by_param[placeholder]
        if not unsharded:
            outputs.append(placeholder)
            continue
        if len(unsharded) != 1:
            raise ValueError(
                "Deferred FSDP gradient sync expects one canonical unshard "
                f"output for {placeholder.name}, got {len(unsharded)}"
            )
        found_unshard = True
        outputs.append(unsharded[0])
    if not found_unshard:
        raise ValueError(
            "Deferred FSDP gradient sync requires a traced FSDP all-gather"
        )
    return outputs


def _loss_outputs(
    gm: fx.GraphModule,
    traced_result: TracedResult,
) -> list[fx.Node]:
    outputs = graph_outputs(gm.graph)
    _, loss_indices = _gradient_and_loss_flat_indices(traced_result)
    losses = [outputs[index] for index in loss_indices]
    if any(not isinstance(loss, fx.Node) for loss in losses):
        raise ValueError("Deferred FSDP gradient sync requires a tensor loss")
    return list(losses)


def _build_first_microbatch_graph(
    gm: fx.GraphModule,
    traced_result: TracedResult,
    *,
    num_flat_parameters: int,
    reuse_unsharded_parameters: bool,
) -> tuple[fx.GraphModule, tuple[_BoundarySpec, ...], int]:
    """Extract the first microbatch graph before gradient synchronization.

    For one gradient boundary, the generated graph is equivalent to::

        loss0, grad0 = first_microbatch(original_inputs)
        return loss0, grad0, *unsharded_params

    ``grad0`` is the value immediately before FSDP gradient synchronization, so
    this graph does not perform the deferred reduction. The outer graph retains
    it as the initial accumulator. When parameter reuse is enabled, it also
    retains the unsharded parameters for the remaining microbatches.
    """
    work = copy.deepcopy(gm)
    boundaries, specs, num_collective_boundaries = _gradient_boundaries(
        work, traced_result
    )
    unsharded = (
        _unsharded_parameter_outputs(
            work,
            num_flat_parameters=num_flat_parameters,
        )
        if reuse_unsharded_parameters
        else []
    )
    first = _extract_graph(
        work,
        [*_loss_outputs(work, traced_result), *boundaries, *unsharded],
        name="deferred_fsdp_first",
    )
    return first, specs, num_collective_boundaries


def _append_accumulator_placeholders(
    gm: fx.GraphModule,
    boundaries: list[fx.Node],
) -> list[fx.Node]:
    first_non_placeholder = next(
        node for node in gm.graph.nodes if node.op != "placeholder"
    )
    accumulators = []
    with gm.graph.inserting_before(first_non_placeholder):
        for index, boundary in enumerate(boundaries):
            accumulator = gm.graph.placeholder(f"deferred_grad_{index}")
            accumulator.meta = copy.copy(boundary.meta)
            accumulators.append(accumulator)
    return accumulators


def _build_middle_microbatch_graph(
    gm: fx.GraphModule,
    traced_result: TracedResult,
    expected_specs: tuple[_BoundarySpec, ...],
) -> tuple[fx.GraphModule, int]:
    """Build a non-final microbatch that updates pre-reduction accumulators.

    For one gradient boundary, the generated graph is equivalent to::

        loss1, grad1 = middle_microbatch(compute_inputs)
        accumulator.add_(grad1)
        return loss1

    The graph accepts one accumulator per boundary and updates each one in
    place. It does not perform the deferred reduction or return the accumulators;
    their mutations are visible to the next middle or final child.
    """
    work = copy.deepcopy(gm)
    boundaries, specs, num_collective_boundaries = _gradient_boundaries(
        work, traced_result
    )
    _assert_matching_boundaries(expected_specs, specs)
    middle = _extract_graph(
        work,
        [*_loss_outputs(work, traced_result), *boundaries],
        name="deferred_fsdp_middle",
    )
    outputs = graph_outputs(middle.graph)
    loss_count = 1
    middle_boundaries = list(outputs[loss_count:])
    if any(not isinstance(boundary, fx.Node) for boundary in middle_boundaries):
        raise ValueError("Deferred FSDP middle graph lost a gradient boundary")
    typed_boundaries = [
        boundary for boundary in middle_boundaries if isinstance(boundary, fx.Node)
    ]
    accumulators = _append_accumulator_placeholders(middle, typed_boundaries)
    for boundary, accumulator in zip(typed_boundaries, accumulators, strict=True):
        with middle.graph.inserting_after(boundary):
            sink = middle.graph.call_function(
                torch.ops.aten.add_.Tensor,
                args=(accumulator, boundary),
            )
        _copy_placeholder_meta(accumulator, sink)
        sink.meta["deferred_fsdp_gradient_accumulation"] = True
    output = middle.graph.find_nodes(op="output")[0]
    output.args = (tuple(outputs[:loss_count]),)
    middle.graph.lint()
    middle.recompile()
    return middle, num_collective_boundaries


def _compute_placeholder_by_original_index(
    gm: fx.GraphModule,
    compute_flat_input_indices: tuple[int, ...],
) -> dict[int, fx.Node]:
    placeholders = gm.graph.find_nodes(op="placeholder")
    if len(placeholders) < len(compute_flat_input_indices):
        raise ValueError(
            "Deferred FSDP compute input metadata has fewer placeholders than "
            f"expected: {len(placeholders)} < {len(compute_flat_input_indices)}"
        )
    return dict(
        zip(
            compute_flat_input_indices,
            placeholders[: len(compute_flat_input_indices)],
            strict=True,
        )
    )


def _insert_final_gradient_sinks(
    gm: fx.GraphModule,
    traced_result: TracedResult,
    *,
    compute_flat_input_indices: tuple[int, ...],
) -> None:
    outputs = graph_outputs(gm.graph)
    output_ranges = _flat_tensor_ranges(
        traced_result.num_flat_outputs,
        traced_result.output_subclass_layouts,
    )
    placeholder_by_input = _compute_placeholder_by_original_index(
        gm, compute_flat_input_indices
    )
    for state_index, (mapping, gradient_output_index) in enumerate(
        zip(
            traced_result.graph_state.mappings,
            _gradient_output_indices(traced_result),
            strict=True,
        )
    ):
        fqn = mapping.fqn
        buffer_indices = mapping.input_indices
        buffer_logical_index = len(traced_result.state_fqns) + state_index
        leaf_offsets, device_mesh_offset = _graph_state_leaf_offsets(
            fqn,
            traced_result.input_subclass_layouts.get(buffer_logical_index),
            traced_result.output_subclass_layouts.get(gradient_output_index),
        )
        gradient_indices = output_ranges[gradient_output_index]
        if device_mesh_offset is not None:
            _validate_device_mesh_leaf(
                fqn,
                placeholder_by_input[buffer_indices[device_mesh_offset]],
                outputs[gradient_indices[device_mesh_offset]],
            )
        for leaf_offset in leaf_offsets:
            buffer = placeholder_by_input[buffer_indices[leaf_offset]]
            gradient = outputs[gradient_indices[leaf_offset]]
            if not isinstance(gradient, fx.Node):
                raise ValueError(f"Gradient output for {fqn!r} is not a tensor")
            _validate_gradient_leaf_mapping(fqn, leaf_offset, gradient)
            with gm.graph.inserting_after(gradient):
                _insert_graph_gradient_sink(
                    gm,
                    fqn=fqn,
                    buffer=buffer,
                    gradient=gradient,
                )


def _build_final_microbatch_graph(
    gm: fx.GraphModule,
    traced_result: TracedResult,
    expected_specs: tuple[_BoundarySpec, ...],
) -> tuple[fx.GraphModule, int]:
    """Build the final microbatch with synchronization after accumulation.

    For one gradient boundary, the generated graph is equivalent to::

        lossN, gradN = final_microbatch(compute_inputs)
        combined = accumulator.add_(gradN)
        reduced_grad = fsdp_reduce(combined)
        return lossN, reduced_grad

    The original FSDP reduction is retained but rewired to consume the complete
    accumulated gradient. Before finalization, the graph returns the loss and
    reduced gradients.
    """
    final = copy.deepcopy(gm)
    boundaries, specs, num_collective_boundaries = _gradient_boundaries(
        final, traced_result
    )
    _assert_matching_boundaries(expected_specs, specs)
    accumulators = _append_accumulator_placeholders(final, boundaries)
    for boundary, accumulator in zip(boundaries, accumulators, strict=True):
        users = tuple(boundary.users)
        with final.graph.inserting_after(boundary):
            combined = final.graph.call_function(
                torch.ops.aten.add_.Tensor,
                args=(accumulator, boundary),
            )
        _copy_placeholder_meta(accumulator, combined)
        combined.meta["deferred_fsdp_final_gradient"] = True
        for user in users:
            user.replace_input_with(boundary, combined)

    final.graph.lint()
    final.recompile()
    return final, num_collective_boundaries


def _finalize_final_microbatch_graph(
    gm: fx.GraphModule,
    traced_result: TracedResult,
    *,
    compute_flat_input_indices: tuple[int, ...],
) -> None:
    """Write final reduced gradients to buffers and leave only the loss output.

    For each final gradient, this changes the graph from::

        return lossN, reduced_grad

    to::

        graph_state_buffer.add_(reduced_grad)
        return lossN

    The transformation mutates ``gm`` in place. The graph-state buffers are the
    persistent gradients exposed to the optimizer.
    """
    tag_deferred_fsdp_gradient_outputs(traced_result, gm)
    _insert_final_gradient_sinks(
        gm,
        traced_result,
        compute_flat_input_indices=compute_flat_input_indices,
    )
    output = gm.graph.find_nodes(op="output")[0]
    output.args = (tuple(_loss_outputs(gm, traced_result)),)
    gm.graph.lint()
    gm.recompile()


def _apply_child_graph_passes(
    gm: fx.GraphModule,
    *,
    graph_passes: Sequence[Callable],
    compile_config: GraphTrainerCompileConfig,
) -> fx.GraphModule:
    if graph_passes:
        gm = apply_graph_passes(
            gm,
            example_inputs_from_placeholders(gm),
            list(graph_passes),
            compile_config=compile_config,
        )
    return gm


def _compile_child(
    gm: fx.GraphModule,
    *,
    compile_config: GraphTrainerCompileConfig,
) -> fx.GraphModule:
    if compile_config.enable and compile_config.enable_passes:
        gm = apply_graph_passes(
            gm,
            example_inputs_from_placeholders(gm),
            final_inductor_compile_passes(compile_config),
            compile_config=compile_config,
        )
    return gm


def _split_gradient_bucketing_passes(
    passes: Sequence[Callable],
) -> tuple[list[Callable], list[Callable]]:
    """Run gradient bucketing after logical-gradient accumulation.

    For example::

        [cleanup, scheduling, bucketing, compilation]
        -> [cleanup, scheduling], [bucketing, compilation]

    The first list transforms the original microbatch graph. The second list
    transforms the generated first, middle, and final graphs, so bucketing sees
    the accumulated gradients while the original pass order is preserved. If
    there is no bucketing pass, the second list is empty.
    """
    bucketing_indices = [
        index
        for index, pass_fn in enumerate(passes)
        if (pass_fn.func if isinstance(pass_fn, functools.partial) else pass_fn)
        is joint_transformer_block_bucketing_reordering_pass
    ]
    if len(bucketing_indices) > 1:
        raise ValueError("Deferred FSDP graph has multiple gradient bucketing passes")
    if not bucketing_indices:
        return list(passes), []
    bucketing_index = bucketing_indices[0]
    return list(passes[:bucketing_index]), list(passes[bucketing_index:])


def _fsdp_collective_counts(gm: fx.GraphModule) -> tuple[int, int, int]:
    nodes = tuple(gm.graph.nodes)
    return (
        sum(is_all_gather_into_tensor(node) for node in nodes),
        sum(is_reduce_scatter_tensor(node) for node in nodes),
        sum(is_all_reduce(node) for node in nodes),
    )


def _copy_placeholder_meta(source: fx.Node, target: fx.Node) -> None:
    target.meta = copy.copy(source.meta)
    for key in ("custom", "unbacked_bindings"):
        if isinstance(value := target.meta.get(key), dict):
            target.meta[key] = copy.copy(value)


def _call_module_outputs(
    graph: fx.Graph,
    target: str,
    args: list[fx.Node],
    output_values: tuple[Any, ...],
) -> list[fx.Node]:
    call = graph.call_module(target, args=tuple(args))
    call.meta["val"] = tuple(
        value.meta.get("val") if isinstance(value, fx.Node) else value
        for value in output_values
    )
    outputs = []
    insertion_point = call
    for index, value in enumerate(output_values):
        with graph.inserting_after(insertion_point):
            item = graph.call_function(operator.getitem, args=(call, index))
        if isinstance(value, fx.Node):
            _copy_placeholder_meta(value, item)
        outputs.append(item)
        insertion_point = item
    return outputs


def _outer_placeholders(
    graph: fx.Graph,
    traced_result: TracedResult,
    *,
    num_microbatches: int,
) -> tuple[list[list[fx.Node]], tuple[Any, ...]]:
    original_placeholders = traced_result.gm.graph.find_nodes(op="placeholder")
    static_count = traced_result.num_static_inputs
    static_nodes = []
    example_inputs: list[Any] = []
    for index, source in enumerate(original_placeholders[:static_count]):
        node = graph.placeholder(f"static_{index}_{source.name}")
        _copy_placeholder_meta(source, node)
        static_nodes.append(node)
        example_inputs.append(traced_result.example_inputs[index])

    calls = []
    for microbatch in range(num_microbatches):
        call_nodes = list(static_nodes)
        for index, source in enumerate(original_placeholders[static_count:]):
            node = graph.placeholder(f"mb{microbatch}_{index}_{source.name}")
            _copy_placeholder_meta(source, node)
            call_nodes.append(node)
            example_inputs.append(traced_result.example_inputs[static_count + index])
        calls.append(call_nodes)
    return calls, tuple(example_inputs)


def _reuse_call_args(
    original_args: list[fx.Node],
    unsharded_params: list[fx.Node],
    *,
    num_reused_parameters: int,
    compute_flat_input_indices: tuple[int, ...],
) -> list[fx.Node]:
    result = []
    for original_index in compute_flat_input_indices:
        if original_index < num_reused_parameters:
            result.append(unsharded_params[original_index])
        else:
            result.append(original_args[original_index])
    return result


def _build_outer_graph(
    traced_result: TracedResult,
    *,
    first: fx.GraphModule,
    middle: fx.GraphModule,
    final: fx.GraphModule,
    num_microbatches: int,
    num_boundaries: int,
    num_reused_parameters: int,
    compute_flat_input_indices: tuple[int, ...],
) -> tuple[fx.GraphModule, tuple[Any, ...]]:
    """Compose the first, middle, and final children into one step graph.

    For four microbatches, the generated graph is equivalent to::

        loss0, *accumulators, *unsharded_params = first(mb0)
        loss1 = middle(mb1, *unsharded_params, *accumulators)
        loss2 = middle(mb2, *unsharded_params, *accumulators)
        loss3 = final(mb3, *unsharded_params, *accumulators)
        return loss0 + loss1 + loss2 + loss3

    The middle children update the accumulators in place. The final child adds
    its gradients, performs deferred synchronization, and writes the results to
    the graph-state buffers. With two microbatches, there are no middle calls.

    The generated graph accepts the static state once, followed by the inputs
    for every microbatch, and returns one summed loss. This function returns the
    graph together with its flattened example inputs.
    """
    graph = fx.Graph()
    call_args, example_inputs = _outer_placeholders(
        graph,
        traced_result,
        num_microbatches=num_microbatches,
    )
    first_values = graph_outputs(first.graph)
    first_outputs = _call_module_outputs(
        graph,
        "first",
        call_args[0],
        first_values,
    )
    loss = first_outputs[0]
    accumulators = first_outputs[1 : 1 + num_boundaries]
    unsharded_params = first_outputs[1 + num_boundaries :]
    if len(unsharded_params) != num_reused_parameters:
        raise ValueError(
            "Deferred FSDP first graph returned the wrong number of unsharded "
            f"parameters: {len(unsharded_params)} != {num_reused_parameters}"
        )

    for microbatch in range(1, num_microbatches - 1):
        middle_args = _reuse_call_args(
            call_args[microbatch],
            unsharded_params,
            num_reused_parameters=num_reused_parameters,
            compute_flat_input_indices=compute_flat_input_indices,
        )
        middle_outputs = _call_module_outputs(
            graph,
            "middle",
            [*middle_args, *accumulators],
            graph_outputs(middle.graph),
        )
        with graph.inserting_after(middle_outputs[0]):
            next_loss = graph.call_function(
                torch.ops.aten.add.Tensor,
                args=(loss, middle_outputs[0]),
            )
        _copy_placeholder_meta(loss, next_loss)
        loss = next_loss

    final_args = _reuse_call_args(
        call_args[-1],
        unsharded_params,
        num_reused_parameters=num_reused_parameters,
        compute_flat_input_indices=compute_flat_input_indices,
    )
    final_outputs = _call_module_outputs(
        graph,
        "final",
        [*final_args, *accumulators],
        graph_outputs(final.graph),
    )
    with graph.inserting_after(final_outputs[0]):
        loss = graph.call_function(
            torch.ops.aten.add.Tensor,
            args=(loss, final_outputs[0]),
        )
    _copy_placeholder_meta(final_outputs[0], loss)
    graph.output((loss,))

    root = nn.Module()
    root.add_module("first", first)
    root.add_module("middle", middle)
    root.add_module("final", final)
    outer = fx.GraphModule(root, graph, "GraphWithDeferredFSDPReductions")
    outer.graph.lint()
    outer.recompile()
    return outer, example_inputs


def build_graph_with_deferred_fsdp_reductions(
    traced_result: TracedResult,
    *,
    num_flat_parameters: int,
    num_microbatches: int,
    compile_config: GraphTrainerCompileConfig,
    enable_cudagraph: bool,
    reuse_unsharded_parameters: bool = True,
    graph_passes: Sequence[Callable] = (),
) -> GraphWithDeferredFSDPReductions:
    """Build one minimal-FX program for an SPMD accumulation step."""
    if num_microbatches < 2:
        raise ValueError("Deferred FSDP gradient sync requires at least two batches")
    (
        pre_accumulation_passes,
        post_accumulation_passes,
    ) = _split_gradient_bucketing_passes(graph_passes)
    if pre_accumulation_passes:
        traced_result.gm = apply_graph_passes(
            traced_result.gm,
            traced_result.example_inputs,
            pre_accumulation_passes,
            compile_config=compile_config,
            respect_disable_passes=compile_config.enable_passes,
        )
    tag_deferred_fsdp_gradient_outputs(traced_result)
    _validate_gradient_output_mapping(traced_result.gm, traced_result)
    input_names = placeholder_names(traced_result.gm)
    if reuse_unsharded_parameters:
        split = split_forward_fsdp_collectives(
            traced_result.gm,
            num_params=num_flat_parameters,
            fwd_input_names=input_names,
            fwd_flat_input_indices=tuple(range(len(input_names))),
        )
        if split.unshard_module is None:
            raise ValueError("Deferred FSDP gradient sync found no FSDP all-gather")
        compute_gm = split.fw_no_fsdp_module
        compute_flat_input_indices = (
            *split.unshard_flat_param_indices,
            *split.fw_no_fsdp_flat_input_indices,
        )
        num_reused_parameters = num_flat_parameters
    else:
        compute_gm = traced_result.gm
        compute_flat_input_indices = tuple(range(len(input_names)))
        num_reused_parameters = 0

    first, boundary_specs, num_collective_boundaries = _build_first_microbatch_graph(
        traced_result.gm,
        traced_result,
        num_flat_parameters=num_flat_parameters,
        reuse_unsharded_parameters=reuse_unsharded_parameters,
    )
    if num_collective_boundaries == 0:
        raise ValueError("Deferred FSDP gradient sync found no gradient reduction")
    middle, middle_collective_boundaries = _build_middle_microbatch_graph(
        compute_gm,
        traced_result,
        boundary_specs,
    )
    final, final_collective_boundaries = _build_final_microbatch_graph(
        compute_gm,
        traced_result,
        boundary_specs,
    )
    if not (
        num_collective_boundaries
        == middle_collective_boundaries
        == final_collective_boundaries
    ):
        raise ValueError(
            "Deferred FSDP microbatch graphs found different gradient "
            "collective-boundary counts: "
            f"{num_collective_boundaries}, {middle_collective_boundaries}, "
            f"{final_collective_boundaries}"
        )

    first_collectives = _fsdp_collective_counts(first)
    middle_collectives = _fsdp_collective_counts(middle)
    final_collectives = _fsdp_collective_counts(final)
    if middle_collectives[0] != final_collectives[0]:
        raise ValueError(
            "Deferred FSDP reused graphs have different non-parameter "
            "all-gather counts: "
            f"{middle_collectives[0]} != {final_collectives[0]}"
        )
    num_all_gathers = (
        first_collectives[0] - middle_collectives[0]
        if reuse_unsharded_parameters
        else first_collectives[0]
    )
    if num_all_gathers <= 0:
        raise ValueError(
            "Deferred FSDP first graph did not retain parameter all-gather"
        )
    if not reuse_unsharded_parameters and (
        first_collectives[0] != middle_collectives[0]
        or middle_collectives[0] != final_collectives[0]
    ):
        raise ValueError(
            "Deferred FSDP resharded graphs have different all-gather counts"
        )

    first_reductions = first_collectives[1:]
    middle_reductions = middle_collectives[1:]
    final_reductions = final_collectives[1:]
    if first_reductions != middle_reductions:
        raise ValueError(
            "Deferred FSDP first and reused non-final graphs have different "
            "non-gradient reduction counts: "
            f"{first_reductions} != {middle_reductions}"
        )
    num_reduce_scatters = final_reductions[0] - middle_reductions[0]
    num_all_reduces = final_reductions[1] - middle_reductions[1]
    if num_reduce_scatters < 0 or num_all_reduces < 0:
        raise ValueError(
            "Deferred FSDP final graph removed reduction collectives: "
            f"middle={middle_reductions}, final={final_reductions}"
        )
    num_gradient_collectives = num_reduce_scatters + num_all_reduces
    if num_gradient_collectives < num_collective_boundaries:
        raise ValueError(
            "Deferred FSDP final graph restored "
            f"{num_gradient_collectives} gradient collectives for "
            f"{num_collective_boundaries} deferred reduction boundaries"
        )

    transformed_children = {
        name: _apply_child_graph_passes(
            child,
            graph_passes=post_accumulation_passes,
            compile_config=compile_config,
        )
        for name, child in (
            ("first", first),
            ("middle", middle),
            ("final", final),
        )
    }
    first_collectives = _fsdp_collective_counts(transformed_children["first"])
    middle_collectives = _fsdp_collective_counts(transformed_children["middle"])
    final_collectives = _fsdp_collective_counts(transformed_children["final"])
    if middle_collectives[0] != final_collectives[0]:
        raise ValueError(
            "Deferred FSDP bucketed graphs have different non-parameter "
            "all-gather counts"
        )
    if first_collectives[1:] != middle_collectives[1:]:
        raise ValueError(
            "Deferred FSDP bucketed non-final graphs have different "
            "non-gradient reduction counts"
        )
    num_all_gathers = (
        first_collectives[0] - middle_collectives[0]
        if reuse_unsharded_parameters
        else first_collectives[0]
    )
    middle_reductions = middle_collectives[1:]
    final_reductions = final_collectives[1:]
    num_reduce_scatters = final_reductions[0] - middle_reductions[0]
    num_all_reduces = final_reductions[1] - middle_reductions[1]
    num_gradient_collectives = num_reduce_scatters + num_all_reduces
    if num_all_gathers <= 0:
        raise ValueError(
            "Deferred FSDP gradient bucketing removed every parameter all-gather"
        )
    if num_reduce_scatters < 0 or num_all_reduces < 0:
        raise ValueError(
            "Deferred FSDP gradient bucketing removed final reductions: "
            f"middle={middle_reductions}, final={final_reductions}"
        )
    if num_gradient_collectives == 0:
        raise ValueError("Deferred FSDP gradient bucketing removed every reduction")
    _finalize_final_microbatch_graph(
        transformed_children["final"],
        traced_result,
        compute_flat_input_indices=compute_flat_input_indices,
    )

    children = {
        name: _compile_child(child, compile_config=compile_config)
        for name, child in transformed_children.items()
    }
    outer, example_inputs = _build_outer_graph(
        traced_result,
        first=children["first"],
        middle=children["middle"],
        final=children["final"],
        num_microbatches=num_microbatches,
        num_boundaries=len(boundary_specs),
        num_reused_parameters=num_reused_parameters,
        compute_flat_input_indices=compute_flat_input_indices,
    )
    cudagraph_compatible = is_cudagraph_compatible(outer) and all(
        is_cudagraph_compatible(child) for child in children.values()
    )
    outer.meta["cudagraph_compatible"] = cudagraph_compatible
    want_annotations = (
        enable_cudagraph
        and cudagraph_compatible
        and compile_config.inductor_compilation != "full"
        and "insert_kernel_annotations_pass" not in compile_config.disable_passes
    )
    if want_annotations:
        for child in children.values():
            insert_kernel_annotations_pass(child)
    tensor_input_indices = tuple(
        index
        for index, value in enumerate(example_inputs)
        if isinstance(value, torch.Tensor)
    )
    if enable_cudagraph:
        outer = cudagraph_pass(
            outer,
            example_inputs,
            static_input_indices=list(range(traced_result.num_static_inputs)),
            tensor_input_indices=list(tensor_input_indices),
        )
    loss_layout = traced_result.output_subclass_layouts.get(0)
    user_call = pytree.tree_unflatten(
        range(traced_result.user_inputs_spec.num_leaves),
        traced_result.user_inputs_spec,
    )
    deferred_user_inputs_spec = pytree.tree_flatten(
        ((user_call,) * num_microbatches, {})
    )[1]
    deferred_traced_result = replace(
        traced_result,
        gm=outer,
        example_inputs=example_inputs,
        num_flat_inputs=len(example_inputs),
        user_inputs_spec=deferred_user_inputs_spec,
        tensor_input_indices=list(tensor_input_indices),
        num_flat_outputs=1,
        output_subclass_layouts={0: loss_layout} if loss_layout is not None else {},
        output_spec=pytree.tree_flatten(0)[1],
    )
    return GraphWithDeferredFSDPReductions(
        gm=outer,
        traced_result=deferred_traced_result,
        num_microbatches=num_microbatches,
        num_all_gathers=num_all_gathers,
        num_reduce_scatters=num_reduce_scatters,
        num_all_reduces=num_all_reduces,
    )
