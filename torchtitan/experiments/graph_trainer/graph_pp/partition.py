# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Stage-local GraphPP graph partitioning.

GraphTrainer traces one flat joint graph per pipeline stage. The joint graph
returns forward user outputs first, then gradients. GraphPP splits that graph
into a forward callable and a backward callable with a stable calling
convention for PP runtime schedules. ``GraphMeta`` records that calling
convention: ordered placeholder names describe each extracted graph API, and
flat input indices map runtime graph inputs back to the traced GraphTrainer
input list.

Terms used by this pass:

- flat inputs: traced GraphTrainer inputs. GraphTrainer orders them as params,
  buffers, then user forward inputs, followed by schedule-supplied backward
  inputs when the trace contains them.
- forward user outputs: values returned by stage forward, such as
  activation_to_next or scalar_loss.
- backward outputs: gradients returned after forward user outputs; parameter
  grads come before input_grads_to_prev when input grads exist.
- backward_only_input_indices: indices into flat inputs for values that must
  not be forward graph inputs because the PP schedule supplies them only while
  executing backward.
- backward_grad_input_indices: indices into the schedule-supplied backward
  input tuple. They are not indices into the full traced input list.
- backward grad inputs: the concrete backward-time values named by
  backward_only_input_indices. For intermediate stages this is
  output_grads_from_next. The last stage relies on autograd's implicit scalar
  loss seed and has no schedule-supplied backward grad input.
- saved_values_for_backward: forward-computed values returned by the forward
  callable so the backward callable can consume them later.
- backward passthrough placeholders: metadata-only placeholders needed to
  rewrap backward outputs, such as DTensor class/layout metadata.

Representative signatures:

First stage, non-last:
  joint(params, buffers, microbatch_inputs, output_grads_from_next)
    -> activation_to_next, param_grads
  forward(params, buffers, microbatch_inputs)
    -> activation_to_next, saved_values_for_backward, mutation_outputs
  backward(saved_values_for_backward, output_grads_from_next)
    -> param_grads

Intermediate stage:
  joint(params, buffers, activation_from_prev, output_grads_from_next)
    -> activation_to_next, param_grads, input_grads_to_prev
  forward(params, buffers, activation_from_prev)
    -> activation_to_next, saved_values_for_backward, mutation_outputs
  backward(saved_values_for_backward, output_grads_from_next)
    -> param_grads, input_grads_to_prev

Last stage:
  joint(params, buffers, activation_from_prev, target_and_loss_inputs)
    -> scalar_loss, param_grads, input_grads_to_prev
  forward(params, buffers, activation_from_prev, target_and_loss_inputs)
    -> scalar_loss, saved_values_for_backward, mutation_outputs
  backward(saved_values_for_backward)
    -> param_grads, input_grads_to_prev

The first stage usually has no input_grads_to_prev because real microbatch
inputs generally do not require gradients.
"""

import copy
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch.fx as fx
from torch._functorch.partitioners import (
    _extract_fwd_bwd_outputs,
    _extract_graph_with_inputs_outputs,
)
from torch.fx._lazy_graph_module import _make_graph_module

from torchtitan.experiments.graph_trainer.graph_pp.utils import (
    base_tensor_for_mutation_target,
    is_getitem_node,
    is_mutation_node,
    node_closure,
    node_order,
    output_names,
    placeholder_dependencies,
    placeholder_names,
    trace_graph_pp_graph,
    unique_in_order,
)

from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult


@dataclass(frozen=True, slots=True)
class GraphMeta:
    """Calling convention for GraphTrainer-native GraphPP fwd/bwd graphs.

    ``fwd_input_names`` names the forward graph placeholders, while
    ``fwd_flat_input_indices`` maps each placeholder to the traced flat input
    list. Since flat inputs are params, buffers, then user inputs, the runner
    can split those indices into state inputs and user inputs using stage
    metadata.

    ``backward_grad_input_names`` names backward placeholders supplied by the
    PP schedule at backward time, not values saved by forward.
    ``backward_grad_input_indices`` indexes into the schedule-provided backward
    tuple: output_grads_from_next for non-last stages. The last stage has no
    entries because scalar losses use autograd's implicit grad seed.

    ``fwd_side_effect_output_names`` names extra forward outputs appended after
    saved values only to keep forward-side mutations live. The runtime ignores
    their values after the graph has executed.

    Attributes:
        fwd_input_names (tuple[str, ...]): Forward graph placeholder names.
        fwd_flat_input_indices (tuple[int, ...]): Indices mapping each forward
            placeholder back to the traced GraphTrainer flat input list.
        num_fwd_user_outputs (int): Number of leading forward graph outputs
            that correspond to real stage forward outputs.
        saved_for_backward_names (tuple[str, ...]): Forward outputs consumed
            later as leading backward graph inputs.
        fwd_side_effect_output_names (tuple[str, ...]): Extra forward outputs
            used only to materialize forward-side mutations.
        backward_grad_input_names (tuple[str, ...]): Backward placeholders
            supplied by the PP schedule, such as output_grads_from_next.
        backward_grad_input_indices (tuple[int, ...]): Indices into the
            schedule-supplied backward input tuple.
        bwd_input_names (tuple[str, ...]): Backward graph placeholder names.
        bwd_output_names (tuple[str, ...]): Backward graph output names.
    """

    fwd_input_names: tuple[str, ...]
    fwd_flat_input_indices: tuple[int, ...]
    num_fwd_user_outputs: int
    saved_for_backward_names: tuple[str, ...]
    fwd_side_effect_output_names: tuple[str, ...]
    backward_grad_input_names: tuple[str, ...]
    backward_grad_input_indices: tuple[int, ...]
    bwd_input_names: tuple[str, ...]
    bwd_output_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.fwd_input_names) != len(self.fwd_flat_input_indices):
            raise ValueError(
                "Forward input metadata must pair each placeholder name with "
                "one traced flat input index: "
                f"{self.fwd_input_names} vs {self.fwd_flat_input_indices}"
            )
        if len(self.backward_grad_input_names) != len(self.backward_grad_input_indices):
            raise ValueError(
                "Backward grad input metadata must pair each placeholder name "
                "with one schedule input index: "
                f"{self.backward_grad_input_names} vs "
                f"{self.backward_grad_input_indices}"
            )

    @property
    def num_fwd_inputs(self) -> int:
        """int: Number of forward graph placeholders."""
        return len(self.fwd_input_names)

    @property
    def num_saved_for_backward(self) -> int:
        """int: Number of forward outputs saved for backward."""
        return len(self.saved_for_backward_names)

    @property
    def num_fwd_side_effect_outputs(self) -> int:
        """int: Number of forward outputs kept only for side effects."""
        return len(self.fwd_side_effect_output_names)

    @property
    def num_bwd_inputs(self) -> int:
        """int: Number of backward graph placeholders."""
        return len(self.bwd_input_names)

    @property
    def num_backward_grad_inputs(self) -> int:
        """int: Number of backward placeholders supplied by the PP schedule."""
        return len(self.backward_grad_input_names)

    @property
    def num_bwd_outputs(self) -> int:
        """int: Number of backward graph outputs."""
        return len(self.bwd_output_names)


def _forward_flat_input_indices(
    gm: fx.GraphModule,
    placeholder_index_by_name: dict[str, int],
) -> tuple[int, ...]:
    """Map extracted forward placeholders to traced flat input positions."""

    flat_input_indices: list[int] = []
    for node in gm.graph.find_nodes(op="placeholder"):
        if node.name not in placeholder_index_by_name:
            raise ValueError(
                "Extracted graph placeholder does not come from the traced "
                f"GraphTrainer flat input list: {node.name}"
            )
        flat_input_indices.append(placeholder_index_by_name[node.name])
    return tuple(flat_input_indices)


def _validate_backward_only_input_indices(
    indices: tuple[int, ...],
    *,
    num_placeholders: int,
) -> None:
    """Validate caller-selected traced inputs that only backward may consume."""

    if len(set(indices)) != len(indices):
        raise ValueError(
            "backward_only_input_indices must be unique, got " f"{list(indices)}"
        )
    invalid_indices = sorted(
        index for index in indices if index < 0 or index >= num_placeholders
    )
    if invalid_indices:
        raise ValueError(
            "backward_only_input_indices must reference traced graph "
            f"placeholders: {invalid_indices} for {num_placeholders} "
            "placeholders"
        )


def _invalid_backward_only_value_names(
    values: Iterable[object],
    backward_only_names: set[str],
) -> list[str]:
    """Return backward-only placeholders that were selected for forward save."""

    return sorted(
        {
            value.name
            for value in values
            if isinstance(value, fx.Node) and value.name in backward_only_names
        }
    )


def _saved_values_for_backward(
    joint: fx.GraphModule,
    *,
    fwd_outputs: Sequence[object],
    bwd_outputs: Sequence[object],
    backward_only_names: set[str],
) -> list[fx.Node]:
    """Select forward-computed values needed by the backward callable.

    A value is saved when it is in the forward dependency closure and has a
    later user that belongs only to the backward dependency closure. This also
    covers values used by both forward and backward: the forward callable
    returns them once, and the PP runtime passes them to backward later.
    """
    forward_nodes = node_closure(fwd_outputs)
    backward_nodes = node_closure(bwd_outputs)
    saved: list[fx.Node] = []
    for node in joint.graph.nodes:
        if node not in forward_nodes or node.name in backward_only_names:
            continue
        if any(
            user in backward_nodes and user not in forward_nodes for user in node.users
        ):
            saved.append(node)
    return saved


def _forward_mutations_to_materialize(
    joint: fx.GraphModule,
    *,
    fwd_outputs: Sequence[object],
    bwd_outputs: Sequence[object],
    backward_only_names: set[str],
) -> tuple[list[fx.Node], list[fx.Node]]:
    """Find forward-side mutations that must execute in the forward callable.

    Chunked loss traces populate the hidden-state gradient accumulator with
    ``copy_`` mutations during the loss forward. The accumulator is later
    consumed by decoder backward, but those writes are not visible in the data
    dependency closure of the loss output. Preserve the mutated base as a saved
    value and keep the mutation nodes as forward-only outputs so extracted
    forward graph execution materializes the post-mutation tensor.

    Other forward mutations, such as MoE expert-count buffer updates, are not
    consumed by backward but still affect training state. Keep those mutation
    nodes as forward-only outputs as well; the runtime ignores the extra values,
    but returning them keeps the mutations live in the extracted graph.
    """

    order = node_order(joint.graph)
    fwd_output_nodes = [value for value in fwd_outputs if isinstance(value, fx.Node)]
    if not fwd_output_nodes:
        return [], []
    fwd_output_barrier = max(order[node] for node in fwd_output_nodes)
    backward_nodes = node_closure(bwd_outputs)
    saved_mutation_bases: list[fx.Node] = []
    mutation_outputs: list[fx.Node] = []

    for node in joint.graph.nodes:
        if (
            order[node] > fwd_output_barrier
            or node in backward_nodes
            or not is_mutation_node(node)
        ):
            continue
        if not node.args:
            continue
        mutated_base = base_tensor_for_mutation_target(node.args[0])
        if mutated_base is None:
            continue
        if mutated_base.name in backward_only_names:
            raise ValueError(
                "Forward mutation cannot target a backward-only input: "
                f"mutation={node.name}, target={mutated_base.name}"
            )
        mutation_outputs.append(node)
        if mutated_base.op != "placeholder" and mutated_base in backward_nodes:
            saved_mutation_bases.append(mutated_base)

    return (
        unique_in_order(saved_mutation_bases),
        unique_in_order(mutation_outputs),
    )


def _backward_passthrough_placeholders(
    *,
    bwd_outputs: Sequence[object],
    backward_only_names: set[str],
) -> list[fx.Node]:
    """Preserve metadata placeholders returned by backward directly.

    minimal_fx_tracer unwraps tensor subclasses into plain graph values. A
    DTensor gradient, for example, may flatten to ``(local_grad, device_mesh)``,
    where ``device_mesh`` and layout state are input placeholders carried
    through only so the runtime can rewrap the output subclass. Since such
    placeholders have no compute users, dependency-based saved-value discovery
    will not select them; they still must be available to the extracted
    backward graph.
    """
    return unique_in_order(
        node
        for node in bwd_outputs
        if isinstance(node, fx.Node)
        and node.op == "placeholder"
        and node.name not in backward_only_names
    )


def _is_tuple_like_node(node: fx.Node) -> bool:
    return isinstance(node.meta.get("val"), (tuple, list))


def _flatten_saved_values_for_backward(
    joint: fx.GraphModule,
    *,
    saved_values: list[fx.Node],
    bwd_outputs: Sequence[object],
) -> list[fx.Node]:
    """Expose saved tuple intermediates as tensor leaves.

    Higher-order ops such as FlexAttention can return nested tuples whose tensor
    leaves are consumed by backward. Keeping the raw tuple as a forward output
    works for interpreted FX, but standalone regional Inductor expects compiled
    regions to expose plain tensor outputs. If backward only observes the tuple
    through getitem chains, save those getitem leaves directly.
    """

    backward_nodes = node_closure(bwd_outputs)
    order = node_order(joint.graph)

    def flatten_if_backward_uses_items(node: fx.Node) -> list[fx.Node]:
        if not _is_tuple_like_node(node):
            return [node]

        backward_users = [user for user in node.users if user in backward_nodes]
        getitem_users = [user for user in backward_users if is_getitem_node(user)]
        if not getitem_users or len(getitem_users) != len(backward_users):
            return [node]

        flattened_items: list[fx.Node] = []
        for user in sorted(getitem_users, key=order.__getitem__):
            flattened_items.extend(flatten_if_backward_uses_items(user))
        return flattened_items

    flattened: list[fx.Node] = []
    for node in saved_values:
        flattened.extend(flatten_if_backward_uses_items(node))
    return unique_in_order(flattened)


def _forward_inputs_from_outputs(
    placeholders: list[fx.Node],
    *,
    fw_outputs: Sequence[object],
    backward_only_names: set[str],
) -> list[fx.Node]:
    """Select forward placeholders and reject backward-only dependencies."""

    fw_input_set = placeholder_dependencies(fw_outputs)
    invalid_names = sorted(
        node.name for node in fw_input_set if node.name in backward_only_names
    )
    if invalid_names:
        raise ValueError(
            "Forward graph outputs require backward-only inputs: " f"{invalid_names}"
        )
    return [node for node in placeholders if node in fw_input_set]


def _backward_grad_inputs_from_schedule(
    backward_only_inputs: list[fx.Node],
    *,
    bwd_outputs: Sequence[object],
) -> tuple[list[fx.Node], tuple[int, ...]]:
    """Return schedule-supplied grad placeholders consumed by backward.

    For non-last stages, the PP schedule supplies output_grads_from_next. The
    last stage has no schedule-supplied backward grad inputs. The returned
    indices are positions in that backward-time tuple, not positions in the
    full traced input list.
    """

    bwd_input_set = placeholder_dependencies(bwd_outputs)
    backward_grad_inputs = [
        node for node in backward_only_inputs if node in bwd_input_set
    ]
    backward_grad_input_indices = tuple(
        index
        for index, node in enumerate(backward_only_inputs)
        if node in bwd_input_set
    )
    return backward_grad_inputs, backward_grad_input_indices


def partition_joint_graph(
    traced: TracedResult,
    *,
    num_fwd_outputs: int,
    backward_only_input_indices: tuple[int, ...] = (),
) -> tuple[fx.GraphModule, fx.GraphModule, GraphMeta]:
    """Partition a post-pass GraphTrainer joint graph into GraphPP callables.

    The caller passes a flat ``minimal_fx_tracer`` graph after any
    metadata-preserving GraphTrainer passes. ``num_fwd_outputs`` counts the raw
    forward user outputs after tensor subclasses are unwrapped. Semantic pytree
    structure is restored by GraphPPRunner at the runtime boundary.

    Args:
        traced (TracedResult): Flat stage-local joint graph produced by
            ``minimal_fx_tracer``.
        num_fwd_outputs (int): Number of leading traced outputs that belong to
            the stage forward result.
        backward_only_input_indices (tuple[int, ...]): Flat input indices for
            values supplied only during backward by the PP schedule. These
            inputs are excluded from the forward graph.

    Returns:
        tuple[fx.GraphModule, fx.GraphModule, GraphMeta]: Forward graph,
        backward graph, and their GraphPP calling-convention metadata.

    Raises:
        ValueError: If the requested output/input partition violates the
            GraphPP calling convention or would silently save backward-only
            inputs through the forward graph.
    """
    if num_fwd_outputs < 1:
        raise ValueError(f"num_fwd_outputs must be positive, got {num_fwd_outputs}")

    joint = copy.deepcopy(traced.gm)
    trace_graph_pp_graph("graph_pp_partition_joint", joint)
    placeholders = list(joint.graph.find_nodes(op="placeholder"))
    placeholder_index_by_name = {
        node.name: index for index, node in enumerate(placeholders)
    }
    _validate_backward_only_input_indices(
        backward_only_input_indices,
        num_placeholders=len(placeholders),
    )
    backward_only_index_set = set(backward_only_input_indices)
    backward_only_inputs = [
        node
        for index, node in enumerate(placeholders)
        if index in backward_only_index_set
    ]
    backward_only_names = {node.name for node in backward_only_inputs}

    (
        fwd_outputs,
        bwd_outputs,
        fwd_output_descs,
        bwd_output_descs,
    ) = _extract_fwd_bwd_outputs(joint, num_fwd_outputs=num_fwd_outputs)
    if len(fwd_outputs) != num_fwd_outputs:
        raise ValueError(
            "num_fwd_outputs exceeds traced graph output count: "
            f"requested {num_fwd_outputs}, found {len(fwd_outputs) + len(bwd_outputs)}"
        )

    # 1. Select values produced by forward and later consumed by backward.
    saved_values = _saved_values_for_backward(
        joint,
        fwd_outputs=fwd_outputs,
        bwd_outputs=bwd_outputs,
        backward_only_names=backward_only_names,
    )

    # 2. Add metadata-only placeholders needed to rewrap backward outputs.
    saved_values.extend(
        _backward_passthrough_placeholders(
            bwd_outputs=bwd_outputs,
            backward_only_names=backward_only_names,
        )
    )

    # 3. Preserve forward tensor mutations whose return values are otherwise
    # dead from the perspective of forward user outputs.
    (mutation_saved_values, fwd_mutation_outputs,) = _forward_mutations_to_materialize(
        joint,
        fwd_outputs=fwd_outputs,
        bwd_outputs=bwd_outputs,
        backward_only_names=backward_only_names,
    )
    saved_values = unique_in_order([*saved_values, *mutation_saved_values])
    invalid_saved_names = _invalid_backward_only_value_names(
        saved_values,
        backward_only_names,
    )
    if invalid_saved_names:
        raise ValueError(
            "saved_values_for_backward must not include backward-only inputs: "
            f"{invalid_saved_names}"
        )

    # 4. Expose tuple saved values as leaves when backward only observes the
    # leaves through getitem chains.
    saved_values = _flatten_saved_values_for_backward(
        joint,
        saved_values=saved_values,
        bwd_outputs=bwd_outputs,
    )

    # 5. Select the concrete calling convention and extract both subgraphs.
    fw_outputs = fwd_outputs + saved_values + fwd_mutation_outputs
    fw_output_descs = fwd_output_descs + [None] * (
        len(saved_values) + len(fwd_mutation_outputs)
    )
    fw_inputs = _forward_inputs_from_outputs(
        placeholders,
        fw_outputs=fw_outputs,
        backward_only_names=backward_only_names,
    )

    (
        backward_grad_inputs,
        backward_grad_input_indices,
    ) = _backward_grad_inputs_from_schedule(
        backward_only_inputs,
        bwd_outputs=bwd_outputs,
    )
    bw_inputs = saved_values + backward_grad_inputs

    fw_graph = _extract_graph_with_inputs_outputs(
        joint.graph,
        fw_inputs,
        fw_outputs,
        fw_output_descs,
        "forward",
        ignore_must_be_in_fw_bw=True,
    )
    bw_graph = _extract_graph_with_inputs_outputs(
        joint.graph,
        bw_inputs,
        bwd_outputs,
        bwd_output_descs,
        "backward",
        ignore_must_be_in_fw_bw=True,
    )
    fw_module = _make_graph_module(joint, fw_graph)
    bw_module = _make_graph_module(joint, bw_graph)
    fw_module.graph.lint()
    bw_module.graph.lint()
    fw_module.recompile()
    bw_module.recompile()
    trace_graph_pp_graph("graph_pp_partition_forward", fw_module)
    trace_graph_pp_graph("graph_pp_partition_backward", bw_module)

    saved_for_backward_names = output_names(fw_module)[
        num_fwd_outputs : num_fwd_outputs + len(saved_values)
    ]
    bwd_input_names = placeholder_names(bw_module)
    expected_bwd_input_names = tuple(node.name for node in bw_inputs)
    if bwd_input_names != expected_bwd_input_names:
        raise ValueError(
            "Backward graph inputs must be saved values followed by "
            "schedule-supplied backward grad inputs: "
            f"expected {expected_bwd_input_names}, got {bwd_input_names}"
        )
    if bwd_input_names[: len(saved_for_backward_names)] != saved_for_backward_names:
        raise ValueError(
            "Forward saved output names must match backward saved input names: "
            f"saved={saved_for_backward_names}, backward={bwd_input_names}"
        )

    return (
        fw_module,
        bw_module,
        GraphMeta(
            fwd_input_names=placeholder_names(fw_module),
            fwd_flat_input_indices=_forward_flat_input_indices(
                fw_module, placeholder_index_by_name
            ),
            num_fwd_user_outputs=num_fwd_outputs,
            saved_for_backward_names=saved_for_backward_names,
            fwd_side_effect_output_names=output_names(fw_module)[
                num_fwd_outputs + len(saved_values) :
            ],
            backward_grad_input_names=tuple(node.name for node in backward_grad_inputs),
            backward_grad_input_indices=backward_grad_input_indices,
            bwd_input_names=bwd_input_names,
            bwd_output_names=output_names(bw_module),
        ),
    )
