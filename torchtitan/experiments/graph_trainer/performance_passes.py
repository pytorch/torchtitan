# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graph passes that improve performance but may change numerics.

Gated behind ``--compile.numerics_changing_optim`` (opt-in, default off).
"""

import logging
import operator
from dataclasses import dataclass

import torch
import torch.fx as fx


logger = logging.getLogger(__name__)


def annotate_rmsnorm_for_regional_inductor_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    rmsnorm_compile_config: dict | None = None,
) -> torch.fx.GraphModule:
    """Tag RMSNorm ops with compile_with_inductor for regional_inductor.

    ``nn.RMSNorm`` calls ``F.rms_norm``, which the autograd dispatch path
    lowers to ``aten._fused_rms_norm`` during ``make_fx`` tracing with
    backward (no explicit decomposition table needed).  This pass finds
    those ``_fused_rms_norm`` / ``_fused_rms_norm_backward`` nodes by
    ``node.target`` and tags them (along with their ``getitem`` users) so
    that ``regional_inductor_pass`` compiles each norm as a fused Inductor
    kernel.

    Args:
        gm: The graph module to annotate.
        example_inputs: Example inputs (unused, required by pass interface).
        rmsnorm_compile_config: Inductor config dict for RMSNorm nodes.
            When provided, wrapped as ``{"inductor_configs": rmsnorm_compile_config}``.
            Default is None (no inductor_configs in the annotation).
    """
    compile_annotation: dict = (
        {"inductor_configs": rmsnorm_compile_config}
        if rmsnorm_compile_config is not None
        else {}
    )

    _RMSNORM_TARGETS = {
        torch.ops.aten._fused_rms_norm.default,
        torch.ops.aten._fused_rms_norm_backward.default,
    }

    num_tagged = 0

    for node in gm.graph.nodes:
        if node.target not in _RMSNORM_TARGETS:
            continue

        node.meta.setdefault("custom", {})["compile_with_inductor"] = compile_annotation
        num_tagged += 1

        # Tag getitem users that extract outputs from the fused op.
        for user in node.users:
            if user.target is operator.getitem:
                user.meta.setdefault("custom", {})[
                    "compile_with_inductor"
                ] = compile_annotation

    if num_tagged > 0:
        logger.info(
            f"Tagged {num_tagged} RMSNorm nodes for regional Inductor compilation"
        )

    return gm


@dataclass
class _ChunkedLossRSGroup:
    collectives: list[fx.Node]
    nodes: set[fx.Node]


def _static_tensor_layout(node: fx.Node) -> tuple | None:
    value = node.meta.get("val")
    if not isinstance(value, torch.Tensor):
        return None
    shape, stride = tuple(value.shape), tuple(value.stride())
    if not all(isinstance(size, int) for size in (*shape, *stride)):
        return None
    return shape, stride, value.dtype, value.device


def _match_chunked_loss_rs(output: fx.Node) -> _ChunkedLossRSGroup | None:
    """Match one tagged gradient's exclusive RS sum chain.

    Only cross layout-preserving ops, not dtype changes, scaling or all-reduces.
    """
    identity_ops = {
        torch.ops.aten.alias.default,
        torch.ops.aten.detach.default,
        torch.ops.aten.clone.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten._to_copy.default,
    }
    add_ops = {torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor}
    rs_target = torch.ops._c10d_functional.reduce_scatter_tensor.default
    wait_target = torch.ops._c10d_functional.wait_tensor.default
    output_layout = _static_tensor_layout(output)
    if output_layout is None or output_layout[2] not in (
        torch.float32,
        torch.float64,
    ):
        return None

    nodes: set[fx.Node] = set()
    collectives: list[fx.Node] = []
    # Head-module annotations can override the loss scope. Derive the module
    # from parameter identities instead of hard-coding the head name.
    parameter_modules = {
        name.rpartition(".")[0]
        for name in output.meta.get("custom", {}).get("parameter_gradient_fqns", ())
        if isinstance(name, str) and "." in name
    }

    def visit(node: fx.Node) -> bool:
        if node in nodes or node.op != "call_function":
            return False
        nodes.add(node)
        if _static_tensor_layout(node) != output_layout:
            return False
        if node.target in identity_ops:
            if len(node.all_input_nodes) != 1:
                return False
            return visit(node.all_input_nodes[0])
        if node.target is torch.ops.aten.copy_.default:
            if len(node.args) != 2 or node.kwargs:
                return False
            destination, source = node.args
            # Only absorb copy_ into a fresh, exclusive AccumulateGrad buffer,
            # never into an existing optimizer buffer.
            if (
                not isinstance(destination, fx.Node)
                or not isinstance(source, fx.Node)
                or destination.target is not torch.ops.aten.new_empty_strided.default
                or destination.all_input_nodes != [source]
                or set(destination.users) != {node}
                or _static_tensor_layout(destination) != output_layout
                or destination in nodes
            ):
                return False
            nodes.add(destination)
            return visit(source)
        if node.target in add_ops:
            if len(node.args) != 2 or set(node.kwargs) - {"alpha"}:
                return False
            if node.kwargs.get("alpha", 1) != 1:
                return False
            lhs, rhs = node.args
            return (
                isinstance(lhs, fx.Node)
                and isinstance(rhs, fx.Node)
                and visit(lhs)
                and visit(rhs)
            )
        if node.target != wait_target or len(node.args) != 1 or node.kwargs:
            return False
        rs = node.args[0]
        if (
            not isinstance(rs, fx.Node)
            or rs.target != rs_target
            or len(rs.args) != 4
            or rs.kwargs
            or rs in nodes
        ):
            return False
        module_fqn = rs.meta.get("custom", {}).get("module_fqn", "")
        if not isinstance(module_fqn, str) or (
            module_fqn != "loss"
            and not module_fqn.startswith("loss.")
            and module_fqn not in parameter_modules
        ):
            return False
        nodes.add(rs)
        collectives.append(rs)
        return True

    if not visit(output) or len(collectives) < 2:
        return None
    # Intermediate aliases must not expose partial sums or mutable RS outputs.
    if any(
        user not in nodes for node in nodes if node is not output for user in node.users
    ):
        return None

    first = collectives[0]
    if (
        first.args[1] != "sum"
        or not isinstance(first.args[2], int)
        or first.args[2] <= 1
        or not isinstance(first.args[0], fx.Node)
    ):
        return None
    input_layout = _static_tensor_layout(first.args[0])
    if input_layout is None or input_layout[2:] != output_layout[2:]:
        return None
    input_shape, _, _, _ = input_layout
    output_shape, _, _, _ = output_layout
    if (
        not input_shape
        or not output_shape
        or input_shape[0] != output_shape[0] * first.args[2]
        or input_shape[1:] != output_shape[1:]
        or not first.args[0].meta["val"].is_contiguous()
    ):
        return None
    for rs in collectives:
        if (
            rs.args[1:] != first.args[1:]
            or not isinstance(rs.args[0], fx.Node)
            or _static_tensor_layout(rs.args[0]) != input_layout
            or _static_tensor_layout(rs) != output_layout
        ):
            return None
    return _ChunkedLossRSGroup(collectives, nodes)


def coalesce_chunked_loss_rs_pass(
    gm: fx.GraphModule,
    example_inputs=None,
) -> fx.GraphModule:
    """Coalesce one parameter's FP32/FP64 SUM reduce-scatters per microbatch.

    Requires gradient-identity metadata; unsupported chains remain unchanged.
    Run after marker removal and DCE, before canonicalization and bucketing.
    Changes floating-point reduction order, not cross-microbatch accumulation.
    """
    del example_inputs
    graph = gm.graph
    order = {node: index for index, node in enumerate(graph.nodes)}
    # Parameter identities are populated by remove_parameter_gradient_markers_pass.
    outputs = [
        node
        for node in graph.nodes
        if node.meta.get("custom", {}).get("parameter_gradient_fqns")
    ]
    removed: set[fx.Node] = set()
    reports: list[dict] = []
    for output in reversed(outputs):
        if output in removed:
            continue
        group = _match_chunked_loss_rs(output)
        if group is None:
            if output.meta.get("custom", {}).get("module_fqn") == "loss":
                logger.warning(
                    "Skipping chunked-loss RS coalescing for %s: gradient chain "
                    "is unsupported or already synchronized in one collective.",
                    output.meta["custom"]["parameter_gradient_fqns"],
                )
            continue
        collectives = sorted(group.collectives, key=order.__getitem__)
        accumulator: fx.Node | None = None
        for rs in collectives:
            gradient = rs.args[0]
            assert isinstance(gradient, fx.Node)
            with graph.inserting_before(rs):
                if accumulator is None:
                    # Own the buffer: the original input may alias a graph input
                    # or saved activation. Never mutate those during replay.
                    accumulator = graph.call_function(
                        torch.ops.aten.clone.default, (gradient,)
                    )
                else:
                    accumulator = graph.call_function(
                        torch.ops.aten.add_.Tensor, (accumulator, gradient)
                    )
                accumulator.meta = {
                    **gradient.meta,
                    "custom": {**gradient.meta.get("custom", {})},
                }
        assert accumulator is not None
        last_rs = collectives[-1]
        with graph.inserting_before(last_rs):
            new_rs = graph.call_function(
                torch.ops._c10d_functional.reduce_scatter_tensor.default,
                (accumulator, *last_rs.args[1:]),
            )
            new_rs.meta = {
                **last_rs.meta,
                "custom": {**last_rs.meta.get("custom", {})},
            }
            new_wait = graph.call_function(
                torch.ops._c10d_functional.wait_tensor.default, (new_rs,)
            )
            new_wait.meta = {
                **output.meta,
                "custom": {**output.meta.get("custom", {})},
            }
        output.replace_all_uses_with(new_wait)
        for node in sorted(group.nodes, key=order.__getitem__, reverse=True):
            graph.erase_node(node)
        removed.update(group.nodes)
        value = accumulator.meta["val"]
        num_bytes = value.numel() * value.element_size()
        report = {
            "parameter_fqns": output.meta["custom"]["parameter_gradient_fqns"],
            "num_rs_before": len(collectives),
            "num_rs_after": 1,
            "input_bytes_before": len(collectives) * num_bytes,
            "input_bytes_after": num_bytes,
        }
        reports.append(report)
        logger.info("Coalesced chunked-loss RS: %s", report)

    gm.meta["chunked_loss_rs_coalescing"] = reports
    if reports:
        graph.lint()
        gm.recompile()
    else:
        logger.warning(
            "Chunked-loss RS coalescing found no supported gradient chain; "
            "leaving the graph unchanged. Requires tagged parameter gradients, "
            "exclusive loss/head FP32/FP64 SUM RS results, matching static "
            "layouts, and no intervening casts, scaling or all-reduces."
        )
    return gm
