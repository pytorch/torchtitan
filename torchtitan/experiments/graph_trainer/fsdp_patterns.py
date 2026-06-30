# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Pattern helpers for GraphPP's current simple-FSDP collective traces.

The matchers intentionally follow c10d functional traces produced by FSDP2:

    param_shard -> all_gather -> wait -> view*/split-cat -> compute
    local_grad -> cast/view* -> reduce_scatter -> wait -> param_grad
    local_grad -> cast/view* -> all_reduce -> wait -> param_grad

They are structural helpers for today's trace shape. A future upstream FSDP or
torch.pipelining annotation should replace this with explicit collective-region
metadata instead of broader pattern matching.
"""

import operator
from typing import Any

import torch
import torch.fx as fx


def is_wait_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
    )


def is_all_gather_into_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_gather_into_tensor.default
    )


def is_reduce_scatter_tensor(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.reduce_scatter_tensor.default
    )


def is_all_reduce(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target is torch.ops._c10d_functional.all_reduce.default
    )


def is_reduce_grad_collective(node: fx.Node) -> bool:
    return is_reduce_scatter_tensor(node) or is_all_reduce(node)


def _find_last_all_gather_in_chain(start_node: fx.Node) -> fx.Node | None:
    """Find the final all-gather in a linear FSDP unshard launch chain."""
    node = start_node
    last_all_gather = None
    while True:
        if len(node.users) != 1:
            break
        user = next(iter(node.users))
        if len(user.all_input_nodes) > 1:
            break
        node = user
        if is_all_gather_into_tensor(node):
            last_all_gather = node
    return last_all_gather


def _find_last_user_in_wait_chain(wait_node: fx.Node) -> fx.Node:
    """Find the last FSDP unshard node before the value enters real compute.

    The traced FSDP unshard has a mostly linear shape:

        flat_param -> ... -> all_gather -> wait -> view* -> compute

    Some models reshape the gathered flat buffer through a split/cat fanout:

        wait -> split -> getitem_0 --+
                      -> getitem_1 --+-> cat -> view* -> compute

    In both cases the FSDP region ends before the first consumer with multiple
    FX inputs. The split/getitem/cat fanout is still part of reconstructing the
    unsharded parameter value, so it is included in the chain.
    """
    node = wait_node
    while True:
        if len(node.users) != 1:
            if (
                node.op == "call_function"
                and node.target == torch.ops.aten.split.Tensor
                and all(
                    user.op == "call_function"
                    and user.target == operator.getitem
                    and len(user.users) == 1
                    for user in node.users
                )
            ):
                getitem_users = [next(iter(user.users)) for user in node.users]
                potential_cat = getitem_users[0]
                if all(user == potential_cat for user in getitem_users) and (
                    potential_cat.op == "call_function"
                    and potential_cat.target == torch.ops.aten.cat.default
                ):
                    node = potential_cat
                    continue
            break

        user = next(iter(node.users))
        if len(user.all_input_nodes) > 1:
            break
        node = user
    return node


def _find_last_non_view_node_in_chain(node: fx.Node) -> fx.Node:
    """Return the value GraphPP should pass across the unshard boundary."""
    result = node
    while hasattr(result.target, "is_view") and result.target.is_view:
        if len(result.all_input_nodes) != 1:
            raise ValueError(f"View node {result.name} should have exactly one input")
        result = result.all_input_nodes[0]
    return result


def find_fsdp_unshard_output(param_placeholder: fx.Node) -> fx.Node | None:
    """Return the extracted unshard output for one flat parameter input.

    GraphPP and the SAC force-save policy must agree on this node. It is the
    same value AutoParallel saves for ``reshard_after_forward=False``: the last
    non-view node in the FSDP all-gather/wait reconstruction chain. Parameters
    without an all-gather are replicated or otherwise already local, so callers
    should keep the original placeholder as that parameter's unsharded value.

    TODO(sanketpurandare): requires upstream change: FSDP trace/passes should
    annotate unshard collective regions for downstream graph extraction.
    """
    last_all_gather = _find_last_all_gather_in_chain(param_placeholder)
    if last_all_gather is None:
        return None

    if len(last_all_gather.users) != 1:
        raise ValueError(
            f"Expected one wait_tensor user for all_gather node {last_all_gather.name}, "
            f"got {len(last_all_gather.users)}"
        )
    wait_node = next(iter(last_all_gather.users))
    if not is_wait_tensor(wait_node):
        raise ValueError(
            f"Expected wait_tensor after all_gather node {last_all_gather.name}, "
            f"got {wait_node.name}"
        )

    wait_chain_user = _find_last_user_in_wait_chain(wait_node)
    return _find_last_non_view_node_in_chain(wait_chain_user)


def find_fsdp_unshard_save_node(param_placeholder: fx.Node) -> fx.Node | None:
    return find_fsdp_unshard_output(param_placeholder)


def find_fsdp_reduce_grad_input(param_grad_output: Any) -> fx.Node | None:
    """Return the split point before an FSDP reduce-grad epilogue.

    The backward FSDP/DDP/HSDP tail is traced as a unary chain ending in the
    synced grad output:

        local_grad -> cast/view* -> reduce_scatter -> wait -> sharded_grad
        local_grad -> cast/view* -> all_reduce -> wait -> replicated_grad
        local_grad -> cast/view* -> all_reduce -> wait -> reduce_scatter
          -> wait -> grad

    GraphPP splits at the input to the earliest grad-sync collective in that
    suffix. The cast remains in ``bw_no_fsdp`` so microbatch accumulation
    happens in FSDP's reduce dtype, and ``reduce_grad`` contains only the
    scheduled collective epilogue. Values that are not FX nodes, such as
    ``None`` parameter-grad slots, are not collective outputs and are preserved
    by the caller.

    TODO(sanketpurandare): requires upstream change: FSDP trace/passes should
    annotate reduce-grad collective regions for downstream graph extraction.
    """
    if not isinstance(param_grad_output, fx.Node):
        return None

    node = param_grad_output
    reduce_grad_input = None
    while isinstance(node, fx.Node) and len(node.all_input_nodes) == 1:
        input_node = node.all_input_nodes[0]
        if len(input_node.users) > 1:
            break
        previous_node = node
        node = input_node
        if is_reduce_grad_collective(previous_node):
            reduce_grad_input = node
    return reduce_grad_input
