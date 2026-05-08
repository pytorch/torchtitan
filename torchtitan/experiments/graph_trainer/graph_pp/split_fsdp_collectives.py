# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
from typing import Any

import torch
import torch.fx.node
import torch.utils._pytree as pytree
from torch._functorch._aot_autograd.descriptors import AOTOutput
from torch._functorch.partitioners import _extract_graph_with_inputs_outputs

from .graph_pp_utils import (
    find_last_all_gather_in_chain,
    find_last_non_view_node_in_chain,
    find_last_user_in_wait_chain,
    is_reduce_scatter_tensor,
    is_wait_tensor,
)


@contextmanager
def exclude_from_fx_side_effectful(exclude_vals: set[Any]):
    original_val = torch.fx.node._side_effectful_functions.copy()
    try:
        torch.fx.node._side_effectful_functions -= exclude_vals
        yield
    finally:
        torch.fx.node._side_effectful_functions.clear()
        torch.fx.node._side_effectful_functions.update(original_val)


exclude_wait_from_fx_side_effectful = partial(
    exclude_from_fx_side_effectful,
    {
        torch.ops._c10d_functional.wait_tensor,
        torch.ops._c10d_functional.wait_tensor.default,
    },
)


@dataclasses.dataclass(frozen=True)
class PrefetchOutput(AOTOutput):
    pass


@dataclasses.dataclass(frozen=True)
class EpilogueInput(AOTOutput):
    pass


def split_fsdp_prefetch(
    gm: torch.fx.GraphModule,
    num_params: int,
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
    g = deepcopy(gm.graph)
    all_g_ins = g.find_nodes(op="placeholder")
    param_g_ins = all_g_ins[:num_params]
    rem_g_ins = all_g_ins[num_params:]

    prefetch_g_outs_map = []

    for param_g_in in param_g_ins:
        # 1. Find last all_gather from each placeholder
        last_ag_node = find_last_all_gather_in_chain(param_g_in)
        if last_ag_node is None:
            prefetch_g_outs_map.append(param_g_in)
        else:
            # 2. Find last wait_tensor from last all_gather
            last_ag_wait_node = next(iter(last_ag_node.users))
            assert is_wait_tensor(last_ag_wait_node)

            # 3. Continue the linear chain from the last wait_tensor
            last_wait_chain_user = find_last_user_in_wait_chain(last_ag_wait_node)

            # 4. Get the last non-view node in the wait chain
            last_non_view_wait_chain_user = find_last_non_view_node_in_chain(
                last_wait_chain_user
            )

            prefetch_g_outs_map.append(last_non_view_wait_chain_user)

    prefetch_g_outs = prefetch_g_outs_map
    prefetch_g_outs_descs: list[AOTOutput] = [
        PrefetchOutput() for _ in range(len(prefetch_g_outs))
    ]
    g_outs = pytree.arg_tree_leaves(*(n.args for n in g.find_nodes(op="output")))
    g_outs_descs = pytree.arg_tree_leaves(
        next(iter(g.find_nodes(op="output"))).meta.get("desc", [None] * len(g_outs))
    )
    with exclude_wait_from_fx_side_effectful():
        prefetch_g = _extract_graph_with_inputs_outputs(
            g,
            param_g_ins,
            prefetch_g_outs,
            prefetch_g_outs_descs,
            ignore_must_be_in_fw_bw=True,
        )

        main_g = _extract_graph_with_inputs_outputs(
            g,
            prefetch_g_outs + rem_g_ins,
            g_outs,
            g_outs_descs,
            ignore_must_be_in_fw_bw=True,
        )
    prefetch_gm = torch.fx._lazy_graph_module._make_graph_module(gm, prefetch_g)
    main_gm = torch.fx._lazy_graph_module._make_graph_module(gm, main_g)
    return prefetch_gm, main_gm


def split_fsdp_reduce_scatters_epilogue(
    gm: torch.fx.GraphModule,
    num_grads: int,
) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
    g = deepcopy(gm.graph)
    g_ins = g.find_nodes(op="placeholder")
    g_outs = pytree.arg_tree_leaves(*(n.args for n in g.find_nodes(op="output")))
    grad_outs = g_outs[:num_grads]
    rem_g_outs = g_outs[num_grads:]
    out_descs = pytree.arg_tree_leaves(
        next(iter(g.find_nodes(op="output"))).meta.get("desc", [None] * len(grad_outs))
    )
    grad_outs_descs = out_descs[:num_grads]
    rem_g_outs_descs = out_descs[num_grads:]

    grad_outs_map = []
    for grad_out in grad_outs:
        n = grad_out
        earliest_rs = None
        while n is not None:
            if len(n.all_input_nodes) != 1:
                break
            n_in = n.all_input_nodes[0]
            if len(n_in.users) > 1:
                break
            prev_n = n
            n = n_in
            # Maybe we also need to track all_reduce?
            if is_reduce_scatter_tensor(prev_n):
                # In AP for mesh dim > 1
                # The reduction of gradients happen in multiple steps
                earliest_rs = n
        if earliest_rs is not None:
            grad_outs_map.append(earliest_rs)
        else:
            grad_outs_map.append(grad_out)

    epi_g_ins = grad_outs_map
    epi_g_ins_descs: list[AOTOutput] = [EpilogueInput() for _ in range(len(epi_g_ins))]

    with exclude_wait_from_fx_side_effectful():
        main_g = _extract_graph_with_inputs_outputs(
            g,
            g_ins,
            epi_g_ins + rem_g_outs,
            epi_g_ins_descs + rem_g_outs_descs,
            ignore_must_be_in_fw_bw=True,
        )
        epi_g = _extract_graph_with_inputs_outputs(
            g,
            epi_g_ins,
            grad_outs,
            grad_outs_descs,
            ignore_must_be_in_fw_bw=True,
        )
    epi_gm = torch.fx._lazy_graph_module._make_graph_module(gm, epi_g)
    main_gm = torch.fx._lazy_graph_module._make_graph_module(gm, main_g)
    return main_gm, epi_gm
