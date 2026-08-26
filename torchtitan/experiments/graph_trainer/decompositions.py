# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone decomposition passes for GraphTrainer FX graphs."""

from __future__ import annotations

import torch
import torch.fx as fx
import torch.fx.traceback as fx_traceback
from torch._subclasses.fake_tensor import fake_tensor_tls
from torch.fx.experimental.proxy_tensor import decompose, make_fx


def _inputs_from_meta(gm: fx.GraphModule):
    inputs = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        if "val" not in node.meta:
            return None
        inputs.append(node.meta["val"])
    return tuple(inputs)


def _fetch_attr(root, target):
    for atom in target.split("."):
        root = getattr(root, atom)
    return root


def _assign_attr(root, target, value) -> None:
    atoms = target.split(".")
    for atom in atoms[:-1]:
        if not hasattr(root, atom):
            setattr(root, atom, torch.nn.Module())
        root = getattr(root, atom)
    setattr(root, atoms[-1], value)


def _copy_getattrs(dst: fx.GraphModule, src: fx.GraphModule) -> None:
    for node in src.graph.nodes:
        if node.op == "get_attr":
            _assign_attr(dst, node.target, _fetch_attr(src, node.target))


def _apply_decompositions(
    gm: fx.GraphModule,
    example_inputs,
    decomposition_table,
) -> None:
    inputs = (
        tuple(example_inputs) if example_inputs is not None else _inputs_from_meta(gm)
    )
    if inputs is None:
        return

    class DecomposeInterpreter(fx.Interpreter):
        def run_node(self, node):
            with decompose(decomposition_table):
                return super().run_node(node)

    def run(*args):
        return DecomposeInterpreter(gm).run(*args)

    old_allow_non_fake = fake_tensor_tls.allow_non_fake_inputs_override
    fake_tensor_tls.allow_non_fake_inputs_override = True
    try:
        with fx_traceback.preserve_node_meta():
            decomposed = make_fx(
                run,
                decomposition_table={},
                _allow_non_fake_inputs=True,
            )(*inputs)
    finally:
        fake_tensor_tls.allow_non_fake_inputs_override = old_allow_non_fake

    _copy_getattrs(gm, decomposed)
    gm.graph = decomposed.graph
    gm.graph.lint()
    gm.recompile()


def apply_decompositions_pass(
    gm: fx.GraphModule,
    example_inputs=None,
    *,
    decomposition_table,
    recurse: bool = False,
    apply_to_root: bool = True,
) -> fx.GraphModule:
    """Apply a decomposition table to the root graph and/or nested subgraphs.

    Args:
        recurse: If ``True``, apply the pass to all nested FX ``GraphModule``
            submodules. The root graph is controlled separately by
            ``apply_to_root``.
    """
    if callable(decomposition_table):
        decomposition_table = decomposition_table()
    if not decomposition_table:
        return gm

    modules = []
    if apply_to_root:
        modules.append(gm)
    if recurse:
        modules.extend(
            module
            for name, module in gm.named_modules()
            if name and isinstance(module, fx.GraphModule)
        )

    for module in modules:
        _apply_decompositions(
            module,
            example_inputs if module is gm else None,
            decomposition_table,
        )
    return gm
