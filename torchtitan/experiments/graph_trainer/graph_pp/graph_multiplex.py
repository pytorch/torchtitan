# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import hashlib
from collections.abc import Mapping
from itertools import dropwhile
from typing import Any

import sympy
import torch
import torch.fx as fx
from torch._dynamo.source import ConstantSource
from torch._logging import trace_structured
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx.graph_module import _assign_attr, _get_attr


def _find_fake_mode(gm: fx.GraphModule) -> FakeTensorMode | None:
    for node in gm.graph.nodes:
        for value in node.meta.values():
            for leaf in _iter_meta_leaves(value):
                if isinstance(leaf, FakeTensor):
                    return leaf.fake_mode
    return None


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


class _MetaShapeEnvTransfer:
    """Move copied FX metadata into the multiplex destination ShapeEnv.

    Contract:
      The multiplex graph is one FX graph, so every copied metadata object must
      belong to the destination graph's FakeTensorMode/ShapeEnv before full
      Inductor sees it.  The caller keeps the forward graph as that destination
      so the forward ShapeEnv retains the dynamic collective-size provenance.

      FakeTensor metadata is copied through FakeTensorMode.from_tensor, which
      delegates tensor size/stride/storage-offset transfer to PyTorch's
      ShapeEnv.transfer_symbols_from_foreign_shape_env path. GraphPP only adds
      the FX-metadata glue that PyTorch does not provide:

      1. Seed foreign unbacked base symbols before any derived expressions.
      2. Recursively copy raw SymInt/Expr metadata leaves from node.meta.

      Seeding base symbols first preserves derived expressions such as
      ``u0 + u1``.  Calling ShapeEnv's raw expression transfer on that derived
      expression before its bases are known would collapse it into one opaque
      unbacked symbol.

      NOTE: FakeTensor/ShapeEnv do not expose a public full-fidelity metadata
      transfer API for arbitrary FX node.meta trees, so GraphPP uses private
      ShapeEnv transfer hooks only for raw non-tensor symbolic metadata.
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
        # Seed bases first; FakeTensorMode.from_tensor handles the actual tensor
        # size/stride/storage-offset transfer through PyTorch ShapeEnv APIs.
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
                # Delegates to ShapeEnv.transfer_symbols_from_foreign_shape_env
                # for tensor shape metadata. The collect/seed phase above
                # preserves base-symbol identity before derived dims are copied.
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


def _copy_prefixed_get_attrs(
    dst: fx.GraphModule,
    src: fx.GraphModule,
    *,
    prefix: str,
) -> dict[str, str]:
    remap: dict[str, str] = {}
    for node in src.graph.find_nodes(op="get_attr"):
        attr_name = str(node.target)
        if attr_name in remap:
            continue
        base_name = f"{prefix}{attr_name.replace('.', '_')}"
        new_attr_name = base_name
        suffix = 0
        while hasattr(dst, new_attr_name):
            suffix += 1
            new_attr_name = f"{base_name}_{suffix}"
        _assign_attr(copy.deepcopy(_get_attr(src, attr_name)), dst, new_attr_name)
        remap[attr_name] = new_attr_name
    return remap


def multiplex_fw_bw_graph(
    fw_gm: fx.GraphModule,
    bw_gm: fx.GraphModule,
) -> fx.GraphModule:
    """Concatenate backward and forward graphs into one boxed GraphPP callable.

    Contract:
      ``OVERLAP_F_B`` schedule actions need one callable that performs a
      backward action for one stage and a forward action for another stage. The
      returned graph preserves the two flat ABIs by ordering placeholders as
      ``bw_inputs + fw_inputs`` and outputs as ``bw_outputs + fw_outputs``.

    Pseudocode:
      deep-copy the forward graph as the destination
      transfer backward metadata into the forward FakeTensorMode/ShapeEnv
      insert copied backward placeholders before the forward placeholders
      copy backward get_attr targets with a prefix to avoid attr collisions
      copy backward compute nodes before the forward compute nodes
      replace the output tuple with backward outputs followed by forward outputs

    The forward graph remains the destination module because its ShapeEnv owns
    the dynamic collective-size constraints needed by full Inductor for MoE
    all-to-all outputs.  The backward graph is inserted in topological order
    before the existing forward compute, with disjoint placeholders and
    prefixed attributes, so this helper does not need dependency analysis or a
    scheduling policy.  EP-overlap annotations are intentionally not applied
    inside this graph; pass ordering keeps EP-overlap on the standalone no-FSDP
    forward/backward graphs.
    """
    old_to_new: dict[fx.Node, fx.Node] = {}
    # Preserve the forward ShapeEnv exactly as traced.  Reconstructing it from
    # copied metadata can lose collective-size hints that full Inductor needs.
    multiplexed_gm = copy.deepcopy(fw_gm)
    dst_fake_mode = _find_fake_mode(multiplexed_gm)
    meta_transfer = _MetaShapeEnvTransfer(dst_fake_mode)
    for node in bw_gm.graph.nodes:
        meta_transfer.collect(node.meta)
    meta_transfer.seed()
    bw_get_attr_remap = _copy_prefixed_get_attrs(
        multiplexed_gm,
        bw_gm,
        prefix="bw",
    )

    fw_placeholders = multiplexed_gm.graph.find_nodes(op="placeholder")
    if not fw_placeholders:
        raise ValueError("GraphPP forward graph has no placeholders to multiplex")
    first_fw_placeholder = fw_placeholders[0]
    insert_point: fx.Node | None = None
    for node in bw_gm.graph.find_nodes(op="placeholder"):
        if insert_point is None:
            with multiplexed_gm.graph.inserting_before(first_fw_placeholder):
                new_placeholder = multiplexed_gm.graph.placeholder(f"bw_{node.name}")
        else:
            with multiplexed_gm.graph.inserting_after(insert_point):
                new_placeholder = multiplexed_gm.graph.placeholder(f"bw_{node.name}")
        new_placeholder.meta = meta_transfer.copy_meta(node.meta)
        old_to_new[node] = new_placeholder
        insert_point = new_placeholder

    first_fw_compute = next(
        (node for node in multiplexed_gm.graph.nodes if node.op != "placeholder"),
        None,
    )
    if first_fw_compute is None:
        raise ValueError("GraphPP forward graph has no output node to multiplex")

    bw_nodes = iter(bw_gm.graph.nodes)
    bw_nodes = dropwhile(lambda node: node.op == "placeholder", bw_nodes)
    insert_point = None
    for node in bw_nodes:
        if node.op == "output":
            break
        if insert_point is None:
            with multiplexed_gm.graph.inserting_before(first_fw_compute):
                new_node = multiplexed_gm.graph.node_copy(
                    node, lambda arg: old_to_new[arg]
                )
        else:
            with multiplexed_gm.graph.inserting_after(insert_point):
                new_node = multiplexed_gm.graph.node_copy(
                    node, lambda arg: old_to_new[arg]
                )
        new_node.meta = meta_transfer.copy_meta(node.meta)
        if new_node.op == "get_attr" and new_node.target in bw_get_attr_remap:
            new_node.target = bw_get_attr_remap[str(new_node.target)]
        old_to_new[node] = new_node
        insert_point = new_node

    multiplexed_output = multiplexed_gm.graph.find_nodes(op="output")[0]
    bw_output_node = bw_gm.graph.find_nodes(op="output")[0]
    bw_outputs = [
        old_to_new[value] if isinstance(value, fx.Node) else value
        for value in bw_output_node.args[0]
    ]
    fw_outputs = list(multiplexed_output.args[0])
    multiplexed_output.args = (tuple(bw_outputs + fw_outputs),)

    multiplexed_gm.graph.eliminate_dead_code()
    multiplexed_gm.graph.lint()
    multiplexed_gm.recompile()
    trace_structured(
        "artifact",
        metadata_fn=lambda: {
            "name": "graph_pp_multiplexed_graph",
            "encoding": "string",
        },
        payload_fn=lambda: multiplexed_gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
        ),
    )
    return multiplexed_gm
