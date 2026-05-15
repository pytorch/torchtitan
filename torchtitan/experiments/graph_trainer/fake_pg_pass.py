# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from typing import Any

import torch
import torch.fx as fx
from torch._inductor.fx_passes.bucketing import _resolve_group_name
from torch.fx.operator_schemas import normalize_function
from torch.utils._ordered_set import OrderedSet

from torchtitan.tools.logging import logger


_DEBUG_FAKE_PG_REGISTRY: dict[str, str] = {}
_DERIVED_PG_REGISTRY: dict[str, OrderedSet[str]] = defaultdict(OrderedSet)


def _register_derived_pg(source_pg_name: str, derived_pg_name: str) -> None:
    """Record that ``derived_pg_name`` semantically belongs to ``source_pg_name``."""
    _DERIVED_PG_REGISTRY[source_pg_name].add(derived_pg_name)


def _get_or_create_debug_fake_pg(source_pg_name: str) -> str:
    """Return a fake-backend PG with the same ranks as ``source_pg_name``."""
    import torch.distributed as dist

    if source_pg_name in _DEBUG_FAKE_PG_REGISTRY:
        return _DEBUG_FAKE_PG_REGISTRY[source_pg_name]

    source_pg = dist.distributed_c10d._resolve_process_group(source_pg_name)
    ranks = dist.get_process_group_ranks(source_pg)
    fake_pg = dist.new_group(
        ranks=ranks,
        backend="fake",
        group_desc="debug_fake_pg",
        use_local_synchronization=True,
    )
    _DEBUG_FAKE_PG_REGISTRY[source_pg_name] = fake_pg.group_name
    logger.info(
        "Created debug fake PG (source: %s, fake: %s, ranks=%d)",
        source_pg_name,
        fake_pg.group_name,
        len(ranks),
    )
    return fake_pg.group_name


def _get_fake_pg_mapping_for_mesh_axes(
    parallel_dims: Any,
    mesh_axes: Iterable[str],
) -> dict[str, str]:
    axes = list(mesh_axes)
    if not axes:
        return {}

    enabled_meshes = parallel_dims.get_all_one_dimensional_meshes()
    source_pg_names: OrderedSet[str] = OrderedSet()
    for axis in axes:
        if axis == "ep":
            raise ValueError(
                "--compile.fake_pg_mesh_axes does not support 'ep'. EP token "
                "exchange uses data-dependent token counts; replacing its PG "
                "with a fake backend can hang or error."
            )
        try:
            mesh = parallel_dims.get_optional_mesh(axis)
        except ValueError as exc:
            raise ValueError(
                f"--compile.fake_pg_mesh_axes got invalid mesh axis '{axis}'."
            ) from exc
        if mesh is None:
            logger.warning(
                "--compile.fake_pg_mesh_axes requested unavailable mesh axis "
                "'%s'; skipping it. Enabled axes are %s.",
                axis,
                sorted(enabled_meshes.keys()),
            )
            continue
        if mesh.ndim != 1:
            raise ValueError(
                f"--compile.fake_pg_mesh_axes only supports one-dimensional "
                f"mesh axes, got '{axis}' with ndim={mesh.ndim}."
            )
        source_pg_names.add(mesh.get_group().group_name)

    expanded_pg_names: OrderedSet[str] = OrderedSet()
    for source_pg_name in source_pg_names:
        expanded_pg_names.add(source_pg_name)
        for derived_pg_name in _DERIVED_PG_REGISTRY.get(source_pg_name, ()):
            expanded_pg_names.add(derived_pg_name)

    pg_mapping = {
        pg_name: _get_or_create_debug_fake_pg(pg_name) for pg_name in expanded_pg_names
    }
    logger.info("Fake PG mesh-axis mapping: axes=%s pg_mapping=%s", axes, pg_mapping)
    return pg_mapping


def _is_c10d_op(node: fx.Node) -> bool:
    if node.op != "call_function":
        return False
    namespace = getattr(node.target, "namespace", None)
    return namespace in {"_c10d_functional", "c10d_functional", "c10d"}


def _schema_arg_index(target: Any, arg_name: str) -> int | None:
    schema = getattr(target, "_schema", None)
    if schema is None:
        return None
    for i, arg in enumerate(schema.arguments):
        if arg.name == arg_name:
            return i
    return None


def _process_group_name_from_arg(value: Any) -> str | None:
    if isinstance(value, fx.Node):
        value = value.meta.get("val")
    if value is None:
        return None
    if isinstance(value, str):
        return _resolve_group_name(value)
    group_name = getattr(value, "group_name", None)
    if group_name is not None:
        return group_name
    return None


def _get_c10d_pg_arg(node: fx.Node) -> tuple[str, str] | None:
    for arg_name in ("group_name", "process_group"):
        idx = _schema_arg_index(node.target, arg_name)
        if idx is None:
            continue
        if arg_name in node.kwargs:
            pg_name = _process_group_name_from_arg(node.kwargs[arg_name])
        elif idx < len(node.args):
            pg_name = _process_group_name_from_arg(node.args[idx])
        else:
            normalized = normalize_function(
                node.target,
                args=node.args,
                kwargs=node.kwargs,
                normalize_to_only_use_kwargs=True,
            )
            if normalized is None:
                pg_name = None
            else:
                _, kwargs = normalized
                pg_name = _process_group_name_from_arg(kwargs.get(arg_name))
        if pg_name is not None:
            return arg_name, pg_name
    return None


def _replacement_pg_arg_value(arg_name: str, fake_pg_name: str) -> Any:
    if arg_name == "group_name":
        return fake_pg_name
    if arg_name == "process_group":
        import torch.distributed as dist

        return dist.distributed_c10d._resolve_process_group(fake_pg_name)
    raise AssertionError(f"Unexpected c10d PG argument name: {arg_name}")


def _schema_arg_value(node: fx.Node, arg_name: str) -> Any:
    idx = _schema_arg_index(node.target, arg_name)
    if idx is None:
        raise ValueError(f"{node.target} has no schema argument {arg_name}")
    if arg_name in node.kwargs:
        return node.kwargs[arg_name]
    if idx < len(node.args):
        return node.args[idx]
    return None


def _new_pg_get_attr(gm: fx.GraphModule, node: fx.Node, process_group: Any) -> fx.Node:
    base = "_debug_fake_process_group"
    idx = 0
    while hasattr(gm, f"{base}_{idx}"):
        idx += 1
    attr_name = f"{base}_{idx}"
    setattr(gm, attr_name, process_group)
    with gm.graph.inserting_before(node):
        pg_node = gm.graph.get_attr(attr_name)
    pg_node.meta["val"] = process_group
    return pg_node


def _set_c10d_pg_arg(
    gm: fx.GraphModule,
    node: fx.Node,
    arg_name: str,
    replacement_value: Any,
) -> None:
    idx = _schema_arg_index(node.target, arg_name)
    if idx is None:
        raise ValueError(f"{node.target} has no schema argument {arg_name}")
    if arg_name == "process_group":
        current_value = _schema_arg_value(node, arg_name)
        if isinstance(current_value, fx.Node) and current_value.op == "get_attr":
            setattr(gm, current_value.target, replacement_value)
            current_value.meta["val"] = replacement_value
            return
        replacement_value = _new_pg_get_attr(gm, node, replacement_value)
    if arg_name in node.kwargs:
        node.kwargs = {**node.kwargs, arg_name: replacement_value}
        return
    if idx < len(node.args):
        args = list(node.args)
        args[idx] = replacement_value
        node.args = tuple(args)
        return

    normalized = normalize_function(
        node.target,
        args=node.args,
        kwargs=node.kwargs,
        normalize_to_only_use_kwargs=True,
    )
    if normalized is None:
        raise ValueError(f"Could not normalize c10d node: {node.format_node()}")
    _, kwargs = normalized
    kwargs[arg_name] = replacement_value
    node.args = ()
    node.kwargs = kwargs


def fake_pg_mesh_axes_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    parallel_dims: Any | None = None,
    mesh_axes: Iterable[str] = (),
    pg_mapping: dict[str, str] | None = None,
) -> torch.fx.GraphModule:
    """Rewrite c10d collectives on selected PGs to fake-backend PGs."""
    del example_inputs
    if pg_mapping is None:
        if parallel_dims is None:
            raise ValueError("fake_pg_mesh_axes_pass requires parallel_dims")
        pg_mapping = _get_fake_pg_mapping_for_mesh_axes(parallel_dims, mesh_axes)
    if not pg_mapping:
        return gm

    rewrite_counts = {source: 0 for source in pg_mapping}
    target_counts: Counter[str] = Counter()
    for node in gm.graph.nodes:
        if not _is_c10d_op(node):
            continue
        pg_arg = _get_c10d_pg_arg(node)
        if pg_arg is None:
            continue
        arg_name, source_pg_name = pg_arg
        fake_pg_name = pg_mapping.get(source_pg_name)
        if fake_pg_name is None:
            continue
        _set_c10d_pg_arg(
            gm,
            node,
            arg_name,
            _replacement_pg_arg_value(arg_name, fake_pg_name),
        )
        rewrite_counts[source_pg_name] += 1
        target_counts[str(node.target)] += 1

    for source, fake in pg_mapping.items():
        logger.info(
            "Rewrote %d c10d node(s) from PG %s to fake PG %s",
            rewrite_counts[source],
            source,
            fake,
        )
    logger.info("Fake PG rewrite counts by target: %s", dict(target_counts))
    gm.graph.lint()
    gm.recompile()
    return gm
