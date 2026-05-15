# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inductor compilation passes for graph_trainer.

Regional and full Inductor compilation, plus FlexAttention annotation for
regional_inductor.
"""

from __future__ import annotations

from typing import Any

import sympy
import torch
from torch.fx.passes.regional_inductor import regional_inductor
from torch.utils._pytree import tree_map
from torch.utils._sympy.functions import FloorDiv

from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    _eval_hint,
    CHUNK_SYMBOL_HINTS_META,
    free_symbols,
)
from torchtitan.tools.logging import logger


def _ops_filter_with_distributed(name: str) -> bool:
    """Ops filter that allows distributed collective ops for serialization.

    The default GraphPickler ops filter only allows aten and fbgemm ops.
    SimpleFSDP uses _c10d_functional collectives that must also be
    allowed for the graph to serialize correctly.  The device_mesh ops
    (e.g. _get_submesh) appear in the backward graph when DTensor
    reconstructs submeshes from tracked ancestor meshes.
    """
    return name.startswith(
        (
            "torch.ops.aten",
            "torch.ops.fbgemm",
            "torch.ops._c10d_functional",
            "torch.ops._dtensor",
            "torch.ops.device_mesh",
            "torch.ops.bucketing",
        )
    )


def _node_metadata_key_filter_distributed(key: str) -> bool:
    """Metadata key filter for regional_inductor with distributed ops.

    Distributed ops (e.g. _get_submesh, mesh_get_process_group) produce
    opaque values (DeviceMesh, ProcessGroup) in node.meta["val"] and
    node.meta["eager_input_vals"] that cannot be pickled.  We strip
    both — they are not needed at runtime.
    """
    if key in ("val", "eager_input_vals"):
        return False
    return key not in ["source_fn_stack", "nn_module_stack", "fwd_source_fn_stack"]


def _localize_regional_chunk_input_symbols(gm: torch.fx.GraphModule) -> None:
    """Use region-local symbols for chunk extents visible only as submodule inputs."""
    selected_hints: dict[object, int] = {}
    for node in gm.graph.nodes:
        hints = node.meta.get(CHUNK_SYMBOL_HINTS_META)
        if isinstance(hints, dict):
            selected_hints.update(hints)
    if not selected_hints:
        return

    half_symbols: dict[tuple[int, sympy.Symbol], sympy.Symbol] = {}
    half_hints: dict[object, int] = {}

    def half_symbol(shape_env: Any, symbol: sympy.Symbol) -> sympy.Symbol:
        key = (id(shape_env), symbol)
        if key in half_symbols:
            return half_symbols[key]
        symint = shape_env.create_unbacked_symint()
        new_symbol = symint.node.expr
        if new_symbol in shape_env.pending_fresh_unbacked_symbols:
            shape_env.pending_fresh_unbacked_symbols.remove(new_symbol)
        hint = max(1, (selected_hints[symbol] + 1) // 2)
        shape_env.var_to_hint_override[new_symbol] = hint
        half_symbols[key] = new_symbol
        half_hints[new_symbol] = hint
        return new_symbol

    def rewrite_symbolic_value(value: object) -> object:
        if not isinstance(value, (torch.SymInt, torch.SymFloat, torch.SymBool)):
            return value
        symbols = free_symbols(value) & selected_hints.keys()
        if not symbols:
            return value
        shape_env = value.node.shape_env
        substitutions = {
            FloorDiv(symbol, 2): half_symbol(shape_env, symbol) for symbol in symbols
        }
        expr = value.node.expr.subs(substitutions)
        if expr == value.node.expr:
            return value
        hints = {**selected_hints, **half_hints}
        hint = _eval_hint(expr, hints)
        if isinstance(value, torch.SymInt):
            return shape_env.create_symintnode(expr, hint=hint)
        if isinstance(value, torch.SymFloat):
            return shape_env.create_symfloatnode(expr, hint=hint)
        return shape_env.create_symboolnode(expr)

    def rewrite_tensor_meta(value: object) -> object:
        if not isinstance(value, torch.Tensor):
            return rewrite_symbolic_value(value)
        changed = False
        shape = []
        for dim in value.shape:
            new_dim = rewrite_symbolic_value(dim)
            changed = changed or new_dim is not dim
            shape.append(new_dim)
        stride = []
        for dim in value.stride():
            new_dim = rewrite_symbolic_value(dim)
            changed = changed or new_dim is not dim
            stride.append(new_dim)
        if not changed:
            return value
        return value.new_empty_strided(tuple(shape), tuple(stride))

    for node in gm.graph.nodes:
        if node.meta.get("chunked_region_role") is None:
            continue
        for key in ("val", "example_value", "eager_input_vals"):
            if key in node.meta:
                node.meta[key] = tree_map(rewrite_tensor_meta, node.meta[key])


def regional_inductor_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple,
    *,
    serializable: bool = False,
    localize_chunk_input_symbols: bool = True,
) -> torch.fx.GraphModule:
    """Compile tagged graph regions with ``regional_inductor``.

    Scans the graph for nodes whose ``node.meta["custom"]`` contains a
    ``compile_with_inductor`` key and compiles those regions with
    TorchInductor.  Nodes without this tag are left unchanged.  If no
    nodes are tagged the pass is a no-op.

    Inductor is configured for bitwise-equal numerics so that the
    compiled regions match eager execution exactly.

    Args:
        gm: The graph module to compile.
        example_inputs: Example inputs for shape propagation.
        serializable: When True (precompile mode), sets
            ``force_autograd_cache`` so that ``regional_inductor`` wraps
            its output in ``RegionalOutputCode``, and overrides the ops
            filter to allow distributed collective ops.
    """
    import torch._inductor.config as ic
    from torch._subclasses.fake_tensor import FakeTensor

    def _get_fake_mode_from_gm(gm: torch.fx.GraphModule):
        """Extract the FakeTensorMode from a graph module's placeholder metadata."""
        for node in gm.graph.nodes:
            if node.op == "placeholder" and "val" in node.meta:
                val = node.meta["val"]
                if isinstance(val, FakeTensor):
                    return val.fake_mode
        return None

    # Ensure inductor produces bitwise-equal numerics vs eager.
    ic.eager_numerics.division_rounding = True
    # Recommended by inductor team — uncomment as needed:
    # ic.emulate_precision_casts = True
    # ic.eager_numerics.disable_ftz = True
    # ic.eager_numerics.use_pytorch_libdevice = True
    # ic.fallback_random = True

    # regional_inductor calls standalone_compile with
    # dynamic_shapes="from_tracing_context", which requires an active
    # TracingContext with a FakeTensorMode.  When this pass is called
    # outside torch.compile (e.g. after make_fx tracing in graph_trainer),
    # no TracingContext exists, so we create one from the graph's fake
    # tensor metadata.
    fake_mode = _get_fake_mode_from_gm(gm)
    tracing_ctx = torch._guards.TracingContext(fake_mode)

    if localize_chunk_input_symbols:
        _localize_regional_chunk_input_symbols(gm)

    if serializable:
        with (
            torch._guards.tracing(tracing_ctx),
            torch._functorch.config.patch("force_autograd_cache", True),
        ):
            result = regional_inductor(gm, example_inputs)
        from torch._inductor.output_code import RegionalOutputCode

        # Override the ops filter after compilation so that
        # serialization (which happens later) allows distributed
        # collective ops like _c10d_functional through GraphPickler.
        if isinstance(result, RegionalOutputCode):
            result._ops_filter = _ops_filter_with_distributed
            result._node_metadata_key_filter = _node_metadata_key_filter_distributed
        else:
            logger.warning(
                "regional_inductor with serializable=True did not produce "
                "RegionalOutputCode; distributed ops may not serialize correctly."
            )
        return result

    with torch._guards.tracing(tracing_ctx):
        gm = regional_inductor(gm, example_inputs)

    # regional_inductor may switch to boxed calling convention; reset to
    # default so the graph can be called with positional args as usual.
    gm.graph.set_codegen(torch.fx.graph.CodeGen())
    gm.recompile()
    return gm


def annotate_flex_attention_for_regional_inductor_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    flex_compile_config: dict | None,
    mask_compile_config: dict | None = None,
) -> torch.fx.GraphModule:
    """Tag flex attention HOPs with compile_with_inductor for regional_inductor.

    Annotates three sets of nodes so that regional_inductor correctly
    scoops and compiles flex attention regions:
    1. The HOP node itself (flex_attention / flex_attention_backward)
    2. The get_attr nodes referencing score_mod / mask_mod submodules.
    3. All nodes inside those submodule graphs.

    Args:
        gm: The graph module to annotate.
        example_inputs: Example inputs (unused, required by pass interface).
        flex_compile_config: Inductor config dict for flex attention HOP
            nodes and their get_attr submodule references. When provided,
            wrapped as ``{"inductor_configs": flex_compile_config}``.
            When None, nodes are tagged with an empty annotation.
        mask_compile_config: Inductor config dict for nodes inside mask_mod
            subgraphs. When provided, wrapped as
            ``{"inductor_configs": mask_compile_config}``.
            When None, nodes are tagged with an empty annotation.
    """
    flex_compile_annotation: dict = (
        {"inductor_configs": flex_compile_config}
        if flex_compile_config is not None
        else {}
    )
    mask_compile_annotation: dict = (
        {"inductor_configs": mask_compile_config}
        if mask_compile_config is not None
        else {}
    )

    for node in gm.graph.nodes:
        if node.target not in {
            torch.ops.higher_order.flex_attention,
            torch.ops.higher_order.flex_attention_backward,
        }:
            continue
        node.meta.setdefault("custom", {})[
            "compile_with_inductor"
        ] = flex_compile_annotation
        for inp in node.all_input_nodes:
            if inp.op != "get_attr":
                continue
            submod = getattr(gm, inp.target, None)
            if not isinstance(submod, torch.fx.GraphModule):
                continue
            inp.meta.setdefault("custom", {})[
                "compile_with_inductor"
            ] = flex_compile_annotation

            # Following are the nodes in mask_mod subgraph
            for sub_node in submod.graph.nodes:
                sub_node.meta.setdefault("custom", {})[
                    "compile_with_inductor"
                ] = mask_compile_annotation
    return gm


def _migrate_cpu_get_attrs_to_cuda(gm: torch.fx.GraphModule) -> None:
    """Move CPU constant tensor get_attrs to CUDA so cudagraph capture works."""
    from torch.fx.graph_module import _assign_attr, _get_attr

    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.find_nodes(op="get_attr"):
            attr = _get_attr(module, node.target)
            if isinstance(attr, torch.Tensor) and attr.device.type == "cpu":
                _assign_attr(attr.cuda(), module, node.target)


def full_inductor_compilation_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple
) -> torch.fx.GraphModule:
    """Apply full Inductor compilation by tagging every node and delegating
    to :func:`regional_inductor_pass`.

    Marks every non-placeholder/output node with the ``compile_with_inductor``
    custom metadata key so ``regional_inductor`` scoops the entire graph as
    one compiled region. This reuses the regional path (which goes through
    ``standalone_compile`` and gets c10d functionalization, PG unboxing,
    decompositions, and caching for free) instead of duplicating that prep
    around a direct ``compile_fx_inner`` call.

    The collapse hides cudagraph-incompatible ops (unpinned D2H copies,
    sm<10 ``_grouped_mm``) inside the opaque ``standalone_compile_inner``
    node, so the later :func:`is_cudagraph_compatible` scan can't see
    them. Snapshot the verdict on the pre-collapse gm and stash it on
    the result so the downstream scan can honor it.

    Must be the **terminal** pass — no FX-graph-level passes (e.g.
    ``custom_codegen_pass``, ``insert_kernel_annotations_pass``) can
    run after this because the FX graph is no longer authoritative.
    """
    import torch._inductor.config as ic

    from torchtitan.experiments.graph_trainer.cudagraph import is_cudagraph_compatible

    pre_collapse_cudagraph_compatible = is_cudagraph_compatible(
        gm, skip_flex_attention_check=True
    )
    _migrate_cpu_get_attrs_to_cuda(gm)
    for module in gm.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        for node in module.graph.nodes:
            if node.op in ("placeholder", "output"):
                continue
            node.meta.setdefault("custom", {}).setdefault(
                "compile_with_inductor", {"inductor_configs": {}}
            )
    # AOT autograd (via ``standalone_compile``) reorders the gm and breaks
    # fwd/bwd interleaving, blowing up the baseline schedule. Re-enable
    # Inductor's reorder pass (disabled globally in ``compile.py``) to fix.
    with ic.patch(reorder_for_peak_memory=True):
        result = regional_inductor_pass(
            gm, example_inputs, localize_chunk_input_symbols=False
        )

    # Carry the pre-collapse cudagraph verdict forward via gm.meta. The
    # collapse is information-destroying; this is how downstream passes
    # know whether the artifact contains hidden cudagraph-incompatible ops.
    result.meta["cudagraph_compatible"] = pre_collapse_cudagraph_compatible
    return result
