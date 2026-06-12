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

import torch
from torch.fx.passes.regional_inductor import regional_inductor

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


def regional_inductor_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple, *, serializable: bool = False
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


def _get_fake_mode(gm: torch.fx.GraphModule):
    """Return the FakeTensorMode backing the graph's placeholder metadata."""
    from torch._subclasses.fake_tensor import FakeTensor

    for node in gm.graph.nodes:
        if node.op == "placeholder" and isinstance(node.meta.get("val"), FakeTensor):
            return node.meta["val"].fake_mode
    return None


def serialize_loss_chunks_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple | None = None
) -> torch.fx.GraphModule:
    """Force the ChunkedCELoss chunks to execute one at a time before full inductor.

    ``ChunkedCELossWithParamGrads`` splits the tokens into ``num_chunks`` so the
    huge ``[tokens, vocab]`` lm_head logits are materialized one chunk at a time.
    The chunks are data-independent, so when full inductor compiles the whole
    graph its scheduler batches all ``num_chunks`` lm_head matmuls together —
    every chunk's forward logits AND backward grad-logits (each
    ``[tokens/num_chunks, vocab]``) end up live simultaneously, reconstituting
    the full logits tensor and OOM-ing DSv3-16B at B=16 (root-caused in
    ``dsv3_scaling_experiment_results.md`` Run 11: ~16 × 1.68 GiB live at the
    loss backward).

    This pass restores the streaming by threading a numerically-zero dependency:
    each chunk's lm_head matmul input is made to depend on the previous chunk's
    forward loss (``nll_loss_forward``), which itself depends on that chunk's
    logits. Inductor can then no longer hoist a later chunk's matmul ahead of an
    earlier chunk's loss, so each chunk's logits are freed before the next
    chunk's are allocated.

    The injected term is ``hidden + nan_to_num(prev_loss) * 0`` (broadcast of a
    scalar zero): ``nan_to_num`` first maps any inf/nan to a finite value so the
    ``* 0`` is exactly ``0`` (a plain ``prev_loss * 0`` would give ``nan`` if a
    chunk loss ever overflowed -> ``inf * 0 = nan`` -> corrupted training). The
    result is numerically identity (``hidden + 0``) yet keeps the data read of
    ``prev_loss`` that orders the chunks. Operates on the lowered (plain-tensor)
    graph, so there are no DTensor placement concerns. Disable with
    ``--compile.disable_passes serialize_loss_chunks_pass``.
    """
    import operator

    aten = torch.ops.aten
    g = gm.graph
    order = {n: i for i, n in enumerate(g.nodes)}

    # Each chunk's forward partial loss is a distinct nll_loss_forward.
    nll_nodes = [n for n in g.nodes if n.target == aten.nll_loss_forward.default]
    if len(nll_nodes) < 2:
        return gm  # not chunked (or single chunk): nothing to serialize

    def find_chunk_matmul(nll: torch.fx.Node) -> torch.fx.Node | None:
        """Walk back from a chunk's nll to the lm_head matmul that produced its
        logits (nll <- log_softmax <- to_copy <- view... <- mm)."""
        seen: set[torch.fx.Node] = set()
        stack = list(nll.all_input_nodes)
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n.op == "call_function" and n.target == aten.mm.default:
                return n
            stack.extend(n.all_input_nodes)
        return None

    def loss_scalar(nll: torch.fx.Node) -> torch.fx.Node:
        """The scalar loss output of nll_loss_forward (getitem 0); fall back to
        the nll node itself."""
        for u in nll.users:
            if u.target is operator.getitem and u.args[1] == 0:
                return u
        return nll

    chunks = []
    for nll in nll_nodes:
        mm = find_chunk_matmul(nll)
        if mm is not None:
            chunks.append((mm, nll))
    if len(chunks) < 2:
        return gm
    chunks.sort(key=lambda mn: order[mn[0]])

    fake_mode = _get_fake_mode(gm)
    num_serialized = 0
    for i in range(1, len(chunks)):
        mm = chunks[i][0]
        anchor = loss_scalar(chunks[i - 1][1])
        hidden = mm.args[0]
        if not isinstance(hidden, torch.fx.Node):
            continue
        hidden_dtype = hidden.meta["val"].dtype
        with g.inserting_before(mm):
            finite = g.call_function(aten.nan_to_num.default, (anchor,))
            # Cast the anchor to the hidden's dtype BEFORE the broadcast add: the
            # loss anchor is fp32, and ``bf16_hidden + fp32_zero`` would promote
            # the matmul input to fp32 and change its numerics. With the cast the
            # added term is a same-dtype zero, so hidden_dep == hidden bitwise.
            finite_cast = g.call_function(
                aten._to_copy.default, (finite,), {"dtype": hidden_dtype}
            )
            zero = g.call_function(aten.mul.Tensor, (finite_cast, 0.0))
            hidden_dep = g.call_function(aten.add.Tensor, (hidden, zero))
        # Fresh fake-tensor metadata so downstream passes / inductor see valid
        # vals. The chain keeps the anchor's (scalar) shape; the broadcast add
        # keeps the hidden input's shape and dtype -> numerically identity.
        if fake_mode is not None:
            with fake_mode:
                av = anchor.meta["val"]
                hv = hidden.meta["val"]
                finite.meta["val"] = torch.nan_to_num(av)
                finite_cast.meta["val"] = finite.meta["val"].to(hidden_dtype)
                zero.meta["val"] = finite_cast.meta["val"] * 0.0
                hidden_dep.meta["val"] = hv + zero.meta["val"]
        mm.replace_input_with(hidden, hidden_dep)
        num_serialized += 1

    g.lint()
    gm.recompile()
    logger.info(
        f"serialize_loss_chunks_pass: serialized {num_serialized + 1} loss "
        f"chunks ({num_serialized} dependency edge(s))"
    )
    return gm


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
        result = regional_inductor_pass(gm, example_inputs)

    # Carry the pre-collapse cudagraph verdict forward via gm.meta. The
    # collapse is information-destroying; this is how downstream passes
    # know whether the artifact contains hidden cudagraph-incompatible ops.
    result.meta["cudagraph_compatible"] = pre_collapse_cudagraph_compatible
    return result
