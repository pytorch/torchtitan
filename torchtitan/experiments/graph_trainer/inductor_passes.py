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
from torch._inductor.compile_fx import compile_fx_inner
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


def full_inductor_compilation_pass(
    gm: torch.fx.GraphModule, example_inputs: tuple
) -> torch.fx.GraphModule:
    """Apply full Inductor compilation with code generation.

    Applies inductor decompositions (e.g. ``aten.t`` → ``aten.permute``),
    then compiles the graph into optimized Triton/C++ kernels via
    ``compile_fx_inner`` and replaces the GraphModule's ``forward``
    with the compiled callable.

    Must be the **terminal** pass — no FX-graph-level passes (e.g.
    ``custom_codegen_pass``, ``insert_kernel_annotations_pass``) can
    run after this because the FX graph is no longer authoritative.
    """

    def _apply_decompositions(
        gm: torch.fx.GraphModule, example_inputs: tuple
    ) -> torch.fx.GraphModule:
        """Retrace with ``select_decomp_table()`` so that ops like ``aten.t``
        are decomposed before ``compile_fx_inner``."""
        from torch._inductor.decomposition import select_decomp_table
        from torch._subclasses.fake_tensor import FakeTensor
        from torch.fx.experimental.proxy_tensor import make_fx

        decomp_table = select_decomp_table()

        fake_mode = None
        for inp in example_inputs:
            if isinstance(inp, FakeTensor):
                fake_mode = inp.fake_mode
                break

        if fake_mode is not None:
            with fake_mode:
                gm = make_fx(
                    gm,
                    decomposition_table=decomp_table,
                    _allow_non_fake_inputs=True,
                )(*example_inputs)

        return gm

    gm = _apply_decompositions(gm, example_inputs)
    output_code = compile_fx_inner(gm, example_inputs)

    # compile_fx_inner returns OutputCode with boxed calling convention
    # (single list arg). Adapt to positional args so the graph trainer's
    # execution path (gm(*flat_inputs)) works unchanged.
    def _compiled_forward(*args):
        return output_code(list(args))

    gm.forward = _compiled_forward
    return gm
