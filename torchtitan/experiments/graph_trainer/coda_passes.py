# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CODA-style FlexGEMM epilogue fusion passes for joint training graphs.

The pass runs after forward/backward tracing, so it can optimize forward and
backward boundaries without relying on FlexGEMM autograd support. It recognizes
the pattern families documented by the CODA investigation and rewrites the
GEMM-rooted portions as ``torch.ops.higher_order.flex_gemm`` calls. Match
eligibility depends only on graph structure and tensor metadata, never module
names or source locations.
"""

from __future__ import annotations

import functools
import operator
from collections import Counter, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from math import gcd, prod
from typing import Any

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch._higher_order_ops.flex_gemm import (
    flex_gemm_hop,
    mark_flex_gemm_body_gemm_node,
)
from torch._inductor.pattern_matcher import CallFunction, KeywordArg, Match, MULTIPLE
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.utils.fuser_utils import fuse_as_graphmodule, validate_partition

from torchtitan.experiments.graph_trainer.coda_registry import (
    _CODA_PATTERNS,
    _quack_config,
    CodaKernel,
    CodaPattern,
    register_coda_pattern,
)

from torchtitan.experiments.graph_trainer.compile_time_benchmark import (
    apply_benchmarked_rewrites,
    BenchmarkCandidateSelection,
    BenchmarkGraphProcessorFn,
    make_rewrite_benchmark_region,
    RewriteBenchmarkRegion,
)
from torchtitan.tools.logging import logger


aten = torch.ops.aten

# Pattern catalog
#
# Each function name is its canonical pattern ID. The decorator derives that
# name and keeps the search and kernel policy beside the docstring that explains
# the match and rewrite.
@register_coda_pattern(
    priority=10,
    search=CallFunction(
        aten._fused_rms_norm.default,
        KeywordArg("norm_input"),
        KeywordArg("normalized_shape"),
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=MULTIPLE,
    ),
    kernels={
        "projection": CodaKernel(
            best_configs={10: _quack_config(256, 192, dynamic=False)}
        ),
        "expansion": CodaKernel(
            best_configs={10: _quack_config(256, 256, dynamic=True)}
        ),
    },
)
def F_mla_qproj_rmsnorm_expand(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a Q projection, RMSNorm, and the following expansion projection.

    Match:   Q projection -> RMSNorm -> expansion projection
    Ops:     ``aten.mm.default -> [_RESHAPE_TARGETS] ->
             aten._fused_rms_norm.default -> aten.mm.default``.
    Rewrite: The first ``flex_gemm`` emits weighted values and partial mean
             squares; the second consumes the resulting rstd in its epilogue.
    """

    def projection_body(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        weight_row: torch.Tensor,
        *,
        output_dtype: torch.dtype,
        root_shape: Sequence[Any],
        width: int,
        reduction_group: int,
    ) -> tuple[torch.Tensor, ...]:
        full = torch.mm(lhs, rhs)
        full_fp32 = full.to(torch.float32)
        weighted = (full_fp32 * weight_row).to(output_dtype)
        grouped = full_fp32.view(
            root_shape[0],
            width // reduction_group,
            reduction_group,
        )
        return full, weighted, (grouped * grouped).mean(dim=-1)

    def expansion_body(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        rstd: torch.Tensor,
        *,
        rows: Any,
        output_shape: Sequence[Any],
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor]:
        expanded = torch.mm(lhs, rhs).view(rows, output_shape[-1])
        output = (expanded.to(torch.float32) * rstd).to(output_dtype)
        return (output.view(output_shape),)

    def replacement(
        projection_lhs: torch.Tensor,
        projection_rhs: torch.Tensor,
        weight_row: torch.Tensor,
        expansion_rhs: torch.Tensor,
        *,
        projection_body: GraphModule,
        expansion_body: GraphModule,
        projection_kernel_options: dict[str, Any],
        expansion_kernel_options: dict[str, Any],
        full_shape: Sequence[Any],
        norm_shape: Sequence[Any],
        expansion_lhs_shape: Sequence[Any],
        expansion_output_shape: Sequence[Any],
        rstd_shape: Sequence[Any],
        eps: Any,
        norm_output_dtype: torch.dtype,
        expansion_output_dtype: torch.dtype,
        reduction_group: int,
    ) -> tuple[torch.Tensor, ...]:
        full_2d, weighted_2d, partial = flex_gemm_hop(
            aten.mm.default,
            projection_body,
            (projection_lhs, projection_rhs, weight_row),
            {},
            projection_kernel_options,
        )
        full = full_2d.view(full_shape)
        weighted = weighted_2d.view(full_shape)

        norm_width = norm_shape[-1]
        if full_shape[-1] != norm_width:
            norm_input = full[..., :norm_width]
            weighted_norm = weighted[..., :norm_width]
            partial = partial[..., : norm_width // reduction_group]
        else:
            norm_input = full
            weighted_norm = weighted

        physical_rstd = torch.rsqrt(partial.mean(dim=-1, keepdim=True) + eps)
        rstd = physical_rstd.view(rstd_shape)
        norm_weight = weight_row[..., :norm_width]
        norm_output = (norm_input.float() * rstd * norm_weight).to(norm_output_dtype)

        expansion_lhs = weighted_norm.view(expansion_lhs_shape)
        (expansion_output,) = flex_gemm_hop(
            aten.mm.default,
            expansion_body,
            (expansion_lhs, expansion_rhs, physical_rstd),
            {},
            expansion_kernel_options,
        )
        return full, norm_output, rstd, expansion_output

    rewrite = functools.partial(
        _rewrite_projection_rmsnorm,
        projection_body=projection_body,
        expansion_body=expansion_body,
        replacement=replacement,
    )
    pattern = F_mla_qproj_rmsnorm_expand.__name__
    for candidate in candidates:
        norm = candidate.output_node()
        if (
            _is_backward(norm)
            or not _valid_forward_rmsnorm(norm)
            or norm.meta.get("coda_consumed")
        ):
            continue
        norm_input = norm.args[0]
        if not isinstance(norm_input, Node):
            continue
        path = _chain_to_mm(norm_input, allow_cast=False)
        if path is None or not _is_reshape_only_path(path):
            continue
        norm_out = _rmsnorm_getitems(norm)
        if norm_out is None:
            continue
        second_use = _downstream_mm(norm_out[0], backward=False)
        if (
            second_use is None
            or second_use.operand != 0
            or _is_backward(second_use.node)
            or not _valid_projection_rmsnorm_use(norm, path, second_use, norm_input)
            or not _coda_nodes_available((*path.nodes, norm, second_use.node))
        ):
            continue
        if not _claim_coda_match(pattern, norm, selection):
            continue
        rewrite(
            gm,
            norm=norm,
            first_path=path,
            second_use=second_use,
            full_output=norm_input,
            pattern=pattern,
            autotune=autotune,
            benchmark_regions=benchmark_regions,
        )
        counts[pattern] += 1


@register_coda_pattern(
    priority=20,
    search=CallFunction(
        aten._fused_rms_norm.default,
        KeywordArg("norm_input"),
        KeywordArg("normalized_shape"),
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=MULTIPLE,
    ),
    kernels={
        "projection": CodaKernel(
            best_configs={10: _quack_config(128, 192, dynamic=False, cluster_m=1)}
        ),
        "expansion": CodaKernel(
            best_configs={10: _quack_config(128, 256, dynamic=True)}
        ),
    },
)
def F_mla_kvproj_rmsnorm_expand(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a segmented KV projection, RMSNorm, and expansion projection.

    Match:   KV projection -> split[0] -> RMSNorm -> expansion projection
    Ops:     ``aten.mm.default -> [_RESHAPE_TARGETS] ->
             aten.split_with_sizes.default -> operator.getitem(..., 0) ->
             aten._fused_rms_norm.default -> aten.mm.default``.
    Rewrite: The first ``flex_gemm`` emits weighted values and partial mean
             squares; the second consumes the resulting rstd in its epilogue.
    """

    def projection_body(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        weight_row: torch.Tensor,
        *,
        output_dtype: torch.dtype,
        root_shape: Sequence[Any],
        width: int,
        reduction_group: int,
    ) -> tuple[torch.Tensor, ...]:
        full = torch.mm(lhs, rhs)
        full_fp32 = full.to(torch.float32)
        weighted = (full_fp32 * weight_row).to(output_dtype)
        grouped = full_fp32.view(
            root_shape[0],
            width // reduction_group,
            reduction_group,
        )
        return full, weighted, (grouped * grouped).mean(dim=-1)

    def expansion_body(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        rstd: torch.Tensor,
        *,
        rows: Any,
        output_shape: Sequence[Any],
        output_dtype: torch.dtype,
    ) -> tuple[torch.Tensor]:
        expanded = torch.mm(lhs, rhs).view(rows, output_shape[-1])
        output = (expanded.to(torch.float32) * rstd).to(output_dtype)
        return (output.view(output_shape),)

    def replacement(
        projection_lhs: torch.Tensor,
        projection_rhs: torch.Tensor,
        weight_row: torch.Tensor,
        expansion_rhs: torch.Tensor,
        *,
        projection_body: GraphModule,
        expansion_body: GraphModule,
        projection_kernel_options: dict[str, Any],
        expansion_kernel_options: dict[str, Any],
        full_shape: Sequence[Any],
        norm_shape: Sequence[Any],
        expansion_lhs_shape: Sequence[Any],
        expansion_output_shape: Sequence[Any],
        rstd_shape: Sequence[Any],
        eps: Any,
        norm_output_dtype: torch.dtype,
        expansion_output_dtype: torch.dtype,
        reduction_group: int,
    ) -> tuple[torch.Tensor, ...]:
        full_2d, weighted_2d, partial = flex_gemm_hop(
            aten.mm.default,
            projection_body,
            (projection_lhs, projection_rhs, weight_row),
            {},
            projection_kernel_options,
        )
        full = full_2d.view(full_shape)
        weighted = weighted_2d.view(full_shape)

        norm_width = norm_shape[-1]
        if full_shape[-1] != norm_width:
            norm_input = full[..., :norm_width]
            weighted_norm = weighted[..., :norm_width]
            partial = partial[..., : norm_width // reduction_group]
        else:
            norm_input = full
            weighted_norm = weighted

        physical_rstd = torch.rsqrt(partial.mean(dim=-1, keepdim=True) + eps)
        rstd = physical_rstd.view(rstd_shape)
        norm_weight = weight_row[..., :norm_width]
        norm_output = (norm_input.float() * rstd * norm_weight).to(norm_output_dtype)

        expansion_lhs = weighted_norm.view(expansion_lhs_shape)
        (expansion_output,) = flex_gemm_hop(
            aten.mm.default,
            expansion_body,
            (expansion_lhs, expansion_rhs, physical_rstd),
            {},
            expansion_kernel_options,
        )
        return full, norm_output, rstd, expansion_output

    rewrite = functools.partial(
        _rewrite_projection_rmsnorm,
        projection_body=projection_body,
        expansion_body=expansion_body,
        replacement=replacement,
    )
    pattern = F_mla_kvproj_rmsnorm_expand.__name__
    for candidate in candidates:
        norm = candidate.output_node()
        if (
            _is_backward(norm)
            or not _valid_forward_rmsnorm(norm)
            or norm.meta.get("coda_consumed")
        ):
            continue
        norm_input = norm.args[0]
        if (
            not isinstance(norm_input, Node)
            or norm_input.target is not operator.getitem
        ):
            continue
        split = norm_input.args[0]
        if (
            not isinstance(split, Node)
            or split.target is not aten.split_with_sizes.default
            or norm_input.args[1] != 0
            or not isinstance(split.args[0], Node)
        ):
            continue
        full_output = split.args[0]
        path = _chain_to_mm(full_output, allow_cast=False)
        if (
            path is None
            or not _is_reshape_only_path(path)
            or not _path_has_phase(path, backward=False)
        ):
            continue
        full_shape = _shape(full_output)
        norm_shape = _shape(norm_input)
        split_sizes = split.args[1]
        split_dim = split.args[2] if len(split.args) > 2 else split.kwargs.get("dim", 0)
        if (
            full_shape is None
            or norm_shape is None
            or not isinstance(split_sizes, (list, tuple))
            or not all(isinstance(size, int) for size in split_sizes)
            or not isinstance(split_dim, int)
        ):
            continue
        split_dim %= len(full_shape)
        if (
            split_dim != len(full_shape) - 1
            or len(split_sizes) < 2
            or split_sizes[0] != norm_shape[-1]
            or sum(split_sizes) != full_shape[-1]
            or norm_shape[:-1] != full_shape[:-1]
        ):
            continue
        norm_out = _rmsnorm_getitems(norm)
        if norm_out is None:
            continue
        second_use = _downstream_mm(norm_out[0], backward=False)
        if (
            second_use is None
            or second_use.operand != 0
            or _is_backward(second_use.node)
            or not _valid_projection_rmsnorm_use(norm, path, second_use, full_output)
            or not _coda_nodes_available((*path.nodes, norm, second_use.node))
        ):
            continue
        if not _claim_coda_match(pattern, norm, selection):
            continue
        rewrite(
            gm,
            norm=norm,
            first_path=path,
            second_use=second_use,
            full_output=full_output,
            pattern=pattern,
            autotune=autotune,
            benchmark_regions=benchmark_regions,
        )
        counts[pattern] += 1


@register_coda_pattern(
    priority=30,
    search=CallFunction(
        aten._fused_rms_norm.default,
        KeywordArg("norm_input"),
        KeywordArg("normalized_shape"),
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=MULTIPLE,
    ),
    kernels={
        # QUACK does not yet support local-reduction outputs for BMM.
        "main": CodaKernel(backend="TRITON", supports_autotune=False),
    },
)
def F_weighted_residual_bmm_prenorm(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a weighted residual BMM with its following RMSNorm.

    Match:   probabilities @ residual values -> cast -> RMSNorm
    Ops:     ``aten.bmm.default -> aten.squeeze.dim ->
             aten._to_copy.default -> aten._fused_rms_norm.default``.
    Rewrite: ``flex_gemm`` performs the BMM and emits partial mean squares;
             the rstd reduction and final normalization remain outside.
    """
    rewrite = _rewrite_forward_rmsnorm
    pattern = F_weighted_residual_bmm_prenorm.__name__
    for candidate in candidates:
        norm = candidate.output_node()
        path = _weighted_residual_rmsnorm_path(norm)
        if path is None or not _claim_coda_match(pattern, norm, selection):
            continue
        rewrite(
            gm,
            norm=norm,
            path=path,
            pattern=pattern,
            autotune=autotune,
            benchmark_regions=benchmark_regions,
        )
        counts[pattern] += 1


@register_coda_pattern(
    priority=40,
    search=CallFunction(
        aten._fused_rms_norm.default,
        KeywordArg("norm_input"),
        KeywordArg("normalized_shape"),
        KeywordArg("norm_weight"),
        KeywordArg("eps"),
        _users=MULTIPLE,
    ),
    kernels={"main": CodaKernel()},
)
def F_mm_residual_rmsnorm(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a GEMM, residual add chain, and RMSNorm.

    Match:   GEMM -> residual add(s) -> RMSNorm
    Ops:     ``aten.mm.default -> [_RESHAPE_TARGETS] -> aten.add.Tensor ->
             aten._fused_rms_norm.default`` with same-shape residuals.
    Rewrite: ``flex_gemm`` performs the add and emits partial mean squares;
             the rstd reduction and final normalization remain outside.
    """
    rewrite = _rewrite_forward_rmsnorm
    pattern = F_mm_residual_rmsnorm.__name__
    for candidate in candidates:
        norm = candidate.output_node()
        path = _residual_rmsnorm_path(norm)
        if path is None:
            continue
        getitems = _rmsnorm_getitems(norm)
        if getitems is None:
            continue
        norm_out, old_rstd = getitems
        region_nodes = (
            *path.nodes,
            norm,
            norm_out,
            *((old_rstd,) if old_rstd is not None else ()),
        )
        if not validate_partition(_ordered_nodes(gm, region_nodes)):
            continue
        if not _claim_coda_match(pattern, norm, selection):
            continue
        rewrite(
            gm,
            norm=norm,
            path=path,
            pattern=pattern,
            autotune=autotune,
            benchmark_regions=benchmark_regions,
        )
        counts[pattern] += 1


@register_coda_pattern(
    priority=50,
    search=CallFunction(
        aten.mul.Tensor,
        KeywordArg("gate_or_up"),
        KeywordArg("up_or_gate"),
        _users=MULTIPLE,
    ),
    kernels={
        "gate": CodaKernel(
            fast_math=True,
            best_configs_by_shape={
                10: {
                    (8192, 2048, 10944): _quack_config(
                        256, 224, dynamic=True, cluster_n=2
                    )
                }
            },
        ),
        "up": CodaKernel(
            fast_math=True,
            best_configs_by_shape={
                10: {(8192, 2048, 10944): _quack_config(256, 224, dynamic=True)}
            },
        ),
    },
)
def F_swiglu(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse two projections and the exact SwiGLU activation topology.

    Match:   ``gate = reshape*(mm(x, gate_weight))`` and
             ``up = reshape*(mm(x, up_weight))``, where both GEMMs share
             ``x`` and ``mul(silu(gate), up)`` is the final output.
    Rewrite: The gate ``flex_gemm`` emits ``silu(gate)``; the up
             ``flex_gemm`` consumes it and emits the final product.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = F_swiglu.__name__
    for candidate in candidates:
        output = candidate.output_node()
        match = _match_swiglu(output)
        if match is None:
            continue
        gate_nodes = {*match.gate.nodes, match.silu}
        up_nodes = {*match.up.nodes, match.output}
        if (
            _find_flex_gemm_partition(
                gm,
                root=match.gate.root,
                body_nodes=gate_nodes,
            )
            is None
            or _find_flex_gemm_partition(
                gm,
                root=match.up.root,
                body_nodes=up_nodes,
            )
            is None
            or not _claim_coda_match(pattern, output, selection)
        ):
            continue
        gate_fused = rewrite(
            gm,
            root=match.gate.root,
            body_nodes=gate_nodes,
            pattern=pattern,
            autotune=autotune,
            kernel="gate",
            benchmark_regions=benchmark_regions,
        )
        if gate_fused is None:
            continue
        up_fused = rewrite(
            gm,
            root=match.up.root,
            body_nodes=up_nodes,
            pattern=pattern,
            autotune=autotune,
            kernel="up",
            benchmark_regions=benchmark_regions,
        )
        if up_fused is None:
            raise AssertionError(f"CODA {pattern} failed after successful validation")
        counts[pattern] += 1


@register_coda_pattern(
    priority=60,
    search=CallFunction(
        aten._to_copy.default,
        KeywordArg("product"),
        dtype=torch.bfloat16,
        _users=MULTIPLE,
    ),
    kernels={
        "gate": CodaKernel(fast_math=True),
        "up": CodaKernel(fast_math=True),
    },
)
def F_situ(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse two projections and the exact SiTU activation topology.

    Match:   ``gate = to_fp32(mm(x, gate_weight))`` and
             ``up = to_fp32(mm(x, up_weight))``, followed by
             ``to_bf16((4 * tanh(gate / 4) * sigmoid(gate)) *
             (25 * tanh(up / 25)))``.
    Rewrite: The gate ``flex_gemm`` emits the complete gate activation; the
             up ``flex_gemm`` consumes it and emits the final BF16 product.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = F_situ.__name__
    for candidate in candidates:
        output = candidate.output_node()
        match = _match_situ(output)
        if match is None:
            continue
        if (
            _find_flex_gemm_partition(
                gm,
                root=match.gate.root,
                body_nodes=match.gate_nodes,
            )
            is None
            or _find_flex_gemm_partition(
                gm,
                root=match.up.root,
                body_nodes=match.up_nodes,
            )
            is None
            or not _claim_coda_match(pattern, output, selection)
        ):
            continue
        gate_fused = rewrite(
            gm,
            root=match.gate.root,
            body_nodes=match.gate_nodes,
            pattern=pattern,
            autotune=autotune,
            kernel="gate",
            benchmark_regions=benchmark_regions,
        )
        if gate_fused is None:
            continue
        up_fused = rewrite(
            gm,
            root=match.up.root,
            body_nodes=match.up_nodes,
            pattern=pattern,
            autotune=autotune,
            kernel="up",
            benchmark_regions=benchmark_regions,
        )
        if up_fused is None:
            raise AssertionError(f"CODA {pattern} failed after successful validation")
        counts[pattern] += 1


@register_coda_pattern(
    priority=70,
    search=CallFunction(
        aten.sigmoid.default,
        KeywordArg("projection"),
        _users=MULTIPLE,
    ),
    kernels={"main": CodaKernel(fast_math=True)},
)
def F_k3_mla_output_gate(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse the MLA output gate into its projection epilogue.

    Match:   projection -> sigmoid -> multiply activation
    Ops:     ``aten.mm.default -> [_VIEW_TARGETS] -> aten.sigmoid.default ->
             [_VIEW_TARGETS] -> aten.mul.Tensor``.
    Rewrite: One ``flex_gemm`` body returns the externally used sigmoid and
             gated output.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = F_k3_mla_output_gate.__name__
    for candidate in candidates:
        sigmoid = candidate.output_node()
        if (
            _is_backward(sigmoid)
            or sigmoid.target is not aten.sigmoid.default
            or sigmoid.meta.get("coda_consumed")
        ):
            continue
        chain = _chain_to_mm(sigmoid.args[0], allow_cast=False)
        mul_match = _find_pointwise_user(sigmoid, aten.mul.Tensor)
        if chain is None or mul_match is None:
            continue
        mul, bridge = mul_match
        if not _claim_coda_match(pattern, sigmoid, selection):
            continue
        fused = rewrite(
            gm,
            root=chain.root,
            body_nodes={*chain.nodes, sigmoid, *bridge, mul},
            pattern=pattern,
            autotune=autotune,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern(
    priority=80,
    search=CallFunction(
        aten.sigmoid.default,
        KeywordArg("projection"),
        _users=MULTIPLE,
    ),
    kernels={
        "main": CodaKernel(
            fast_math=True,
            best_configs={10: _quack_config(256, 256, dynamic=True)},
        )
    },
)
def F_router_sigmoid_bias(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse router sigmoid and expert bias into the projection epilogue.

    Match:   router projection -> sigmoid -> add expert bias
    Ops:     ``aten.mm.default -> [_VIEW_TARGETS] -> aten.sigmoid.default ->
             [_VIEW_TARGETS] -> aten.add.Tensor``.
    Rewrite: One ``flex_gemm`` body returns the externally used sigmoid and
             bias-adjusted scores.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = F_router_sigmoid_bias.__name__
    for candidate in candidates:
        sigmoid = candidate.output_node()
        if (
            _is_backward(sigmoid)
            or sigmoid.target is not aten.sigmoid.default
            or sigmoid.meta.get("coda_consumed")
        ):
            continue
        chain = _chain_to_mm(sigmoid.args[0], allow_cast=False)
        add_match = _find_pointwise_user(sigmoid, aten.add.Tensor)
        if chain is None or add_match is None:
            continue
        add, bridge = add_match
        if not _claim_coda_match(pattern, sigmoid, selection):
            continue
        fused = rewrite(
            gm,
            root=chain.root,
            body_nodes={*chain.nodes, sigmoid, *bridge, add},
            pattern=pattern,
            autotune=autotune,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern(
    priority=110,
    search=CallFunction(
        aten._to_copy.default,
        KeywordArg("projection"),
        dtype=torch.float32,
        _users=MULTIPLE,
    ),
    kernels={
        "main": CodaKernel(best_configs={10: _quack_config(256, 256, dynamic=True)})
    },
)
def B_reshape_bf16_to_fp32(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a reshaped GEMM output's FP32 cast into the GEMM.

    Match:   BF16 GEMM -> reshape -> FP32 cast
    Ops:     backward ``aten.mm.default -> [_RESHAPE_TARGETS] ->
             aten._to_copy.default(dtype=torch.float32)`` with a same-numel
             shape change and no transpose.
    Rewrite: ``flex_gemm`` writes FP32 after preserving the BF16 rounding point.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = B_reshape_bf16_to_fp32.__name__
    for candidate in candidates:
        cast = candidate.output_node()
        if (
            not _is_backward(cast)
            or cast.target is not _CAST_TARGET
            or _cast_dtype(cast) is not torch.float32
            or cast.meta.get("coda_consumed")
        ):
            continue
        chain = _chain_to_mm(cast)
        if (
            chain is None
            or not _path_has_phase(chain, backward=True)
            or _dtype(chain.root) is not torch.bfloat16
        ):
            continue
        has_transpose = any(
            node.target in {aten.t.default, aten.transpose.int} for node in chain.nodes
        )
        root_shape = _shape(chain.root)
        cast_shape = _shape(cast)
        reshaped_output = (
            root_shape is not None
            and cast_shape is not None
            and root_shape != cast_shape
            and _same_numel(root_shape, cast_shape)
        )
        if has_transpose or not reshaped_output:
            continue
        if not _claim_coda_match(pattern, cast, selection):
            continue
        fused = rewrite(
            gm,
            root=chain.root,
            body_nodes=chain.nodes,
            pattern=pattern,
            autotune=autotune,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern(
    priority=120,
    search=CallFunction(
        aten.silu_backward.default,
        CallFunction(
            aten.mul.Tensor,
            KeywordArg("branch_grad_or_saved_gate"),
            KeywordArg("saved_gate_or_branch_grad"),
        ),
        KeywordArg("saved_silu_output"),
        _users=MULTIPLE,
    ),
    kernels={
        "main": CodaKernel(
            fast_math=True,
            best_configs={10: _quack_config(128, 128, dynamic=True, swap_ab=True)},
            best_configs_by_shape={
                10: {(8192, 2048, 10944): _quack_config(256, 224, dynamic=True)}
            },
        )
    },
)
def B_swiglu_backward_activation(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse the exact SwiGLU pointwise backward after its gradient GEMM.

    Match:   one backward ``mm`` has exactly two ``mul`` users. One multiply
             is the externally used up gradient; the other is consumed only
             by ``aten.silu_backward.default`` to produce the gate gradient.
    Rewrite: One ``flex_gemm`` emits the up and gate gradients.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = B_swiglu_backward_activation.__name__
    for candidate in candidates:
        gate_grad = candidate.output_node()
        gate_product = gate_grad.args[0]
        if not isinstance(gate_product, Node):
            continue
        root = next(
            (
                node
                for node in gate_product.all_input_nodes
                if node.target is _MM_TARGET
            ),
            None,
        )
        if root is None:
            continue
        if root.meta.get("coda_consumed"):
            continue
        match = _match_swiglu_backward(root)
        if (
            match is None
            or gate_grad not in match.outputs
            or not _coda_nodes_available(match.body_nodes)
        ):
            continue
        ordered = _ordered_nodes(gm, match.body_nodes)
        if not validate_partition(ordered):
            continue
        if not _claim_coda_match(pattern, root, selection):
            continue
        fused = rewrite(
            gm,
            root=root,
            body_nodes=match.body_nodes,
            pattern=pattern,
            autotune=autotune,
            fused_outputs=match.outputs,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern(
    priority=130,
    search=CallFunction(
        aten.add.Tensor,
        KeywordArg("lhs"),
        KeywordArg("rhs"),
        _users=MULTIPLE,
    ),
    kernels={"main": CodaKernel()},
)
def B_parallel_mm_dx_merge(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse two parallel GEMM input-gradient branches at their final add.

    Match:   {dX GEMM, dX GEMM} -> add
    Ops:     two backward ``aten.mm.default`` chains into ``aten.add.Tensor``.
    Rewrite: One contributing GEMM and the add become one ``flex_gemm``.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = B_parallel_mm_dx_merge.__name__
    for candidate in candidates:
        add = candidate.output_node()
        if (
            not _is_backward(add)
            or add.target is not aten.add.Tensor
            or add.meta.get("coda_consumed")
        ):
            continue
        chains = [
            chain for arg in add.args[:2] if (chain := _chain_to_mm(arg)) is not None
        ]
        add_shape = _shape(add)
        add_inputs = [arg for arg in add.args[:2] if isinstance(arg, Node)]
        if (
            len(chains) != 2
            or len(add_inputs) != 2
            or add_shape is None
            or any(_shape(arg) != add_shape for arg in add_inputs)
            or any(not _path_has_phase(chain, backward=True) for chain in chains)
            or any(_dtype(chain.root) is not torch.bfloat16 for chain in chains)
            or not _claim_coda_match(pattern, add, selection)
        ):
            continue
        chain = chains[-1]
        fused = rewrite(
            gm,
            root=chain.root,
            body_nodes={*chain.nodes, add},
            pattern=pattern,
            autotune=autotune,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern(
    priority=140,
    search=CallFunction(
        aten.sigmoid_backward.default,
        CallFunction(
            aten.mul.Tensor,
            KeywordArg("gated_grad_or_saved_attention"),
            KeywordArg("saved_attention_or_gated_grad"),
        ),
        KeywordArg("saved_sigmoid_gate"),
        _users=MULTIPLE,
    ),
    kernels={"main": CodaKernel()},
)
def B_k3_mla_output_gate(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse MLA output-gate backward into the output-projection dX GEMM.

    Match:   one backward ``mm`` has exactly two ``mul`` users. One emits the
             attention gradient; the other is consumed only by
             ``aten.sigmoid_backward.default`` and uses the distinct saved
             attention input. Both branches have the GEMM output shape.
    Rewrite: One ``flex_gemm`` emits the attention and gate gradients.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = B_k3_mla_output_gate.__name__
    for candidate in candidates:
        expected_gate_grad = candidate.output_node()
        gate_product = expected_gate_grad.args[0]
        if not isinstance(gate_product, Node):
            continue
        root = next(
            (
                node
                for node in gate_product.all_input_nodes
                if node.target is _MM_TARGET
            ),
            None,
        )
        if root is None:
            continue
        if (
            not _is_backward(root)
            or root.target is not _MM_TARGET
            or root.meta.get("coda_consumed")
        ):
            continue
        mul_users = [user for user in root.users if user.target is aten.mul.Tensor]
        if len(mul_users) != 2 or set(root.users) != set(mul_users):
            continue
        match = None
        for gate_product, attention_grad in (
            (mul_users[0], mul_users[1]),
            (mul_users[1], mul_users[0]),
        ):
            gate_grad = _single_user_with_target(
                gate_product, aten.sigmoid_backward.default
            )
            if (
                gate_grad is not expected_gate_grad
                or gate_grad.args[0] is not gate_product
            ):
                continue
            sigmoid_gate = gate_grad.args[1]
            attention_gate = _other_node_input(attention_grad, root)
            saved_attention = _other_node_input(gate_product, root)
            if (
                not isinstance(sigmoid_gate, Node)
                or attention_gate is None
                or saved_attention is None
                or _alias_source(attention_gate) is not _alias_source(sigmoid_gate)
                or _alias_source(saved_attention) is _alias_source(sigmoid_gate)
            ):
                continue
            root_shape = _shape(root)
            if root_shape is None or any(
                _shape(node) != root_shape
                for node in (
                    gate_product,
                    attention_grad,
                    gate_grad,
                    sigmoid_gate,
                    attention_gate,
                    saved_attention,
                )
            ):
                continue
            match = gate_product, attention_grad, gate_grad
            break
        if match is None:
            continue
        gate_product, attention_grad, gate_grad = match
        body_nodes = {root, gate_product, attention_grad, gate_grad}
        if (
            set(_boundary_outputs(_ordered_nodes(gm, body_nodes)))
            != {attention_grad, gate_grad}
            or not _coda_nodes_available(body_nodes)
            or not _claim_coda_match(pattern, root, selection)
        ):
            continue
        fused = rewrite(
            gm,
            root=root,
            body_nodes=body_nodes,
            pattern=pattern,
            autotune=autotune,
            fused_outputs=(attention_grad, gate_grad),
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern(
    priority=150,
    search=CallFunction(
        aten._to_copy.default,
        CallFunction(
            aten.tanh_backward.default,
            KeywordArg("branch_grad_fp32"),
            KeywordArg("saved_tanh"),
        ),
        dtype=KeywordArg("output_dtype"),
        _users=MULTIPLE,
    ),
    kernels={"main": CodaKernel(fast_math=True)},
)
def B_k3_situ_backward_activation(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse the exact SiTU pointwise backward after its gradient GEMM.

    Match:   ``mm -> to_fp32`` with exactly two users:
             ``tanh_backward -> to_output_dtype`` and
             ``sigmoid_backward -> to_output_dtype``. The two casts are the
             only boundary outputs and every matched tensor has the MM shape.
    Rewrite: One ``flex_gemm`` emits the tanh and sigmoid branch gradients.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = B_k3_situ_backward_activation.__name__
    for candidate in candidates:
        branch_grad = candidate.kwargs["branch_grad_fp32"]
        if not isinstance(branch_grad, Node):
            continue
        root = _single_tensor_input(branch_grad)
        if root is None:
            continue
        if root.meta.get("coda_consumed"):
            continue
        match = _match_situ_backward(root)
        if (
            match is None
            or candidate.output_node() not in match.outputs
            or not _coda_nodes_available(match.body_nodes)
        ):
            continue
        ordered = _ordered_nodes(gm, match.body_nodes)
        if not validate_partition(ordered):
            continue
        if not _claim_coda_match(pattern, root, selection):
            continue
        fused = rewrite(
            gm,
            root=root,
            body_nodes=match.body_nodes,
            pattern=pattern,
            autotune=autotune,
            fused_outputs=match.outputs,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern(
    priority=160,
    search=CallFunction(
        aten.add.Tensor,
        KeywordArg("projection_grad"),
        KeywordArg("residual_grad"),
        _users=MULTIPLE,
    ),
    kernels={
        "main": CodaKernel(
            best_configs={10: _quack_config(256, 256, dynamic=True, cluster_n=2)}
        )
    },
)
def B_mm_dx_residual_add(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a GEMM input-gradient branch with a residual-gradient add.

    Match:   dX GEMM branch + residual gradient branch
    Ops:     backward ``aten.add.Tensor`` with exactly one input tracing through
             ``_VIEW_TARGETS`` or ``aten._to_copy.default`` to
             ``aten.mm.default``.
    Rewrite: The GEMM chain and add become one ``flex_gemm`` body.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = B_mm_dx_residual_add.__name__
    for candidate in candidates:
        add = candidate.output_node()
        if (
            not _is_backward(add)
            or add.target is not aten.add.Tensor
            or add.meta.get("coda_consumed")
        ):
            continue
        chains = [
            chain for arg in add.args[:2] if (chain := _chain_to_mm(arg)) is not None
        ]
        add_shape = _shape(add)
        add_inputs = [arg for arg in add.args[:2] if isinstance(arg, Node)]
        if (
            len(chains) != 1
            or len(add_inputs) != 2
            or add_shape is None
            or any(_shape(arg) != add_shape for arg in add_inputs)
            or not _path_has_phase(chains[0], backward=True)
        ):
            continue
        chain = chains[0]
        body_nodes = {*chain.nodes, add}
        if (
            not _is_supported_flex_gemm_root(chain.root)
            or not _coda_nodes_available(body_nodes)
            or not validate_partition(_ordered_nodes(gm, body_nodes))
        ):
            continue
        if not _claim_coda_match(pattern, add, selection):
            continue
        fused = rewrite(
            gm,
            root=chain.root,
            body_nodes=body_nodes,
            pattern=pattern,
            autotune=autotune,
            benchmark_regions=benchmark_regions,
        )
        if fused is not None:
            counts[pattern] += 1


@register_coda_pattern(
    priority=170,
    search=CallFunction(
        aten._fused_rms_norm_backward.default,
        KeywordArg("grad"),
        KeywordArg("norm_input"),
        KeywordArg("normalized_shape"),
        KeywordArg("rstd"),
        KeywordArg("norm_weight"),
        KeywordArg("output_mask"),
        _users=MULTIPLE,
    ),
    kernels={"main": CodaKernel()},
)
def B_mm_dx_rmsnorm(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a GEMM input gradient with RMSNorm backward.

    Match:   dX GEMM -> RMSNorm backward
    Ops:     backward ``aten.mm.default -> [_RESHAPE_TARGETS] ->
             aten._fused_rms_norm_backward.default``.
    Rewrite: ``flex_gemm`` emits d_x_hat and partial row dots; the remaining
             reduction and pointwise operations produce input and weight grads.
    """
    rewrite = _rewrite_backward_rmsnorm
    pattern = B_mm_dx_rmsnorm.__name__
    for candidate in candidates:
        node = candidate.output_node()
        if (
            not _is_backward(node)
            or not _valid_backward_rmsnorm(node)
            or node.meta.get("coda_consumed")
        ):
            continue
        path = _chain_to_mm(node.args[0], allow_cast=False)
        if (
            path is None
            or not _is_reshape_only_path(path)
            or not _path_has_phase(path, backward=True)
            or not _coda_nodes_available((*path.nodes, node))
            or not _claim_coda_match(pattern, node, selection)
        ):
            continue
        rewrite(
            gm,
            norm_backward=node,
            path=path,
            pattern=pattern,
            autotune=autotune,
            benchmark_regions=benchmark_regions,
        )
        counts[pattern] += 1


@register_coda_pattern(
    priority=180,
    search=CallFunction(
        aten._to_copy.default,
        KeywordArg("weight_grad"),
        dtype=torch.float32,
        _users=MULTIPLE,
    ),
    kernels={
        "main": CodaKernel(
            best_configs_by_shape={
                10: {
                    (2048, 8192, 10944): _quack_config(
                        256, 256, dynamic=True, cluster_n=2
                    ),
                    (10944, 8192, 2048): _quack_config(
                        256, 192, dynamic=False, swap_ab=True
                    ),
                    (102400, 1024, 2048): _quack_config(
                        256, 256, dynamic=False, swap_ab=True
                    ),
                }
            },
        )
    },
)
def B_linear_dw_bf16_to_fp32(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a linear weight-gradient FP32 cast into its BF16 GEMM.

    Match:   BF16 weight-gradient GEMM -> optional view/transpose -> FP32 cast
    Ops:     backward ``aten.mm.default -> [_VIEW_TARGETS] ->
             aten._to_copy.default(dtype=torch.float32)`` excluding the
             reshape-only LM-head dX case.
    Rewrite: ``flex_gemm`` writes FP32 after preserving the BF16 rounding point;
             transpose and shape-only operations remain outside when needed.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    rewrite_transposed = _rewrite_transposed_cast
    pattern = B_linear_dw_bf16_to_fp32.__name__
    for candidate in candidates:
        cast = candidate.output_node()
        if (
            not _is_backward(cast)
            or cast.target is not _CAST_TARGET
            or _cast_dtype(cast) is not torch.float32
            or cast.meta.get("coda_consumed")
        ):
            continue
        chain = _chain_to_mm(cast)
        if (
            chain is None
            or not _path_has_phase(chain, backward=True)
            or _dtype(chain.root) is not torch.bfloat16
        ):
            continue
        has_transpose = any(
            node.target in {aten.t.default, aten.transpose.int} for node in chain.nodes
        )
        root_shape = _shape(chain.root)
        cast_shape = _shape(cast)
        reshaped_output = (
            root_shape is not None
            and cast_shape is not None
            and root_shape != cast_shape
            and _same_numel(root_shape, cast_shape)
        )
        if reshaped_output and not has_transpose:
            continue
        if not _claim_coda_match(pattern, cast, selection):
            continue
        if has_transpose:
            if not rewrite_transposed(
                gm,
                chain=chain,
                cast=cast,
                pattern=pattern,
                autotune=autotune,
                benchmark_regions=benchmark_regions,
            ):
                continue
        else:
            fused = rewrite(
                gm,
                root=chain.root,
                body_nodes=chain.nodes,
                pattern=pattern,
                autotune=autotune,
                benchmark_regions=benchmark_regions,
            )
            if fused is None:
                continue
        counts[pattern] += 1


# Public order is explicit and independent of source insertion order.
CODA_PATTERN_NAMES = tuple(
    pattern.name
    for pattern in sorted(
        _CODA_PATTERNS.values(),
        key=lambda pattern: (pattern.priority, pattern.name),
    )
)


# Implementation details

RMSNORM_FORWARD_GROUP = 64
RMSNORM_BACKWARD_GROUP = 128
_CODA_INDUCTOR_REGION = "coda_flex_gemm"
_VIEW_TARGETS = {
    aten.alias.default,
    aten.reshape.default,
    aten.t.default,
    aten.transpose.int,
    aten.view.default,
    aten._unsafe_view.default,
}
_RESHAPE_TARGETS = {
    aten.alias.default,
    aten.reshape.default,
    aten.view.default,
    aten._unsafe_view.default,
}
_CAST_TARGET = aten._to_copy.default
_MM_TARGET = aten.mm.default
_BMM_TARGET = aten.bmm.default


@dataclass(frozen=True)
class _MatmulChain:
    root: Node
    nodes: tuple[Node, ...]


@dataclass(frozen=True)
class _MatmulUse:
    node: Node
    operand: int
    wrappers: tuple[Node, ...]


@dataclass(frozen=True)
class _SwiGLUMatch:
    gate: _MatmulChain
    silu: Node
    up: _MatmulChain
    output: Node


@dataclass(frozen=True)
class _SiTUMatch:
    gate: _MatmulChain
    gate_nodes: frozenset[Node]
    up: _MatmulChain
    up_nodes: frozenset[Node]
    output: Node


@dataclass(frozen=True)
class _PointwiseBackwardMatch:
    root: Node
    body_nodes: frozenset[Node]
    outputs: tuple[Node, ...]


@dataclass(frozen=True)
class _FlexGemmPartition:
    nodes: tuple[Node, ...]
    outputs: tuple[Node, ...]


@dataclass(frozen=True)
class _InsertedFlexGemm:
    outputs: tuple[Node, ...]
    body_name: str
    region: str


@dataclass(frozen=True)
class _ForwardRMSNormOutputs:
    output: Node
    rstd: Node
    physical_rstd: Node


@dataclass(frozen=True)
class _BackwardRMSNormOutputs:
    input_grad: Node
    weight_grad: Node


def _gemm_shape(root: Node) -> tuple[int, int, int] | None:
    if root.target is not _MM_TARGET or len(root.args) < 2:
        return None
    lhs, rhs = root.args[:2]
    if not isinstance(lhs, Node) or not isinstance(rhs, Node):
        return None
    lhs_shape = _shape(lhs)
    rhs_shape = _shape(rhs)
    if (
        lhs_shape is None
        or rhs_shape is None
        or len(lhs_shape) != 2
        or len(rhs_shape) != 2
    ):
        return None
    gemm_shape = (lhs_shape[0], lhs_shape[1], rhs_shape[1])
    if not all(isinstance(dim, int) for dim in gemm_shape):
        return None
    return gemm_shape


def _best_config(
    pattern: str,
    kernel: str,
    root: Node | None = None,
) -> dict[str, Any] | None:
    if not torch.cuda.is_available():
        return None
    device_capacity = torch.cuda.get_device_capability()[0]
    if device_capacity == 11:
        device_capacity = 10
    registered = _CODA_PATTERNS[pattern].kernels[kernel]
    if root is not None:
        shape = _gemm_shape(root)
        shape_configs = registered.best_configs_by_shape.get(device_capacity, {})
        if shape in shape_configs:
            return shape_configs[shape]
    return registered.best_configs.get(device_capacity)


def _kernel_options(
    pattern: str,
    *,
    kernel: str = "main",
    root: Node | None = None,
    autotune: bool = False,
) -> dict[str, Any]:
    try:
        policy = _CODA_PATTERNS[pattern].kernels[kernel]
    except KeyError as error:
        raise AssertionError(
            f"CODA pattern {pattern!r} has no kernel site {kernel!r}"
        ) from error
    options: dict[str, Any] = {"backend": policy.backend}
    if policy.backend == "QUACK":
        options["tuned"] = autotune and policy.supports_autotune
        config = None if autotune else _best_config(pattern, kernel, root)
        if config is not None:
            options["tuned"] = False
            options["config"] = dict(config)
    if policy.fast_math:
        options["fast_math"] = True
    return options


def _is_backward(node: Node) -> bool:
    return bool(node.meta.get("autograd_backward"))


def _claim_coda_match(
    pattern: str,
    anchor: Node,
    selection: BenchmarkCandidateSelection | None,
) -> bool:
    if selection is None:
        return True
    match_id = f"{pattern}:{anchor.name}"
    if match_id in selection.rejected:
        return False
    if selection.selected is None:
        selection.selected = match_id
    return selection.selected == match_id


def _path_has_phase(path: _MatmulChain, *, backward: bool) -> bool:
    return all(_is_backward(node) is backward for node in path.nodes)


def _tensor_value(node: Node) -> torch.Tensor | None:
    value = node.meta.get("val")
    return value if isinstance(value, torch.Tensor) else None


def _shape(node: Node) -> tuple[Any, ...] | None:
    value = _tensor_value(node)
    return tuple(value.shape) if value is not None else None


def _static_numel(shape: Sequence[Any]) -> int | None:
    result = 1
    for dim in shape:
        if not isinstance(dim, int):
            return None
        result *= dim
    return result


def _same_numel(lhs: Sequence[Any], rhs: Sequence[Any]) -> bool:
    lhs_static = _static_numel(lhs)
    rhs_static = _static_numel(rhs)
    if lhs_static is not None and rhs_static is not None:
        return lhs_static == rhs_static
    return statically_known_true(prod(lhs) == prod(rhs))


def _dtype(node: Node) -> torch.dtype | None:
    value = _tensor_value(node)
    return value.dtype if value is not None else None


def _single_tensor_input(node: Node) -> Node | None:
    inputs = [arg for arg in node.all_input_nodes]
    return inputs[0] if len(inputs) == 1 else None


def _alias_source(node: Node) -> Node:
    while node.target is aten.alias.default:
        source = _single_tensor_input(node)
        if source is None:
            break
        node = source
    return node


def _other_node_input(node: Node, known: Node) -> Node | None:
    inputs = list(node.all_input_nodes)
    if len(inputs) != 2 or known not in inputs:
        return None
    return inputs[1] if inputs[0] is known else inputs[0]


def _chain_to_mm(node: object, *, allow_cast: bool = True) -> _MatmulChain | None:
    if not isinstance(node, Node):
        return None
    reverse_path: list[Node] = []
    current = node
    while current.target in _VIEW_TARGETS or (
        allow_cast and current.target is _CAST_TARGET
    ):
        reverse_path.append(current)
        current = _single_tensor_input(current)
        if current is None:
            return None
    if current.target is not _MM_TARGET:
        return None
    return _MatmulChain(current, tuple([current, *reversed(reverse_path)]))


def _find_mm_path(node: Node) -> _MatmulChain | None:
    direct = _chain_to_mm(node)
    if direct is not None:
        return direct
    frontier = deque([(node, ())])
    visited: set[Node] = set()
    while frontier:
        current, outer_adds = frontier.popleft()
        if current in visited or current.target is not aten.add.Tensor:
            continue
        visited.add(current)
        for arg in current.all_input_nodes:
            path = _chain_to_mm(arg)
            if path is not None:
                return _MatmulChain(
                    path.root,
                    (*path.nodes, current, *reversed(outer_adds)),
                )
        frontier.extend(
            (arg, (*outer_adds, current)) for arg in current.all_input_nodes
        )
    return None


def _is_reshape_only_path(path: _MatmulChain) -> bool:
    return all(
        node is path.root
        or node.target in _RESHAPE_TARGETS
        or node.target is aten.add.Tensor
        for node in path.nodes
    )


def _has_exact_add_shapes(path: _MatmulChain) -> bool:
    for node in path.nodes:
        if node.target is not aten.add.Tensor:
            continue
        inputs = [arg for arg in node.args[:2] if isinstance(arg, Node)]
        output_shape = _shape(node)
        if len(inputs) != 2 or output_shape is None:
            return False
        if any(_shape(arg) != output_shape for arg in inputs):
            return False
    return True


def _ordered_nodes(gm: GraphModule, nodes: Iterable[Node]) -> list[Node]:
    selected = set(nodes)
    return [node for node in gm.graph.nodes if node in selected]


def _coda_nodes_available(nodes: Iterable[Node]) -> bool:
    return all(not node.meta.get("coda_consumed") for node in nodes)


def _mark_coda_owned(pattern: str, nodes: Iterable[Node]) -> None:
    for node in nodes:
        owner = node.meta.get("coda_owner")
        if owner is not None and owner != pattern:
            raise AssertionError(
                f"CODA node {node.name} is already owned by pattern {owner}"
            )
        node.meta["coda_consumed"] = True
        node.meta["coda_owner"] = pattern


def _single_user_with_target(node: Node, target: Any) -> Node | None:
    matches = [user for user in node.users if user.target is target]
    return matches[0] if len(matches) == 1 else None


def _boundary_outputs(body_nodes: Sequence[Node]) -> list[Node]:
    body_set = set(body_nodes)
    return [
        node for node in body_nodes if any(user not in body_set for user in node.users)
    ]


def _unique_submodule_name(gm: GraphModule, prefix: str) -> str:
    index = 0
    while hasattr(gm, f"{prefix}_{index}"):
        index += 1
    return f"{prefix}_{index}"


def _copy_meta(dst: Node, src: Node) -> None:
    dst.meta = src.meta.copy()
    custom = dst.meta.get("custom")
    if isinstance(custom, dict):
        dst.meta["custom"] = custom.copy()
        compile_with_inductor = custom.get("compile_with_inductor")
        if isinstance(compile_with_inductor, dict):
            dst.meta["custom"]["compile_with_inductor"] = compile_with_inductor.copy()


def _call_after(
    graph: torch.fx.Graph,
    cursor: Node,
    target: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
) -> Node:
    with graph.inserting_after(cursor):
        return graph.call_function(target, args=args, kwargs=kwargs or {})


def _set_view_meta(node: Node, source: Node, shape: Sequence[Any]) -> None:
    value = _tensor_value(source)
    if value is not None:
        node.meta["val"] = aten.view.default(value, list(shape))


def _set_empty_meta(
    node: Node,
    source: Node,
    shape: Sequence[Any],
    *,
    dtype: torch.dtype,
) -> None:
    value = _tensor_value(source)
    if value is not None:
        node.meta["val"] = value.new_empty(tuple(shape), dtype=dtype)


def _view_region_input(
    gm: GraphModule,
    node: Node,
    shape: Sequence[Any],
    existing_nodes: set[Node],
) -> Node:
    node_shape = _shape(node)
    if node_shape == tuple(shape):
        return node
    value = _tensor_value(node)
    if value is None:
        raise AssertionError(f"CODA input {node.name} requires tensor metadata")
    try:
        viewed_value = aten.view.default(value, list(shape))
    except RuntimeError as error:
        raise AssertionError(
            f"CODA input {node.name} cannot be viewed as {tuple(shape)}"
        ) from error
    reshaped = _call_after(
        gm.graph,
        node,
        aten.view.default,
        (node, list(shape)),
    )
    reshaped.meta["val"] = viewed_value
    existing_nodes.add(reshaped)
    return reshaped


def _register_body(gm: GraphModule, body: GraphModule, pattern: str) -> str:
    name = _unique_submodule_name(gm, f"coda_{pattern.lower()}_body")
    gm.add_submodule(name, body)
    return name


def _trace_replacement(
    replacement: Callable[..., Any],
    inputs: Sequence[torch.Tensor],
) -> GraphModule:
    fake_modes = {value.fake_mode for value in inputs if isinstance(value, FakeTensor)}
    if len(fake_modes) != 1 or not all(
        isinstance(value, FakeTensor) for value in inputs
    ):
        raise AssertionError("CODA replacement inputs must share one FakeTensorMode")
    fake_mode = next(iter(fake_modes))
    shape_env = fake_mode.shape_env
    pending_unbacked = None
    ignorable_unbacked = None
    if shape_env is not None:
        pending_unbacked = list(shape_env.pending_fresh_unbacked_symbols)
        ignorable_unbacked = list(shape_env.ignorable_fresh_unbacked_symbols)
        shape_env.pending_fresh_unbacked_symbols.clear()
        shape_env.ignorable_fresh_unbacked_symbols.clear()
    try:
        with fake_mode, enable_python_dispatcher():
            return make_fx(replacement)(*inputs)
    finally:
        if shape_env is not None:
            assert pending_unbacked is not None
            assert ignorable_unbacked is not None
            shape_env.pending_fresh_unbacked_symbols[:] = pending_unbacked
            shape_env.ignorable_fresh_unbacked_symbols[:] = ignorable_unbacked


def _inline_flex_gemm_replacement(
    gm: GraphModule,
    replacement: GraphModule,
    inputs: Sequence[Node],
    outputs: Sequence[Node],
    body: GraphModule,
    body_nodes: set[Node],
    *,
    pattern: str,
) -> Node:
    placeholders = list(replacement.graph.find_nodes(op="placeholder"))
    if len(placeholders) != len(inputs):
        raise AssertionError(
            f"CODA {pattern} replacement has {len(placeholders)} inputs, "
            f"expected {len(inputs)}"
        )
    (
        source_body,
        source_body_attr,
        source_fused,
        source_nodes,
    ) = _mark_flex_gemm_replacement(replacement, pattern=pattern)
    body_name = _register_body(gm, body, pattern)
    env = dict(zip(placeholders, inputs, strict=True))
    with gm.graph.inserting_before(outputs[0]):
        body_attr = gm.graph.get_attr(body_name)
        _copy_meta(body_attr, source_body_attr)
        env[source_body_attr] = body_attr
        replacement_outputs = gm.graph.graph_copy(replacement.graph, env)

    if not isinstance(replacement_outputs, tuple):
        replacement_outputs = (replacement_outputs,)
    if len(replacement_outputs) != len(outputs):
        raise AssertionError(
            f"CODA {pattern} replacement has {len(replacement_outputs)} outputs, "
            f"expected {len(outputs)}"
        )
    fused = env[source_fused]
    new_nodes = [env[node] for node in source_nodes]
    for source, copied in zip(source_nodes, new_nodes, strict=True):
        _copy_meta(copied, source)

    _mark_inductor_region(body, body_attr, fused, pattern=pattern)
    _mark_nodes_for_inductor(new_nodes, group=body_name)
    for old, new in zip(outputs, replacement_outputs, strict=True):
        _copy_meta(new, old)
        new.meta.setdefault("custom", {})["coda_pattern"] = pattern
        for user in list(old.users):
            if user not in body_nodes:
                user.replace_input_with(old, new)
    return fused


def _inline_traced_coda_replacement(
    gm: GraphModule,
    replacement: GraphModule,
    inputs: Sequence[Node],
    cursor: Node,
    *,
    pattern: str,
) -> tuple[Node, ...]:
    """Inline a traced replacement containing one or more FlexGEMM calls."""
    placeholders = list(replacement.graph.find_nodes(op="placeholder"))
    if len(placeholders) != len(inputs):
        raise AssertionError(
            f"CODA {pattern} replacement has {len(placeholders)} inputs, "
            f"expected {len(inputs)}"
        )

    env = dict(zip(placeholders, inputs, strict=True))
    copied_nodes: list[Node] = []
    bodies: dict[Node, tuple[GraphModule, Node, str]] = {}
    output_cursor = cursor
    for source in replacement.graph.nodes:
        if source.op in ("placeholder", "output"):
            continue
        if source.op == "get_attr":
            source_body = getattr(replacement, source.target)
            if not isinstance(source_body, GraphModule):
                raise AssertionError(
                    f"CODA {pattern} replacement attribute {source.target!r} "
                    "is not a GraphModule"
                )
            body_name = _register_body(gm, source_body, pattern)
            with gm.graph.inserting_after(output_cursor):
                copied = gm.graph.get_attr(body_name)
            bodies[source] = (source_body, copied, body_name)
        else:
            with gm.graph.inserting_after(output_cursor):
                copied = gm.graph.node_copy(source, lambda node: env[node])
        _copy_meta(copied, source)
        env[source] = copied
        copied_nodes.append(copied)
        output_cursor = copied

    flex_gemms = [
        source for source in replacement.graph.nodes if source.target is flex_gemm_hop
    ]
    if not flex_gemms:
        raise AssertionError(f"CODA {pattern} replacement has no FlexGEMM call")
    first_group = next(iter(bodies.values()))[2]
    for source_fused in flex_gemms:
        source_body_attr = source_fused.args[1]
        if not isinstance(source_body_attr, Node) or source_body_attr not in bodies:
            raise AssertionError(f"CODA {pattern} FlexGEMM has no registered body")
        body, body_attr, _ = bodies[source_body_attr]
        fused = env[source_fused]
        _mark_inductor_region(body, body_attr, fused, pattern=pattern)
        _mark_nodes_for_inductor(body.graph.nodes, group=first_group)
    _mark_nodes_for_inductor(copied_nodes, group=first_group)

    source_output = next(
        node for node in replacement.graph.nodes if node.op == "output"
    ).args[0]
    replacement_outputs = torch.fx.node.map_arg(source_output, lambda node: env[node])
    if not isinstance(replacement_outputs, (list, tuple)):
        replacement_outputs = (replacement_outputs,)
    if not all(isinstance(output, Node) for output in replacement_outputs):
        raise AssertionError(f"CODA {pattern} replacement outputs must be FX nodes")
    return tuple(replacement_outputs)


def _mark_flex_gemm_replacement(
    replacement: GraphModule,
    *,
    pattern: str,
) -> tuple[GraphModule, Node, Node, list[Node]]:
    """Mark an isolated FlexGEMM replacement for regional Inductor."""
    body_attrs = list(replacement.graph.find_nodes(op="get_attr"))
    if len(body_attrs) != 1:
        raise AssertionError(
            f"CODA {pattern} replacement expected one FlexGEMM body, "
            f"found {len(body_attrs)}"
        )
    source_body_attr = body_attrs[0]
    source_body = getattr(replacement, source_body_attr.target)
    source_fused = next(
        (node for node in replacement.graph.nodes if node.target is flex_gemm_hop),
        None,
    )
    if not isinstance(source_body, GraphModule) or source_fused is None:
        raise AssertionError(f"CODA {pattern} replacement has no FlexGEMM body")
    source_nodes = [
        node
        for node in replacement.graph.nodes
        if node.op not in ("placeholder", "output")
    ]
    _mark_inductor_region(
        source_body,
        source_body_attr,
        source_fused,
        pattern=pattern,
    )
    _mark_nodes_for_inductor(source_nodes, group=str(source_body_attr.target))
    return source_body, source_body_attr, source_fused, source_nodes


def _make_local_rewrite_benchmark_region(
    gm: GraphModule,
    nodes: Iterable[Node],
    pattern: str,
    rewrite: Callable[[GraphModule, dict[str, Node]], None],
) -> RewriteBenchmarkRegion:
    """Build baseline and candidate graphs from only the matched region."""
    ordered = _ordered_nodes(gm, nodes)
    baseline, _, _ = fuse_as_graphmodule(
        gm,
        ordered,
        f"BenchmarkBaseline_{pattern}",
        always_return_tuple=True,
    )
    candidate, _, _ = fuse_as_graphmodule(
        gm,
        ordered,
        f"BenchmarkCandidate_{pattern}",
        always_return_tuple=True,
    )
    for region in (baseline, candidate):
        for node in region.graph.nodes:
            _copy_meta(node, node)
    candidate_nodes = {node.name: node for node in candidate.graph.nodes}
    rewrite(candidate, candidate_nodes)
    _prepare_coda_graph(candidate)
    return make_rewrite_benchmark_region(baseline, candidate)


def _propagate_body_meta(body: GraphModule, inputs: Sequence[Node]) -> None:
    values = [node.meta.get("val") for node in inputs]
    if not all(isinstance(value, torch.Tensor) for value in values):
        raise AssertionError("CODA FlexGEMM body inputs require tensor metadata")
    fake_modes = {value.fake_mode for value in values if isinstance(value, FakeTensor)}
    if len(fake_modes) == 1 and all(isinstance(value, FakeTensor) for value in values):
        fake_mode = next(iter(fake_modes))
        FakeTensorProp(body, mode=fake_mode).propagate_dont_convert_inputs(*values)
        return
    if not fake_modes:
        FakeTensorProp(body).propagate(*values)
        return

    fake_mode = FakeTensorMode()
    normalized_values = []
    for value in values:
        meta_value = torch.empty_strided(
            value.shape,
            value.stride(),
            dtype=value.dtype,
            device="meta",
        )
        normalized_values.append(
            fake_mode.fake_tensor_converter.from_meta_and_device(
                fake_mode, meta_value, value.device
            )
        )
    FakeTensorProp(body, mode=fake_mode).propagate_dont_convert_inputs(
        *normalized_values
    )


def _body_output_values(body: GraphModule, pattern: str) -> tuple[torch.Tensor, ...]:
    output = next(node for node in body.graph.nodes if node.op == "output")
    result_nodes = output.args[0]
    if not isinstance(result_nodes, (list, tuple)):
        result_nodes = (result_nodes,)
    values = tuple(
        _tensor_value(node) if isinstance(node, Node) else None for node in result_nodes
    )
    if not all(value is not None for value in values):
        raise AssertionError(f"CODA {pattern} body outputs require tensor metadata")
    return tuple(value for value in values if isinstance(value, torch.Tensor))


def _validate_body_outputs(
    body: GraphModule,
    expected_values: Sequence[torch.Tensor | None],
    pattern: str,
) -> tuple[torch.Tensor, ...]:
    actual_values = _body_output_values(body, pattern)
    if len(actual_values) != len(expected_values):
        raise AssertionError(
            f"CODA {pattern} body returns {len(actual_values)} outputs, "
            f"expected {len(expected_values)}"
        )
    for index, (actual, expected) in enumerate(
        zip(actual_values, expected_values, strict=True)
    ):
        if expected is None:
            raise AssertionError(
                f"CODA {pattern} output {index} requires tensor metadata"
            )
        actual_spec = (
            tuple(actual.shape),
            tuple(actual.stride()),
            actual.dtype,
            actual.device,
        )
        expected_spec = (
            tuple(expected.shape),
            tuple(expected.stride()),
            expected.dtype,
            expected.device,
        )
        if actual_spec != expected_spec:
            raise AssertionError(
                f"CODA {pattern} output {index} has spec {actual_spec}, "
                f"expected {expected_spec}"
            )
    return actual_values


def _mark_inductor_region(
    body: GraphModule,
    body_attr: Node,
    fused: Node,
    *,
    pattern: str,
    region: str = _CODA_INDUCTOR_REGION,
) -> None:
    group = str(body_attr.target)
    _mark_nodes_for_inductor((body_attr, fused), region=region, group=group)
    fused.meta["custom"]["coda_pattern"] = pattern
    _mark_nodes_for_inductor(body.graph.nodes, region=region, group=group)


def _mark_nodes_for_inductor(
    nodes: Iterable[Node],
    *,
    region: str = _CODA_INDUCTOR_REGION,
    group: str | None = None,
) -> None:
    annotation = {"inductor_region": region}
    for node in nodes:
        custom = node.meta.setdefault("custom", {})
        custom["compile_with_inductor"] = annotation
        if group is not None:
            custom["coda_region_group"] = group


def _coda_region_group(node: Node) -> str | None:
    custom = node.meta.get("custom", {})
    group = custom.get("coda_region_group") if isinstance(custom, dict) else None
    return group if isinstance(group, str) else None


def _assign_coda_inductor_regions(gm: GraphModule) -> None:
    groups: dict[str, list[Node]] = {}
    for node in gm.graph.nodes:
        group = _coda_region_group(node)
        if group is not None:
            groups.setdefault(group, []).append(node)

    graph_order = {node: index for index, node in enumerate(gm.graph.nodes)}
    ordered_groups = sorted(
        groups,
        key=lambda group: min(graph_order[node] for node in groups[group]),
    )
    assignments = {
        group: f"{_CODA_INDUCTOR_REGION}_{index}"
        for index, group in enumerate(ordered_groups)
    }

    for module in gm.modules():
        if not isinstance(module, GraphModule):
            continue
        for node in module.graph.nodes:
            group = _coda_region_group(node)
            if group not in assignments:
                continue
            custom = node.meta["custom"]
            custom["compile_with_inductor"]["inductor_region"] = assignments[group]


def _infer_new_node_meta(node: Node) -> None:
    if node.op != "call_function" or "val" in node.meta:
        return

    def load_value(arg: Node) -> Any:
        if "val" not in arg.meta:
            raise AssertionError(f"CODA input {arg.name} is missing value metadata")
        return arg.meta["val"]

    args, kwargs = torch.fx.node.map_arg((node.args, node.kwargs), load_value)
    if node.target is operator.getitem:
        node.meta["val"] = node.target(*args, **kwargs)
        return

    fake_mode = FakeTensorMode()

    def normalize(value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        device = value.fake_device if isinstance(value, FakeTensor) else value.device
        meta_value = torch.empty_strided(
            value.shape,
            value.stride(),
            dtype=value.dtype,
            device="meta",
        )
        return fake_mode.fake_tensor_converter.from_meta_and_device(
            fake_mode, meta_value, device
        )

    args, kwargs = torch.utils._pytree.tree_map(normalize, (args, kwargs))
    with fake_mode:
        node.meta["val"] = node.target(*args, **kwargs)


def _mark_new_nodes_for_inductor(
    gm: GraphModule,
    existing_nodes: set[Node],
    *,
    region: str = _CODA_INDUCTOR_REGION,
    group: str | None = None,
) -> None:
    new_nodes = [node for node in gm.graph.nodes if node not in existing_nodes]
    for node in new_nodes:
        _infer_new_node_meta(node)
        custom = node.meta.setdefault("custom", {})
        custom.setdefault(
            "compile_with_inductor",
            {"inductor_region": region},
        )
        if group is not None:
            custom.setdefault("coda_region_group", group)


def _is_supported_flex_gemm_root(root: Node) -> bool:
    if root.target not in {_MM_TARGET, _BMM_TARGET}:
        return False
    inputs = root.all_input_nodes
    if len(inputs) < 2:
        return False
    lhs_dtype = _dtype(inputs[0])
    rhs_dtype = _dtype(inputs[1])
    return lhs_dtype == rhs_dtype and lhs_dtype in {
        torch.bfloat16,
        torch.float16,
    }


def _find_flex_gemm_partition(
    gm: GraphModule,
    *,
    root: Node,
    body_nodes: Iterable[Node],
    fused_outputs: Iterable[Node] | None = None,
) -> _FlexGemmPartition | None:
    if not _is_supported_flex_gemm_root(root):
        return None
    ordered = _ordered_nodes(gm, body_nodes)
    if not _coda_nodes_available(ordered) or not validate_partition(ordered):
        return None
    outputs = (
        _ordered_nodes(gm, fused_outputs)
        if fused_outputs is not None
        else _boundary_outputs(ordered)
    )
    if not outputs:
        return None
    return _FlexGemmPartition(tuple(ordered), tuple(outputs))


def _rewrite_matched_epilogue_as_flex_gemm(
    gm: GraphModule,
    *,
    root: Node,
    body_nodes: Iterable[Node],
    pattern: str,
    autotune: bool = False,
    kernel: str = "main",
    fused_outputs: Iterable[Node] | None = None,
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> Node | None:
    """Fuse an already matched subgraph while preserving its exact FX nodes.

    Fixed replacement formulas are traced from tensor functions below. This
    generic path intentionally copies nodes because it accepts variable
    epilogues selected by the registered structural matcher.
    """
    partition = _find_flex_gemm_partition(
        gm,
        root=root,
        body_nodes=body_nodes,
        fused_outputs=fused_outputs,
    )
    if partition is None:
        return None
    ordered = list(partition.nodes)
    body_set = set(ordered)
    outputs = list(partition.outputs)

    external_inputs: list[Node] = []
    for arg in root.all_input_nodes:
        if arg not in external_inputs:
            external_inputs.append(arg)
    for node in ordered:
        for arg in node.all_input_nodes:
            if arg not in body_set and arg not in external_inputs:
                external_inputs.append(arg)

    root_shape = _shape(root)
    if root_shape is None:
        raise AssertionError(f"CODA {pattern} requires GEMM shape metadata")
    actual_inputs = external_inputs

    body_graph = torch.fx.Graph()
    env: dict[Node, Node] = {}
    for index, (external, actual) in enumerate(zip(external_inputs, actual_inputs)):
        placeholder = body_graph.placeholder(f"arg{index}")
        _copy_meta(placeholder, actual)
        env[external] = placeholder
    for node in ordered:
        copied = body_graph.node_copy(node, lambda old: env[old])
        _copy_meta(copied, node)
        env[node] = copied
    outputs.sort(key=lambda node: not _same_numel(_shape(node) or (), root_shape))
    body_results: list[Node] = []
    restore_shapes: list[tuple[Any, ...] | None] = []
    for output in outputs:
        output_shape = _shape(output)
        if (
            output_shape is not None
            and output_shape != root_shape
            and _same_numel(output_shape, root_shape)
        ):
            body_results.append(
                body_graph.call_function(
                    aten.view.default, args=(env[output], list(root_shape))
                )
            )
            restore_shapes.append(output_shape)
        else:
            body_results.append(env[output])
            restore_shapes.append(None)
    body_graph.output(tuple(body_results))
    body = GraphModule(torch.nn.Module(), body_graph)
    _propagate_body_meta(body, actual_inputs)
    physical_values = []
    for output, restore_shape in zip(outputs, restore_shapes, strict=True):
        value = _tensor_value(output)
        if value is not None and restore_shape is not None:
            value = aten.view.default(value, list(root_shape))
        physical_values.append(value)
    body_values = _validate_body_outputs(body, physical_values, pattern)
    mark_flex_gemm_body_gemm_node(body, root.target)
    options = _kernel_options(
        pattern,
        kernel=kernel,
        root=root,
        autotune=autotune,
    )
    body_inputs = tuple(
        node.meta.get("val") for node in body.graph.nodes if node.op == "placeholder"
    )
    if not all(isinstance(value, torch.Tensor) for value in body_inputs):
        raise AssertionError(f"CODA {pattern} body inputs require tensor metadata")

    def restore_outputs(results: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        return tuple(
            (
                aten.view.default(result, list(restore_shape))
                if restore_shape is not None
                else result
            )
            for result, restore_shape in zip(results, restore_shapes, strict=True)
        )

    def baseline(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return restore_outputs(body(*args))

    def replacement(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return restore_outputs(
            flex_gemm_hop(
                root.target,
                body,
                tuple(args),
                {},
                options,
            )
        )

    replacement_graph = _trace_replacement(replacement, body_inputs)
    if benchmark_regions is not None:
        baseline_graph = _trace_replacement(baseline, body_inputs)
        _mark_flex_gemm_replacement(replacement_graph, pattern=pattern)
        _prepare_coda_graph(replacement_graph)
        benchmark_regions.append(
            make_rewrite_benchmark_region(baseline_graph, replacement_graph)
        )
        return root

    fused = _inline_flex_gemm_replacement(
        gm,
        replacement_graph,
        actual_inputs,
        outputs,
        body,
        body_set,
        pattern=pattern,
    )
    _mark_coda_owned(pattern, ordered)
    return fused


def _insert_traced_flex_gemm(
    gm: GraphModule,
    *,
    body: GraphModule,
    root: Node,
    inputs: Sequence[Node],
    expected_outputs: Sequence[torch.Tensor | None],
    cursor: Node,
    pattern: str,
    kernel: str,
    autotune: bool,
) -> _InsertedFlexGemm:
    """Insert a traced FlexGEMM body and materialize each tuple output."""
    body_values = _validate_body_outputs(body, expected_outputs, pattern)
    mark_flex_gemm_body_gemm_node(body, root.target)
    body_name = _register_body(gm, body, pattern)
    region = _CODA_INDUCTOR_REGION

    with gm.graph.inserting_after(cursor):
        body_attr = gm.graph.get_attr(body_name)
    with gm.graph.inserting_after(body_attr):
        fused = gm.graph.call_function(
            flex_gemm_hop,
            args=(
                root.target,
                body_attr,
                tuple(inputs),
                {},
                _kernel_options(
                    pattern,
                    kernel=kernel,
                    root=root,
                    autotune=autotune,
                ),
            ),
        )
    fused.meta["val"] = body_values
    _mark_inductor_region(
        body,
        body_attr,
        fused,
        pattern=pattern,
        region=region,
    )

    outputs: list[Node] = []
    output_cursor = fused
    for index, value in enumerate(body_values):
        output_cursor = _call_after(
            gm.graph,
            output_cursor,
            operator.getitem,
            (fused, index),
        )
        output_cursor.meta["val"] = value
        outputs.append(output_cursor)
    _mark_nodes_for_inductor(outputs, region=region, group=body_name)
    return _InsertedFlexGemm(tuple(outputs), body_name, region)


def _match_swiglu_backward(root: Node) -> _PointwiseBackwardMatch | None:
    if not _is_backward(root) or root.target is not _MM_TARGET:
        return None
    mul_users = [user for user in root.users if user.target is aten.mul.Tensor]
    if len(mul_users) != 2 or set(root.users) != set(mul_users):
        return None

    for gate_product, up_grad in (
        (mul_users[0], mul_users[1]),
        (mul_users[1], mul_users[0]),
    ):
        saved_gate = _other_node_input(gate_product, root)
        saved_up = _other_node_input(up_grad, root)
        gate_grad = _single_user_with_target(gate_product, aten.silu_backward.default)
        if (
            saved_gate is None
            or saved_up is None
            or gate_grad is None
            or gate_grad.args[0] is not gate_product
            or not isinstance(gate_grad.args[1], Node)
        ):
            continue
        body_nodes = frozenset((root, gate_product, up_grad, gate_grad))
        root_shape = _shape(root)
        if (
            root_shape is None
            or any(_shape(node) != root_shape for node in body_nodes)
            or any(
                _shape(node) != root_shape
                for node in (saved_gate, saved_up, gate_grad.args[1])
            )
            or not all(_is_backward(node) for node in body_nodes)
            or set(_boundary_outputs(body_nodes)) != {up_grad, gate_grad}
        ):
            continue
        return _PointwiseBackwardMatch(root, body_nodes, (up_grad, gate_grad))
    return None


def _match_situ_backward(root: Node) -> _PointwiseBackwardMatch | None:
    if not _is_backward(root) or root.target is not _MM_TARGET:
        return None
    grad_fp32 = _single_user_with_target(root, _CAST_TARGET)
    if (
        grad_fp32 is None
        or set(root.users) != {grad_fp32}
        or _cast_dtype(grad_fp32) is not torch.float32
    ):
        return None

    tanh_grad = _single_user_with_target(grad_fp32, aten.tanh_backward.default)
    sigmoid_grad = _single_user_with_target(grad_fp32, aten.sigmoid_backward.default)
    if (
        tanh_grad is None
        or sigmoid_grad is None
        or tanh_grad.args[0] is not grad_fp32
        or sigmoid_grad.args[0] is not grad_fp32
        or not isinstance(tanh_grad.args[1], Node)
        or not isinstance(sigmoid_grad.args[1], Node)
        or set(grad_fp32.users) != {tanh_grad, sigmoid_grad}
    ):
        return None

    tanh_output = _single_user_with_target(tanh_grad, _CAST_TARGET)
    sigmoid_output = _single_user_with_target(sigmoid_grad, _CAST_TARGET)
    if (
        tanh_output is None
        or sigmoid_output is None
        or set(tanh_grad.users) != {tanh_output}
        or set(sigmoid_grad.users) != {sigmoid_output}
        or _dtype(tanh_output) != _dtype(root)
        or _dtype(sigmoid_output) != _dtype(root)
    ):
        return None

    body_nodes = frozenset(
        (root, grad_fp32, tanh_grad, tanh_output, sigmoid_grad, sigmoid_output)
    )
    root_shape = _shape(root)
    if (
        root_shape is None
        or any(_shape(node) != root_shape for node in body_nodes)
        or _shape(tanh_grad.args[1]) != root_shape
        or _shape(sigmoid_grad.args[1]) != root_shape
        or not all(_is_backward(node) for node in body_nodes)
        or set(_boundary_outputs(body_nodes)) != {tanh_output, sigmoid_output}
    ):
        return None
    return _PointwiseBackwardMatch(
        root,
        body_nodes,
        (tanh_output, sigmoid_output),
    )


def _match_swiglu(output: Node) -> _SwiGLUMatch | None:
    if output.target is not aten.mul.Tensor or _is_backward(output):
        return None
    for gate_output, up_output in (
        (output.args[0], output.args[1]),
        (output.args[1], output.args[0]),
    ):
        if (
            not isinstance(gate_output, Node)
            or gate_output.target is not aten.silu.default
        ):
            continue
        gate = _chain_to_mm(gate_output.args[0], allow_cast=False)
        up = _chain_to_mm(up_output, allow_cast=False)
        if gate is None or up is None:
            continue
        if any(node.target not in _RESHAPE_TARGETS for node in gate.nodes[1:]):
            continue
        if any(node.target not in _RESHAPE_TARGETS for node in up.nodes[1:]):
            continue
        gate_input = gate.root.args[0]
        up_input = up.root.args[0]
        output_shape = _shape(output)
        if (
            gate_input is not up_input
            or output_shape is None
            or _shape(gate_output) != output_shape
            or _shape(up_output) != output_shape
        ):
            continue
        return _SwiGLUMatch(gate, gate_output, up, output)
    return None


def _scalar_input(node: object, target: Any, scalar: float) -> Node | None:
    if not isinstance(node, Node) or node.target is not target or len(node.args) < 2:
        return None
    lhs, rhs = node.args[:2]
    if isinstance(lhs, Node) and rhs == scalar:
        return lhs
    if isinstance(rhs, Node) and lhs == scalar:
        return rhs
    return None


def _match_situ(output: Node) -> _SiTUMatch | None:
    if output.target is not _CAST_TARGET or _cast_dtype(output) is not torch.bfloat16:
        return None
    product = _single_tensor_input(output)
    if product is None or product.target is not aten.mul.Tensor:
        return None
    for gate_output, up_output in (
        (product.args[0], product.args[1]),
        (product.args[1], product.args[0]),
    ):
        if (
            not isinstance(gate_output, Node)
            or gate_output.target is not aten.mul.Tensor
        ):
            continue
        gate_scaled, gate_sigmoid = gate_output.args[:2]
        if (
            not isinstance(gate_sigmoid, Node)
            or gate_sigmoid.target is not aten.sigmoid.default
        ):
            gate_scaled, gate_sigmoid = gate_sigmoid, gate_scaled
        gate_tanh = _scalar_input(gate_scaled, aten.mul.Tensor, 4.0)
        gate_div = _single_tensor_input(gate_tanh) if gate_tanh is not None else None
        gate_cast = _scalar_input(gate_div, aten.div.Tensor, 4.0)
        if (
            gate_tanh is None
            or gate_tanh.target is not aten.tanh.default
            or gate_cast is None
            or not isinstance(gate_sigmoid, Node)
            or gate_sigmoid.target is not aten.sigmoid.default
            or gate_sigmoid.args[0] is not gate_cast
            or gate_cast.target is not _CAST_TARGET
            or _cast_dtype(gate_cast) is not torch.float32
        ):
            continue
        gate = _chain_to_mm(gate_cast.args[0], allow_cast=False)

        up_tanh = _scalar_input(up_output, aten.mul.Tensor, 25.0)
        up_div = _single_tensor_input(up_tanh) if up_tanh is not None else None
        up_cast = _scalar_input(up_div, aten.div.Tensor, 25.0)
        if (
            up_tanh is None
            or up_tanh.target is not aten.tanh.default
            or up_cast is None
            or up_cast.target is not _CAST_TARGET
            or _cast_dtype(up_cast) is not torch.float32
        ):
            continue
        up = _chain_to_mm(up_cast.args[0], allow_cast=False)
        if gate is None or up is None or gate.root.args[0] is not up.root.args[0]:
            continue
        output_shape = _shape(product)
        if (
            output_shape is None
            or _shape(gate_output) != output_shape
            or _shape(up_output) != output_shape
        ):
            continue
        gate_nodes = frozenset(
            {
                *gate.nodes,
                gate_cast,
                gate_div,
                gate_tanh,
                gate_scaled,
                gate_sigmoid,
                gate_output,
            }
        )
        up_nodes = frozenset(
            {*up.nodes, up_cast, up_div, up_tanh, up_output, product, output}
        )
        return _SiTUMatch(gate, gate_nodes, up, up_nodes, output)
    return None


def _find_pointwise_user(
    node: Node, target: Any
) -> tuple[Node, tuple[Node, ...]] | None:
    frontier = deque((user, ()) for user in node.users)
    visited: set[Node] = set()
    while frontier:
        current, path = frontier.popleft()
        if current in visited:
            continue
        visited.add(current)
        if current.target is target:
            return current, path
        if current.target in _VIEW_TARGETS:
            frontier.extend((user, (*path, current)) for user in current.users)
    return None


def _cast_dtype(node: Node) -> torch.dtype | None:
    dtype = node.kwargs.get("dtype")
    return dtype if isinstance(dtype, torch.dtype) else None


def _rewrite_transposed_cast(
    gm: GraphModule,
    *,
    chain: _MatmulChain,
    cast: Node,
    pattern: str,
    autotune: bool = False,
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> bool:
    """Fuse the FP32 cast into GEMM while leaving shape-only ops outside."""
    if not _coda_nodes_available(chain.nodes):
        return False
    if not validate_partition(list(chain.nodes)):
        return False
    if benchmark_regions is not None:

        def rewrite(
            candidate: GraphModule,
            candidate_nodes: dict[str, Node],
        ) -> None:
            candidate_chain = _MatmulChain(
                candidate_nodes[chain.root.name],
                tuple(candidate_nodes[node.name] for node in chain.nodes),
            )
            if not _rewrite_transposed_cast(
                candidate,
                chain=candidate_chain,
                cast=candidate_nodes[cast.name],
                pattern=pattern,
                autotune=autotune,
            ):
                raise AssertionError(
                    f"CODA {pattern} failed to build its benchmark candidate"
                )

        benchmark_regions.append(
            _make_local_rewrite_benchmark_region(
                gm,
                chain.nodes,
                pattern,
                rewrite,
            )
        )
        return True

    existing_nodes = set(gm.graph.nodes)
    root = chain.root
    external_inputs = list(root.all_input_nodes)
    root_value = _tensor_value(root)
    if root_value is None:
        raise AssertionError(f"CODA {pattern} requires GEMM value metadata")

    def body(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor]:
        return (root.target(lhs, rhs).to(torch.float32),)

    traced_body = _trace_coda_body(body, external_inputs, pattern)

    graph_order = {node: index for index, node in enumerate(gm.graph.nodes)}
    latest_input = max(external_inputs, key=graph_order.__getitem__)
    inserted = _insert_traced_flex_gemm(
        gm,
        body=traced_body,
        root=root,
        inputs=external_inputs,
        expected_outputs=(root_value.to(torch.float32),),
        cursor=latest_input,
        pattern=pattern,
        kernel="main",
        autotune=autotune,
    )
    (extracted,) = inserted.outputs
    body_name = inserted.body_name
    region = inserted.region
    replacements = {root: extracted}
    cursor = extracted
    for old in chain.nodes[1:]:
        if old is cast:
            break
        with gm.graph.inserting_after(cursor):
            copied = gm.graph.node_copy(old, lambda arg: replacements.get(arg, arg))
        _copy_meta(copied, old)
        old_shape = _shape(old)
        if old_shape is not None:
            _set_empty_meta(copied, root, old_shape, dtype=torch.float32)
        replacements[old] = copied
        cursor = copied
    _copy_meta(cursor, cast)
    cursor.meta.setdefault("custom", {})["coda_pattern"] = pattern
    cast.replace_all_uses_with(cursor)
    _mark_new_nodes_for_inductor(gm, existing_nodes, region=region, group=body_name)
    _mark_coda_owned(pattern, chain.nodes)
    return True


def _rmsnorm_getitems(node: Node) -> tuple[Node, Node | None] | None:
    by_index: dict[int, Node] = {}
    for user in node.users:
        if user.target is operator.getitem and len(user.args) >= 2:
            index = user.args[1]
            if not isinstance(index, int) or index not in {0, 1} or index in by_index:
                return None
            by_index[index] = user
    if 0 not in by_index:
        return None
    return by_index[0], by_index.get(1)


def _valid_forward_rmsnorm(node: Node) -> bool:
    if node.target is not aten._fused_rms_norm.default or len(node.args) < 4:
        return False
    norm_input, normalized_shape, weight, eps = node.args[:4]
    if (
        not isinstance(norm_input, Node)
        or not isinstance(weight, Node)
        or not isinstance(eps, (float, int))
        or not isinstance(normalized_shape, (list, tuple))
        or _rmsnorm_getitems(node) is None
    ):
        return False
    input_shape = _shape(norm_input)
    return (
        input_shape is not None
        and list(normalized_shape) == [input_shape[-1]]
        and _shape(weight) == (input_shape[-1],)
    )


def _valid_backward_rmsnorm(node: Node) -> bool:
    if node.target is not aten._fused_rms_norm_backward.default or len(node.args) < 6:
        return False
    grad, norm_input, normalized_shape, rstd, weight, output_mask = node.args[:6]
    if (
        not all(isinstance(arg, Node) for arg in (grad, norm_input, rstd, weight))
        or not isinstance(normalized_shape, (list, tuple))
        or not isinstance(output_mask, (list, tuple))
        or list(output_mask) != [True, True]
    ):
        return False
    grad_shape = _shape(grad)
    input_shape = _shape(norm_input)
    outputs = _rmsnorm_getitems(node)
    return (
        grad_shape is not None
        and input_shape == grad_shape
        and list(normalized_shape) == [grad_shape[-1]]
        and _shape(weight) == (grad_shape[-1],)
        and _shape(rstd) == (*grad_shape[:-1], 1)
        and outputs is not None
        and outputs[1] is not None
    )


def _residual_rmsnorm_path(norm: Node) -> _MatmulChain | None:
    if (
        _is_backward(norm)
        or not _valid_forward_rmsnorm(norm)
        or norm.meta.get("coda_consumed")
    ):
        return None
    hidden = norm.args[0]
    if not isinstance(hidden, Node):
        return None
    path = _find_mm_path(hidden)
    if (
        path is None
        or not _is_reshape_only_path(path)
        or not _has_exact_add_shapes(path)
        or not _coda_nodes_available((*path.nodes, norm))
    ):
        return None
    path_set = set(path.nodes)
    residuals = [
        input_node
        for path_node in path.nodes
        if path_node.target is aten.add.Tensor
        for input_node in path_node.all_input_nodes
        if input_node not in path_set
    ]
    if not residuals:
        return None
    return path


def _weighted_residual_rmsnorm_path(norm: Node) -> _MatmulChain | None:
    if (
        _is_backward(norm)
        or not _valid_forward_rmsnorm(norm)
        or norm.meta.get("coda_consumed")
    ):
        return None
    hidden = norm.args[0]
    if not isinstance(hidden, Node) or hidden.target is not _CAST_TARGET:
        return None
    squeeze = _single_tensor_input(hidden)
    if (
        squeeze is None
        or squeeze.target is not aten.squeeze.dim
        or len(squeeze.args) < 2
        or squeeze.args[1] not in {1, -2}
    ):
        return None
    bmm = _single_tensor_input(squeeze)
    if bmm is None or bmm.target is not _BMM_TARGET or len(bmm.args) < 2:
        return None
    norm_outputs = _rmsnorm_getitems(norm)
    if norm_outputs is None:
        return None
    expected_norm_users = {norm_outputs[0]}
    if norm_outputs[1] is not None:
        expected_norm_users.add(norm_outputs[1])
    if set(norm.users) != expected_norm_users:
        return None
    lhs, rhs = bmm.args[:2]
    lhs_shape = _shape(lhs) if isinstance(lhs, Node) else None
    rhs_shape = _shape(rhs) if isinstance(rhs, Node) else None
    bmm_shape = _shape(bmm)
    hidden_shape = _shape(hidden)
    if (
        lhs_shape is None
        or rhs_shape is None
        or bmm_shape is None
        or hidden_shape is None
        or len(lhs_shape) != 3
        or len(rhs_shape) != 3
        or lhs_shape[1] != 1
        or lhs_shape[0] != rhs_shape[0]
        or lhs_shape[2] != rhs_shape[1]
        or bmm_shape != (lhs_shape[0], 1, rhs_shape[2])
        or hidden_shape != (bmm_shape[0], bmm_shape[2])
        or set(bmm.users) != {squeeze}
        or set(squeeze.users) != {hidden}
        or not _coda_nodes_available((bmm, squeeze, hidden, norm))
    ):
        return None
    return _MatmulChain(bmm, (bmm, squeeze, hidden))


def _downstream_mm(node: Node, *, backward: bool | None = None) -> _MatmulUse | None:
    frontier = deque((user, ()) for user in node.users)
    visited: set[Node] = set()
    matches: list[_MatmulUse] = []
    while frontier:
        user, wrappers = frontier.popleft()
        if user in visited:
            continue
        visited.add(user)
        if user.target is _MM_TARGET:
            if backward is not None and _is_backward(user) is not backward:
                continue
            operands = [index for index, arg in enumerate(user.args[:2]) if arg is node]
            if wrappers:
                operands = [
                    index
                    for index, arg in enumerate(user.args[:2])
                    if arg is wrappers[-1]
                ]
            if len(operands) == 1:
                matches.append(_MatmulUse(user, operands[0], wrappers))
            continue
        if user.target in _RESHAPE_TARGETS:
            frontier.extend(
                (downstream, (*wrappers, user)) for downstream in user.users
            )
    return matches[0] if len(matches) == 1 else None


def _valid_projection_rmsnorm_use(
    norm: Node,
    first_path: _MatmulChain,
    second_use: _MatmulUse,
    full_output: Node,
) -> bool:
    getitems = _rmsnorm_getitems(norm)
    if getitems is None or second_use.operand != 0:
        return False
    norm_input = norm.args[0]
    second_mm = second_use.node
    second_lhs, second_rhs = second_mm.args[:2]
    if not all(isinstance(node, Node) for node in (norm_input, second_lhs, second_rhs)):
        return False
    root_shape = _shape(first_path.root)
    full_shape = _shape(full_output)
    norm_shape = _shape(norm_input)
    norm_output_shape = _shape(getitems[0])
    lhs_shape = _shape(second_lhs)
    rhs_shape = _shape(second_rhs)
    output_shape = _shape(second_mm)
    if not all(
        shape is not None
        for shape in (
            root_shape,
            full_shape,
            norm_shape,
            norm_output_shape,
            lhs_shape,
            rhs_shape,
            output_shape,
        )
    ):
        return False
    if not all(
        len(shape) == 2 for shape in (root_shape, lhs_shape, rhs_shape, output_shape)
    ):
        return False
    return bool(
        _same_numel(root_shape, full_shape)
        and norm_output_shape == norm_shape
        and _same_numel(norm_shape, lhs_shape)
        and lhs_shape[-1] == rhs_shape[0]
        and output_shape == (lhs_shape[0], rhs_shape[1])
    )


def _trace_coda_values(
    function: Callable[..., Any],
    values: Sequence[torch.Tensor],
    pattern: str,
) -> GraphModule:
    """Trace tensor code after normalizing inputs to one FakeTensor mode."""
    fake_modes = {value.fake_mode for value in values if isinstance(value, FakeTensor)}
    if len(fake_modes) == 1 and all(isinstance(value, FakeTensor) for value in values):
        return _trace_replacement(function, values)

    fake_mode = FakeTensorMode()
    fake_values = tuple(
        fake_mode.fake_tensor_converter.from_meta_and_device(
            fake_mode,
            torch.empty_strided(
                value.shape,
                value.stride(),
                dtype=value.dtype,
                device="meta",
            ),
            value.device,
        )
        for value in values
    )
    return _trace_replacement(function, fake_values)


def _trace_coda_body(
    function: Callable[..., Any],
    inputs: Sequence[Node],
    pattern: str,
) -> GraphModule:
    """Trace readable tensor code from FX node metadata."""
    values = tuple(_tensor_value(node) for node in inputs)
    if not all(isinstance(value, torch.Tensor) for value in values):
        missing = [
            node.name
            for node, value in zip(inputs, values, strict=True)
            if not isinstance(value, torch.Tensor)
        ]
        raise AssertionError(
            f"CODA {pattern} body inputs require tensor metadata: {missing}"
        )
    return _trace_coda_values(function, values, pattern)


def _trace_forward_rmsnorm_body(
    path: _MatmulChain,
    ordered: Sequence[Node],
    input_pairs: Sequence[tuple[Node, Node]],
    *,
    root_shape: Sequence[Any],
    width: int,
    pattern: str,
) -> GraphModule:
    logical_inputs = tuple(logical for logical, _ in input_pairs)

    def body(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        env = dict(zip(logical_inputs, args, strict=True))
        hidden = path.root.target(env[path.root.args[0]], env[path.root.args[1]])
        previous = path.root
        for node in ordered[1:]:
            if node.target in _VIEW_TARGETS or node.target is aten.squeeze.dim:
                previous = node
                continue
            if node.target is _CAST_TARGET:
                hidden = _CAST_TARGET(hidden, **node.kwargs)
                previous = node
                continue
            if node.target is not aten.add.Tensor:
                raise AssertionError(
                    f"CODA {pattern} found unsupported residual-path op {node.target}"
                )
            if node.args[0] is previous:
                residual = node.args[1]
                lhs_is_accumulator = True
            elif node.args[1] is previous:
                residual = node.args[0]
                lhs_is_accumulator = False
            else:
                raise AssertionError(f"CODA {pattern} lost the residual add path")
            if not isinstance(residual, Node):
                raise AssertionError(f"CODA {pattern} expected a tensor residual")
            add_args = (
                (hidden, env[residual])
                if lhs_is_accumulator
                else (env[residual], hidden)
            )
            hidden = aten.add.Tensor(*add_args, *node.args[2:], **node.kwargs)
            previous = node

        hidden_fp32 = hidden.to(torch.float32)
        grouped = hidden_fp32.view(
            *root_shape[:-1],
            width // RMSNORM_FORWARD_GROUP,
            RMSNORM_FORWARD_GROUP,
        )
        return hidden, (grouped * grouped).mean(dim=-1)

    return _trace_coda_body(
        body,
        tuple(physical for _, physical in input_pairs),
        pattern,
    )


def _trace_backward_rmsnorm_body(
    path: _MatmulChain,
    input_pairs: Sequence[tuple[Node, Node]],
    *,
    norm_input: Node,
    rstd: Node,
    weight: Node,
    root_shape: Sequence[Any],
    width: int,
    pattern: str,
) -> GraphModule:
    logical_inputs = tuple(logical for logical, _ in input_pairs)

    def body(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        env = dict(zip(logical_inputs, args, strict=True))
        grad = path.root.target(env[path.root.args[0]], env[path.root.args[1]])
        grad_fp32 = grad.to(torch.float32)
        x_hat = env[norm_input].to(torch.float32) * env[rstd]
        grad_x_hat = grad_fp32 * env[weight].to(torch.float32)
        dot = x_hat * grad_x_hat
        grouped = dot.view(
            root_shape[0],
            width // RMSNORM_BACKWARD_GROUP,
            RMSNORM_BACKWARD_GROUP,
        )
        return grad, grouped.sum(dim=-1)

    return _trace_coda_body(
        body,
        tuple(physical for _, physical in input_pairs),
        pattern,
    )


def _emit_forward_rmsnorm(
    gm: GraphModule,
    *,
    norm_input: Node,
    partial: Node,
    weight: Node,
    eps: object,
    cursor: Node,
    physical_meta: Node,
    physical_prefix_shape: Sequence[Any],
    logical_rstd_shape: Sequence[Any],
    output_dtype: torch.dtype,
    output_meta: Node,
    rstd_meta: Node | None,
) -> _ForwardRMSNormOutputs:
    """Finish RMSNorm from row-wise partial mean squares."""
    partial_mean = _call_after(
        gm.graph,
        cursor,
        aten.mean.dim,
        (partial, [-1], True),
    )
    variance = _call_after(
        gm.graph,
        partial_mean,
        aten.add.Scalar,
        (partial_mean, eps),
    )
    physical_rstd = _call_after(
        gm.graph,
        variance,
        aten.rsqrt.default,
        (variance,),
    )
    _set_empty_meta(
        physical_rstd,
        physical_meta,
        [*physical_prefix_shape, 1],
        dtype=torch.float32,
    )
    rstd = _call_after(
        gm.graph,
        physical_rstd,
        aten.view.default,
        (physical_rstd, list(logical_rstd_shape)),
    )
    input_fp32 = _call_after(
        gm.graph,
        rstd,
        _CAST_TARGET,
        (norm_input,),
        {"dtype": torch.float32},
    )
    weight_fp32 = _call_after(
        gm.graph,
        input_fp32,
        _CAST_TARGET,
        (weight,),
        {"dtype": torch.float32},
    )
    normalized = _call_after(
        gm.graph,
        weight_fp32,
        aten.mul.Tensor,
        (input_fp32, rstd),
    )
    weighted = _call_after(
        gm.graph,
        normalized,
        aten.mul.Tensor,
        (normalized, weight_fp32),
    )
    output = _call_after(
        gm.graph,
        weighted,
        _CAST_TARGET,
        (weighted,),
        {"dtype": output_dtype},
    )
    _copy_meta(output, output_meta)
    if rstd_meta is not None:
        _copy_meta(rstd, rstd_meta)
    return _ForwardRMSNormOutputs(output, rstd, physical_rstd)


def _emit_backward_rmsnorm(
    gm: GraphModule,
    *,
    grad_2d: Node,
    partial: Node,
    grad: Node,
    norm_input: Node,
    rstd: Node,
    weight: Node,
    grad_shape: Sequence[Any],
    width: int,
    input_grad_dtype: torch.dtype,
    weight_grad_dtype: torch.dtype,
    input_grad_meta: Node,
    weight_grad_meta: Node,
    pattern: str,
) -> _BackwardRMSNormOutputs:
    """Finish RMSNorm backward from row-wise partial dot products."""
    new_grad = _call_after(
        gm.graph,
        partial,
        aten.view.default,
        (grad_2d, list(grad_shape)),
    )
    _copy_meta(new_grad, grad)
    grad_fp32 = _call_after(
        gm.graph,
        new_grad,
        _CAST_TARGET,
        (new_grad,),
        {"dtype": torch.float32},
    )
    input_fp32 = _call_after(
        gm.graph,
        grad_fp32,
        _CAST_TARGET,
        (norm_input,),
        {"dtype": torch.float32},
    )
    x_hat = _call_after(gm.graph, input_fp32, aten.mul.Tensor, (input_fp32, rstd))
    weight_fp32 = _call_after(
        gm.graph,
        x_hat,
        _CAST_TARGET,
        (weight,),
        {"dtype": torch.float32},
    )
    grad_x_hat = _call_after(
        gm.graph,
        weight_fp32,
        aten.mul.Tensor,
        (grad_fp32, weight_fp32),
    )
    row_dot_2d = _call_after(
        gm.graph,
        grad_x_hat,
        aten.sum.dim_IntList,
        (partial, [-1], True),
    )
    rstd_shape = _shape(rstd)
    if rstd_shape is None:
        raise AssertionError(f"CODA {pattern} requires RMSNorm rstd shape metadata")
    row_dot = _call_after(
        gm.graph,
        row_dot_2d,
        aten.view.default,
        (row_dot_2d, list(rstd_shape)),
    )
    scaled_x = _call_after(gm.graph, row_dot, aten.div.Scalar, (x_hat, width))
    correction = _call_after(gm.graph, scaled_x, aten.mul.Tensor, (scaled_x, row_dot))
    centered = _call_after(
        gm.graph,
        correction,
        aten.sub.Tensor,
        (grad_x_hat, correction),
    )
    input_grad_fp32 = _call_after(
        gm.graph,
        centered,
        aten.mul.Tensor,
        (centered, rstd),
    )
    input_grad = _call_after(
        gm.graph,
        input_grad_fp32,
        _CAST_TARGET,
        (input_grad_fp32,),
        {"dtype": input_grad_dtype},
    )
    weight_grad_terms = _call_after(
        gm.graph,
        input_grad,
        aten.mul.Tensor,
        (grad_fp32, x_hat),
    )
    weight_grad_fp32 = _call_after(
        gm.graph,
        weight_grad_terms,
        aten.sum.dim_IntList,
        (weight_grad_terms, list(range(len(grad_shape) - 1)), False),
    )
    weight_grad = _call_after(
        gm.graph,
        weight_grad_fp32,
        _CAST_TARGET,
        (weight_grad_fp32,),
        {"dtype": weight_grad_dtype},
    )
    _copy_meta(input_grad, input_grad_meta)
    _copy_meta(weight_grad, weight_grad_meta)
    return _BackwardRMSNormOutputs(input_grad, weight_grad)


def _rewrite_projection_rmsnorm(
    gm: GraphModule,
    *,
    norm: Node,
    first_path: _MatmulChain,
    second_use: _MatmulUse,
    full_output: Node,
    projection_body: Callable[..., tuple[torch.Tensor, ...]],
    expansion_body: Callable[..., tuple[torch.Tensor]],
    replacement: Callable[..., tuple[torch.Tensor, ...]],
    pattern: str,
    autotune: bool = False,
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    getitems = _rmsnorm_getitems(norm)
    if getitems is None:
        raise AssertionError(f"CODA {pattern} expected RMSNorm tuple outputs")
    old_norm_out, old_rstd = getitems
    second_mm = second_use.node
    if second_use.operand != 0:
        raise AssertionError(f"CODA {pattern} requires RMSNorm as GEMM operand 0")
    norm_input = norm.args[0]
    weight = norm.args[2]
    if not isinstance(norm_input, Node) or not isinstance(weight, Node):
        raise AssertionError(f"CODA {pattern} expected tensor RMSNorm inputs")

    if benchmark_regions is not None:
        bridge_nodes = []
        bridge = norm_input
        while bridge is not full_output:
            bridge_nodes.append(bridge)
            parent = _single_tensor_input(bridge)
            if parent is None:
                raise AssertionError(
                    f"CODA {pattern} cannot isolate the projection RMSNorm path"
                )
            bridge = parent
        region_nodes = (
            *first_path.nodes,
            *bridge_nodes,
            norm,
            old_norm_out,
            *((old_rstd,) if old_rstd is not None else ()),
            *second_use.wrappers,
            second_mm,
        )

        def rewrite_candidate(
            candidate: GraphModule,
            candidate_nodes: dict[str, Node],
        ) -> None:
            _rewrite_projection_rmsnorm(
                candidate,
                norm=candidate_nodes[norm.name],
                first_path=_MatmulChain(
                    candidate_nodes[first_path.root.name],
                    tuple(candidate_nodes[node.name] for node in first_path.nodes),
                ),
                second_use=_MatmulUse(
                    candidate_nodes[second_use.node.name],
                    second_use.operand,
                    tuple(candidate_nodes[node.name] for node in second_use.wrappers),
                ),
                full_output=candidate_nodes[full_output.name],
                projection_body=projection_body,
                expansion_body=expansion_body,
                replacement=replacement,
                pattern=pattern,
                autotune=autotune,
            )

        benchmark_regions.append(
            _make_local_rewrite_benchmark_region(
                gm,
                region_nodes,
                pattern,
                rewrite_candidate,
            )
        )
        return

    second_lhs = second_mm.args[0]
    second_rhs = second_mm.args[1]
    if not isinstance(second_lhs, Node) or not isinstance(second_rhs, Node):
        raise AssertionError(f"CODA {pattern} expected tensor expansion GEMM inputs")

    root_shape = _shape(first_path.root)
    full_shape = _shape(full_output)
    norm_shape = _shape(norm_input)
    expansion_lhs_shape = _shape(second_lhs)
    expansion_output_shape = _shape(second_mm)
    norm_output_dtype = _dtype(old_norm_out)
    expansion_output_dtype = _dtype(second_mm)
    if (
        root_shape is None
        or full_shape is None
        or norm_shape is None
        or expansion_lhs_shape is None
        or expansion_output_shape is None
        or norm_output_dtype is None
        or expansion_output_dtype is None
    ):
        raise AssertionError(f"CODA {pattern} requires projection tensor metadata")
    full_width = full_shape[-1]
    norm_width = norm_shape[-1]
    if not isinstance(full_width, int) or not isinstance(norm_width, int):
        raise AssertionError(
            f"CODA {pattern} unsupported widths full={full_width}, norm={norm_width}"
        )
    reduction_group = gcd(RMSNORM_FORWARD_GROUP, full_width, norm_width)
    rstd_shape = _shape(old_rstd) if old_rstd is not None else (*norm_shape[:-1], 1)

    graph_order = {node: index for index, node in enumerate(gm.graph.nodes)}
    cursor = max(
        [*first_path.root.all_input_nodes, weight],
        key=graph_order.__getitem__,
    )
    padded_weight = weight
    if full_width != norm_width:
        padded_weight = _call_after(
            gm.graph,
            cursor,
            aten.constant_pad_nd.default,
            (weight, [0, full_width - norm_width], 1.0),
        )
        padded_value = _tensor_value(weight)
        if padded_value is not None:
            padded_weight.meta["val"] = aten.constant_pad_nd.default(
                padded_value,
                [0, full_width - norm_width],
                1.0,
            )
        cursor = padded_weight
    weight_row = _call_after(
        gm.graph,
        cursor,
        aten.view.default,
        (padded_weight, [1, full_width]),
    )
    _set_view_meta(weight_row, padded_weight, [1, full_width])
    weight_row_fp32 = _call_after(
        gm.graph,
        weight_row,
        _CAST_TARGET,
        (weight_row,),
        {"dtype": torch.float32},
    )
    _set_empty_meta(
        weight_row_fp32,
        weight_row,
        [1, full_width],
        dtype=torch.float32,
    )

    def configured_projection_body(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        weight_row: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return projection_body(
            lhs,
            rhs,
            weight_row,
            output_dtype=norm_output_dtype,
            root_shape=root_shape,
            width=full_width,
            reduction_group=reduction_group,
        )

    traced_projection_body = _trace_coda_body(
        configured_projection_body,
        (*first_path.root.all_input_nodes, weight_row_fp32),
        pattern,
    )
    mark_flex_gemm_body_gemm_node(traced_projection_body, first_path.root.target)

    def configured_expansion_body(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        rstd: torch.Tensor,
    ) -> tuple[torch.Tensor]:
        return expansion_body(
            lhs,
            rhs,
            rstd,
            rows=root_shape[0],
            output_shape=expansion_output_shape,
            output_dtype=expansion_output_dtype,
        )

    expansion_lhs_value = _tensor_value(second_lhs)
    expansion_rhs_value = _tensor_value(second_rhs)
    root_value = _tensor_value(first_path.root)
    full_value = _tensor_value(full_output)
    norm_output_value = _tensor_value(old_norm_out)
    expansion_output_value = _tensor_value(second_mm)
    if not all(
        isinstance(value, torch.Tensor)
        for value in (
            expansion_lhs_value,
            expansion_rhs_value,
            root_value,
            full_value,
            norm_output_value,
            expansion_output_value,
        )
    ):
        raise AssertionError(f"CODA {pattern} requires projection tensor values")
    physical_rstd_value = root_value.new_empty(
        [root_shape[0], 1],
        dtype=torch.float32,
    )
    logical_rstd_value = root_value.new_empty(rstd_shape, dtype=torch.float32)
    traced_expansion_body = _trace_coda_values(
        configured_expansion_body,
        (expansion_lhs_value, expansion_rhs_value, physical_rstd_value),
        pattern,
    )
    mark_flex_gemm_body_gemm_node(traced_expansion_body, second_mm.target)

    projection_options = _kernel_options(
        pattern,
        kernel="projection",
        root=first_path.root,
        autotune=autotune,
    )
    expansion_options = _kernel_options(
        pattern,
        kernel="expansion",
        root=second_mm,
        autotune=autotune,
    )

    def configured_replacement(
        projection_lhs: torch.Tensor,
        projection_rhs: torch.Tensor,
        weight_row: torch.Tensor,
        expansion_rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return replacement(
            projection_lhs,
            projection_rhs,
            weight_row,
            expansion_rhs,
            projection_body=traced_projection_body,
            expansion_body=traced_expansion_body,
            projection_kernel_options=projection_options,
            expansion_kernel_options=expansion_options,
            full_shape=full_shape,
            norm_shape=norm_shape,
            expansion_lhs_shape=expansion_lhs_shape,
            expansion_output_shape=expansion_output_shape,
            rstd_shape=rstd_shape,
            eps=norm.args[3],
            norm_output_dtype=norm_output_dtype,
            expansion_output_dtype=expansion_output_dtype,
            reduction_group=reduction_group,
        )

    replacement_inputs = (
        *first_path.root.all_input_nodes,
        weight_row_fp32,
        second_rhs,
    )
    replacement_values = tuple(_tensor_value(node) for node in replacement_inputs)
    if not all(isinstance(value, torch.Tensor) for value in replacement_values):
        raise AssertionError(f"CODA {pattern} replacement inputs require metadata")
    replacement_graph = _trace_coda_values(
        configured_replacement,
        replacement_values,
        pattern,
    )
    _validate_body_outputs(
        replacement_graph,
        (
            full_value,
            norm_output_value,
            logical_rstd_value,
            expansion_output_value,
        ),
        pattern,
    )
    graph_order = {node: index for index, node in enumerate(gm.graph.nodes)}
    cursor = max(replacement_inputs, key=graph_order.__getitem__)
    new_full, new_norm_out, new_rstd, new_expansion = _inline_traced_coda_replacement(
        gm,
        replacement_graph,
        replacement_inputs,
        cursor,
        pattern=pattern,
    )
    _copy_meta(new_full, full_output)
    _copy_meta(new_norm_out, old_norm_out)
    if old_rstd is not None:
        _copy_meta(new_rstd, old_rstd)
    _copy_meta(new_expansion, second_mm)

    full_output.replace_all_uses_with(new_full)
    old_norm_out.replace_all_uses_with(new_norm_out)
    if old_rstd is not None:
        old_rstd.replace_all_uses_with(new_rstd)
    second_mm.replace_all_uses_with(new_expansion)
    _mark_coda_owned(
        pattern,
        (*first_path.nodes, norm, old_norm_out, second_mm),
    )
    if old_rstd is not None:
        _mark_coda_owned(pattern, (old_rstd,))


def _rewrite_forward_rmsnorm(
    gm: GraphModule,
    *,
    norm: Node,
    path: _MatmulChain,
    pattern: str,
    autotune: bool = False,
    kernel: str = "main",
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    existing_nodes = set(gm.graph.nodes)
    hidden = norm.args[0]
    weight = norm.args[2]
    if not isinstance(hidden, Node) or not isinstance(weight, Node):
        raise AssertionError(f"CODA {pattern} expected tensor RMSNorm inputs")
    getitems = _rmsnorm_getitems(norm)
    if getitems is None:
        raise AssertionError(f"CODA {pattern} expected RMSNorm tuple outputs")
    norm_out, old_rstd = getitems
    if benchmark_regions is not None:
        region_nodes = (
            *path.nodes,
            norm,
            norm_out,
            *((old_rstd,) if old_rstd is not None else ()),
        )

        def rewrite(
            candidate: GraphModule,
            candidate_nodes: dict[str, Node],
        ) -> None:
            _rewrite_forward_rmsnorm(
                candidate,
                norm=candidate_nodes[norm.name],
                path=_MatmulChain(
                    candidate_nodes[path.root.name],
                    tuple(candidate_nodes[node.name] for node in path.nodes),
                ),
                pattern=pattern,
                autotune=autotune,
                kernel=kernel,
            )

        benchmark_regions.append(
            _make_local_rewrite_benchmark_region(
                gm,
                region_nodes,
                pattern,
                rewrite,
            )
        )
        return
    norm_output_dtype = _dtype(norm_out)
    if norm_output_dtype is None:
        raise AssertionError(f"CODA {pattern} requires output dtype metadata")
    hidden_shape = _shape(hidden)
    if hidden_shape is None:
        raise AssertionError(f"CODA {pattern} requires shape metadata")
    width = hidden_shape[-1]
    if not isinstance(width, int) or width % RMSNORM_FORWARD_GROUP:
        raise AssertionError(f"CODA {pattern} unsupported RMSNorm width {width}")

    ordered = _ordered_nodes(gm, path.nodes)
    body_set = set(ordered)
    external_inputs: list[Node] = []
    for arg in path.root.all_input_nodes:
        if arg not in external_inputs:
            external_inputs.append(arg)
    for body_node in ordered:
        for arg in body_node.all_input_nodes:
            if arg not in body_set and arg not in external_inputs:
                external_inputs.append(arg)

    root_shape = _shape(path.root)
    if root_shape is None:
        raise AssertionError(f"CODA {pattern} requires GEMM shape metadata")
    root_inputs = set(path.root.all_input_nodes)
    body_input_pairs = [
        (
            external,
            (
                external
                if external in root_inputs
                else _view_region_input(gm, external, root_shape, existing_nodes)
            ),
        )
        for external in external_inputs
    ]
    graph_order = {node: index for index, node in enumerate(gm.graph.nodes)}
    cursor = max(
        [*(actual for _, actual in body_input_pairs), weight],
        key=graph_order.__getitem__,
    )

    gemm_target = path.root.target
    if gemm_target not in {_MM_TARGET, _BMM_TARGET}:
        raise AssertionError(f"CODA {pattern} expected MM or BMM root")
    body = _trace_forward_rmsnorm_body(
        path,
        ordered,
        body_input_pairs,
        root_shape=root_shape,
        width=width,
        pattern=pattern,
    )
    root_value = _tensor_value(path.root)
    if root_value is None:
        raise AssertionError(f"CODA {pattern} requires GEMM value metadata")
    hidden_dtype = _dtype(hidden)
    if hidden_dtype is None:
        raise AssertionError(f"CODA {pattern} requires RMSNorm input dtype metadata")
    hidden_value = root_value.new_empty(root_shape, dtype=hidden_dtype)
    partial_shape = [*root_shape[:-1], width // RMSNORM_FORWARD_GROUP]
    partial_value = root_value.new_empty(
        partial_shape,
        dtype=torch.float32,
    )
    inserted = _insert_traced_flex_gemm(
        gm,
        body=body,
        root=path.root,
        inputs=tuple(actual for _, actual in body_input_pairs),
        expected_outputs=(hidden_value, partial_value),
        cursor=cursor,
        pattern=pattern,
        kernel=kernel,
        autotune=autotune,
    )
    hidden_physical, partial_out = inserted.outputs
    body_name = inserted.body_name
    region = inserted.region
    _set_empty_meta(hidden_physical, path.root, root_shape, dtype=hidden_dtype)
    _set_empty_meta(
        partial_out,
        path.root,
        partial_shape,
        dtype=torch.float32,
    )
    new_hidden = _call_after(
        gm.graph,
        partial_out,
        aten.view.default,
        (hidden_physical, list(hidden_shape)),
    )
    _copy_meta(new_hidden, hidden)
    rstd_shape = _shape(old_rstd) if old_rstd is not None else (*hidden_shape[:-1], 1)
    rmsnorm = _emit_forward_rmsnorm(
        gm,
        norm_input=new_hidden,
        partial=partial_out,
        weight=weight,
        eps=norm.args[3],
        cursor=new_hidden,
        physical_meta=path.root,
        physical_prefix_shape=root_shape[:-1],
        logical_rstd_shape=rstd_shape,
        output_dtype=norm_output_dtype,
        output_meta=norm_out,
        rstd_meta=old_rstd,
    )
    new_norm_out = rmsnorm.output
    new_rstd = rmsnorm.rstd

    body_nodes = set(ordered)
    for user in list(hidden.users):
        if user not in body_nodes and user is not norm:
            user.replace_input_with(hidden, new_hidden)
    norm_out.replace_all_uses_with(new_norm_out)
    if old_rstd is not None:
        old_rstd.replace_all_uses_with(new_rstd)
    _mark_new_nodes_for_inductor(gm, existing_nodes, region=region, group=body_name)
    _mark_coda_owned(pattern, (*path.nodes, norm, norm_out))
    if old_rstd is not None:
        _mark_coda_owned(pattern, (old_rstd,))


def _rewrite_backward_rmsnorm(
    gm: GraphModule,
    *,
    norm_backward: Node,
    path: _MatmulChain,
    pattern: str,
    autotune: bool = False,
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> None:
    existing_nodes = set(gm.graph.nodes)
    grad = norm_backward.args[0]
    norm_input = norm_backward.args[1]
    rstd = norm_backward.args[3]
    weight = norm_backward.args[4]
    if not all(isinstance(arg, Node) for arg in (grad, norm_input, rstd, weight)):
        raise AssertionError(f"CODA {pattern} expected tensor RMSNorm inputs")
    outputs = _rmsnorm_getitems(norm_backward)
    if outputs is None:
        raise AssertionError(f"CODA {pattern} expected RMSNorm backward outputs")
    old_grad_input, old_grad_weight = outputs
    if old_grad_weight is None:
        raise AssertionError(f"CODA {pattern} expected RMSNorm weight gradient")
    if benchmark_regions is not None:
        region_nodes = (
            *path.nodes,
            norm_backward,
            old_grad_input,
            old_grad_weight,
        )

        def rewrite(
            candidate: GraphModule,
            candidate_nodes: dict[str, Node],
        ) -> None:
            _rewrite_backward_rmsnorm(
                candidate,
                norm_backward=candidate_nodes[norm_backward.name],
                path=_MatmulChain(
                    candidate_nodes[path.root.name],
                    tuple(candidate_nodes[node.name] for node in path.nodes),
                ),
                pattern=pattern,
                autotune=autotune,
            )

        benchmark_regions.append(
            _make_local_rewrite_benchmark_region(
                gm,
                region_nodes,
                pattern,
                rewrite,
            )
        )
        return
    grad_input_dtype = _dtype(old_grad_input)
    grad_weight_dtype = _dtype(old_grad_weight)
    if grad_input_dtype is None or grad_weight_dtype is None:
        raise AssertionError(f"CODA {pattern} requires output dtype metadata")
    grad_shape = _shape(grad)
    if grad_shape is None:
        raise AssertionError(f"CODA {pattern} requires shape metadata")
    width = grad_shape[-1]
    if not isinstance(width, int) or width % RMSNORM_BACKWARD_GROUP:
        raise AssertionError(f"CODA {pattern} unsupported RMSNorm width {width}")

    ordered = _ordered_nodes(gm, path.nodes)
    body_set = set(ordered)
    external_inputs: list[Node] = []
    for arg in path.root.all_input_nodes:
        if arg not in external_inputs:
            external_inputs.append(arg)
    for external in (norm_input, rstd, weight):
        if external not in external_inputs:
            external_inputs.append(external)
    for body_node in ordered:
        for arg in body_node.all_input_nodes:
            if arg not in body_set and arg not in external_inputs:
                external_inputs.append(arg)

    root_shape = _shape(path.root)
    if root_shape is None:
        raise AssertionError(f"CODA {pattern} requires GEMM shape metadata")
    root_inputs = set(path.root.all_input_nodes)
    desired_shapes = {
        norm_input: root_shape,
        rstd: (root_shape[0], 1),
        weight: (1, root_shape[-1]),
    }
    body_input_pairs = [
        (
            external,
            (
                external
                if external in root_inputs
                else _view_region_input(
                    gm,
                    external,
                    desired_shapes.get(external, root_shape),
                    existing_nodes,
                )
            ),
        )
        for external in external_inputs
    ]
    graph_order = {node: index for index, node in enumerate(gm.graph.nodes)}
    cursor = max(
        (actual for _, actual in body_input_pairs),
        key=graph_order.__getitem__,
    )

    body = _trace_backward_rmsnorm_body(
        path,
        body_input_pairs,
        norm_input=norm_input,
        rstd=rstd,
        weight=weight,
        root_shape=root_shape,
        width=width,
        pattern=pattern,
    )
    root_value = _tensor_value(path.root)
    if root_value is None:
        raise AssertionError(f"CODA {pattern} requires GEMM value metadata")
    partial_value = root_value.new_empty(
        [root_shape[0], width // RMSNORM_BACKWARD_GROUP],
        dtype=torch.float32,
    )
    inserted = _insert_traced_flex_gemm(
        gm,
        body=body,
        root=path.root,
        inputs=tuple(actual for _, actual in body_input_pairs),
        expected_outputs=(root_value, partial_value),
        cursor=cursor,
        pattern=pattern,
        kernel="main",
        autotune=autotune,
    )
    grad_2d, partial_out = inserted.outputs
    body_name = inserted.body_name
    region = inserted.region
    _copy_meta(grad_2d, path.root)
    _set_empty_meta(
        partial_out,
        path.root,
        [root_shape[0], width // RMSNORM_BACKWARD_GROUP],
        dtype=torch.float32,
    )
    rmsnorm = _emit_backward_rmsnorm(
        gm,
        grad_2d=grad_2d,
        partial=partial_out,
        grad=grad,
        norm_input=norm_input,
        rstd=rstd,
        weight=weight,
        grad_shape=grad_shape,
        width=width,
        input_grad_dtype=grad_input_dtype,
        weight_grad_dtype=grad_weight_dtype,
        input_grad_meta=old_grad_input,
        weight_grad_meta=old_grad_weight,
        pattern=pattern,
    )
    old_grad_input.replace_all_uses_with(rmsnorm.input_grad)
    old_grad_weight.replace_all_uses_with(rmsnorm.weight_grad)
    _mark_new_nodes_for_inductor(gm, existing_nodes, region=region, group=body_name)
    _mark_coda_owned(
        pattern,
        (*path.nodes, norm_backward, old_grad_input, old_grad_weight),
    )


def _prepare_coda_graph(gm: GraphModule) -> GraphModule:
    _stable_topological_sort(gm.graph, {})
    gm.graph.eliminate_dead_code()
    _assign_coda_inductor_regions(gm)
    gm.graph.lint()
    gm.recompile()
    return gm


def _finalize_coda_graph(
    gm: GraphModule,
    counts: Counter[str],
    *,
    log_matches: bool = True,
) -> GraphModule:
    _prepare_coda_graph(gm)
    gm.meta["coda_pattern_counts"] = dict(counts)
    if log_matches:
        matched = {name: count for name, count in counts.items() if count}
        logger.info(f"CODA FlexGEMM matched {sum(matched.values())} groups: {matched}")
    return gm


def _apply_coda_pattern(
    gm: GraphModule,
    pattern: CodaPattern,
    *,
    log_matches: bool = True,
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    selection: BenchmarkCandidateSelection | None = None,
    autotune: bool = False,
) -> GraphModule:
    counts: Counter[str] = Counter({name: 0 for name in CODA_PATTERN_NAMES})
    counts.update(gm.meta.get("coda_pattern_counts", {}))
    pattern.apply(gm, counts, benchmark_regions, selection, autotune)
    if benchmark_regions is not None:
        return gm
    return _finalize_coda_graph(gm, counts, log_matches=log_matches)


def _apply_coda_candidate(
    gm: GraphModule,
    selection: BenchmarkCandidateSelection,
    benchmark_regions: list[RewriteBenchmarkRegion] | None,
    *,
    pattern: CodaPattern,
    autotune: bool,
) -> GraphModule:
    return _apply_coda_pattern(
        gm,
        pattern,
        log_matches=False,
        benchmark_regions=benchmark_regions,
        selection=selection,
        autotune=autotune,
    )


def _resolve_coda_patterns(patterns: Iterable[str] | None) -> tuple[str, ...]:
    requested = list(CODA_PATTERN_NAMES if patterns is None else patterns)
    duplicates = sorted(name for name in set(requested) if requested.count(name) > 1)
    if duplicates:
        raise ValueError(f"Duplicate CODA pattern entries: {duplicates}")
    unknown = sorted(set(requested) - _CODA_PATTERNS.keys())
    if unknown:
        raise ValueError(
            f"Unknown CODA pattern entries: {unknown}; "
            f"supported patterns: {sorted(CODA_PATTERN_NAMES)}"
        )
    enabled = (_CODA_PATTERNS[name] for name in requested)
    return tuple(
        pattern.name
        for pattern in sorted(enabled, key=lambda item: (item.priority, item.name))
    )


def _kernel_policy_summary(
    pattern: CodaPattern,
    *,
    autotune: bool,
) -> dict[str, str]:
    summary: dict[str, str] = {}
    for name, kernel in pattern.kernels.items():
        if kernel.backend != "QUACK":
            summary[name] = kernel.backend
        elif autotune and kernel.supports_autotune:
            summary[name] = "QUACK autotune"
        elif kernel.best_configs_by_shape:
            summary[name] = "QUACK shape-specific config"
        elif kernel.best_configs:
            summary[name] = "QUACK pinned config"
        else:
            summary[name] = "QUACK default config"
    return summary


def _configured_coda_pass(
    pattern: CodaPattern,
    *,
    compile_time_benchmark: bool,
    benchmark_strict: bool,
    coda_autotune: bool,
    benchmark_graph_processor: BenchmarkGraphProcessorFn | None,
) -> Callable:
    @functools.wraps(_apply_coda_pattern)
    def apply(
        gm: GraphModule,
        example_inputs: tuple | None = None,
    ) -> GraphModule:
        logger.info(
            f"CODA {pattern.name} kernel policies: "
            f"{_kernel_policy_summary(pattern, autotune=coda_autotune)}"
        )
        if compile_time_benchmark:
            return apply_benchmarked_rewrites(
                gm,
                name=pattern.name,
                apply_candidate=functools.partial(
                    _apply_coda_candidate,
                    pattern=pattern,
                    autotune=coda_autotune,
                ),
                cache_key=(
                    "coda_flex_gemm",
                    coda_autotune,
                    benchmark_graph_processor,
                ),
                strict=benchmark_strict,
                process_baseline=benchmark_graph_processor,
                process_candidate=benchmark_graph_processor,
            )
        return _apply_coda_pattern(gm, pattern, autotune=coda_autotune)

    apply.__name__ = pattern.name
    return apply


def get_coda_pattern_passes(
    patterns: Iterable[str] | None = None,
    *,
    compile_time_benchmark: bool = True,
    benchmark_strict: bool = False,
    coda_autotune: bool = False,
    benchmark_graph_processor: BenchmarkGraphProcessorFn | None = None,
) -> list[Callable]:
    """Build canonical CODA passes; ``None`` selects all and ``[]`` selects none."""
    return [
        _configured_coda_pass(
            _CODA_PATTERNS[name],
            compile_time_benchmark=compile_time_benchmark,
            benchmark_strict=benchmark_strict,
            coda_autotune=coda_autotune,
            benchmark_graph_processor=benchmark_graph_processor,
        )
        for name in _resolve_coda_patterns(patterns)
    ]


def coda_flex_gemm_pass(
    gm: GraphModule,
    example_inputs: tuple | None = None,
    *,
    patterns: Iterable[str] | None = None,
    compile_time_benchmark: bool = True,
    benchmark_strict: bool = False,
    coda_autotune: bool = False,
    benchmark_graph_processor: BenchmarkGraphProcessorFn | None = None,
) -> GraphModule:
    """Apply canonical CODA patterns to a joint training graph in priority order."""
    for pattern_pass in get_coda_pattern_passes(
        patterns,
        compile_time_benchmark=compile_time_benchmark,
        benchmark_strict=benchmark_strict,
        coda_autotune=coda_autotune,
        benchmark_graph_processor=benchmark_graph_processor,
    ):
        gm = pattern_pass(gm, example_inputs)
    return gm
