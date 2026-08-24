# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CODA-style FlexGEMM epilogue fusion passes for joint training graphs.
"""

from __future__ import annotations

import functools
import logging
import operator
from collections import Counter, deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from math import gcd, prod
from typing import Any

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._guards import detect_fake_mode
from torch._higher_order_ops.flex_gemm import (
    flex_gemm_hop,
    mark_flex_gemm_body_gemm_node,
)
from torch._inductor.pattern_matcher import (
    CallFunction,
    KeywordArg,
    Match,
    MULTIPLE,
    PatternExpr,
)
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.fx import GraphModule, Node
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.utils.fuser_utils import fuse_as_graphmodule, validate_partition

from torchtitan.experiments.graph_trainer.compile_time_benchmark import (
    apply_benchmarked_rewrites,
    make_rewrite_benchmark_region,
    RewriteBenchmarkRegion,
    RewriteCandidate,
)

aten = torch.ops.aten
logger = logging.getLogger(__name__)


CodaMatcher = Callable[
    [
        GraphModule,
        tuple[Match, ...],
        Counter[str],
        set[Node],
        list[RewriteCandidate] | None,
        bool,
    ],
    None,
]


@dataclass(frozen=True)
class CodaKernel:
    """Backend and tuning policy for one named GEMM emitted by a pattern."""

    backend: str = "QUACK"
    fast_math: bool = False
    supports_autotune: bool = True
    pretuned_configs: dict[int, dict[str, Any]] = field(default_factory=dict)
    pretuned_configs_by_shape: dict[
        int, dict[tuple[int, int, int], dict[str, Any]]
    ] = field(default_factory=dict)


@dataclass(frozen=True)
class CodaPattern:
    """Declarative search, structural rewrite, and kernel policy for a pattern."""

    name: str
    matcher: CodaMatcher
    priority: int
    search: PatternExpr
    kernels: dict[str, CodaKernel]

    def apply(
        self,
        gm: GraphModule,
        counts: Counter[str],
        nodes_to_remove: set[Node],
        prepared_candidates: list[RewriteCandidate] | None = None,
        autotune: bool = False,
    ) -> None:
        """Run this registration's search, then validate and rewrite its matches."""
        candidates = tuple(
            match
            for node in list(gm.graph.nodes)
            if isinstance((match := self.search.match(node)), Match)
        )
        self.matcher(
            gm,
            candidates,
            counts,
            nodes_to_remove,
            prepared_candidates,
            autotune,
        )


_CODA_PATTERNS: dict[str, CodaPattern] = {}


def register_coda_pattern(
    *,
    priority: int,
    search: PatternExpr,
    kernels: dict[str, CodaKernel],
) -> Callable[[CodaMatcher], CodaMatcher]:
    """Register a named handler, its search, and its complete kernel policy."""

    def register(matcher: CodaMatcher) -> CodaMatcher:
        name = matcher.__name__
        if name in _CODA_PATTERNS:
            raise ValueError(f"CODA pattern {name!r} is already registered")
        if any(pattern.priority == priority for pattern in _CODA_PATTERNS.values()):
            raise ValueError(f"CODA pattern priority {priority} is already registered")
        if not kernels:
            raise ValueError(f"CODA pattern {name!r} must declare a kernel site")
        _CODA_PATTERNS[name] = CodaPattern(
            name,
            matcher,
            priority,
            search,
            kernels,
        )
        return matcher

    return register


def _quack_config(
    tile_m: int,
    tile_n: int,
    *,
    dynamic: bool,
    cluster_m: int = 2,
    cluster_n: int = 1,
    swap_ab: bool = False,
) -> dict[str, Any]:
    return {
        "tile_m": tile_m,
        "tile_n": tile_n,
        "tile_k": None,
        "num_warps": None,
        "pingpong": False,
        "is_dynamic_persistent": dynamic,
        "cluster_m": cluster_m,
        "cluster_n": cluster_n,
        "cluster_k": 1,
        "swap_ab": swap_ab,
        "max_swizzle_size": 8,
        "device_capacity": 10,
        "use_tma_gather": False,
    }


# Patterns registrar
#
# Function name is its canonical pattern ID.
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
            pretuned_configs={10: _quack_config(256, 192, dynamic=False)}
        ),
        "expansion": CodaKernel(
            pretuned_configs={10: _quack_config(256, 256, dynamic=True)}
        ),
    },
)
def F_mla_qproj_rmsnorm_expand(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a Q projection, RMSNorm, and the following expansion projection.

    Input::

        q_a = mm(x, wq_a)
        q_norm, rstd = rms_norm(q_a, norm_weight, eps)
        q = mm(q_norm, wq_b)

    Output::

        def projection_epilogue(acc):
            acc_fp32 = acc.float()
            weighted = (acc_fp32 * norm_weight).to(output_dtype)
            partial_ms = acc_fp32.view(..., RMSNORM_GROUP).square().mean(-1)
            return acc, weighted, partial_ms

        q_a, weighted_q_a, partial_ms = flex_gemm(
            x, wq_a, epilogue=projection_epilogue
        )
        rstd = rsqrt(mean(partial_ms) + eps)
        q_norm = (q_a.float() * rstd * norm_weight).to(output_dtype)

        def expansion_epilogue(acc):
            return (acc.float() * rstd).to(output_dtype)

        q = flex_gemm(weighted_q_a, wq_b, epilogue=expansion_epilogue)

    Description::

        Sites: Low-rank Q down/up projections: ``NUM_MLA_LAYERS``.
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
            or norm in nodes_to_remove
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
            or not _coda_nodes_available(
                (*path.nodes, norm, second_use.node), nodes_to_remove
            )
        ):
            continue
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            norm,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            norm=norm,
            first_path=path,
            second_use=second_use,
            full_output=norm_input,
            pattern=pattern,
            autotune=autotune,
        )


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
            pretuned_configs={10: _quack_config(128, 192, dynamic=False, cluster_m=1)}
        ),
        "expansion": CodaKernel(
            pretuned_configs={10: _quack_config(128, 256, dynamic=True)}
        ),
    },
)
def F_mla_kvproj_rmsnorm_expand(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a segmented KV projection, RMSNorm, and expansion projection.

    Input::

        compressed_kv = mm(x, wkv_a)
        kv_latent, k_rope = split(compressed_kv)
        kv_norm, rstd = rms_norm(kv_latent, norm_weight, eps)
        kv = mm(kv_norm, wkv_b)

    Output::

        def projection_epilogue(acc):
            acc_fp32 = acc.float()
            weighted = (acc_fp32 * norm_weight).to(output_dtype)
            partial_ms = acc_fp32.view(..., RMSNORM_GROUP).square().mean(-1)
            return acc, weighted, partial_ms

        compressed_kv, weighted_kv, partial_ms = flex_gemm(
            x, wkv_a, epilogue=projection_epilogue
        )
        kv_latent, k_rope = split(compressed_kv)
        weighted_kv_latent = weighted_kv[..., :KV_LORA_RANK]
        partial_ms = partial_ms[..., :NUM_KV_GROUPS]
        rstd = rsqrt(mean(partial_ms) + eps)
        kv_norm = (kv_latent.float() * rstd * norm_weight).to(output_dtype)

        def expansion_epilogue(acc):
            return (acc.float() * rstd).to(output_dtype)

        kv = flex_gemm(weighted_kv_latent, wkv_b, epilogue=expansion_epilogue)

    Description::

        Sites: MLA KV down/up projections: ``NUM_MLA_LAYERS``.
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
            or norm in nodes_to_remove
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
            or not _coda_nodes_available(
                (*path.nodes, norm, second_use.node), nodes_to_remove
            )
        ):
            continue
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            norm,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            norm=norm,
            first_path=path,
            second_use=second_use,
            full_output=full_output,
            pattern=pattern,
            autotune=autotune,
        )


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
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a weighted residual BMM with its following RMSNorm.

    Input::

        residual = bmm(probabilities, values).squeeze(1).to(output_dtype)
        normalized, rstd = rms_norm(residual, norm_weight, eps)

    Output::

        def epilogue(acc):
            residual = acc.to(output_dtype)
            partial_ms = (
                residual.float().view(..., RMSNORM_GROUP).square().mean(-1)
            )
            return residual, partial_ms

        residual, partial_ms = flex_gemm(
            probabilities, values, epilogue=epilogue
        )
        residual = residual.squeeze(1)
        rstd = rsqrt(mean(partial_ms) + eps)
        normalized = (residual.float() * rstd * norm_weight).to(output_dtype)

    Description::

        Sites: Kimi K3 attention and FFN prenorms: ``2 * NUM_LAYERS``.
    """
    rewrite = _rewrite_forward_rmsnorm
    pattern = F_weighted_residual_bmm_prenorm.__name__
    for candidate in candidates:
        norm = candidate.output_node()
        path = _weighted_residual_rmsnorm_path(norm, nodes_to_remove)
        if path is None:
            continue
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            norm,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            norm=norm,
            path=path,
            pattern=pattern,
            autotune=autotune,
        )


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
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a GEMM, residual add chain, and RMSNorm.

    Input::

        projected = mm(x, weight)
        hidden = projected + residual_0 + ... + residual_n
        normalized, rstd = rms_norm(hidden, norm_weight, eps)

    Output::

        def epilogue(acc):
            hidden = acc + residual_0 + ... + residual_n
            partial_ms = hidden.float().view(..., RMSNORM_GROUP).square().mean(-1)
            return hidden, partial_ms

        hidden, partial_ms = flex_gemm(x, weight, epilogue=epilogue)
        rstd = rsqrt(mean(partial_ms) + eps)
        normalized = (hidden.float() * rstd * norm_weight).to(output_dtype)

    Description::

        Sites: Post-attention and post-FFN residual prenorms: ``2 * NUM_LAYERS``.
    """
    rewrite = _rewrite_forward_rmsnorm
    pattern = F_mm_residual_rmsnorm.__name__
    for candidate in candidates:
        norm = candidate.output_node()
        path = _residual_rmsnorm_path(norm, nodes_to_remove)
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
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            norm,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            norm=norm,
            path=path,
            pattern=pattern,
            autotune=autotune,
        )


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
            pretuned_configs_by_shape={
                10: {
                    (8192, 2048, 10944): _quack_config(
                        256, 224, dynamic=True, cluster_n=2
                    )
                }
            },
        ),
        "up": CodaKernel(
            fast_math=True,
            pretuned_configs_by_shape={
                10: {(8192, 2048, 10944): _quack_config(256, 224, dynamic=True)}
            },
        ),
    },
)
def F_swiglu(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse two projections and the exact SwiGLU activation topology.

    Input::

        gate = mm(x, gate_weight)
        up = mm(x, up_weight)
        output = silu(gate) * up

    Output::

        activated_gate = flex_gemm(
            x, gate_weight, epilogue=lambda acc: silu(acc)
        )
        output = flex_gemm(
            x,
            up_weight,
            epilogue=lambda acc: acc * activated_gate,
        )

    Description::

        Sites: Separate gate/up SwiGLU FFNs:
        ``NUM_DENSE_FFNS + NUM_SHARED_EXPERT_FFNS``.
    """

    def match_swiglu(
        output: Node,
    ) -> tuple[_MatmulChain, Node, _MatmulChain] | None:
        if _is_backward(output):
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
            output_shape = _shape(output)
            if (
                gate.root.args[0] is not up.root.args[0]
                or output_shape is None
                or _shape(gate_output) != output_shape
                or _shape(up_output) != output_shape
            ):
                continue
            return gate, gate_output, up
        return None

    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = F_swiglu.__name__
    for candidate in candidates:
        output = candidate.output_node()
        match = match_swiglu(output)
        if match is None:
            continue
        gate, silu, up = match
        gate_nodes = {*gate.nodes, silu}
        up_nodes = {*up.nodes, output}
        if (
            _find_flex_gemm_partition(
                gm,
                root=gate.root,
                body_nodes=gate_nodes,
                nodes_to_remove=nodes_to_remove,
            )
            is None
            or _find_flex_gemm_partition(
                gm,
                root=up.root,
                body_nodes=up_nodes,
                nodes_to_remove=nodes_to_remove,
            )
            is None
        ):
            continue
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            output,
            counts,
            nodes_to_remove,
            prepared_candidates,
            _combine_coda_rewrites(
                functools.partial(
                    rewrite,
                    gm,
                    root=gate.root,
                    body_nodes=gate_nodes,
                    pattern=pattern,
                    autotune=autotune,
                    kernel="gate",
                ),
                functools.partial(
                    rewrite,
                    gm,
                    root=up.root,
                    body_nodes=up_nodes,
                    pattern=pattern,
                    autotune=autotune,
                    kernel="up",
                ),
            ),
        )


@register_coda_pattern(
    priority=60,
    search=CallFunction(
        aten._to_copy.default,
        KeywordArg("product"),
        dtype=KeywordArg("output_dtype"),
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
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse two projections and the exact SiTU activation topology.

    Input::

        gate = mm(x, gate_weight).float()
        up = mm(x, up_weight).float()
        gate = 4 * tanh(gate / 4) * sigmoid(gate)
        output = (gate * (25 * tanh(up / 25))).to(output_dtype)

    Output::

        def gate_epilogue(acc):
            gate = acc.float()
            return 4 * tanh(gate / 4) * sigmoid(gate)

        activated_gate = flex_gemm(x, gate_weight, epilogue=gate_epilogue)

        def up_epilogue(acc):
            up = 25 * tanh(acc.float() / 25)
            return (activated_gate * up).to(output_dtype)

        output = flex_gemm(x, up_weight, epilogue=up_epilogue)

    Description::

        Sites: Separate gate/up SiTU FFNs:
        ``NUM_DENSE_FFNS + NUM_SHARED_EXPERT_FFNS``.
    """

    def scalar_input(node: object, target: Any, scalar: float) -> Node | None:
        if (
            not isinstance(node, Node)
            or node.target is not target
            or len(node.args) < 2
        ):
            return None
        lhs, rhs = node.args[:2]
        if isinstance(lhs, Node) and rhs == scalar:
            return lhs
        if isinstance(rhs, Node) and lhs == scalar:
            return rhs
        return None

    def match_situ(
        output: Node,
    ) -> tuple[_MatmulChain, frozenset[Node], _MatmulChain, frozenset[Node],] | None:
        output_dtype = _cast_dtype(output)
        if output_dtype is None:
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
            gate_tanh = scalar_input(gate_scaled, aten.mul.Tensor, 4.0)
            gate_div = (
                _single_tensor_input(gate_tanh) if gate_tanh is not None else None
            )
            gate_cast = scalar_input(gate_div, aten.div.Tensor, 4.0)
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

            up_tanh = scalar_input(up_output, aten.mul.Tensor, 25.0)
            up_div = _single_tensor_input(up_tanh) if up_tanh is not None else None
            up_cast = scalar_input(up_div, aten.div.Tensor, 25.0)
            if (
                up_tanh is None
                or up_tanh.target is not aten.tanh.default
                or up_cast is None
                or up_cast.target is not _CAST_TARGET
                or _cast_dtype(up_cast) is not torch.float32
            ):
                continue
            up = _chain_to_mm(up_cast.args[0], allow_cast=False)
            if (
                gate is None
                or up is None
                or gate.root.args[0] is not up.root.args[0]
                or _dtype(gate.root) is not output_dtype
                or _dtype(up.root) is not output_dtype
            ):
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
            return gate, gate_nodes, up, up_nodes
        return None

    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = F_situ.__name__
    for candidate in candidates:
        output = candidate.output_node()
        match = match_situ(output)
        if match is None:
            continue
        gate, gate_nodes, up, up_nodes = match
        if (
            _find_flex_gemm_partition(
                gm,
                root=gate.root,
                body_nodes=gate_nodes,
                nodes_to_remove=nodes_to_remove,
            )
            is None
            or _find_flex_gemm_partition(
                gm,
                root=up.root,
                body_nodes=up_nodes,
                nodes_to_remove=nodes_to_remove,
            )
            is None
        ):
            continue
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            output,
            counts,
            nodes_to_remove,
            prepared_candidates,
            _combine_coda_rewrites(
                functools.partial(
                    rewrite,
                    gm,
                    root=gate.root,
                    body_nodes=gate_nodes,
                    pattern=pattern,
                    autotune=autotune,
                    kernel="gate",
                ),
                functools.partial(
                    rewrite,
                    gm,
                    root=up.root,
                    body_nodes=up_nodes,
                    pattern=pattern,
                    autotune=autotune,
                    kernel="up",
                ),
            ),
        )


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
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse the MLA output gate into its projection epilogue.

    Input::

        gate = sigmoid(mm(x, gate_weight))
        gated_attention = attention * gate

    Output::

        def epilogue(acc):
            gate = sigmoid(acc)
            return gate, attention * gate

        gate, gated_attention = flex_gemm(x, gate_weight, epilogue=epilogue)

    Description::

        Sites: Kimi K3 MLA output-gate projections: ``NUM_MLA_LAYERS``.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = F_k3_mla_output_gate.__name__
    for candidate in candidates:
        sigmoid = candidate.output_node()
        if (
            _is_backward(sigmoid)
            or sigmoid.target is not aten.sigmoid.default
            or sigmoid in nodes_to_remove
        ):
            continue
        chain = _chain_to_mm(sigmoid.args[0], allow_cast=False)
        mul_match = _find_pointwise_user(sigmoid, aten.mul.Tensor)
        if chain is None or mul_match is None:
            continue
        mul, bridge = mul_match
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            sigmoid,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            root=chain.root,
            body_nodes={*chain.nodes, sigmoid, *bridge, mul},
            pattern=pattern,
            autotune=autotune,
        )


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
            pretuned_configs={10: _quack_config(256, 256, dynamic=True)},
        )
    },
)
def F_router_sigmoid_bias(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse router sigmoid and expert bias into the projection epilogue.

    Input::

        scores = sigmoid(mm(tokens, router_weight))
        biased_scores = scores + expert_bias

    Output::

        def epilogue(acc):
            scores = sigmoid(acc)
            return scores, scores + expert_bias

        scores, biased_scores = flex_gemm(
            tokens, router_weight, epilogue=epilogue
        )

    Description::

        Sites: Sigmoid-routed MoE router projections: ``NUM_MOE_LAYERS``.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = F_router_sigmoid_bias.__name__
    for candidate in candidates:
        sigmoid = candidate.output_node()
        if (
            _is_backward(sigmoid)
            or sigmoid.target is not aten.sigmoid.default
            or sigmoid in nodes_to_remove
        ):
            continue
        chain = _chain_to_mm(sigmoid.args[0], allow_cast=False)
        add_match = _find_pointwise_user(sigmoid, aten.add.Tensor)
        if chain is None or add_match is None:
            continue
        add, bridge = add_match
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            sigmoid,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            root=chain.root,
            body_nodes={*chain.nodes, sigmoid, *bridge, add},
            pattern=pattern,
            autotune=autotune,
        )


@register_coda_pattern(
    priority=110,
    search=CallFunction(
        aten._to_copy.default,
        KeywordArg("projection"),
        dtype=torch.float32,
        _users=MULTIPLE,
    ),
    kernels={
        "main": CodaKernel(pretuned_configs={10: _quack_config(256, 256, dynamic=True)})
    },
)
def B_reshape_bf16_to_fp32(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a reshaped GEMM output's FP32 cast into the GEMM.

    Input::

        grad_2d = mm(grad_output, weight)
        grad_fp32 = reshape(grad_2d, input_shape).float()

    Output::

        grad_2d_fp32 = flex_gemm(
            grad_output,
            weight,
            epilogue=lambda acc: acc.float(),
        )
        grad_fp32 = reshape(grad_2d_fp32, input_shape)

    Description::

        Sites: Reshaped LM-head input gradients: ``NUM_LM_HEADS``.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = B_reshape_bf16_to_fp32.__name__
    for candidate in candidates:
        cast = candidate.output_node()
        if (
            not _is_backward(cast)
            or cast.target is not _CAST_TARGET
            or _cast_dtype(cast) is not torch.float32
            or cast in nodes_to_remove
        ):
            continue
        chain = _chain_to_mm(cast)
        if (
            chain is None
            or not _path_has_phase(chain, backward=True)
            or not _is_supported_flex_gemm_root(chain.root)
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
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            cast,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            root=chain.root,
            body_nodes=chain.nodes,
            pattern=pattern,
            autotune=autotune,
        )


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
            pretuned_configs={10: _quack_config(128, 128, dynamic=True, swap_ab=True)},
            pretuned_configs_by_shape={
                10: {(8192, 2048, 10944): _quack_config(256, 224, dynamic=True)}
            },
        )
    },
)
def B_swiglu_backward_activation(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse the exact SwiGLU pointwise backward after its gradient GEMM.

    Input::

        branch_grad = mm(grad_output, down_weight)
        up_grad = branch_grad * saved_silu_gate
        gate_grad = silu_backward(branch_grad * saved_up, saved_gate)

    Output::

        def epilogue(acc):
            up_grad = acc * saved_silu_gate
            gate_grad = silu_backward(acc * saved_up, saved_gate)
            return up_grad, gate_grad

        up_grad, gate_grad = flex_gemm(
            grad_output, down_weight, epilogue=epilogue
        )

    Description::

        Sites: SwiGLU down-projection input gradients:
        ``NUM_DENSE_FFNS + NUM_SHARED_EXPERT_FFNS``.
    """

    def match_swiglu_backward(
        root: Node,
    ) -> tuple[frozenset[Node], tuple[Node, ...]] | None:
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
            gate_grad = _single_user_with_target(
                gate_product, aten.silu_backward.default
            )
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
            return body_nodes, (up_grad, gate_grad)
        return None

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
        if root in nodes_to_remove:
            continue
        match = match_swiglu_backward(root)
        if match is None:
            continue
        body_nodes, outputs = match
        if gate_grad not in outputs or not _coda_nodes_available(
            body_nodes, nodes_to_remove
        ):
            continue
        ordered = _ordered_nodes(gm, body_nodes)
        if not validate_partition(ordered):
            continue
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            root,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            root=root,
            body_nodes=body_nodes,
            pattern=pattern,
            autotune=autotune,
            fused_outputs=outputs,
        )


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
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse two parallel GEMM input-gradient branches at their final add.

    Input::

        lhs_grad = mm(lhs_branch_grad, lhs_weight)
        rhs_grad = mm(rhs_branch_grad, rhs_weight)
        input_grad = lhs_grad + rhs_grad

    Output::

        lhs_grad = mm(lhs_branch_grad, lhs_weight)
        input_grad = flex_gemm(
            rhs_branch_grad,
            rhs_weight,
            epilogue=lambda acc: acc + lhs_grad,
        )

    Description::

        Sites: Separate gate/up GLU input-gradient merges:
        ``NUM_DENSE_FFNS + NUM_SHARED_EXPERT_FFNS``.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = B_parallel_mm_dx_merge.__name__
    for candidate in candidates:
        add = candidate.output_node()
        if (
            not _is_backward(add)
            or add.target is not aten.add.Tensor
            or add in nodes_to_remove
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
            or any(not _is_supported_flex_gemm_root(chain.root) for chain in chains)
            or len({_dtype(chain.root) for chain in chains}) != 1
        ):
            continue
        chain = chains[-1]
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            add,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            root=chain.root,
            body_nodes={*chain.nodes, add},
            pattern=pattern,
            autotune=autotune,
        )


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
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse MLA output-gate backward into the output-projection dX GEMM.

    Input::

        gated_grad = mm(grad_output, output_weight)
        attention_grad = gated_grad * saved_sigmoid_gate
        gate_grad = sigmoid_backward(
            gated_grad * saved_attention, saved_sigmoid_gate
        )

    Output::

        def epilogue(acc):
            attention_grad = acc * saved_sigmoid_gate
            gate_grad = sigmoid_backward(
                acc * saved_attention, saved_sigmoid_gate
            )
            return attention_grad, gate_grad

        attention_grad, gate_grad = flex_gemm(
            grad_output, output_weight, epilogue=epilogue
        )

    Description::

        Sites: Kimi K3 MLA output-projection input gradients: ``NUM_MLA_LAYERS``.
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
            or root in nodes_to_remove
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
        if set(_boundary_outputs(_ordered_nodes(gm, body_nodes))) != {
            attention_grad,
            gate_grad,
        } or not _coda_nodes_available(body_nodes, nodes_to_remove):
            continue
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            root,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            root=root,
            body_nodes=body_nodes,
            pattern=pattern,
            autotune=autotune,
            fused_outputs=(attention_grad, gate_grad),
        )


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
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse the exact SiTU pointwise backward after its gradient GEMM.

    Input::

        branch_grad_fp32 = mm(grad_output, down_weight).float()
        up_grad = tanh_backward(branch_grad_fp32, saved_tanh).to(output_dtype)
        gate_grad = sigmoid_backward(
            branch_grad_fp32, saved_sigmoid
        ).to(output_dtype)

    Output::

        def epilogue(acc):
            branch_grad_fp32 = acc.float()
            up_grad = tanh_backward(branch_grad_fp32, saved_tanh).to(output_dtype)
            gate_grad = sigmoid_backward(
                branch_grad_fp32, saved_sigmoid
            ).to(output_dtype)
            return up_grad, gate_grad

        up_grad, gate_grad = flex_gemm(
            grad_output, down_weight, epilogue=epilogue
        )

    Description::

        Sites: SiTU down-projection input gradients:
        ``NUM_DENSE_FFNS + NUM_SHARED_EXPERT_FFNS``.
    """

    def match_situ_backward(
        root: Node,
    ) -> tuple[frozenset[Node], tuple[Node, ...]] | None:
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
        sigmoid_grad = _single_user_with_target(
            grad_fp32, aten.sigmoid_backward.default
        )
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
        return body_nodes, (tanh_output, sigmoid_output)

    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = B_k3_situ_backward_activation.__name__
    for candidate in candidates:
        branch_grad = candidate.kwargs["branch_grad_fp32"]
        if not isinstance(branch_grad, Node):
            continue
        root = _single_tensor_input(branch_grad)
        if root is None:
            continue
        if root in nodes_to_remove:
            continue
        match = match_situ_backward(root)
        if match is None:
            continue
        body_nodes, outputs = match
        if candidate.output_node() not in outputs or not _coda_nodes_available(
            body_nodes, nodes_to_remove
        ):
            continue
        ordered = _ordered_nodes(gm, body_nodes)
        if not validate_partition(ordered):
            continue
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            root,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            root=root,
            body_nodes=body_nodes,
            pattern=pattern,
            autotune=autotune,
            fused_outputs=outputs,
        )


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
            pretuned_configs={10: _quack_config(256, 256, dynamic=True, cluster_n=2)}
        )
    },
)
def B_mm_dx_residual_add(  # noqa: N802
    gm: GraphModule,
    candidates: tuple[Match, ...],
    counts: Counter[str],
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a GEMM input-gradient branch with a residual-gradient add.

    Input::

        projection_grad = mm(branch_grad, weight)
        input_grad = projection_grad + residual_grad

    Output::

        input_grad = flex_gemm(
            branch_grad,
            weight,
            epilogue=lambda acc: acc + residual_grad,
        )

    Description::

        Sites: MoE router input-gradient residual adds: ``NUM_MOE_LAYERS``.
    """
    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    pattern = B_mm_dx_residual_add.__name__
    for candidate in candidates:
        add = candidate.output_node()
        if (
            not _is_backward(add)
            or add.target is not aten.add.Tensor
            or add in nodes_to_remove
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
            or not _coda_nodes_available(body_nodes, nodes_to_remove)
            or not validate_partition(_ordered_nodes(gm, body_nodes))
        ):
            continue
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            add,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            root=chain.root,
            body_nodes=body_nodes,
            pattern=pattern,
            autotune=autotune,
        )


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
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a GEMM input gradient with RMSNorm backward.

    Input::

        norm_output_grad = mm(grad_output, expansion_weight)
        input_grad, weight_grad = rms_norm_backward(
            norm_output_grad, norm_input, rstd, norm_weight
        )

    Output::

        def epilogue(acc):
            x_hat = norm_input.float() * rstd
            grad_x_hat = acc.float() * norm_weight.float()
            partial_row_dots = (
                (x_hat * grad_x_hat).view(..., RMSNORM_GROUP).sum(-1)
            )
            return acc, partial_row_dots

        norm_output_grad, partial_row_dots = flex_gemm(
            grad_output, expansion_weight, epilogue=epilogue
        )
        input_grad, weight_grad = finish_rms_norm_backward(
            norm_output_grad, partial_row_dots, norm_input, rstd, norm_weight
        )

    Description::

        Sites: MLA Q/KV expansion input gradients: ``NUM_EXPANDED_LATENT_NORMS``.
    """
    pattern = B_mm_dx_rmsnorm.__name__

    def rewrite(
        candidate_gm: GraphModule,
        *,
        norm_backward: Node,
        path: _MatmulChain,
        nodes_to_remove: set[Node],
        benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    ) -> bool:
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

            def apply_rewrite(
                candidate: GraphModule,
                candidate_nodes: dict[str, Node],
                candidate_nodes_to_remove: set[Node],
            ) -> None:
                rewrite(
                    candidate,
                    norm_backward=candidate_nodes[norm_backward.name],
                    path=_MatmulChain(
                        candidate_nodes[path.root.name],
                        tuple(candidate_nodes[node.name] for node in path.nodes),
                    ),
                    nodes_to_remove=candidate_nodes_to_remove,
                )

            benchmark_regions.append(
                _make_local_rewrite_benchmark_region(
                    candidate_gm,
                    region_nodes,
                    pattern,
                    apply_rewrite,
                )
            )
            nodes_to_remove.update(region_nodes)
            return True
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

        ordered = _ordered_nodes(candidate_gm, path.nodes)
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
        body_input_values = []
        for external in external_inputs:
            value = (
                _tensor_value(external)
                if external in root_inputs
                else _view_coda_value(
                    external,
                    desired_shapes.get(external, root_shape),
                )
            )
            if value is None:
                raise AssertionError(
                    f"CODA input {external.name} requires tensor metadata"
                )
            body_input_values.append((external, value))
        body = _trace_backward_rmsnorm_body(
            path,
            body_input_values,
            norm_input=norm_input,
            rstd=rstd,
            weight=weight,
            root_shape=root_shape,
            width=width,
            pattern=pattern,
        )
        rstd_shape = _shape(rstd)
        if rstd_shape is None:
            raise AssertionError(f"CODA {pattern} requires RMSNorm rstd shape metadata")
        mark_flex_gemm_body_gemm_node(body, path.root.target)
        options = _kernel_options(
            pattern,
            kernel="main",
            root=path.root,
            autotune=autotune,
        )

        def replacement(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
            env = dict(zip(external_inputs, args, strict=True))
            body_inputs = tuple(
                env[external]
                if external in root_inputs
                else env[external].view(desired_shapes.get(external, root_shape))
                for external in external_inputs
            )
            grad_2d, partial = flex_gemm_hop(
                path.root.target,
                body,
                body_inputs,
                {},
                options,
            )
            return _backward_rmsnorm_fn_to_trace(
                grad_2d,
                partial,
                env[norm_input],
                env[rstd],
                env[weight],
                grad_shape=grad_shape,
                rstd_shape=rstd_shape,
                width=width,
                input_grad_dtype=grad_input_dtype,
                weight_grad_dtype=grad_weight_dtype,
            )

        new_grad_input, new_grad_weight = _insert_traced_replacement(
            candidate_gm,
            replacement,
            external_inputs,
            (_tensor_value(old_grad_input), _tensor_value(old_grad_weight)),
            pattern=pattern,
        )
        _copy_meta(new_grad_input, old_grad_input)
        _copy_meta(new_grad_weight, old_grad_weight)
        old_grad_input.replace_all_uses_with(new_grad_input)
        old_grad_weight.replace_all_uses_with(new_grad_weight)
        nodes_to_remove.update(
            (*path.nodes, norm_backward, old_grad_input, old_grad_weight)
        )
        return True

    for candidate in candidates:
        node = candidate.output_node()
        if (
            not _is_backward(node)
            or not _valid_backward_rmsnorm(node)
            or node in nodes_to_remove
        ):
            continue
        path = _chain_to_mm(node.args[0], allow_cast=False)
        if (
            path is None
            or not _is_reshape_only_path(path)
            or not _path_has_phase(path, backward=True)
            or not _coda_nodes_available((*path.nodes, node), nodes_to_remove)
        ):
            continue
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            node,
            counts,
            nodes_to_remove,
            prepared_candidates,
            rewrite,
            gm,
            norm_backward=node,
            path=path,
        )


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
            pretuned_configs_by_shape={
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
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None = None,
    autotune: bool = False,
) -> None:
    """Fuse a linear weight-gradient FP32 cast into its low-precision GEMM.

    Input::

        weight_grad = mm(grad_output.T, input)
        weight_grad_fp32 = optional_view_or_transpose(weight_grad).float()

    Output::

        weight_grad_fp32 = flex_gemm(
            grad_output.T,
            input,
            epilogue=lambda acc: acc.float(),
        )
        weight_grad_fp32 = optional_view_or_transpose(weight_grad_fp32)

    Description::

        Sites: Dense-linear weight gradients: ``NUM_ELIGIBLE_DENSE_LINEAR_WEIGHTS``.
    """
    pattern = B_linear_dw_bf16_to_fp32.__name__

    def rewrite_transposed(
        candidate_gm: GraphModule,
        *,
        chain: _MatmulChain,
        cast: Node,
        nodes_to_remove: set[Node],
        benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
    ) -> bool:
        if not _coda_nodes_available(chain.nodes, nodes_to_remove):
            return False
        if not validate_partition(list(chain.nodes)):
            return False
        if benchmark_regions is not None:

            def apply_rewrite(
                candidate: GraphModule,
                candidate_nodes: dict[str, Node],
                candidate_nodes_to_remove: set[Node],
            ) -> None:
                candidate_chain = _MatmulChain(
                    candidate_nodes[chain.root.name],
                    tuple(candidate_nodes[node.name] for node in chain.nodes),
                )
                if not rewrite_transposed(
                    candidate,
                    chain=candidate_chain,
                    cast=candidate_nodes[cast.name],
                    nodes_to_remove=candidate_nodes_to_remove,
                ):
                    raise AssertionError(
                        f"CODA {pattern} failed to build its benchmark candidate"
                    )

            benchmark_regions.append(
                _make_local_rewrite_benchmark_region(
                    candidate_gm,
                    chain.nodes,
                    pattern,
                    apply_rewrite,
                )
            )
            nodes_to_remove.update(chain.nodes)
            return True

        root = chain.root
        external_inputs = list(root.all_input_nodes)
        cast_value = _tensor_value(cast)
        if cast_value is None:
            raise AssertionError(f"CODA {pattern} requires GEMM value metadata")

        def body(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor]:
            return (root.target(lhs, rhs).to(torch.float32),)

        traced_body = _trace_coda_function(body, external_inputs, pattern)
        mark_flex_gemm_body_gemm_node(traced_body, root.target)
        options = _kernel_options(
            pattern,
            kernel="main",
            root=root,
            autotune=autotune,
        )

        def replacement(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor]:
            (output,) = flex_gemm_hop(
                root.target,
                traced_body,
                (lhs, rhs),
                {},
                options,
            )
            previous = root
            for node in chain.nodes[1:]:
                if node is cast:
                    break
                if not node.args or node.args[0] is not previous:
                    raise AssertionError(
                        f"CODA {pattern} expected a unary shape-only chain"
                    )
                output = node.target(output, *node.args[1:], **node.kwargs)
                previous = node
            return (output,)

        (extracted,) = _insert_traced_replacement(
            candidate_gm,
            replacement,
            external_inputs,
            (cast_value,),
            pattern=pattern,
        )
        _copy_meta(extracted, cast)
        extracted.meta.setdefault("custom", {})["coda_pattern"] = pattern
        cast.replace_all_uses_with(extracted)
        nodes_to_remove.update(chain.nodes)
        return True

    rewrite = _rewrite_matched_epilogue_as_flex_gemm
    for candidate in candidates:
        cast = candidate.output_node()
        if (
            not _is_backward(cast)
            or cast.target is not _CAST_TARGET
            or _cast_dtype(cast) is not torch.float32
            or cast in nodes_to_remove
        ):
            continue
        chain = _chain_to_mm(cast)
        if (
            chain is None
            or not _path_has_phase(chain, backward=True)
            or not _is_supported_flex_gemm_root(chain.root)
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
        if has_transpose:
            candidate_rewrite = functools.partial(
                rewrite_transposed,
                gm,
                chain=chain,
                cast=cast,
            )
        else:
            candidate_rewrite = functools.partial(
                rewrite,
                gm,
                root=chain.root,
                body_nodes=chain.nodes,
                pattern=pattern,
                autotune=autotune,
            )
        _apply_or_prepare_coda_candidate(
            gm,
            pattern,
            cast,
            counts,
            nodes_to_remove,
            prepared_candidates,
            candidate_rewrite,
        )


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
class _FlexGemmPartition:
    nodes: tuple[Node, ...]
    outputs: tuple[Node, ...]


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


def _pretuned_configs(
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
        shape_configs = registered.pretuned_configs_by_shape.get(device_capacity, {})
        if shape in shape_configs:
            return shape_configs[shape]
    return registered.pretuned_configs.get(device_capacity)


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
        config = None if autotune else _pretuned_configs(pattern, kernel, root)
        if config is not None:
            options["tuned"] = False
            options["config"] = dict(config)
    if policy.fast_math:
        options["fast_math"] = True
    return options


def _is_backward(node: Node) -> bool:
    return bool(node.meta.get("autograd_backward"))


def _combine_coda_rewrites(*rewrites: Callable[..., Any]) -> Callable[..., Any]:
    def combined(
        *,
        nodes_to_remove: set[Node],
        benchmark_regions: list[RewriteBenchmarkRegion] | None,
    ) -> bool:
        for rewrite in rewrites:
            result = rewrite(
                nodes_to_remove=nodes_to_remove,
                benchmark_regions=benchmark_regions,
            )
            if result is None or result is False:
                return False
        return True

    return combined


def _apply_or_prepare_coda_candidate(
    gm: GraphModule,
    pattern: str,
    anchor: Node,
    counts: Counter[str],
    nodes_to_remove: set[Node],
    prepared_candidates: list[RewriteCandidate] | None,
    rewrite: Callable[..., Any],
    /,
    *rewrite_args: Any,
    **rewrite_kwargs: Any,
) -> None:
    def run_rewrite(
        candidate_nodes_to_remove: set[Node],
        benchmark_regions: list[RewriteBenchmarkRegion] | None,
    ) -> bool:
        result = rewrite(
            *rewrite_args,
            nodes_to_remove=candidate_nodes_to_remove,
            benchmark_regions=benchmark_regions,
            **rewrite_kwargs,
        )
        return result is not None and result is not False

    if prepared_candidates is None:
        if run_rewrite(nodes_to_remove, None):
            counts[pattern] += 1
        return

    benchmark_regions: list[RewriteBenchmarkRegion] = []
    if not run_rewrite(nodes_to_remove, benchmark_regions):
        return
    if not benchmark_regions:
        raise AssertionError(f"CODA {pattern} declared no benchmark regions")
    for region in benchmark_regions:
        _assign_coda_inductor_regions(region.candidate)

    def apply() -> None:
        candidate_nodes_to_remove: set[Node] = set()
        if not run_rewrite(candidate_nodes_to_remove, None):
            raise AssertionError(f"CODA {pattern} candidate could not be reapplied")
        _erase_nodes(gm, candidate_nodes_to_remove)
        counts[pattern] += 1
        gm.meta["coda_pattern_counts"] = dict(counts)

    prepared_candidates.append(
        RewriteCandidate(
            f"{pattern}:{anchor.name}",
            tuple(benchmark_regions),
            apply,
        )
    )


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


def _coda_nodes_available(nodes: Iterable[Node], nodes_to_remove: set[Node]) -> bool:
    return all(node not in nodes_to_remove for node in nodes)


def _erase_nodes(gm: GraphModule, nodes_to_remove: set[Node]) -> None:
    ordered = _ordered_nodes(gm, nodes_to_remove)
    for node in ordered:
        external_users = set(node.users) - nodes_to_remove
        if external_users:
            users = sorted(user.name for user in external_users)
            raise AssertionError(
                f"CODA cannot erase {node.name}; remaining users: {users}"
            )
    for node in reversed(ordered):
        gm.graph.erase_node(node)


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


def _view_coda_value(node: Node, shape: Sequence[Any]) -> torch.Tensor:
    value = _tensor_value(node)
    if value is None:
        raise AssertionError(f"CODA input {node.name} requires tensor metadata")
    if tuple(value.shape) == tuple(shape):
        return value
    try:
        return aten.view.default(value, list(shape))
    except RuntimeError as error:
        raise AssertionError(
            f"CODA input {node.name} cannot be viewed as {tuple(shape)}"
        ) from error


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
    graph_order = {node: index for index, node in enumerate(gm.graph.nodes)}
    insertion_point = min(outputs, key=graph_order.__getitem__)
    insertion_index = graph_order[insertion_point]
    nodes_to_move: set[Node] = set()

    def collect_late_producers(node: Node) -> None:
        if graph_order[node] < insertion_index or node in nodes_to_move:
            return
        if node in body_nodes:
            raise AssertionError(
                f"CODA {pattern} replacement input {node.name} depends on its body"
            )
        for input_node in node.all_input_nodes:
            collect_late_producers(input_node)
        nodes_to_move.add(node)

    for input_node in inputs:
        collect_late_producers(input_node)
    for node in sorted(nodes_to_move, key=graph_order.__getitem__):
        insertion_point.prepend(node)

    with gm.graph.inserting_before(insertion_point):
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
    rewrite: Callable[[GraphModule, dict[str, Node], set[Node]], None],
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
    candidate_nodes_to_remove: set[Node] = set()
    rewrite(candidate, candidate_nodes, candidate_nodes_to_remove)
    _erase_nodes(candidate, candidate_nodes_to_remove)
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

    missing = next((arg for arg in node.all_input_nodes if "val" not in arg.meta), None)
    if missing is not None:
        raise AssertionError(f"CODA input {missing.name} is missing value metadata")
    args, kwargs = torch.fx.node.map_arg(
        (node.args, node.kwargs), lambda arg: arg.meta["val"]
    )
    if node.target is operator.getitem:
        node.meta["val"] = node.target(*args, **kwargs)
        return

    fake_mode = detect_fake_mode()
    if fake_mode is None:
        fake_mode = next(
            (
                value.fake_mode
                for value in torch.utils._pytree.tree_leaves((args, kwargs))
                if isinstance(value, FakeTensor)
            ),
            None,
        )
    fake_mode = fake_mode or FakeTensorMode()

    def normalize(value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return value
        if isinstance(value, FakeTensor) and value.fake_mode is fake_mode:
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
    nodes_to_remove: set[Node],
    fused_outputs: Iterable[Node] | None = None,
) -> _FlexGemmPartition | None:
    if not _is_supported_flex_gemm_root(root):
        return None
    ordered = _ordered_nodes(gm, body_nodes)
    if not _coda_nodes_available(ordered, nodes_to_remove) or not validate_partition(
        ordered
    ):
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
    nodes_to_remove: set[Node],
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
        nodes_to_remove=nodes_to_remove,
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
        benchmark_regions.append(
            make_rewrite_benchmark_region(baseline_graph, replacement_graph)
        )
        nodes_to_remove.update(ordered)
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
    nodes_to_remove.update(ordered)
    return fused


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


def _residual_rmsnorm_path(
    norm: Node, nodes_to_remove: set[Node]
) -> _MatmulChain | None:
    if (
        _is_backward(norm)
        or not _valid_forward_rmsnorm(norm)
        or norm in nodes_to_remove
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
        or not _coda_nodes_available((*path.nodes, norm), nodes_to_remove)
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


def _weighted_residual_rmsnorm_path(
    norm: Node, nodes_to_remove: set[Node]
) -> _MatmulChain | None:
    if (
        _is_backward(norm)
        or not _valid_forward_rmsnorm(norm)
        or norm in nodes_to_remove
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
        or not _coda_nodes_available((bmm, squeeze, hidden, norm), nodes_to_remove)
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


def _trace_coda_function(
    fn_to_trace: Callable[..., Any],
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
            f"CODA {pattern} function inputs require tensor metadata: {missing}"
        )
    return _trace_coda_values(fn_to_trace, values, pattern)


def _insert_traced_replacement(
    gm: GraphModule,
    fn_to_trace: Callable[..., Any],
    inputs: Sequence[Node],
    expected_outputs: Sequence[torch.Tensor | None],
    *,
    pattern: str,
) -> tuple[Node, ...]:
    traced = _trace_coda_function(fn_to_trace, inputs, pattern)
    _validate_body_outputs(traced, expected_outputs, pattern)
    graph_order = {node: index for index, node in enumerate(gm.graph.nodes)}
    cursor = max(inputs, key=graph_order.__getitem__)
    return _inline_traced_coda_replacement(
        gm,
        traced,
        inputs,
        cursor,
        pattern=pattern,
    )


def _trace_forward_rmsnorm_body(
    path: _MatmulChain,
    ordered: Sequence[Node],
    input_values: Sequence[tuple[Node, torch.Tensor]],
    *,
    root_shape: Sequence[Any],
    width: int,
    pattern: str,
) -> GraphModule:
    logical_inputs = tuple(logical for logical, _ in input_values)

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

    return _trace_coda_values(
        body,
        tuple(value for _, value in input_values),
        pattern,
    )


def _trace_backward_rmsnorm_body(
    path: _MatmulChain,
    input_values: Sequence[tuple[Node, torch.Tensor]],
    *,
    norm_input: Node,
    rstd: Node,
    weight: Node,
    root_shape: Sequence[Any],
    width: int,
    pattern: str,
) -> GraphModule:
    logical_inputs = tuple(logical for logical, _ in input_values)

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

    return _trace_coda_values(
        body,
        tuple(value for _, value in input_values),
        pattern,
    )


def _forward_rmsnorm_fn_to_trace(
    hidden: torch.Tensor,
    partial: torch.Tensor,
    weight: torch.Tensor,
    *,
    hidden_shape: Sequence[Any],
    rstd_shape: Sequence[Any],
    eps: Any,
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    hidden = hidden.view(hidden_shape)
    rstd = torch.rsqrt(partial.mean(dim=-1, keepdim=True) + eps).view(rstd_shape)
    output = (hidden.float() * rstd * weight.float()).to(output_dtype)
    return hidden, output, rstd


def _backward_rmsnorm_fn_to_trace(
    grad: torch.Tensor,
    partial: torch.Tensor,
    norm_input: torch.Tensor,
    rstd: torch.Tensor,
    weight: torch.Tensor,
    *,
    grad_shape: Sequence[Any],
    rstd_shape: Sequence[Any],
    width: int,
    input_grad_dtype: torch.dtype,
    weight_grad_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    grad = grad.view(grad_shape).float()
    x_hat = norm_input.float() * rstd
    grad_x_hat = grad * weight.float()
    row_dot = partial.sum(dim=-1, keepdim=True).view(rstd_shape)
    input_grad = ((grad_x_hat - x_hat / width * row_dot) * rstd).to(input_grad_dtype)
    weight_grad = (
        (grad * x_hat).sum(dim=tuple(range(len(grad_shape) - 1))).to(weight_grad_dtype)
    )
    return input_grad, weight_grad


def _rewrite_projection_rmsnorm(
    gm: GraphModule,
    *,
    norm: Node,
    first_path: _MatmulChain,
    second_use: _MatmulUse,
    full_output: Node,
    nodes_to_remove: set[Node],
    projection_body: Callable[..., tuple[torch.Tensor, ...]],
    expansion_body: Callable[..., tuple[torch.Tensor]],
    replacement: Callable[..., tuple[torch.Tensor, ...]],
    pattern: str,
    autotune: bool = False,
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> bool:
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
            candidate_nodes_to_remove: set[Node],
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
                nodes_to_remove=candidate_nodes_to_remove,
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
        nodes_to_remove.update(
            (
                *first_path.nodes,
                norm,
                old_norm_out,
                *((old_rstd,) if old_rstd is not None else ()),
                second_mm,
            )
        )
        return True

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

    def prepare_weight(norm_weight: torch.Tensor) -> torch.Tensor:
        if full_width != norm_width:
            norm_weight = torch.nn.functional.pad(
                norm_weight,
                (0, full_width - norm_width),
                value=1.0,
            )
        return norm_weight.view(1, full_width).float()

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

    projection_values = tuple(
        _tensor_value(node) for node in (*first_path.root.all_input_nodes, weight)
    )
    if not all(isinstance(value, torch.Tensor) for value in projection_values):
        raise AssertionError(f"CODA {pattern} projection inputs require metadata")
    traced_projection_body = _trace_coda_values(
        configured_projection_body,
        (*projection_values[:-1], prepare_weight(projection_values[-1])),
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
        norm_weight: torch.Tensor,
        expansion_rhs: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return replacement(
            projection_lhs,
            projection_rhs,
            prepare_weight(norm_weight),
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
        weight,
        second_rhs,
    )
    new_full, new_norm_out, new_rstd, new_expansion = _insert_traced_replacement(
        gm,
        configured_replacement,
        replacement_inputs,
        (
            full_value,
            norm_output_value,
            logical_rstd_value,
            expansion_output_value,
        ),
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
    nodes_to_remove.update(
        (
            *first_path.nodes,
            norm,
            old_norm_out,
            *((old_rstd,) if old_rstd is not None else ()),
            second_mm,
        ),
    )
    return True


def _rewrite_forward_rmsnorm(
    gm: GraphModule,
    *,
    norm: Node,
    path: _MatmulChain,
    nodes_to_remove: set[Node],
    pattern: str,
    autotune: bool = False,
    kernel: str = "main",
    benchmark_regions: list[RewriteBenchmarkRegion] | None = None,
) -> bool:
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
            candidate_nodes_to_remove: set[Node],
        ) -> None:
            _rewrite_forward_rmsnorm(
                candidate,
                norm=candidate_nodes[norm.name],
                path=_MatmulChain(
                    candidate_nodes[path.root.name],
                    tuple(candidate_nodes[node.name] for node in path.nodes),
                ),
                nodes_to_remove=candidate_nodes_to_remove,
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
        nodes_to_remove.update(region_nodes)
        return True
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
    body_input_values = []
    for external in external_inputs:
        value = (
            _tensor_value(external)
            if external in root_inputs
            else _view_coda_value(external, root_shape)
        )
        if value is None:
            raise AssertionError(f"CODA input {external.name} requires tensor metadata")
        body_input_values.append((external, value))
    gemm_target = path.root.target
    if gemm_target not in {_MM_TARGET, _BMM_TARGET}:
        raise AssertionError(f"CODA {pattern} expected MM or BMM root")
    body = _trace_forward_rmsnorm_body(
        path,
        ordered,
        body_input_values,
        root_shape=root_shape,
        width=width,
        pattern=pattern,
    )
    mark_flex_gemm_body_gemm_node(body, gemm_target)
    root_value = _tensor_value(path.root)
    hidden_value = _tensor_value(hidden)
    norm_output_value = _tensor_value(norm_out)
    if root_value is None or hidden_value is None or norm_output_value is None:
        raise AssertionError(f"CODA {pattern} requires RMSNorm tensor metadata")
    rstd_shape = _shape(old_rstd) if old_rstd is not None else (*hidden_shape[:-1], 1)
    if rstd_shape is None:
        raise AssertionError(f"CODA {pattern} requires RMSNorm rstd shape metadata")
    rstd_value = (
        _tensor_value(old_rstd)
        if old_rstd is not None
        else root_value.new_empty(rstd_shape, dtype=torch.float32)
    )
    options = _kernel_options(
        pattern,
        kernel=kernel,
        root=path.root,
        autotune=autotune,
    )
    replacement_inputs = [*external_inputs]
    if weight not in replacement_inputs:
        replacement_inputs.append(weight)

    def replacement(*args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        env = dict(zip(replacement_inputs, args, strict=True))
        body_inputs = tuple(
            env[external] if external in root_inputs else env[external].view(root_shape)
            for external in external_inputs
        )
        hidden_physical, partial = flex_gemm_hop(
            gemm_target,
            body,
            body_inputs,
            {},
            options,
        )
        return _forward_rmsnorm_fn_to_trace(
            hidden_physical,
            partial,
            env[weight],
            hidden_shape=hidden_shape,
            rstd_shape=rstd_shape,
            eps=norm.args[3],
            output_dtype=norm_output_dtype,
        )

    new_hidden, new_norm_out, new_rstd = _insert_traced_replacement(
        gm,
        replacement,
        replacement_inputs,
        (hidden_value, norm_output_value, rstd_value),
        pattern=pattern,
    )
    _copy_meta(new_hidden, hidden)
    _copy_meta(new_norm_out, norm_out)
    if old_rstd is not None:
        _copy_meta(new_rstd, old_rstd)

    body_nodes = set(ordered)
    for user in list(hidden.users):
        if user not in body_nodes and user is not norm:
            user.replace_input_with(hidden, new_hidden)
    norm_out.replace_all_uses_with(new_norm_out)
    if old_rstd is not None:
        old_rstd.replace_all_uses_with(new_rstd)
    nodes_to_remove.update(
        (
            *path.nodes,
            norm,
            norm_out,
            *((old_rstd,) if old_rstd is not None else ()),
        ),
    )
    return True


def _apply_coda_pattern(
    gm: GraphModule,
    pattern: CodaPattern,
    *,
    autotune: bool = False,
) -> GraphModule:
    counts: Counter[str] = Counter({name: 0 for name in CODA_PATTERN_NAMES})
    counts.update(gm.meta.get("coda_pattern_counts", {}))
    nodes_to_remove: set[Node] = set()
    pattern.apply(
        gm,
        counts,
        nodes_to_remove,
        None,
        autotune,
    )
    _erase_nodes(gm, nodes_to_remove)
    _assign_coda_inductor_regions(gm)
    gm.meta["coda_pattern_counts"] = dict(counts)
    return gm


def _prepare_coda_candidates(
    gm: GraphModule,
    pattern: CodaPattern,
    *,
    autotune: bool,
) -> list[RewriteCandidate]:
    counts: Counter[str] = Counter({name: 0 for name in CODA_PATTERN_NAMES})
    counts.update(gm.meta.get("coda_pattern_counts", {}))
    prepared_candidates: list[RewriteCandidate] = []
    pattern.apply(
        gm,
        counts,
        set(),
        prepared_candidates,
        autotune,
    )
    return prepared_candidates


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
        elif kernel.pretuned_configs_by_shape:
            summary[name] = "QUACK shape-specific config"
        elif kernel.pretuned_configs:
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
    benchmark_graph_processor: Callable[
        [GraphModule, tuple[torch.Tensor, ...]], Callable[..., Any]
    ]
    | None,
) -> Callable:
    @functools.wraps(_apply_coda_pattern)
    def apply(
        gm: GraphModule,
        example_inputs: tuple | None = None,
    ) -> GraphModule:
        previous_count = gm.meta.get("coda_pattern_counts", {}).get(pattern.name, 0)
        logger.info(
            f"CODA {pattern.name} kernel policies: "
            f"{_kernel_policy_summary(pattern, autotune=coda_autotune)}"
        )
        if compile_time_benchmark:
            gm = apply_benchmarked_rewrites(
                gm,
                name=pattern.name,
                prepare_candidates=functools.partial(
                    _prepare_coda_candidates,
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
            _assign_coda_inductor_regions(gm)
        else:
            gm = _apply_coda_pattern(gm, pattern, autotune=coda_autotune)
        count = gm.meta.get("coda_pattern_counts", {}).get(pattern.name, 0)
        logger.info(f"CODA {pattern.name} applied {count - previous_count} groups")
        return gm

    apply.__name__ = pattern.name
    return apply


def coda_flex_gemm_pass(
    gm: GraphModule,
    example_inputs: tuple | None = None,
    *,
    patterns: Iterable[str] | None = None,
    compile_time_benchmark: bool = True,
    benchmark_strict: bool = False,
    coda_autotune: bool = False,
    benchmark_graph_processor: Callable[
        [GraphModule, tuple[torch.Tensor, ...]], Callable[..., Any]
    ]
    | None = None,
) -> GraphModule:
    """Apply canonical CODA patterns to a joint training graph in priority order."""
    enabled = set(CODA_PATTERN_NAMES if patterns is None else patterns)
    if unknown := enabled.difference(CODA_PATTERN_NAMES):
        raise ValueError(f"Unknown CODA pattern entries: {sorted(unknown)}")
    for name in CODA_PATTERN_NAMES:
        if name not in enabled:
            continue
        gm = _configured_coda_pass(
            _CODA_PATTERNS[name],
            compile_time_benchmark=compile_time_benchmark,
            benchmark_strict=benchmark_strict,
            coda_autotune=coda_autotune,
            benchmark_graph_processor=benchmark_graph_processor,
        )(gm, example_inputs)
    gm.recompile()
    return gm
