# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager producer for the EP-overlap chunk metadata contract."""

from __future__ import annotations

import fnmatch
from typing import Any

import torch
import torch.nn as nn
from torch.fx.traceback import annotate_fn

from torchtitan.experiments.graph_trainer.configs import (
    EpOverlapChunkDim,
    validate_ep_overlap_config,
)


_WRAPPED_ATTR = "_torchtitan_ep_overlap_eager_chunked"


def _matches_module_fqn(pattern: str, fqn: str) -> bool:
    pattern_parts = pattern.split(".")
    fqn_parts = fqn.split(".")
    return len(pattern_parts) == len(fqn_parts) and all(
        fnmatch.fnmatchcase(fqn_part, pattern_part)
        for pattern_part, fqn_part in zip(pattern_parts, fqn_parts)
    )


def _chunk_dim_index(mode: EpOverlapChunkDim) -> int:
    return 0 if mode == "batch" else 1


def _cat_chunked_outputs(outputs: list[Any], dim: int) -> Any:
    first = outputs[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(outputs, dim=dim)
    if isinstance(first, tuple):
        return tuple(
            _cat_chunked_outputs([output[i] for output in outputs], dim)
            for i in range(len(first))
        )
    if isinstance(first, list):
        return [
            _cat_chunked_outputs([output[i] for output in outputs], dim)
            for i in range(len(first))
        ]
    raise TypeError(
        "ep_overlap eager chunking only supports Tensor, tuple, or list outputs"
    )


def _wrap_forward_with_eager_chunking(
    module: nn.Module,
    *,
    root_fqn: str,
    chunk_dim: EpOverlapChunkDim,
) -> None:
    if getattr(module, _WRAPPED_ATTR, False):
        return

    dim = _chunk_dim_index(chunk_dim)
    inner_forward = module.forward

    def chunked_forward(*args: Any, **kwargs: Any) -> Any:
        tensor_inputs = [
            value
            for value in [*args, *kwargs.values()]
            if isinstance(value, torch.Tensor) and value.dim() > dim
        ]
        if not tensor_inputs:
            raise ValueError(
                f"ep_overlap eager chunking found no tensor inputs for {root_fqn!r}"
            )
        full_extent = tensor_inputs[0].shape[dim]
        if full_extent % 2 != 0:
            raise ValueError(
                f"ep_overlap eager chunking requires an even {chunk_dim} extent "
                f"for {root_fqn!r}, got {full_extent}"
            )
        chunk_extent = full_extent // 2

        def split_if_chunked(value: Any) -> list[Any]:
            if (
                isinstance(value, torch.Tensor)
                and value.dim() > dim
                and value.shape[dim] == full_extent
            ):
                split = annotate_fn(
                    {
                        "chunked_region_fqn": root_fqn,
                        "chunked_region_role": "split_boundary",
                    }
                )(torch.split)
                # MoE blocks flatten activations with view(); seq chunks are
                # non-contiguous views, so materialize the chunk boundary.
                return [
                    chunk.contiguous()
                    for chunk in split(value, [chunk_extent, chunk_extent], dim=dim)
                ]
            return [value, value]

        split_args = [split_if_chunked(arg) for arg in args]
        split_kwargs = {key: split_if_chunked(value) for key, value in kwargs.items()}
        outputs = []
        for chunk_id in (0, 1):
            body = annotate_fn(
                {
                    "chunk_id": chunk_id,
                    "chunked_region_fqn": root_fqn,
                    "chunked_region_role": "body",
                }
            )(inner_forward)
            outputs.append(
                body(
                    *(chunks[chunk_id] for chunks in split_args),
                    **{key: chunks[chunk_id] for key, chunks in split_kwargs.items()},
                )
            )

        materialize = annotate_fn(
            {
                "chunked_region_fqn": root_fqn,
                "chunked_region_role": "materialization",
            }
        )(_cat_chunked_outputs)
        return materialize(outputs, dim)

    setattr(chunked_forward, _WRAPPED_ATTR, True)
    module.forward = chunked_forward
    setattr(module, _WRAPPED_ATTR, True)


def apply_ep_overlap_eager_chunking(
    model: nn.Module,
    compile_config: Any,
) -> None:
    """Wrap selected module forwards so tracing observes eager chunking."""
    if "ep_overlap" not in getattr(compile_config, "passes", []):
        return
    chunk_dim, chunk_strategy, module_fqn = validate_ep_overlap_config(compile_config)
    if chunk_strategy != "eager":
        return

    matched = False
    for fqn, module in model.named_modules():
        if _matches_module_fqn(module_fqn, fqn):
            if getattr(module, "moe_enabled", True) is False:
                continue
            _wrap_forward_with_eager_chunking(
                module,
                root_fqn=fqn,
                chunk_dim=chunk_dim,
            )
            matched = True
    if not matched:
        raise ValueError(f"ep_overlap eager chunking matched no modules: {module_fqn}")


def import_eager_chunk_metadata_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
) -> torch.fx.GraphModule:
    """Promote eager traceback chunk annotations into pass metadata."""
    del example_inputs
    for node in gm.graph.nodes:
        custom = node.meta.get("custom")
        if not isinstance(custom, dict):
            continue
        for key in ("chunk_id", "chunked_region_fqn", "chunked_region_role"):
            if key in custom:
                node.meta[key] = custom[key]
    return gm
