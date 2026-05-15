# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager producer for the EP-overlap chunk metadata contract.

Algorithm:
1. Validate the EP-overlap config and find modules whose FQN matches the single
   configured pattern.
2. Replace each selected forward with a thin wrapper that splits matching tensor
   inputs into two equal chunks.
3. Call the original forward once per chunk under traceback annotations.
4. Cat Tensor outputs recursively and populate those traceback annotations into
   normal node metadata after tracing.
"""

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
from torchtitan.tools.logging import logger


_WRAPPED_ATTR = "_torchtitan_ep_overlap_eager_chunked"


def _matches_module_fqn(pattern: str, fqn: str) -> bool:
    """Match one module FQN against a component-wise fnmatch pattern."""
    pattern_parts = pattern.split(".")
    fqn_parts = fqn.split(".")
    return len(pattern_parts) == len(fqn_parts) and all(
        fnmatch.fnmatchcase(fqn_part, pattern_part)
        for pattern_part, fqn_part in zip(pattern_parts, fqn_parts)
    )


def _chunk_dim_index(mode: EpOverlapChunkDim) -> int:
    """Map logical EP chunk mode to the tensor dimension used by eager wrappers."""
    return 0 if mode == "batch" else 1


def _cat_chunked_outputs(outputs: list[Any], dim: int, root_fqn: str) -> Any:
    """Recursively cat Tensor outputs while preserving tuple/list structure."""
    first = outputs[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(outputs, dim=dim)
    if isinstance(first, tuple):
        if any(
            not isinstance(output, tuple) or len(output) != len(first)
            for output in outputs
        ):
            raise TypeError(
                "ep_overlap eager chunking expected matching tuple outputs "
                f"for {root_fqn!r}"
            )
        return tuple(
            _cat_chunked_outputs([output[i] for output in outputs], dim, root_fqn)
            for i in range(len(first))
        )
    if isinstance(first, list):
        if any(
            not isinstance(output, list) or len(output) != len(first)
            for output in outputs
        ):
            raise TypeError(
                "ep_overlap eager chunking expected matching list outputs "
                f"for {root_fqn!r}"
            )
        return [
            _cat_chunked_outputs([output[i] for output in outputs], dim, root_fqn)
            for i in range(len(first))
        ]
    raise TypeError(
        "ep_overlap eager chunking only supports Tensor, tuple, or list outputs "
        f"for {root_fqn!r}; got {type(first).__name__}"
    )


def _wrap_forward_with_eager_chunking(
    module: nn.Module,
    *,
    root_fqn: str,
    chunk_dim: EpOverlapChunkDim,
) -> None:
    """Install an idempotent two-chunk forward wrapper on ``module``.

    Eager chunking deliberately uses a narrow same-extent split rule for the
    selected `layers.*` TransformerBlock and `layers.*.moe` MoE roots. Those
    module inputs are activation-shaped in the supported model sources. This
    rule is not a general module wrapper contract.
    """
    if getattr(module, _WRAPPED_ATTR, False):
        return

    dim = _chunk_dim_index(chunk_dim)
    inner_forward = module.forward

    def chunked_forward(*args: Any, **kwargs: Any) -> Any:
        # Step 2: infer the full extent from the first tensor input that carries
        # the requested logical chunk dimension.
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
        torch._check(
            full_extent % 2 == 0,
            lambda: (
                f"ep_overlap eager chunking requires an even {chunk_dim} extent "
                f"for {root_fqn!r}, got {full_extent}"
            ),
        )
        chunk_extent = full_extent // 2

        def split_if_chunked(value: Any) -> list[Any]:
            """Split tensors with the selected full extent; duplicate others."""
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

        # Step 3: call the original module body once per chunk under metadata
        # annotations consumed by ``populate_eager_chunk_metadata_pass``.
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

        # Step 4: materialize the module output back to the full tensor form.
        materialize = annotate_fn(
            {
                "chunked_region_fqn": root_fqn,
                "chunked_region_role": "materialization",
            }
        )(_cat_chunked_outputs)
        return materialize(outputs, dim, root_fqn)

    setattr(chunked_forward, _WRAPPED_ATTR, True)
    module.forward = chunked_forward
    setattr(module, _WRAPPED_ATTR, True)
    logger.debug("Installed eager EP chunk wrapper on %s", root_fqn)


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

    matched: list[str] = []
    for fqn, module in model.named_modules():
        if _matches_module_fqn(module_fqn, fqn):
            if getattr(module, "moe_enabled", True) is False:
                continue
            _wrap_forward_with_eager_chunking(
                module,
                root_fqn=fqn,
                chunk_dim=chunk_dim,
            )
            matched.append(fqn)
    if not matched:
        raise ValueError(f"ep_overlap eager chunking matched no modules: {module_fqn}")
    logger.info(
        "Applied eager EP chunking to %d module(s): pattern=%s, chunk_dim=%s",
        len(matched),
        module_fqn,
        chunk_dim,
    )


def populate_eager_chunk_metadata_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
) -> torch.fx.GraphModule:
    """Promote eager traceback chunk annotations into pass metadata."""
    del example_inputs
    imported = 0
    for node in gm.graph.nodes:
        custom = node.meta.get("custom")
        if not isinstance(custom, dict):
            continue
        for key in ("chunk_id", "chunked_region_fqn", "chunked_region_role"):
            if key in custom:
                node.meta[key] = custom[key]
        if "chunked_region_role" in node.meta:
            node.meta["chunked_region_producer"] = "eager"
            imported += 1
    if imported:
        logger.debug("Populated eager EP chunk metadata for %d node(s)", imported)
    return gm
