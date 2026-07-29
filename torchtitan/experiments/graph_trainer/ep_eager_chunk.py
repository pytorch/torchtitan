# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager producer for the EP-overlap chunk metadata contract.

Eager chunking is intentionally narrow: it supports the model input contracts
used by ``layers.*`` TransformerBlock roots and ``layers.*.moe`` MoE roots, and
fails loudly if those upstream forward signatures drift.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.fx.traceback import annotate_fn

from torchtitan.experiments.graph_trainer.common_utils import matches_module_fqn_pattern
from torchtitan.experiments.graph_trainer.configs import (
    EpOverlapChunkDim,
    GraphTrainerCompileConfig,
    MOE_BLOCK_FQN,
    TRANSFORMER_BLOCK_FQN,
    validate_ep_overlap_config,
)
from torchtitan.models.common.decoder import TransformerBlock
from torchtitan.tools.logging import logger


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


def _describe_value(value: Any) -> str:
    if isinstance(value, torch.Tensor):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device})"
        )
    if isinstance(value, dict):
        items = ", ".join(
            f"{key!r}: {_describe_value(item)}" for key, item in value.items()
        )
        return f"dict({{{items}}})"
    return type(value).__name__


def _describe_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    arg_desc = ", ".join(_describe_value(arg) for arg in args)
    kwarg_desc = ", ".join(
        f"{key}={_describe_value(value)}" for key, value in kwargs.items()
    )
    return f"args=[{arg_desc}], kwargs={{{kwarg_desc}}}"


def _expected_contract(root_kind: str) -> str:
    if root_kind == "moe":
        return "MoE.forward(x_BLD: Tensor[B, L, D])"
    return (
        "TransformerBlock.forward(x: Tensor[B, L, D], "
        "attention_masks: BlockMask|dict[BlockMask]|None, "
        "positions: Tensor[B, L]|None)"
    )


def _contract_error(
    *,
    root_fqn: str,
    root_kind: str,
    chunk_dim: EpOverlapChunkDim,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    reason: str,
) -> ValueError:
    return ValueError(
        "ep_overlap eager chunking only supports the current "
        f"{_expected_contract(root_kind)} contract for root {root_fqn!r} "
        f"with chunk_dim={chunk_dim!r}. {reason}. Observed "
        f"{_describe_inputs(args, kwargs)}. If this is a real model path, "
        "the upstream MoE.forward or TransformerBlock.forward contract likely "
        "changed and the eager EP chunking wrapper must be updated."
    )


class _EagerChunkedForward:
    def __init__(
        self,
        inner_forward: Callable[..., Any],
        *,
        root_fqn: str,
        chunk_dim: EpOverlapChunkDim,
        root_kind: str,
    ) -> None:
        self.inner_forward = inner_forward
        self.root_fqn = root_fqn
        self.chunk_dim = chunk_dim
        self.root_kind = root_kind
        self.dim = _chunk_dim_index(chunk_dim)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        def split_block_mask(value: Any) -> list[Any] | None:
            from torch.nn.attention.flex_attention import BlockMask

            if isinstance(value, BlockMask):
                from torch.distributed.pipelining.microbatch import _split_block_mask

                # Pipeline microbatching owns the BlockMask metadata splitting
                # logic, including batch-index offsetting inside mask_mod.
                return _split_block_mask(value, 2)
            if isinstance(value, dict) and any(
                isinstance(item, BlockMask) for item in value.values()
            ):
                split_items = {
                    key: split_block_mask(item) or [item, item]
                    for key, item in value.items()
                }
                return [
                    {key: chunks[chunk_id] for key, chunks in split_items.items()}
                    for chunk_id in (0, 1)
                ]
            return None

        def split_tensor(
            value: torch.Tensor, *, logical_name: str
        ) -> list[torch.Tensor]:
            if value.dim() <= self.dim:
                raise _contract_error(
                    root_fqn=self.root_fqn,
                    root_kind=self.root_kind,
                    chunk_dim=self.chunk_dim,
                    args=args,
                    kwargs=kwargs,
                    reason=(
                        f"{logical_name} rank must include selected dim {self.dim}"
                    ),
                )
            full_extent = value.shape[self.dim]
            torch._check(
                full_extent % 2 == 0,
                lambda: (
                    f"ep_overlap eager chunking requires an even {self.chunk_dim} "
                    f"extent for {logical_name} in {self.root_fqn!r}, "
                    f"got {full_extent}"
                ),
            )
            chunk_extent = full_extent // 2
            split = annotate_fn(
                {
                    "chunked_region_fqn": self.root_fqn,
                    "chunked_region_role": "split_boundary",
                }
            )(torch.split)
            # MoE blocks flatten activations with view(); seq chunks are
            # non-contiguous views, so materialize the chunk boundary.
            return [
                chunk.contiguous()
                for chunk in split(value, [chunk_extent, chunk_extent], dim=self.dim)
            ]

        def split_moe_inputs() -> tuple[list[list[Any]], dict[str, list[Any]]]:
            if len(args) != 1 or kwargs or not isinstance(args[0], torch.Tensor):
                raise _contract_error(
                    root_fqn=self.root_fqn,
                    root_kind=self.root_kind,
                    chunk_dim=self.chunk_dim,
                    args=args,
                    kwargs=kwargs,
                    reason="expected exactly one positional activation tensor",
                )
            return [split_tensor(args[0], logical_name="x_BLD")], {}

        def split_transformer_block_inputs() -> tuple[
            list[list[Any]], dict[str, list[Any]]
        ]:
            if self.chunk_dim != "batch":
                raise _contract_error(
                    root_fqn=self.root_fqn,
                    root_kind=self.root_kind,
                    chunk_dim=self.chunk_dim,
                    args=args,
                    kwargs=kwargs,
                    reason="TransformerBlock eager chunking only supports batch chunking",
                )
            if not args or len(args) > 3 or not isinstance(args[0], torch.Tensor):
                raise _contract_error(
                    root_fqn=self.root_fqn,
                    root_kind=self.root_kind,
                    chunk_dim=self.chunk_dim,
                    args=args,
                    kwargs=kwargs,
                    reason=(
                        "expected positional x tensor followed by optional "
                        "attention_masks and positions"
                    ),
                )
            if any(key not in ("attention_masks", "positions") for key in kwargs):
                raise _contract_error(
                    root_fqn=self.root_fqn,
                    root_kind=self.root_kind,
                    chunk_dim=self.chunk_dim,
                    args=args,
                    kwargs=kwargs,
                    reason="unexpected keyword argument",
                )
            if len(args) > 1 and "attention_masks" in kwargs:
                raise _contract_error(
                    root_fqn=self.root_fqn,
                    root_kind=self.root_kind,
                    chunk_dim=self.chunk_dim,
                    args=args,
                    kwargs=kwargs,
                    reason="attention_masks was passed both positionally and by keyword",
                )
            if len(args) > 2 and "positions" in kwargs:
                raise _contract_error(
                    root_fqn=self.root_fqn,
                    root_kind=self.root_kind,
                    chunk_dim=self.chunk_dim,
                    args=args,
                    kwargs=kwargs,
                    reason="positions was passed both positionally and by keyword",
                )

            def split_attention_masks(value: Any) -> list[Any]:
                if value is None:
                    return [None, None]
                chunks = split_block_mask(value)
                if chunks is not None:
                    return chunks
                raise _contract_error(
                    root_fqn=self.root_fqn,
                    root_kind=self.root_kind,
                    chunk_dim=self.chunk_dim,
                    args=args,
                    kwargs=kwargs,
                    reason=(
                        "attention_masks must be None, BlockMask, or dict "
                        "containing BlockMask"
                    ),
                )

            def split_positions(value: Any) -> list[Any]:
                if value is None:
                    return [None, None]
                if isinstance(value, torch.Tensor):
                    return split_tensor(value, logical_name="positions")
                raise _contract_error(
                    root_fqn=self.root_fqn,
                    root_kind=self.root_kind,
                    chunk_dim=self.chunk_dim,
                    args=args,
                    kwargs=kwargs,
                    reason="positions must be None or a tensor",
                )

            split_args = [split_tensor(args[0], logical_name="x")]
            if len(args) > 1:
                split_args.append(split_attention_masks(args[1]))
            if len(args) > 2:
                split_args.append(split_positions(args[2]))
            split_kwargs = {
                key: (
                    split_attention_masks(value)
                    if key == "attention_masks"
                    else split_positions(value)
                )
                for key, value in kwargs.items()
            }
            return split_args, split_kwargs

        if self.root_kind == "moe":
            split_args, split_kwargs = split_moe_inputs()
        else:
            split_args, split_kwargs = split_transformer_block_inputs()

        from torchtitan.distributed.minimal_async_ep.api import _use_buffer_set

        outputs = []
        for chunk_id in (0, 1):
            body = annotate_fn(
                {
                    "chunk_id": chunk_id,
                    "chunked_region_fqn": self.root_fqn,
                    "chunked_region_role": "body",
                }
            )(self.inner_forward)
            with _use_buffer_set(chunk_id):
                outputs.append(
                    body(
                        *(chunks[chunk_id] for chunks in split_args),
                        **{
                            key: chunks[chunk_id]
                            for key, chunks in split_kwargs.items()
                        },
                    )
                )

        materialize = annotate_fn(
            {
                "chunked_region_fqn": self.root_fqn,
                "chunked_region_role": "materialization",
            }
        )(_cat_chunked_outputs)
        return materialize(outputs, self.dim, self.root_fqn)


def _wrap_forward_with_eager_chunking(
    module: nn.Module,
    *,
    root_fqn: str,
    chunk_dim: EpOverlapChunkDim,
    root_kind: str,
) -> None:
    """Install an idempotent two-chunk forward wrapper on ``module``."""
    if isinstance(module.forward, _EagerChunkedForward):
        return

    module.forward = _EagerChunkedForward(
        module.forward,
        root_fqn=root_fqn,
        chunk_dim=chunk_dim,
        root_kind=root_kind,
    )
    logger.debug("Installed eager EP chunk wrapper on %s", root_fqn)


def maybe_apply_ep_overlap_eager_chunking(
    model: nn.Module,
    compile_config: GraphTrainerCompileConfig,
) -> None:
    """Wrap selected module forwards so tracing observes eager chunking."""
    if not compile_config.enable or not compile_config.ep_overlap.enabled:
        return
    chunk_dim, chunk_strategy, module_fqn = validate_ep_overlap_config(
        compile_config.ep_overlap
    )
    if chunk_strategy != "eager":
        return

    root_kind = "moe" if module_fqn == MOE_BLOCK_FQN else "transformer_block"
    matched: list[str] = []
    for fqn, module in model.named_modules():
        if matches_module_fqn_pattern(module_fqn, fqn):
            if (
                module_fqn == TRANSFORMER_BLOCK_FQN
                and isinstance(module, TransformerBlock)
                and getattr(module, "moe", None) is None
            ):
                continue
            _wrap_forward_with_eager_chunking(
                module,
                root_fqn=fqn,
                chunk_dim=chunk_dim,
                root_kind=root_kind,
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
