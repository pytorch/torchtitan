# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Split MoE token inputs before tracing and annotate the two chunk bodies."""

# Tensor axes: T = tokens, D = hidden width.

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn
from torch.fx.traceback import annotate_fn

from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_dense_sp_enabled, spmd_mesh_size
from torchtitan.experiments.graph_trainer.common_utils import matches_module_fqn_pattern
from torchtitan.experiments.graph_trainer.configs import (
    GraphTrainerCompileConfig,
    validate_ep_overlap_config,
)
from torchtitan.experiments.graph_trainer.simple_fsdp import _cached_unshard_parameters
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe.token_dispatcher import HybridEPTokenDispatcher

logger = logging.getLogger(__name__)


def _split_padding_mask(
    padding_mask_T: torch.Tensor, *, num_token_shards: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match two local token chunks while keeping the mask replicated.

    Dense SP partitions activations across TP before this wrapper. Split each
    shard's mask, then flatten: MoE's R -> S(0) transition must give each rank
    the mask for its own activation chunk, not a slice of a global half.
    """
    mask_ST = padding_mask_T.reshape(num_token_shards, -1)
    first_ST, second_ST = mask_ST.chunk(2, dim=1)
    return first_ST.reshape(-1), second_ST.reshape(-1)


class _EagerChunkedForward:
    """A MoE forward that preserves token order across two annotated calls."""

    def __init__(
        self,
        module: MoE,
        *,
        root_fqn: str,
    ) -> None:
        self.inner_forward = module.forward
        self.module = module
        self.root_fqn = root_fqn

    def __call__(
        self,
        x_TD: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
        aux_loss_denominator: torch.Tensor | None = None,
        **router_kwargs: Any,
    ) -> torch.Tensor:
        """Split token-aligned inputs and share all other router arguments.

        Args:
            x_TD: MoE input with an even token count and unchanged hidden width.
            padding_mask_T: Optional token-aligned mask, replicated across TP
                even when sequence parallelism shards the activation.
            aux_loss_denominator: Optional full-call denominator shared by chunks.
            **router_kwargs: Additional arguments forwarded unchanged to the router.

        Returns:
            Concatenated MoE outputs in the original token order.
        """
        if x_TD.ndim != 2:
            raise ValueError(
                f"EP overlap expects (T, D) input for {self.root_fqn}, "
                f"got shape {tuple(x_TD.shape)}"
            )
        num_tokens = x_TD.shape[0]
        torch._check(
            num_tokens % 2 == 0,
            lambda: f"EP overlap requires an even token count for {self.root_fqn}",
        )
        num_token_shards = (
            spmd_mesh_size(MeshAxisName.TP) if spmd_dense_sp_enabled() else 1
        )
        mask_num_tokens = num_tokens * num_token_shards
        if padding_mask_T is not None and padding_mask_T.shape != (mask_num_tokens,):
            raise ValueError(
                f"EP overlap padding mask for {self.root_fqn} must have shape "
                f"({mask_num_tokens},), got {tuple(padding_mask_T.shape)}"
            )

        boundary_meta = {
            "chunked_region_fqn": self.root_fqn,
            "chunked_region_role": "split_boundary",
        }
        split = annotate_fn(boundary_meta)(torch.split)
        chunk_sizes = [num_tokens // 2, num_tokens // 2]
        inputs = split(x_TD, chunk_sizes, dim=0)
        masks = (
            [None, None]
            if padding_mask_T is None
            else annotate_fn(boundary_meta)(_split_padding_mask)(
                padding_mask_T, num_token_shards=num_token_shards
            )
        )
        outputs = []
        with _cached_unshard_parameters(self.module):
            for chunk_id in (0, 1):
                body = annotate_fn(
                    {
                        "chunk_id": chunk_id,
                        "chunked_region_fqn": self.root_fqn,
                        "chunked_region_role": "body",
                    }
                )(self.inner_forward)
                kwargs = dict(router_kwargs)
                if masks[chunk_id] is not None:
                    kwargs["padding_mask_T"] = masks[chunk_id]
                if aux_loss_denominator is not None:
                    kwargs["aux_loss_denominator"] = aux_loss_denominator
                outputs.append(body(inputs[chunk_id], **kwargs))

        materialize = annotate_fn(
            {
                "chunked_region_fqn": self.root_fqn,
                "chunked_region_role": "materialization",
            }
        )(torch.cat)
        return materialize(outputs, dim=0)


def maybe_apply_ep_overlap_eager_chunking(
    model: nn.Module,
    compile_config: GraphTrainerCompileConfig,
) -> None:
    """Wrap selected MoE forwards once so tracing observes token chunking."""
    if not compile_config.ep_overlap.enabled:
        return
    module_fqn = validate_ep_overlap_config(compile_config.ep_overlap)
    matched = []
    for fqn, module in model.named_modules():
        if isinstance(module, MoE) and matches_module_fqn_pattern(module_fqn, fqn):
            dispatcher = module.routed_experts.token_dispatcher
            if not isinstance(dispatcher, HybridEPTokenDispatcher):
                raise ValueError(f"EP overlap requires HybridEP in {fqn}")
            if dispatcher.non_blocking_capacity_factor is None:
                raise ValueError(f"EP overlap requires non-blocking HybridEP in {fqn}")
            dispatcher.async_op = True
            if not isinstance(module.forward, _EagerChunkedForward):
                module.forward = _EagerChunkedForward(module, root_fqn=fqn)
            matched.append(fqn)
    if not matched:
        raise ValueError(
            f"EP overlap eager chunking matched no MoE modules: {module_fqn}"
        )
    logger.info("Applied eager EP token chunking to %d MoE modules", len(matched))


def populate_eager_chunk_metadata_pass(
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
