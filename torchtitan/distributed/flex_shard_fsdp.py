# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data parallelism with FlexShard in place of FSDP2's ``fully_shard``.

``apply_flex_shard_to_decoder`` gives FlexShard (meta-pytorch/flex_shard) one
``fsdp2_compatible`` bucket for each group that ``apply_fsdp_to_decoder`` passes
to ``fully_shard``, with the same placements, mixed precision and reshard
policy. The buckets' collectives then match FSDP2's byte for byte, so training
is bitwise identical. The one difference is that FSDP2 skips the embedding's
backward all-gather (``set_unshard_in_backward``), which FlexShard can't yet; it
gathers the embedding again in backward, with the same values. FlexShard is
imported only when ``parallelism.fsdp_backend='flex_shard'``.
"""

import logging
import sys
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan.distributed.fsdp import (
    get_fsdp_reshard_after_forward_policy,
    linear_param_shard_placements,
    routed_expert_param_placements,
)
from torchtitan.models.common.moe import MoE

if TYPE_CHECKING:
    from torchtitan.models.common.decoder import Decoder

__all__ = [
    "apply_flex_shard_to_decoder",
    "as_fsdp2_dtensor",
    "grad_as_fsdp2_dtensor",
]

logger = logging.getLogger(__name__)


def apply_flex_shard_to_decoder(
    model: "Decoder",
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    reshard_after_forward_policy: str = "default",
) -> None:
    """Shard a decoder with FlexShard as ``apply_fsdp_to_decoder`` does with FSDP2.

    Each ``fully_shard`` group becomes a bucket naming the same modules: the
    embedding (with the norm and output projection when weights are tied), the
    norm with the output projection, each transformer block, and the parameters
    left to the root group. Gradients are summed over ``dp_mesh`` without
    division, as ``disable_fsdp_gradient_division`` makes FSDP2 do.

    Args:
        model: The decoder to shard.
        dp_mesh: The 1D ``dp_shard`` mesh to shard over.
        param_dtype: The dtype of the unsharded parameters.
        reduce_dtype: The dtype of gradient reduction.
        pp_enabled: Whether pipeline parallelism is enabled.
        reshard_after_forward_policy: "default", "always" or "never", as for
            ``apply_fsdp_to_decoder``.
    """
    # flex_shard is an optional dependency, needed only for this backend.
    # pyrefly: ignore [missing-import]
    from flex_shard import BucketSpec, flex_shard, MixedPrecisionPolicy

    # pyrefly: ignore [missing-import]
    from flex_shard.custom_placements.shard import Shard as FlexShardShard

    if dp_mesh.mesh_dim_names != ("dp_shard",):
        raise ValueError(
            "FlexShard shards over a 1D mesh with the dp_shard axis, but got "
            f"mesh axes {dp_mesh.mesh_dim_names}."
        )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )
    # As mixed_precision_policy does for FSDP2, preserved parameters compute in
    # the wider of their own dtype and param_dtype.
    param_dtype_overrides = {
        fqn: dtype
        for fqn, param in model.named_preserve_dtype_params()
        if (dtype := torch.promote_types(param.dtype, param_dtype)) != param_dtype
    }
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        param_dtype_overrides=param_dtype_overrides or None,
    )
    # Shard dims other than 0, which FSDP2 gets from shard_placement_fn.
    shard_dims: dict[nn.Parameter, int] = {}

    def placement_fn(named_params, mesh):
        del mesh
        return {
            fqn: (FlexShardShard(shard_dims.get(param, 0)),)
            for fqn, param in named_params
        }

    buckets: list[BucketSpec] = []
    bucketed_params: set[nn.Parameter] = set()

    def add_bucket(patterns: list[str], reshard_after_forward: bool) -> None:
        buckets.append(
            BucketSpec(
                patterns,
                placement_fn=placement_fn,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                gradient_divide_factor=1.0,
                reshard_after_forward=reshard_after_forward,
                fsdp2_compatible=True,
            )
        )

    def add_module_bucket(module_fqns: list[str], reshard_after_forward: bool) -> None:
        for fqn in module_fqns:
            bucketed_params.update(model.get_submodule(fqn).parameters())
        add_bucket(module_fqns, reshard_after_forward)

    if model.enable_weight_tying:
        add_module_bucket(
            [
                name
                for name in ("tok_embeddings", "norm", "lm_head")
                if getattr(model, name) is not None
            ],
            reshard_after_forward_policy == "always",
        )
    else:
        if model.tok_embeddings is not None:
            add_module_bucket(["tok_embeddings"], reshard_after_forward)
        if model.norm is not None and model.lm_head is not None:
            add_module_bucket(
                ["norm", "lm_head"], reshard_after_forward_policy == "always"
            )
    for layer_id, transformer_block in model.layers.items():
        placements = linear_param_shard_placements(transformer_block)
        moe = next(
            (m for m in transformer_block.children() if isinstance(m, MoE)), None
        )
        if moe is not None:
            placements.update(
                routed_expert_param_placements(
                    moe.routed_experts,
                    num_experts=moe.num_experts,
                    expert_sharding_size=dp_mesh.size(),
                )
            )
        shard_dims.update((param, p.dim) for param, p in placements.items())
        add_module_bucket([f"layers.{layer_id}"], reshard_after_forward)
    # fully_shard(model) puts the remaining parameters in the root group, which
    # FSDP2 does not reshard after forward.
    root_fqns = [
        fqn for fqn, param in model.named_parameters() if param not in bucketed_params
    ]
    if root_fqns:
        add_bucket(root_fqns, reshard_after_forward=False)

    flex_shard(model, buckets=buckets)
    logger.info("Applied FlexShard to the model")


def as_fsdp2_dtensor(param: torch.Tensor) -> torch.Tensor:
    """Return a FlexShard parameter as the DTensor FSDP2 would give it.

    FlexShard stores a parameter as its plain local shard. The returned DTensor
    views that shard, on its bucket's mesh with its ``Shard`` placement, so code
    written for FSDP2's DTensor parameters (e.g. parallelism-agnostic
    initialization) behaves the same. Other tensors are returned unchanged.
    FlexShard parameters exist only once ``flex_shard`` is imported.
    """
    return _as_fsdp2_dtensor(param, param)


def grad_as_fsdp2_dtensor(param: torch.Tensor) -> torch.Tensor | None:
    """Return ``param``'s grad, as the DTensor FSDP2 would give it.

    Viewed as the DTensors FSDP2 would produce, FlexShard grads reduce their
    norm over their meshes as FSDP2's grads do. Other grads, and ``None``, are
    returned unchanged.
    """
    if param.grad is None:
        return None
    return _as_fsdp2_dtensor(param.grad, param)


def _as_fsdp2_dtensor(tensor: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """View ``tensor``, laid out as ``param``'s local shard, as a DTensor if
    ``param`` is a FlexShard parameter."""
    flex_shard = sys.modules.get("flex_shard")
    if flex_shard is None or not flex_shard.is_flex_shard_param(param):
        return tensor
    global_shape = flex_shard.get_global_shape(param)
    return DTensor.from_local(
        tensor,
        flex_shard.get_mesh(param),
        tuple(Shard(p.dim) for p in flex_shard.get_placements(param)),
        run_check=False,
        shape=global_shape,
        stride=make_contiguous_strides_for(global_shape),
    )
