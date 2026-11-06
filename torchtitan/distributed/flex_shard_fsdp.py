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
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from torchtitan.distributed.fsdp import decoder_fsdp_groups

if TYPE_CHECKING:
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.loss import Loss
    from torchtitan.components.optim import Optim
    from torchtitan.config.configs import DebugConfig, TrainingConfig
    from torchtitan.models.common.decoder import Decoder

__all__ = [
    "apply_flex_shard_to_decoder",
    "as_fsdp2_dtensor",
    "validate_flex_shard_training",
]

logger = logging.getLogger(__name__)


def apply_flex_shard_to_decoder(
    model: "Decoder",
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    reshard_after_forward_policy: str,
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

    # ParallelismConfig allows only dp_shard with flex_shard.
    assert dp_mesh.mesh_dim_names == ("dp_shard",), dp_mesh.mesh_dim_names
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
    embedding, head, blocks = decoder_fsdp_groups(
        model,
        dp_mesh,
        reshard_after_forward_policy=reshard_after_forward_policy,
        pp_enabled=pp_enabled,
    )
    groups = [group for group in (embedding, head) if group is not None] + blocks
    # Shard dims other than 0, which FSDP2 gets from shard_placement_fn.
    shard_dims = {
        param: placement.dim
        for group in groups
        for param, placement in group.placements.items()
    }

    def placement_fn(named_params, mesh):
        del mesh
        return {
            fqn: (FlexShardShard(shard_dims.get(param, 0)),)
            for fqn, param in named_params
        }

    def bucket(patterns: list[str], reshard_after_forward: bool) -> BucketSpec:
        return BucketSpec(
            patterns,
            placement_fn=placement_fn,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            gradient_divide_factor=1.0,
            reshard_after_forward=reshard_after_forward,
            fsdp2_compatible=True,
        )

    module_fqns = {module: fqn for fqn, module in model.named_modules()}
    buckets = [
        bucket(
            [module_fqns[module] for module in group.modules],
            group.reshard_after_forward,
        )
        for group in groups
    ]
    # fully_shard(model) puts the remaining parameters in the root group, which
    # FSDP2 does not reshard after forward.
    grouped_params = {
        param
        for group in groups
        for module in group.modules
        for param in module.parameters()
    }
    root_fqns = [
        fqn for fqn, param in model.named_parameters() if param not in grouped_params
    ]
    if root_fqns:
        buckets.append(bucket(root_fqns, reshard_after_forward=False))

    flex_shard(model, buckets=buckets)
    logger.info("Applied FlexShard to the model")


def as_fsdp2_dtensor(tensor: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """View ``tensor``, laid out as ``param``'s local shard, as the DTensor
    FSDP2 would give it if ``param`` is a FlexShard parameter.

    FlexShard stores a parameter, and its grad, as the plain local shard. The
    returned DTensor views that shard on its bucket's mesh with its ``Shard``
    placement, so code written for FSDP2's DTensor parameters and grads (e.g.
    parallelism-agnostic initialization, grad-norm clipping) behaves the same.
    Other tensors are returned unchanged.
    """
    # TODO: Use FlexShard's own DTensor view of a parameter once it has one.
    # FlexShard parameters exist only once flex_shard is imported, which
    # happens only when it is selected.
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
        stride=torch.empty(global_shape, device="meta").stride(),
    )


def validate_flex_shard_training(
    *,
    training: "TrainingConfig",
    loss: "Loss.Config",
    checkpointer: "CheckpointManager.Config | None",
    optim: "Optim.Config",
    debug: "DebugConfig",
) -> None:
    """Reject training features that ``parallelism.fsdp_backend='flex_shard'``
    does not support yet."""
    from torchtitan.components.loss import ChunkedLossWrapper
    from torchtitan.components.optim.optimizer import DistMuon
    from torchtitan.distributed.accelerator_graph import accelerator_graphs_supported

    unsupported = []
    if not training.disable_accelerator_graphs and accelerator_graphs_supported():
        unsupported.append(
            "accelerator graphs (set training.disable_accelerator_graphs=True)"
        )
    if isinstance(loss, ChunkedLossWrapper.Config):
        unsupported.append("ChunkedLossWrapper")
    if checkpointer is not None:
        unsupported.append("checkpointing")
    if optim.ema is not None:
        unsupported.append("optim.ema")
    if any(isinstance(o, DistMuon.Config) for o in optim.optimizer.optimizers):
        # DistMuon needs DTensor parameters.
        unsupported.append("DistMuon")
    if debug.spmd_typechecking:
        unsupported.append("debug.spmd_typechecking")
    if unsupported:
        raise ValueError(
            "parallelism.fsdp_backend='flex_shard' does not support "
            f"{', '.join(unsupported)} yet."
        )
