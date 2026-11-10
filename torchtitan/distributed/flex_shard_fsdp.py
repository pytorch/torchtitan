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
is bitwise identical. Two differences change only memory and communication.
FSDP2 skips the embedding's backward all-gather (``set_unshard_in_backward``),
which FlexShard can't yet; it gathers the embedding again in backward, with the
same values. And FlexShard has no equivalent of ``set_is_last_backward``: during
gradient accumulation, every backward waits for its gradient reductions and
releases its prefetch. FlexShard is imported only when
``parallelism.fsdp_backend`` is a ``FlexShardConfig``.
"""

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

if TYPE_CHECKING:
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.loss import Loss
    from torchtitan.components.optim import Optim
    from torchtitan.config.configs import DebugConfig, TrainingConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.common.decoder import Decoder

__all__ = [
    "FlexShardConfig",
    "apply_flex_shard_to_decoder",
    "as_fsdp2_dtensor",
    "validate_flex_shard_training",
]

logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class FlexShardConfig:
    """Shard parameters with FlexShard (meta-pytorch/flex_shard) in place of
    FSDP2's ``fully_shard``.

    Each FSDP2 group becomes an ``fsdp2_compatible`` bucket with the FSDP
    settings in ``parallelism``, so training is bitwise identical to FSDP2.
    """


def apply_flex_shard_to_decoder(
    model: "Decoder",
    dp_mesh: DeviceMesh,
    *,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward_policy: str,
) -> None:
    """Shard a decoder with FlexShard as ``apply_fsdp_to_decoder`` does with FSDP2.

    Each ``fully_shard`` group becomes a bucket naming the same modules: the
    embedding, the norm with the output projection, each transformer block, and
    the parameters left to the root group. Gradients are summed over ``dp_mesh``
    without division, as ``disable_fsdp_gradient_division`` makes FSDP2 do.

    Args:
        model: The decoder to shard.
        dp_mesh: The 1D ``dp_shard`` mesh to shard over.
        param_dtype: The dtype of the unsharded parameters.
        reduce_dtype: The dtype of gradient reduction.
        reshard_after_forward_policy: "default", "always" or "never", as for
            ``apply_fsdp_to_decoder``.
    """
    from torchtitan.distributed.fsdp import decoder_fsdp_groups
    from torchtitan.quantization._fsdp_tensor import _ShardedFSDPTensor

    # FlexShard's Shard placement all-gathers a bucket in one dtype.
    if any(model.named_preserve_dtype_params()):
        raise ValueError(
            "parallelism.fsdp_backend=FlexShardConfig() does not support "
            "preserve_dtype_params yet."
        )
    # FlexShard stores parameters as plain tensors, which would drop the
    # quantized all-gather that MXFP8 and NVFP4 weights implement for FSDP2.
    quantized = [
        fqn
        for fqn, param in model.named_parameters()
        if isinstance(param, _ShardedFSDPTensor)
    ]
    if quantized:
        raise ValueError(
            "parallelism.fsdp_backend=FlexShardConfig() does not support quantized "
            f"weights (MXFP8, NVFP4) yet; got {quantized[0]} and "
            f"{len(quantized) - 1} more."
        )

    # flex_shard is an optional dependency, needed only for this backend.
    # pyrefly: ignore [missing-import]
    from flex_shard import BucketSpec, flex_shard, MixedPrecisionPolicy

    # pyrefly: ignore [missing-import]
    from flex_shard.custom_placements.shard import Shard as FlexShardShard

    # validate_flex_shard_training allows only dp_shard.
    assert dp_mesh.mesh_dim_names == ("dp_shard",), dp_mesh.mesh_dim_names
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    embedding, head, blocks = decoder_fsdp_groups(
        model,
        dp_mesh,
        reshard_after_forward_policy=reshard_after_forward_policy,
        # validate_flex_shard_training rejects pipeline parallelism.
        pp_enabled=False,
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
    # TODO(<flex_shard issue link>): Use FlexShard's own DTensor view of a
    # parameter once it has one.
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
    parallelism: "ParallelismConfig",
    training: "TrainingConfig",
    loss: "Loss.Config",
    checkpointer: "CheckpointManager.Config | None",
    optim: "Optim.Config",
    debug: "DebugConfig",
) -> None:
    """Reject parallelisms and training features that
    FlexShard does not support yet.

    These checks run from ``TrainingEngine.Config.__post_init__``, which the
    config loader runs again after recipes assign fields, rather than from
    ``ParallelismConfig.__post_init__``, which runs only at construction.
    """
    from torchtitan.components.loss import ChunkedLossWrapper
    from torchtitan.components.optim.optimizer import DistMuon
    from torchtitan.distributed.accelerator_graph import accelerator_graphs_supported

    unsupported = [
        f"parallelism.{name}={degree}"
        for name, degree in (
            (
                "data_parallel_replicate_degree",
                parallelism.data_parallel_replicate_degree,
            ),
            ("tensor_parallel_degree", parallelism.tensor_parallel_degree),
            ("context_parallel_degree", parallelism.context_parallel_degree),
            ("expert_parallel_degree", parallelism.expert_parallel_degree),
            ("pipeline_parallel_degree", parallelism.pipeline_parallel_degree),
        )
        if degree != 1
    ]
    if parallelism.fsdp_symm_mem_scope is not None:
        unsupported.append(
            f"parallelism.fsdp_symm_mem_scope={parallelism.fsdp_symm_mem_scope!r}"
        )
    if parallelism.fsdp_enable_cpu_offloading:
        unsupported.append("parallelism.fsdp_enable_cpu_offloading=True")
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
            "parallelism.fsdp_backend=FlexShardConfig() does not support "
            f"{', '.join(unsupported)} yet."
        )
