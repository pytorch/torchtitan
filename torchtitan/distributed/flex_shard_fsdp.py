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
is bitwise identical. With expert parallelism, ``fully_shard`` gives an MoE
block one group per mesh, and the block gets one bucket per group. Three
differences change only memory and communication. FSDP2 skips the embedding's
backward all-gather (``set_unshard_in_backward``), which FlexShard can't yet; it
gathers the embedding again in backward, with the same values. FlexShard has no
equivalent of ``set_is_last_backward``: during gradient accumulation, every
backward waits for its gradient reductions and releases its prefetch. And an MoE
block's expert bucket hooks the routed experts, not the block, so FlexShard
reshards them as soon as the routed experts finish forward or backward, while
FSDP2 keeps the block's expert group unsharded until the whole block finishes.
FlexShard is imported only when ``parallelism.fsdp_backend`` is a
``FlexShardConfig``.
"""

import logging
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeGuard

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard

if TYPE_CHECKING:
    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.optim import Optim
    from torchtitan.config.configs import DebugConfig, TrainingConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.models.common.decoder import Decoder

__all__ = [
    "FlexShardConfig",
    "apply_flex_shard_to_decoder",
    "as_fsdp2_dtensor",
    "flex_shard_partial_backwards",
    "is_flex_shard_module",
    "validate_flex_shard_training",
]

logger = logging.getLogger(__name__)

# For as_fsdp2_dtensor: the sparse storage mesh (edp_shard, ep) on which FSDP2
# lays out routed experts, keyed by the edp_shard mesh of FlexShard's expert
# buckets. A FlexShard parameter records only its bucket's mesh, and to_empty()
# replaces the parameters before initialization, so this can't be recorded on
# them. It goes away with the TODO in as_fsdp2_dtensor.
_EXPERT_STORAGE_MESHES: dict[DeviceMesh, DeviceMesh] = {}


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
    ep_degree: int,
    edp_mesh: DeviceMesh | None,
) -> None:
    """Shard a decoder with FlexShard as ``apply_fsdp_to_decoder`` does with FSDP2.

    Each ``fully_shard`` group becomes a bucket naming the same modules: the
    embedding, the norm with the output projection, each transformer block, and
    the parameters left to the root group. Gradients are summed over ``dp_mesh``
    without division, as ``disable_fsdp_gradient_division`` makes FSDP2 do.

    With expert parallelism, ``fully_shard`` splits an MoE block into one group
    per mesh: the routed experts on ``edp_mesh``'s ``edp_shard`` axis, and the
    rest of the block on ``dp_mesh``. The block then gets a bucket for each, and
    the buckets prefetch as ``apply_fsdp_to_decoder`` sets FSDP2 to.

    FlexShard stores parameters as plain local shards, which DCP would save as
    whole tensors. The model's ``state_dict()`` declares where each local shard
    sits in the full parameter (DCP's ``CheckpointableTensor``), as FSDP2's
    DTensors do, so FSDP2 and FlexShard save the same checkpoints.

    Args:
        model: The decoder to shard.
        dp_mesh: The 1D ``dp_shard`` mesh to shard over.
        param_dtype: The dtype of the unsharded parameters.
        reduce_dtype: The dtype of gradient reduction.
        reshard_after_forward_policy: "default", "always" or "never", as for
            ``apply_fsdp_to_decoder``.
        ep_degree: The expert-parallel degree.
        edp_mesh: With ``ep_degree > 1``, the sparse storage mesh, with the
            ``edp_shard`` and ``ep`` axes.
    """
    from torchtitan.distributed.fsdp import (
        decoder_ep_prefetch_order,
        decoder_fsdp_groups,
    )
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
    from flex_shard import (
        BucketSpec,
        flex_shard,
        GlobalLayout,
        MixedPrecisionPolicy,
        set_global_layout,
        set_state_dict_global_layouts,
    )

    # pyrefly: ignore [missing-import]
    from flex_shard.custom_placements.shard import Shard as FlexShardShard

    # validate_flex_shard_training allows only dp_shard and ep.
    assert dp_mesh.mesh_dim_names == ("dp_shard",), dp_mesh.mesh_dim_names
    expert_mesh = ep_mesh = None
    if ep_degree > 1:
        assert edp_mesh is not None
        assert edp_mesh.mesh_dim_names == ("edp_shard", "ep"), edp_mesh.mesh_dim_names
        expert_mesh, ep_mesh = edp_mesh["edp_shard"], edp_mesh["ep"]
        _EXPERT_STORAGE_MESHES[expert_mesh] = edp_mesh
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    embedding, head, blocks = decoder_fsdp_groups(
        model,
        dp_mesh,
        reshard_after_forward_policy=reshard_after_forward_policy,
        # validate_flex_shard_training rejects pipeline parallelism.
        pp_enabled=False,
        ep_degree=ep_degree,
        edp_mesh=edp_mesh,
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

    def bucket(
        patterns: list[str], reshard_after_forward: bool, mesh: DeviceMesh = dp_mesh
    ) -> BucketSpec:
        return BucketSpec(
            patterns,
            placement_fn=placement_fn,
            mesh=mesh,
            mp_policy=mp_policy,
            gradient_divide_factor=1.0,
            reshard_after_forward=reshard_after_forward,
            fsdp2_compatible=True,
        )

    module_fqns = {module: fqn for fqn, module in model.named_modules()}
    param_fqns = {param: fqn for fqn, param in model.named_parameters()}
    group_buckets: list[list[BucketSpec]] = []
    for group in groups:
        if expert_mesh is None or group.routed_experts is None:
            patterns = [module_fqns[module] for module in group.modules]
            group_buckets.append([bucket(patterns, group.reshard_after_forward)])
            continue
        # The expert bucket hooks the routed experts. The other bucket names
        # the rest of the block's parameters, so it hooks their deepest common
        # module, the block.
        expert_params = set(group.routed_experts.parameters())
        dense_fqns = [
            param_fqns[param]
            for module in group.modules
            for param in module.parameters()
            if param not in expert_params
        ]
        group_buckets.append(
            [
                bucket(dense_fqns, group.reshard_after_forward),
                bucket(
                    [module_fqns[group.routed_experts]],
                    group.reshard_after_forward,
                    expert_mesh,
                ),
            ]
        )
    buckets = [spec for specs in group_buckets for spec in specs]
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

    if ep_mesh is not None:
        # EP leaves each ep rank an even split of the routed experts' dim 0.
        # Declared before flex_shard(), this layout composes with the expert
        # bucket's split over edp_shard, so checkpoints place each local shard
        # in the full experts, as FSDP2's DTensors do.
        # TODO(<flex_shard issue link>): Drop once FlexShard derives the layout
        # itself.
        ep_rank, ep = ep_mesh.get_local_rank(), ep_mesh.size()
        for group in groups:
            if group.routed_experts is None:
                continue
            for param in group.routed_experts.parameters():
                num_local_experts, *rest = param.shape
                zeros = (0,) * len(rest)
                set_global_layout(
                    param,
                    GlobalLayout(
                        global_shape=(num_local_experts * ep, *rest),
                        global_offsets=((ep_rank * num_local_experts, *zeros),),
                        local_offsets=((0, *zeros),),
                        local_sizes=(tuple(param.shape),),
                    ),
                )

    flex_shard_module = flex_shard(model, buckets=buckets)
    if expert_mesh is not None:
        # Every bucket holds parameters, so there is one storage per bucket.
        assert len(flex_shard_module.sharded_bucket_storages) == len(buckets)
        storages = iter(flex_shard_module.sharded_bucket_storages)
        group_storages = {}
        for group, specs in zip(groups, group_buckets):
            storages_of_group = [next(storages) for _ in specs]
            group_storages.update(dict.fromkeys(group.modules, storages_of_group))
            # FSDP2 unshards a block's expert group with the block, so the
            # expert bucket prefetches nothing itself.
            for storage in storages_of_group[1:]:
                storage.set_buckets_to_forward_prefetch([])
                storage.set_buckets_to_backward_prefetch([])
        # FlexShard's default prefetches one bucket ahead, so the next block
        # would start gathering only at this block's routed experts, after EP's
        # device-to-host sync. FSDP2 gathers the whole next block from this
        # block's start.
        order = [group_storages[module] for module in decoder_ep_prefetch_order(model)]
        for current, following in zip(order, order[1:]):
            current[0].set_buckets_to_forward_prefetch(following)
            # In backward, FSDP2 prefetches a module's groups in reverse.
            following[0].set_buckets_to_backward_prefetch(current[::-1])

    def declare_global_layouts(module, state_dict, prefix, local_metadata) -> None:
        # TODO(<flex_shard issue link>): Drop once FlexShard's state_dict()
        # declares them.
        set_state_dict_global_layouts(
            module,
            {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            },
        )

    model.register_state_dict_post_hook(declare_global_layouts)
    logger.info("Applied FlexShard to the model")


def as_fsdp2_dtensor(tensor: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    """View ``tensor``, laid out as ``param``'s local shard, as the DTensor
    FSDP2 would give it if ``param`` is a FlexShard parameter.

    FlexShard stores a parameter, and its grad, as the plain local shard. The
    returned DTensor views that shard on its bucket's mesh with its ``Shard``
    placement, or for routed experts under expert parallelism, on the sparse
    storage mesh as FSDP2 lays them out. Code written for FSDP2's DTensor
    parameters and grads (e.g. parallelism-agnostic initialization, grad-norm
    clipping) then behaves the same. Other tensors are returned unchanged.
    """
    # TODO(<flex_shard issue link>): Use FlexShard's own DTensor view of a
    # parameter once it has one.
    # FlexShard parameters exist only once flex_shard is imported, which
    # happens only when it is selected.
    flex_shard = sys.modules.get("flex_shard")
    if flex_shard is None or not flex_shard.is_flex_shard_param(param):
        return tensor
    global_shape = flex_shard.get_global_shape(param)
    mesh = flex_shard.get_mesh(param)
    placements = tuple(Shard(p.dim) for p in flex_shard.get_placements(param))
    if (storage_mesh := _EXPERT_STORAGE_MESHES.get(mesh)) is not None:
        # EP shards the experts' dim 0, so FlexShard's global shape holds one
        # ep rank's experts. As in FSDP2, an edp_shard placement on dim 0
        # shards within each ep rank's experts.
        ep = storage_mesh["ep"].size()
        (placement,) = placements
        placements = (
            _StridedShard(0, split_factor=ep) if placement.dim == 0 else placement,
            Shard(0),
        )
        global_shape = torch.Size((global_shape[0] * ep, *global_shape[1:]))
        mesh = storage_mesh
    return DTensor.from_local(
        tensor,
        mesh,
        placements,
        run_check=False,
        shape=global_shape,
        stride=torch.empty(global_shape, device="meta").stride(),
    )


def is_flex_shard_module(module: nn.Module | None) -> TypeGuard[nn.Module]:
    """Whether ``module`` is a model that ``apply_flex_shard_to_decoder``
    sharded, which holds all its buckets."""
    # FlexShard modules exist only once flex_shard is imported, which happens
    # only when parallelism.fsdp_backend selects it.
    if "flex_shard" not in sys.modules:
        return False
    # TODO(<flex_shard issue link>): Import FlexShardModule from flex_shard once
    # it exports it.
    # pyrefly: ignore [missing-import]
    from flex_shard.flex_shard.flex_shard import FlexShardModule

    return isinstance(module, FlexShardModule)


@contextmanager
def flex_shard_partial_backwards(
    module: nn.Module,
    *,
    flex_shard_module: nn.Module,
    run_backward: bool,
    outer_backward_follows: bool,
) -> Iterator[Callable[[], None]]:
    """``fsdp_partial_backwards`` for ``module`` of ``flex_shard_module``, with
    the collectives FSDP2 issues. Yields ``before_last_forward``.

    A call of ``module`` on its own leaves its bucket's forward open, so
    FlexShard runs no post-forward for it and finishes the bucket as soon as
    the call's input grads are ready: with gradient sync, it reduce-scatters
    and reshards; without, it keeps the grads and reshards only with
    ``reshard_after_backward``. That backward finishes no other bucket, so
    unlike FSDP2, the other buckets and the root need no changes. Pipeline
    schedules' manual finalization isn't handled, as FlexShard rejects pipeline
    parallelism.

    Without backward, the bucket's forward stays open, as FSDP2 leaves its
    group's; ``post_validate_fsdp_for_accelerator_graph`` closes both.
    """
    if run_backward and not outer_backward_follows:
        # Calls whose inputs don't require grad have no trigger that finishes
        # the bucket. FSDP2 uses complete_fsdp_grouped_forward instead.
        raise ValueError(
            "parallelism.fsdp_backend=FlexShardConfig() does not support "
            "ChunkedLossWrapper with a frozen decoder yet."
        )
    # pyrefly: ignore [not-callable]
    storage = flex_shard_module.bucket_storage_of(next(module.parameters()))
    # TODO(<flex_shard issue link>): Read the bucket's gradient sync setting, and
    # whether it keeps grads from backwards without sync (FSDP2's
    # _post_backward_pending), through public FlexShard APIs once it has them.
    requires_gradient_sync = storage._requires_gradient_sync
    has_kept_grads = storage._runtime()[1].needs_sync
    reshard_after_backward = storage.reshard_after_backward

    def before_last_forward() -> None:
        # Kept grads wait for the outer backward, which reduce-scatters the
        # whole bucket at once, as FSDP2 does.
        if not has_kept_grads:
            storage.set_requires_gradient_sync(requires_gradient_sync)

    try:
        storage.set_requires_gradient_sync(False)
        storage.set_reshard_after_backward(False)
        yield before_last_forward
    finally:
        storage.set_requires_gradient_sync(requires_gradient_sync)
        storage.set_reshard_after_backward(reshard_after_backward)
        if not run_backward or reshard_after_backward:
            storage.reshard()


def validate_flex_shard_training(
    *,
    parallelism: "ParallelismConfig",
    training: "TrainingConfig",
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
    from torchtitan.components.checkpointer import CheckpointManager
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
    if checkpointer is not None:
        if not isinstance(checkpointer, CheckpointManager.Config):
            # TODO(<torch_checkpointing issue link>): torch_checkpointing's
            # resharder ignores the shard layouts FlexShard's state dicts declare.
            unsupported.append("checkpointers other than CheckpointManager")
        elif checkpointer.async_mode == "async_with_pinned_mem":
            # TODO(<pytorch issue link>): The save process receives the state
            # dict through torch.multiprocessing, which drops the shard layouts.
            unsupported.append("checkpointer.async_mode='async_with_pinned_mem'")
        # The state dict adapters need DTensor parameters.
        if checkpointer.initial_load_in_hf:
            unsupported.append("checkpointer.initial_load_in_hf=True")
        if checkpointer.last_save_in_hf:
            unsupported.append("checkpointer.last_save_in_hf=True")
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
