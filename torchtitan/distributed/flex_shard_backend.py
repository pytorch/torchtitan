# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FlexShard (meta-pytorch/flex_shard) as a ``ShardingBackend``, in place of
FSDP2's ``fully_shard``.

With ``parallelism.fsdp_backend=FlexShardBackend.Config()``, ``apply_to_decoder``
gives FlexShard one ``fsdp2_compatible`` bucket for each unit FSDP2 would pass to
``fully_shard``, with the same placements, mixed precision and reshard policy.
The buckets' collectives then match FSDP2's byte for byte, so training from the
same parameters is bitwise identical.

With expert parallelism, ``fully_shard`` gives an MoE block one group per mesh,
and the block gets one bucket per group: its routed experts on ``edp_shard``,
the rest on ``dp_shard``. EP has already split each routed-expert parameter over
``ep``, and torchtitan declares that split as the parameter's outer layout.
Likewise under tensor parallelism, each parameter is its tp rank's shard, and
declares TP's split as its outer layout, or tp as the group its grad is partial
over. Under context parallelism, the buckets shard over ``dp_shard`` and ``cp``
flattened into one axis, as ``fully_shard`` does.

FlexShard parameters and grads are plain local shards, with no DTensor.
``apply_to_decoder`` wraps each FlexShard parameter's ``param_init`` function
with flex_shard's ``init_from_full``: it fills the full parameter and the shard
keeps its part, as one device would initialize it, while each of FSDP2's DTensor
shards draws its own random values. ``flex_shard_grad_norm`` reduces a bucket's
grad norm as DTensor does. Two more differences change only memory and
communication. FSDP2 skips the embedding's backward all-gather
(``set_unshard_in_backward``), which FlexShard can't yet, so it gathers the
embedding again in backward, with the same values. And an MoE block's expert
bucket hooks the routed experts, not the block, so FlexShard reshards them as
soon as the routed experts finish forward or backward, while FSDP2 keeps the
block's expert group unsharded until the whole block finishes.
``flex_shard_partial_backwards`` keeps ``lm_head``'s bucket unsharded across
``ChunkedLossWrapper``'s chunks, as ``fsdp_partial_backwards`` keeps FSDP2's
group. FlexShard is imported only when it is selected.

FlexShard's state dicts hold plain local shards too, so each declares where its
shards sit in the full tensors (DCP's ``CheckpointableTensor``): the model's,
through ``declare_state_dict_layouts``, and the optimizer's, through
``declare_optimizer_state_layouts``. DCP then saves the same checkpoints as for
FSDP2's DTensors, and either backend loads the other's.
``load_hf_into_local_shards`` loads a Hugging Face checkpoint into those local
shards: it loads full tensors, a group of parameters at a time, and each shard
keeps its regions.
"""

import logging
import math
import sys
from collections import defaultdict
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from importlib.util import find_spec
from typing import Any, cast, TYPE_CHECKING

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from torchtitan.distributed.sharding_backend import ShardingBackend

if TYPE_CHECKING:
    from torch.distributed.fsdp import DataParallelMeshDims

    from torchtitan.components.checkpointer import CheckpointManager
    from torchtitan.components.optim import Optim
    from torchtitan.config import DebugConfig, PrecisionConfig
    from torchtitan.config.parallelism import ParallelismConfig
    from torchtitan.distributed.fsdp import ShardingUnit
    from torchtitan.distributed.fsdp_lifecycle import _PartialBackwardHooks
    from torchtitan.distributed.parallelism_context import ParallelismContext
    from torchtitan.distributed.sharding_backend import PartialBackwards
    from torchtitan.models.base.model import BaseModel
    from torchtitan.models.base.state_dict_adapter import BaseStateDictAdapter
    from torchtitan.models.common.decoder import Decoder

logger = logging.getLogger(__name__)

__all__ = [
    "FlexShardBackend",
    "declare_optimizer_state_layouts",
    "declare_state_dict_layouts",
    "flex_shard_bucket",
    "flex_shard_bucket_mesh",
    "flex_shard_grad_norm",
    "flex_shard_partial_backwards",
    "flex_shard_reduce_meshes",
    "holds_flex_shard_params",
    "load_hf_into_local_shards",
]


def flex_shard_bucket_mesh(
    dp_mesh: DeviceMesh, dp_mesh_dims: "DataParallelMeshDims | None"
) -> DeviceMesh:
    """The mesh of the dense FlexShard buckets, from the dense storage mesh and
    DP dims that ``resolve_fsdp_mesh`` returns."""
    if dp_mesh_dims is None:
        # One rank, where fully_shard takes the mesh as it is.
        return dp_mesh
    # validate_model rejects HSDP.
    assert not dp_mesh_dims.replicate_names, dp_mesh_dims
    shard_names = dp_mesh_dims.shard_names
    # As fully_shard does. DeviceMesh caches a flattened mesh by name, so both
    # backends reduce on one process group.
    if len(shard_names) == 1:
        return dp_mesh[shard_names[0]]
    return dp_mesh[shard_names]._flatten("_".join(shard_names))


def flex_shard_reduce_meshes(
    parallelism_context: "ParallelismContext",
) -> dict[DeviceMesh, DeviceMesh]:
    """For each mesh of FlexShard buckets whose parameters another parallelism
    split before FlexShard sharded them, the mesh to reduce their grad norm over
    (``flex_shard_grad_norm``): the mesh of FSDP2's DTensor grads for them, with
    the buckets' ranks as its first dim, or that mesh flattened if their DTensor
    norm reduces over it in one all-reduce. The grad norms of other buckets
    reduce over the buckets' own mesh.

    FlexShard knows only each bucket's mesh, so the meshes come from
    ``parallelism_context``, resolved as ``FlexShardBackend.apply_to_decoder``
    resolves the meshes it shards over. Tensor parallelism splits the dense
    parameters over ``tp``, and expert parallelism splits the routed experts
    over ``ep``, whose buckets are on ``edp_shard``.
    """
    from torchtitan.distributed.fsdp import resolve_fsdp_mesh, resolve_sparse_fsdp_mesh

    meshes = {}
    dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
    tp_mesh = dp_mesh["tp"] if "tp" in (dp_mesh.mesh_dim_names or ()) else None
    if tp_mesh is not None:
        # FSDP2 lays the dense grads out as DTensors on (bucket mesh, tp). No
        # flattened mesh spans tp, so DTensor reduces their norm over the
        # bucket mesh, then tp.
        bucket_mesh = flex_shard_bucket_mesh(dp_mesh, dp_mesh_dims)
        # TODO: Use a public DeviceMesh API once there is one. _concatenate
        # reuses the meshes' process groups instead of creating new ones.
        meshes[bucket_mesh] = DeviceMesh._concatenate([bucket_mesh, tp_mesh])
    edp_mesh, _ = resolve_sparse_fsdp_mesh(parallelism_context)
    if edp_mesh is not None:
        # FSDP2's routed-expert grads are DTensors on edp_mesh. Without TP,
        # torchtitan's loss mesh spans the same ranks, and DTensor finds it by
        # layout and reduces their norm over it in one all-reduce. _flatten
        # returns that mesh, and raises if its layout differs. With TP, no
        # flattened mesh does, so it reduces over edp_shard, then ep. Flattening
        # one here would make DTensor reduce differently with FSDP2 in the same
        # process.
        meshes[edp_mesh["edp_shard"]] = (
            edp_mesh if tp_mesh is not None else edp_mesh._flatten("loss_mesh")
        )
    return meshes


def flex_shard_grad_norm(
    params: list[torch.Tensor],
    *,
    mesh: DeviceMesh,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
) -> torch.Tensor:
    """The norm of the grads of FlexShard parameters ``params``, reduced over
    ``mesh``: their buckets' 1D mesh (``flex_shard.get_mesh``), or the mesh
    ``flex_shard_reduce_meshes`` gives for it.

    The grads are plain local shards. Like FSDP2's DTensor grads, each is
    sharded over ``mesh``'s first dim, and over the others if
    ``_is_outer_sharded``. This computes the norm as ``get_total_norm`` does for
    DTensor grads with that layout, so the result is bitwise equal. Each grad's
    local norm is partial over the dims that shard it. Stacking the norms makes
    them partial over every dim that shards any grad: a norm replicated on such
    a dim is divided by the dim's size ** (1 / p), so that its copies sum to it.
    The stacked norms are reduced over each dim in turn, and their norm is taken
    last. Taking it before the reduction isn't bitwise equal.
    """
    grads = [p.grad for p in params if p.grad is not None]
    replicated = [not _is_outer_sharded(p) for p in params if p.grad is not None]
    # From flex_shard's examples/deepseek_v3/train.py.
    if foreach is False:
        norm_list = [torch.linalg.vector_norm(g, norm_type) for g in grads]
    else:
        norm_list = list(torch._foreach_norm(grads, norm_type))
    # The dims that shard some grad, over which the stacked norms are partial.
    # TODO: Use flex_shard's grad norm for its layouts once it has one. This
    # copies DTensor's scaling of replicated norms (_NormPartial), which has to
    # stay in step with DTensor's for the norm to stay bitwise equal.
    dims = range(1 if all(replicated) else mesh.ndim)
    for dim in dims[1:]:
        # Each norm on its own, so that no mask has to be copied to the device.
        scale = math.pow(mesh.size(dim), 1.0 / norm_type)
        norm_list = [
            norm / scale if is_replicated else norm
            for norm, is_replicated in zip(norm_list, replicated, strict=True)
        ]
    norms = torch.stack(norm_list)
    for dim in dims:
        # DTensor skips dims of size 1.
        if mesh.size(dim) == 1:
            continue
        if math.isinf(norm_type):
            dist.all_reduce(norms, op=dist.ReduceOp.MAX, group=mesh.get_group(dim))
        else:
            norms **= norm_type
            dist.all_reduce(norms, group=mesh.get_group(dim))
            norms **= 1.0 / norm_type
    total_norm = torch.linalg.vector_norm(norms, norm_type)
    if error_if_nonfinite and not torch.isfinite(total_norm):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from `parameters` "
            "is non-finite, so it cannot be clipped. To disable this error and "
            "scale the gradients by the non-finite norm anyway, set "
            "`error_if_nonfinite=False`"
        )
    return total_norm


def _is_outer_sharded(param: torch.Tensor) -> bool:
    """Whether FlexShard parameter ``param`` is a shard of an outer sharding,
    tensor or expert parallelism, which declared its layout before
    ``flex_shard()``. Else its grad is the same on the ranks of each dim of the
    mesh ``flex_shard_grad_norm`` reduces it over but the first, e.g. a norm
    weight's under TP."""
    # pyrefly: ignore [missing-import]
    from flex_shard import get_outer_layout

    return get_outer_layout(param) is not None


def holds_flex_shard_params(module: nn.Module) -> bool:
    """Whether FlexShard sharded ``module``'s parameters. Outside forward and
    backward only: the gathered parameters of a bucket aren't FlexShard's."""
    # flex_shard is imported only when it is selected, so without it no
    # parameter is FlexShard's.
    flex_shard = sys.modules.get("flex_shard")
    return flex_shard is not None and any(
        flex_shard.is_flex_shard_param(param) for param in module.parameters()
    )


def flex_shard_bucket(model: nn.Module, module_fqn: str) -> Any:
    """The FlexShard bucket that holds the parameters of ``model``'s submodule
    ``module_fqn``, e.g. ``lm_head``'s, which also holds the final norm's, for
    ``flex_shard_partial_backwards``. FlexShard must have sharded ``model``,
    and the buckets name their parameters by their names in ``model``. A
    shared parameter, e.g. ``lm_head``'s tied weight, goes by its first name,
    ``tok_embeddings.weight``, with the others among its shared names."""
    # TODO: Use flex_shard's lookup of a module's bucket once it has one, so
    # fsdp_partial_backwards can find it from the module as for FSDP2.
    prefix = f"{module_fqn}."
    buckets = [
        bucket
        for bucket in cast(Any, model).sharded_bucket_storages
        if any(
            name.startswith(prefix)
            for fqn, info in bucket.param_infos.items()
            for name in (fqn, *info.shared_fqns)
        )
    ]
    if len(buckets) != 1:
        raise ValueError(
            f"Expected one FlexShard bucket to hold {module_fqn}'s parameters, "
            f"but {len(buckets)} do."
        )
    return buckets[0]


@contextmanager
def flex_shard_partial_backwards(
    bucket: Any, *, run_backward: bool, outer_backward_follows: bool
) -> Iterator["_PartialBackwardHooks"]:
    """``fsdp_partial_backwards`` for a module whose FSDP2 group is the
    FlexShard ``bucket`` (``flex_shard_bucket``), with the collectives FSDP2
    issues for its group.

    A call of the module on its own leaves its bucket's forward open, so
    FlexShard runs no post-forward for it and finishes the bucket as soon as
    the call's input grads are ready: with gradient sync, it reduce-scatters
    and reshards; without, it keeps the grads, and reshards only with
    ``reshard_after_backward``. That backward finishes no other bucket, so
    unlike FSDP2, the other buckets need no changes. Gradient sync is on
    outside, since ``validate_model`` rejects deferred gradient
    reduction, so the last call's backward reduce-scatters the grads of all
    calls, as FSDP2's does.

    Without backward, the bucket's forward stays open, as FSDP2 leaves its
    group's; ``reshard_after_eval`` closes both.
    """
    from torchtitan.distributed.fsdp_lifecycle import _PartialBackwardHooks

    if run_backward and not outer_backward_follows:
        # Calls whose inputs don't require grad have no trigger that finishes
        # the bucket. FSDP2 completes each call's grouped forward instead.
        raise ValueError(
            "parallelism.fsdp_backend=FlexShardBackend.Config() does not support "
            "ChunkedLossWrapper with a frozen decoder yet."
        )
    reshard_after_backward = bucket.reshard_after_backward
    # TODO: Gradient sync is set to True, not restored, because FlexShard has
    # no public getter for it, and validate_model rejects
    # deferred gradient reduction, so it's always on outside. Allowing deferral
    # needs the caller's setting restored, and FlexShard's state of grads kept
    # without sync, to leave the last reduce-scatter to the decoder backward
    # when earlier microbatches kept grads, as FSDP2 does.
    try:
        bucket.set_requires_gradient_sync(False)
        bucket.set_reshard_after_backward(False)
        # The bucket's forward stays open, so outputs need no hooks.
        yield _PartialBackwardHooks(
            lambda: bucket.set_requires_gradient_sync(True), lambda output: output
        )
    finally:
        bucket.set_requires_gradient_sync(True)
        bucket.set_reshard_after_backward(reshard_after_backward)
        # The last call's backward has resharded the bucket, unless there was
        # no backward.
        bucket.reshard()


def declare_state_dict_layouts(model: nn.Module) -> None:
    """Make ``model.state_dict()`` declare where each FlexShard parameter's
    local shard sits in the full parameter. Call it after ``flex_shard(model)``.

    Without the layouts, DCP saves and loads each local shard as the whole
    parameter. Every state dict ``CheckpointManager`` saves or loads comes from
    ``model.state_dict()``, including the one it takes again after a load.
    """
    # pyrefly: ignore [missing-import]
    from flex_shard import set_state_dict_global_layouts

    # TODO: Drop once a FlexShard module's state_dict() declares its layouts.
    def hook(module, state_dict, prefix, local_metadata) -> None:
        set_state_dict_global_layouts(
            module,
            {
                key.removeprefix(prefix): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            },
        )

    model.register_state_dict_post_hook(hook)


def declare_optimizer_state_layouts(
    optimizer: torch.optim.Optimizer, model: nn.Module
) -> None:
    """Make ``optimizer.state_dict()`` declare where each state of a FlexShard
    parameter sits in the full state, if FlexShard sharded ``model``, which
    holds the parameters.

    ``step`` declares no layout, so DCP saves it as replicated, as for FSDP2.
    flex_shard raises ``NotImplementedError`` for optimizers other than Adam
    and AdamW. Call it outside forward and backward, as for
    ``holds_flex_shard_params``.
    """
    if holds_flex_shard_params(model):
        # pyrefly: ignore [missing-import]
        from flex_shard import register_optimizer_checkpoint_hook

        register_optimizer_checkpoint_hook(optimizer, model)


# The full tensors of one group of parameters in a Hugging Face load take at most
# this many bytes, unless the group is one parameter that takes more.
_HF_LOAD_GROUP_BYTES = 1 << 30


def load_hf_into_local_shards(
    sd_adapter: "BaseStateDictAdapter",
    state_dict: dict[str, Any],
    storage_reader: dcp.StorageReader,
) -> None:
    """Load a Hugging Face checkpoint into ``state_dict`` in place, where
    ``state_dict`` is a model state dict whose local shards declare their
    layouts, as FlexShard's does (``declare_state_dict_layouts``).

    The state dict adapters convert full tensors (and FSDP2's DTensors), so
    they would take each local shard for the whole tensor. Instead, the
    parameters load as full tensors, a group at a time, through the adapters'
    conversions of full tensors, and each shard keeps its regions of its full
    tensor. Each conversion sees every key, as in a full load: the group's full
    tensors, and meta tensors of the same shapes, which hold no data, for the
    other keys. One rank reads each group, in turn, and broadcasts its full
    tensors to the other ranks, so the checkpoint is read once. A shard that
    ``from_hf`` gives back from another key's Hugging Face tensors, such as
    ``lm_head.weight``'s with tied weights, loads with that key's group. Entries
    without a layout load last, as for FSDP2.
    """
    # pyrefly: ignore [missing-import]
    from flex_shard import get_global_layout

    shards, others = {}, {}
    for key, value in state_dict.items():
        layout = get_global_layout(value)
        if layout is None:
            others[key] = value
        else:
            shards[key] = (value, layout)
    meta = {
        key: torch.empty(layout.global_shape, dtype=value.dtype, device="meta")
        for key, (value, layout) in shards.items()
    } | {key: value.to("meta") for key, value in others.items()}
    # Every rank holds the same keys, and the groups depend only on the keys and
    # full shapes, so every rank loads as often.
    groups: list[list[str]] = []
    group_bytes = 0
    for key, (value, layout) in shards.items():
        nbytes = math.prod(layout.global_shape) * value.element_size()
        if not groups or group_bytes + nbytes > _HF_LOAD_GROUP_BYTES:
            groups.append([])
            group_bytes = 0
        groups[-1].append(key)
        group_bytes += nbytes
    # TODO: Every rank receives each group's full tensors. Sending each rank
    # only its regions would cut that traffic to the size of its shards.
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    device = next(iter(shards.values()))[0].device
    filled = set()
    for i, group in enumerate(groups):
        reader = i % world_size
        full = _read_and_broadcast(
            sd_adapter,
            meta
            | (
                {key: shards[key][0].new_empty(meta[key].shape) for key in group}
                if rank == reader
                else {}
            ),
            shards.keys(),
            storage_reader,
            reader,
            device,
        )
        with torch.no_grad():
            for key, value in full.items():
                local, layout = shards[key]
                for global_offset, local_offset, size in zip(
                    layout.global_offsets,
                    layout.local_offsets,
                    layout.local_sizes,
                    strict=True,
                ):
                    local[_box(local_offset, size)].copy_(
                        value[_box(global_offset, size)]
                    )
                filled.add(key)
        del full
    if missing := sorted(shards.keys() - filled):
        raise ValueError(
            f"{type(sd_adapter).__name__} gives no Hugging Face tensors for "
            f"{missing}, so FlexShard can't load them."
        )
    if others:
        state_dict.update(
            _read_and_broadcast(
                sd_adapter,
                meta | others,
                others.keys(),
                storage_reader,
                0,
                next(iter(others.values())).device,
            )
        )


def _read_and_broadcast(
    sd_adapter: "BaseStateDictAdapter",
    state_dict: dict[str, Any],
    keys: Collection[str],
    storage_reader: dcp.StorageReader,
    reader: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """The tensors of ``keys`` other than meta ones that ``from_hf`` gives back
    from ``to_hf(state_dict)``, whose tensors other than meta ones ``dcp.load``
    fills from the checkpoint. Only rank ``reader`` converts and reads, then
    broadcasts the tensors to the other ranks, which allocate them on
    ``device``. A tensor that ``from_hf`` gives back for several keys, such as
    a tied weight, is sent once. If the reader fails, every rank raises."""
    distributed = dist.is_initialized() and dist.get_world_size() > 1
    # aliases maps a key to an earlier key whose tensor from_hf gave it.
    full, aliases, error = {}, {}, None
    if not distributed or dist.get_rank() == reader:
        try:
            hf_state_dict = sd_adapter.to_hf(state_dict)
            if loaded := {k: v for k, v in hf_state_dict.items() if not v.is_meta}:
                dcp.load(loaded, storage_reader=storage_reader, no_dist=True)
            first: dict[int, str] = {}
            for key, value in sd_adapter.from_hf(hf_state_dict).items():
                if key not in keys or value.is_meta:
                    continue
                if id(value) in first:
                    aliases[key] = first[id(value)]
                else:
                    first[id(value)] = key
                    full[key] = value.to(device).contiguous()
        except Exception as e:
            if not distributed:
                raise
            error = e
    if distributed:
        specs = [
            None
            if error is not None
            else ({key: (v.shape, v.dtype) for key, v in full.items()}, aliases)
        ]
        dist.broadcast_object_list(specs, src=reader)
        if specs[0] is None:
            if error is not None:
                raise error
            raise RuntimeError(
                f"Rank {reader} failed to read its part of the Hugging Face "
                "checkpoint."
            )
        shapes, aliases = specs[0]
        for key, (shape, dtype) in shapes.items():
            if key not in full:
                full[key] = torch.empty(shape, dtype=dtype, device=device)
            dist.broadcast(full[key], src=reader)
    return full | {key: full[source] for key, source in aliases.items()}


def _box(offsets, sizes) -> tuple[slice, ...]:
    return tuple(
        slice(offset, offset + size)
        for offset, size in zip(offsets, sizes, strict=True)
    )


def _raise_if_unsupported(unsupported: list[str]) -> None:
    if unsupported:
        raise ValueError(
            "parallelism.fsdp_backend=FlexShardBackend.Config() does not support "
            f"{', '.join(unsupported)} yet."
        )


class FlexShardBackend(ShardingBackend):
    """FlexShard, bitwise identical to FSDP2's ``fully_shard`` for what it
    supports. ``validate_model`` and ``validate_trainer`` reject the rest. Each
    rejection is temporary: it goes once FlexShard supports the feature."""

    @dataclass(kw_only=True, slots=True)
    class Config(ShardingBackend.Config):
        pass

    def validate_model(
        self,
        model_config: "BaseModel.Config",
        *,
        parallelism: "ParallelismConfig",
        debug: "DebugConfig",
    ) -> None:
        from torchtitan.models.base.module import Module
        from torchtitan.models.common.decoder import Decoder
        from torchtitan.models.common.moe import MoE, RoutedExperts
        from torchtitan.models.common.moe.token_dispatcher import (
            AllToAllTokenDispatcher,
        )
        from torchtitan.quantization.utils import has_quantization

        # TODO: Remove each rejection once FlexShard supports the feature.
        unsupported = [
            f"parallelism.{name}={degree}"
            for name, degree in (
                (
                    "data_parallel_replicate_degree",
                    parallelism.data_parallel_replicate_degree,
                ),
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
        if parallelism.fsdp_unshard_in_current_stream:
            unsupported.append("parallelism.fsdp_unshard_in_current_stream=True")
        if parallelism.fsdp_defer_gradient_reduction:
            # FlexShard has no equivalent of FSDP2's set_is_last_backward.
            # TODO: When allowing it, change flex_shard_partial_backwards, which
            # sets lm_head's bucket's gradient sync to True instead of restoring it.
            unsupported.append("parallelism.fsdp_defer_gradient_reduction=True")
        # Decoder._apply_fsdp is the only path that applies fsdp_backend, so a
        # model that overrides it would shard with FSDP2.
        model_cls = model_config._owner
        if getattr(model_cls, "_apply_fsdp", None) is not Decoder._apply_fsdp:
            name = (
                type(model_config).__qualname__
                if model_cls is None
                else model_cls.__name__
            )
            unsupported.append(f"models that override Decoder._apply_fsdp ({name})")
        if getattr(model_config, "mtp_layers", None):
            # TODO(<flex_shard issue link>): lm_head runs once per objective, and
            # FlexShard sums the calls' grads in bf16, where FSDP2 sums them in fp32.
            # The other grads differ from FSDP2's too, from the first step.
            unsupported.append("MTP layers")
        # FlexShard's Shard placement all-gathers a bucket in one dtype.
        if any(
            module_config.preserve_dtype_params
            for _, module_config, _, _ in model_config.traverse(
                Module.Config, recurse=True
            )
        ):
            unsupported.append("preserve_dtype_params")
        # FlexShard parameters are plain local shards, and only param_init
        # initializes them as one device would: apply_to_decoder wraps it with
        # init_from_full. reset_parameters() would draw every rank's shard as the
        # whole parameter, with the same values.
        if any(
            module_config.param_init is None
            and callable(getattr(module_config._owner, "reset_parameters", None))
            for _, module_config, _, _ in model_config.traverse(
                Module.Config, recurse=True
            )
        ):
            unsupported.append(
                "modules initialized by reset_parameters() (set param_init)"
            )
        # FlexShard stores parameters as plain tensors, which would drop the
        # quantized all-gather that MXFP8 and NVFP4 weights implement for FSDP2.
        if has_quantization(model_config):
            unsupported.append("MXFP8 and NVFP4 weights")
        if parallelism.expert_parallel_degree > 1:
            # Only RoutedExperts with AllToAllTokenDispatcher has been checked
            # against FSDP2 under EP. FlexShard's expert bucket treats every
            # routed-expert parameter as split over ep, but output_postprocess
            # parameters are replicated over ep (models/common/moe/SHARDING.md),
            # so their grads would never be summed over it.
            for _, moe, _, _ in model_config.traverse(MoE.Config):
                routed_experts = moe.routed_experts
                if type(routed_experts) is not RoutedExperts.Config:
                    feature = type(routed_experts).__qualname__
                elif (
                    type(routed_experts.token_dispatcher)
                    is not AllToAllTokenDispatcher.Config
                ):
                    feature = type(routed_experts.token_dispatcher).__qualname__
                elif routed_experts.output_postprocess is not None:
                    feature = "routed-expert output_postprocess"
                else:
                    continue
                entry = f"{feature} with expert parallelism"
                if entry not in unsupported:
                    unsupported.append(entry)

        if debug.spmd_typechecking:
            unsupported.append("debug.spmd_typechecking")
        _raise_if_unsupported(unsupported)
        # Checked here because flex_shard is imported only once the model is
        # sharded, after distributed init and the model build.
        if find_spec("flex_shard") is None:
            raise ImportError(
                "parallelism.fsdp_backend=FlexShardBackend.Config() needs flex_shard "
                "(github.com/meta-pytorch/flex_shard), which isn't installed. "
                ".github/workflows/unit_test_gpu.yaml installs the commit torchtitan "
                "is tested with."
            )

    def validate_trainer(
        self,
        *,
        checkpointer: "CheckpointManager.Config | None",
        optim: "Optim.Config",
        trainer: type | None,
    ) -> None:
        from torchtitan.components.checkpointer import CheckpointManager
        from torchtitan.components.optim import DistMuon, GradientClipper
        from torchtitan.trainer import Trainer

        # TODO: Remove each rejection once FlexShard supports the feature. They
        # read the checkpointer, optimizers and trainer, which
        # validate_model_training_config (config/validation.py), where
        # validate_model runs, doesn't receive, so TrainingEngine.Config's
        # __post_init__ calls this method instead.
        unsupported = []
        if checkpointer is not None:
            # FlexShard's state dicts hold local shards that declare their layouts
            # to DCP. Each of these would lose the layouts and treat every rank's
            # shard as the whole tensor, without an error.
            if not isinstance(checkpointer, CheckpointManager.Config):
                # TODO: Allow once torch_checkpointing's resharder reads the
                # layouts, not only DTensor placements.
                unsupported.append("checkpointers other than CheckpointManager")
            elif checkpointer.async_mode == "async_with_pinned_mem":
                # TODO: Allow once DCP's process-based async save keeps the
                # layouts, which torch.multiprocessing drops on the way to the save
                # process.
                unsupported.append("checkpointer.async_mode='async_with_pinned_mem'")
            # load_hf_into_local_shards doesn't load quantized checkpoints.
            if checkpointer.initial_load_in_hf_quantized:
                unsupported.append("checkpointer.initial_load_in_hf_quantized=True")
            # The state dict adapters need DTensor parameters.
            if checkpointer.last_save_in_hf:
                unsupported.append("checkpointer.last_save_in_hf=True")
        if optim.ema is not None:
            unsupported.append("optim.ema")
        if any(isinstance(o, DistMuon.Config) for o in optim.optimizer.optimizers):
            # DistMuon needs DTensor parameters.
            unsupported.append("DistMuon")
        # Only GradientClipper is known to clip through the sharding backend.
        # Another clipper might clip FlexShard's plain local shards by each rank's
        # own norm, with no error.
        if type(optim.gradient_clipper) is not GradientClipper.Config:
            unsupported.append(
                "gradient clippers other than GradientClipper "
                f"({type(optim.gradient_clipper).__qualname__})"
            )
        # Other trainers shard, reduce or publish parameters in ways that FlexShard
        # doesn't support yet: GraphTrainer shards with SimpleFSDP, TorchFT's
        # cross-replica all-reduce hooks FSDP2's parameter groups, and the RL
        # trainer publishes state_dict(), which holds local shards, as full weights.
        if trainer is not Trainer:
            name = (
                "None"
                if trainer is None
                else f"{trainer.__module__}.{trainer.__qualname__}"
            )
            unsupported.append(
                f"trainers other than torchtitan.trainer.Trainer ({name})"
            )
        _raise_if_unsupported(unsupported)

    def apply_to_decoder(
        self,
        model: "Decoder",
        *,
        parallelism_context: "ParallelismContext",
        precision: "PrecisionConfig",
        parallelism: "ParallelismConfig",
    ) -> None:
        from torchtitan.config import TORCH_DTYPE_MAP
        from torchtitan.distributed.fsdp import (
            decoder_sharding_units,
            resolve_fsdp_mesh,
            resolve_sparse_fsdp_mesh,
        )

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallelism_context)
        edp_mesh, _ = resolve_sparse_fsdp_mesh(parallelism_context)
        embedding, norm_and_lm_head, blocks = decoder_sharding_units(
            model,
            dp_mesh=dp_mesh,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            pp_enabled=parallelism_context.pp_enabled,
            ep_degree=parallelism_context.ep,
            edp_mesh=edp_mesh,
        )
        _apply_flex_shard(
            model,
            dp_mesh=dp_mesh,
            units=[unit for unit in (embedding, norm_and_lm_head) if unit is not None]
            + blocks,
            param_dtype=TORCH_DTYPE_MAP[precision.param_dtype],
            reduce_dtype=TORCH_DTYPE_MAP[precision.reduce_dtype],
            dp_mesh_dims=dp_mesh_dims,
            edp_mesh=edp_mesh if parallelism_context.ep > 1 else None,
        )

    def clip_grad_norm_(
        self,
        parameters: list[torch.nn.Parameter],
        max_norm: float,
        *,
        foreach: bool | None,
        parallelism_context: "ParallelismContext",
    ) -> torch.Tensor:
        # validate_model has checked that flex_shard is installed.
        # pyrefly: ignore [missing-import]
        import flex_shard

        from torchtitan.distributed import utils as dist_utils

        # FlexShard grads are plain local shards over their bucket's mesh.
        params_by_mesh: dict[DeviceMesh, list[torch.Tensor]] = defaultdict(list)
        for p in parameters:
            if p.grad is not None:
                mesh = flex_shard.get_mesh(p)
                # The root bucket holds every parameter that no other bucket does.
                assert mesh is not None, tuple(p.shape)
                params_by_mesh[mesh].append(p)
        reduce_meshes = flex_shard_reduce_meshes(parallelism_context)
        norms = [
            flex_shard_grad_norm(
                params,
                mesh=reduce_meshes.get(mesh, mesh),
                norm_type=2.0,
                error_if_nonfinite=False,
                foreach=foreach,
            )
            for mesh, params in params_by_mesh.items()
        ]
        # A rank without grads, e.g. a PP rank, contributes a norm of 0.
        if not norms:
            norms = [torch.nn.utils.get_total_norm([], foreach=foreach)]
        return dist_utils.clip_grads_with_mesh_norms_(
            list(params_by_mesh.values()),
            norms,
            max_norm=max_norm,
            norm_type=2.0,
            foreach=foreach,
            pp_mesh=parallelism_context.get_optional_mesh("pp"),
        )

    def partial_backwards(
        self, model: nn.Module, module_fqn: str
    ) -> "PartialBackwards":
        return partial(
            flex_shard_partial_backwards, flex_shard_bucket(model, module_fqn)
        )

    def reshard_after_eval(self, model_parts: list[nn.Module]) -> None:
        # Closes the forward of norm and lm_head's bucket, which ChunkedLossWrapper
        # leaves open, as FSDP2's post_validate_fsdp_for_accelerator_graph does
        # for their group.
        for model in model_parts:
            cast(Any, model).reshard()


def _declare_tensor_parallelism(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    *,
    routed_expert_params: set[nn.Parameter],
) -> None:
    """Declare to FlexShard how tensor parallelism splits each parameter of
    ``model`` and its grad. Call before ``flex_shard()``.

    ``Module._parallelize`` leaves each parameter as its tp rank's local shard,
    with the SPMD type that FSDP2 lays it out by:

    - A parameter that TP splits declares where its shard sits in the full
      parameter, as its outer layout. flex_shard composes it with the bucket's
      split in the layouts the wrapped ``param_init`` functions use.
    - A parameter typed ``R`` on tp, e.g. a norm weight under sequence
      parallelism, has a grad that's partial over tp. FSDP2 types the grad
      ``P`` and all-reduces it over tp before its reduce-scatter. FlexShard does
      the same for a parameter that declares tp as its partial-grad group.
    - A parameter typed ``I`` on tp declares nothing: each tp rank computes
      its whole grad.
    - Any other type on tp, or none, raises. Untyped, a parameter would declare
      itself whole, and the grad norm would count it once per tp rank.

    Routed experts are skipped. MoE models need ``ep >= tp``, so the routed
    experts are typed on the sparse mesh, and keep their EP layout.
    """
    # pyrefly: ignore [missing-import]
    from flex_shard import set_partial_grad_group

    # TODO: Drop once flex_shard() reads the parameters' SPMD types itself, as
    # fully_shard(..., dp_mesh_dims=...) does. Until then torchtitan declares TP's
    # layouts, through a flex_shard adapter that flex_shard doesn't export.
    # pyrefly: ignore [missing-import]
    from flex_shard.layout_adapters.spmd_types import spmd_types_to_global_layout

    tp_group = tp_mesh.get_group()
    tp_axis = spmd.MeshAxis.of(tp_group)
    layouts = {}
    for fqn, param in model.named_parameters():
        if param in routed_expert_params:
            continue
        local_type = spmd.get_local_type(param)
        tp_type = local_type.get(tp_axis)
        if tp_type is spmd.R:
            set_partial_grad_group(param, tp_group)
        elif tp_type is spmd.V or isinstance(tp_type, spmd.Shard):
            # Split by TP: V with its dim in the PartitionSpec, or S(dim). The
            # adapter raises for V without a dim.
            layouts[fqn] = spmd.SpmdType(
                dict(local_type), spmd.get_partition_spec(param)
            )
        elif tp_type is not spmd.I:
            raise ValueError(
                "parallelism.fsdp_backend=FlexShardBackend.Config() supports only "
                "parameters typed R, I, V or S(dim) on tp under tensor "
                f"parallelism, but {fqn} is {tp_type}."
            )
    spmd_types_to_global_layout(model, layouts, tp_mesh)


def _apply_flex_shard(
    model: "Decoder",
    *,
    dp_mesh: DeviceMesh,
    units: list["ShardingUnit"],
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    dp_mesh_dims: "DataParallelMeshDims | None",
    edp_mesh: DeviceMesh | None,
) -> None:
    """Shard ``model`` with FlexShard (meta-pytorch/flex_shard), one
    ``fsdp2_compatible`` bucket per sharding unit.

    Each bucket names the unit's modules, with its placements and reshard
    setting, and a root bucket takes the parameters left over, as
    ``fully_shard(model)`` does. ``dp_mesh`` and ``dp_mesh_dims`` are the dense
    storage mesh and its data-parallel axes, as ``fully_shard`` gets them. The
    buckets shard over the mesh ``fully_shard`` shards over: ``dp_shard``, or
    under context parallelism, ``dp_shard`` and ``cp`` flattened into one axis.
    Gradients are summed over it without division, as
    ``disable_fsdp_gradient_division`` makes FSDP2 do. The buckets' collectives
    then match FSDP2's byte for byte. Each sharded parameter's ``param_init``
    function is wrapped with flex_shard's ``init_from_full``, so the shards start
    as one device's initialization. ``model.state_dict()`` declares each local
    shard's place in the full parameter, so DCP saves the same checkpoints as for
    FSDP2.

    Under tensor parallelism, each parameter is its tp rank's local shard, and
    declares before ``flex_shard()`` how TP splits it and its grad (see
    ``_declare_tensor_parallelism``).

    With expert parallelism, ``edp_mesh`` is the sparse mesh, with the
    ``edp_shard`` and ``ep`` axes. ``fully_shard`` splits an MoE block into one
    group per mesh, so the block gets two buckets: its routed experts on
    ``edp_shard``, and its other parameters on the dense buckets' mesh. The
    buckets prefetch as ``apply_fsdp_to_decoder`` sets FSDP2's groups to.
    """
    # flex_shard is an optional dependency, needed only for this backend.
    # pyrefly: ignore [missing-import]
    from flex_shard import (
        BucketSpec,
        custom_placements,
        flex_shard,
        get_flex_shard_global_layouts,
        GlobalLayout,
        init_from_full,
        MixedPrecisionPolicy,
        set_global_layout,
    )

    from torchtitan.distributed.fsdp import decoder_prefetch_order
    from torchtitan.models.base.module import Module
    from torchtitan.models.common.param_init import skip_param_init

    # flex_shard_reduce_meshes gives the mesh to reduce the grad norm over for
    # the parameters of buckets that another parallelism split first.
    bucket_mesh = flex_shard_bucket_mesh(dp_mesh, dp_mesh_dims)
    tp_mesh = dp_mesh["tp"] if "tp" in (dp_mesh.mesh_dim_names or ()) else None
    if tp_mesh is not None:
        _declare_tensor_parallelism(
            model,
            tp_mesh,
            routed_expert_params={
                param
                for unit in units
                if unit.routed_experts is not None
                for param in unit.routed_experts.parameters()
            },
        )
    expert_mesh = None
    if edp_mesh is not None:
        assert edp_mesh.mesh_dim_names == ("edp_shard", "ep"), edp_mesh.mesh_dim_names
        expert_mesh = edp_mesh["edp_shard"]
        # EP has split each routed-expert parameter's dim 0, the experts, over
        # ep. Declaring this rank's part makes the layouts that the wrapped
        # param_init functions use cover all experts, so each shard is its slice
        # of one device's init, which briefly holds the parameter with all experts.
        ep, ep_rank = edp_mesh["ep"].size(), edp_mesh["ep"].get_local_rank()
        for unit in units:
            if unit.routed_experts is None:
                continue
            for param in unit.routed_experts.parameters():
                num_local_experts, *rest = param.shape
                set_global_layout(
                    param,
                    GlobalLayout(
                        global_shape=(num_local_experts * ep, *rest),
                        global_offsets=(
                            (num_local_experts * ep_rank, *[0] * len(rest)),
                        ),
                        local_offsets=((0,) * param.ndim,),
                        local_sizes=(tuple(param.shape),),
                    ),
                )
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    # Shard dims other than 0, which FSDP2 gets from shard_placement_fn.
    shard_dims = {
        param: placement.dim
        for unit in units
        for param, placement in unit.placements.items()
    }

    def placement_fn(named_params, mesh):
        del mesh
        return {
            fqn: (custom_placements.Shard(shard_dims.get(param, 0)),)
            for fqn, param in named_params
        }

    def bucket(
        patterns: list[str],
        reshard_after_forward: bool,
        mesh: DeviceMesh = bucket_mesh,
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
    unit_buckets: list[list[BucketSpec]] = []
    for unit in units:
        if expert_mesh is None or unit.routed_experts is None:
            patterns = [module_fqns[module] for module in unit.modules]
            unit_buckets.append([bucket(patterns, unit.reshard_after_forward)])
            continue
        # The expert bucket hooks the routed experts. The other bucket names
        # the rest of the block's parameters, so it hooks their deepest common
        # module, the block.
        expert_params = set(unit.routed_experts.parameters())
        dense_fqns = [
            param_fqns[param]
            for module in unit.modules
            for param in module.parameters()
            if param not in expert_params
        ]
        unit_buckets.append(
            [
                bucket(dense_fqns, unit.reshard_after_forward),
                bucket(
                    [module_fqns[unit.routed_experts]],
                    unit.reshard_after_forward,
                    expert_mesh,
                ),
            ]
        )
    buckets = [spec for specs in unit_buckets for spec in specs]
    # fully_shard(model) puts the remaining parameters in the root unit, which
    # FSDP2 does not reshard after forward.
    params_in_units = {
        param
        for unit in units
        for module in unit.modules
        for param in module.parameters()
    }
    root_fqns = [
        fqn for fqn, param in model.named_parameters() if param not in params_in_units
    ]
    if root_fqns:
        buckets.append(bucket(root_fqns, reshard_after_forward=False))

    flex_shard_module = flex_shard(model, buckets=buckets)
    # FlexShard parameters are plain local shards. Wrap each one's param_init
    # function so that it fills the full parameter and the shard keeps its part:
    # one device's initialization, whatever the sharding. to_empty() replaces
    # the parameters but keeps the modules, so the wrapped functions survive it.
    # validate_model rejects modules without param_init, whose
    # reset_parameters() would bypass this.
    for fqn, layout in get_flex_shard_global_layouts(model).items():
        module_fqn, _, name = fqn.rpartition(".")
        module = model.get_submodule(module_fqn)
        assert isinstance(module, Module) and module._param_init is not None, fqn
        # A skipped init, e.g. tok_embeddings' with tied weights, which lm_head
        # initializes, stays a no-op on the shard.
        if module._param_init.get(name, skip_param_init) is not skip_param_init:
            # A new dict: modules built from one Config share its param_init.
            module._param_init = {
                **module._param_init,
                name: partial(
                    init_from_full, layout=layout, init_fn=module._param_init[name]
                ),
            }
    declare_state_dict_layouts(model)
    if expert_mesh is not None:
        # Every bucket holds parameters, so there is one storage per bucket.
        assert len(flex_shard_module.sharded_bucket_storages) == len(buckets)
        storages = iter(flex_shard_module.sharded_bucket_storages)
        unit_storages = {}
        for unit, specs in zip(units, unit_buckets):
            storages_of_unit = [next(storages) for _ in specs]
            unit_storages.update(dict.fromkeys(unit.modules, storages_of_unit))
            # FSDP2 unshards a block's expert group with the block, so the
            # expert bucket prefetches nothing itself.
            for storage in storages_of_unit[1:]:
                storage.set_buckets_to_forward_prefetch([])
                storage.set_buckets_to_backward_prefetch([])
        # FlexShard's default prefetches the next bucket in forward order. A
        # block's expert bucket would then gather the next block's dense bucket
        # before EP's device-to-host sync, but the next block's expert bucket
        # only from that block's start, after the sync. These lists match
        # FSDP2's schedule instead, which gathers the whole next block from this
        # block's start.
        order = [unit_storages[module] for module in decoder_prefetch_order(model)]
        for current, following in zip(order, order[1:]):
            current[0].set_buckets_to_forward_prefetch(following)
            # In backward, FSDP2 prefetches a module's groups in reverse.
            following[0].set_buckets_to_backward_prefetch(current[::-1])
    logger.info("Applied FlexShard to the model")
