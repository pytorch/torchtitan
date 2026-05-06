# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import fnmatch
import logging
import sys
from collections.abc import Callable, Generator

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch._prims_common import make_contiguous_strides_for
from torch.distributed.fsdp import DataParallelMeshDims
from .placements import (
    _MixedPrecisionCast,
    _with_fqn,
    disable_active_parametrization,
    EagerAllGatherResult,
    EagerReduceScatterResult,
    Placement,
    Shard,
)
from .utils import (
    _validate_bucket_placements,
    _validate_eager_params,
    _validate_flex_shard_mesh,
    _validate_placements,
)


if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


@dataclass
class EagerAllGatherContext:
    """Communication streams for eager batched collectives."""

    all_gather_stream: torch.Stream
    reduce_scatter_stream: torch.Stream
    buckets: list[EagerAllGatherBucket] = field(default_factory=list)
    pending: PendingEagerAllGather | None = None
    reduce_scatter_states: list[EagerReduceScatterResult] = field(default_factory=list)
    reduce_scatter_callback_queued: bool = False


@dataclass
class EagerAllGatherBucket:
    """Runtime metadata for one eager batched all-gather bucket."""

    storage: DStorage
    entries: list[tuple[nn.Module, str, nn.Module, ParamInfo]]
    infos: list[ParamInfo]
    debug_fqn: str | None
    use_autograd_unshard: bool


@dataclass
class PendingEagerAllGather:
    """The single one-bucket-ahead eager all-gather in flight."""

    bucket: EagerAllGatherBucket
    result: EagerAllGatherResult
    recompute: bool


@dataclass
class EagerBucketAllGatherRuntime:
    """Runtime metadata passed to eager RAF bucket autograd."""

    prefetched_result: EagerAllGatherResult | None
    infos: list[ParamInfo]
    param_refs: list[tuple[nn.Module, str]]
    mesh: DeviceMesh
    context: EagerAllGatherContext
    debug_fqn: str | None


def _queue_reduce_scatter_wait(context: EagerAllGatherContext) -> None:
    """Queue a post-backward wait for eager reduce-scatter work."""
    if context.reduce_scatter_callback_queued:
        return
    context.reduce_scatter_callback_queued = True

    def _wait_for_reduce_scatter() -> None:
        try:
            for result in context.reduce_scatter_states:
                Shard.wait_for_reduce_grad(result)
                Shard.release_reduce_grad_buffers(
                    result,
                    release_sharded_grads=True,
                )
        finally:
            context.reduce_scatter_states.clear()
            context.reduce_scatter_callback_queued = False

    torch.autograd.Variable._execution_engine.queue_callback(_wait_for_reduce_scatter)


def _wait_and_clear_reduce_scatter_states(
    context: EagerAllGatherContext,
    debug_fqn: str | None,
) -> None:
    """Wait for prior eager reduce-scatter states and release their buffers."""
    if not context.reduce_scatter_states:
        return
    with torch.profiler.record_function(
        _with_fqn("FlexShard::post_backward_rs_wait", debug_fqn)
    ):
        for result in context.reduce_scatter_states:
            Shard.wait_for_reduce_grad(result)
            Shard.release_reduce_grad_buffers(
                result,
                release_sharded_grads=True,
            )
        context.reduce_scatter_states.clear()


class _EagerBucketAllGather(torch.autograd.Function):
    """Autograd boundary for eager RAF bucket all-gather.

    Forward consumes a raw all-gather result, either prefetched by the previous
    bucket or launched on demand. Backward packs full-parameter gradients and
    launches one explicit bucket reduce-scatter.
    """

    @staticmethod
    def forward(
        ctx: Any,
        runtime: EagerBucketAllGatherRuntime,
        *local_shards: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.runtime = runtime
        ctx.num_inputs = len(local_shards)

        result = runtime.prefetched_result
        runtime.prefetched_result = None
        if result is None:
            result = Shard.begin_unshard(
                [shard.detach() for shard in local_shards],
                runtime.infos,
                runtime.mesh,
                runtime.context.all_gather_stream,
                debug_fqn=runtime.debug_fqn,
            )
        full_params = Shard.finish_unshard(result)
        return tuple(full_params)

    @staticmethod
    def backward(
        ctx: Any,
        *full_param_grads: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        runtime: EagerBucketAllGatherRuntime = ctx.runtime
        grads: list[torch.Tensor] = []
        valid_infos: list[ParamInfo] = []
        valid_param_refs: list[tuple[nn.Module, str]] = []
        for grad, info, param_ref in zip(
            full_param_grads,
            runtime.infos,
            runtime.param_refs,
            strict=True,
        ):
            if grad is None:
                continue
            grads.append(grad.contiguous())
            valid_infos.append(info)
            valid_param_refs.append(param_ref)

        if grads:
            with torch.no_grad():
                _wait_and_clear_reduce_scatter_states(
                    runtime.context,
                    runtime.debug_fqn,
                )
                result = Shard.begin_reduce_grad(
                    grads,
                    valid_infos,
                    runtime.mesh,
                    runtime.context.reduce_scatter_stream,
                    debug_fqn=runtime.debug_fqn,
                )
                stored_grads: list[torch.Tensor] = []
                with torch.cuda.stream(runtime.context.reduce_scatter_stream):
                    for (leaf, name), grad in zip(
                        valid_param_refs,
                        result.sharded_grads,
                        strict=True,
                    ):
                        param = leaf._parameters[name]
                        if grad.dtype != param.dtype:
                            grad = grad.to(param.dtype)
                        stored_grads.append(grad)
                        if param.grad is None:
                            param.grad = grad
                        else:
                            param.grad += grad
                    result.sharded_grads = stored_grads
                    result.event = torch.cuda.Event()
                    result.event.record(runtime.context.reduce_scatter_stream)
                runtime.context.reduce_scatter_states.append(result)
                _queue_reduce_scatter_wait(runtime.context)

        # Gradients are accumulated into the original sharded parameters above
        # so the autograd input grads can stay empty and avoid blocking here.
        return (None, *([None] * ctx.num_inputs))


__all__ = [
    "auto_buckets",
    "BucketSpec",
    "disable_active_parametrization",
    "flex_shard",
    "get_global_shape",
    "get_placements",
    "is_flex_shard_param",
    "MixedPrecisionPolicy",
    "set_sharding_info",
]


# Module attribute names for storing DStorage
_DSTORAGE_ATTR = "_dstorage"
_DSTORAGES_ATTR = "_dstorages"

# Hidden attribute names for placement metadata on plain tensors
_PLACEMENTS_ATTR = "_placements"
_GLOBAL_SHAPE_ATTR = "_global_shape"
_GLOBAL_STRIDE_ATTR = "_global_stride"
_MESH_ATTR = "_mesh"
_REQUIRES_EAGER_BATCHED_UNSHARD_ATTR = "_flex_shard_requires_eager_batched_unshard"
_EAGER_BATCHED_HOOK_REGISTERED_ATTR = "_flex_shard_eager_batched_hook_registered"
_EAGER_COMM_CONTEXTS_ATTR = "_flex_shard_eager_comm_contexts"
_PARAM_FQN_ATTR = "_flex_shard_param_fqn"
_BUCKET_FQN_ATTR = "_flex_shard_bucket_fqn"
_EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR = "_flex_shard_eager_autograd_bucket_unshard"


@dataclass(frozen=True)
class FlexShardMeshInfo:
    """Mesh metadata for FlexShard's data-parallel shard view.

    ``dp_shard_mesh`` is the one-dimensional mesh used for FlexShard
    collectives and storage sharding.
    """

    dp_shard_mesh: DeviceMesh


def _get_submesh(mesh: DeviceMesh, names: tuple[str, ...]) -> DeviceMesh:
    """Return one mesh dim or flatten several named mesh dims."""
    if len(names) == 1:
        return mesh[names[0]]
    return mesh[names]._flatten("_".join(names))


def _get_flex_shard_mesh_info(
    mesh: DeviceMesh,
    dp_mesh_dims: DataParallelMeshDims,
) -> FlexShardMeshInfo:
    """Derive FlexShard's DP shard mesh from the input mesh."""
    _validate_flex_shard_mesh(mesh, dp_mesh_dims)

    assert mesh.mesh_dim_names is not None
    shard_names = dp_mesh_dims.shard_names
    return FlexShardMeshInfo(
        dp_shard_mesh=_get_submesh(mesh, shard_names),
    )


def _get_device_from_mesh(mesh: DeviceMesh) -> torch.device:
    """Return the current rank's device for ``mesh``."""
    if mesh.device_type == "cpu":
        return torch.device("cpu")
    if mesh.device_type == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    try:
        device_module = torch.get_device_module(mesh.device_type)
    except (AttributeError, RuntimeError):
        return torch.device(mesh.device_type)
    return torch.device(mesh.device_type, device_module.current_device())


# ---------------------------------------------------------------------------
# Phase 2b: Bucket assignment and validation
# ---------------------------------------------------------------------------


def _assign_params_to_buckets(
    param_fqns: list[str],
    buckets: list[BucketSpec],
) -> list[list[str]]:
    """Assign each param FQN to exactly one bucket via fnmatch.

    Returns:
        List of lists: assignments[i] = [fqn, ...] for bucket i.

    Raises:
        ValueError: if any param matches zero or multiple buckets.
    """
    param_to_buckets: dict[str, list[int]] = {fqn: [] for fqn in param_fqns}
    for bucket_idx, bucket in enumerate(buckets):
        for fqn in param_fqns:
            for pattern in bucket.patterns:
                if fnmatch.fnmatch(fqn, pattern):
                    param_to_buckets[fqn].append(bucket_idx)
                    break  # one match per bucket is enough

    # Check for orphans
    orphans = [fqn for fqn, idxs in param_to_buckets.items() if len(idxs) == 0]
    if orphans:
        orphan_list = "\n  ".join(orphans)
        raise ValueError(
            f"flex_shard: {len(orphans)} parameters not covered by any bucket:\n"
            f"  {orphan_list}\n"
            'Add these to an existing bucket or add a catch-all bucket: ["*"]'
        )

    # Check for overlaps
    overlaps = {fqn: idxs for fqn, idxs in param_to_buckets.items() if len(idxs) > 1}
    if overlaps:
        lines = []
        for fqn, idxs in overlaps.items():
            bucket_descs = ", ".join(f"bucket {i} {buckets[i].patterns}" for i in idxs)
            lines.append(f"  {fqn} -> {bucket_descs}")
        overlap_list = "\n".join(lines)
        raise ValueError(
            f"flex_shard: {len(overlaps)} parameters matched multiple buckets:\n"
            f"{overlap_list}\n"
            "Ensure each parameter matches exactly one bucket."
        )

    # Build assignments
    assignments: list[list[str]] = [[] for _ in buckets]
    for fqn, idxs in param_to_buckets.items():
        assignments[idxs[0]].append(fqn)

    return assignments


# ---------------------------------------------------------------------------
# Phase 2a: Property-based parametrization registration
# ---------------------------------------------------------------------------

_reshard_checkpoint_enabled: ContextVar[bool] = ContextVar(
    "_reshard_checkpoint_enabled",
    default=True,
)
_reshard_checkpoint_recompute: ContextVar[bool] = ContextVar(
    "_reshard_checkpoint_recompute",
    default=False,
)


@contextmanager
def _mark_reshard_checkpoint_recompute(ctx: Any) -> Generator[None, None, None]:
    """Mark execution as FlexShard checkpoint recomputation."""
    token = _reshard_checkpoint_recompute.set(True)
    try:
        with ctx:
            yield
    finally:
        _reshard_checkpoint_recompute.reset(token)


def _is_graph_capture_active() -> bool:
    """Return whether unsupported graph capture is active."""
    if torch.compiler.is_compiling():
        return True
    try:
        return torch._guards.TracingContext.try_get() is not None
    except AttributeError:
        return False


def _raise_graph_capture_unsupported() -> None:
    raise ValueError(
        "FlexShard currently supports eager execution only; torch.compile and "
        "graph capture are not supported yet."
    )


def _raise_missing_eager_batched_unshard(parametrization: nn.Module) -> None:
    param_fqn = getattr(parametrization, _PARAM_FQN_ATTR, "<unknown>")
    bucket_fqn = getattr(parametrization, _BUCKET_FQN_ATTR, None)
    bucket_msg = f" in bucket {bucket_fqn!r}" if bucket_fqn else ""
    raise RuntimeError(
        "FlexShard eager mode would fall back to per-parameter "
        f"_c10d_functional collectives for parameter {param_fqn!r}{bucket_msg}, "
        "but eager Shard(0) parameters must be served by a batched "
        "all-gather hook. This usually means the BucketSpec boundary does not "
        "match the module hook/checkpoint execution unit. Split the bucket to "
        "match forward module boundaries."
    )


def _get_or_create_eager_comm_context(
    root_module: nn.Module,
    device: torch.device,
) -> EagerAllGatherContext:
    contexts = getattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, None)
    if contexts is None:
        contexts = {}
        setattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, contexts)

    context = contexts.get(device)
    if context is None:
        context = EagerAllGatherContext(
            all_gather_stream=torch.cuda.Stream(device=device, priority=-1),
            reduce_scatter_stream=torch.cuda.Stream(device=device, priority=-1),
        )
        contexts[device] = context
    return context


_wrap_class_counter = 0


def _register_parametrization(
    module: nn.Module,
    param_parametrizations: dict[str, nn.Module],
) -> None:
    """Register per-parameter property getters that call parametrization forward.

    Uses dynamic subclass creation (not nn.utils.parametrize) to avoid
    state_dict key mangling. state_dict reads self._parameters directly,
    bypassing property getters.

    Args:
        module: The leaf module owning the parameters.
        param_parametrizations: Maps parameter name to its parametrization module.
    """
    global _wrap_class_counter
    _wrap_class_counter += 1

    def _make_getter(pn, p):
        def getter(self):
            # In eager batched mode, _pre_gathered is set on the
            # parametrization by the batched all-gather pre_forward hook.
            pre = getattr(p, "_pre_gathered", None)
            if pre is not None:
                p._pre_gathered = None
                param_dtype = getattr(p, "param_dtype", None)
                reduce_dtype = getattr(p, "reduce_dtype", None)
                if getattr(p, _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR, False):
                    if param_dtype is not None or reduce_dtype is not None:
                        pre = _MixedPrecisionCast.apply(pre, param_dtype, reduce_dtype)
                    return pre

                if param_dtype is not None and pre.dtype != param_dtype:
                    pre = pre.to(param_dtype)
                unsharded = pre.detach().requires_grad_(True)
                if (
                    torch.is_grad_enabled()
                    and getattr(p, "_unsharded_for_reduce", None) is None
                ):
                    p._unsharded_for_reduce = unsharded
                return unsharded
            if _is_graph_capture_active():
                _raise_graph_capture_unsupported()
            if getattr(p, _REQUIRES_EAGER_BATCHED_UNSHARD_ATTR, False) and not getattr(
                p, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, False
            ):
                _raise_missing_eager_batched_unshard(p)
            return p(self._parameters[pn])

        return getter

    param_name_to_property = {
        param_name: property(_make_getter(param_name, param))
        for param_name, param in param_parametrizations.items()
    }
    module_cls = type(
        f"FlexShard{module.__class__.__name__}_{_wrap_class_counter}",
        (module.__class__,),
        param_name_to_property,
    )
    module.__class__ = module_cls
    sys.modules[module_cls.__module__].__dict__[module_cls.__name__] = module_cls


def set_sharding_info(
    tensor: torch.Tensor,
    placements: tuple[Placement, ...],
    global_shape: torch.Size,
    global_stride: tuple[int, ...],
    mesh: DeviceMesh,
) -> None:
    """Annotate a tensor with FlexShard placement metadata."""
    tensor._placements = placements
    tensor._global_shape = global_shape
    tensor._global_stride = global_stride
    tensor._mesh = mesh


def get_placements(tensor: torch.Tensor) -> tuple[Placement, ...] | None:
    """Get FlexShard placements from a tensor, or None if not annotated."""
    return getattr(tensor, _PLACEMENTS_ATTR, None)


def get_global_shape(tensor: torch.Tensor) -> torch.Size | None:
    """Get the global (unsharded) shape from a tensor, or None if not annotated."""
    return getattr(tensor, _GLOBAL_SHAPE_ATTR, None)


def is_flex_shard_param(tensor: torch.Tensor) -> bool:
    """Check if a tensor has FlexShard placement metadata."""
    return hasattr(tensor, _PLACEMENTS_ATTR)


class FlexShardModule:
    """Mixin added to modules after flex_shard()."""

    @property
    def dstorage(self) -> DStorage:
        """First (or only) DStorage. For multi-bucket, use .dstorages."""
        return getattr(self, _DSTORAGE_ATTR)

    @property
    def dstorages(self) -> list:
        """All DStorage instances (one per bucket)."""
        return getattr(self, _DSTORAGES_ATTR)

    @property
    def eager_comm_contexts(self) -> dict[torch.device, EagerAllGatherContext]:
        """Root-owned eager communication contexts keyed by CUDA device."""
        return getattr(self, _EAGER_COMM_CONTEXTS_ATTR)


@dataclass(frozen=True)
class MixedPrecisionPolicy:
    """Mixed precision policy for FlexShard buckets.

    Args:
        param_dtype: Dtype for forward compute. Parameters are all-gathered
            in storage dtype, then cast to param_dtype. If None, no cast.
        reduce_dtype: Dtype for gradient reduction. Gradients are cast to
            this dtype before reduce-scatter. If None, uses param_dtype
            (or storage dtype if param_dtype is also None).
    """

    param_dtype: torch.dtype | None = None
    reduce_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class OffloadPolicy:
    """CPU offload policy for FlexShard buckets.

    When set on a BucketSpec, the bucket's byte storage is allocated on
    CPU (optionally pinned). The parametrization handles H2D transfer
    before all-gather; backward autograd handles D2H automatically.

    Args:
        pin_memory: Whether to pin CPU memory for faster H2D/D2H
            transfers via DMA. Set to False if insufficient CPU memory.
            Default True.
    """

    pin_memory: bool = True


@dataclass(frozen=True)
class BucketSpec:
    """Specification for a parameter communication bucket.

    Args:
        patterns: fnmatch glob patterns matched against parameter FQNs.
            A parameter matches this bucket if its FQN matches any pattern.
        mp_policy: Mixed precision policy for this bucket.
        offload_policy: CPU offload policy for this bucket.
        reshard_after_forward: Whether to free this bucket's unsharded
            parameters after forward and recompute them in backward.
    """

    patterns: list[str]
    mp_policy: MixedPrecisionPolicy | None = None
    offload_policy: OffloadPolicy | None = None
    reshard_after_forward: bool = True


# ---------------------------------------------------------------------------
# Eager reshard-after-forward via checkpoint + selective AC policy
# ---------------------------------------------------------------------------

# These produce the unsharded param tensors that we want freed per-layer.
_FLEX_SHARD_COLLECTIVE_OPS = {
    torch.ops._c10d_functional.all_gather_into_tensor.default,
    torch.ops._c10d_functional.wait_tensor.default,
    torch.ops._c10d_functional.broadcast.default,
}


def _flex_shard_reshard_policy(ctx, func, *args, **kwargs):
    """Checkpoint policy for per-layer reshard-after-forward.

    Marks collective ops (all-gather, broadcast, wait_tensor) for
    recomputation — checkpoint discards their outputs after each layer's
    forward. All other ops (matmul, attention, etc.) are saved, avoiding
    redundant compute recomputation in backward.
    """
    from torch.utils.checkpoint import CheckpointPolicy

    if func in _FLEX_SHARD_COLLECTIVE_OPS:
        return CheckpointPolicy.MUST_RECOMPUTE
    # PREFER_RECOMPUTE lets checkpoint decide what to save vs recompute
    # for non-collective ops, matching standard AC behavior.
    return CheckpointPolicy.PREFER_RECOMPUTE


def _compose_reshard_with_ac_policy(ac_context_fn):
    """Compose FlexShard reshard policy with an existing AC context_fn.

    Returns a new context_fn that wraps the AC policy: FlexShard collective
    ops are forced to MUST_RECOMPUTE, everything else delegates to the
    original AC policy. The two op sets are disjoint so no conflicts arise.
    """

    def merged_context_fn():
        from torch.utils.checkpoint import CheckpointPolicy

        contexts = ac_context_fn()
        for ctx in contexts:
            original_policy = getattr(ctx, "policy_fn", None)
            if original_policy is None:
                continue

            def merged_policy(sctx, func, *args, _orig=original_policy, **kwargs):
                if func in _FLEX_SHARD_COLLECTIVE_OPS:
                    return CheckpointPolicy.MUST_RECOMPUTE
                return _orig(sctx, func, *args, **kwargs)

            ctx.policy_fn = merged_policy
        forward_ctx, recompute_ctx = contexts
        return forward_ctx, _mark_reshard_checkpoint_recompute(recompute_ctx)

    return merged_context_fn


@dataclass
class ParamInfo:
    """Metadata for a parameter in chunked storage."""

    fqn: str
    global_shape: torch.Size
    global_stride: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool
    placements: tuple[Placement, ...]
    local_shape: torch.Size = field(default_factory=lambda: torch.Size([]))
    local_numel: int = 0
    byte_offset: int = 0  # byte offset into the sharded storage
    global_numel: int = 0  # total elements in unsharded param


class DStorage:
    """
    Manages a byte buffer that backs one bucket of sharded parameters.

    All parameters in a storage must share the same dtype and use Shard(0)
    placement. Each parameter's local shard is a typed view into this buffer at
    its sequential byte offset.

    Communication is delegated to eager hooks and parametrization modules; this
    storage object owns buffer layout and metadata.
    """

    def __init__(
        self,
        byte_storage: torch.Tensor,
        param_infos: dict[str, ParamInfo],
        mesh: DeviceMesh,
        total_bytes: int,
        module: nn.Module,
        reshard_after_forward: bool = True,
    ) -> None:
        if byte_storage.dtype != torch.uint8:
            raise ValueError(f"Expected uint8 storage, got {byte_storage.dtype}")
        self._byte_storage = byte_storage
        self._param_infos = param_infos
        self._mesh = mesh
        self._total_bytes = total_bytes
        self._module = module
        self._reshard_after_forward = reshard_after_forward

    @property
    def byte_storage(self) -> torch.Tensor:
        """The underlying unified byte storage tensor (sharded)."""
        return self._byte_storage

    @property
    def flat_storage(self) -> torch.Tensor:
        """Alias for byte_storage for backwards compatibility."""
        return self._byte_storage

    @property
    def total_bytes(self) -> int:
        """Total bytes in the sharded storage."""
        return self._total_bytes

    @property
    def numel(self) -> int:
        """Total number of bytes (for compatibility, returns byte count)."""
        return self._byte_storage.numel()

    @property
    def param_infos(self) -> dict[str, ParamInfo]:
        """Metadata for each parameter."""
        return self._param_infos

    @property
    def world_size(self) -> int:
        """World size of the mesh."""
        return self._mesh.size()

    def get_local_view(self, fqn: str) -> torch.Tensor:
        """Get the local tensor view for a parameter by FQN (from sharded storage)."""
        info = self._param_infos[fqn]
        num_bytes = info.local_numel * info.dtype.itemsize
        byte_view = self._byte_storage[info.byte_offset : info.byte_offset + num_bytes]
        typed_flat = byte_view.view(info.dtype)
        return typed_flat.view(info.local_shape)


def _storage_requires_eager_batched_unshard(storage: DStorage) -> bool:
    """Return whether eager execution must use pre-gathered bucket tensors."""
    infos = list(storage._param_infos.values())
    if not infos:
        return False
    return storage.byte_storage.device.type != "cpu"


def _storage_uses_eager_autograd_unshard(storage: DStorage) -> bool:
    """Return whether eager RAF should use the custom bucket autograd path."""
    if not storage._reshard_after_forward:
        return False
    if not storage._param_infos:
        return False
    return _get_eager_raf_custom_bucket_unsupported_reason(storage) is None


def _get_eager_raf_custom_bucket_unsupported_reason(
    storage: DStorage,
) -> str | None:
    """Return why a RAF eager bucket cannot use the custom autograd path."""
    infos = list(storage._param_infos.values())
    if not infos:
        return None
    if storage.byte_storage.device.type != "cuda":
        return f"storage is on {storage.byte_storage.device.type}"
    ptype = type(infos[0].placements[0])
    if ptype is not Shard:
        return f"placement type is {ptype.__name__}"
    shard_dims = sorted({info.placements[0].dim for info in infos})
    if shard_dims != [0]:
        return f"Shard dimension is {shard_dims}"
    return None


def _raise_unsupported_eager_raf_custom_bucket(storage: DStorage) -> None:
    reason = _get_eager_raf_custom_bucket_unsupported_reason(storage)
    bucket_fqn = _get_storage_debug_fqn(storage)
    bucket_msg = f" for bucket {bucket_fqn!r}" if bucket_fqn else ""
    raise NotImplementedError(
        "FlexShard eager reshard_after_forward currently supports only CUDA "
        f"Shard(0) buckets in the custom autograd bucket path{bucket_msg}; "
        f"{reason}. Use reshard_after_forward=False for this bucket or add "
        "support for this placement before using eager RAF."
    )


def _compute_local_info(
    global_shape: torch.Size,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
) -> tuple[torch.Size, int]:
    """Compute local shape and numel for a parameter on current rank."""
    rank = mesh.get_local_rank()
    world_size = mesh.size()
    placement = placements[0]
    local_shape = placement.compute_local_shape(global_shape, rank, world_size)
    local_numel = placement.compute_local_numel(global_shape, rank, world_size)
    return local_shape, local_numel


def auto_buckets(module: nn.Module) -> list[BucketSpec]:
    """Generate one bucket per direct child module.

    Returns a list of ``BucketSpec`` objects suitable for the ``buckets``
    parameter of :func:`flex_shard`. Each bucket contains a single
    ``"child_name.*"`` pattern matching all parameters under that child.

    Example::

        >>> buckets = auto_buckets(model)
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     dp_mesh_dims,
        ...     shard_placement_fn=per_param_placements,
        ...     buckets=buckets,
        ... )
    """
    children = list(module.named_children())
    if not children:
        return [BucketSpec(["*"])]
    return [BucketSpec([f"{name}.*"]) for name, _ in children]


def _create_param_infos(
    named_params: list[tuple[str, nn.Parameter]],
    mesh_info: FlexShardMeshInfo,
    param_placements: dict[str, tuple[Placement, ...]],
) -> tuple[dict[str, ParamInfo], int]:
    """
    Create ParamInfo for each parameter, computing local shapes and byte offsets.

    The caller validates that each bucket uses Shard(0) and a uniform dtype, so
    parameters are laid out sequentially in the byte buffer.

    Args:
        named_params: List of (fqn, param) tuples
        mesh_info: Mesh metadata for sharding
        param_placements: Dict mapping FQN to placement tuple for each parameter

    Returns:
        param_infos: dict mapping FQN to ParamInfo
        total_bytes: total bytes needed for the sharded buffer
    """
    param_infos: dict[str, ParamInfo] = {}
    current_byte_offset = 0

    for fqn, param in named_params:
        placements = param_placements[fqn]
        global_shape = param.shape
        global_stride = make_contiguous_strides_for(global_shape)
        local_shape, local_numel = _compute_local_info(
            global_shape, mesh_info.dp_shard_mesh, placements
        )
        dtype = param.dtype
        global_numel = param.numel()

        # Sharded buffer: only allocate if this rank has data
        if local_numel > 0:
            byte_offset = current_byte_offset
            current_byte_offset += local_numel * dtype.itemsize
        else:
            byte_offset = 0

        info = ParamInfo(
            fqn=fqn,
            global_shape=global_shape,
            global_stride=tuple(global_stride),
            dtype=dtype,
            requires_grad=param.requires_grad,
            placements=placements,
            local_shape=local_shape,
            local_numel=local_numel,
            byte_offset=byte_offset,
            global_numel=global_numel,
        )
        param_infos[fqn] = info

    return param_infos, current_byte_offset


def _create_sharded_view(
    local_view: torch.Tensor,
    info: ParamInfo,
    mesh_info: FlexShardMeshInfo,
) -> torch.Tensor:
    """Annotate a local tensor view with placement metadata."""
    set_sharding_info(
        local_view,
        placements=info.placements,
        global_shape=info.global_shape,
        global_stride=info.global_stride,
        mesh=mesh_info.dp_shard_mesh,
    )
    return local_view


def _set_param_on_module(
    root_module: nn.Module,
    fqn: str,
    param: nn.Parameter,
) -> None:
    """Navigate to submodule by FQN and set parameter."""
    parts = fqn.split(".")
    module = root_module
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], param)


def _write_params_to_dstorage(
    byte_storage: torch.Tensor,
    named_params: list[tuple[str, nn.Parameter]],
    param_infos: dict[str, ParamInfo],
    mesh_info: FlexShardMeshInfo,
) -> None:
    """Pack original parameter data into byte storage.

    Calls placement.extract_local_shard() to get each rank's typed local shard,
    then copies it as uint8 into the byte buffer.
    """
    mesh = mesh_info.dp_shard_mesh
    my_rank = mesh.get_local_rank()
    world_size = mesh.size()

    for fqn, param in named_params:
        info = param_infos[fqn]
        param_data = param.detach()
        if param_data.device.type == "meta":
            continue
        if not param_data.is_contiguous():
            param_data = param_data.contiguous()
        shard = info.placements[0].extract_local_shard(param_data, my_rank, world_size)
        if shard.numel() > 0:
            nbytes = shard.numel() * shard.element_size()
            byte_storage[info.byte_offset : info.byte_offset + nbytes].copy_(
                shard.reshape(-1).view(torch.uint8)
            )


def _get_managed_named_params(
    module: nn.Module,
) -> list[tuple[str, nn.Parameter]]:
    """
    Collect parameters that should be managed by this module's DStorage.

    This excludes parameters from child modules that already have their own
    DStorage (i.e., already wrapped with flex_shard).

    Similar to FSDP2's _get_managed_modules/_get_managed_states pattern.
    """
    managed_params: list[tuple[str, nn.Parameter]] = []

    # Find child modules that already have DStorage
    wrapped_prefixes: set[str] = set()
    for name, child in module.named_modules():
        if name and getattr(child, _DSTORAGE_ATTR, None) is not None:
            # This child is already wrapped; skip its parameters
            wrapped_prefixes.add(name + ".")

    # Collect parameters not in wrapped submodules
    for fqn, param in module.named_parameters():
        is_wrapped = any(fqn.startswith(prefix) for prefix in wrapped_prefixes)
        if not is_wrapped:
            managed_params.append((fqn, param))

    return managed_params


PlacementFn = Callable[
    [list[tuple[str, nn.Parameter]], "DeviceMesh"],
    dict[str, tuple[Placement, ...]],
]

logger = logging.getLogger(__name__)


def flex_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    dp_mesh_dims: DataParallelMeshDims,
    shard_placement_fn: PlacementFn,
    buckets: list[BucketSpec],
) -> FlexShardModule:
    """
    Apply flat-storage FSDP sharding to a module.

    This function:
    1. Collects parameters from the module (excluding already-wrapped submodules)
    2. Groups parameters into communication buckets (one per bucket, or all in one)
    3. Creates a unified byte buffer per bucket for all its parameters
    4. Replaces each parameter with a plain tensor annotated with placement metadata
    5. Registers property-based parametrization for eager parameter access
    6. Stores DStorages on the module (accessible via module.dstorages)

    Each bucket gets its own byte buffer and DStorage, enabling independent
    all-gather operations per bucket.

    Nested wrapping is supported: apply flex_shard to inner modules first,
    then to outer modules. The outer module's storage will exclude parameters
    from already-wrapped inner modules.

    Args:
        module: The module to shard. Can have real or meta device parameters.
        mesh: The named device mesh for sharding.
        dp_mesh_dims: Names for the data-parallel dimensions in ``mesh``.
            FlexShard derives its DP shard mesh from ``mesh``.
        shard_placement_fn: Required callable that maps
            ``(named_params, dp_shard_mesh)`` to per-parameter placements.
            The minimal eager path currently supports ``Shard(0)`` placements,
            for example via ``per_param_placements``.
        buckets: Required list of bucket specifications. Use
            ``[BucketSpec(["*"])]`` for a single whole-module bucket or
            ``auto_buckets()`` to generate one bucket per direct child module.

    Returns:
        The module (mutated in-place). Use module.dstorages to access internals.

    Example::

        >>> mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("fsdp",))
        >>> model = Transformer(args)
        >>> model.to("cuda")
        >>> # Single bucket:
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     DataParallelMeshDims(shard="fsdp"),
        ...     shard_placement_fn=per_param_placements,
        ...     buckets=[BucketSpec(["*"])],
        ... )
        >>> # Explicit buckets:
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     DataParallelMeshDims(shard="fsdp"),
        ...     shard_placement_fn=per_param_placements,
        ...     buckets=[BucketSpec(["attn.*"]), BucketSpec(["ffn.*"])],
        ... )
        >>> # Auto buckets (one per child):
        >>> flex_shard(
        ...     model,
        ...     mesh,
        ...     DataParallelMeshDims(shard="fsdp"),
        ...     shard_placement_fn=per_param_placements,
        ...     buckets=auto_buckets(model),
        ... )

    Note:
        - Each bucket must contain only ``Shard(0)`` placements.
        - Each bucket must contain one original parameter dtype. Split mixed
          dtype parameters into separate buckets.
        - Parameters on meta device will have uninitialized storage
    """
    # Check if module is already wrapped
    if getattr(module, _DSTORAGES_ATTR, None) is not None:
        raise ValueError(
            f"Module {type(module).__name__} already has DStorage. "
            "Cannot apply flex_shard twice to the same module."
        )

    # Collect parameters (excluding those from already-wrapped submodules)
    named_params = _get_managed_named_params(module)
    if not named_params:
        raise ValueError(
            f"Module {type(module).__name__} has no parameters to shard. "
            "All parameters may belong to already-wrapped submodules."
        )
    mesh_info = _get_flex_shard_mesh_info(mesh, dp_mesh_dims)
    shard_mesh = mesh_info.dp_shard_mesh

    # Determine device - use meta only if all params are meta, otherwise use mesh device.
    all_params_meta = all(param.device.type == "meta" for _, param in named_params)
    if all_params_meta:
        device = torch.device("meta")
    else:
        device = _get_device_from_mesh(shard_mesh)
    _validate_eager_params(
        named_params,
        expected_device=None if all_params_meta else device,
    )

    # Resolve placements for all params
    param_placements = shard_placement_fn(named_params, shard_mesh)
    expected_fqns = {fqn for fqn, _ in named_params}
    actual_fqns = set(param_placements)
    missing_fqns = expected_fqns - actual_fqns
    extra_fqns = actual_fqns - expected_fqns
    if missing_fqns or extra_fqns:
        msg_parts = []
        if missing_fqns:
            msg_parts.append(f"missing placements for {sorted(missing_fqns)}")
        if extra_fqns:
            msg_parts.append(f"unexpected placements for {sorted(extra_fqns)}")
        raise ValueError(
            "shard_placement_fn must return placements for exactly the managed "
            f"parameters; {', '.join(msg_parts)}."
        )
    _validate_placements(param_placements, named_params, shard_mesh)

    if not buckets:
        raise ValueError("flex_shard requires at least one BucketSpec in buckets.")
    if not all(isinstance(bucket, BucketSpec) for bucket in buckets):
        raise TypeError("flex_shard buckets must be a list of BucketSpec objects.")

    param_fqns = [fqn for fqn, _ in named_params]
    bucket_assignments = _assign_params_to_buckets(param_fqns, buckets)
    _validate_bucket_placements(
        bucket_assignments,
        param_placements,
        buckets,
        named_params,
    )

    # Log bucket coverage
    if logger.isEnabledFor(logging.DEBUG):
        lines = ["flex_shard bucket coverage:"]
        total_params = 0
        for i, fqns in enumerate(bucket_assignments):
            patterns = buckets[i].patterns
            lines.append(f"  bucket {i} {patterns}: {len(fqns)} params")
            total_params += len(fqns)
        lines.append(f"  total: {total_params} params across {len(buckets)} buckets")
        logger.debug("\n".join(lines))

    # Per-bucket: create param_infos, byte buffer, replace params, create DStorage
    named_params_dict = dict(named_params)
    storages: list[DStorage] = []
    fqn_to_mp_policy: dict[str, MixedPrecisionPolicy | None] = {}
    fqn_to_offload_policy: dict[str, OffloadPolicy | None] = {}

    for bucket_idx, bucket_fqns in enumerate(bucket_assignments):
        if not bucket_fqns:
            continue

        # Extract per-bucket policies
        bucket_mp_policy = None
        bucket_offload_policy = None
        bucket_spec = buckets[bucket_idx]
        bucket_mp_policy = bucket_spec.mp_policy
        bucket_offload_policy = bucket_spec.offload_policy
        bucket_reshard_after_forward = bucket_spec.reshard_after_forward
        for fqn in bucket_fqns:
            fqn_to_mp_policy[fqn] = bucket_mp_policy
            fqn_to_offload_policy[fqn] = bucket_offload_policy

        bucket_named_params = [(fqn, named_params_dict[fqn]) for fqn in bucket_fqns]
        bucket_placements = {fqn: param_placements[fqn] for fqn in bucket_fqns}

        param_infos, total_bytes = _create_param_infos(
            bucket_named_params, mesh_info, bucket_placements
        )

        if bucket_offload_policy is not None:
            byte_storage = torch.empty(
                total_bytes,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=bucket_offload_policy.pin_memory,
            )
        else:
            byte_storage = torch.empty(total_bytes, dtype=torch.uint8, device=device)
        _write_params_to_dstorage(
            byte_storage, bucket_named_params, param_infos, mesh_info
        )

        for fqn, info in param_infos.items():
            local_view = byte_storage[
                info.byte_offset : info.byte_offset
                + info.local_numel * info.dtype.itemsize
            ]
            typed_view = local_view.view(info.dtype).view(info.local_shape)
            new_param = nn.Parameter(typed_view, requires_grad=info.requires_grad)
            expected_param_device = (
                torch.device("cpu") if bucket_offload_policy is not None else device
            )
            if new_param.device != expected_param_device:
                raise AssertionError(
                    f"Expected sharded parameter {fqn!r} on "
                    f"{expected_param_device}, but got {new_param.device}"
                )
            _create_sharded_view(new_param, info, mesh_info)
            _set_param_on_module(module, fqn, new_param)

        storage = DStorage(
            byte_storage,
            param_infos,
            shard_mesh,
            total_bytes,
            module,
            reshard_after_forward=bucket_reshard_after_forward,
        )
        storages.append(storage)

    # Store DStorages on module
    setattr(module, _DSTORAGES_ATTR, storages)
    setattr(module, _DSTORAGE_ATTR, storages[0] if storages else None)
    setattr(module, _EAGER_COMM_CONTEXTS_ATTR, {})

    # Change module class to include FlexShardModule mixin
    cls = type(module)
    if not issubclass(cls, FlexShardModule):
        module.__class__ = type(cls.__name__, (cls, FlexShardModule), {})

    # Register property-based parametrization.
    group_name = shard_mesh.get_group().group_name
    world_size = shard_mesh.size()

    # Group parametrizations by leaf module (across all buckets)
    module_param_map: dict[nn.Module, dict[str, nn.Module]] = {}

    for s in storages:
        requires_eager_batched_unshard = _storage_requires_eager_batched_unshard(s)
        uses_eager_autograd_unshard = _storage_uses_eager_autograd_unshard(s)
        bucket_fqn = _get_storage_debug_fqn(s)
        for fqn, info in s._param_infos.items():
            mp = fqn_to_mp_policy.get(fqn)
            offload = fqn_to_offload_policy.get(fqn)
            param_dtype = mp.param_dtype if mp else None
            reduce_dtype = mp.reduce_dtype if mp else None
            compute_device = torch.device(device) if offload is not None else None
            placement = info.placements[0]
            p = placement.create_parametrization(
                info,
                group_name,
                world_size,
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                compute_device=compute_device,
            )

            setattr(
                p,
                _REQUIRES_EAGER_BATCHED_UNSHARD_ATTR,
                requires_eager_batched_unshard,
            )
            setattr(
                p,
                _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR,
                uses_eager_autograd_unshard,
            )
            setattr(p, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, False)
            setattr(p, _PARAM_FQN_ATTR, fqn)
            setattr(p, _BUCKET_FQN_ATTR, bucket_fqn)

            # Find the leaf module owning this param
            parts = fqn.split(".")
            leaf_mod = module
            for part in parts[:-1]:
                leaf_mod = getattr(leaf_mod, part)
            local_name = parts[-1]

            if leaf_mod not in module_param_map:
                module_param_map[leaf_mod] = {}
            module_param_map[leaf_mod][local_name] = p

    for mod, param_map in module_param_map.items():
        _register_parametrization(mod, param_map)

    # Reshard-after-forward: in eager mode, wrap each layer in checkpoint with
    # a selective policy that recomputes only collective ops (all-gather,
    # broadcast), saving compute ops to avoid redundant work.
    reshard_storages = [s for s in storages if s._reshard_after_forward]
    if _reshard_checkpoint_enabled.get() and reshard_storages:
        _apply_reshard_checkpoint(module, reshard_storages)

    # Install batched all-gather hooks for eager mode when the storage layout
    # supports the batched Placement.unshard() path.
    _install_batched_allgather_hooks(storages, module_param_map)

    return module


def _wrap_with_reshard(child: nn.Module) -> nn.Module:
    """Wrap a single module with reshard checkpoint, composing with AC if present.

    If the child is already wrapped by AC's CheckpointWrapper, unwraps it,
    merges the AC policy with FlexShard's reshard policy, and re-wraps once.
    If no AC wrapper exists, wraps with reshard-only policy.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointWrapper,
    )
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    def _reshard_only_context_fn():
        forward_ctx, recompute_ctx = create_selective_checkpoint_contexts(
            _flex_shard_reshard_policy
        )
        return forward_ctx, _mark_reshard_checkpoint_recompute(recompute_ctx)

    if isinstance(child, CheckpointWrapper):
        # AC already applied — unwrap, merge policies, re-wrap
        inner = child._checkpoint_wrapped_module
        ac_kwargs = dict(child.checkpoint_fn.keywords)
        ac_kwargs.pop("use_reentrant", None)
        ac_context_fn = ac_kwargs.pop("context_fn", None)
        if ac_context_fn is not None:
            # Selective AC — merge with reshard policy
            merged_fn = _compose_reshard_with_ac_policy(ac_context_fn)
        else:
            # Full AC — add reshard policy via selective context
            merged_fn = _reshard_only_context_fn
        return checkpoint_wrapper(inner, context_fn=merged_fn, **ac_kwargs)

    # No AC — reshard-only wrapping
    return checkpoint_wrapper(child, context_fn=_reshard_only_context_fn)


def _module_path_common_prefix(paths: list[str]) -> str:
    """Return the common module path prefix for parameter-owner module paths."""
    if not paths:
        return ""
    common_parts = paths[0].split(".") if paths[0] else []
    for path in paths[1:]:
        parts = path.split(".") if path else []
        limit = min(len(common_parts), len(parts))
        i = 0
        while i < limit and common_parts[i] == parts[i]:
            i += 1
        common_parts = common_parts[:i]
        if not common_parts:
            break
    return ".".join(common_parts)


def _strip_checkpoint_wrapped_module_path(path: str) -> str:
    """Remove CheckpointWrapper internals from a dotted module path."""
    return ".".join(
        part for part in path.split(".") if part != "_checkpoint_wrapped_module"
    )


def _get_module_by_path(module: nn.Module, path: str) -> nn.Module:
    """Resolve a dotted module path from a root module."""
    result = module
    for part in path.split("."):
        if part:
            result = getattr(result, part)
    return result


def _set_module_by_path(module: nn.Module, path: str, child: nn.Module) -> None:
    """Set a dotted module path on a root module."""
    parts = path.split(".")
    parent = (
        _get_module_by_path(module, ".".join(parts[:-1])) if len(parts) > 1 else module
    )
    name = parts[-1]
    if isinstance(parent, (nn.ModuleList, nn.Sequential)) and name.isdigit():
        parent[int(name)] = child
    elif isinstance(parent, nn.ModuleDict):
        parent[name] = child
    else:
        setattr(parent, name, child)


def _top_level_owner_path(module: nn.Module, owner_path: str) -> str:
    """Choose the outer module to checkpoint for a parameter owner path."""
    parts = owner_path.split(".")
    if not parts or not parts[0]:
        return ""
    child = getattr(module, parts[0])
    if (
        isinstance(child, (nn.ModuleDict, nn.ModuleList, nn.Sequential))
        and len(parts) > 1
    ):
        return ".".join(parts[:2])
    return parts[0]


def _get_storage_reshard_module_paths(storage: DStorage) -> list[str]:
    """Return module paths to checkpoint for one resharding bucket."""
    owner_paths = sorted(
        {
            _strip_checkpoint_wrapped_module_path(".".join(fqn.split(".")[:-1]))
            for fqn in storage._param_infos
            if "." in fqn
        }
    )
    if not owner_paths:
        return []

    common = _module_path_common_prefix(owner_paths)
    if common:
        target = _get_module_by_path(storage._module, common)
        if isinstance(target, (nn.ModuleDict, nn.ModuleList)):
            return sorted(
                {
                    _top_level_owner_path(storage._module, owner_path)
                    for owner_path in owner_paths
                }
            )
        return [common]
    return sorted(
        {
            _top_level_owner_path(storage._module, owner_path)
            for owner_path in owner_paths
        }
    )


def _get_storage_debug_fqn(storage: DStorage) -> str | None:
    """Return a concise module/bucket FQN for profiler annotations."""
    owner_paths = sorted(
        {
            _strip_checkpoint_wrapped_module_path(".".join(fqn.split(".")[:-1]))
            for fqn in storage._param_infos
        }
    )
    if not owner_paths:
        return None
    common = _module_path_common_prefix(owner_paths)
    if common:
        return common
    top_level_paths = sorted(
        {
            _top_level_owner_path(storage._module, owner_path)
            for owner_path in owner_paths
        }
    )
    top_level_paths = [path for path in top_level_paths if path]
    if not top_level_paths:
        return None
    return ", ".join(top_level_paths)


def _apply_reshard_checkpoint(
    module: nn.Module,
    reshard_storages: list[DStorage],
) -> None:
    """Wrap FlexShard-managed bucket modules in checkpoint for reshard.

    Each selected bucket's owning module gets wrapped with a checkpoint policy that marks
    collective ops (all-gather, broadcast, wait_tensor) as MUST_RECOMPUTE
    so unsharded params are freed after each layer's forward.

    Composes with activation checkpointing: if a child is already wrapped
    by AC's CheckpointWrapper, the two policies are merged into a single
    wrapper (FlexShard collectives → MUST_RECOMPUTE, AC compute ops →
    MUST_SAVE, everything else → PREFER_RECOMPUTE).
    """
    paths: set[str] = set()
    for storage in reshard_storages:
        storage_paths = _get_storage_reshard_module_paths(storage)
        if storage_paths:
            paths.update(storage_paths)
        else:
            paths.update(name for name, _ in module.named_children())

    for path in sorted(paths, key=lambda p: (p.count("."), p)):
        child = _get_module_by_path(module, path)
        _set_module_by_path(module, path, _wrap_with_reshard(child))


def _install_batched_allgather_hooks(
    storages: list,
    module_param_map: dict[nn.Module, dict[str, nn.Module]],
) -> None:
    """Install pre/post forward hooks for batched per-bucket all-gather.

    In eager mode, each DStorage's pre-forward hook runs a single batched
    Placement.unshard() call (one NCCL collective per bucket), then sets
    _pre_gathered on each parametrization module so the property getter
    skips the per-param all-gather.

    Skipped under graph capture. FlexShard currently supports eager execution
    only, so parameter access will raise before collectives are emitted.
    """
    for storage in storages:
        infos = list(storage._param_infos.values())
        if not infos:
            continue

        ptype = type(infos[0].placements[0])
        # Skip batching when the placement/storage does not support the eager
        # batched path. Parametrization remains the source of truth.
        if not _storage_requires_eager_batched_unshard(storage):
            continue
        if storage._reshard_after_forward and not _storage_uses_eager_autograd_unshard(
            storage
        ):
            _raise_unsupported_eager_raf_custom_bucket(storage)

        # Pre-compute (leaf_module, param_name, parametrization, info) for
        # each param in this bucket. Captured at flex_shard() time (before
        # checkpoint wrapping changes the module tree).
        param_entries: list[tuple[nn.Module, str, nn.Module, ParamInfo]] = []
        for info in infos:
            parts = info.fqn.split(".")
            leaf_mod = storage._module
            for part in parts[:-1]:
                child = getattr(leaf_mod, part, None)
                if child is None:
                    wrapped = getattr(leaf_mod, "_checkpoint_wrapped_module", None)
                    if wrapped is not None:
                        leaf_mod = getattr(wrapped, part)
                    else:
                        leaf_mod = getattr(leaf_mod, part)
                else:
                    leaf_mod = child
            local_name = parts[-1]
            # Unwrap CheckpointWrapper to find the original module
            # that's in module_param_map
            if hasattr(leaf_mod, "_checkpoint_wrapped_module"):
                leaf_mod = leaf_mod._checkpoint_wrapped_module
            if leaf_mod in module_param_map:
                param_p = module_param_map[leaf_mod].get(local_name)
                if param_p is not None:
                    param_entries.append((leaf_mod, local_name, param_p, info))

        logger.debug(f"Batched hooks: {len(param_entries)}/{len(infos)} params matched")
        if not param_entries:
            continue

        ag_context = None
        if ptype is Shard and storage.byte_storage.device.type == "cuda":
            device = storage.byte_storage.device
            ag_context = _get_or_create_eager_comm_context(storage._module, device)
        ag_bucket = None
        if ag_context is not None:
            ag_bucket = EagerAllGatherBucket(
                storage=storage,
                entries=param_entries,
                infos=infos,
                debug_fqn=_get_storage_debug_fqn(storage),
                use_autograd_unshard=_storage_uses_eager_autograd_unshard(storage),
            )
        use_autograd_unshard = _storage_uses_eager_autograd_unshard(storage)

        def make_hooks(
            s,
            entries,
            pt,
            all_gather_context,
            all_gather_bucket,
            use_autograd_bucket,
        ):
            # Collected grads from AccumulateGrad hooks (indexed by position)
            collected_grads: dict[int, torch.Tensor] = {}

            def _begin_bucket_unshard(bucket):
                local_shards = [
                    leaf._parameters[name].data for leaf, name, _, _ in bucket.entries
                ]
                return Shard.begin_unshard(
                    local_shards,
                    bucket.infos,
                    bucket.storage._mesh,
                    all_gather_context.all_gather_stream,
                    debug_fqn=bucket.debug_fqn,
                )

            def _wait_bucket_unshard(result):
                Shard.wait_for_unshard(result)
                Shard.release_unshard_buffers(result)

            def _prefetch_next_bucket():
                if all_gather_context.pending is not None:
                    return
                prefetch_order = all_gather_context.buckets
                if _reshard_checkpoint_recompute.get():
                    prefetch_order = prefetch_order[::-1]
                for idx, bucket in enumerate(prefetch_order):
                    if bucket is all_gather_bucket:
                        break
                else:
                    return
                next_idx = idx + 1
                if next_idx >= len(prefetch_order):
                    return
                next_bucket = prefetch_order[next_idx]
                all_gather_context.pending = PendingEagerAllGather(
                    bucket=next_bucket,
                    result=_begin_bucket_unshard(next_bucket),
                    recompute=_reshard_checkpoint_recompute.get(),
                )

            def _take_pending_for_current_bucket():
                pending = all_gather_context.pending
                if pending is None:
                    return None
                if (
                    pending.bucket is all_gather_bucket
                    and pending.recompute == _reshard_checkpoint_recompute.get()
                ):
                    all_gather_context.pending = None
                    return pending.result
                _wait_bucket_unshard(pending.result)
                all_gather_context.pending = None
                return None

            def _reduce_fn():
                """Batched reduce-scatter using collected grads."""
                grads, valid = [], []
                for idx, (leaf, name, param_p, info) in enumerate(entries):
                    g = collected_grads.pop(idx, None)
                    if g is not None:
                        grads.append(g)
                        valid.append((leaf, name, info))
                    param_p._unsharded_for_reduce = None
                if not grads:
                    return
                valid_infos = [i for _, _, i in valid]
                with torch.no_grad():
                    if (
                        all_gather_context is not None
                        and all_gather_bucket is not None
                        and pt is Shard
                    ):
                        _wait_and_clear_reduce_scatter_states(
                            all_gather_context,
                            all_gather_bucket.debug_fqn,
                        )
                        result = Shard.begin_reduce_grad(
                            grads,
                            valid_infos,
                            s._mesh,
                            all_gather_context.reduce_scatter_stream,
                            debug_fqn=all_gather_bucket.debug_fqn,
                        )
                        stored_grads: list[torch.Tensor] = []
                        with torch.cuda.stream(
                            all_gather_context.reduce_scatter_stream
                        ):
                            for (leaf, name, _), rg in zip(
                                valid, result.sharded_grads, strict=True
                            ):
                                param = leaf._parameters[name]
                                if rg.dtype != param.dtype:
                                    rg = rg.to(param.dtype)
                                stored_grads.append(rg)
                                if param.grad is None:
                                    param.grad = rg
                                else:
                                    param.grad += rg
                            result.sharded_grads = stored_grads
                            result.event = torch.cuda.Event()
                            result.event.record(
                                all_gather_context.reduce_scatter_stream
                            )
                        all_gather_context.reduce_scatter_states.append(result)
                        _queue_reduce_scatter_wait(all_gather_context)
                    else:
                        reduced = pt.reduce_grad(grads, valid_infos, s._mesh)
                        for (leaf, name, _), rg in zip(valid, reduced, strict=True):
                            param = leaf._parameters[name]
                            if rg.dtype != param.dtype:
                                rg = rg.to(param.dtype)
                            if param.grad is None:
                                param.grad = rg
                            else:
                                param.grad += rg

            def pre_forward_hook(mod, args):
                if torch.compiler.is_compiling():
                    return
                # Batched all-gather
                local_shards = [
                    (
                        leaf._parameters[name]
                        if use_autograd_bucket
                        else leaf._parameters[name].data
                    )
                    for leaf, name, _, _ in entries
                ]
                entry_infos = [info for _, _, _, info in entries]
                if (
                    all_gather_context is not None
                    and all_gather_bucket is not None
                    and pt is Shard
                ):
                    if use_autograd_bucket:
                        prefetched_result = _take_pending_for_current_bucket()
                        runtime = EagerBucketAllGatherRuntime(
                            prefetched_result=prefetched_result,
                            infos=entry_infos,
                            param_refs=[(leaf, name) for leaf, name, _, _ in entries],
                            mesh=s._mesh,
                            context=all_gather_context,
                            debug_fqn=all_gather_bucket.debug_fqn,
                        )
                        full_params = list(
                            _EagerBucketAllGather.apply(runtime, *local_shards)
                        )
                        _prefetch_next_bucket()
                    else:
                        with torch.no_grad():
                            result = _take_pending_for_current_bucket()
                            if result is None:
                                result = Shard.begin_unshard(
                                    local_shards,
                                    entry_infos,
                                    s._mesh,
                                    all_gather_context.all_gather_stream,
                                    debug_fqn=all_gather_bucket.debug_fqn,
                                )
                            full_params = Shard.finish_unshard(result)
                            _prefetch_next_bucket()
                else:
                    with torch.no_grad():
                        full_params = pt.unshard(local_shards, entry_infos, s._mesh)
                for (_, _, param_p, _), full_param in zip(
                    entries, full_params, strict=True
                ):
                    param_p._pre_gathered = full_param

            def post_forward_hook(mod, args, output):
                if torch.compiler.is_compiling():
                    return
                for _, _, param_p, _ in entries:
                    param_p._pre_gathered = None
                if use_autograd_bucket:
                    return

                # Register AccumulateGrad hooks on detached leaf params.
                # Each hook stores its grad by index. The last hook fires
                # _reduce_fn with all collected grads (u.grad is None at
                # hook time — grads must be captured from the hook argument).
                if torch.is_grad_enabled():
                    collected_grads.clear()
                    leaf_indices = []
                    for idx, (_, _, param_p, _) in enumerate(entries):
                        u = getattr(param_p, "_unsharded_for_reduce", None)
                        if u is not None and u.requires_grad:
                            leaf_indices.append((idx, u))

                    if leaf_indices:
                        grad_count = [0]
                        n = len(leaf_indices)

                        def _make_hook(i):
                            def _on_grad(grad):
                                collected_grads[i] = grad
                                grad_count[0] += 1
                                if grad_count[0] >= n:
                                    _reduce_fn()
                                    grad_count[0] = 0

                            return _on_grad

                        for idx, leaf in leaf_indices:
                            leaf.register_hook(_make_hook(idx))

            return pre_forward_hook, post_forward_hook

        pre_hook, post_hook = make_hooks(
            storage,
            param_entries,
            ptype,
            ag_context,
            ag_bucket,
            use_autograd_unshard,
        )

        # Register hooks on the bucket's child module (not root) so they
        # fire during checkpoint recomputation in backward too.
        # Navigate through CheckpointWrapper to the inner module.
        target = _get_bucket_module(storage)
        if (
            storage._reshard_after_forward
            and target is storage._module
            and any(
                hasattr(child, "_checkpoint_wrapped_module")
                for child in storage._module.modules()
                if child is not storage._module
            )
        ):
            logger.debug(
                "Skipping root-level batched all-gather hook because child "
                "checkpoint recomputation would not replay the root hook.",
            )
            continue
        inner = getattr(target, "_checkpoint_wrapped_module", target)
        inner.register_forward_pre_hook(pre_hook)
        inner.register_forward_hook(post_hook)
        if ag_context is not None and ag_bucket is not None:
            ag_context.buckets.append(ag_bucket)
        for _, _, param_p, _ in param_entries:
            setattr(param_p, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, True)


def _get_bucket_module(storage) -> nn.Module:
    """Find the deepest common ancestor module for a bucket's params.

    For bucket "layers.0.*", returns model.layers[0].
    For bucket "norm.*, output.*" (no common prefix), returns root.
    """
    fqns = list(storage._param_infos.keys())
    # Get module-level prefixes (strip param name)
    prefixes = [".".join(fqn.split(".")[:-1]) for fqn in fqns]
    # Find common prefix
    if not prefixes:
        return storage._module
    common = prefixes[0]
    for p in prefixes[1:]:
        # Find common prefix character by character, then trim to last "."
        i = 0
        while i < len(common) and i < len(p) and common[i] == p[i]:
            i += 1
        common = common[:i]
    # Trim to last complete component (don't split mid-name)
    if "." in common:
        common = common[: common.rfind(".") + 1].rstrip(".")
    elif common and common not in prefixes:
        # Partial match — not a complete component
        common = ""
    if not common:
        return storage._module
    mod = storage._module
    for part in common.split("."):
        mod = getattr(mod, part)
    return mod
