# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn

from .comm_buffer_lifetime import AsyncAllGatherResult, AsyncReduceScatterResult
from .module_wrapping import EagerParamAccessState
from .sharding_metadata import (
    _BUCKET_FQN_ATTR,
    _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR,
    _EAGER_BATCHED_HOOK_REGISTERED_ATTR,
    _EAGER_COMM_CONTEXTS_ATTR,
    _PARAM_FQN_ATTR,
)
from .placements import Shard
from .reshard_after_forward import _reshard_after_forward_recompute
from .storage import BucketSpec, DStorage, ParamInfo
from .utils import _get_storage_debug_fqn, _with_fqn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.getLogger(__name__)

ParamEntry = tuple[nn.Module, str, EagerParamAccessState, ParamInfo]


@dataclass
class AllGatherBucket:
    """Runtime metadata for one batched all-gather bucket."""

    storage: DStorage
    entries: list[ParamEntry]
    infos: list[ParamInfo]
    debug_fqn: str | None
    use_autograd_unshard: bool


@dataclass
class PendingAllGather:
    """The single one-bucket-ahead all-gather in flight."""

    bucket: AllGatherBucket
    result: AsyncAllGatherResult
    recompute: bool


@dataclass
class AllGatherContext:
    """Communication streams for batched collectives."""

    all_gather_stream: torch.Stream
    reduce_scatter_stream: torch.Stream
    buckets: list[AllGatherBucket] = field(default_factory=list)
    pending: PendingAllGather | None = None
    reduce_scatter_states: list[AsyncReduceScatterResult] = field(default_factory=list)
    reduce_scatter_callback_queued: bool = False


@dataclass
class BucketAllGatherRuntime:
    """Runtime metadata passed to RAF bucket autograd."""

    prefetched_result: AsyncAllGatherResult | None
    infos: list[ParamInfo]
    param_refs: list[tuple[nn.Module, str]]
    mesh: DeviceMesh
    context: AllGatherContext
    debug_fqn: str | None


def _queue_reduce_scatter_wait(context: AllGatherContext) -> None:
    """Queue a post-backward wait for reduce-scatter work."""
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
    context: AllGatherContext,
    debug_fqn: str | None,
) -> None:
    """Wait for prior reduce-scatter states and release their buffers."""
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


def _accumulate_sharded_grads(
    param_refs: list[tuple[nn.Module, str]],
    sharded_grads: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Cast sharded grads to local param dtype and accumulate into .grad."""
    stored_grads: list[torch.Tensor] = []
    for (leaf, name), grad in zip(param_refs, sharded_grads, strict=True):
        param = leaf._parameters[name]
        if grad.dtype != param.dtype:
            grad = grad.to(param.dtype)
        stored_grads.append(grad)
        if param.grad is None:
            param.grad = grad
        else:
            param.grad += grad
    return stored_grads


class _BucketAllGather(torch.autograd.Function):
    """Autograd boundary for RAF bucket all-gather.

    Forward consumes a raw all-gather result, either prefetched by the previous
    bucket or launched on demand. Backward packs full-parameter gradients and
    launches one explicit bucket reduce-scatter.
    """

    @staticmethod
    def forward(
        ctx: Any,
        runtime: BucketAllGatherRuntime,
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
        runtime: BucketAllGatherRuntime = ctx.runtime
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
                with torch.cuda.stream(runtime.context.reduce_scatter_stream):
                    result.sharded_grads = _accumulate_sharded_grads(
                        valid_param_refs,
                        result.sharded_grads,
                    )
                    result.event = torch.cuda.Event()
                    result.event.record(runtime.context.reduce_scatter_stream)
                runtime.context.reduce_scatter_states.append(result)
                _queue_reduce_scatter_wait(runtime.context)

        # Gradients are accumulated into the original sharded parameters above
        # so the autograd input grads can stay empty and avoid blocking here.
        return (None, *([None] * ctx.num_inputs))


def _get_or_create_comm_context(
    root_module: nn.Module,
    device: torch.device,
) -> AllGatherContext:
    contexts = getattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, None)
    if contexts is None:
        contexts = {}
        setattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, contexts)

    context = contexts.get(device)
    if context is None:
        context = AllGatherContext(
            all_gather_stream=torch.cuda.Stream(device=device, priority=-1),
            reduce_scatter_stream=torch.cuda.Stream(device=device, priority=-1),
        )
        contexts[device] = context
    return context


def _storage_requires_batched_unshard(storage: DStorage) -> bool:
    """Return whether parameter access must use hook-provided tensors."""
    return bool(storage._param_infos)


def _storage_uses_bucket_autograd_unshard(storage: DStorage) -> bool:
    """Return whether RAF should use the custom bucket autograd path."""
    if not storage._reshard_after_forward:
        return False
    if not storage._param_infos:
        return False
    return _get_bucket_autograd_unshard_unsupported_reason(storage) is None


def _get_bucket_autograd_unshard_unsupported_reason(
    storage: DStorage,
) -> str | None:
    """Return why a RAF bucket cannot use the custom autograd path."""
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


def _raise_unsupported_bucket_autograd_unshard(storage: DStorage) -> None:
    reason = _get_bucket_autograd_unshard_unsupported_reason(storage)
    bucket_fqn = _get_storage_debug_fqn(storage)
    bucket_msg = f" for bucket {bucket_fqn!r}" if bucket_fqn else ""
    raise NotImplementedError(
        "FlexShard eager reshard_after_forward currently supports only CUDA "
        f"Shard(0) buckets in the custom autograd bucket path{bucket_msg}; "
        f"{reason}. Use reshard_after_forward=False for this bucket or add "
        "support for this placement before using eager RAF."
    )


def _get_bucket_module(storage: DStorage) -> nn.Module:
    """Find the deepest common ancestor module for a bucket's params.

    For bucket "layers.0.*", returns model.layers[0].
    For bucket "norm.*, output.*" (no common prefix), returns root.
    """
    fqns = list(storage._param_infos.keys())
    prefixes = [".".join(fqn.split(".")[:-1]) for fqn in fqns]
    if not prefixes:
        return storage._module
    common = prefixes[0]
    for prefix in prefixes[1:]:
        i = 0
        while i < len(common) and i < len(prefix) and common[i] == prefix[i]:
            i += 1
        common = common[:i]
    # Trim to the last complete component so a partial name match is ignored.
    if "." in common:
        common = common[: common.rfind(".") + 1].rstrip(".")
    elif common and common not in prefixes:
        common = ""
    if not common:
        return storage._module
    mod = storage._module
    for part in common.split("."):
        mod = getattr(mod, part)
    return mod


def _get_hook_target_module(storage: DStorage) -> nn.Module | None:
    """Return the module whose forward should trigger a bucket all-gather."""
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
        return None
    return getattr(target, "_checkpoint_wrapped_module", target)


def _get_param_leaf_module(
    module: nn.Module,
    fqn: str,
) -> tuple[nn.Module, str]:
    """Return the leaf module and local parameter name for a parameter FQN."""
    parts = fqn.split(".")
    leaf_module = module
    for part in parts[:-1]:
        leaf_module = getattr(leaf_module, part)
    return leaf_module, parts[-1]


def _get_bucket_param_entries(
    storage: DStorage,
    module_param_map: dict[nn.Module, dict[str, EagerParamAccessState]],
) -> list[ParamEntry]:
    """Return params in a bucket with their owning module and access state."""
    param_entries: list[ParamEntry] = []
    for info in storage._param_infos.values():
        parts = info.fqn.split(".")
        leaf_module = storage._module
        for part in parts[:-1]:
            child = getattr(leaf_module, part, None)
            if child is None:
                wrapped = getattr(leaf_module, "_checkpoint_wrapped_module", None)
                if wrapped is not None:
                    leaf_module = getattr(wrapped, part)
                else:
                    leaf_module = getattr(leaf_module, part)
            else:
                leaf_module = child
        local_name = parts[-1]
        if hasattr(leaf_module, "_checkpoint_wrapped_module"):
            leaf_module = leaf_module._checkpoint_wrapped_module
        if leaf_module in module_param_map:
            param_state = module_param_map[leaf_module].get(local_name)
            if param_state is not None:
                param_entries.append((leaf_module, local_name, param_state, info))
    return param_entries


def _create_eager_param_states(
    module: nn.Module,
    storages: list[DStorage],
    fqn_to_bucket_spec: dict[str, BucketSpec],
    device: torch.device,
) -> dict[nn.Module, dict[str, EagerParamAccessState]]:
    """Create eager parameter access state grouped by owning leaf module."""
    module_param_map: dict[nn.Module, dict[str, EagerParamAccessState]] = {}

    for storage in storages:
        uses_bucket_autograd_unshard = _storage_uses_bucket_autograd_unshard(storage)
        bucket_fqn = _get_storage_debug_fqn(storage)
        for fqn, info in storage._param_infos.items():
            bucket_spec = fqn_to_bucket_spec[fqn]
            mp_policy = bucket_spec.mp_policy
            param_dtype = mp_policy.param_dtype if mp_policy else None
            reduce_dtype = mp_policy.reduce_dtype if mp_policy else None
            compute_device = (
                torch.device(device) if bucket_spec.offload_policy is not None else None
            )
            param_state = EagerParamAccessState(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                compute_device=compute_device,
            )

            setattr(
                param_state,
                _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR,
                uses_bucket_autograd_unshard,
            )
            setattr(param_state, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, False)
            setattr(param_state, _PARAM_FQN_ATTR, fqn)
            setattr(param_state, _BUCKET_FQN_ATTR, bucket_fqn)

            leaf_module, local_name = _get_param_leaf_module(module, fqn)
            module_param_map.setdefault(leaf_module, {})[local_name] = param_state

    return module_param_map


def _install_batched_allgather_hooks(
    storages: list,
    module_param_map: dict[nn.Module, dict[str, EagerParamAccessState]],
) -> None:
    """Install pre/post forward hooks for batched per-bucket all-gather.

    In eager mode, each DStorage's pre-forward hook runs a single batched
    Placement.unshard() call (one NCCL collective per bucket), then sets
    _pre_gathered on each parameter access state so the property getter can
    return the hook-provided tensor.

    Skipped under graph capture. FlexShard currently supports eager execution
    only, so parameter access will raise before collectives are emitted.
    """
    for storage in storages:
        infos = list(storage._param_infos.values())
        if not infos:
            continue

        ptype = type(infos[0].placements[0])
        if not _storage_requires_batched_unshard(storage):
            continue
        if storage._reshard_after_forward and not _storage_uses_bucket_autograd_unshard(
            storage
        ):
            _raise_unsupported_bucket_autograd_unshard(storage)

        param_entries = _get_bucket_param_entries(storage, module_param_map)
        logger.debug(f"Batched hooks: {len(param_entries)}/{len(infos)} params matched")
        if not param_entries:
            continue

        ag_context = None
        if ptype is Shard and storage.byte_storage.device.type == "cuda":
            device = storage.byte_storage.device
            ag_context = _get_or_create_comm_context(storage._module, device)
        ag_bucket = None
        if ag_context is not None:
            ag_bucket = AllGatherBucket(
                storage=storage,
                entries=param_entries,
                infos=infos,
                debug_fqn=_get_storage_debug_fqn(storage),
                use_autograd_unshard=_storage_uses_bucket_autograd_unshard(storage),
            )
        use_autograd_unshard = _storage_uses_bucket_autograd_unshard(storage)

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
                if _reshard_after_forward_recompute.get():
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
                all_gather_context.pending = PendingAllGather(
                    bucket=next_bucket,
                    result=_begin_bucket_unshard(next_bucket),
                    recompute=_reshard_after_forward_recompute.get(),
                )

            def _take_pending_for_current_bucket():
                pending = all_gather_context.pending
                if pending is None:
                    return None
                if (
                    pending.bucket is all_gather_bucket
                    and pending.recompute == _reshard_after_forward_recompute.get()
                ):
                    all_gather_context.pending = None
                    return pending.result
                _wait_bucket_unshard(pending.result)
                all_gather_context.pending = None
                return None

            def _reduce_fn():
                """Batched reduce-scatter using collected grads."""
                grads, valid = [], []
                for idx, (leaf, name, param_state, info) in enumerate(entries):
                    g = collected_grads.pop(idx, None)
                    if g is not None:
                        grads.append(g)
                        valid.append((leaf, name, info))
                    param_state._unsharded_for_reduce = None
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
                        with torch.cuda.stream(
                            all_gather_context.reduce_scatter_stream
                        ):
                            result.sharded_grads = _accumulate_sharded_grads(
                                [(leaf, name) for leaf, name, _ in valid],
                                result.sharded_grads,
                            )
                            result.event = torch.cuda.Event()
                            result.event.record(
                                all_gather_context.reduce_scatter_stream
                            )
                        all_gather_context.reduce_scatter_states.append(result)
                        _queue_reduce_scatter_wait(all_gather_context)
                    else:
                        reduced = pt.reduce_grad(grads, valid_infos, s._mesh)
                        _accumulate_sharded_grads(
                            [(leaf, name) for leaf, name, _ in valid],
                            reduced,
                        )

            def pre_forward_hook(mod, args):
                if torch.compiler.is_compiling():
                    return
                # Batched all-gather
                local_shards = []
                for leaf, name, param_state, _ in entries:
                    local_shard = (
                        leaf._parameters[name]
                        if use_autograd_bucket
                        else leaf._parameters[name].data
                    )
                    if (
                        param_state.compute_device is not None
                        and local_shard.device != param_state.compute_device
                    ):
                        local_shard = local_shard.to(
                            param_state.compute_device,
                            non_blocking=True,
                        )
                    local_shards.append(local_shard)
                entry_infos = [info for _, _, _, info in entries]
                if (
                    all_gather_context is not None
                    and all_gather_bucket is not None
                    and pt is Shard
                ):
                    if use_autograd_bucket:
                        prefetched_result = _take_pending_for_current_bucket()
                        runtime = BucketAllGatherRuntime(
                            prefetched_result=prefetched_result,
                            infos=entry_infos,
                            param_refs=[(leaf, name) for leaf, name, _, _ in entries],
                            mesh=s._mesh,
                            context=all_gather_context,
                            debug_fqn=all_gather_bucket.debug_fqn,
                        )
                        full_params = list(
                            _BucketAllGather.apply(runtime, *local_shards)
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
                for (_, _, param_state, _), full_param in zip(
                    entries, full_params, strict=True
                ):
                    param_state._pre_gathered = full_param

            def post_forward_hook(mod, args, output):
                if torch.compiler.is_compiling():
                    return
                for _, _, param_state, _ in entries:
                    param_state._pre_gathered = None
                if use_autograd_bucket:
                    return

                # Register AccumulateGrad hooks on detached leaf params.
                # Each hook stores its grad by index. The last hook fires
                # _reduce_fn with all collected grads (u.grad is None at
                # hook time — grads must be captured from the hook argument).
                if torch.is_grad_enabled():
                    collected_grads.clear()
                    leaf_indices = []
                    for idx, (_, _, param_state, _) in enumerate(entries):
                        u = param_state._unsharded_for_reduce
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

        target = _get_hook_target_module(storage)
        if target is None:
            continue
        target.register_forward_pre_hook(pre_hook)
        target.register_forward_hook(post_hook)
        if ag_context is not None and ag_bucket is not None:
            ag_context.buckets.append(ag_bucket)
        for _, _, param_state, _ in param_entries:
            setattr(param_state, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, True)
