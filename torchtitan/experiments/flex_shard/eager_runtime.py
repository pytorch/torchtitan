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

from .metadata import (
    _EAGER_BATCHED_HOOK_REGISTERED_ATTR,
    _EAGER_COMM_CONTEXTS_ATTR,
)
from .placements import (
    EagerAllGatherResult,
    EagerReduceScatterResult,
    Shard,
)
from .reshard import _get_storage_debug_fqn, _reshard_checkpoint_recompute
from .storage import DStorage, ParamInfo
from .utils import _with_fqn

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


logger = logging.getLogger(__name__)


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
