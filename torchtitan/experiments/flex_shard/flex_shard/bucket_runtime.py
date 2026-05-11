# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle

from .bucket_collectives import (
    AllGatherUnshardHandle,
    begin_all_gather_unshard,
    begin_reduce_scatter_grad,
    ReduceScatterGradHandle,
)
from .bucket_storage import BucketSpec, DStorage, ParamInfo
from .param_access import (
    _BUCKET_FQN_ATTR,
    _EAGER_AUTOGRAD_BUCKET_UNSHARD_ATTR,
    _EAGER_BATCHED_HOOK_REGISTERED_ATTR,
    _EAGER_COMM_CONTEXTS_ATTR,
    _PARAM_FQN_ATTR,
    EagerParamAccessState,
    ParamModuleInfo,
)
from .reshard_after_forward import _is_reshard_after_forward_recompute
from .utils import _get_storage_debug_fqn, _with_fqn


logger = logging.getLogger(__name__)

ParamEntry = tuple[ParamModuleInfo, EagerParamAccessState, ParamInfo]


@dataclass
class PendingAllGather:
    """The single one-bucket-ahead all-gather in flight."""

    bucket: BucketRuntime
    result: AllGatherUnshardHandle
    recompute: bool


@dataclass
class PendingReduceGrad:
    """One in-flight reduce-grad result."""

    result: ReduceScatterGradHandle


@dataclass
class BucketCommContext:
    """Communication streams for batched collectives."""

    device_handle: ModuleType
    all_gather_stream: torch.Stream
    reduce_scatter_stream: torch.Stream
    buckets: list[BucketRuntime] = field(default_factory=list)
    pending: PendingAllGather | None = None
    reduce_scatter_states: list[PendingReduceGrad] = field(default_factory=list)
    reduce_scatter_callback_queued: bool = False

    @classmethod
    def get_or_create(
        cls,
        root_module: nn.Module,
        device: torch.device,
    ) -> BucketCommContext:
        contexts = getattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, None)
        if contexts is None:
            contexts = {}
            setattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, contexts)

        context = contexts.get(device)
        if context is None:
            device_handle = _get_device_handle(device.type)
            context = cls(
                device_handle=device_handle,
                all_gather_stream=device_handle.Stream(priority=-1),
                reduce_scatter_stream=device_handle.Stream(priority=-1),
            )
            contexts[device] = context
        return context

    def queue_reduce_scatter_wait(self) -> None:
        """Queue a post-backward wait for reduce-scatter work."""
        if self.reduce_scatter_callback_queued:
            return
        self.reduce_scatter_callback_queued = True

        def _wait_for_reduce_scatter() -> None:
            try:
                for pending in self.reduce_scatter_states:
                    pending.result.wait()
                    pending.result.release_buffers(
                        release_sharded_grads=True,
                    )
            finally:
                self.reduce_scatter_states.clear()
                self.reduce_scatter_callback_queued = False

        torch.autograd.Variable._execution_engine.queue_callback(
            _wait_for_reduce_scatter
        )

    def wait_and_clear_reduce_scatter_states(
        self,
        debug_fqn: str | None,
    ) -> None:
        """Wait for prior reduce-scatter states and release their buffers."""
        if not self.reduce_scatter_states:
            return
        with torch.profiler.record_function(
            _with_fqn("FlexShard::post_backward_rs_wait", debug_fqn)
        ):
            for pending in self.reduce_scatter_states:
                pending.result.wait()
                pending.result.release_buffers(
                    release_sharded_grads=True,
                )
            self.reduce_scatter_states.clear()


@dataclass
class BucketAllGatherRuntime:
    """Runtime metadata passed to reshard-after-forward bucket autograd."""

    bucket: BucketRuntime
    prefetched_result: AllGatherUnshardHandle | None


def _accumulate_sharded_grads(
    param_refs: list[ParamModuleInfo],
    sharded_grads: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Cast sharded grads to local param dtype and accumulate into .grad."""
    stored_grads: list[torch.Tensor] = []
    for module_info, grad in zip(param_refs, sharded_grads, strict=True):
        param = module_info.module._parameters[module_info.param_name]
        if grad.dtype != param.dtype:
            grad = grad.to(param.dtype)
        stored_grads.append(grad)
        if param.grad is None:
            param.grad = grad
        else:
            param.grad += grad
    return stored_grads


@dataclass
class BucketRuntime:
    """Runtime state and hooks for one FlexShard bucket."""

    storage: DStorage
    entries: list[ParamEntry]
    context: BucketCommContext
    debug_fqn: str | None
    use_autograd_unshard: bool
    collected_grads: dict[int, torch.Tensor] = field(default_factory=dict)

    @classmethod
    def from_storage(
        cls,
        storage: DStorage,
        module_param_map: dict[nn.Module, dict[str, EagerParamAccessState]],
        context: BucketCommContext | None = None,
    ) -> BucketRuntime | None:
        """Create runtime state for one storage bucket."""
        entries = cls._get_param_entries(storage, module_param_map)
        logger.debug(
            f"Batched hooks: {len(entries)}/{len(storage._param_infos)} params matched"
        )
        if not entries:
            return None
        if context is None:
            context = BucketCommContext.get_or_create(
                storage._module,
                cls._comm_device(storage, entries),
            )
        return cls(
            storage=storage,
            entries=entries,
            context=context,
            debug_fqn=_get_storage_debug_fqn(storage),
            use_autograd_unshard=_storage_uses_bucket_autograd_unshard(storage),
        )

    @staticmethod
    def _get_param_entries(
        storage: DStorage,
        module_param_map: dict[nn.Module, dict[str, EagerParamAccessState]],
    ) -> list[ParamEntry]:
        """Return params in a bucket with their owning module and access state."""
        param_entries: list[ParamEntry] = []
        for info in storage._param_infos.values():
            module_info = ParamModuleInfo.resolve(storage._module, info.fqn)
            if module_info.module in module_param_map:
                param_state = module_param_map[module_info.module].get(
                    module_info.param_name
                )
                if param_state is not None:
                    param_entries.append((module_info, param_state, info))
        return param_entries

    @staticmethod
    def _comm_device(
        storage: DStorage,
        entries: list[ParamEntry],
    ) -> torch.device:
        """Return the device used for this bucket's collectives."""
        comm_device = storage.byte_storage.device
        for _, param_state, _ in entries:
            if param_state.compute_device is not None:
                return param_state.compute_device
        return comm_device

    @property
    def infos(self) -> list[ParamInfo]:
        return [info for _, _, info in self.entries]

    @property
    def param_refs(self) -> list[ParamModuleInfo]:
        return [module_info for module_info, _, _ in self.entries]

    def _local_shards(self, use_autograd: bool) -> list[torch.Tensor]:
        local_shards: list[torch.Tensor] = []
        for module_info, param_state, _ in self.entries:
            param = module_info.module._parameters[module_info.param_name]
            local_shard = param if use_autograd else param.data
            if (
                param_state.compute_device is not None
                and local_shard.device != param_state.compute_device
            ):
                local_shard = local_shard.to(
                    param_state.compute_device,
                    non_blocking=True,
                )
            local_shards.append(local_shard)
        return local_shards

    def bucket_module(self) -> nn.Module:
        """Find the deepest common ancestor module for this bucket's params."""
        fqns = list(self.storage._param_infos.keys())
        prefixes = [".".join(fqn.split(".")[:-1]) for fqn in fqns]
        if not prefixes:
            return self.storage._module
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
            return self.storage._module
        mod = self.storage._module
        for part in common.split("."):
            mod = getattr(mod, part)
        return mod

    def hook_target_module(self) -> nn.Module | None:
        """Return the module whose forward should trigger this bucket."""
        target = self.bucket_module()
        # TODO: Avoid registering bucket hooks on passive containers such as
        # ModuleList or ModuleDict. Catch-all buckets can resolve to those
        # containers, whose hooks may never run when forward iterates children.
        if (
            self.storage._reshard_after_forward
            and target is self.storage._module
            and any(
                hasattr(child, "_checkpoint_wrapped_module")
                for child in self.storage._module.modules()
                if child is not self.storage._module
            )
        ):
            logger.debug(
                "Skipping root-level batched all-gather hook because child "
                "checkpoint recomputation would not replay the root hook.",
            )
            return None
        return getattr(target, "_checkpoint_wrapped_module", target)

    def begin_unshard(self) -> AllGatherUnshardHandle:
        """Begin this bucket's all-gather unshard on the shared stream."""
        return begin_all_gather_unshard(
            self._local_shards(use_autograd=False),
            self.infos,
            self.storage._mesh,
            self.context.all_gather_stream,
            debug_fqn=self.debug_fqn,
        )

    def wait_pending(self, result: AllGatherUnshardHandle) -> None:
        """Wait for and release an unused pending unshard result."""
        result.wait()
        result.release_buffers()

    def in_reshard_after_forward_recompute(self) -> bool:
        """Return whether this bucket is in reshard-after-forward recompute."""
        return (
            self.storage._reshard_after_forward
            and _is_reshard_after_forward_recompute(id(self.storage))
        )

    def prefetch_next(self) -> None:
        """Start the next bucket's all-gather if no prefetch is in flight."""
        if self.context.pending is not None:
            return
        prefetch_order = self.context.buckets
        is_recompute = self.in_reshard_after_forward_recompute()
        if is_recompute:
            prefetch_order = prefetch_order[::-1]
        for idx, bucket in enumerate(prefetch_order):
            if bucket is self:
                break
        else:
            return
        next_idx = idx + 1
        if next_idx >= len(prefetch_order):
            return
        next_bucket = prefetch_order[next_idx]
        self.context.pending = PendingAllGather(
            bucket=next_bucket,
            result=next_bucket.begin_unshard(),
            recompute=is_recompute,
        )

    def take_pending(self) -> AllGatherUnshardHandle | None:
        """Return a matching pending result, releasing stale pending work."""
        pending = self.context.pending
        if pending is None:
            return None
        if (
            pending.bucket is self
            and pending.recompute == self.in_reshard_after_forward_recompute()
        ):
            self.context.pending = None
            return pending.result
        self.wait_pending(pending.result)
        self.context.pending = None
        return None

    def reduce_grads(
        self,
        grads: list[torch.Tensor],
        infos: list[ParamInfo],
        param_refs: list[ParamModuleInfo],
    ) -> None:
        """Reduce full-parameter grads and accumulate local sharded grads."""
        if not grads:
            return
        with torch.no_grad():
            self.context.wait_and_clear_reduce_scatter_states(self.debug_fqn)
            # TODO: Thread the bucket's reduce_dtype into the non-reshard-after-forward
            # path before launching reduce-scatter. Reshard-after-forward uses
            # _MixedPrecisionCast backward, but detached non-reshard-after-forward
            # leaves can arrive in param_dtype.
            result = begin_reduce_scatter_grad(
                grads,
                infos,
                self.storage._mesh,
                self.context.reduce_scatter_stream,
                debug_fqn=self.debug_fqn,
            )
            with self.context.device_handle.stream(self.context.reduce_scatter_stream):
                sharded_grads = result.finish()
                result.record_sharded_grads(
                    _accumulate_sharded_grads(
                        param_refs,
                        sharded_grads,
                    ),
                    self.context.reduce_scatter_stream,
                )
            self.context.reduce_scatter_states.append(PendingReduceGrad(result))
            self.context.queue_reduce_scatter_wait()

    def _reduce_collected_grads(self) -> None:
        grads: list[torch.Tensor] = []
        infos: list[ParamInfo] = []
        param_refs: list[ParamModuleInfo] = []
        for idx, (module_info, param_state, info) in enumerate(self.entries):
            grad = self.collected_grads.pop(idx, None)
            if grad is not None:
                grads.append(grad)
                infos.append(info)
                param_refs.append(module_info)
            param_state._unsharded_for_reduce = None
        self.reduce_grads(grads, infos, param_refs)

    def pre_forward_hook(self, mod, args) -> None:
        if torch.compiler.is_compiling():
            return
        local_shards = self._local_shards(use_autograd=self.use_autograd_unshard)
        if self.use_autograd_unshard:
            prefetched_result = self.take_pending()
            runtime = BucketAllGatherRuntime(
                bucket=self,
                prefetched_result=prefetched_result,
            )
            full_params = list(_BucketAllGather.apply(runtime, *local_shards))
            self.prefetch_next()
        else:
            with torch.no_grad():
                result = self.take_pending()
                if result is None:
                    result = begin_all_gather_unshard(
                        local_shards,
                        self.infos,
                        self.storage._mesh,
                        self.context.all_gather_stream,
                        debug_fqn=self.debug_fqn,
                    )
                full_params = result.finish()
                self.prefetch_next()
        for (_, param_state, _), full_param in zip(
            self.entries, full_params, strict=True
        ):
            param_state._pre_gathered = full_param

    def post_forward_hook(self, mod, args, output) -> None:
        if torch.compiler.is_compiling():
            return
        for _, param_state, _ in self.entries:
            param_state._pre_gathered = None
        if self.use_autograd_unshard:
            return

        if torch.is_grad_enabled():
            self.collected_grads.clear()
            leaf_indices = []
            for idx, (_, param_state, _) in enumerate(self.entries):
                leaf = param_state._unsharded_for_reduce
                if leaf is not None and leaf.requires_grad:
                    leaf_indices.append((idx, leaf))

            if leaf_indices:
                indices = [idx for idx, _ in leaf_indices]
                leaves = [leaf for _, leaf in leaf_indices]

                def _on_grads(grads):
                    self.collected_grads.clear()
                    for idx, grad in zip(indices, grads, strict=True):
                        if grad is not None:
                            self.collected_grads[idx] = grad
                    self._reduce_collected_grads()

                torch.autograd.graph.register_multi_grad_hook(
                    leaves,
                    _on_grads,
                )


class _BucketAllGather(torch.autograd.Function):
    """Autograd boundary for reshard-after-forward bucket all-gather.

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
            result = begin_all_gather_unshard(
                [shard.detach() for shard in local_shards],
                runtime.bucket.infos,
                runtime.bucket.storage._mesh,
                runtime.bucket.context.all_gather_stream,
                debug_fqn=runtime.bucket.debug_fqn,
            )
        full_params = result.finish()
        return tuple(full_params)

    @staticmethod
    def backward(
        ctx: Any,
        *full_param_grads: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        runtime: BucketAllGatherRuntime = ctx.runtime
        bucket = runtime.bucket
        grads: list[torch.Tensor] = []
        valid_infos: list[ParamInfo] = []
        valid_param_refs: list[ParamModuleInfo] = []
        for grad, info, param_ref in zip(
            full_param_grads,
            bucket.infos,
            bucket.param_refs,
            strict=True,
        ):
            if grad is None:
                continue
            grads.append(grad.contiguous())
            valid_infos.append(info)
            valid_param_refs.append(param_ref)

        if grads:
            bucket.reduce_grads(grads, valid_infos, valid_param_refs)

        # Gradients are accumulated into the original sharded parameters above
        # so the autograd input grads can stay empty and avoid blocking here.
        return (None, *([None] * ctx.num_inputs))


def _storage_requires_batched_unshard(storage: DStorage) -> bool:
    """Return whether parameter access must use hook-provided tensors."""
    return bool(storage._param_infos)


def _storage_uses_bucket_autograd_unshard(storage: DStorage) -> bool:
    """Return whether reshard-after-forward should use the custom bucket autograd path."""
    if not storage._reshard_after_forward:
        return False
    if not storage._param_infos:
        return False
    return _get_bucket_autograd_unshard_unsupported_reason(storage) is None


def _get_bucket_autograd_unshard_unsupported_reason(
    storage: DStorage,
) -> str | None:
    """Return why a reshard-after-forward bucket cannot use the custom autograd path."""
    infos = list(storage._param_infos.values())
    if not infos:
        return None
    if storage.byte_storage.device.type != "cuda":
        return f"storage is on {storage.byte_storage.device.type}"
    return None


def _raise_unsupported_bucket_autograd_unshard(storage: DStorage) -> None:
    reason = _get_bucket_autograd_unshard_unsupported_reason(storage)
    bucket_fqn = _get_storage_debug_fqn(storage)
    bucket_msg = f" for bucket {bucket_fqn!r}" if bucket_fqn else ""
    raise NotImplementedError(
        "FlexShard eager reshard_after_forward requires placement support "
        f"for the custom autograd bucket path{bucket_msg}; "
        f"{reason}. Use reshard_after_forward=False for this bucket or add "
        "support for this placement before using eager reshard-after-forward."
    )


def _raise_unreplayable_reshard_hook(storage: DStorage) -> None:
    bucket_fqn = _get_storage_debug_fqn(storage)
    bucket_msg = f" for bucket {bucket_fqn!r}" if bucket_fqn else ""
    raise RuntimeError(
        "FlexShard eager reshard_after_forward could not register a "
        f"recomputation-safe batched all-gather hook{bucket_msg}. "
        "Bucket hooks must run both in the original forward and in activation "
        "checkpoint recomputation. Split the bucket to match forward execution "
        "units, or set reshard_after_forward=False for this bucket."
    )


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
                requires_grad=info.requires_grad,
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

            module_info = ParamModuleInfo.resolve(module, fqn)
            module_param_map.setdefault(module_info.module, {})[
                module_info.param_name
            ] = param_state

    return module_param_map


def _install_batched_allgather_hooks(
    storages: list,
    module_param_map: dict[nn.Module, dict[str, EagerParamAccessState]],
) -> None:
    """Install pre/post forward hooks for batched per-bucket all-gather.

    In eager mode, each DStorage's pre-forward hook starts a single batched
    all-gather unshard call (one collective per bucket), then sets
    _pre_gathered on each parameter access state so the property getter can
    return the hook-provided tensor.

    Skipped under graph capture. FlexShard currently supports eager execution
    only, so parameter access will raise before collectives are emitted.
    """
    for storage in storages:
        if not _storage_requires_batched_unshard(storage):
            continue
        if (
            storage._reshard_after_forward
            and not _storage_uses_bucket_autograd_unshard(storage)
        ):
            _raise_unsupported_bucket_autograd_unshard(storage)

        bucket_runtime = BucketRuntime.from_storage(
            storage=storage,
            module_param_map=module_param_map,
        )
        if bucket_runtime is None:
            continue

        target = bucket_runtime.hook_target_module()
        if target is None:
            _raise_unreplayable_reshard_hook(storage)
        target.register_forward_pre_hook(bucket_runtime.pre_forward_hook)
        target.register_forward_hook(bucket_runtime.post_forward_hook)
        bucket_runtime.context.buckets.append(bucket_runtime)
        for _, param_state, _ in bucket_runtime.entries:
            setattr(param_state, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, True)
