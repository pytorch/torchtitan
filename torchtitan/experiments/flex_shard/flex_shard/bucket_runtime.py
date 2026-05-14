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
    launch_reduce_scatter_grad,
    prepare_reduce_scatter_grad,
    PreparedReduceScatterGrad,
    ReduceScatterGradHandle,
)
from .bucket_storage import BucketSpec, DStorage, ParamInfo
from .param_access import (
    _BUCKET_FQN_ATTR,
    _EAGER_BATCHED_HOOK_REGISTERED_ATTR,
    _EAGER_COMM_CONTEXTS_ATTR,
    _PARAM_FQN_ATTR,
    EagerParamAccessState,
    ParamModuleInfo,
)
from .utils import _get_storage_debug_fqn, _record_function_if_eager


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParamEntry:
    """Per-parameter bucket runtime state.

    module_info locates the live nn.Parameter for local shard reads and grad
    writes. access_state holds mutable forward/backward parameter access state.
    param_info carries immutable storage and placement metadata for collectives.
    Keeping them together preserves bucket order and avoids repeated FQN
    resolution in hooks.
    """

    module_info: ParamModuleInfo
    access_state: EagerParamAccessState
    param_info: ParamInfo


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
class PendingReduceScatterLaunch:
    """One packed reduce-grad request waiting for a backward all-gather first."""

    bucket: BucketRuntime
    prepared: PreparedReduceScatterGrad
    param_refs: list[ParamModuleInfo]


@dataclass
class BucketCommContext:
    """Communication streams shared by buckets on one root module/device."""

    device_handle: ModuleType
    all_gather_stream: torch.Stream
    reduce_scatter_stream: torch.Stream
    buckets: list[BucketRuntime] = field(default_factory=list)
    pending: PendingAllGather | None = None
    pending_reduce_scatter_launches: list[PendingReduceScatterLaunch] = field(
        default_factory=list
    )
    reduce_scatter_states: list[PendingReduceGrad] = field(default_factory=list)
    reduce_scatter_callback_queued: bool = False

    def next_backward_all_gather_bucket(
        self,
        bucket: BucketRuntime,
    ) -> BucketRuntime | None:
        """Return the next bucket whose backward all-gather has priority.

        Buckets execute forward in ``self.buckets`` order and backward in reverse
        order. After bucket ``i`` produces gradients, bucket ``i - 1``'s
        all-gather is the next critical-path backward communication when that
        bucket uses reshard-after-forward. In that case bucket ``i``'s
        reduce-scatter should be delayed until after the previous bucket's
        all-gather has launched.
        """
        idx = next(
            (i for i, candidate in enumerate(self.buckets) if candidate is bucket),
            None,
        )
        if idx is None:
            return None
        if idx == 0:
            return None
        next_bucket = self.buckets[idx - 1]
        if not next_bucket.storage._reshard_after_forward:
            return None
        return next_bucket

    def should_defer_reduce_scatter_for_backward_prefetch(
        self,
        bucket: BucketRuntime,
    ) -> bool:
        """Return whether reduce-scatter should wait for backward prefetch."""
        return self.next_backward_all_gather_bucket(bucket) is not None

    @classmethod
    def get(
        cls,
        root_module: nn.Module,
        device: torch.device,
    ) -> BucketCommContext | None:
        contexts = getattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, None)
        if contexts is None:
            return None
        return contexts.get(device)

    @classmethod
    def create(
        cls,
        root_module: nn.Module,
        device: torch.device,
    ) -> BucketCommContext:
        contexts = getattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, None)
        if contexts is None:
            contexts = {}
            setattr(root_module, _EAGER_COMM_CONTEXTS_ATTR, contexts)
        if device in contexts:
            raise AssertionError(
                f"Communication context for device {device} already exists."
            )

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
                self.flush_pending_reduce_scatter_launches(max_to_flush=None)
                for pending in self.reduce_scatter_states:
                    pending.result.wait()
                    pending.result.release_buffers(
                        release_sharded_grads=True,
                    )
            finally:
                self.reduce_scatter_states.clear()
                self.pending_reduce_scatter_launches.clear()
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
        with _record_function_if_eager("FlexShard::post_backward_rs_wait", debug_fqn):
            for pending in self.reduce_scatter_states:
                pending.result.wait()
                pending.result.release_buffers(
                    release_sharded_grads=True,
                )
            self.reduce_scatter_states.clear()

    def _launch_pending_reduce_scatter(
        self,
        pending: PendingReduceScatterLaunch,
    ) -> None:
        bucket = pending.bucket
        self.wait_and_clear_reduce_scatter_states(bucket.debug_fqn)
        result = launch_reduce_scatter_grad(
            pending.prepared,
            self.reduce_scatter_stream,
        )
        with self.device_handle.stream(self.reduce_scatter_stream):
            sharded_grads = result.finish()
            result.record_sharded_grads(
                _accumulate_sharded_grads(
                    pending.param_refs,
                    sharded_grads,
                ),
                self.reduce_scatter_stream,
            )
        self.reduce_scatter_states.append(PendingReduceGrad(result))

    def queue_reduce_scatter_launch(
        self,
        bucket: BucketRuntime,
        grads: list[torch.Tensor],
        infos: list[ParamInfo],
        param_refs: list[ParamModuleInfo],
    ) -> None:
        """Pack reduce-scatter input and delay NCCL launch until after next all-gather."""
        if not grads:
            return
        with torch.no_grad():
            prepared = prepare_reduce_scatter_grad(
                grads,
                infos,
                bucket.storage._mesh,
                debug_fqn=bucket.debug_fqn,
            )
        self.pending_reduce_scatter_launches.append(
            PendingReduceScatterLaunch(
                bucket=bucket,
                prepared=prepared,
                param_refs=param_refs,
            )
        )
        self.queue_reduce_scatter_wait()

    def flush_pending_reduce_scatter_launches(
        self,
        max_to_flush: int | None,
    ) -> None:
        """Launch deferred reduce-scatters after all-gather prefetch has priority."""
        num_flushed = 0
        while self.pending_reduce_scatter_launches and (
            max_to_flush is None or num_flushed < max_to_flush
        ):
            pending = self.pending_reduce_scatter_launches.pop(0)
            with torch.no_grad():
                self._launch_pending_reduce_scatter(pending)
            num_flushed += 1
        if num_flushed:
            self.queue_reduce_scatter_wait()


@dataclass
class BucketAllGatherRuntime:
    """Runtime metadata passed to bucket all-gather autograd."""

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
            comm_device = cls._comm_device(storage, entries)
            context = BucketCommContext.get(storage._module, comm_device)
            if context is None:
                context = BucketCommContext.create(storage._module, comm_device)
        return cls(
            storage=storage,
            entries=entries,
            context=context,
            debug_fqn=_get_storage_debug_fqn(storage),
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
                    param_entries.append(
                        ParamEntry(
                            module_info=module_info,
                            access_state=param_state,
                            param_info=info,
                        )
                    )
        return param_entries

    @staticmethod
    def _comm_device(
        storage: DStorage,
        entries: list[ParamEntry],
    ) -> torch.device:
        """Return the device used for this bucket's collectives."""
        comm_device = storage.byte_storage.device
        for entry in entries:
            if entry.access_state.compute_device is not None:
                return entry.access_state.compute_device
        return comm_device

    @property
    def infos(self) -> list[ParamInfo]:
        return [entry.param_info for entry in self.entries]

    @property
    def param_refs(self) -> list[ParamModuleInfo]:
        return [entry.module_info for entry in self.entries]

    def _local_shards(self, *, use_autograd: bool) -> list[torch.Tensor]:
        local_shards: list[torch.Tensor] = []
        for entry in self.entries:
            module_info = entry.module_info
            param_state = entry.access_state
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

    def resolve_bucket_forward_hook_module(self) -> nn.Module | None:
        """Return the module whose forward should trigger this bucket."""
        # Register hooks on the deepest common ancestor module for the bucket's
        # params so one pre-forward all-gather covers their parameter accesses.
        # For example, a bucket with "layers.0.attn.weight" and
        # "layers.0.mlp.weight" hooks "layers.0".
        target = self.bucket_module()
        # TODO: Avoid registering bucket hooks on passive containers such as
        # ModuleList or ModuleDict. Catch-all buckets can resolve to those
        # containers, whose hooks may never run when forward iterates children.
        # Reshard-after-forward needs the bucket hook to rerun during activation
        # checkpoint recompute. For example, a bucket like
        # ["embed.*", "layers.1.*", "output.*"] resolves to the root module. If
        # "layers.1" is checkpoint-wrapped, backward recompute calls only the
        # layer, not the root module's __call__, so a root hook would not
        # re-all-gather the layer params after post-forward resharding.
        # Grouped root/rest buckets need replay fragments that share one
        # physical bucket all-gather/reduce-scatter, so reject them until that
        # support exists instead of silently installing an unreplayable hook.
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
        recompute_state = self.storage._reshard_after_forward_recompute_state
        return (
            self.storage._reshard_after_forward
            and recompute_state is not None
            and recompute_state.is_recomputing(id(self.storage))
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

    def pre_forward_hook(self, mod, args) -> None:
        local_shards = self._local_shards(use_autograd=True)
        is_compiling = torch.compiler.is_compiling()
        prefetched_result = None if is_compiling else self.take_pending()
        runtime = BucketAllGatherRuntime(
            bucket=self,
            prefetched_result=prefetched_result,
        )
        full_params = list(_BucketAllGather.apply(runtime, *local_shards))
        if not is_compiling:
            self.prefetch_next()
            self.context.flush_pending_reduce_scatter_launches(max_to_flush=1)
        for entry, full_param in zip(self.entries, full_params, strict=True):
            entry.access_state._pre_gathered = full_param

    def post_forward_hook(self, mod, args, output) -> None:
        for entry in self.entries:
            entry.access_state._pre_gathered = None


class _BucketAllGather(torch.autograd.Function):
    """Autograd boundary for bucket all-gather.

    Forward consumes a raw all-gather result, either prefetched by the previous
    bucket or launched on demand. Backward packs full-parameter gradients and
    either launches or defers one explicit bucket reduce-scatter. The reduced
    sharded grads are accumulated into the original sharded parameters
    asynchronously instead of returned through autograd, so later backward
    compute can overlap with the reduce-scatter.
    """

    @staticmethod
    def forward(
        ctx: Any,
        runtime: BucketAllGatherRuntime,
        *local_shards: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.runtime = runtime
        ctx.num_inputs = len(local_shards)
        ctx.local_shard_dtypes = tuple(shard.dtype for shard in local_shards)

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
        frozen_params = [
            full_param
            for full_param, info in zip(full_params, runtime.bucket.infos, strict=True)
            if not info.requires_grad
        ]
        if frozen_params:
            ctx.mark_non_differentiable(*frozen_params)
        return tuple(full_params)

    @staticmethod
    def backward(
        ctx: Any,
        *full_param_grads: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        runtime: BucketAllGatherRuntime = ctx.runtime
        bucket = runtime.bucket
        input_grads: list[torch.Tensor | None] = [None] * ctx.num_inputs
        grads: list[torch.Tensor] = []
        valid_infos: list[ParamInfo] = []
        valid_param_refs: list[ParamModuleInfo] = []
        valid_indices: list[int] = []
        for idx, (grad, entry) in enumerate(
            zip(
                full_param_grads,
                bucket.entries,
                strict=True,
            )
        ):
            if grad is None:
                continue
            valid_indices.append(idx)
            grads.append(grad.contiguous())
            valid_infos.append(entry.param_info)
            valid_param_refs.append(entry.module_info)

        if grads and torch.compiler.is_compiling():
            result = begin_reduce_scatter_grad(
                grads,
                valid_infos,
                bucket.storage._mesh,
                bucket.context.reduce_scatter_stream,
                debug_fqn=bucket.debug_fqn,
            )
            sharded_grads = result.finish()
            for input_idx, sharded_grad in zip(
                valid_indices,
                sharded_grads,
                strict=True,
            ):
                input_dtype = ctx.local_shard_dtypes[input_idx]
                if sharded_grad.dtype != input_dtype:
                    sharded_grad = sharded_grad.to(input_dtype)
                input_grads[input_idx] = sharded_grad
            return (None, *input_grads)

        if grads:
            if bucket.context.should_defer_reduce_scatter_for_backward_prefetch(bucket):
                bucket.context.queue_reduce_scatter_launch(
                    bucket,
                    grads,
                    valid_infos,
                    valid_param_refs,
                )
            else:
                bucket.reduce_grads(grads, valid_infos, valid_param_refs)

        return (None, *([None] * ctx.num_inputs))


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

    CUDA buckets use the same custom autograd bucket path in eager and compile
    so Dynamo traces the same bucket pre-hook and parameter access logic.
    """
    for storage in storages:
        if not storage._param_infos:
            raise AssertionError("Expected FlexShard bucket storage to own parameters.")
        if storage.byte_storage.device.type != "cuda":
            raise AssertionError("Expected FlexShard bucket storage to be on CUDA.")

        bucket_runtime = BucketRuntime.from_storage(
            storage=storage,
            module_param_map=module_param_map,
        )
        if bucket_runtime is None:
            continue

        target = bucket_runtime.resolve_bucket_forward_hook_module()
        if target is None:
            _raise_unreplayable_reshard_hook(storage)
        target.register_forward_pre_hook(bucket_runtime.pre_forward_hook)
        target.register_forward_hook(bucket_runtime.post_forward_hook)
        bucket_runtime.context.buckets.append(bucket_runtime)
        for entry in bucket_runtime.entries:
            setattr(entry.access_state, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, True)
