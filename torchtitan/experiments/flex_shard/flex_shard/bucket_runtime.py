# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import _get_device_handle

from .bucket_comm import (
    begin_bucket_unshard,
    begin_reduce_grad,
    launch_reduce_grad,
    prepare_reduce_grad,
    PreparedReduceGrad,
    ReduceGradHandle,
    UnshardHandle,
)
from .bucket_storage import BucketSpec, ParamInfo, ShardedBucketStorage
from .unsharded_param_getters import UnshardedParamSlot
from .utils import (
    _get_bucket_storage_debug_fqn,
    _record_function_if_eager,
    _strip_checkpoint_wrapped_module_path,
    _suppress_eager_profiling,
    _top_level_owner_path,
)


logger = logging.getLogger(__name__)


_EAGER_COMM_CONTEXTS_ATTR = "_flex_shard_eager_comm_contexts"


def _max_pending_reduce_grads_from_env() -> int:
    raw = os.environ.get("FLEX_SHARD_MAX_PENDING_REDUCE_GRADS")
    if raw is None:
        return 1
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "Ignoring invalid FLEX_SHARD_MAX_PENDING_REDUCE_GRADS=%r; using 1.",
            raw,
        )
        return 1


@dataclass
class ParamOwnerRef:
    """Owning module and local parameter name for a managed parameter."""

    module: nn.Module
    param_name: str

    @classmethod
    def resolve(cls, root_module: nn.Module, fqn: str) -> ParamOwnerRef:
        """Resolve a parameter FQN from root, unwrapping checkpoint wrappers."""
        parts = fqn.split(".")
        leaf_module = root_module
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
        if hasattr(leaf_module, "_checkpoint_wrapped_module"):
            leaf_module = leaf_module._checkpoint_wrapped_module
        return cls(leaf_module, parts[-1])


@dataclass(frozen=True)
class BucketParam:
    """Per-parameter bucket runtime state.

    param_owner locates the live nn.Parameter for local shard reads and grad
    writes. unsharded_param_slot is the hook-to-getter handoff for the current
    unsharded parameter.
    param_info carries immutable bucket storage and placement metadata for collectives.
    Keeping them together preserves bucket order and avoids repeated FQN
    resolution in hooks.
    """

    param_owner: ParamOwnerRef
    unsharded_param_slot: UnshardedParamSlot
    param_info: ParamInfo


@dataclass
class PendingUnshard:
    """One in-flight one-bucket-ahead unshard."""

    bucket: BucketRuntime
    result: UnshardHandle
    recompute: bool


@dataclass(frozen=True)
class PendingUnshardKey:
    """Key separating forward and RAF recompute prefetches for one bucket."""

    bucket_id: int
    recompute: bool


@dataclass
class PendingReduceGrad:
    """One in-flight reduce-grad result."""

    result: ReduceGradHandle


@dataclass
class PendingReduceGradLaunch:
    """One packed reduce-grad request waiting for a backward unshard first."""

    bucket: BucketRuntime
    prepared: PreparedReduceGrad
    param_owners: list[ParamOwnerRef]


@dataclass
class BucketCommContext:
    """Communication streams shared by buckets on one root module/device."""

    device_handle: ModuleType
    unshard_stream: torch.Stream
    reduce_grad_stream: torch.Stream
    reduce_grad_release_stream: torch.Stream
    buckets: list[BucketRuntime] = field(default_factory=list)
    pending_unshards: dict[PendingUnshardKey, PendingUnshard] = field(
        default_factory=dict
    )
    pending_reduce_grad_launches: list[PendingReduceGradLaunch] = field(
        default_factory=list
    )
    reduce_grad_states: list[PendingReduceGrad] = field(default_factory=list)
    max_pending_reduce_grads: int = field(
        default_factory=_max_pending_reduce_grads_from_env
    )
    reduce_grad_callback_queued: bool = False
    raf_saved_unshard_cache: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    raf_saved_unshard_cache_callback_queued: bool = False
    _forward_bucket_indices: dict[int, int] | None = None
    _recompute_prefetch_buckets: list[BucketRuntime] | None = None
    _recompute_prefetch_bucket_indices: dict[int, int] | None = None

    def add_bucket(self, bucket: BucketRuntime) -> None:
        """Append a bucket and invalidate cached scheduling metadata."""
        self.buckets.append(bucket)
        self._forward_bucket_indices = None
        self._recompute_prefetch_buckets = None
        self._recompute_prefetch_bucket_indices = None

    def next_backward_unshard_bucket(
        self,
        bucket: BucketRuntime,
    ) -> BucketRuntime | None:
        """Return the next bucket whose backward unshard has priority.

        Reshard-after-forward backward recomputes top-level execution units in
        reverse order, while replaying bucket hooks inside each unit in forward
        order. Non-reshard buckets are not recomputed, but their backward may
        still be the point where the first recompute bucket should get priority.
        """
        recompute_order = self.recompute_prefetch_buckets()
        if not recompute_order:
            return None

        recompute_idx = self.recompute_prefetch_bucket_index(bucket)
        if recompute_idx is not None:
            next_idx = recompute_idx + 1
            if next_idx >= len(recompute_order):
                return None
            return recompute_order[next_idx]

        bucket_forward_idx = self.forward_bucket_index(bucket)
        if bucket_forward_idx is None:
            return None
        for candidate in recompute_order:
            candidate_forward_idx = self.forward_bucket_index(candidate)
            if candidate_forward_idx is not None and (
                candidate_forward_idx < bucket_forward_idx
            ):
                return candidate
        return None

    def forward_bucket_index(self, bucket: BucketRuntime) -> int | None:
        """Return bucket's index in forward execution order."""
        if self._forward_bucket_indices is None:
            self._forward_bucket_indices = {
                id(candidate): idx for idx, candidate in enumerate(self.buckets)
            }
        return self._forward_bucket_indices.get(id(bucket))

    def recompute_prefetch_bucket_index(self, bucket: BucketRuntime) -> int | None:
        """Return bucket's index in RAF recompute prefetch order."""
        if self._recompute_prefetch_bucket_indices is None:
            self._recompute_prefetch_bucket_indices = {
                id(candidate): idx
                for idx, candidate in enumerate(self.recompute_prefetch_buckets())
            }
        return self._recompute_prefetch_bucket_indices.get(id(bucket))

    def recompute_prefetch_buckets(self) -> list[BucketRuntime]:
        """Return RAF recompute prefetch order.

        Forward execution order lists buckets inside a top-level unit in the
        order their hooks fire. During backward recompute, top-level units are
        visited in reverse, but each unit still replays its forward hook order.
        """
        if self._recompute_prefetch_buckets is not None:
            return self._recompute_prefetch_buckets

        unit_order: list[object] = []
        buckets_by_unit: dict[object, list[BucketRuntime]] = {}
        for bucket in self.buckets:
            if not bucket.bucket_storage._reshard_after_forward:
                continue
            unit_key = self.recompute_prefetch_unit_key(bucket)
            if unit_key not in buckets_by_unit:
                unit_order.append(unit_key)
                buckets_by_unit[unit_key] = []
            buckets_by_unit[unit_key].append(bucket)

        recompute_order: list[BucketRuntime] = []
        for unit_key in reversed(unit_order):
            recompute_order.extend(buckets_by_unit[unit_key])
        self._recompute_prefetch_buckets = recompute_order
        self._recompute_prefetch_bucket_indices = {
            id(bucket): idx for idx, bucket in enumerate(recompute_order)
        }
        return recompute_order

    @staticmethod
    def recompute_prefetch_unit_key(bucket: BucketRuntime) -> object:
        """Return the top-level execution unit key for a bucket."""
        key_fn = getattr(bucket, "recompute_prefetch_unit_key", None)
        if callable(key_fn):
            return key_fn()
        return id(bucket)

    def should_defer_reduce_grad_for_backward_prefetch(
        self,
        bucket: BucketRuntime,
    ) -> bool:
        """Return whether reduce-grad should wait for backward prefetch."""
        next_bucket = self.next_backward_unshard_bucket(bucket)
        if next_bucket is None or not self.should_prefetch_bucket(next_bucket):
            return False
        backward_prefetch_key = next_bucket.pending_unshard_key(recompute=True)
        return backward_prefetch_key not in self.pending_unshards

    def should_prefetch_bucket(self, bucket: BucketRuntime) -> bool:
        """Return whether this bucket can be unsharded from another module hook."""
        _ = bucket
        return True

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
            unshard_stream=device_handle.Stream(priority=-1),
            reduce_grad_stream=device_handle.Stream(priority=-1),
            reduce_grad_release_stream=device_handle.Stream(priority=-1),
        )
        contexts[device] = context
        return context

    def queue_reduce_grad_wait(self) -> None:
        """Queue a post-backward wait for reduce-grad work."""
        if self.reduce_grad_callback_queued:
            return
        self.reduce_grad_callback_queued = True

        def _wait_for_reduce_grad() -> None:
            try:
                self.flush_pending_reduce_grad_launches(max_to_flush=None)
                self.wait_and_clear_reduce_grad_states(debug_fqn=None)
            finally:
                self.pending_reduce_grad_launches.clear()
                self.reduce_grad_callback_queued = False

        torch.autograd.Variable._execution_engine.queue_callback(_wait_for_reduce_grad)

    def queue_raf_saved_unshard_cache_clear(self) -> None:
        """Queue cleanup for RAF saved-tensor backward unshard values."""
        if self.raf_saved_unshard_cache_callback_queued:
            return
        self.raf_saved_unshard_cache_callback_queued = True

        def _clear_raf_saved_unshard_cache() -> None:
            try:
                self.raf_saved_unshard_cache.clear()
            finally:
                self.raf_saved_unshard_cache_callback_queued = False

        torch.autograd.Variable._execution_engine.queue_callback(
            _clear_raf_saved_unshard_cache
        )

    def wait_and_clear_reduce_grad_states(
        self,
        debug_fqn: str | None,
    ) -> None:
        """Host-retire prior reduce-grad states and release their buffers."""
        if not self.reduce_grad_states:
            return
        with _record_function_if_eager(
            "FlexShard::post_backward_reduce_grad_wait",
            debug_fqn,
        ):
            for pending in self.reduce_grad_states:
                self.synchronize_and_release_reduce_grad_state(pending)
            self.reduce_grad_states.clear()

    def synchronize_and_release_reduce_grad_state(
        self,
        pending: PendingReduceGrad,
    ) -> None:
        pending.result.synchronize()
        with self.device_handle.stream(self.reduce_grad_release_stream):
            pending.result.release_buffers(
                release_sharded_grads=True,
            )

    def drain_reduce_grad_states_if_needed(
        self,
        debug_fqn: str | None,
    ) -> None:
        if self.max_pending_reduce_grads <= 0:
            return
        while len(self.reduce_grad_states) >= self.max_pending_reduce_grads:
            pending = self.reduce_grad_states.pop(0)
            with _record_function_if_eager(
                "FlexShard::post_backward_reduce_grad_retire",
                debug_fqn,
            ):
                self.synchronize_and_release_reduce_grad_state(pending)

    def _launch_pending_reduce_grad(
        self,
        pending: PendingReduceGradLaunch,
    ) -> None:
        bucket = pending.bucket
        self.drain_reduce_grad_states_if_needed(bucket.debug_fqn)
        result = launch_reduce_grad(
            pending.prepared,
            self.reduce_grad_stream,
        )
        with self.device_handle.stream(self.reduce_grad_stream):
            sharded_grads = result.finish()
            result.record_sharded_grads(
                _accumulate_sharded_grads(
                    pending.param_owners,
                    sharded_grads,
                ),
                self.reduce_grad_stream,
            )
        self.reduce_grad_states.append(PendingReduceGrad(result))

    def queue_reduce_grad_launch(
        self,
        bucket: BucketRuntime,
        grads: list[torch.Tensor],
        infos: list[ParamInfo],
        param_owners: list[ParamOwnerRef],
    ) -> None:
        """Pack reduce-grad input and delay launch until after next unshard."""
        if not grads:
            return
        with torch.no_grad():
            prepared = prepare_reduce_grad(
                grads,
                infos,
                bucket.bucket_storage._mesh,
                debug_fqn=bucket.debug_fqn,
            )
        self.pending_reduce_grad_launches.append(
            PendingReduceGradLaunch(
                bucket=bucket,
                prepared=prepared,
                param_owners=param_owners,
            )
        )
        self.queue_reduce_grad_wait()

    def flush_pending_reduce_grad_launches(
        self,
        max_to_flush: int | None,
    ) -> None:
        """Launch deferred reduce-grads after unshard prefetch has priority."""
        num_flushed = 0
        while self.pending_reduce_grad_launches and (
            max_to_flush is None or num_flushed < max_to_flush
        ):
            pending = self.pending_reduce_grad_launches.pop(0)
            with torch.no_grad():
                self._launch_pending_reduce_grad(pending)
            num_flushed += 1
        if num_flushed:
            self.queue_reduce_grad_wait()


@dataclass
class BucketUnshardRuntime:
    """Runtime metadata passed to bucket unshard autograd."""

    bucket: BucketRuntime
    prefetched_result: UnshardHandle | None


@dataclass(frozen=True)
class RafSavedUnshardedParam:
    """Saved-tensor hook handle for a RAF full parameter."""

    bucket: BucketRuntime
    param_index: int

    def unpack_raf_saved_tensor(self) -> torch.Tensor:
        bucket_id = id(self.bucket.bucket_storage)
        full_params = self.bucket.context.raf_saved_unshard_cache.get(bucket_id)
        if full_params is None:
            recompute_state = (
                self.bucket.bucket_storage._reshard_after_forward_recompute_state
            )
            token = None
            if recompute_state is not None:
                token = recompute_state.enter_recompute(frozenset({bucket_id}))
            try:
                full_params = self.bucket.recompute_unshard_for_saved_tensor()
            finally:
                if token is not None and recompute_state is not None:
                    recompute_state.exit_recompute(token)
            self.bucket.context.raf_saved_unshard_cache[bucket_id] = full_params
            self.bucket.context.queue_raf_saved_unshard_cache_clear()

        full_param = full_params[self.param_index]
        slot = self.bucket.bucket_params[self.param_index].unsharded_param_slot
        return slot.apply_unsharded_param_policy(full_param)


@dataclass
class BucketRuntime:
    """Runtime state and hooks for one FlexShard bucket."""

    bucket_storage: ShardedBucketStorage
    bucket_params: list[BucketParam]
    context: BucketCommContext
    debug_fqn: str | None

    @classmethod
    def from_bucket_storage(
        cls,
        bucket_storage: ShardedBucketStorage,
        module_param_slots: dict[nn.Module, dict[str, UnshardedParamSlot]],
        context: BucketCommContext | None = None,
    ) -> BucketRuntime | None:
        """Create runtime state for one bucket storage."""
        bucket_params = cls._get_bucket_params(bucket_storage, module_param_slots)
        logger.debug(
            f"Batched hooks: {len(bucket_params)}/{len(bucket_storage._param_infos)} params matched"
        )
        if not bucket_params:
            return None
        if context is None:
            comm_device = cls._comm_device(bucket_storage, bucket_params)
            context = BucketCommContext.get(bucket_storage._module, comm_device)
            if context is None:
                context = BucketCommContext.create(bucket_storage._module, comm_device)
        return cls(
            bucket_storage=bucket_storage,
            bucket_params=bucket_params,
            context=context,
            debug_fqn=_get_bucket_storage_debug_fqn(bucket_storage),
        )

    @staticmethod
    def _get_bucket_params(
        bucket_storage: ShardedBucketStorage,
        module_param_slots: dict[nn.Module, dict[str, UnshardedParamSlot]],
    ) -> list[BucketParam]:
        """Return params in a bucket with owning module refs and unsharded slots."""
        bucket_params: list[BucketParam] = []
        for info in bucket_storage._param_infos.values():
            param_owner = ParamOwnerRef.resolve(bucket_storage._module, info.fqn)
            if param_owner.module in module_param_slots:
                unsharded_param_slot = module_param_slots[param_owner.module].get(
                    param_owner.param_name
                )
                if unsharded_param_slot is not None:
                    bucket_params.append(
                        BucketParam(
                            param_owner=param_owner,
                            unsharded_param_slot=unsharded_param_slot,
                            param_info=info,
                        )
                    )
        return bucket_params

    @staticmethod
    def _comm_device(
        bucket_storage: ShardedBucketStorage,
        bucket_params: list[BucketParam],
    ) -> torch.device:
        """Return the device used for this bucket's collectives."""
        comm_device = bucket_storage.byte_storage.device
        for bucket_param in bucket_params:
            if bucket_param.unsharded_param_slot.compute_device is not None:
                return bucket_param.unsharded_param_slot.compute_device
        return comm_device

    @property
    def infos(self) -> list[ParamInfo]:
        return [bucket_param.param_info for bucket_param in self.bucket_params]

    @property
    def param_owners(self) -> list[ParamOwnerRef]:
        return [bucket_param.param_owner for bucket_param in self.bucket_params]

    def _local_shards(self, *, use_autograd: bool) -> list[torch.Tensor]:
        local_shards: list[torch.Tensor] = []
        for bucket_param in self.bucket_params:
            param_owner = bucket_param.param_owner
            unsharded_param_slot = bucket_param.unsharded_param_slot
            param = param_owner.module._parameters[param_owner.param_name]
            local_shard = param if use_autograd else param.data
            if (
                unsharded_param_slot.compute_device is not None
                and local_shard.device != unsharded_param_slot.compute_device
            ):
                local_shard = local_shard.to(
                    unsharded_param_slot.compute_device,
                    non_blocking=True,
                )
            local_shards.append(local_shard)
        return local_shards

    def recompute_prefetch_unit_key(self) -> object:
        """Return a key for the top-level module replayed during recompute."""
        owner_paths = sorted(
            {
                _strip_checkpoint_wrapped_module_path(
                    ".".join(info.fqn.split(".")[:-1])
                )
                for info in self.bucket_storage._param_infos.values()
                if "." in info.fqn
            }
        )
        top_level_paths = tuple(
            sorted(
                {
                    _top_level_owner_path(self.bucket_storage._module, owner_path)
                    for owner_path in owner_paths
                }
            )
        )
        if len(top_level_paths) == 1:
            return top_level_paths[0]
        return top_level_paths or id(self)

    def bucket_module(self) -> nn.Module:
        """Find the deepest common ancestor module for this bucket's params."""
        fqns = list(self.bucket_storage._param_infos.keys())
        prefixes = [".".join(fqn.split(".")[:-1]) for fqn in fqns]
        if not prefixes:
            return self.bucket_storage._module
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
            return self.bucket_storage._module
        mod = self.bucket_storage._module
        for part in common.split("."):
            mod = getattr(mod, part)
        return mod

    def resolve_bucket_forward_hook_module(self) -> nn.Module | None:
        """Return the module whose forward should trigger this bucket."""
        # Register hooks on the deepest common ancestor module for the bucket's
        # params so one pre-forward unshard covers their parameter accesses.
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
        # re-unshard the layer params after post-forward resharding.
        # Grouped root/rest buckets need replay fragments that share one
        # physical bucket unshard/reduce-grad, so reject them until that
        # support exists instead of silently installing an unreplayable hook.
        if (
            self.bucket_storage._reshard_after_forward
            and target is self.bucket_storage._module
            and any(
                hasattr(child, "_checkpoint_wrapped_module")
                for child in self.bucket_storage._module.modules()
                if child is not self.bucket_storage._module
            )
        ):
            logger.debug(
                "Skipping root-level bucket unshard hook because child "
                "checkpoint recomputation would not replay the root hook.",
            )
            return None
        return getattr(target, "_checkpoint_wrapped_module", target)

    def begin_unshard(
        self,
        *,
        sac_transparent: bool | None = None,
    ) -> UnshardHandle:
        """Begin this bucket's unshard on the shared stream."""
        return self.begin_unshard_from_tensors(
            self._local_shards(use_autograd=False),
            sac_transparent=sac_transparent,
        )

    def begin_unshard_from_tensors(
        self,
        local_shards: list[torch.Tensor],
        *,
        sac_transparent: bool | None = None,
    ) -> UnshardHandle:
        """Begin a physical bucket unshard, hidden from SAC when needed."""
        if sac_transparent is None:
            sac_transparent = self.bucket_storage._reshard_after_forward
        if sac_transparent and not torch.compiler.is_compiling():
            # Selective activation checkpointing requires the same
            # replay-visible op stream in forward and recompute. RAF prefetch
            # deliberately moves the physical all-gather between module hooks,
            # so keep low-level allocation/c10d/profiler ops out of SAC storage.
            with torch._C._DisableTorchDispatch():
                with _suppress_eager_profiling():
                    return self._begin_unshard_from_tensors(local_shards)
        return self._begin_unshard_from_tensors(local_shards)

    def _begin_unshard_from_tensors(
        self,
        local_shards: list[torch.Tensor],
    ) -> UnshardHandle:
        return begin_bucket_unshard(
            local_shards,
            self.infos,
            self.bucket_storage._mesh,
            self.context.unshard_stream,
            debug_fqn=self.debug_fqn,
        )

    def recompute_unshard_for_saved_tensor(self) -> list[torch.Tensor]:
        """Recreate full params for autograd saved-tensor unpack."""
        if torch.compiler.is_compiling():
            raise AssertionError("RAF saved-tensor unpack is eager-only.")

        result = self.take_pending()
        if result is None:
            result = self.begin_unshard(sac_transparent=False)
        full_params = result.finish()
        self.prefetch_next()
        return full_params

    def wait_pending(self, result: UnshardHandle) -> None:
        """Wait for and release an unused pending unshard result."""
        result.wait()
        result.release_buffers()

    def pending_unshard_key(self, *, recompute: bool) -> PendingUnshardKey:
        """Return the pending-unshard key for this bucket and execution phase."""
        return PendingUnshardKey(
            bucket_id=id(self.bucket_storage),
            recompute=recompute,
        )

    def clear_stale_pending_unshards(self) -> None:
        """Drain unused pending unshards so stale work cannot cross phases."""
        for pending in self.context.pending_unshards.values():
            self.wait_pending(pending.result)
        self.context.pending_unshards.clear()

    def in_reshard_after_forward_recompute(self) -> bool:
        """Return whether this bucket is in reshard-after-forward recompute."""
        recompute_state = self.bucket_storage._reshard_after_forward_recompute_state
        return (
            self.bucket_storage._reshard_after_forward
            and recompute_state is not None
            and recompute_state.is_recomputing(id(self.bucket_storage))
        )

    def prefetch_next(self) -> None:
        """Start the next bucket's unshard if no prefetch is in flight."""
        if self.context.pending_unshards:
            return
        is_recompute = self.in_reshard_after_forward_recompute()
        prefetch_order = (
            self.context.recompute_prefetch_buckets()
            if is_recompute
            else self.context.buckets
        )
        for idx, bucket in enumerate(prefetch_order):
            if bucket is self:
                break
        else:
            return
        next_idx = idx + 1
        if next_idx >= len(prefetch_order):
            return
        next_bucket = prefetch_order[next_idx]
        if not self.context.should_prefetch_bucket(next_bucket):
            return
        key = next_bucket.pending_unshard_key(recompute=is_recompute)
        self.context.pending_unshards[key] = PendingUnshard(
            bucket=next_bucket,
            result=next_bucket.begin_unshard(
                sac_transparent=(
                    self.bucket_storage._reshard_after_forward
                    or next_bucket.bucket_storage._reshard_after_forward
                ),
            ),
            recompute=is_recompute,
        )

    def take_pending(self) -> UnshardHandle | None:
        """Return a matching pending result, releasing stale pending work."""
        is_recompute = self.in_reshard_after_forward_recompute()
        key = self.pending_unshard_key(recompute=is_recompute)
        pending = self.context.pending_unshards.pop(key, None)
        if pending is None:
            self.clear_stale_pending_unshards()
            return None
        assert pending.bucket is self
        assert pending.recompute == is_recompute
        return pending.result

    def reduce_grads(
        self,
        grads: list[torch.Tensor],
        infos: list[ParamInfo],
        param_owners: list[ParamOwnerRef],
    ) -> None:
        """Reduce full-parameter grads and accumulate local sharded grads."""
        if not grads:
            return
        with torch.no_grad():
            self.context.drain_reduce_grad_states_if_needed(self.debug_fqn)
            result = begin_reduce_grad(
                grads,
                infos,
                self.bucket_storage._mesh,
                self.context.reduce_grad_stream,
                debug_fqn=self.debug_fqn,
            )
            with self.context.device_handle.stream(self.context.reduce_grad_stream):
                sharded_grads = result.finish()
                result.record_sharded_grads(
                    _accumulate_sharded_grads(
                        param_owners,
                        sharded_grads,
                    ),
                    self.context.reduce_grad_stream,
                )
            self.context.reduce_grad_states.append(PendingReduceGrad(result))
            self.context.queue_reduce_grad_wait()

    def schedule_reduce_grad_tensors(
        self,
        grads: list[torch.Tensor],
        infos: list[ParamInfo],
        param_owners: list[ParamOwnerRef],
    ) -> None:
        """Launch or defer a pre-packed bucket reduce-grad request."""
        if self.context.should_defer_reduce_grad_for_backward_prefetch(self):
            self.context.queue_reduce_grad_launch(
                self,
                grads,
                infos,
                param_owners,
            )
        else:
            self.reduce_grads(grads, infos, param_owners)

    def pre_forward_hook(self, mod, args) -> None:
        local_shards = self._local_shards(use_autograd=True)
        is_compiling = torch.compiler.is_compiling()
        is_recompute = self.in_reshard_after_forward_recompute()
        prefetched_result = None if is_compiling else self.take_pending()
        runtime = BucketUnshardRuntime(
            bucket=self,
            prefetched_result=prefetched_result,
        )
        full_params = list(_BucketUnshard.apply(runtime, *local_shards))
        if not is_compiling:
            self.prefetch_next()
        if not is_compiling and not is_recompute:
            self.context.flush_pending_reduce_grad_launches(max_to_flush=1)
        for param_index, (bucket_param, full_param) in enumerate(
            zip(self.bucket_params, full_params, strict=True)
        ):
            saved_tensor_handle = (
                RafSavedUnshardedParam(self, param_index)
                if self.bucket_storage._reshard_after_forward
                else None
            )
            bucket_param.unsharded_param_slot.push_unsharded_param(
                full_param,
                saved_tensor_handle=saved_tensor_handle,
            )

    def post_forward_hook(self, mod, args, output) -> None:
        for bucket_param in self.bucket_params:
            bucket_param.unsharded_param_slot.pop_unsharded_param()


class _BucketUnshard(torch.autograd.Function):
    """Autograd boundary for bucket unshard.

    Forward consumes a raw unshard result, either prefetched by the previous
    bucket or launched on demand. Backward packs full-parameter gradients and
    either launches or defers one explicit bucket reduce-grad. The reduced
    sharded grads are accumulated into the original sharded parameters
    asynchronously instead of returned through autograd, so later backward
    compute can overlap with the reduce-grad.
    """

    @staticmethod
    def forward(
        ctx: Any,
        runtime: BucketUnshardRuntime,
        *local_shards: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.runtime = runtime
        ctx.num_inputs = len(local_shards)
        ctx.local_shard_dtypes = tuple(shard.dtype for shard in local_shards)

        result = runtime.prefetched_result
        runtime.prefetched_result = None
        if result is None:
            result = runtime.bucket.begin_unshard_from_tensors(
                [shard.detach() for shard in local_shards],
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
        runtime: BucketUnshardRuntime = ctx.runtime
        bucket = runtime.bucket
        is_compiling = torch.compiler.is_compiling()
        input_grads: list[torch.Tensor | None] | None = (
            [None] * ctx.num_inputs if is_compiling else None
        )
        grads: list[torch.Tensor] = []
        valid_infos: list[ParamInfo] = []
        valid_param_owners: list[ParamOwnerRef] = []
        valid_indices: list[int] = []
        for idx, (grad, bucket_param) in enumerate(
            zip(
                full_param_grads,
                bucket.bucket_params,
                strict=True,
            )
        ):
            if grad is None:
                continue
            if is_compiling:
                valid_indices.append(idx)
            grads.append(grad)
            valid_infos.append(bucket_param.param_info)
            valid_param_owners.append(bucket_param.param_owner)

        if grads and is_compiling:
            if input_grads is None:
                raise AssertionError("Expected compile backward input gradients.")
            result = begin_reduce_grad(
                grads,
                valid_infos,
                bucket.bucket_storage._mesh,
                bucket.context.reduce_grad_stream,
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
            bucket.schedule_reduce_grad_tensors(
                grads,
                valid_infos,
                valid_param_owners,
            )

        return (None, *([None] * ctx.num_inputs))


def _accumulate_sharded_grads(
    param_owners: list[ParamOwnerRef],
    sharded_grads: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Cast sharded grads to local param dtype and accumulate into .grad."""
    stored_grads: list[torch.Tensor] = []
    for param_owner, grad in zip(param_owners, sharded_grads, strict=True):
        param = param_owner.module._parameters[param_owner.param_name]
        if grad.dtype != param.dtype:
            grad = grad.to(param.dtype)
        stored_grads.append(grad)
        if param.grad is None:
            param.grad = grad
        else:
            param.grad += grad
    return stored_grads


def _raise_unreplayable_reshard_hook(bucket_storage: ShardedBucketStorage) -> None:
    bucket_fqn = _get_bucket_storage_debug_fqn(bucket_storage)
    bucket_msg = f" for bucket {bucket_fqn!r}" if bucket_fqn else ""
    raise RuntimeError(
        "FlexShard eager reshard_after_forward could not register a "
        f"recomputation-safe bucket unshard hook{bucket_msg}. "
        "Bucket hooks must run both in the original forward and in activation "
        "checkpoint recomputation. Split the bucket to match forward execution "
        "units, or set reshard_after_forward=False for this bucket."
    )


def _create_unsharded_param_slots(
    module: nn.Module,
    bucket_storages: list[ShardedBucketStorage],
    fqn_to_bucket_spec: dict[str, BucketSpec],
    device: torch.device,
) -> dict[nn.Module, dict[str, UnshardedParamSlot]]:
    """Create unsharded parameter slots grouped by owning leaf module."""
    module_param_slots: dict[nn.Module, dict[str, UnshardedParamSlot]] = {}

    for bucket_storage in bucket_storages:
        bucket_fqn = _get_bucket_storage_debug_fqn(bucket_storage)
        for fqn, info in bucket_storage._param_infos.items():
            bucket_spec = fqn_to_bucket_spec[fqn]
            compute_device = (
                torch.device(device) if bucket_spec.offload_policy is not None else None
            )
            unsharded_param_slot = UnshardedParamSlot(
                param_fqn=fqn,
                bucket_fqn=bucket_fqn,
                param_dtype=info.param_dtype,
                compute_device=compute_device,
            )

            param_owner = ParamOwnerRef.resolve(module, fqn)
            module_param_slots.setdefault(param_owner.module, {})[
                param_owner.param_name
            ] = unsharded_param_slot

    return module_param_slots


def _install_bucket_unshard_hooks(
    bucket_storages: list[ShardedBucketStorage],
    module_param_slots: dict[nn.Module, dict[str, UnshardedParamSlot]],
) -> None:
    """Install pre/post forward hooks for per-bucket unshard.

    In eager mode, each ShardedBucketStorage's pre-forward hook starts a single bucket
    unshard call (one collective per bucket), then fills each
    UnshardedParamSlot so the property getter can return the hook-provided
    tensor.

    CUDA buckets use the same custom autograd bucket path in eager and compile
    so Dynamo traces the same bucket pre-hook and parameter access logic.
    """
    for bucket_storage in bucket_storages:
        if not bucket_storage._param_infos:
            raise AssertionError("Expected FlexShard bucket storage to own parameters.")
        if bucket_storage.byte_storage.device.type != "cuda":
            raise AssertionError("Expected FlexShard bucket storage to be on CUDA.")

        bucket_runtime = BucketRuntime.from_bucket_storage(
            bucket_storage=bucket_storage,
            module_param_slots=module_param_slots,
        )
        if bucket_runtime is None:
            continue

        target = bucket_runtime.resolve_bucket_forward_hook_module()
        if target is None:
            _raise_unreplayable_reshard_hook(bucket_storage)
        target.register_forward_pre_hook(bucket_runtime.pre_forward_hook)
        target.register_forward_hook(
            bucket_runtime.post_forward_hook,
            always_call=True,
        )
        bucket_runtime.context.add_bucket(bucket_runtime)
        for bucket_param in bucket_runtime.bucket_params:
            bucket_param.unsharded_param_slot.bucket_unshard_hook_registered = True
