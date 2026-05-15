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
from .utils import _get_bucket_storage_debug_fqn, _record_function_if_eager


logger = logging.getLogger(__name__)


_EAGER_COMM_CONTEXTS_ATTR = "_flex_shard_eager_comm_contexts"


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
    """The single one-bucket-ahead unshard in flight."""

    bucket: BucketRuntime
    result: UnshardHandle
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
    buckets: list[BucketRuntime] = field(default_factory=list)
    pending: PendingUnshard | None = None
    pending_reduce_grad_launches: list[PendingReduceGradLaunch] = field(
        default_factory=list
    )
    reduce_grad_states: list[PendingReduceGrad] = field(default_factory=list)
    reduce_grad_callback_queued: bool = False

    def next_backward_unshard_bucket(
        self,
        bucket: BucketRuntime,
    ) -> BucketRuntime | None:
        """Return the next bucket whose backward unshard has priority.

        Buckets execute forward in ``self.buckets`` order and backward in reverse
        order. After bucket ``i`` produces gradients, bucket ``i - 1``'s
        unshard is the next critical-path backward communication when that
        bucket uses reshard-after-forward. In that case bucket ``i``'s
        reduce-grad should be delayed until after the previous bucket's
        unshard has launched.
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
        if not next_bucket.bucket_storage._reshard_after_forward:
            return None
        return next_bucket

    def should_defer_reduce_grad_for_backward_prefetch(
        self,
        bucket: BucketRuntime,
    ) -> bool:
        """Return whether reduce-grad should wait for backward prefetch."""
        return self.next_backward_unshard_bucket(bucket) is not None

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
                for pending in self.reduce_grad_states:
                    pending.result.wait()
                    pending.result.release_buffers(
                        release_sharded_grads=True,
                    )
            finally:
                self.reduce_grad_states.clear()
                self.pending_reduce_grad_launches.clear()
                self.reduce_grad_callback_queued = False

        torch.autograd.Variable._execution_engine.queue_callback(_wait_for_reduce_grad)

    def wait_and_clear_reduce_grad_states(
        self,
        debug_fqn: str | None,
    ) -> None:
        """Wait for prior reduce-grad states and release their buffers."""
        if not self.reduce_grad_states:
            return
        with _record_function_if_eager(
            "FlexShard::post_backward_reduce_grad_wait",
            debug_fqn,
        ):
            for pending in self.reduce_grad_states:
                pending.result.wait()
                pending.result.release_buffers(
                    release_sharded_grads=True,
                )
            self.reduce_grad_states.clear()

    def _launch_pending_reduce_grad(
        self,
        pending: PendingReduceGradLaunch,
    ) -> None:
        bucket = pending.bucket
        self.wait_and_clear_reduce_grad_states(bucket.debug_fqn)
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

    def begin_unshard(self) -> UnshardHandle:
        """Begin this bucket's unshard on the shared stream."""
        return begin_bucket_unshard(
            self._local_shards(use_autograd=False),
            self.infos,
            self.bucket_storage._mesh,
            self.context.unshard_stream,
            debug_fqn=self.debug_fqn,
        )

    def wait_pending(self, result: UnshardHandle) -> None:
        """Wait for and release an unused pending unshard result."""
        result.wait()
        result.release_buffers()

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
        self.context.pending = PendingUnshard(
            bucket=next_bucket,
            result=next_bucket.begin_unshard(),
            recompute=is_recompute,
        )

    def take_pending(self) -> UnshardHandle | None:
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
        param_owners: list[ParamOwnerRef],
    ) -> None:
        """Reduce full-parameter grads and accumulate local sharded grads."""
        if not grads:
            return
        with torch.no_grad():
            self.context.wait_and_clear_reduce_grad_states(self.debug_fqn)
            # TODO: Thread the bucket's reduce_dtype into the non-reshard-after-forward
            # path before launching reduce-grad. Reshard-after-forward uses
            # _MixedPrecisionCast backward, but detached non-reshard-after-forward
            # leaves can arrive in param_dtype.
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

    def pre_forward_hook(self, mod, args) -> None:
        if torch.compiler.is_compiling():
            return
        local_shards = self._local_shards(use_autograd=True)
        prefetched_result = self.take_pending()
        runtime = BucketUnshardRuntime(
            bucket=self,
            prefetched_result=prefetched_result,
        )
        full_params = list(_BucketUnshard.apply(runtime, *local_shards))
        self.prefetch_next()
        self.context.flush_pending_reduce_grad_launches(max_to_flush=1)
        for bucket_param, full_param in zip(
            self.bucket_params, full_params, strict=True
        ):
            bucket_param.unsharded_param_slot._current_unsharded_param = full_param

    def post_forward_hook(self, mod, args, output) -> None:
        if torch.compiler.is_compiling():
            return
        for bucket_param in self.bucket_params:
            bucket_param.unsharded_param_slot._current_unsharded_param = None


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

        result = runtime.prefetched_result
        runtime.prefetched_result = None
        if result is None:
            result = begin_bucket_unshard(
                [shard.detach() for shard in local_shards],
                runtime.bucket.infos,
                runtime.bucket.bucket_storage._mesh,
                runtime.bucket.context.unshard_stream,
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
        runtime: BucketUnshardRuntime = ctx.runtime
        bucket = runtime.bucket
        grads: list[torch.Tensor] = []
        valid_infos: list[ParamInfo] = []
        valid_param_owners: list[ParamOwnerRef] = []
        for grad, bucket_param in zip(
            full_param_grads,
            bucket.bucket_params,
            strict=True,
        ):
            if grad is None:
                continue
            grads.append(grad.contiguous())
            valid_infos.append(bucket_param.param_info)
            valid_param_owners.append(bucket_param.param_owner)

        if grads:
            if bucket.context.should_defer_reduce_grad_for_backward_prefetch(bucket):
                bucket.context.queue_reduce_grad_launch(
                    bucket,
                    grads,
                    valid_infos,
                    valid_param_owners,
                )
            else:
                bucket.reduce_grads(grads, valid_infos, valid_param_owners)

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
            mp_policy = bucket_spec.mp_policy
            param_dtype = mp_policy.param_dtype if mp_policy else None
            reduce_dtype = mp_policy.reduce_dtype if mp_policy else None
            compute_device = (
                torch.device(device) if bucket_spec.offload_policy is not None else None
            )
            unsharded_param_slot = UnshardedParamSlot(
                param_fqn=fqn,
                bucket_fqn=bucket_fqn,
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
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
        target.register_forward_hook(bucket_runtime.post_forward_hook)
        bucket_runtime.context.buckets.append(bucket_runtime)
        for bucket_param in bucket_runtime.bucket_params:
            bucket_param.unsharded_param_slot.bucket_unshard_hook_registered = True
