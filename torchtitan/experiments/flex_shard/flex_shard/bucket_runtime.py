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
from .utils import (
    _get_storage_debug_fqn,
    _module_path_common_prefix,
    _record_function_if_eager,
    _strip_checkpoint_wrapped_module_path,
    _top_level_owner_path,
)


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
    """One packed reduce-grad request waiting for an all-gather to launch first."""

    bucket: BucketRuntime
    prepared: PreparedReduceScatterGrad
    param_refs: list[ParamModuleInfo]


@dataclass
class BucketCommContext:
    """Communication streams shared by buckets on one root module/device."""

    device_handle: ModuleType
    all_gather_stream: torch.Stream
    reduce_scatter_stream: torch.Stream
    buckets: list[BucketReplayFragment] = field(default_factory=list)
    pending: PendingAllGather | None = None
    pending_reduce_scatter_launches: list[PendingReduceScatterLaunch] = field(
        default_factory=list
    )
    reduce_scatter_states: list[PendingReduceGrad] = field(default_factory=list)
    reduce_scatter_callback_queued: bool = False

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


@dataclass
class BucketGradAccumulator:
    """Accumulate replay-fragment grads before one physical bucket RS."""

    bucket: BucketRuntime
    expected_indices: set[int]
    received_indices: set[int] = field(default_factory=set)
    grads_by_index: dict[int, torch.Tensor] = field(default_factory=dict)
    launched: bool = False

    def report(
        self,
        fragment: BucketReplayFragment,
        full_param_grads: tuple[torch.Tensor | None, ...],
    ) -> None:
        for entry_idx, entry, grad in zip(
            fragment.entry_indices,
            fragment.entries,
            full_param_grads,
            strict=True,
        ):
            if not entry.param_info.requires_grad:
                continue
            if entry_idx in self.received_indices:
                raise AssertionError(
                    "Received duplicate full-parameter grad for "
                    f"{entry.param_info.fqn!r} in bucket {self.bucket.debug_fqn!r}."
                )
            self.received_indices.add(entry_idx)
            if grad is not None:
                self.grads_by_index[entry_idx] = grad.contiguous()

        if self.expected_indices.issubset(self.received_indices):
            if self.launched:
                raise AssertionError(
                    f"Reduce-scatter for bucket {self.bucket.debug_fqn!r} "
                    "was already launched."
                )
            self.launched = True
            self.bucket.reduce_accumulated_grads(self.grads_by_index)


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


def _entry_owner_path(entry: ParamEntry) -> str:
    """Return the module path that owns this parameter."""
    fqn = entry.param_info.fqn
    return _strip_checkpoint_wrapped_module_path(".".join(fqn.split(".")[:-1]))


def _entry_belongs_to_module_path(entry: ParamEntry, module_path: str) -> bool:
    """Return whether this entry is owned by the module path or its children."""
    owner_path = _entry_owner_path(entry)
    return owner_path == module_path or owner_path.startswith(module_path + ".")


@dataclass
class BucketRuntime:
    """Physical runtime state for one user FlexShard bucket."""

    storage: DStorage
    entries: list[ParamEntry]
    context: BucketCommContext
    debug_fqn: str | None
    fragments: list[BucketReplayFragment] = field(default_factory=list)
    active_full_params: tuple[torch.Tensor, ...] | None = None
    active_recompute: bool | None = None
    grad_accumulator: BucketGradAccumulator | None = None

    @classmethod
    def from_storage_fragments(
        cls,
        storage: DStorage,
        module_param_map: dict[nn.Module, dict[str, EagerParamAccessState]],
        context: BucketCommContext | None = None,
        *,
        allow_root_fragmentation: bool,
    ) -> list[BucketReplayFragment]:
        """Create replay fragments backed by one physical storage bucket."""
        entries = cls._get_param_entries(storage, module_param_map)
        logger.debug(
            f"Batched hooks: {len(entries)}/{len(storage._param_infos)} params matched"
        )
        if not entries:
            return []
        if context is None:
            comm_device = cls._comm_device(storage, entries)
            context = BucketCommContext.get(storage._module, comm_device)
            if context is None:
                context = BucketCommContext.create(storage._module, comm_device)
        runtime = cls(
            storage=storage,
            entries=entries,
            context=context,
            debug_fqn=_get_storage_debug_fqn(storage),
        )
        fragment_paths = runtime._root_replay_fragment_paths(
            allow_root_fragmentation=allow_root_fragmentation,
        )
        if not fragment_paths:
            return [
                runtime._create_fragment(
                    entries=entries,
                    entry_indices=list(range(len(entries))),
                    debug_fqn=runtime.debug_fqn,
                )
            ]

        fragments: list[BucketReplayFragment] = []
        for path in fragment_paths:
            indexed_entries = [
                (idx, entry)
                for idx, entry in enumerate(entries)
                if _entry_belongs_to_module_path(entry, path)
            ]
            fragment_indices = [idx for idx, _ in indexed_entries]
            fragment_entries = [entry for _, entry in indexed_entries]
            if not fragment_entries:
                raise AssertionError(
                    f"Expected replay fragment {path!r} to own parameters."
                )
            fragments.append(
                runtime._create_fragment(
                    entries=fragment_entries,
                    entry_indices=fragment_indices,
                    debug_fqn=path,
                )
            )
        return fragments

    def _create_fragment(
        self,
        entries: list[ParamEntry],
        entry_indices: list[int],
        debug_fqn: str | None,
    ) -> BucketReplayFragment:
        fragment = BucketReplayFragment(
            bucket=self,
            entries=entries,
            entry_indices=entry_indices,
            debug_fqn=debug_fqn,
        )
        self.fragments.append(fragment)
        return fragment

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

    def _root_replay_fragment_paths(
        self,
        *,
        allow_root_fragmentation: bool,
    ) -> list[str]:
        """Return top-level replay fragments for a root-level bucket, if any."""
        if not allow_root_fragmentation or not self.storage._reshard_after_forward:
            return []
        if self.bucket_module() is not self.storage._module:
            return []
        has_checkpointed_child = any(
            hasattr(child, "_checkpoint_wrapped_module")
            for child in self.storage._module.modules()
            if child is not self.storage._module
        )
        if not has_checkpointed_child:
            return []

        owner_paths = [_entry_owner_path(entry) for entry in self.entries]
        if not owner_paths or any(not path for path in owner_paths):
            return []
        if _module_path_common_prefix(owner_paths):
            return []

        fragment_paths = sorted(
            {
                _top_level_owner_path(self.storage._module, owner_path)
                for owner_path in owner_paths
            }
        )
        fragment_paths = [path for path in fragment_paths if path]
        # Keep mixed root/layer buckets rejected for now. A ModuleList child
        # such as layers.0 needs a replay fragment under a container, which
        # has different ordering and prefetch tradeoffs from top-level
        # root/rest fragments.
        if len(fragment_paths) <= 1 or any("." in path for path in fragment_paths):
            return []
        return fragment_paths

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

    def bucket_module(self, entries: list[ParamEntry] | None = None) -> nn.Module:
        """Find the deepest common ancestor module for this bucket's params."""
        if entries is None:
            entries = self.entries
        fqns = [entry.param_info.fqn for entry in entries]
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

    def resolve_bucket_forward_hook_module(
        self,
        entries: list[ParamEntry],
    ) -> nn.Module | None:
        """Return the module whose forward should trigger this bucket."""
        # Register hooks on the deepest common ancestor module for the bucket's
        # params so one pre-forward all-gather covers their parameter accesses.
        # For example, a bucket with "layers.0.attn.weight" and
        # "layers.0.mlp.weight" hooks "layers.0".
        target = self.bucket_module(entries)
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

    def has_active_full_params_for(self, is_recompute: bool) -> bool:
        return (
            self.active_full_params is not None
            and self.active_recompute == is_recompute
        )

    def release_active_full_params(self) -> None:
        self.active_full_params = None
        self.active_recompute = None

    def in_reshard_after_forward_recompute(self) -> bool:
        """Return whether this bucket is in reshard-after-forward recompute."""
        recompute_state = self.storage._reshard_after_forward_recompute_state
        return (
            self.storage._reshard_after_forward
            and recompute_state is not None
            and recompute_state.is_recomputing(id(self.storage))
        )

    def prefetch_next(self, current_fragment: BucketReplayFragment) -> None:
        """Start the next bucket's all-gather if no prefetch is in flight."""
        if self.context.pending is not None:
            return
        prefetch_order = self.context.buckets
        is_recompute = self.in_reshard_after_forward_recompute()
        if is_recompute:
            prefetch_order = prefetch_order[::-1]
        for idx, fragment in enumerate(prefetch_order):
            if fragment is current_fragment:
                break
        else:
            return
        next_bucket: BucketRuntime | None = None
        for next_fragment in prefetch_order[idx + 1 :]:
            candidate = next_fragment.bucket
            if candidate is self:
                continue
            if candidate.has_active_full_params_for(is_recompute):
                continue
            next_bucket = candidate
            break
        if next_bucket is None:
            return
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

    def materialize_full_params_for_fragment(
        self,
        fragment: BucketReplayFragment,
    ) -> list[torch.Tensor]:
        """Return fragment full params from this physical bucket's AG result."""
        is_recompute = self.in_reshard_after_forward_recompute()
        if self.active_full_params is not None and self.active_recompute != is_recompute:
            self.release_active_full_params()

        if self.active_full_params is None:
            local_shards = self._local_shards(use_autograd=True)
            prefetched_result = self.take_pending()
            runtime = BucketAllGatherRuntime(
                bucket=self,
                prefetched_result=prefetched_result,
            )
            self.active_full_params = tuple(
                _BucketAllGather.apply(runtime, *local_shards)
            )
            self.active_recompute = is_recompute

        selected = [self.active_full_params[idx] for idx in fragment.entry_indices]
        if self.storage._reshard_after_forward:
            return list(_BucketFragmentAccess.apply(fragment, *selected))
        return selected

    def should_release_after_fragment(self, fragment: BucketReplayFragment) -> bool:
        """Return whether this fragment ends the current bucket window."""
        if len(self.fragments) == 1:
            return True
        if self.in_reshard_after_forward_recompute():
            return fragment is self.fragments[0]
        return fragment is self.fragments[-1]

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

    def report_fragment_grads(
        self,
        fragment: BucketReplayFragment,
        full_param_grads: tuple[torch.Tensor | None, ...],
    ) -> None:
        """Accumulate fragment grads and launch one physical bucket RS."""
        if self.grad_accumulator is None:
            expected_indices = {
                idx
                for idx, entry in enumerate(self.entries)
                if entry.param_info.requires_grad
            }
            self.grad_accumulator = BucketGradAccumulator(
                bucket=self,
                expected_indices=expected_indices,
            )
        self.grad_accumulator.report(fragment, full_param_grads)

    def reduce_accumulated_grads(
        self,
        grads_by_index: dict[int, torch.Tensor],
    ) -> None:
        """Launch one reduce-scatter for accumulated physical bucket grads."""
        grads: list[torch.Tensor] = []
        valid_infos: list[ParamInfo] = []
        valid_param_refs: list[ParamModuleInfo] = []
        for idx, entry in enumerate(self.entries):
            grad = grads_by_index.get(idx)
            if grad is None:
                continue
            grads.append(grad)
            valid_infos.append(entry.param_info)
            valid_param_refs.append(entry.module_info)

        if grads:
            self.context.queue_reduce_scatter_launch(
                self,
                grads,
                valid_infos,
                valid_param_refs,
            )
        self.grad_accumulator = None
        self.release_active_full_params()


@dataclass
class BucketReplayFragment:
    """Replay-safe hook runtime for one execution unit in a physical bucket."""

    bucket: BucketRuntime
    entries: list[ParamEntry]
    entry_indices: list[int]
    debug_fqn: str | None
    hook_module: nn.Module | None = None

    def resolve_bucket_forward_hook_module(self) -> nn.Module | None:
        return self.bucket.resolve_bucket_forward_hook_module(self.entries)

    def pre_forward_hook(self, mod, args) -> None:
        if torch.compiler.is_compiling():
            return
        full_params = self.bucket.materialize_full_params_for_fragment(self)
        self.bucket.prefetch_next(self)
        self.bucket.context.flush_pending_reduce_scatter_launches(max_to_flush=1)
        for entry, full_param in zip(self.entries, full_params, strict=True):
            entry.access_state._pre_gathered = full_param

    def post_forward_hook(self, mod, args, output) -> None:
        if torch.compiler.is_compiling():
            return
        for entry in self.entries:
            entry.access_state._pre_gathered = None
        if self.bucket.should_release_after_fragment(self):
            self.bucket.release_active_full_params()


class _BucketFragmentAccess(torch.autograd.Function):
    """Autograd boundary for one replay fragment's parameter accesses."""

    @staticmethod
    def forward(
        ctx: Any,
        fragment: BucketReplayFragment,
        *full_params: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        ctx.fragment = fragment
        ctx.num_inputs = len(full_params)
        frozen_params = [
            full_param
            for full_param, entry in zip(full_params, fragment.entries, strict=True)
            if not entry.param_info.requires_grad
        ]
        if frozen_params:
            ctx.mark_non_differentiable(*frozen_params)
        return tuple(full_param.view_as(full_param) for full_param in full_params)

    @staticmethod
    def backward(
        ctx: Any,
        *full_param_grads: torch.Tensor | None,
    ) -> tuple[Any, ...]:
        fragment: BucketReplayFragment = ctx.fragment
        fragment.bucket.report_fragment_grads(fragment, full_param_grads)
        return (None, *([None] * ctx.num_inputs))


class _BucketAllGather(torch.autograd.Function):
    """Autograd boundary for bucket all-gather.

    Forward consumes a raw all-gather result, either prefetched by the previous
    bucket or launched on demand. For reshard_after_forward=False, backward
    launches the bucket reduce-scatter directly. For reshard_after_forward=True,
    replay-fragment access nodes aggregate grads into the physical bucket so one
    reduce-scatter covers every fragment in the user bucket.
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
        grads: list[torch.Tensor] = []
        valid_infos: list[ParamInfo] = []
        valid_param_refs: list[ParamModuleInfo] = []
        for grad, entry in zip(
            full_param_grads,
            bucket.entries,
            strict=True,
        ):
            if grad is None:
                continue
            grads.append(grad.contiguous())
            valid_infos.append(entry.param_info)
            valid_param_refs.append(entry.module_info)

        if grads and not bucket.storage._reshard_after_forward:
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


def _sort_context_buckets_by_module_order(
    root_module: nn.Module,
    context: BucketCommContext,
) -> None:
    """Sort fragment prefetch order by registered hook module order."""
    module_order = {id(module): idx for idx, module in enumerate(root_module.modules())}
    indexed_fragments = list(enumerate(context.buckets))
    indexed_fragments.sort(
        key=lambda item: (
            module_order.get(id(item[1].hook_module), len(module_order)),
            item[0],
        )
    )
    context.buckets[:] = [fragment for _, fragment in indexed_fragments]

    fragments_by_bucket: dict[int, list[BucketReplayFragment]] = {}
    physical_buckets: dict[int, BucketRuntime] = {}
    for fragment in context.buckets:
        bucket_id = id(fragment.bucket)
        fragments_by_bucket.setdefault(bucket_id, []).append(fragment)
        physical_buckets[bucket_id] = fragment.bucket
    for bucket_id, fragments in fragments_by_bucket.items():
        physical_buckets[bucket_id].fragments[:] = fragments


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

    In eager mode, a bucket's pre-forward hook starts a batched all-gather and
    sets _pre_gathered on each parameter access state so the property getter can
    return the hook-provided tensor. A root-level bucket with disjoint top-level
    execution units may expand into multiple replay fragments, but the physical
    bucket still owns one all-gather/reduce-scatter contract.
    """
    contexts_by_id: dict[int, BucketCommContext] = {}
    allow_root_fragmentation = len(storages) > 1
    for storage in storages:
        if not storage._param_infos:
            raise AssertionError("Expected FlexShard bucket storage to own parameters.")
        if storage.byte_storage.device.type != "cuda":
            raise AssertionError("Expected FlexShard bucket storage to be on CUDA.")

        bucket_fragments = BucketRuntime.from_storage_fragments(
            storage=storage,
            module_param_map=module_param_map,
            allow_root_fragmentation=allow_root_fragmentation,
        )
        if not bucket_fragments:
            continue

        for fragment in bucket_fragments:
            target = fragment.resolve_bucket_forward_hook_module()
            if target is None:
                _raise_unreplayable_reshard_hook(storage)
            target.register_forward_pre_hook(fragment.pre_forward_hook)
            target.register_forward_hook(fragment.post_forward_hook)
            fragment.hook_module = target
            fragment.bucket.context.buckets.append(fragment)
            contexts_by_id[id(fragment.bucket.context)] = fragment.bucket.context
            for entry in fragment.entries:
                setattr(entry.access_state, _EAGER_BATCHED_HOOK_REGISTERED_ATTR, True)

    for context in contexts_by_id.values():
        _sort_context_buckets_by_module_order(storages[0]._module, context)
