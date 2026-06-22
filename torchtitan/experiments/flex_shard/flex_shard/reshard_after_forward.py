# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

import torch
import torch.nn as nn
from torch.utils._python_dispatch import TorchDispatchMode

from .bucket_storage import ShardedBucketStorage
from .utils import (
    _module_path_common_prefix,
    _strip_checkpoint_wrapped_module_path,
    _top_level_owner_path,
)
from .unsharded_param_getters import flex_shard_raf_saved_tensors


_RAF_SAVED_TENSOR_HOOKS_INSTALLED_ATTR = "_flex_shard_raf_saved_tensor_hooks_installed"


class _ReshardAfterForwardRecomputeState:
    """Per-flex_shard dynamic state for reshard-after-forward recompute."""

    def __init__(self) -> None:
        self._bucket_ids: ContextVar[frozenset[int]] = ContextVar(
            "_flex_shard_reshard_after_forward_recompute_bucket_ids",
            default=frozenset(),
        )

    @torch.compiler.assume_constant_result
    def is_recomputing(self, bucket_id: int) -> bool:
        """Return whether this bucket is in reshard-after-forward recompute."""
        return bucket_id in self._bucket_ids.get()

    def enter_recompute(self, recompute_bucket_ids: frozenset[int]) -> Any:
        active_bucket_ids = self._bucket_ids.get()
        return self._bucket_ids.set(active_bucket_ids | recompute_bucket_ids)

    def exit_recompute(self, token: Any) -> None:
        self._bucket_ids.reset(token)


# These produce the unsharded param tensors that we want freed per-layer.
_FLEX_SHARD_COLLECTIVE_OPS = {
    torch.ops._c10d_functional.all_gather_into_tensor.default,
    torch.ops._c10d_functional.wait_tensor.default,
    # Eager path (legacy in-place collectives).
    torch.ops.c10d._allgather_base_.default,
    torch.ops.c10d.allgather_.default,
    torch.ops.c10d.broadcast_.default,
}

# Non-FlexShard collectives inside a RAF-wrapped module should not be replayed
# just because parameter unshards are recomputed. In DeepSeek V3 MoE layers,
# replaying token-dispatch all-to-all creates extra backward-window A2A launches
# versus FSDP, which only re-unshards parameters.
_FLEX_SHARD_MUST_SAVE_OPS = {
    torch.ops._c10d_functional.all_to_all_single.default,
    torch.ops.c10d.alltoall_base_.default,
    torch.ops.c10d.alltoall_.default,
}

_FLEX_SHARD_PREFER_SAVE_OPS = {
    torch.ops.aten._fused_rms_norm.default,
    torch.ops.aten._grouped_mm.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten.add.Tensor,
    torch.ops.aten.bmm.default,
    torch.ops.aten.linear.default,
    torch.ops.aten.matmul.default,
    torch.ops.aten.mm.default,
    torch.ops.aten.mul.Tensor,
    torch.ops.aten.rms_norm.default,
    torch.ops.aten.scaled_dot_product_attention.default,
    torch.ops.aten.silu.default,
}


def _is_mutating_op(func: Any) -> bool:
    schema = getattr(func, "_schema", None)
    if schema is None:
        return False
    for argument in schema.arguments:
        alias_info = argument.alias_info
        if alias_info is not None and alias_info.is_write:
            return True
    return False


def _apply_reshard_after_forward(
    module: nn.Module,
    reshard_bucket_storages: list[ShardedBucketStorage],
) -> None:
    """Wrap FlexShard-managed bucket modules for reshard-after-forward.

    Each selected bucket's owning module gets wrapped with an activation
    recompute policy that marks collective ops (all-gather, broadcast,
    wait_tensor) as MUST_RECOMPUTE so unsharded params are freed after each
    layer's forward.

    Composes with activation checkpointing: if a child is already wrapped
    by AC's CheckpointWrapper, the two policies are merged into a single
    wrapper (FlexShard collectives -> MUST_RECOMPUTE, token-dispatch
    collectives -> MUST_SAVE, selected compute-heavy ops -> PREFER_SAVE unless
    an existing activation-checkpointing policy is present).
    """
    recompute_state = _ReshardAfterForwardRecomputeState()
    bucket_ids_by_path: dict[str, set[int]] = {}
    child_paths = [name for name, _ in module.named_children()]
    for bucket_storage in reshard_bucket_storages:
        bucket_storage._reshard_after_forward_recompute_state = recompute_state
        bucket_storage_paths = _get_module_paths_to_wrap(bucket_storage)
        if not bucket_storage_paths:
            bucket_storage_paths = child_paths
        for path in bucket_storage_paths:
            bucket_ids_by_path.setdefault(path, set()).add(id(bucket_storage))

    for path in sorted(bucket_ids_by_path, key=lambda p: (p.count("."), p)):
        child = _get_module_by_path(module, path)
        _set_module_by_path(
            module,
            path,
            _wrap_module(
                child,
                recompute_state,
                frozenset(bucket_ids_by_path[path]),
            ),
        )


def _reshard_after_forward_policy(ctx, func, *args, **kwargs):
    """Activation recompute policy for per-layer reshard-after-forward.

    Marks FlexShard unshard collectives (all-gather, broadcast, wait_tensor)
    for recomputation; activation checkpointing discards their outputs after
    each layer's forward. MoE token-dispatch all-to-all outputs are saved so
    RAF recompute does not replay non-FlexShard communication.
    """
    from torch.utils.checkpoint import CheckpointPolicy

    if func in _FLEX_SHARD_COLLECTIVE_OPS:
        return CheckpointPolicy.MUST_RECOMPUTE
    if func in _FLEX_SHARD_MUST_SAVE_OPS:
        return CheckpointPolicy.MUST_SAVE
    if _is_mutating_op(func):
        return CheckpointPolicy.PREFER_RECOMPUTE
    if func in _FLEX_SHARD_PREFER_SAVE_OPS:
        return CheckpointPolicy.PREFER_SAVE
    # RAF-only should replay parameter unshards, not the wrapped module's
    # compute-heavy ops. Factory and indexing ops are still recomputed since
    # selective checkpointing rejects cached tensors that are mutated. If
    # activation checkpointing is already present, the composed policy below
    # delegates non-FlexShard ops to that original AC policy.
    return CheckpointPolicy.PREFER_RECOMPUTE


def _compose_with_ac_policy(
    ac_context_fn,
    recompute_state: _ReshardAfterForwardRecomputeState,
    recompute_bucket_ids: frozenset[int],
):
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
                if func in _FLEX_SHARD_MUST_SAVE_OPS:
                    return CheckpointPolicy.MUST_SAVE
                return _orig(sctx, func, *args, **kwargs)

            ctx.policy_fn = merged_policy
        forward_ctx, recompute_ctx = contexts
        return forward_ctx, _MarkRecomputeTorchDispatchMode(
            recompute_ctx,
            recompute_state,
            recompute_bucket_ids,
        )

    return merged_context_fn


def _make_reshard_only_context_fn(
    recompute_state: _ReshardAfterForwardRecomputeState,
    recompute_bucket_ids: frozenset[int],
):
    def reshard_only_context_fn():
        from torch.utils.checkpoint import create_selective_checkpoint_contexts

        forward_ctx, recompute_ctx = create_selective_checkpoint_contexts(
            _reshard_after_forward_policy
        )
        return forward_ctx, _MarkRecomputeTorchDispatchMode(
            recompute_ctx,
            recompute_state,
            recompute_bucket_ids,
        )

    return reshard_only_context_fn


class _MarkRecomputeTorchDispatchMode(TorchDispatchMode):
    """Context wrapper that marks bucket-specific recomputation during tracing."""

    supports_higher_order_operators = True

    @classmethod
    def ignore_compile_internals(cls) -> bool:
        return True

    def __init__(
        self,
        ctx: Any,
        recompute_state: _ReshardAfterForwardRecomputeState,
        recompute_bucket_ids: frozenset[int],
    ) -> None:
        super().__init__()
        self.ctx = ctx
        self.recompute_state = recompute_state
        self.recompute_bucket_ids = recompute_bucket_ids
        self._token: Any | None = None

    def __enter__(self) -> _MarkRecomputeTorchDispatchMode:
        self.ctx.__enter__()
        self._token = self.recompute_state.enter_recompute(self.recompute_bucket_ids)
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        suppress = False
        try:
            suppress = bool(super().__exit__(exc_type, exc_val, exc_tb))
        finally:
            token = self._token
            self._token = None
            if token is not None:
                self.recompute_state.exit_recompute(token)
            suppress = bool(self.ctx.__exit__(exc_type, exc_val, exc_tb)) or suppress
        return suppress

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        return func(*args, **kwargs)


def _wrap_module(
    child: nn.Module,
    recompute_state: _ReshardAfterForwardRecomputeState,
    recompute_bucket_ids: frozenset[int],
) -> nn.Module:
    """Wrap a module to implement reshard-after-forward.

    If the child is already wrapped by AC's CheckpointWrapper, unwraps it,
    merges the AC policy with FlexShard's reshard policy, and re-wraps once.
    If no AC wrapper exists, use saved-tensor hooks so backward replays only
    FlexShard parameter unshards instead of the whole module op stream.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointWrapper,
    )

    if isinstance(child, CheckpointWrapper):
        # AC already applied: unwrap, merge policies, re-wrap.
        inner = child._checkpoint_wrapped_module
        ac_kwargs = dict(child.checkpoint_fn.keywords)
        ac_kwargs.pop("use_reentrant", None)
        ac_context_fn = ac_kwargs.pop("context_fn", None)
        if ac_context_fn is not None:
            # Selective AC: merge with reshard policy.
            merged_fn = _compose_with_ac_policy(
                ac_context_fn,
                recompute_state,
                recompute_bucket_ids,
            )
        else:
            # Full AC: add reshard policy via selective context.
            merged_fn = _make_reshard_only_context_fn(
                recompute_state,
                recompute_bucket_ids,
            )
        return checkpoint_wrapper(inner, context_fn=merged_fn, **ac_kwargs)

    _install_raf_saved_tensor_hooks(child)
    return child


def _install_raf_saved_tensor_hooks(child: nn.Module) -> None:
    """Install saved-tensor hooks around one RAF-only module forward."""
    if getattr(child, _RAF_SAVED_TENSOR_HOOKS_INSTALLED_ATTR, False):
        return

    active_contexts: list[Any] = []

    def pre_forward_hook(module, args):
        _ = module, args
        context = flex_shard_raf_saved_tensors()
        context.__enter__()
        active_contexts.append(context)

    def post_forward_hook(module, args, output):
        _ = module, args
        context = active_contexts.pop()
        context.__exit__(None, None, None)
        return output

    child.register_forward_pre_hook(pre_forward_hook)
    child.register_forward_hook(post_forward_hook, always_call=True)
    setattr(child, _RAF_SAVED_TENSOR_HOOKS_INSTALLED_ATTR, True)


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


def _get_module_paths_to_wrap(bucket_storage: ShardedBucketStorage) -> list[str]:
    """Return module paths to wrap for one reshard-after-forward bucket."""
    owner_paths = sorted(
        {
            _strip_checkpoint_wrapped_module_path(".".join(fqn.split(".")[:-1]))
            for fqn in bucket_storage._param_infos
            if "." in fqn
        }
    )
    if not owner_paths:
        return []

    common = _module_path_common_prefix(owner_paths)
    if common:
        top_level_paths = sorted(
            {
                _top_level_owner_path(bucket_storage._module, owner_path)
                for owner_path in owner_paths
            }
        )
        top_level_paths = [path for path in top_level_paths if path]
        if len(top_level_paths) == 1 and top_level_paths[0] != common:
            return top_level_paths
        target = _get_module_by_path(bucket_storage._module, common)
        if isinstance(target, (nn.ModuleDict, nn.ModuleList)):
            return sorted(
                {
                    _top_level_owner_path(bucket_storage._module, owner_path)
                    for owner_path in owner_paths
                }
            )
        return [common]
    return sorted(
        {
            _top_level_owner_path(bucket_storage._module, owner_path)
            for owner_path in owner_paths
        }
    )
