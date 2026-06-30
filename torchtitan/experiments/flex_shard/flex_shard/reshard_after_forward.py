# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from contextlib import nullcontext
from contextvars import ContextVar
from typing import Any

import torch
import torch.nn as nn

from .bucket_storage import ShardedBucketStorage
from .ops import is_unshard_bucket_op
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


def _apply_reshard_after_forward(
    module: nn.Module,
    reshard_bucket_storages: list[ShardedBucketStorage],
) -> None:
    """Wrap FlexShard-managed bucket modules for reshard-after-forward.

    Each selected bucket's owning module either gets saved-tensor hooks
    (RAF-only path) or composes with the user's activation checkpointing
    wrapper. In the AC path, FlexShard only marks its semantic unshard op as
    MUST_RECOMPUTE and leaves normal module compute policy to the user's AC.
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


def _compose_with_ac_policy(
    ac_context_fn,
    recompute_state: _ReshardAfterForwardRecomputeState,
    recompute_bucket_ids: frozenset[int],
):
    """Compose FlexShard reshard policy with an existing AC context_fn.

    Returns a new context_fn that wraps the AC policy: FlexShard semantic
    unshard ops are forced to MUST_RECOMPUTE, and everything else delegates to
    the original AC policy.
    """

    def merged_context_fn():
        from torch.utils.checkpoint import CheckpointPolicy

        contexts = ac_context_fn()
        for ctx in contexts:
            original_policy = getattr(ctx, "policy_fn", None)
            if original_policy is None:
                continue

            def merged_policy(sctx, func, *args, _orig=original_policy, **kwargs):
                if is_unshard_bucket_op(func):
                    return CheckpointPolicy.MUST_RECOMPUTE
                return _orig(sctx, func, *args, **kwargs)

            ctx.policy_fn = merged_policy
        forward_ctx, recompute_ctx = contexts
        return forward_ctx, _MarkRecomputeContext(
            recompute_ctx,
            recompute_state,
            recompute_bucket_ids,
        )

    return merged_context_fn


def _make_full_ac_recompute_context_fn(
    recompute_state: _ReshardAfterForwardRecomputeState,
    recompute_bucket_ids: frozenset[int],
):
    def full_ac_recompute_context_fn():
        return nullcontext(), _MarkRecomputeContext(
            nullcontext(),
            recompute_state,
            recompute_bucket_ids,
        )

    return full_ac_recompute_context_fn


class _MarkRecomputeContext:
    """Context wrapper that marks bucket-specific recomputation."""

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

    def __enter__(self) -> _MarkRecomputeContext:
        self.ctx.__enter__()
        self._token = self.recompute_state.enter_recompute(self.recompute_bucket_ids)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        token = self._token
        self._token = None
        if token is not None:
            self.recompute_state.exit_recompute(token)
        return self.ctx.__exit__(exc_type, exc_val, exc_tb)


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
            # Full AC: preserve full recomputation and only mark RAF replay.
            merged_fn = _make_full_ac_recompute_context_fn(
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
