# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch.nn as nn

from .bucket_storage import DStorage
from .reshard_provenance import _maybe_mark_policy_outputs, _with_flex_shard_provenance
from .utils import (
    _module_path_common_prefix,
    _strip_checkpoint_wrapped_module_path,
    _top_level_owner_path,
)


class _ReshardAfterForwardRecomputeState:
    """Per-flex_shard dynamic state for reshard-after-forward recompute."""

    def __init__(self) -> None:
        self._bucket_ids: ContextVar[frozenset[int]] = ContextVar(
            "_flex_shard_reshard_after_forward_recompute_bucket_ids",
            default=frozenset(),
        )

    def is_recomputing(self, bucket_id: int) -> bool:
        """Return whether this bucket is in reshard-after-forward recompute."""
        return bucket_id in self._bucket_ids.get()

    def enter_recompute(self, recompute_bucket_ids: frozenset[int]) -> Any:
        active_bucket_ids = self._bucket_ids.get()
        return self._bucket_ids.set(active_bucket_ids | recompute_bucket_ids)

    def exit_recompute(self, token: Any) -> None:
        self._bucket_ids.reset(token)


@contextmanager
def _mark_recompute(
    ctx: Any,
    recompute_state: _ReshardAfterForwardRecomputeState,
    recompute_bucket_ids: frozenset[int],
) -> Generator[None, None, None]:
    """Mark bucket-specific FlexShard reshard-after-forward recomputation."""
    token = recompute_state.enter_recompute(recompute_bucket_ids)
    try:
        with ctx:
            yield
    finally:
        recompute_state.exit_recompute(token)


def _reshard_after_forward_policy(ctx, func, *args, **kwargs):
    """Activation recompute policy for per-layer reshard-after-forward.

    Only FlexShard-derived tensors are forced to recompute: bucket all-gather
    full params and their alias/view outputs such as transposes. Other ops,
    including non-FlexShard collectives, use the normal checkpoint policy.
    """
    from torch.utils.checkpoint import CheckpointPolicy

    if _maybe_mark_policy_outputs(ctx, func, args, kwargs):
        return CheckpointPolicy.MUST_RECOMPUTE
    # PREFER_RECOMPUTE lets checkpoint decide what to save vs recompute
    # for non-collective ops, matching standard AC behavior.
    return CheckpointPolicy.PREFER_RECOMPUTE


def _compose_with_ac_policy(
    ac_context_fn,
    recompute_state: _ReshardAfterForwardRecomputeState,
    recompute_bucket_ids: frozenset[int],
):
    """Compose FlexShard reshard policy with an existing AC context_fn.

    Returns a new context_fn that wraps the AC policy: FlexShard-derived
    full params and their aliases are forced to MUST_RECOMPUTE, everything
    else delegates to the original AC policy.
    """

    def merged_context_fn():
        from torch.utils.checkpoint import CheckpointPolicy

        contexts = ac_context_fn()
        for ctx in contexts:
            original_policy = getattr(ctx, "policy_fn", None)
            if original_policy is None:
                continue

            def merged_policy(sctx, func, *args, _orig=original_policy, **kwargs):
                if _maybe_mark_policy_outputs(sctx, func, args, kwargs):
                    return CheckpointPolicy.MUST_RECOMPUTE
                return _orig(sctx, func, *args, **kwargs)

            ctx.policy_fn = merged_policy
        forward_ctx, recompute_ctx = contexts
        return _with_flex_shard_provenance(forward_ctx), _mark_recompute(
            _with_flex_shard_provenance(recompute_ctx),
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
        return _with_flex_shard_provenance(forward_ctx), _mark_recompute(
            _with_flex_shard_provenance(recompute_ctx),
            recompute_state,
            recompute_bucket_ids,
        )

    return reshard_only_context_fn


def _wrap_module(
    child: nn.Module,
    recompute_state: _ReshardAfterForwardRecomputeState,
    recompute_bucket_ids: frozenset[int],
) -> nn.Module:
    """Wrap a module to implement reshard-after-forward via activation recompute.

    If the child is already wrapped by AC's CheckpointWrapper, unwraps it,
    merges the AC policy with FlexShard's reshard policy, and re-wraps once.
    If no AC wrapper exists, wraps with a reshard-after-forward-only policy.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointWrapper,
    )

    if isinstance(child, CheckpointWrapper):
        # AC already applied — unwrap, merge policies, re-wrap
        inner = child._checkpoint_wrapped_module
        ac_kwargs = dict(child.checkpoint_fn.keywords)
        ac_kwargs.pop("use_reentrant", None)
        ac_context_fn = ac_kwargs.pop("context_fn", None)
        if ac_context_fn is not None:
            # Selective AC — merge with reshard policy
            merged_fn = _compose_with_ac_policy(
                ac_context_fn,
                recompute_state,
                recompute_bucket_ids,
            )
        else:
            # Full AC — add reshard policy via selective context
            merged_fn = _make_reshard_only_context_fn(
                recompute_state,
                recompute_bucket_ids,
            )
        return checkpoint_wrapper(inner, context_fn=merged_fn, **ac_kwargs)

    # No AC — reshard-only wrapping
    return checkpoint_wrapper(
        child,
        context_fn=_make_reshard_only_context_fn(
            recompute_state,
            recompute_bucket_ids,
        ),
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


def _get_module_paths_to_wrap(storage: DStorage) -> list[str]:
    """Return module paths to wrap for one reshard-after-forward bucket."""
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
        top_level_paths = sorted(
            {
                _top_level_owner_path(storage._module, owner_path)
                for owner_path in owner_paths
            }
        )
        top_level_paths = [path for path in top_level_paths if path]
        if len(top_level_paths) == 1 and top_level_paths[0] != common:
            return top_level_paths
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


def _apply_reshard_after_forward(
    module: nn.Module,
    reshard_storages: list[DStorage],
) -> None:
    """Wrap FlexShard-managed bucket modules for reshard-after-forward.

    Each selected bucket's owning module gets wrapped with an activation
    recompute policy that marks FlexShard-derived all-gather full params and
    their aliases as MUST_RECOMPUTE so unsharded params are freed after each
    layer's forward.

    Composes with activation checkpointing: if a child is already wrapped
    by AC's CheckpointWrapper, the two policies are merged into a single
    wrapper (FlexShard-derived tensors → MUST_RECOMPUTE, AC compute ops →
    MUST_SAVE, everything else → PREFER_RECOMPUTE).
    """
    recompute_state = _ReshardAfterForwardRecomputeState()
    bucket_ids_by_path: dict[str, set[int]] = {}
    child_paths = [name for name, _ in module.named_children()]
    for storage in reshard_storages:
        storage._reshard_after_forward_recompute_state = recompute_state
        storage_paths = _get_module_paths_to_wrap(storage)
        if not storage_paths:
            storage_paths = child_paths
        for path in storage_paths:
            bucket_ids_by_path.setdefault(path, set()).add(id(storage))

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
