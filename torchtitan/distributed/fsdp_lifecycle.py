# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast, NamedTuple

import spmd_types as spmd
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup


class _GroupPolicy(NamedTuple):
    group: FSDPParamGroup
    post_forward_mesh_info: FSDPMeshInfo | None
    reshard_after_backward: bool
    reduce_grads: bool
    all_reduce_grads: bool


class _ChunkedLossHooks(NamedTuple):
    before_last_chunk: Callable[[], None]
    after_forward: Callable[[Any], Any]


def complete_fsdp_grouped_forward(module: FSDPModule, output: Any) -> Any:
    """Finish a partial FSDP grouped forward, retaining its output hooks."""
    state = module._get_fsdp_state()
    if state._modules_to_run_forward:
        output = state._post_forward(cast(nn.Module, module), None, output)
        state._modules_to_run_forward.clear()
    return output


@contextmanager
def chunked_loss_fsdp(
    lm_head: nn.Module, *, run_backward: bool, decoder_requires_grad: bool
) -> Iterator[_ChunkedLossHooks | None]:
    """Isolate head-only backwards and restore the caller's FSDP policies.

    The final head chunk reduces with the caller's policy unless earlier full
    or HSDP partial gradients require coalescing with the decoder backward.
    Manual finalization retains full gradients for deferred replica reduction.
    The decoder owns automatic finalization whenever its backward still follows.
    """
    if not isinstance(lm_head, FSDPModule):
        yield None
        return

    # Public setters do not expose policy snapshots, and the sync setter also
    # overwrites HSDP's independently controlled replica all-reduce policy.
    state = lm_head._get_fsdp_state()
    state._lazy_init()
    state_ctx = state._state_ctx
    is_last_backward = state_ctx.is_last_backward
    auto_reshard_after_forward = state._auto_reshard_after_forward
    head_groups = state._fsdp_param_groups
    states = state_ctx.all_states if decoder_requires_grad else [state]
    policies = [
        _GroupPolicy(
            group,
            group.post_forward_mesh_info,
            group.reshard_after_backward,
            group.reduce_grads,
            group.all_reduce_grads,
        )
        for owner in states
        for group in owner._fsdp_param_groups
    ]
    defer_head_reduction = (
        (
            state_ctx.manual_backward_finalization
            and any(
                not policy.all_reduce_grads
                for policy in policies
                if policy.group in head_groups
            )
        )
        or decoder_requires_grad
        and any(
            policy.group._post_backward_pending
            or policy.group._partial_reduce_output is not None
            for policy in policies
        )
    )

    def before_last_chunk() -> None:
        if not defer_head_reduction:
            for policy in policies:
                if policy.group in head_groups:
                    policy.group.reduce_grads = policy.reduce_grads
                    policy.group.all_reduce_grads = policy.all_reduce_grads

    def after_forward(output: Any) -> Any:
        if run_backward and not decoder_requires_grad:
            # Frozen inputs have no input backward hook. Register the group's
            # output hook even when only the head ran in this chunk.
            return complete_fsdp_grouped_forward(cast(FSDPModule, lm_head), output)
        return output

    try:
        lm_head.set_reshard_after_forward(False, recurse=False)
        for policy in policies:
            policy.group.reshard_after_backward = False
            policy.group.reduce_grads = False
        if decoder_requires_grad:
            # A chunk's root callback also visits decoder groups. Preserve their
            # gradients, prefetched weights and forward order until backward.
            state_ctx.is_last_backward = False
        # Explicit unshard avoids leaving an eager all-gather event for the
        # next FSDP module when CUDA graph capture follows the warmup.
        with spmd.no_typecheck():
            lm_head.unshard()
        yield _ChunkedLossHooks(before_last_chunk, after_forward)
    finally:
        state_ctx.is_last_backward = is_last_backward
        state._auto_reshard_after_forward = auto_reshard_after_forward
        for policy in policies:
            group = policy.group
            group.post_forward_mesh_info = policy.post_forward_mesh_info
            group.reshard_after_backward = policy.reshard_after_backward
            group.reduce_grads = policy.reduce_grads
            group.all_reduce_grads = policy.all_reduce_grads
        for policy in policies:
            if policy.group in head_groups and (
                policy.reshard_after_backward or not run_backward
            ):
                # Reshard releases weights; the autograd leaf retains gradients
                # until decoder backward or caller finalization reduces them.
                policy.group.reshard()
