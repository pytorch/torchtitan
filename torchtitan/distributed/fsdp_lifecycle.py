# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Runtime FSDP lifecycle helpers that isolate private upstream state."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any, cast, NamedTuple

import spmd_types as spmd
import torch.nn as nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_common import FSDPMeshInfo
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

from torchtitan.distributed.flex_shard_fsdp import (
    flex_shard_partial_backwards,
    is_flex_shard_module,
)


class _GroupPolicy(NamedTuple):
    group: FSDPParamGroup
    post_forward_mesh_info: FSDPMeshInfo | None
    reshard_after_backward: bool
    reduce_grads: bool
    all_reduce_grads: bool


class _PartialBackwardHooks(NamedTuple):
    before_last_forward: Callable[[], None]
    after_forward: Callable[[Any], Any]


def complete_fsdp_grouped_forward(module: FSDPModule, output: Any) -> Any:
    """Finish a partial FSDP grouped forward, retaining its output hooks."""
    # TODO: Use public partial-group forward completion for repeated standalone
    # calls. https://github.com/pytorch/pytorch/issues/203176
    state = module._get_fsdp_state()
    if state._modules_to_run_forward:
        output = state._post_forward(cast(nn.Module, module), None, output)
        state._modules_to_run_forward.clear()
    return output


def fsdp_partial_backwards(
    module: nn.Module,
    *,
    fsdp_root: nn.Module | None,
    run_backward: bool,
    outer_backward_follows: bool,
) -> AbstractContextManager[_PartialBackwardHooks]:
    """Keep one module unsharded across partial backwards, preserving policy.

    Only ``module`` runs in the inner forwards and backwards. Call
    ``before_last_forward`` before its last forward and ``after_forward``
    on each output. An outer backward, when present, follows all inner passes
    and owns root finalization. Without an outer backward, the caller's
    finalization mode remains in effect.

    The final inner backward uses the caller's reduction policy unless pending
    full or HSDP partial gradients must coalesce with the outer backward.
    Manual finalization owns all reductions, including the first inner pass,
    and leaves parameter residency to the schedule.
    Forward-only calls complete the same scope without running backward.

    ``fsdp_root`` is the module that FSDP sharded and that holds ``module``.
    FSDP2 finds ``module``'s group from ``module``; FlexShard finds its bucket
    from ``fsdp_root`` (see ``flex_shard_partial_backwards``).
    """
    if is_flex_shard_module(fsdp_root):
        return flex_shard_partial_backwards(
            module,
            flex_shard_module=fsdp_root,
            run_backward=run_backward,
            outer_backward_follows=outer_backward_follows,
        )
    return _fsdp2_partial_backwards(
        module, run_backward=run_backward, outer_backward_follows=outer_backward_follows
    )


@contextmanager
def _fsdp2_partial_backwards(
    module: nn.Module, *, run_backward: bool, outer_backward_follows: bool
) -> Iterator[_PartialBackwardHooks]:
    """``fsdp_partial_backwards`` for FSDP2."""
    if not isinstance(module, FSDPModule):
        yield _PartialBackwardHooks(lambda: None, lambda output: output)
        return

    # TODO: Use scoped partial backward and value-based policy restoration.
    # https://github.com/pytorch/pytorch/issues/203176
    # https://github.com/pytorch/pytorch/issues/203177
    # The public sync setter also overwrites the separate HSDP all-reduce policy.
    state = module._get_fsdp_state()
    state._lazy_init()
    state_ctx = state._state_ctx
    is_last_backward = state_ctx.is_last_backward
    auto_reshard_after_forward = state._auto_reshard_after_forward
    module_groups = state._fsdp_param_groups
    states = state_ctx.all_states if outer_backward_follows else [state]
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
    manual_finalization = state_ctx.manual_backward_finalization
    defer_reduction = (
        manual_finalization
        or outer_backward_follows
        and any(
            policy.group._post_backward_pending
            or policy.group._partial_reduce_output is not None
            for policy in policies
        )
    )

    def before_last_forward() -> None:
        if not defer_reduction:
            for policy in policies:
                if policy.group in module_groups:
                    policy.group.reduce_grads = policy.reduce_grads
                    policy.group.all_reduce_grads = policy.all_reduce_grads

    def after_forward(output: Any) -> Any:
        if run_backward and not outer_backward_follows:
            # Detached inputs have no input backward hook. Register the group's
            # output hook even when only this module ran in the inner forward.
            return complete_fsdp_grouped_forward(cast(FSDPModule, module), output)
        return output

    try:
        module.set_reshard_after_forward(False, recurse=False)
        for policy in policies:
            policy.group.reshard_after_backward = False
            policy.group.reduce_grads = False
        if outer_backward_follows:
            # Inner root callbacks also visit other groups. Preserve their
            # gradients, prefetched weights and forward order for outer backward.
            state_ctx.is_last_backward = False
        # Explicit unshard avoids leaving an eager all-gather event for the
        # next FSDP module when CUDA graph capture follows the warmup.
        with spmd.no_typecheck():
            module.unshard()
        yield _PartialBackwardHooks(before_last_forward, after_forward)
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
            if policy.group in module_groups and (
                not run_backward
                or (not manual_finalization and policy.reshard_after_backward)
            ):
                # Reshard releases weights; the autograd leaf retains gradients
                # until outer backward or caller finalization reduces them.
                policy.group.reshard()
