# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Named ACable regions (out-of-tree) on dynamo_bypassing_wrapper.

Lets you NAME a region and control save/recompute of that whole region by name
-- robust to composite ops (e.g. nn.Linear on a 3-D input) that decompose to
view/addmm/view, which a per-tensor name() approach cannot save (the name lands
on a view).

    def block(x):
        y = unit(attn, x, name="attn")   # a named region; policy decides its fate
        return mlp(y)

    out = checkpoint(block, x, policy={"attn": CheckpointPolicy.MUST_SAVE})

The region body can be a plain composite (e.g. nn.Linear) or a custom
autograd.Function -- the latter is the cleanest, since it is already an atomic
autograd node whose backward needs are defined by ctx.save_for_backward.

Two backends behind one surface:
  - compile: the region body inlines during AOT (Inductor still fuses); we tag
    every decomposed node with the region's policy. The partitioner ignores views
    for free (inf save-weight), so only real compute nodes are saved.
  - eager: a plain use_reentrant=False checkpoint recomputes by default. To SAVE a
    region we (a) shadow the checkpoint's pack/unpack with identity hooks while
    running the body on the FORWARD pass (so the region's internal saves bypass the
    checkpoint frame), and (b) on the RECOMPUTE pass skip the body via if/else and
    return the stashed output.

Regions must be disjoint -- a unit must not contain another unit. The eager replay
log is consumed in unit-entry order on recompute but written in unit-completion
order on forward (an inner unit completes before its enclosing one), so a nested
outer unit would be skipped on recompute and hand back the inner unit's stash,
silently corrupting any downstream recomputed op. The wiring in
activation_checkpoint.py enforces this when it installs units.

Both paths are verified on full transformer blocks (attention + SDPA + FFN,
multiple saved regions per block, multiple checkpointed blocks): eager forward and
grads match a no-AC run, and saved regions are not recomputed in backward; compile
saves the same regions via the partitioner.

Source: P2376565922.
"""

import functools

import torch
from torch.utils.checkpoint import checkpoint as _checkpoint, CheckpointPolicy

_dbw = torch.ops.higher_order.dynamo_bypassing_wrapper


# ===========================================================================
# region policy plumbing
# ===========================================================================
# The name->policy dict is made ambient for the duration of the checkpointed
# function so the nested unit() calls can resolve their region by name. unit()
# reads it in *traced* code, so under compile Dynamo guards the read and a policy
# change recompiles instead of silently reusing a stale graph.
_REGION_POLICY: dict = {}


# DEVIATION FROM P2376565922: the paste keeps the eager forward/recompute replay
# log in single module globals, which only supports one checkpointed region.
# torchtitan checkpoints every transformer block as its own checkpoint() call, so
# the log must be per-invocation. We give each checkpoint() call an _EagerState and
# install it (via a stack) while its fn runs -- on both the forward and the
# recompute pass, since the recompute re-invokes the same wrapped fn. The compile
# path never touches this state (it tags fx nodes instead), so it is unchanged.
class _EagerState:
    def __init__(self):
        self.log: list = []  # per-forward (is_save, stashed_output)
        self.recomp_gid = None
        self.idx = 0


_eager_state_stack: list = []


def checkpoint(fn, *args, policy=None, use_reentrant=False, **kwargs):
    """use_reentrant=False checkpoint with a name-keyed region `policy`."""
    if policy is None:
        return _checkpoint(fn, *args, use_reentrant=use_reentrant, **kwargs)
    global _REGION_POLICY
    old = _REGION_POLICY
    _REGION_POLICY = dict(policy)
    state = _EagerState()

    def wrapped(*a, **k):
        # Compile folds is_compiling() to True and prunes this block, so the
        # eager state machinery never enters the traced graph.
        if torch.compiler.is_compiling():
            return fn(*a, **k)
        gid = torch._C._current_graph_task_id()
        if gid == -1:
            state.log.clear()  # forward pass
        elif state.recomp_gid != gid:
            state.recomp_gid = gid
            state.idx = 0  # recompute pass start
        _eager_state_stack.append(state)
        try:
            return fn(*a, **k)
        finally:
            _eager_state_stack.pop()

    try:
        return _checkpoint(wrapped, *args, use_reentrant=use_reentrant, **kwargs)
    finally:
        _REGION_POLICY = old


def _is_save(pol):
    return pol in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE)


class _identity_hooks(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self):
        super().__init__(lambda x: x, lambda x: x)


def _unit_impl(pol, inner_fn):
    # Bypassed by Dynamo; runs at AOT trace time (compile) or eagerly. `pol` is
    # resolved in unit() (traced code) so Dynamo guards it -> a policy change
    # recompiles instead of silently reusing a stale graph.
    def run(*args, **kwargs):
        from torch.fx.experimental.proxy_tensor import get_proxy_mode

        pmode = get_proxy_mode()
        if pmode is not None:
            # compile, proxy-tracing pass: tag every node the body decomposes into
            graph = pmode.tracer.graph
            before = len(graph.nodes)
            out = inner_fn(*args, **kwargs)
            if pol is not None:
                for n in list(graph.nodes)[before:]:
                    if n.op == "call_function":
                        n.meta["recompute"] = pol
            return out

        if torch.compiler.is_compiling():
            # other AOT pass (e.g. metadata collection): no graph to tag, just run
            return inner_fn(*args, **kwargs)

        # eager
        state = _eager_state_stack[-1]
        gid = torch._C._current_graph_task_id()
        if gid != -1:
            # recompute pass: replay the forward decision from the log
            is_save, val = state.log[state.idx]
            state.idx += 1
            if is_save:
                return val  # skip body, return stashed output
            return inner_fn(*args, **kwargs)

        # forward pass
        if _is_save(pol):
            with _identity_hooks():
                out = inner_fn(*args, **kwargs)
            state.log.append((True, out))
            return out
        out = inner_fn(*args, **kwargs)
        state.log.append((False, None))
        return out

    return run


def unit(fn, *args, name, **kwargs):
    # Resolve the policy here (traced under compile) so Dynamo guards the read
    # and recompiles when the policy passed to checkpoint() changes.
    pol = _REGION_POLICY.get(name)
    if pol is None and not _eager_state_stack:
        # Truly outside any region checkpoint: pass through with no HOP so unit()
        # is free to sit in model forwards unconditionally. We must NOT short-
        # circuit merely because pol is None: the eager recompute pass runs during
        # backward, after checkpoint() has torn down _REGION_POLICY, so pol is None
        # there even for saved regions. While a region checkpoint is live (its
        # _EagerState is on the stack, on both the forward and recompute passes) we
        # always route through the HOP so run() can replay the per-invocation log
        # and keep the forward/recompute pack order identical. Dynamo guards the
        # dict miss, so adding a policy for `name` recompiles.
        return fn(*args, **kwargs)
    return _dbw(functools.partial(_unit_impl, pol), fn, *args, **kwargs)
