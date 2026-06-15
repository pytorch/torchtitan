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
    region we (a) stash its output and skip the body on the RECOMPUTE pass; and (b)
    on the FORWARD pass replace each tensor the region saved that can be refilled
    from the recompute -- the input activations, the module parameters, and views of
    either (an N-D input is saved as a reshape view, nn.Linear saves weight.t()) --
    with a slot, freeing it now and refilling it (via as_strided for views) when the
    recompute reaches the region. A dummy tensor saved last (see _Trigger) makes the
    recompute fire before any region's backward reads a slot. Freeing parameter
    views matters under FSDP: a pinned weight.t() would hold the gathered parameter
    resident; instead it is refilled from the re-gathered parameter.

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
        self.log: list = []  # per-forward (is_save, stashed_output, slots)
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
            # Add a dummy save (frame holder) last, so its backward -- the first
            # node in the block's backward -- triggers the checkpoint's recompute
            # before any saved region reads its slots.
            return _add_trigger(fn(*a, **k))
        finally:
            _eager_state_stack.pop()

    try:
        return _checkpoint(wrapped, *args, use_reentrant=use_reentrant, **kwargs)
    finally:
        _REGION_POLICY = old


def _is_save(pol):
    return pol in (CheckpointPolicy.MUST_SAVE, CheckpointPolicy.PREFER_SAVE)


class _Slot:
    """Placeholder stored in a saved region's grad_fn in place of a saved tensor, so
    that tensor is not pinned resident after the forward. Filled from the recompute
    (run()'s recompute branch) when the block recompute reaches the region; the
    _Trigger dummy save guarantees that recompute runs before any slot is read."""

    __slots__ = ("value",)

    def __init__(self):
        self.value = None


class _Trigger(torch.autograd.Function):
    """Saves a dummy tensor as the LAST tensor of the checkpointed forward. Its
    backward is therefore the FIRST node in the block's backward; unpacking the
    dummy (a frame holder) triggers the checkpoint's own recompute before any saved
    region's backward reads its slots. Because the dummy is the last frame holder,
    early-stop cannot cut the recompute short -- it reaches every region. This needs
    nothing from the checkpoint frame's internals."""

    @staticmethod
    def forward(ctx, x, dummy):
        ctx.save_for_backward(dummy)
        return x

    @staticmethod
    def backward(ctx, grad):
        ctx.saved_tensors  # unpack -> triggers the frame recompute (fills slots)
        return grad, None


def _add_trigger(out):
    if isinstance(out, torch.Tensor):
        return _Trigger.apply(out, out.new_zeros(1))
    if isinstance(out, (tuple, list)):
        return type(out)(
            _add_trigger(o) if isinstance(o, torch.Tensor) else o for o in out
        )
    return out


def _resolve_source(x, id_to_arg, id_to_param):
    """Return how to refill a saved tensor from the recompute, or None to pin it:
    ("arg"/"param", index, view) where view is None for a direct save or
    (size, stride, storage_offset) for a view of an arg/param (e.g. a reshaped
    activation or nn.Linear's weight.t())."""
    i = id_to_arg.get(id(x))
    if i is not None:
        return ("arg", i, None)
    j = id_to_param.get(id(x))
    if j is not None:
        return ("param", j, None)
    base = x._base
    if base is not None:
        spec = (x.size(), x.stride(), x.storage_offset())
        i = id_to_arg.get(id(base))
        if i is not None:
            return ("arg", i, spec)
        j = id_to_param.get(id(base))
        if j is not None:
            return ("param", j, spec)
    return None


class _region_save_hooks(torch.autograd.graph.saved_tensors_hooks):
    """Save-tensor hooks for a MUST_SAVE region: replace every saved tensor that can
    be refilled from the recompute -- the input activations, the module parameters,
    and views of either (an N-D input is saved as a reshape view; nn.Linear saves
    weight.t()) -- with a slot (freeing it now), and pin anything else (a true
    intermediate). Each slot records an index-based source (see _resolve_source); no
    tensors or ids are retained past the forward.

    Freeing parameter views matters under FSDP: pinning a saved weight.t() would
    hold the gathered parameter resident and defeat resharding; it is refilled from
    the re-gathered parameter on recompute instead.
    """

    def __init__(self, args, module, slots):
        # ids are valid only while inputs/params are alive (this forward); used only
        # to resolve the index-based sources now, then dropped on __exit__.
        id_to_arg = {
            id(a): i for i, a in enumerate(args) if isinstance(a, torch.Tensor)
        }
        id_to_param = {}
        if isinstance(module, torch.nn.Module):
            id_to_param = {id(p): j for j, p in enumerate(module.parameters())}
        self._maps = (id_to_arg, id_to_param)

        def pack(x):
            src = _resolve_source(x, id_to_arg, id_to_param)
            if src is None:
                return x  # pin (a true intermediate, not an input or parameter)
            slot = _Slot()
            slots.append((slot, src))
            return slot

        def unpack(s):
            if type(s) is _Slot:
                if s.value is None:
                    raise RuntimeError(
                        "region SAC: a saved region's slot was read before the block "
                        "recompute filled it (or read twice, e.g. double-backward, "
                        "which region SAC does not support). The _Trigger dummy save "
                        "should make the recompute run first."
                    )
                # Release the refilled tensor as soon as the consumer's backward
                # reads it, rather than pinning it in state.log for the whole
                # block backward. Each slot is read once per backward (one saved
                # tensor -> one backward node; verified for shared inputs across
                # regions), so this frees the recomputed activation (e.g. the FFN
                # product feeding w2) immediately instead of ~100 events later.
                v = s.value
                s.value = None
                return v
            return s

        super().__init__(pack, unpack)

    def __exit__(self, *exc):
        # Sources are resolved; drop the ids so none are kept past the forward.
        for m in self._maps:
            m.clear()
        return super().__exit__(*exc)


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
            idx = state.idx
            is_save, val, slots = state.log[idx]
            state.idx += 1
            if is_save:
                # refill the saved region's slots from the recomputed sources
                params = None
                for slot, (base_kind, index, view) in slots:
                    if base_kind == "arg":
                        base = args[index]
                    else:
                        if params is None:
                            params = list(inner_fn.parameters())
                        base = params[index]
                    t = base if view is None else base.as_strided(*view)
                    slot.value = (
                        t.detach()
                        if isinstance(t, torch.Tensor) and t.requires_grad
                        else t
                    )
                # Drop state.log's reference to the stashed output now that the
                # recompute has handed it back: the recomputed graph holds `val`,
                # so pinning it in the log would keep it resident for the rest of
                # the block backward (read once per backward, like the slots).
                state.log[idx] = (True, None, slots)
                return val  # skip body, return stashed output
            return inner_fn(*args, **kwargs)

        # forward pass
        if _is_save(pol):
            slots: list = []
            with _region_save_hooks(args, inner_fn, slots):
                out = inner_fn(*args, **kwargs)
            state.log.append((True, out, slots))
            return out
        out = inner_fn(*args, **kwargs)
        state.log.append((False, None, None))
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
