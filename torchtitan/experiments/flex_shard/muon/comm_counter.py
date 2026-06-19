# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Process-global communication-byte counter for the distributed-Muon benchmark.

Monkeypatches two disjoint families of collectives so each call adds this rank's
**send volume** (the input buffer's bytes) to a global accumulator:

1. Eager ``torch.distributed`` collectives (``_VOLUME_ARG``) -- the all-gather /
   reduce-scatter / broadcast / reduce variants the FlexShard engine and the raw
   collective calls use.
2. Functional collectives (``torch.ops._c10d_functional.*``, ``_FUNCTIONAL_COLLECTIVES``)
   -- the ops core ``fully_shard`` (FSDP2) and **DTensor redistribute / full_tensor**
   dispatch to. These do NOT route through the eager ``torch.distributed`` python
   symbols above (they call straight into the c10d functional ops), so patching only
   family (1) would register **zero** bytes for the DTensor / FSDP2 baselines -- e.g.
   :class:`DTensorMuon`'s in-step ``update.full_tensor()`` all-gather -- making them
   look as collective-free as comm-efficient ``Owned`` Muon. The two families dispatch
   through separate namespaces (``c10d`` vs ``_c10d_functional``), so patching both does
   not double-count. The op packet is the single chokepoint every functional-collective
   python wrapper (eager, autograd, coalesced) funnels through, so patching it there --
   rather than the many wrappers -- catches all paths and adds zero overhead to
   non-collective ops (unlike a process-wide ``TorchDispatchMode``).

Reading the accumulator before/after a region attributes comm bytes to that region
(e.g. the optimizer step vs the rest of the iteration); see ``_BenchMixin`` in ``bench.py``.

Counting this rank's send bytes (input tensor) gives a consistent per-rank metric
across collectives: for ``all_gather`` / ``all_gather_single`` / ``reduce_scatter``
that is the local contribution; for ``broadcast`` / ``reduce`` it is the buffer the
root sends / each rank contributes. Absolute numbers are approximate (one rank, send
side only) but the cross-config comparison -- the point of the benchmark -- is clean.

Limitation: under ``torch.compile`` the collectives inside a compiled region are
captured by Dynamo and do not call these python wrappers, so they are not counted.
The metric the benchmark turns on -- ``step_comm``, dominated by the eager optimizer
step (NS gather for the gather baselines, ``full_tensor()`` for DTensorMuon) -- runs
outside compiled regions and is counted.

This lives entirely in experiment code and patches only ``torch.distributed`` symbols
and the ``_c10d_functional`` op packets; it does not modify the FlexShard engine or any
core torchtitan code. ``install()`` is idempotent and a no-op until called, so importing
this module changes nothing.
"""

from __future__ import annotations

import torch
import torch.distributed as dist


# (positional index, keyword name) of the input/volume tensor for each collective.
_VOLUME_ARG = {
    "all_gather": (1, "tensor"),
    "all_gather_single": (1, "input"),
    "reduce_scatter_tensor": (1, "input"),
    "broadcast": (0, "tensor"),
    "reduce": (0, "tensor"),
    # FSDP2 (fully_shard) collectives, so the same counter works for the
    # vanilla-FSDP2 + AdamW reference baseline.
    "all_gather_into_tensor": (1, "input_tensor"),
    "_all_gather_base": (1, "input"),
    "_reduce_scatter_base": (1, "input"),
}

# Functional collectives on ``torch.ops._c10d_functional``. Every one takes the local
# send/input tensor as the first positional arg (coalesced variants pass a list of them),
# so a single args[0] rule covers all of them.
_FUNCTIONAL_COLLECTIVES = (
    "all_gather_into_tensor",
    "all_gather_into_tensor_coalesced",
    "reduce_scatter_tensor",
    "reduce_scatter_tensor_coalesced",
    "all_reduce",
    "all_reduce_coalesced",
    "broadcast",
    "all_to_all_single",
)

_state = {"bytes": 0, "installed": False}


def reset() -> None:
    """Zero the accumulated byte count."""
    _state["bytes"] = 0


def read() -> int:
    """Return the bytes accumulated on this rank since the last ``reset()``."""
    return _state["bytes"]


def _volume_bytes(name: str, args: tuple, kwargs: dict) -> int:
    idx, key = _VOLUME_ARG[name]
    tensor = kwargs.get(key)
    if tensor is None and len(args) > idx:
        tensor = args[idx]
    if isinstance(tensor, torch.Tensor):
        return tensor.numel() * tensor.element_size()
    return 0


def _wrap(name: str, fn):
    def wrapped(*args, **kwargs):
        # Count before the call; a failed collective would raise anyway.
        _state["bytes"] += _volume_bytes(name, args, kwargs)
        return fn(*args, **kwargs)

    return wrapped


def _functional_volume_bytes(args: tuple) -> int:
    """Send bytes for a ``_c10d_functional`` op: its first positional arg (a tensor, or
    a list of tensors for the coalesced variants)."""
    tensor = args[0] if args else None
    if isinstance(tensor, (list, tuple)):
        return sum(
            t.numel() * t.element_size() for t in tensor if isinstance(t, torch.Tensor)
        )
    if isinstance(tensor, torch.Tensor):
        return tensor.numel() * tensor.element_size()
    return 0


def _wrap_functional(fn):
    def wrapped(*args, **kwargs):
        _state["bytes"] += _functional_volume_bytes(args)
        return fn(*args, **kwargs)

    return wrapped


def install() -> None:
    """Patch the eager + functional collectives to accumulate send bytes (idempotent)."""
    if _state["installed"]:
        return
    for name in _VOLUME_ARG:
        fn = getattr(dist, name, None)
        if fn is not None:
            setattr(dist, name, _wrap(name, fn))
    # Functional collectives: wrap the op packet on the namespace (the chokepoint all
    # python wrappers funnel through). Touch the attribute first so it is materialized and
    # cached on the namespace before we overwrite it.
    functional = torch.ops._c10d_functional
    for name in _FUNCTIONAL_COLLECTIVES:
        fn = getattr(functional, name, None)
        if fn is not None:
            setattr(functional, name, _wrap_functional(fn))
    _state["installed"] = True
