# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamo: storing a new attribute on an object defined outside a checkpoint HOP body fails with a
generic "Observed exception" (CPU repro).

Inside a higher-order op (here `torch.utils.checkpoint`, the `tag_activation_checkpoint` HOP), Dynamo
defers a constant attribute store on an outer-scope object and snapshots the attribute's original value
first (`snapshot_attr_mutation`, `torch/_dynamo/side_effects.py:335-366`). For a new attribute that read
raises AttributeError; only NotImplementedError is caught, so it escapes as the generic graph break
"Observed exception" (gb0088). A store to an existing attribute instead fails with the intended
"HOP: Non-nullified side effect" (`side_effects.py:368-383`). The same code compiles at top level.

Where we hit it: TorchTitan's `_spmd_mesh_stack()` (`torchtitan/distributed/spmd_types.py:91-96`)
lazily creates a list on a module-level threading.local.

Usage: python kb_hop_outer_attr_store.py
Prints one line per variant: OK, or the graph-break type and the raised exception.

Expected (CPU, torch 2.15.0.dev20260926; see ../README.md, Known behaviors): top-level and read-only
variants and the threading.local subclass print OK; every new-attribute store inside the HOP prints
"Observed exception" with AttributeError(... has no attribute ...); a non-constant value prints "HOP: Unsafe side effect"
(store_attr snapshots only constants); the existing-attribute store prints
"HOP: Non-nullified side effect"; with fullgraph=False the whole function runs eagerly
(0 frames compiled).
"""

import threading

import torch
from torch._dynamo.testing import CompileCounter
from torch.utils.checkpoint import checkpoint


class Plain:
    pass


class _DefaultTLS(threading.local):
    def __init__(self):
        super().__init__()
        self.mesh_stack = []


def lazy_init_fn(obj):
    def fn(x):
        stack = getattr(obj, "mesh_stack", None)
        if stack is None:
            stack = []
            obj.mesh_stack = stack
        return x * (2 + len(stack))

    return fn


def read_only_fn(obj):
    def fn(x):
        return x * 2 if getattr(obj, "mesh_stack", None) is None else x * 3

    return fn


def store_fn(obj, name):
    def fn(x):
        setattr(obj, name, 1)
        return x * 2

    return fn


G_TLS = threading.local()
G_PLAIN = Plain()
G_TLS_EXISTING = threading.local()
G_TLS_EXISTING.flag = 0


def g_store_new_tls(x):
    G_TLS.other = 1
    return x * 2


def g_store_new_plain(x):
    G_PLAIN.other = 1
    return x * 2


def g_store_new_tls_tensor(x):
    G_TLS.tensor = x.detach()
    return x * 2


def g_store_existing_tls(x):
    G_TLS_EXISTING.flag = 1
    return x * 2


def describe(e):
    lines = str(e).splitlines()
    raised = [
        l.split("raised exception", 1)[1].strip()
        for l in lines
        if "raised exception" in l
    ]
    return f"{type(e).__name__}: {lines[0][:40]}" + (
        f" | {raised[0][:80]}" if raised else ""
    )


def run(name, fn, in_hop=True, fullgraph=True):
    torch._dynamo.reset()
    counter = CompileCounter()

    def wrapper(x):
        y = x.sin()
        return checkpoint(fn, y, use_reentrant=False) if in_hop else fn(y)

    x = torch.randn(4, requires_grad=True)
    try:
        backend = "eager" if fullgraph else counter
        torch.compile(wrapper, fullgraph=fullgraph, backend=backend)(x).sum().backward()
        extra = f" (frames compiled: {counter.frame_count})" if not fullgraph else ""
        print(f"{name:66s} OK{extra}")
    except Exception as e:  # noqa: BLE001
        print(f"{name:66s} {describe(e)}")


def main():
    run(
        "lazy init, fresh threading.local, top level",
        lazy_init_fn(threading.local()),
        in_hop=False,
    )
    run(
        "read-only getattr default, fresh threading.local, in HOP",
        read_only_fn(threading.local()),
    )
    run(
        "lazy init, fresh threading.local (closure), in HOP",
        lazy_init_fn(threading.local()),
    )
    run("store new attr, global threading.local, in HOP", g_store_new_tls)
    run("store new attr, global plain object, in HOP", g_store_new_plain)
    run("store new attr, plain object (closure), in HOP", store_fn(Plain(), "other"))
    run("store new attr, non-constant value (tensor), in HOP", g_store_new_tls_tensor)
    run("store existing attr, global threading.local, in HOP", g_store_existing_tls)
    run(
        "lazy init, threading.local subclass with __init__ default, in HOP",
        lazy_init_fn(_DefaultTLS()),
    )
    run(
        "lazy init, fresh threading.local, in HOP, fullgraph=False",
        lazy_init_fn(threading.local()),
        fullgraph=False,
    )


if __name__ == "__main__":
    main()
