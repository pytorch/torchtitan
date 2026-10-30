# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`torch._dynamo.maybe_mark_dynamic` is @forbid_in_graph by design (_dynamo/decorators.py:1347), so
a helper that calls it fails when the helper itself is compiled (block compile, GraphTrainer).
Guarding the call with `torch.compiler.is_compiling()` traces fine (Dynamo folds the branch).

Expected on 1x GB300 (torch 2.15.0.dev20260926+cu130, triton 3.8.0; see ../README.md, Known behaviors):
unguarded: AssertionError 'Attempt to trace forbidden callable'; guarded by is_compiling(): compiles.
"""
import torch


def wrapper(x):
    torch._dynamo.maybe_mark_dynamic(x, 0)
    return x * 2


def guarded_wrapper(x):
    if not torch.compiler.is_compiling():
        torch._dynamo.maybe_mark_dynamic(x, 0)
    return x * 2


for name, fn in (
    ("unguarded", wrapper),
    ("guarded by is_compiling()", guarded_wrapper),
):
    try:
        torch.compile(fn, fullgraph=True)(torch.randn(4, device="cuda"))
        print(f"{name}: compiled fine")
    except Exception as e:
        print(f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:100]}")
