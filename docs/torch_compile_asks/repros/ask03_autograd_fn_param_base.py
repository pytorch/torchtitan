# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamo + AOTAutograd fail on an autograd.Function forward that inspects the parameter behind a view.

Pattern from dist_moe (`api.py:3614` -> `_execution.py:405-411` and `:478`, at ab56bce): the forward
resolves the leaf parameter that owns ``.grad`` (``w._base``), checks ``is_leaf``/``requires_grad``, and
reads ``grad_dtype``. Dynamo graph-breaks on the ``grad_dtype`` read (it returns a dtype) and records
the ``_base`` access as graph ops; with backend aot_eager or inductor the graph then fails with
``AttributeError: 'NoneType' object has no attribute 'is_leaf'`` (dist_moe: ``... '_base'``).
backend="eager" works, and so does the same code without the ``grad_dtype`` read.

Usage (CPU): python ask03_autograd_fn_param_base.py

Expected (CPU, torch 2.15.0.dev20260926; see ../README.md, ask 3): with the grad_dtype read,
eager ok, aot_eager and inductor FAIL (BackendCompilerFailed ... 'NoneType' object has no attribute
'is_leaf'); without it, all three backends ok; with the read but no view (the parameter itself), all ok.
Then explain output (3 graphs / 2 breaks with the read, 1 / 0 without) and the innermost frames of the
aot_eager failure (AOTAutograd's metadata pass).
"""
import traceback
import weakref

import torch


def make_fn(read_grad_dtype: bool):
    class Fn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w):
            param = w if w._base is None else w._base
            if not param.is_leaf or not param.requires_grad:
                raise ValueError("needs a leaf")
            ctx.ref = weakref.ref(param)
            if read_grad_dtype:
                ctx.dtype = param.grad_dtype
            return x * 2

        @staticmethod
        def backward(ctx, g):
            return g * 2, None

    return Fn


class M(torch.nn.Module):
    def __init__(self, fn, use_view=True):
        super().__init__()
        self.fn = fn
        self.use_view = use_view
        self.w = torch.nn.Parameter(
            torch.randn(6, 4) if not use_view else torch.randn(2, 3, 4)
        )

    def forward(self, x):
        return self.fn.apply(x, self.w.flatten(0, 1) if self.use_view else self.w) + 1


def explain(read_grad_dtype):
    """Graph count, break reasons and captured graph code (a capturing backend, no AOTAutograd)."""
    torch._dynamo.reset()
    m = M(make_fn(read_grad_dtype))
    x = torch.randn(6, 4, requires_grad=True)
    ex = torch._dynamo.explain(m)(x)
    print(
        f"  explain, grad_dtype read={read_grad_dtype}: graphs={ex.graph_count} breaks={ex.graph_break_count}"
    )
    for b in ex.break_reasons[:1]:
        print("    break:", str(b.reason).splitlines()[0][:90])
    torch._dynamo.reset()
    codes = []
    torch.compile(m, backend=lambda gm, _: (codes.append(gm.code), gm.forward)[1])(x)
    for i, code in enumerate(codes):
        body = [
            l.strip()
            for l in code.splitlines()
            if " = " in l or l.strip().startswith("return")
        ]
        print(f"    graph {i}: " + " ; ".join(body)[:240])


def innermost_torch_frames():
    """Innermost frames of the aot_eager failure, to show where None meets the attribute read."""
    torch._dynamo.reset()
    m = M(make_fn(True))
    x = torch.randn(6, 4, requires_grad=True)
    try:
        torch.compile(m, backend="aot_eager")(x)
    except Exception as e:  # noqa: BLE001
        exc = e
        while exc.__cause__ is not None:
            exc = exc.__cause__
        frames = traceback.extract_tb(exc.__traceback__)
        print(
            f"  innermost frames of {type(exc).__name__}: {str(exc).splitlines()[1] if len(str(exc).splitlines()) > 1 else exc}"
        )
        for f in frames[-8:]:
            path = f.filename.split("site-packages/")[-1]
            print(f"    {path}:{f.lineno} in {f.name}: {(f.line or '').strip()[:80]}")


def main():
    for read_grad_dtype, use_view in ((True, True), (False, True), (True, False)):
        m = M(make_fn(read_grad_dtype), use_view)
        x = torch.randn(6, 4, requires_grad=True)
        m(x).sum().backward()
        label = (
            ("with" if read_grad_dtype else "without")
            + " grad_dtype read"
            + ("" if use_view else ", no view")
        )
        for backend in ("eager", "aot_eager", "inductor"):
            torch._dynamo.reset()
            m.w.grad = None
            try:
                torch.compile(m, backend=backend)(x).sum().backward()
                print(f"{label:33s} {backend:9s} ok")
            except Exception as e:  # noqa: BLE001
                lines = str(e).splitlines()
                print(
                    f"{label:33s} {backend:9s} FAIL {type(e).__name__} | {' | '.join(lines[:2])[:200]}"
                )
    explain(True)
    explain(False)
    innermost_torch_frames()


if __name__ == "__main__":
    main()
