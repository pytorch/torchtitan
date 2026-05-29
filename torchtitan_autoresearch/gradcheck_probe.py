"""Standalone correctness probe for custom kernels (the ``kernel`` class).

For custom Triton ops (the pairwise-RoPE / SwiGLU class) the strongest check is
not training at all but verifying the backward directly. This runs on a single
device, imports the candidate op from the editable Qwen3 files (so it tests the
real code), and supports two modes:

  - ``gradcheck``: ``torch.autograd.gradcheck`` in float64. Rigorous, but only
    works if the op runs in double precision.
  - ``reference``: compare the candidate's forward and input-gradients against a
    trusted eager reference impl within tolerance. Use this for Triton kernels
    that cannot execute in float64.

Inputs are constructed by the harness (not the agent) via ``--kind`` so the
declared shapes can't be cherry-picked to dodge edge cases.

    python -m torchtitan_autoresearch.gradcheck_probe \
        --mode reference --kind rope \
        --target torchtitan.models.qwen3.sharding:pairwise_rope \
        --reference torchtitan_autoresearch.gradcheck_probe:rope_reference \
        --seq 128 --heads 8 --head-dim 128 --batch 4
"""

from __future__ import annotations

import argparse
import importlib
import sys

import torch


def _load(spec: str):
    """Import ``module:attr`` and return the attribute."""
    mod, attr = spec.split(":")
    return getattr(importlib.import_module(mod), attr)


def rope_reference(xq, xk, rope_cache):
    """Reference half-rotation RoPE for comparison.

    This is a generic pairwise rotate-half over the last dim; a candidate's
    Qwen3 RoPE layout should match it up to numerical tolerance. Adapt to the
    actual cache layout if the candidate uses a different convention.
    """
    cos, sin = rope_cache[..., 0], rope_cache[..., 1]

    def rotate(x):
        x1, x2 = x.chunk(2, dim=-1)
        rot = torch.cat((-x2, x1), dim=-1)
        return x * cos + rot * sin

    return rotate(xq), rotate(xk)


def _make_rope_inputs(args, dtype, device):
    g = torch.Generator(device=device).manual_seed(args.seed)
    shape = (args.batch, args.seq, args.heads, args.head_dim)
    xq = torch.randn(shape, generator=g, dtype=dtype, device=device, requires_grad=True)
    xk = torch.randn(shape, generator=g, dtype=dtype, device=device, requires_grad=True)
    half = args.head_dim
    cache = torch.randn(
        (args.batch, args.seq, 1, half, 2), generator=g, dtype=dtype, device=device
    )
    return (xq, xk, cache)


def _make_swiglu_inputs(args, dtype, device):
    g = torch.Generator(device=device).manual_seed(args.seed)
    shape = (args.batch * args.seq, args.head_dim)
    x = torch.randn(shape, generator=g, dtype=dtype, device=device, requires_grad=True)
    gate = torch.randn(
        shape, generator=g, dtype=dtype, device=device, requires_grad=True
    )
    return (x, gate)


_MAKERS = {"rope": _make_rope_inputs, "swiglu": _make_swiglu_inputs}


def run_gradcheck(fn, inputs) -> bool:
    inputs = tuple(
        i.double().detach().requires_grad_(True) if i.is_floating_point() else i
        for i in inputs
    )
    return torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-4, rtol=1e-3)


def run_reference(fn, ref, inputs, rtol: float, atol: float) -> tuple[bool, dict]:
    """Compare candidate vs reference forward and input grads via allclose."""
    cand_out = fn(*inputs)
    ref_inputs = tuple(
        i.detach().clone().requires_grad_(i.requires_grad)
        if isinstance(i, torch.Tensor)
        else i
        for i in inputs
    )
    ref_out = ref(*ref_inputs)
    cand_out = cand_out if isinstance(cand_out, tuple) else (cand_out,)
    ref_out = ref_out if isinstance(ref_out, tuple) else (ref_out,)

    fwd_ok = all(
        torch.allclose(c.float(), r.float(), rtol=rtol, atol=atol)
        for c, r in zip(cand_out, ref_out)
    )
    # Backprop a fixed scalar through both and compare input gradients.
    sum(c.float().sum() for c in cand_out).backward()
    sum(r.float().sum() for r in ref_out).backward()
    grad_ok = True
    for ci, ri in zip(inputs, ref_inputs):
        if isinstance(ci, torch.Tensor) and ci.grad is not None:
            grad_ok &= torch.allclose(
                ci.grad.float(), ri.grad.float(), rtol=rtol, atol=atol
            )
    return fwd_ok and grad_ok, {"forward_ok": fwd_ok, "grad_ok": grad_ok}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gradcheck", "reference"], default="reference")
    ap.add_argument("--kind", choices=list(_MAKERS), required=True)
    ap.add_argument("--target", required=True, help="module:callable for candidate")
    ap.add_argument("--reference", help="module:callable trusted impl (reference mode)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fn = _load(args.target)

    if args.mode == "gradcheck":
        inputs = _MAKERS[args.kind](args, torch.float64, device)
        ok = run_gradcheck(fn, inputs)
        print(f"VERIFY-KERNEL: mode=gradcheck status={'pass' if ok else 'FAIL'}")
        return 0 if ok else 1

    if not args.reference:
        print("VERIFY-KERNEL: FAIL reference mode requires --reference")
        return 2
    ref = _load(args.reference)
    inputs = _MAKERS[args.kind](args, torch.float32, device)
    ok, detail = run_reference(fn, ref, inputs, args.rtol, args.atol)
    print(f"VERIFY-KERNEL: mode=reference status={'pass' if ok else 'FAIL'} {detail}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
