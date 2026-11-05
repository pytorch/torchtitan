# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers: load the dumped FX graph and classify nodes."""

import ast
import gzip
import json
import math
import re

DTYPE_BYTES = {
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.int16": 2,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.bool": 1,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e5m2": 1,
    "torch.complex64": 8,
}

RECOMPUTE_RE = re.compile(r"_recomputed(_\d+)?$")

GEMM_TARGETS = {
    "aten.mm.default",
    "aten.mm.dtype",
    "aten.bmm.dtype",
    "aten.addmm.dtype",
    "aten.addmm.default",
    "aten.bmm.default",
    "aten.baddbmm.default",
    "aten._grouped_mm.default",
    "aten._scaled_mm.default",
    "aten._scaled_mm_v2.default",
    "aten._scaled_grouped_mm.default",
    "aten._scaled_grouped_mm_v2.default",
}

# Ops that never launch a kernel (pure metadata / aliasing).
VIEW_TARGETS = {
    "aten.view.default",
    "aten._unsafe_view.default",
    "aten.t.default",
    "aten.transpose.int",
    "aten.permute.default",
    "aten.expand.default",
    "aten.unsqueeze.default",
    "aten.squeeze.dim",
    "aten.squeeze.dims",
    "aten.squeeze.default",
    "aten.slice.Tensor",
    "aten.select.int",
    "aten.alias.default",
    "aten.as_strided.default",
    "aten.detach.default",
    "aten.split.Tensor",
    "aten.split_with_sizes.default",
    "aten.unbind.int",
    "aten.view_as_real.default",
    "aten.view_as_complex.default",
    "aten.unflatten.int",
    "aten.chunk.default",
    "aten.diagonal.default",
    "aten.view.dtype",
    "_operator.getitem",
}


def is_comm_target(t: str) -> bool:
    return (
        t.startswith("_c10d_functional.")
        or t.startswith("c10d_functional.")
        or t.startswith("_dtensor.")
        or "wait_tensor" in t
        or t.startswith("torchtitan.")
        and "token_exchange" in t
    )


def is_fused_moe_target(t: str) -> bool:
    """Dist-MoE's fused dispatch + expert GEMMs + combine ops (not its weight-prepare casts)."""
    return t.startswith("dist_moe.") and not t.startswith("dist_moe.prepare_")


def is_attention_target(t: str) -> bool:
    tl = t.lower()
    return (
        "flex_attention" in tl
        or "scaled_dot_product" in tl
        or "flash_attention" in tl
        or "efficient_attention" in tl
        or "cudnn_attention" in tl
        or "varlen_attn" in tl
    )


def strip_repr(s):
    if s is None:
        return None
    try:
        v = ast.literal_eval(s)
        return v
    except Exception:  # noqa: BLE001
        return s


def load_nodes(path):
    nodes = []
    with gzip.open(path, "rt") as f:
        for line in f:
            nodes.append(json.loads(line))
    by_name = {n["name"]: n for n in nodes}
    for n in nodes:
        custom = n.get("custom") or {}
        fqn = strip_repr(custom.get("module_fqn")) if custom else None
        n["module_fqn"] = fqn if isinstance(fqn, str) else ""
        m = re.match(r"^layers\.(\d+)(\.|$)", n["module_fqn"])
        n["layer"] = int(m.group(1)) if m else None
        n["phase"] = phase_of(n)
        n["input_nodes"] = list(
            dict.fromkeys(iter_arg_nodes(n["args"]) + iter_arg_nodes(n["kwargs"]))
        )
    return nodes, by_name


def phase_of(n):
    if n["op"] in ("placeholder", "output", "get_attr"):
        return n["op"]
    if RECOMPUTE_RE.search(n["name"]):
        return "recompute"
    if n.get("autograd_backward"):
        return "backward"
    return "forward"


def iter_arg_nodes(a):
    out = []

    def rec(x):
        if isinstance(x, dict):
            if set(x.keys()) == {"node"}:
                out.append(x["node"])
            else:
                for v in x.values():
                    rec(v)
        elif isinstance(x, list):
            for v in x:
                rec(v)

    rec(a)
    return out


def tensor_vals(val):
    """Flatten a dumped ``val`` into a list of tensor dicts."""
    out = []

    def rec(v):
        if isinstance(v, dict) and v.get("t") == "tensor":
            out.append(v)
        elif isinstance(v, list):
            for x in v:
                rec(x)

    rec(val)
    return out


def _int(x, sym=None):
    if isinstance(x, int):
        return x
    if sym is not None and x in sym:
        return sym[x]
    try:
        return int(x)
    except Exception:  # noqa: BLE001
        return None


def tensor_bytes(t, *, unique=True, sym=None):
    """Bytes of a dumped tensor. ``unique`` drops stride-0 (expanded) dims."""
    if t is None:
        return 0
    item = DTYPE_BYTES.get(t["dtype"], 4)
    shape = t["shape"]
    stride = t.get("stride") or [1] * len(shape)
    numel = 1
    for s, st in zip(shape, stride):
        si = _int(s, sym)
        if si is None:
            return None
        if unique and st == 0:
            continue
        numel *= si
    return numel * item


def numel(t, sym=None):
    n = 1
    for s in t["shape"]:
        si = _int(s, sym)
        if si is None:
            return None
        n *= si
    return n


def gemm_flops(n, by_name, sym=None):
    t = n["target"]
    ins = [by_name[x] for x in iter_arg_nodes(n["args"]) if x in by_name]
    ivals = [tensor_vals(i["val"])[0] if tensor_vals(i["val"]) else None for i in ins]
    out = tensor_vals(n["val"])
    if not out:
        return 0
    o = out[0]
    try:
        if t in (
            "aten.mm.default",
            "aten.mm.dtype",
            "aten._scaled_mm.default",
            "aten._scaled_mm_v2.default",
        ):
            a = ivals[0]
            return 2 * numel(o, sym) * _int(a["shape"][-1], sym)
        if t in ("aten.addmm.default", "aten.addmm.dtype"):
            a = ivals[1]
            return 2 * numel(o, sym) * _int(a["shape"][-1], sym)
        if t in ("aten.bmm.default", "aten.bmm.dtype", "aten.baddbmm.default"):
            a = ivals[0] if t == "aten.bmm.default" else ivals[1]
            return 2 * numel(o, sym) * _int(a["shape"][-1], sym)
        if t in (
            "aten._grouped_mm.default",
            "aten._scaled_grouped_mm.default",
            "aten._scaled_grouped_mm_v2.default",
        ):
            a, b = ivals[0], ivals[1]
            if len(a["shape"]) == 2 and len(b["shape"]) == 3:
                return (
                    2
                    * _int(a["shape"][0], sym)
                    * _int(a["shape"][1], sym)
                    * _int(b["shape"][-1], sym)
                )
            if len(a["shape"]) == 2 and len(b["shape"]) == 2:
                # dW: [M, Ttot] x [Ttot, N] grouped over offs -> [G, M, N]
                return (
                    2
                    * _int(a["shape"][0], sym)
                    * _int(a["shape"][1], sym)
                    * _int(b["shape"][1], sym)
                )
            if len(a["shape"]) == 3 and len(b["shape"]) == 3:
                return 2 * numel(o, sym) * _int(a["shape"][-1], sym)
            if len(a["shape"]) == 3 and len(b["shape"]) == 2:
                return 2 * numel(o, sym) * _int(a["shape"][-1], sym)
    except Exception:  # noqa: BLE001
        return None
    return None


def node_io_bytes(n, by_name, sym=None):
    """(bytes_read, bytes_written) for a single node, standalone."""
    rd = 0
    for x in n["input_nodes"]:
        p = by_name.get(x)
        if p is None:
            continue
        for tv in tensor_vals(p["val"]):
            b = tensor_bytes(tv, sym=sym)
            if b is None:
                return None, None
            rd += b
    wr = 0
    for tv in tensor_vals(n["val"]):
        b = tensor_bytes(tv, unique=False, sym=sym)
        if b is None:
            return rd, None
        wr += b
    return rd, wr


def gemm_peak_tflops(n, by_name, cal):
    """GEMM peak for node ``n``: the MXFP8 peak when an operand is float8, else bf16."""
    dtypes = {
        v["dtype"]
        for x in iter_arg_nodes(n["args"])
        if x in by_name
        for v in tensor_vals(by_name[x]["val"])
    }
    if any(d.startswith("torch.float8") for d in dtypes):
        if "mxfp8_gemm_tflops" not in cal:
            raise ValueError(
                f"{n['name']} is a float8 GEMM; rerun calibrate_roofline.py with --mxfp8"
            )
        return cal["mxfp8_gemm_tflops"]
    return cal["bf16_gemm_tflops"]


def ideal_us(bytes_total, flops, bw_gbps, tflops):
    t_mem = (bytes_total or 0) / (bw_gbps * 1e3)  # bytes / (GB/s) -> us
    t_cmp = (flops or 0) / (tflops * 1e6)  # flops / (TFLOP/s) -> us
    return max(t_mem, t_cmp)


def normalize_fqn(fqn):
    return re.sub(r"^layers\.\d+", "layers.*", fqn or "")


def short_src(stack_trace):
    """Innermost torchtitan frame of a stack trace: file:line in func."""
    if not stack_trace:
        return ""
    pat = re.compile(r'^\s*File "(.+)", line (\d+), in (.+)$')
    lines = stack_trace.strip().split("\n")
    for line in reversed(lines):
        m = pat.match(line)
        if m and "/torch/" not in m.group(1) and "site-packages" not in m.group(1):
            path = m.group(1)
            if "torchtitan/" in path:
                path = path[path.index("torchtitan/") :]
            return f"{path}:{m.group(2)} in {m.group(3)}"
    return ""


def log2ceil(x):
    return int(math.ceil(math.log2(max(x, 1))))
