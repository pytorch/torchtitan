"""LM-head grad padding for the FSDP reduce-scatter (Qwen3 235B-A22B, 256-way FSDP) as one CuTe DSL copy.

reference: split(in0 [151936, 4096] bf16, 594) -> 255 chunks of 594 rows + one of 466 rows; every chunk goes
through constant_pad_nd (zero pads except the last chunk, padded with 128 zero rows), then cat -> [152064, 4096].
That is in0 followed by 128 zero rows (151936 = 255 * 594 + 466, 152064 = 256 * 594).

A pure byte copy plus a zero tail (bitwise exact by construction; the pad value 0.0 is +0.0 in bf16, all zero
bytes). CTA c of 38016 handles output bytes [c*32 KiB, +32 KiB): the first 37984 CTAs copy the same bytes of in0
(each of 256 threads issues UNROLL 32-byte loads then UNROLL 32-byte stores), the last 32 write zeros.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

ROWS_IN = 151936
ROWS_OUT = 152064
COLS = 4096
THREADS = 256
VEC = 32  # bytes per access
UNROLL = 4
CTA_BYTES = THREADS * VEC * UNROLL
IN_BYTES = ROWS_IN * COLS * 2
OUT_BYTES = ROWS_OUT * COLS * 2
COPY_CTAS = IN_BYTES // CTA_BYTES
NCTA = OUT_BYTES // CTA_BYTES
assert IN_BYTES % CTA_BYTES == 0 and OUT_BYTES % CTA_BYTES == 0


def _vals(args, loc, ip):
    return [a.ir_value(loc=loc, ip=ip) for a in args]


@dsl_user_op
def ld_v8(addr, *, loc=None, ip=None):
    st = ir.Type.parse("!llvm.struct<(" + ", ".join(["i32"] * 8) + ")>")
    res = llvm.inline_asm(
        st, _vals([Int64(addr)], loc, ip),
        "ld.global.nc.L1::no_allocate.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
        "=r,=r,=r,=r,=r,=r,=r,=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return [Uint32(llvm.extractvalue(T.i32(), res, [k], loc=loc, ip=ip)) for k in range(8)]


@dsl_user_op
def st_v8(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(
        None, _vals([Int64(addr)] + [Uint32(x) for x in v], loc, ip),
        "st.global.v8.b32 [$0], {$1, $2, $3, $4, $5, $6, $7, $8};",
        "l,r,r,r,r,r,r,r,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@cute.kernel
def _pad_kernel(mIn: cute.Tensor, mOut: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    c, _, _ = cute.arch.block_idx()
    off = Int64(c) * CTA_BYTES + Int64(tidx) * VEC
    dst = mOut.iterator.toint() + off
    if c < COPY_CTAS:
        src = mIn.iterator.toint() + off
        vals = [ld_v8(src + u * THREADS * VEC) for u in range(UNROLL)]
        for u in cutlass.range_constexpr(UNROLL):
            st_v8(dst + u * THREADS * VEC, vals[u])
    else:
        z = [Uint32(0)] * 8
        for u in cutlass.range_constexpr(UNROLL):
            st_v8(dst + u * THREADS * VEC, z)


@cute.jit
def _launch(mIn: cute.Tensor, mOut: cute.Tensor, stream):
    _pad_kernel(mIn, mOut).launch(grid=[NCTA, 1, 1], block=[THREADS, 1, 1], stream=stream)


_CACHE = {}


def _fake(n):
    return cute.runtime.make_fake_tensor(cutlass.BFloat16, (n,), stride=(1,), assumed_align=32)


def run(in0):
    if not (in0.is_cuda and in0.dtype == torch.bfloat16 and tuple(in0.shape) == (ROWS_IN, COLS)
            and in0.is_contiguous() and in0.data_ptr() % 32 == 0):
        raise ValueError(f"in0: expected contiguous 32B-aligned cuda bf16 ({ROWS_IN}, {COLS}), got {in0.dtype} "
                         f"{tuple(in0.shape)} stride {in0.stride()} on {in0.device}")
    dev = in0.device
    out = torch.empty((ROWS_OUT, COLS), dtype=torch.bfloat16, device=dev)
    fn = _CACHE.get(dev.index)
    if fn is None:
        fn = cute.compile(
            _launch, _fake(ROWS_IN * COLS), _fake(ROWS_OUT * COLS),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev.index] = fn
    with torch.cuda.device(dev):
        fn(in0.view(-1), out.view(-1))
    return out
