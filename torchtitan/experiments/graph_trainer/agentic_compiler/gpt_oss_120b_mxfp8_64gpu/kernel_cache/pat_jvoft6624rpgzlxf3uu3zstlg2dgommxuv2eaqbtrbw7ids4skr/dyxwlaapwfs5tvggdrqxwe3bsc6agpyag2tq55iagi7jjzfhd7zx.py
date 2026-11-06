"""G-011 MoE combine backward (GPT-OSS 120B MXFP8): one cooperative CuTe DSL kernel.

Reference (eager aten, deterministic):
    v5 = gather(in0, 0, in1)                 [N, D] bf16, in1 = token ids (stride [1, 0])
    out0 = bf16(float(v5) * in3)             in3 = routing scores [N, 1] f32
    v11 = sum(float(v5) * float(in2), dim=1) [N] f32
    out1 = index_put(zeros(N), [in4], v11, accumulate=True).reshape(T, K)

Numerics (bitwise):
  * Row sum in torch's Reduce.cuh order for [N, 2880] f32 over dim 1 (block 32x16,
    input_vec_size 4, one warp per row): lane l owns 4-vectors v = l + 32 j, 4 accumulators
    (one per vector element) from +0, ((a0 + a1) + a2) + a3, shfl_down 16, 8, 4, 2, 1.
    Products x*y are fma.rn.f32.bf16(x, y, -0.0) (== mul.rn.f32 of the widened values for all
    inputs: exact product plus -0, rounded once), adds/muls are explicit .rn PTX (packed
    .f32x2 where possible), no FMA contraction.
  * out0 = cvt.rn.bf16x2(mul.rn(x, s)).
  * index_put(accumulate) with slice size 1 = indexing_backward_kernel_stride_1: duplicates in
    ascending row order (stable radix sort); < 32 -> sequential chain from +0; >= 32 -> 32 lane
    partials (k-th duplicate to lane k % 32 for k < 32*(n//32)), shfl_down tree, sequential
    remainder; out = 0 + chain. One row: out = 0 + (0 + v); no row: +0. Negative in4 wraps;
    out-of-range indices trap (the reference device-asserts).

Fast path (in1 stride-0 along D, in0/in2 rows 16 B aligned; the training layout), one
cooperative launch, 1 CTA x 8 warps per SM:
  (1) CTAs zero a per-call count array, grid barrier, count rows by destination with global
      atomics (count + one row), grid barrier.
  (2) Warps grab chunks of 8 consecutive destination slots from a global counter and resolve
      them lane-parallel: one row -> work item (row, token, score); no row -> out1 = +0;
      several rows -> exact duplicate path inline (rescans in4 in row order).
  (3) Work items stream through a 2-deep ring of (in0 token row, in2 row) bulk copies
      (cp.async.bulk + mbarrier) per warp; out0 rows are stored from registers and
      out1[q] = 0 + (0 + rowsum). Rows are visited in destination order, so the K rows of a
      token (adjacent slots in training: in4 is the argsort permutation, in1 = in4 // K,
      token_dispatcher._local_reorder) share one DRAM read of the in0 row.
  The visiting order does not change any value. Other layouts use a three-kernel general path.
"""

import threading

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

TT = 16384  # tokens
KK = 4  # top-k
N = TT * KK  # routed rows
D = 2880
RB = D * 2  # bytes per bf16 row
NV = D // 4  # 4-element vectors per row (720)
JF = NV // 32  # full vector slots per lane (22)
JT = NV - 32 * JF  # lanes holding one extra slot (16)
J = JF + (1 if JT else 0)
WARPS = 8
TAIL_THREADS = 256
WARPS1 = 8  # warps per CTA, fused kernel
STAGES = 2  # bulk-copy ring depth per warp
CH = 8  # destination slots per dynamically scheduled chunk
LOOK = 8  # resolve the next chunk when this many items are left
CAP = 32  # per-warp work-item ring (>= LOOK + CH + STAGES)


def _asm(ret, args, text, cons, side=False, loc=None, ip=None):
    return llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        text,
        cons,
        has_side_effects=side,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _st(types):
    return ir.Type.parse("!llvm.struct<(" + ", ".join(types) + ")>")


def _ex(res, i, ty, cls, loc, ip):
    return cls(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip))


@dsl_user_op
def ld_v2p(addr, pred, *, loc=None, ip=None):
    res = _asm(_st(["i32", "i32"]), [Int64(addr), Int32(pred)],
               "{ .reg .pred p; setp.ne.s32 p, $3, 0; mov.b32 $0, 0; mov.b32 $1, 0;\n"
               "@p ld.global.nc.v2.b32 {$0, $1}, [$2]; }", "=r,=r,l,r", loc=loc, ip=ip)
    return _ex(res, 0, T.i32(), Uint32, loc, ip), _ex(res, 1, T.i32(), Uint32, loc, ip)


@dsl_user_op
def policy(kind, *, loc=None, ip=None):
    return Int64(_asm(T.i64(), [], f"createpolicy.fractional.L2::{kind}.b64 $0, 1.0;", "=l", loc=loc, ip=ip))


@dsl_user_op
def ld_v2h(addr, pol, pred, *, loc=None, ip=None):
    """8 B load with an L2 cache policy (and L1 no_allocate), zero when pred == 0."""
    res = _asm(_st(["i32", "i32"]), [Int64(addr), Int64(pol), Int32(pred)],
               "{ .reg .pred p; setp.ne.s32 p, $4, 0; mov.b32 $0, 0; mov.b32 $1, 0;\n"
               "@p ld.global.nc.L1::no_allocate.L2::cache_hint.v2.b32 {$0, $1}, [$2], $3; }", "=r,=r,l,l,r",
               loc=loc, ip=ip)
    return _ex(res, 0, T.i32(), Uint32, loc, ip), _ex(res, 1, T.i32(), Uint32, loc, ip)


@dsl_user_op
def st_v2h(addr, w0, w1, pol, pred, *, loc=None, ip=None):
    _asm(None, [Int64(addr), Uint32(w0), Uint32(w1), Int64(pol), Int32(pred)],
         "{ .reg .pred p; setp.ne.s32 p, $4, 0;\n"
         "@p st.global.L1::no_allocate.L2::cache_hint.v2.b32 [$0], {$1, $2}, $3; }", "l,r,r,l,r",
         side=True, loc=loc, ip=ip)


@dsl_user_op
def ld_u16p(addr, pred, *, loc=None, ip=None):
    return Uint32(_asm(T.i32(), [Int64(addr), Int32(pred)],
                       "{ .reg .pred p; .reg .b16 h; setp.ne.s32 p, $2, 0; mov.b16 h, 0;\n"
                       "@p ld.global.nc.b16 h, [$1]; cvt.u32.u16 $0, h; }", "=r,l,r", loc=loc, ip=ip))


@dsl_user_op
def ld_s64(addr, *, loc=None, ip=None):
    return Int64(_asm(T.i64(), [Int64(addr)], "ld.global.nc.s64 $0, [$1];", "=l,l", loc=loc, ip=ip))


@dsl_user_op
def ld_s64p(addr, pred, *, loc=None, ip=None):
    return Int64(_asm(T.i64(), [Int64(addr), Int32(pred)],
                      "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.b64 $0, 0;\n"
                      "@p ld.global.nc.s64 $0, [$1]; }", "=l,l,r", loc=loc, ip=ip))


@dsl_user_op
def ld_f32(addr, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Int64(addr)], "ld.global.nc.f32 $0, [$1];", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def ld_u32(addr, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Int64(addr)], "ld.global.nc.u32 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def st_f32(addr, v, *, loc=None, ip=None):
    _asm(None, [Int64(addr), Float32(v)], "st.global.f32 [$0], $1;", "l,f", side=True, loc=loc, ip=ip)


@dsl_user_op
def st_u32(addr, v, *, loc=None, ip=None):
    _asm(None, [Int64(addr), Int32(v)], "st.global.u32 [$0], $1;", "l,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def red_add_u32(addr, *, loc=None, ip=None):
    _asm(None, [Int64(addr)], "red.global.add.u32 [$0], 1;", "l", side=True, loc=loc, ip=ip)


@dsl_user_op
def checked(i, n, *, loc=None, ip=None):
    """i if 0 <= i < n, else trap (gather index)."""
    return Int64(_asm(T.i64(), [Int64(i), Int64(n)],
                      "{ .reg .pred p; setp.ge.u64 p, $1, $2; @p trap; mov.b64 $0, $1; }", "=l,l,l",
                      side=True, loc=loc, ip=ip))


@dsl_user_op
def wrapped(i, n, *, loc=None, ip=None):
    """index_put index: i + n if i < 0; trap unless the result is in [0, n)."""
    return Int64(_asm(T.i64(), [Int64(i), Int64(n)],
                      "{ .reg .pred p; .reg .s64 t; add.s64 t, $1, $2; setp.lt.s64 p, $1, 0;\n"
                      "selp.b64 $0, t, $1, p; setp.ge.u64 p, $0, $2; @p trap; }", "=l,l,l",
                      side=True, loc=loc, ip=ip))


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc=loc, ip=ip))


@dsl_user_op
def fsel(pred, a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Int32(pred), Float32(a), Float32(b)],
                        "{ .reg .pred p; setp.ne.s32 p, $1, 0; selp.f32 $0, $2, $3, p; }", "=f,r,f,f",
                        loc=loc, ip=ip))


@dsl_user_op
def vec_step(x01, x23, y01, y23, s, a0, a1, a2, a3, *, loc=None, ip=None):
    """4 elements: acc_e += x_e * y_e (fp32 .rn), out_e = bf16(x_e * s) packed as 2 words."""
    res = _asm(
        _st(["f32", "f32", "f32", "f32", "i32", "i32"]),
        [Uint32(x01), Uint32(x23), Uint32(y01), Uint32(y23), Float32(s),
         Float32(a0), Float32(a1), Float32(a2), Float32(a3)],
        "{ .reg .b32 x0, x1, x2, x3, y0, y1, y2, y3, p, o0, o1, o2, o3;\n"
        "shl.b32 x0, $6, 16; and.b32 x1, $6, 0xFFFF0000;\n"
        "shl.b32 x2, $7, 16; and.b32 x3, $7, 0xFFFF0000;\n"
        "shl.b32 y0, $8, 16; and.b32 y1, $8, 0xFFFF0000;\n"
        "shl.b32 y2, $9, 16; and.b32 y3, $9, 0xFFFF0000;\n"
        "mul.rn.f32 p, x0, y0; add.rn.f32 $0, $11, p;\n"
        "mul.rn.f32 p, x1, y1; add.rn.f32 $1, $12, p;\n"
        "mul.rn.f32 p, x2, y2; add.rn.f32 $2, $13, p;\n"
        "mul.rn.f32 p, x3, y3; add.rn.f32 $3, $14, p;\n"
        "mul.rn.f32 o0, x0, $10; mul.rn.f32 o1, x1, $10;\n"
        "mul.rn.f32 o2, x2, $10; mul.rn.f32 o3, x3, $10;\n"
        "cvt.rn.bf16x2.f32 $4, o1, o0; cvt.rn.bf16x2.f32 $5, o3, o2; }",
        "=f,=f,=f,=f,=r,=r,r,r,r,r,f,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return tuple(_ex(res, i, T.f32(), Float32, loc, ip) for i in range(4)) + (
        _ex(res, 4, T.i32(), Uint32, loc, ip), _ex(res, 5, T.i32(), Uint32, loc, ip))


@dsl_user_op
def vec_step2(x01, x23, y01, y23, s, a0, a1, a2, a3, *, loc=None, ip=None):
    """vec_step with fewer instructions, bitwise identical:
    x*y products as fma.rn.f32.bf16(x, y, -0.0) (== mul.rn.f32 of the widened values for all
    inputs: the exact product plus -0 rounds once, signed zeros preserved), accumulator adds and
    the x*s products as packed .f32x2 .rn ops."""
    res = _asm(
        _st(["f32", "f32", "f32", "f32", "i32", "i32"]),
        [Uint32(x01), Uint32(x23), Uint32(y01), Uint32(y23), Float32(s),
         Float32(a0), Float32(a1), Float32(a2), Float32(a3)],
        "{ .reg .b16 xa, xb, xc, xd, ya, yb, yc, yd;\n"
        ".reg .b32 x0, x1, x2, x3, p0, p1, p2, p3, o0, o1, o2, o3;\n"
        ".reg .b64 X01, X23, SS, P01, P23, A01, A23, O01, O23;\n"
        "mov.b32 {xa, xb}, $6; mov.b32 {xc, xd}, $7; mov.b32 {ya, yb}, $8; mov.b32 {yc, yd}, $9;\n"
        "fma.rn.f32.bf16 p0, xa, ya, 0f80000000; fma.rn.f32.bf16 p1, xb, yb, 0f80000000;\n"
        "fma.rn.f32.bf16 p2, xc, yc, 0f80000000; fma.rn.f32.bf16 p3, xd, yd, 0f80000000;\n"
        "mov.b64 P01, {p0, p1}; mov.b64 P23, {p2, p3};\n"
        "mov.b64 A01, {$11, $12}; mov.b64 A23, {$13, $14};\n"
        "add.rn.f32x2 A01, A01, P01; add.rn.f32x2 A23, A23, P23;\n"
        "mov.b64 {$0, $1}, A01; mov.b64 {$2, $3}, A23;\n"
        "shl.b32 x0, $6, 16; and.b32 x1, $6, 0xFFFF0000;\n"
        "shl.b32 x2, $7, 16; and.b32 x3, $7, 0xFFFF0000;\n"
        "mov.b64 X01, {x0, x1}; mov.b64 X23, {x2, x3}; mov.b64 SS, {$10, $10};\n"
        "mul.rn.f32x2 O01, X01, SS; mul.rn.f32x2 O23, X23, SS;\n"
        "mov.b64 {o0, o1}, O01; mov.b64 {o2, o3}, O23;\n"
        "cvt.rn.bf16x2.f32 $4, o1, o0; cvt.rn.bf16x2.f32 $5, o3, o2; }",
        "=f,=f,=f,=f,=r,=r,r,r,r,r,f,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    return tuple(_ex(res, i, T.f32(), Float32, loc, ip) for i in range(4)) + (
        _ex(res, 4, T.i32(), Uint32, loc, ip), _ex(res, 5, T.i32(), Uint32, loc, ip))


@dsl_user_op
def ld_v2c(addr, pol, pred, *, loc=None, ip=None):
    """8 B load (L1-allocating) with an L2 cache policy, zero when pred == 0."""
    res = _asm(_st(["i32", "i32"]), [Int64(addr), Int64(pol), Int32(pred)],
               "{ .reg .pred p; setp.ne.s32 p, $4, 0; mov.b32 $0, 0; mov.b32 $1, 0;\n"
               "@p ld.global.nc.L2::cache_hint.v2.b32 {$0, $1}, [$2], $3; }", "=r,=r,l,l,r",
               loc=loc, ip=ip)
    return _ex(res, 0, T.i32(), Uint32, loc, ip), _ex(res, 1, T.i32(), Uint32, loc, ip)


# ---- shared memory, mbarrier and bulk-copy helpers (pipelined row kernel) ----
@dsl_user_op
def lds_v2(addr, *, loc=None, ip=None):
    res = _asm(_st(["i32", "i32"]), [Int32(addr)], "ld.shared.v2.b32 {$0, $1}, [$2];", "=r,=r,r",
               loc=loc, ip=ip)
    return _ex(res, 0, T.i32(), Uint32, loc, ip), _ex(res, 1, T.i32(), Uint32, loc, ip)


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    res = _asm(_st(["i32", "i32", "i32", "i32"]), [Int32(addr)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
               "=r,=r,=r,=r,r", loc=loc, ip=ip)
    return tuple(_ex(res, i, T.i32(), Int32, loc, ip) for i in range(4))


@dsl_user_op
def sts_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm(None, [Int32(addr), Int32(a), Int32(b), Int32(c), Int32(d)],
         "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def i2f(x, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Int32(x)], "mov.b32 $0, $1;", "=f,r", loc=loc, ip=ip))


@dsl_user_op
def mbar_init(addr, *, loc=None, ip=None):
    _asm(None, [Int32(addr)], "mbarrier.init.shared::cta.b64 [$0], 1;", "r", side=True, loc=loc, ip=ip)


@dsl_user_op
def fence_init(*, loc=None, ip=None):
    _asm(None, [], "fence.mbarrier_init.release.cluster;\nfence.proxy.async.shared::cta;", "", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def bulk_rows(mbar, sdst0, gsrc0, pol0, sdst1, gsrc1, pol1, *, loc=None, ip=None):
    """Expect 2*RB bytes on mbar, then two RB-byte bulk copies global -> shared."""
    _asm(None, [Int32(mbar), Int32(sdst0), Int64(gsrc0), Int64(pol0), Int32(sdst1), Int64(gsrc1), Int64(pol1)],
         "fence.proxy.async.shared::cta;\n"
         f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {2 * RB};\n"
         f"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [$1], [$2], {RB}, [$0], $3;\n"
         f"cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [$4], [$5], {RB}, [$0], $6;",
         "r,r,l,l,r,l,l", side=True, loc=loc, ip=ip)


@dsl_user_op
def mbar_wait(addr, parity, *, loc=None, ip=None):
    _asm(None, [Int32(addr), Int32(parity)],
         "{ .reg .pred P1;\nLAB_WAIT:\n"
         "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1;\n"
         "@P1 bra DONE;\nbra LAB_WAIT;\nDONE: }", "r,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def popc(x, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Int32(x)], "popc.b32 $0, $1;", "=r,r", loc=loc, ip=ip))


@dsl_user_op
def ballot(p, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Int32(p)],
                      "{ .reg .pred q; setp.ne.s32 q, $1, 0; vote.sync.ballot.b32 $0, q, 0xffffffff; }", "=r,r",
                      side=True, loc=loc, ip=ip))


@dsl_user_op
def ld_u32p(addr, pred, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Int64(addr), Int32(pred)],
                      "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.b32 $0, 0;\n"
                      "@p ld.global.nc.u32 $0, [$1]; }", "=r,l,r", loc=loc, ip=ip))


@dsl_user_op
def st_f32p(addr, v, pred, *, loc=None, ip=None):
    _asm(None, [Int64(addr), Float32(v), Int32(pred)],
         "{ .reg .pred p; setp.ne.s32 p, $2, 0; @p st.global.f32 [$0], $1; }", "l,f,r", side=True,
         loc=loc, ip=ip)


@dsl_user_op
def sel32(pred, a, b, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Int32(pred), Int32(a), Int32(b)],
                      "{ .reg .pred p; setp.ne.s32 p, $1, 0; selp.b32 $0, $2, $3, p; }", "=r,r,r,r",
                      loc=loc, ip=ip))


@dsl_user_op
def shfl_idx_f32(v, src, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(v), Int32(src)], "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;",
                        "=f,f,r", side=True, loc=loc, ip=ip))


@dsl_user_op
def ffs(x, *, loc=None, ip=None):
    """index of the lowest set bit of a nonzero word"""
    return Int32(_asm(T.i32(), [Int32(x)], "{ .reg .b32 t; brev.b32 t, $1; bfind.shiftamt.u32 $0, t; }",
                      "=r,r", loc=loc, ip=ip))


@cute.kernel
def _k_count(in4: cute.Tensor, counts: cute.Tensor, who: cute.Tensor, s40: Int64):
    """counts[d] = #rows with destination d (wrapped in4), who[d] = one such row."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    r = Int64(bidx * TAIL_THREADS + tidx)
    d = wrapped(ld_s64(in4.iterator.toint() + r * s40 * 8), Int64(N))
    red_add_u32(counts.iterator.toint() + d * 4)
    st_u32(who.iterator.toint() + d * 4, Int32(r))


def _row(r, lane, s, a0, a1, a2, ao, r2mul, s10, s11, pol_last, pol_first, bc1):
    """One routed row: writes out0[r] and returns the torch-ordered row sum (valid in lane 0).

    Plain Python (traced inline at each call site): every loop and branch here is static."""
    row2 = a2 + r * r2mul * RB
    rowo = ao + r * RB
    tailp = Int32(lane < JT)
    xs = []
    ys = []
    if bc1:
        tok = checked(ld_s64(a1 + r * s10 * 8), Int64(TT))
        row0 = a0 + tok * RB
        for j in range(J):
            off = Int64(lane + 32 * j) * 8
            pj = Int32(1) if (j < JF) else tailp
            xs.append(ld_v2c(row0 + off, pol_last, pj))
            ys.append(ld_v2h(row2 + off, pol_first, pj))
    else:
        # general in1 layout: per-element gather in0[in1[r, c], c]
        for j in range(J):
            off = Int64(lane + 32 * j) * 8
            pj = Int32(1) if (j < JF) else tailp
            hs = []
            for e in range(4):
                c = Int64((lane + 32 * j) * 4 + e)
                ti = ld_s64p(a1 + (r * s10 + c * s11) * 8, pj)
                ti = checked(ti, Int64(TT))
                hs.append(ld_u16p(a0 + (ti * D + c) * 2, pj))
            xs.append((hs[0] | (hs[1] << 16), hs[2] | (hs[3] << 16)))
            ys.append(ld_v2p(row2 + off, pj))
    acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
    for j in range(J):
        off = Int64(lane + 32 * j) * 8
        n0, n1, n2, n3, o01, o23 = vec_step(xs[j][0], xs[j][1], ys[j][0], ys[j][1], s,
                                            acc[0], acc[1], acc[2], acc[3])
        if (j < JF):
            acc = [n0, n1, n2, n3]
            st_v2h(rowo + off, o01, o23, pol_first, Int32(1))
        else:
            acc = [fsel(tailp, n0, acc[0]), fsel(tailp, n1, acc[1]), fsel(tailp, n2, acc[2]),
                   fsel(tailp, n3, acc[3])]
            st_v2h(rowo + off, o01, o23, pol_first, tailp)
    t = fadd(fadd(fadd(acc[0], acc[1]), acc[2]), acc[3])
    for k in range(5):
        t = fadd(t, cute.arch.shuffle_sync_down(t, offset=16 >> k))
    return t


@cute.kernel
def _k_rows(in0: cute.Tensor, in1: cute.Tensor, in2: cute.Tensor, in3: cute.Tensor, in4: cute.Tensor,
            out0: cute.Tensor, out1: cute.Tensor, tmp: cute.Tensor, counts: cute.Tensor, who: cute.Tensor,
            r2mul: Int64, s10: Int64, s11: Int64, s30: Int64, s40: Int64, bc1: cutlass.Constexpr):
    """Warp per destination slot q. Rows are visited in destination order so the K rows of one
    token (adjacent slots in training) run back to back and share the gathered in0 row in L1/L2.
    Slot q processes (a) row who[q] if q has exactly one row -> writes out1[q] directly, and
    (b) row q itself if q's own destination has >= 2 rows -> v11 to tmp for the tail kernel.
    Every row is processed exactly once; the row math does not depend on the visiting order."""
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    warp = tidx // 32
    q = Int64(bidx * WARPS + warp)
    a0 = in0.iterator.toint()
    a1 = in1.iterator.toint()
    a2 = in2.iterator.toint()
    a3 = in3.iterator.toint()
    ao = out0.iterator.toint()
    ac = counts.iterator.toint()
    pol_last = policy("evict_last")
    pol_first = policy("evict_first")
    cq = ld_u32(ac + q * 4)
    dq = wrapped(ld_s64(in4.iterator.toint() + q * s40 * 8), Int64(N))
    cd = ld_u32(ac + dq * 4)
    if cq == 1:
        r = Int64(ld_u32(who.iterator.toint() + q * 4))
        s = ld_f32(a3 + r * s30 * 4)
        t = _row(r, lane, s, a0, a1, a2, ao, r2mul, s10, s11, pol_last, pol_first, bc1)
        if lane == 0:
            st_f32(out1.iterator.toint() + q * 4, fadd(Float32(0.0), fadd(Float32(0.0), t)))
    if cq == 0:
        if lane == 0:
            st_f32(out1.iterator.toint() + q * 4, Float32(0.0))
    if cd > 1:
        s = ld_f32(a3 + q * s30 * 4)
        t = _row(q, lane, s, a0, a1, a2, ao, r2mul, s10, s11, pol_last, pol_first, bc1)
        if lane == 0:
            st_f32(tmp.iterator.toint() + q * 4, t)


@dsl_user_op
def trap_if(bad, *, loc=None, ip=None):
    _asm(None, [Int32(bad)], "{ .reg .pred p; setp.ne.s32 p, $0, 0; @p trap; }", "r", side=True, loc=loc, ip=ip)


@dsl_user_op
def ld_u32_cg(addr, *, loc=None, ip=None):
    """L2-coherent load (data written by the preceding count kernel)."""
    return Int32(_asm(T.i32(), [Int64(addr)], "ld.global.cg.u32 $0, [$1];", "=r,l", side=True, loc=loc, ip=ip))


@dsl_user_op
def grid_barrier(bar, inc, *, loc=None, ip=None):
    """Software grid barrier for the cooperative launch (all CTAs co-resident), called by one
    thread per CTA between CTA barriers; the cooperative_groups grid-sync scheme. The CTA
    increments of one barrier sum to 2^31, so with the low 31 bits of the word at 0 bit 31
    flips exactly at the last arrival and the low bits return to 0. The word is a per-device
    persistent int32 zeroed once (see _barrier_word); two launches of this kernel can never
    overlap on a device (each needs every SM), so the word is never shared concurrently.
    A barrier that spins for ~2^30 polls traps instead of hanging."""
    _asm(None, [Int64(bar), Int32(inc)],
         "{ .reg .u32 old, cur, x, n; .reg .pred p;\n"
         "fence.acq_rel.gpu;\n"
         "atom.add.release.gpu.u32 old, [$0], $1;\n"
         "mov.u32 n, 0;\n"
         "LAB_SPIN:\n"
         "ld.acquire.gpu.u32 cur, [$0];\n"
         "add.u32 n, n, 1; setp.gt.u32 p, n, 1073741824; @p trap;\n"
         "xor.b32 x, cur, old; and.b32 x, x, 0x80000000; setp.eq.u32 p, x, 0;\n"
         "@p bra LAB_SPIN;\n"
         "fence.acq_rel.gpu; }", "l,r", side=True, loc=loc, ip=ip)


@dsl_user_op
def count_row(d, bad, gcnt, gwho, r, *, loc=None, ip=None):
    """d = in4[r] (raw): wrap negatives, return bad | (out of range); global count[d] += 1, who[d] = r."""
    return Int32(_asm(T.i32(), [Int64(d), Int32(bad), Int64(gcnt), Int64(gwho), Int32(r)],
         "{ .reg .pred p; .reg .s64 d, t, o, a;\n"
         f"add.s64 t, $1, {N}; setp.lt.s64 p, $1, 0; selp.b64 d, t, $1, p;\n"
         f"setp.ge.u64 p, d, {N}; selp.u32 $0, 1, 0, p; or.b32 $0, $0, $2;\n"
         f"min.u64 d, d, {N - 1}; shl.b64 o, d, 2;\n"
         "add.s64 a, $3, o; red.global.add.u32 [a], 1;\n"
         "add.s64 a, $4, o; st.global.u32 [a], $5; }",
         "=r,l,r,l,l,r", side=True, loc=loc, ip=ip))


@dsl_user_op
def atom_add_u32(addr, v, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Int64(addr), Int32(v)], "atom.global.add.u32 $0, [$1], $2;", "=r,l,r",
                      side=True, loc=loc, ip=ip))


@dsl_user_op
def shfl_idx_i32(v, src, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Int32(v), Int32(src)], "shfl.sync.idx.b32 $0, $1, $2, 31, 0xffffffff;",
                      "=r,r,r", side=True, loc=loc, ip=ip))


@cute.jit
def _dup_dest(q, lane, a0, a1, a2, a3, a4, ao, ao1, gcnt, r2mul, s10, s30, s40, pol_last, pol_first, stt, kk):
    """Destination q with n >= 2 rows (never in training): scan in4 in ascending row order and
    reproduce indexing_backward_kernel_stride_1 (lane partials for the first 32*(n//32) rows,
    shfl_down tree, sequential remainder, out = 0 + sum)."""
    n = ld_u32_cg(gcnt + Int64(q) * 4)
    npart = (n // 32) * 32
    stt[0] = Float32(0.0)
    stt[1] = Float32(0.0)
    kk[0] = Int32(0)
    for rb in cutlass.range(N // 32):
        rr = Int64(rb * 32 + lane)
        key = wrapped(ld_s64(a4 + rr * s40 * 8), Int64(N))
        kk[1] = ballot(Int32(key == Int64(q)))
        for _b in cutlass.range(32):
            if kk[1] != 0:
                m = kk[1]
                r = Int64(rb * 32 + ffs(m))
                kk[1] = m & (m - 1)
                s = ld_f32(a3 + r * s30 * 4)
                t = _row(r, lane, s, a0, a1, a2, ao, r2mul, s10, Int64(0), pol_last, pol_first, True)
                v = shfl_idx_f32(t, Int32(0))
                k = kk[0]
                if k < npart:
                    if lane == k % 32:
                        stt[0] = fadd(stt[0], v)
                    if k == npart - 1:
                        p = stt[0]
                        for o in cutlass.range_constexpr(5):
                            p = fadd(p, cute.arch.shuffle_sync_down(p, offset=16 >> o))
                        stt[1] = p
                else:
                    stt[1] = fadd(stt[1], v)
                kk[0] = k + 1
    if lane == 0:
        st_f32(ao1 + Int64(q) * 4, fadd(Float32(0.0), stt[1]))


@dsl_user_op
def ld_u32p_cg(addr, pred, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Int64(addr), Int32(pred)],
                      "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.b32 $0, 0;\n"
                      "@p ld.global.cg.u32 $0, [$1]; }", "=r,l,r", side=True, loc=loc, ip=ip))


@cute.kernel
def _k_fused(a0: Int64, a1: Int64, a2: Int64, a3: Int64, a4: Int64, ao: Int64, ao1: Int64, gcnt: Int64,
             gbar: Int64, r2mul: cutlass.Constexpr, s10: cutlass.Constexpr, s30: cutlass.Constexpr,
             s40: cutlass.Constexpr, ncta: cutlass.Constexpr):
    """Whole region in one persistent kernel (bc1 layout, 16 B aligned rows).

    CTA c owns destination slots [c*S, c*S + S). (1) Cooperative launch: CTAs zero a per-call
    count array, grid barrier, count rows [c*S, c*S + S) by destination with global atomics
    (count + one row), grid barrier. (2) Warp w resolves its slots
    (c*S + w + WARPS1*k): one row -> work item (row, token, score), no row -> out1 = +0,
    several rows -> duplicate list. (3) Work items stream through an STAGES-deep ring of
    (in0 token row, in2 row) bulk copies; out0 rows are stored from registers and
    out1[q] = 0 + (0 + rowsum) (index_put of a single value). (4) Duplicate destinations:
    the warp scans in4 in ascending row order and reproduces indexing_backward_kernel_stride_1
    (lane partials for 32*(n//32) rows, shfl_down tree, sequential remainder, 0 + sum).
    Rows are visited in destination order, so the K rows of a token (adjacent slots in
    training) run concurrently and share the gathered in0 row through L2; the row math does
    not depend on the visiting order.
    """
    S = (N + ncta - 1) // ncta
    WB = (STAGES * 2 * RB + CAP * 16 + STAGES * 8 + 127) // 128 * 128
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    warp = tidx // 32
    smem = cutlass.utils.SmemAllocator()
    sbuf = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(WARPS1 * WB), byte_alignment=128)
    sbase = sbuf.iterator.toint()
    sb = sbase + warp * WB
    slist = sb + STAGES * 2 * RB
    smb = slist + CAP * 16
    base = Int32(bidx * S)
    nown = Int32(S)
    gwho = gcnt + 4 * N
    gwork = gcnt + 8 * N
    pol_last = policy("evict_last")
    pol_first = policy("evict_first")
    if lane == 0:
        for st in cutlass.range_constexpr(STAGES):
            mbar_init(smb + st * 8)
        fence_init()
    # ---- (1) per-destination counts in a per-call workspace (cooperative launch) ----
    binc = sel32(Int32(bidx == 0), Int32(2**31 - (ncta - 1)), Int32(1))  # increments sum to 2^31
    for i in cutlass.range_constexpr((S + 32 * WARPS1 - 1) // (32 * WARPS1)):
        idx = tidx + i * 32 * WARPS1
        if idx < S:
            if base + idx < N:
                st_u32(gcnt + Int64(base + idx) * 4, Int32(0))
    if bidx == 0:
        if tidx == 0:
            st_u32(gwork, Int32(0))
    cute.arch.sync_threads()
    if tidx == 0:
        grid_barrier(gbar, binc)
    cute.arch.sync_threads()
    bad = Int32(0)
    for i in cutlass.range_constexpr((S + 32 * WARPS1 - 1) // (32 * WARPS1)):
        idx = tidx + i * 32 * WARPS1
        rown = base + idx
        pr = Int32(idx < S) & Int32(rown < N)
        rs = sel32(pr, rown, Int32(0))
        dv = ld_s64p(a4 + Int64(rs) * s40 * 8, pr)
        if pr != 0:
            bad = count_row(dv, bad, gcnt, gwho, rs)
    trap_if(bad)
    cute.arch.sync_threads()
    if tidx == 0:
        grid_barrier(gbar, binc)
    cute.arch.sync_threads()
    # ---- (2)-(4) dynamically scheduled destination slots ----
    # Warps grab chunks of CH consecutive slots from a global counter (load balance across SMs;
    # the K rows of a token stay in one chunk) and resolve them lane-parallel: one row -> work
    # item in a per-warp smem ring, no row -> out1 = +0, several rows -> _dup_dest inline.
    # Items stream through the bulk-copy ring; the next chunk is resolved when <= LOOK items
    # are left. All loop state is warp-uniform.
    lt = (Int32(1) << lane) - 1
    tailp = Int32(lane < JT)
    stv = cute.make_rmem_tensor(cute.make_layout(4), Int32)  # [consumed, resolved, issued, done]
    for z in cutlass.range_constexpr(4):
        stv[z] = Int32(0)
    stt = cute.make_rmem_tensor(cute.make_layout(4), Float32)
    kk = cute.make_rmem_tensor(cute.make_layout(3), Int32)  # [_dup_dest k, _dup_dest mask, dup lanes]
    while (stv[0] < stv[1]) | (stv[3] == 0):
        if (stv[3] == 0) & (stv[1] - stv[0] <= LOOK):
            cw = Int32(0)
            if lane == 0:
                cw = atom_add_u32(gwork, Int32(1))
            cw = shfl_idx_i32(cw, Int32(0))
            q0 = cw * CH
            if q0 >= N:
                stv[3] = Int32(1)
            else:
                q = q0 + lane
                valid = Int32(lane < CH) & Int32(q < N)
                qs = sel32(valid, q, Int32(0))
                c = ld_u32p_cg(gcnt + Int64(qs) * 4, valid)
                pa = valid & Int32(c == 1)
                wr = ld_u32p_cg(gwho + Int64(qs) * 4, pa)  # who[q] is written only for counted slots
                pz = valid & Int32(c == 0)
                pd = valid & Int32(c > 1)
                st_f32p(ao1 + Int64(qs) * 4, Float32(0.0), pz)
                ra = Int64(sel32(pa, wr, Int32(0)))
                tok = checked(ld_s64p(a1 + ra * s10 * 8, pa), Int64(TT))
                sv = ld_u32p(a3 + ra * s30 * 4, pa)
                ba = ballot(pa)
                tail = stv[1]
                if pa != 0:
                    sts_v4(slist + ((tail + popc(ba & lt)) % CAP) * 16, Int32(ra), Int32(tok), sv, qs)
                stv[1] = tail + popc(ba)
                cute.arch.sync_warp()
                kk[2] = ballot(pd)
                for _d in cutlass.range(CH):
                    if kk[2] != 0:
                        md = kk[2]
                        kk[2] = md & (md - 1)
                        _dup_dest(q0 + ffs(md), lane, a0, a1, a2, a3, a4, ao, ao1, gcnt, r2mul, s10, s30, s40,
                                  pol_last, pol_first, stt, kk)
        # issue bulk copies for items [issued, min(resolved, consumed + STAGES))
        for _u in cutlass.range_constexpr(STAGES):
            jn = stv[2]
            if (jn < stv[1]) & (jn < stv[0] + STAGES):
                if lane == 0:
                    stn = jn % STAGES
                    itn = lds_v4(slist + (jn % CAP) * 16)
                    bulk_rows(smb + stn * 8, sb + stn * 2 * RB, a0 + Int64(itn[1]) * RB, pol_last,
                              sb + stn * 2 * RB + RB, a2 + Int64(itn[0]) * r2mul * RB, pol_first)
                stv[2] = jn + 1
        i = stv[0]
        if i < stv[1]:
            st = i % STAGES
            mbar_wait(smb + st * 8, (i // STAGES) % 2)
            itm = lds_v4(slist + (i % CAP) * 16)
            r = Int64(itm[0])
            s = i2f(itm[2])
            sx = sb + st * 2 * RB
            sy = sx + RB
            rowo = ao + r * RB
            acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
            for j in cutlass.range_constexpr(J):
                off = (lane + 32 * j) * 8
                if cutlass.const_expr(j < JF):
                    x01, x23 = lds_v2(sx + off)
                    y01, y23 = lds_v2(sy + off)
                    n0, n1, n2, n3, o01, o23 = vec_step2(x01, x23, y01, y23, s, acc[0], acc[1], acc[2], acc[3])
                    acc = [n0, n1, n2, n3]
                    st_v2h(rowo + Int64(off), o01, o23, pol_first, Int32(1))
                else:
                    offc = sel32(tailp, off, Int32(0))
                    x01, x23 = lds_v2(sx + offc)
                    y01, y23 = lds_v2(sy + offc)
                    n0, n1, n2, n3, o01, o23 = vec_step2(x01, x23, y01, y23, s, acc[0], acc[1], acc[2], acc[3])
                    acc = [fsel(tailp, n0, acc[0]), fsel(tailp, n1, acc[1]), fsel(tailp, n2, acc[2]),
                           fsel(tailp, n3, acc[3])]
                    st_v2h(rowo + Int64(off), o01, o23, pol_first, tailp)
            t = fadd(fadd(fadd(acc[0], acc[1]), acc[2]), acc[3])
            for k in cutlass.range_constexpr(5):
                t = fadd(t, cute.arch.shuffle_sync_down(t, offset=16 >> k))
            if lane == 0:
                st_f32(ao1 + Int64(itm[3]) * 4, fadd(Float32(0.0), fadd(Float32(0.0), t)))
            cute.arch.sync_warp()
            stv[0] = i + 1


@cute.kernel
def _k_dest(in4: cute.Tensor, tmp: cute.Tensor, counts: cute.Tensor, who: cute.Tensor, out1: cute.Tensor,
            s40: Int64):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    d = Int64(bidx * TAIL_THREADS + tidx)
    at = tmp.iterator.toint()
    c = ld_u32(counts.iterator.toint() + d * 4)
    if c > 1:
        # exact duplicate path: scan rows in ascending order (stable-sort order)
        npart = (c // 32) * 32
        parts = cute.make_rmem_tensor(cute.make_layout(32), Float32)
        for l in cutlass.range_constexpr(32):
            parts[l] = Float32(0.0)
        st = cute.make_rmem_tensor(cute.make_layout(2), Float32)  # [g]
        st[0] = Float32(0.0)
        kk = cute.make_rmem_tensor(cute.make_layout(1), Int32)
        kk[0] = Int32(0)
        a4 = in4.iterator.toint()
        for rr in cutlass.range(N):
            key = wrapped(ld_s64(a4 + Int64(rr) * s40 * 8), Int64(N))
            if key == d:
                v = ld_f32(at + Int64(rr) * 4)
                k = kk[0]
                if k < npart:
                    li = k % 32
                    parts[li] = fadd(parts[li], v)
                    if k == npart - 1:
                        p = [parts[l] for l in range(32)]
                        p = [fadd(p[l], p[l + 16]) for l in range(16)]
                        p = [fadd(p[l], p[l + 8]) for l in range(8)]
                        p = [fadd(p[l], p[l + 4]) for l in range(4)]
                        p = [fadd(p[l], p[l + 2]) for l in range(2)]
                        p = [fadd(p[0], p[1])]
                        st[0] = p[0]
                else:
                    st[0] = fadd(st[0], v)
                kk[0] = k + 1
        st_f32(out1.iterator.toint() + d * 4, fadd(Float32(0.0), st[0]))


@cute.jit
def _launch(in0, in1, in2, in3, in4, out0, tmp, counts, who, out1, r2mul: Int64, s10: Int64, s11: Int64,
            s30: Int64, s40: Int64, stream, bc1: cutlass.Constexpr):
    """General layout (in1 not stride-0 along D, or rows not 16 B aligned): three kernels."""
    _k_count(in4, counts, who, s40).launch(
        grid=[N // TAIL_THREADS, 1, 1], block=[TAIL_THREADS, 1, 1], stream=stream)
    _k_rows(in0, in1, in2, in3, in4, out0, out1, tmp, counts, who, r2mul, s10, s11, s30, s40, bc1).launch(
        grid=[N // WARPS, 1, 1], block=[32 * WARPS, 1, 1], stream=stream)
    _k_dest(in4, tmp, counts, who, out1, s40).launch(
        grid=[N // TAIL_THREADS, 1, 1], block=[TAIL_THREADS, 1, 1], stream=stream)


@cute.jit
def _launch_fused(a0: Int64, a1: Int64, a2: Int64, a3: Int64, a4: Int64, ao: Int64, ao1: Int64, gcnt: Int64,
                  gbar: Int64, anchor: cute.Tensor, stream, r2mul: cutlass.Constexpr, s10: cutlass.Constexpr,
                  s30: cutlass.Constexpr, s40: cutlass.Constexpr, ncta: cutlass.Constexpr):
    _k_fused(a0, a1, a2, a3, a4, ao, ao1, gcnt, gbar, r2mul, s10, s30, s40, ncta).launch(
        grid=[ncta, 1, 1], block=[32 * WARPS1, 1, 1], stream=stream, cooperative=True)


_CACHE = {}


def _fake(dt, shape, stride, align):
    return cute.runtime.make_fake_tensor(dt, shape, stride=stride, assumed_align=align)


def _compile(bc1):
    sym = cute.sym_int64
    bf, f32, i64, i32 = cutlass.BFloat16, cutlass.Float32, cutlass.Int64, cutlass.Int32
    args = (
        _fake(bf, (TT, D), (D, 1), 8),
        _fake(i64, (N, D), (sym(), sym()), 8),
        _fake(bf, (sym(), D), (D, 1), 8),
        _fake(f32, (N, 1), (sym(), sym()), 4),
        _fake(i64, (N,), (sym(),), 8),
        _fake(bf, (N, D), (D, 1), 16),
        _fake(f32, (N,), (1,), 16),
        _fake(i32, (N,), (1,), 16),
        _fake(i32, (N,), (1,), 16),
        _fake(f32, (TT, KK), (KK, 1), 16),
        Int64(0), Int64(0), Int64(0), Int64(0), Int64(0),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        bc1,
    )
    return cute.compile(_launch, *args, options="--enable-tvm-ffi")


def _compile_fused(ncta, r2mul, s10, s30, s40):
    return cute.compile(_launch_fused, *([Int64(0)] * 9),
                        _fake(cutlass.Float32, (TT, KK), (KK, 1), 16),
                        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), r2mul, s10, s30, s40, ncta,
                        options="--enable-tvm-ffi")


_BAR = {}
# Serializes all lazy init (cute.compile is not thread-safe; one barrier word per device). The
# warm path only reads the caches, which hold complete entries only.
_INIT_LOCK = threading.Lock()


def _barrier_word(dev):
    """Per-device persistent grid-barrier word (int32, zeroed once; see grid_barrier). Cached
    only after the zeroing has run: inside CUDA-graph capture the synchronize raises (the
    zeroing was only captured), nothing is cached, and the next eager call starts over. The
    caller holds _INIT_LOCK."""
    t = _BAR.get(dev)
    if t is None:
        t = torch.zeros((1,), dtype=torch.int32, device=dev)
        torch.cuda.synchronize(dev)
        _BAR[dev] = t
    return t.data_ptr()


def _check(name, t, dtype, shape, dev):
    if not isinstance(t, torch.Tensor) or t.dtype != dtype or tuple(t.shape) != shape or t.device != dev:
        raise ValueError(f"{name}: expected {dtype} {shape} on {dev}, got "
                         f"{getattr(t, 'dtype', type(t))} {tuple(getattr(t, 'shape', ()))}")


_FAST = {}  # input metadata -> (compiled fused launcher, barrier word, device)


def _key(in0, in1, in2, in3, in4):
    return (in0.device, in0.dtype, in0.shape, in0.stride(), in1.device, in1.dtype, in1.shape, in1.stride(),
            in2.device, in2.dtype, in2.shape, in2.stride(), in3.device, in3.dtype, in3.shape, in3.stride(),
            in4.device, in4.dtype, in4.shape, in4.stride())


def run(in0, in1, in2, in3, in4):
    ent = _FAST.get(_key(in0, in1, in2, in3, in4)) if isinstance(in4, torch.Tensor) else None
    p0 = in0.data_ptr() if ent is not None else 1
    p2 = in2.data_ptr() if ent is not None else 1
    if ent is None or (p0 | p2) & 15:
        return _run_general(in0, in1, in2, in3, in4)
    fn, bar, dev = ent
    fill = torch.utils.deterministic.fill_uninitialized_memory
    if fill:
        torch.utils.deterministic.fill_uninitialized_memory = False  # every element is written
    try:
        out0 = torch.empty((N, D), dtype=torch.bfloat16, device=dev)
        # out1 [T, K] f32 followed by the kernel's per-call workspace (counts, rows, work counter)
        buf = torch.empty((3 * N + 32,), dtype=torch.float32, device=dev)
    finally:
        if fill:
            torch.utils.deterministic.fill_uninitialized_memory = True
    out1 = buf.as_strided((TT, KK), (KK, 1))
    pb = buf.data_ptr()
    fn(p0, in1.data_ptr(), p2, in3.data_ptr(), in4.data_ptr(), out0.data_ptr(), pb, pb + 4 * N, bar, out1)
    return out0, out1


def _run_general(in0, in1, in2, in3, in4):
    """Validation, compilation, and the three-kernel general path (in1 not stride-0 along D)."""
    dev = in0.device
    _check("in0", in0, torch.bfloat16, (TT, D), dev)
    _check("in1", in1, torch.int64, (N, D), dev)
    _check("in3", in3, torch.float32, (N, 1), dev)
    _check("in4", in4, torch.int64, (N,), dev)
    if not isinstance(in2, torch.Tensor) or in2.dtype != torch.bfloat16 or in2.dim() != 2 or in2.shape[1] != D \
            or in2.device != dev:
        raise ValueError(f"in2: expected bf16 [R, {D}] on {dev}")
    if in2.shape[0] not in (1, N):
        # same failure as the reference's broadcasting multiply
        raise RuntimeError(f"The size of tensor a ({N}) must match the size of tensor b ({in2.shape[0]}) "
                           "at non-singleton dimension 0")
    for name, t in (("in0", in0), ("in2", in2)):
        if t.stride(1) != 1 or (t.shape[0] > 1 and t.stride(0) != D) or t.data_ptr() % 8:
            raise ValueError(f"{name}: rows must be contiguous and 8 B aligned, got stride {t.stride()}")
    bc1 = in1.stride(1) == 0
    r2mul = 0 if in2.shape[0] == 1 else 1
    if bc1:
        key = _key(in0, in1, in2, in3, in4)
        if key not in _FAST:
            with _INIT_LOCK:
                if key not in _FAST:
                    ncta = torch.cuda.get_device_properties(dev).multi_processor_count
                    _FAST[key] = (_compile_fused(ncta, r2mul, in1.stride(0), in3.stride(0), in4.stride(0)),
                                  _barrier_word(dev), dev)
        if in0.data_ptr() % 16 == 0 and in2.data_ptr() % 16 == 0:
            return run(in0, in1, in2, in3, in4)
    prev = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = False  # every element is written below
    try:
        out0 = torch.empty((N, D), dtype=torch.bfloat16, device=dev)
        out1 = torch.empty((TT, KK), dtype=torch.float32, device=dev)
        ws = torch.empty((2, N), dtype=torch.int32, device=dev)
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = prev
    fn = _CACHE.get(bc1)
    if fn is None:
        with _INIT_LOCK:
            fn = _CACHE.get(bc1)
            if fn is None:
                fn = _CACHE[bc1] = _compile(bc1)
    counts = torch.zeros((N,), dtype=torch.int32, device=dev)
    fn(in0, in1, in2, in3, in4, out0, ws[0].view(torch.float32), counts, ws[1], out1,
       r2mul, in1.stride(0), in1.stride(1), in3.stride(0), in4.stride(0))
    return out0, out1
