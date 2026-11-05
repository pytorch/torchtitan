"""Fused MoE router backward (DSv3 671B, DistMoE MXFP8) -- one CuTe DSL kernel.

One warp per token row. Lane l owns score columns [4l, 4l+4) and [128+4l, 128+4l+4),
which is exactly the per-thread slice torch's vectorized row reduction
(Reduce.cuh: block 32x16, input_vec_size=4, step 32) uses for sum(dim=1) over
256 fp32 values, so the shuffle tree below reproduces torch's summation order.
The 8-wide sum uses torch's (8 lanes, no vectorization) order: lanes 0..7,
shfl_down 4, 2, 1.

All fp32 arithmetic is emitted as explicit `.rn` PTX (mul/add/sub/fma) so no
FMA contraction or approximate math can change rounding vs. the eager aten kernels.

Division: every divisor is a per-row value (in7 row / in10 row), so the reciprocal
refinement of ptxas's `div.rn.f32` fast path (MUFU.RCP + 2 FFMA) is hoisted per row and
each quotient uses the remaining fast-path FFMA steps (q0 = x*r; res = fma(-n, q0, x);
q = fma(r, res, q0)). This is exactly the instruction sequence ptxas emits for
`div.rn.f32`; it is only used when every operand of the lane lies in a conservative
range (2^-20 <= |n| <= 2^20, dividends 0 or 2^-60 <= |x| <= 2^60), where it equals
`div.rn.f32` bit for bit (validated on 3.4e10 random + structured pairs, see
ada/scratch/divtest.py). Zero dividends return x*r (the correctly signed zero). Any lane
with an operand outside that range recomputes all its quotients with `div.rn.f32`.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

ROWS = 4096
NE = 256  # experts
TOPK = 8
WARPS = 4  # rows (warps) per CTA

# float32 bit patterns for the fast-division operand ranges
_X_LO, _X_HI = "0f21800000", "0f5D800000"  # 2^-60, 2^60 (dividends)
_X6_LO, _X6_HI = "0f2B800000", "0f53800000"  # 2^-40, 2^40 (dividends that are divided twice)
_N_LO, _N_HI = "0f35800000", "0f49800000"  # 2^-20, 2^20 (divisors)


def _asm(ret_type, args, text, cons, loc=None, ip=None):
    return llvm.inline_asm(
        ret_type,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        text,
        cons,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fmul(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(a), Float32(b)], "mul.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def fsub(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(a), Float32(b)], "sub.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def fdiv(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(a), Float32(b)], "div.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def fneg(a, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(a)], "neg.f32 $0, $1;", "=f,f", loc, ip))


@dsl_user_op
def frcp_refined(n, *, loc=None, ip=None):
    """The divisor-only part of ptxas's div.rn.f32 fast path: r0 = MUFU.RCP(n); r = r0 + r0*(1 - n*r0)."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(n)],
            "{ .reg .f32 r0, e, nn;\n"
            "neg.f32 nn, $1;\n"
            "rcp.approx.ftz.f32 r0, $1;\n"
            "fma.rn.f32 e, nn, r0, 0f3F800000;\n"
            "fma.rn.f32 $0, r0, e, r0; }",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def fdiv_fast(x, n, r, *, loc=None, ip=None):
    """Quotient half of ptxas's div.rn.f32 fast path with hoisted reciprocal r; zero x -> x*r."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(x), Float32(n), Float32(r)],
            "{ .reg .f32 q0, res, q1, nn; .reg .pred pz;\n"
            "neg.f32 nn, $2;\n"
            "mul.rn.f32 q0, $1, $3;\n"
            "fma.rn.f32 res, nn, q0, $1;\n"
            "fma.rn.f32 q1, $3, res, q0;\n"
            "setp.eq.f32 pz, $1, 0f00000000;\n"
            "selp.f32 $0, q0, q1, pz; }",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


def _range_asm(lo, hi, allow_zero):
    z = "setp.eq.or.f32 p, $1, 0f00000000, p;\n" if allow_zero else ""
    return (
        "{ .reg .f32 a; .reg .pred p;\n"
        "abs.f32 a, $1;\n"
        f"setp.ge.f32 p, a, {lo};\n"
        f"setp.le.and.f32 p, a, {hi}, p;\n"
        f"{z}"
        "selp.u32 $0, 1, 0, p; }"
    )


@dsl_user_op
def ok_dividend(x, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Float32(x)], _range_asm(_X_LO, _X_HI, True), "=r,f", loc, ip))


@dsl_user_op
def ok_dividend2(x, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Float32(x)], _range_asm(_X6_LO, _X6_HI, True), "=r,f", loc, ip))


@dsl_user_op
def ok_divisor(n, *, loc=None, ip=None):
    return Int32(_asm(T.i32(), [Float32(n)], _range_asm(_N_LO, _N_HI, False), "=r,f", loc, ip))


@dsl_user_op
def fsgn(a, *, loc=None, ip=None):
    # torch sign: (0 < x) - (x < 0) as float; NaN and +-0 give +0.
    return Float32(
        _asm(
            T.f32(),
            [Float32(a)],
            "{ .reg .f32 p, q;\n"
            "set.gt.f32.f32 p, $1, 0f00000000;\n"
            "set.lt.f32.f32 q, $1, 0f00000000;\n"
            "sub.rn.f32 $0, p, q; }",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def fsel_nz(c, a, b, *, loc=None, ip=None):
    """c != 0 ? a : b (c is an Int32)."""
    return Float32(
        _asm(
            T.f32(),
            [Int32(c), Float32(a), Float32(b)],
            "{ .reg .pred p; setp.ne.s32 p, $1, 0; selp.f32 $0, $2, $3, p; }",
            "=f,r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def fsel_gt_1em12(x, a, *, loc=None, ip=None):
    """x > float32(1e-12) ? a : 0.0"""
    return Float32(
        _asm(
            T.f32(),
            [Float32(x), Float32(a)],
            "{ .reg .pred p; setp.gt.f32 p, $1, 0f2B8CBCCC; selp.f32 $0, $2, 0f00000000, p; }",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def bf16_split2(x, y, *, loc=None, ip=None):
    """hi/mid/lo bf16 split of fp32 pair (x = even col, y = odd col).

    Returns 3 packed b32 words {lo16: col x, hi16: col y} for hi, mid, lo.
    Matches: h = bf16(v); r = v - float(h); m = bf16(r); r2 = r - float(m); l = bf16(r2).
    """
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32)>"),
        [Float32(x), Float32(y)],
        "{ .reg .b32 d0, d1, ax, ay, rx, ry;\n"
        "cvt.rn.bf16x2.f32 d0, $4, $3;\n"
        "shl.b32 ax, d0, 16;\n"
        "and.b32 ay, d0, 0xFFFF0000;\n"
        "sub.rn.f32 rx, $3, ax;\n"
        "sub.rn.f32 ry, $4, ay;\n"
        "cvt.rn.bf16x2.f32 d1, ry, rx;\n"
        "shl.b32 ax, d1, 16;\n"
        "and.b32 ay, d1, 0xFFFF0000;\n"
        "sub.rn.f32 rx, rx, ax;\n"
        "sub.rn.f32 ry, ry, ay;\n"
        "cvt.rn.bf16x2.f32 $2, ry, rx;\n"
        "mov.b32 $0, d0;\n"
        "mov.b32 $1, d1; }",
        "=r,=r,=r,f,f",
        loc,
        ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
    )


class RouterBwd:
    def __init__(self, bcast7: bool):
        self.bcast7 = bcast7

    @cute.jit
    def __call__(self, mIn0, mIn1, mIn3, mIn4, mIn5, mIn6, mIn7, mIn8, mIn9, mIn10, mIn11, mIn12, mOut, stream):
        self.kernel(mIn0, mIn1, mIn3, mIn4, mIn5, mIn6, mIn7, mIn8, mIn9, mIn10, mIn11, mIn12, mOut).launch(
            grid=[ROWS // WARPS, 1, 1], block=[32 * WARPS, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, mIn0, mIn1, mIn3, mIn4, mIn5, mIn6, mIn7, mIn8, mIn9, mIn10, mIn11, mIn12, mOut):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        r = bidx * WARPS + warp

        smem = cutlass.utils.SmemAllocator()
        sScat = smem.allocate_tensor(Float32, cute.make_layout((NE, WARPS)), byte_alignment=16)

        # (4 cols, 32 lanes, 2 halves) view of a 256-wide row
        colL = cute.make_layout((4, 32, 2), stride=(1, 4, 128))
        fragL = cute.make_layout((4, 2), stride=(1, 4))

        x6 = cute.make_rmem_tensor(fragL, Float32)
        y12 = cute.make_rmem_tensor(fragL, Float32)
        w4 = cute.make_rmem_tensor(fragL, Float32)
        cute.autovec_copy(cute.composition(mIn6[r, None], colL)[None, lane, None], x6)
        cute.autovec_copy(cute.composition(mIn12[r, None], colL)[None, lane, None], y12)
        cute.autovec_copy(cute.composition(mIn4, colL)[None, lane, None], w4)
        n7 = cute.make_rmem_tensor(fragL, Float32)
        if cutlass.const_expr(self.bcast7):
            nv = mIn7[r, 0]
            for e in cutlass.range_constexpr(8):
                n7[e] = nv
        else:
            cute.autovec_copy(cute.composition(mIn7[r, None], colL)[None, lane, None], n7)

        m5 = Int32(mIn5[r, 0])
        m1 = Int32(mIn1[r, 0])
        s3 = mIn3[0]
        g8 = mIn8[r, 0]
        d10 = mIn10[r, 0]
        k = lane % TOPK
        a0 = mIn0[r, k]
        a9 = mIn9[r, k]
        idx = Int32(mIn11[r, k])

        # ---- fast-path divisions (hoisted reciprocals) ----
        r7 = cute.make_rmem_tensor(fragL, Float32)
        if cutlass.const_expr(self.bcast7):
            rv = frcp_refined(n7[0])
            ok = ok_divisor(n7[0])
            for e in cutlass.range_constexpr(8):
                r7[e] = rv
        else:
            ok = Int32(1)
            for e in cutlass.range_constexpr(8):
                r7[e] = frcp_refined(n7[e])
                ok = ok & ok_divisor(n7[e])
        r10 = frcp_refined(d10)
        ok = ok & ok_divisor(d10)

        s15 = fmul(Float32(1.0), s3)
        m5f = fsel_nz(m5, Float32(1.0), Float32(0.0))
        v20 = cute.make_rmem_tensor(fragL, Float32)
        v23 = cute.make_rmem_tensor(fragL, Float32)
        v25 = cute.make_rmem_tensor(fragL, Float32)
        for e in cutlass.range_constexpr(8):
            v17 = fmul(s15, w4[e])
            v20[e] = fmul(v17, m5f)
            ok = ok & ok_dividend2(x6[e]) & ok_dividend(v20[e])
            v22 = fdiv_fast(x6[e], n7[e], r7[e])
            v23[e] = fdiv_fast(v22, n7[e], r7[e])
            v25[e] = fdiv_fast(v20[e], n7[e], r7[e])

        v13 = fsel_nz(m1, Float32(0.0), a0)
        v33 = fmul(v13, Float32(2.5))
        ok = ok & ok_dividend2(a9) & ok_dividend(v33)
        v36 = fdiv_fast(fdiv_fast(a9, d10, r10), d10, r10)
        v38 = fdiv_fast(v33, d10, r10)

        if ok == Int32(0):
            # out-of-range operand somewhere in this lane: exact div.rn.f32 for everything
            for e in cutlass.range_constexpr(8):
                v23[e] = fdiv(fdiv(x6[e], n7[e]), n7[e])
                v25[e] = fdiv(v20[e], n7[e])
            v36 = fdiv(fdiv(a9, d10), d10)
            v38 = fdiv(v33, d10)

        # ---- 256-wide score path: v24 and torch-ordered row sum ----
        v24 = cute.make_rmem_tensor(fragL, Float32)
        for e in cutlass.range_constexpr(8):
            v24[e] = fmul(fneg(v20[e]), v23[e])
        acc0 = fadd(fadd(Float32(0.0), v24[0]), v24[4])
        acc1 = fadd(fadd(Float32(0.0), v24[1]), v24[5])
        acc2 = fadd(fadd(Float32(0.0), v24[2]), v24[6])
        acc3 = fadd(fadd(Float32(0.0), v24[3]), v24[7])
        t = fadd(fadd(fadd(acc0, acc1), acc2), acc3)
        for off in cutlass.range_constexpr(5):
            t = fadd(t, cute.arch.shuffle_sync_down(t, offset=16 >> off))
        v26 = cute.arch.shuffle_sync(t, 0)
        v29 = fsel_gt_1em12(g8, v26)

        # ---- 8-wide top-k path ----
        v37 = fmul(fneg(v33), v36)
        t8 = fadd(Float32(0.0), v37)
        for off in cutlass.range_constexpr(3):
            t8 = fadd(t8, cute.arch.shuffle_sync_down(t8, offset=4 >> off))
        v39 = cute.arch.shuffle_sync(t8, 0)
        v41 = fadd(v38, v39)
        sval = fadd(Float32(0.0), v41)  # scatter_add into zeros: 0 + v41

        # ---- scatter via per-warp smem row ----
        sRow = sScat[None, warp]
        sView = cute.composition(sRow, colL)[None, lane, None]
        zf = cute.make_rmem_tensor(fragL, Float32)
        for e in cutlass.range_constexpr(8):
            zf[e] = Float32(0.0)
        cute.autovec_copy(zf, sView)
        cute.arch.sync_warp()
        if lane < TOPK:
            sRow[idx] = sval
        cute.arch.sync_warp()
        v43 = cute.make_rmem_tensor(fragL, Float32)
        cute.autovec_copy(sView, v43)

        # ---- epilogue: add, sigmoid_backward, hi/mid/lo bf16 split ----
        v46 = cute.make_rmem_tensor(fragL, Float32)
        for e in cutlass.range_constexpr(8):
            v31 = fmul(fsgn(x6[e]), v29)
            v32 = fadd(v25[e], v31)
            v44 = fadd(v32, v43[e])
            y = y12[e]
            v46[e] = fmul(fmul(v44, fsub(Float32(1.0), y)), y)

        # output words: (2 words, 32 lanes, 2 halves, 3 slices) in a 384-word row
        outL = cute.make_layout((2, 32, 2, 3), stride=(1, 2, 64, 128))
        mOutW = cute.recast_tensor(mOut, Uint32)
        oFrag = cute.make_rmem_tensor(cute.make_layout((2, 2, 3), stride=(1, 2, 4)), Uint32)
        for h in cutlass.range_constexpr(2):
            for p in cutlass.range_constexpr(2):
                hi, mid, lo = bf16_split2(v46[2 * p + 4 * h], v46[2 * p + 1 + 4 * h])
                oFrag[p, h, 0] = hi
                oFrag[p, h, 1] = mid
                oFrag[p, h, 2] = lo
        cute.autovec_copy(oFrag, cute.composition(mOutW[r, None], outL)[None, lane, None, None])


_DT = {torch.float32: cutlass.Float32, torch.uint8: cutlass.Uint8, torch.int64: cutlass.Int64, torch.bfloat16: cutlass.BFloat16}
_CACHE = {}


def _fake(t):
    align = 16 if t.data_ptr() % 16 == 0 else t.element_size()
    return cute.runtime.make_fake_tensor(_DT[t.dtype], tuple(t.shape), stride=tuple(t.stride()), assumed_align=align)


def _key(ts):
    return tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.data_ptr() % 16 == 0) for t in ts)


def run(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12):
    out = torch.empty((ROWS, 3 * NE), dtype=torch.bfloat16, device=in6.device)
    in1u = in1.view(torch.uint8)
    in5u = in5.view(torch.uint8)
    in3v = in3.view(1)
    args = (in0, in1u, in3v, in4, in5u, in6, in7, in8, in9, in10, in11, in12, out)
    key = _key(args)
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(
            RouterBwd(in7.stride(1) == 0),
            *[_fake(t) for t in args],
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[key] = fn
    fn(*args)
    return (out, out[:, 0:NE].t(), out[:, NE : 2 * NE].t(), out[:, 2 * NE :].t())
