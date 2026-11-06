"""Fused MoE router + aux-loss backward (Qwen3 30B-A3B, 32768 tokens x 128 experts, top-8).

One CuTe DSL kernel. Each CTA is one warp that owns NGW groups of GR = 4 consecutive token rows:
lane l owns score columns [4l, 4l+4) of a row, and in the top-k phase lane l owns slot l % 8 of
row l // 8 of a group. All NGW groups' score blocks (in6, in12: 2 KB each per group) arrive by
1-D bulk async copies (cp.async.bulk + mbarrier) into shared memory, issued at warp start, so
the later groups' loads are hidden behind the earlier groups' math and in-flight data costs no
registers; the small per-row operands are plain loads issued meanwhile. One-warp CTAs let the
hardware scheduler balance the 4096 CTAs across SMs. Per group, the four rows' butterfly
reductions run together (tree4), and the fast or exact (IEEE div.rn) path is chosen once per
group by a warp vote over operand ranges.

Reduction orders (checked against torch on this build):
  * sum(v24, 1) over 128 fp32: Reduce.cuh vectorized input (vec 4), block (32, 16): lane l sums
    its float4 in slot order, then a shfl_down 16, 8, 4, 2, 1 tree (computed as the equivalent
    xor butterfly, which gives every lane lane 0's value).
  * sum(v37, 1) over 8: lanes 0..7 (no vectorization), shfl_down 4, 2, 1 (as xor butterfly).
  * _softmax_backward_data (persistent warp softmax): tmp = grad * out; lane l sums
    tmp[l + 32 it] for it = 0..3 sequentially, xor butterfly 16..1; grad_in = fma(-out, sum, tmp).
    tmp is transposed through a per-warp smem row for that sum.
  * scatter_add (deterministic): duplicate indices in a row accumulate in slot order from 0.
"0 + x" accumulator seeds of torch's reductions only change the sign of an exact zero sum. No
zero sign of v22..v41 can reach the output: v44 = v32 + S, where S (the scatter_add result) is
+0 or 0 + v41 and is never -0, so v44 is never -0 either. Likewise v24 = (-v20) * v23 is summed
as -(sum of v20 * v23) (round-to-nearest is odd-symmetric, so the sum is exactly negated).
The softmax-backward sum does need its seed: tmp = v44 * in12 is -0 in every column of a row
when v44 = +0 and in12 <= -0, or v44 < 0 and in12 = +0; torch's sum is then +0 (not -0), which
decides the sign of zero outputs through fma(-out, sum, tmp). It is kept as (0 + t0) + ...

All fp32 arithmetic is explicit round-to-nearest PTX (`.rn`, scalar or packed f32x2, which
rounds each element exactly like the scalar op), so no FMA contraction changes rounding.
v15 = 1 * in3 and v33 = v13 * 1.0 are skipped: they only change NaN payloads, which the bf16
conversion of the output canonicalizes (cvt.rn.bf16 gives 0x7FFF for every NaN, as torch).

Division: per-row divisors (in7 row, in10 row) use ptxas's div.rn.f32 fast-path sequence with
the divisor's refined reciprocal hoisted (q0 = x*r; res = fma(-n, q0, x); q = fma(r, res, q0)),
when every operand of the lane is in a range where that sequence equals div.rn.f32 (dividends
0 or 2^-60 <= |x| <= 2^60, twice-divided dividends 2^-40..2^40, divisors 2^-20..2^20);
otherwise the lane recomputes with div.rn.f32. NaN dividends give NaN on both paths. Zero
dividends may get a wrongly signed zero from the fast path, which cannot reach the output.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

ROWS = 32768
NE = 128  # experts
TOPK = 8
GR = 4  # rows per warp; the 32 lanes cover the GR x 8 top-k slots of these rows
NGROUPS = ROWS // GR
WARPS = 1  # warps per CTA
NGW = 1  # groups per warp

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
def _bulk_g2s(dst, src, nbytes, mbar, *, loc=None, ip=None):
    """1-D bulk async copy global -> this CTA's smem, completing nbytes on mbar."""
    llvm.inline_asm(
        None,
        [dst.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), src.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
         mbar.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)],
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], {nbytes}, [$2];",
        "r,l,r",
        has_side_effects=True,
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
def fnegmul(a, b, *, loc=None, ip=None):
    """(-a) * b"""
    return Float32(
        _asm(T.f32(), [Float32(a), Float32(b)], "{ .reg .f32 t; neg.f32 t, $1; mul.rn.f32 $0, t, $2; }", "=f,f,f", loc, ip)
    )


@dsl_user_op
def ffma_neg(a, b, c, *, loc=None, ip=None):
    """fma(-a, b, c)"""
    return Float32(
        _asm(
            T.f32(),
            [Float32(a), Float32(b), Float32(c)],
            "{ .reg .f32 t; neg.f32 t, $1; fma.rn.f32 $0, t, $2, $3; }",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def fdiv(a, b, *, loc=None, ip=None):
    return Float32(_asm(T.f32(), [Float32(a), Float32(b)], "div.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def frcp_refined(n, *, loc=None, ip=None):
    """Divisor-only part of ptxas's div.rn.f32 fast path: r0 = MUFU.RCP(n); r = r0 + r0*(1 - n*r0)."""
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
    """Quotient half of ptxas's div.rn.f32 fast path with hoisted reciprocal r (zero sign not kept)."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(x), Float32(n), Float32(r)],
            "{ .reg .f32 q0, res, nn;\n"
            "neg.f32 nn, $2;\n"
            "mul.rn.f32 q0, $1, $3;\n"
            "fma.rn.f32 res, nn, q0, $1;\n"
            "fma.rn.f32 $0, $3, res, q0; }",
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
def fsgn_mul(a, b, *, loc=None, ip=None):
    """torch.sgn(a) * b, sgn = (0 < a) - (a < 0) as float (NaN and +-0 give +0)."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(a), Float32(b)],
            "{ .reg .f32 p, q, s;\n"
            "set.gt.f32.f32 p, $1, 0f00000000;\n"
            "set.lt.f32.f32 q, $1, 0f00000000;\n"
            "sub.rn.f32 s, p, q;\n"
            "mul.rn.f32 $0, s, $2; }",
            "=f,f,f",
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
    """hi/mid/lo bf16 split of an fp32 pair (x = even col, y = odd col), packed {lo16: x, hi16: y}.

    h = bf16(v); r = v - float(h); m = bf16(r); r2 = r - float(m); l = bf16(r2).
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


@dsl_user_op
def fabsmax(acc, x, *, loc=None, ip=None):
    """max(acc, |x|) (NaN x leaves acc)."""
    return Float32(
        _asm(T.f32(), [Float32(acc), Float32(x)], "{ .reg .f32 a; abs.f32 a, $2; max.f32 $0, $1, a; }", "=f,f,f", loc, ip)
    )


@dsl_user_op
def fabsmin(acc, x, *, loc=None, ip=None):
    """min(acc, |x|) (NaN x leaves acc)."""
    return Float32(
        _asm(T.f32(), [Float32(acc), Float32(x)], "{ .reg .f32 a; abs.f32 a, $2; min.f32 $0, $1, a; }", "=f,f,f", loc, ip)
    )


@dsl_user_op
def in_range(amin, amax, lo, hi, *, loc=None, ip=None):
    """1 if lo <= amin and amax <= hi else 0 (lo/hi are PTX float literals)."""
    return Int32(
        _asm(
            T.i32(),
            [Float32(amin), Float32(amax)],
            "{ .reg .pred p;\n"
            f"setp.ge.f32 p, $1, {lo};\n"
            f"setp.le.and.f32 p, $2, {hi}, p;\n"
            "selp.u32 $0, 1, 0, p; }",
            "=r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def fsgn_mul_sel(x, v, z, *, loc=None, ip=None):
    """torch.sgn(x) * v given z = 0 * v: |x| > 0 ? v with x's sign applied : z.

    sgn(x) * v is exactly +-v for nonzero non-NaN x (up to NaN payload/sign, which the bf16
    output conversion canonicalizes) and 0 * v for x in {+0, -0, NaN}.
    """
    return Float32(
        _asm(
            T.f32(),
            [Float32(x), Float32(v), Float32(z)],
            "{ .reg .pred p; .reg .f32 a; .reg .b32 t;\n"
            "abs.f32 a, $1;\n"
            "setp.gt.f32 p, a, 0f00000000;\n"
            "and.b32 t, $1, 0x80000000;\n"
            "xor.b32 t, t, $2;\n"
            "selp.b32 $0, t, $3, p; }",
            "=f,f,f,f",
            loc,
            ip,
        )
    )


def p_mul(a, b):
    return cute.arch.mul_packed_f32x2(a, b, rnd="rn")


def p_add(a, b):
    return cute.arch.add_packed_f32x2(a, b, rnd="rn")


def p_sub(a, b):
    return cute.arch.sub_packed_f32x2(a, b, rnd="rn")


def p_fma(a, b, c):
    return cute.arch.fma_packed_f32x2(a, b, c, rnd="rn")


def p_div_fast(x, nn, r):
    """Packed quotient half of the div.rn.f32 fast path: nn = (-n, -n), r = refined 1/n."""
    q0 = p_mul(x, r)
    return p_fma(r, p_fma(nn, q0, x), q0)


@dsl_user_op
def cvt_bf16x2(x, y, *, loc=None, ip=None):
    """{lo16: bf16(x), hi16: bf16(y)} plus float(bf16(x)), float(bf16(y))."""
    res = _asm(
        ir.Type.parse("!llvm.struct<(i32, f32, f32)>"),
        [Float32(x), Float32(y)],
        "{ .reg .b32 d, a, b;\n"
        "cvt.rn.bf16x2.f32 d, $4, $3;\n"
        "shl.b32 a, d, 16;\n"
        "and.b32 b, d, 0xFFFF0000;\n"
        "mov.b32 $0, d;\n"
        "mov.b32 $1, a;\n"
        "mov.b32 $2, b; }",
        "=r,=f,=f,f,f",
        loc,
        ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_bf16x2_last(x, y, *, loc=None, ip=None):
    return Uint32(_asm(T.i32(), [Float32(x), Float32(y)], "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,f,f", loc, ip))


def bf16_split_pair(v):
    """hi/mid/lo bf16 split of an fp32 pair: h = bf16(v); r = v - float(h); m = bf16(r);
    r2 = r - float(m); l = bf16(r2). Returns packed words {lo16: v[0], hi16: v[1]}."""
    hi, ax, ay = cvt_bf16x2(v[0], v[1])
    r = p_sub(v, (ax, ay))
    mid, bx, by = cvt_bf16x2(r[0], r[1])
    r2 = p_sub(r, (bx, by))
    lo = cvt_bf16x2_last(r2[0], r2[1])
    return hi, mid, lo


@dsl_user_op
def mbar_wait(mbar, phase, *, loc=None, ip=None):
    """Spin on mbarrier.try_wait.parity (no nanosleep back-off)."""
    llvm.inline_asm(
        None,
        [mbar.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip), Int32(phase).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred p;\n"
        "LAB_WAIT:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n"
        "@p bra DONE;\n"
        "bra LAB_WAIT;\n"
        "DONE:\n"
        "}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def fxorsign(v, x, *, loc=None, ip=None):
    """v with x's sign bit xor-ed in: sgn(x) * v for nonzero non-NaN x (NaN payload/sign aside)."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(v), Float32(x)],
            "{ .reg .b32 t; and.b32 t, $2, 0x80000000; xor.b32 $0, t, $1; }",
            "=f,f,f",
            loc,
            ip,
        )
    )


def tree4(v, b4, b3):
    """Torch's 32-lane xor-butterfly sum (16, 8, 4, 2, 1) of 4 rows at once, broadcast to all lanes.

    Levels 16 and 8 trade halves between lane pairs so that each shuffle carries one row: after
    them, lanes with (lane & 24) == 0, 16, 8, 24 hold rows 0, 1, 2, 3. Each add combines the same
    two partials as the per-row butterfly (fp add is commutative), so every result is bitwise
    the per-row tree's.
    """
    s01 = fsel_nz(b4, v[0], v[1])
    k01 = fsel_nz(b4, v[1], v[0])
    s23 = fsel_nz(b4, v[2], v[3])
    k23 = fsel_nz(b4, v[3], v[2])
    a01 = fadd(k01, cute.arch.shuffle_sync_bfly(s01, 16))
    a23 = fadd(k23, cute.arch.shuffle_sync_bfly(s23, 16))
    sx = fsel_nz(b3, a01, a23)
    kx = fsel_nz(b3, a23, a01)
    a = fadd(kx, cute.arch.shuffle_sync_bfly(sx, 8))
    for off in (4, 2, 1):
        a = fadd(a, cute.arch.shuffle_sync_bfly(a, off))
    return [cute.arch.shuffle_sync(a, 0), cute.arch.shuffle_sync(a, 16), cute.arch.shuffle_sync(a, 8), cute.arch.shuffle_sync(a, 24)]


class RouterBwd:
    def __init__(self, bcast7: bool):
        self.bcast7 = bcast7
        self.nblk = 2 if bcast7 else 3  # 2 KB score blocks per group: in6, in12 (and in7 rows)

    @cute.jit
    def __call__(self, mIn0, mIn1, mIn3, mIn4, mIn5, mIn6, mIn7, mIn8, mIn9, mIn10, mIn11, mIn12, mOut, stream):
        self.kernel(mIn0, mIn1, mIn3, mIn4, mIn5, mIn6, mIn7, mIn8, mIn9, mIn10, mIn11, mIn12, mOut).launch(
            grid=[NGROUPS // NGW, 1, 1], block=[32, 1, 1], stream=stream
        )

    @cute.jit
    def group(self, fast: cutlass.Constexpr, lane, warp, r0, sBlk, sW, a0, a9, idx, d10, m1, m5w, g8r, n7r, v17, mOut):
        """Process this warp's GR rows. fast: div.rn fast path (the caller checked every operand
        range, and x6 has no zeros or infinities); otherwise IEEE div.rn and exact sgn."""
        colL = cute.make_layout((4, 32), stride=(1, 4))
        fragL = cute.make_layout(4)
        jt = lane // TOPK

        # ---- top-k: v36, v38, 8-wide sums (8-lane groups = rows), scatter_add into zeros ----
        v33 = fsel_nz(m1, Float32(0.0), a0)
        if cutlass.const_expr(fast):
            r10 = frcp_refined(d10)
            v36 = fdiv_fast(fdiv_fast(a9, d10, r10), d10, r10)
            v38 = fdiv_fast(v33, d10, r10)
        else:
            v36 = fdiv(fdiv(a9, d10), d10)
            v38 = fdiv(v33, d10)
        t8 = fnegmul(v33, v36)
        for off in cutlass.range_constexpr(3):
            t8 = fadd(t8, cute.arch.shuffle_sync_bfly(t8, 4 >> off))
        v41 = fadd(v38, t8)
        mm = cute.arch.match_sync(0xFFFFFFFF, idx + jt * NE)
        has_dup = cute.arch.vote_any_sync((mm & (mm - Uint32(1))) != Uint32(0))
        zf = cute.make_rmem_tensor(fragL, Float32)
        for e in cutlass.range_constexpr(4):
            zf[e] = Float32(0.0)
        for j in cutlass.range_constexpr(GR):
            cute.autovec_copy(zf, cute.composition(sW[None, j], colL)[None, lane])
        cute.arch.sync_warp()
        if has_dup:
            # deterministic scatter_add: duplicates accumulate in slot order from 0
            for src in cutlass.range_constexpr(32):
                ik = cute.arch.shuffle_sync(idx, src)
                vk = cute.arch.shuffle_sync(v41, src)
                if lane == 0:
                    if (ik & Int32(-NE)) == Int32(0):
                        sW[ik, src // TOPK] = fadd(sW[ik, src // TOPK], vk)
        else:
            # an index outside [0, NE) makes the reference raise; never write outside the row
            if (idx & Int32(-NE)) == Int32(0):
                sW[idx, jt] = fadd(Float32(0.0), v41)
        cute.arch.sync_warp()

        outL = cute.make_layout((2, 32, 3), stride=(1, 2, 64))
        mOutW = cute.recast_tensor(mOut, Uint32)
        b4 = Int32(lane & 16)
        b3 = Int32(lane & 8)

        def blk(b, j):
            return cute.composition(cute.make_tensor(sBlk[None, b, warp].iterator + j * NE, cute.make_layout(NE)), colL)[None, lane]

        # ---- P1: u = v20 * v23 (v24 = -u), v25 = v20 / n7; in-lane partial of sum(u) ----
        v25 = cute.make_rmem_tensor(cute.make_layout((4, GR), stride=(1, 4)), Float32)
        tpart = []
        for j in cutlass.range_constexpr(GR):
            x6 = cute.make_rmem_tensor(fragL, Float32)
            cute.autovec_copy(blk(0, j), x6)
            n7 = cute.make_rmem_tensor(fragL, Float32)
            if cutlass.const_expr(self.bcast7):
                for e in cutlass.range_constexpr(4):
                    n7[e] = n7r[j]
            else:
                cute.autovec_copy(blk(2, j), n7)
            m5f = fsel_nz(Int32((m5w >> Uint32(8 * j)) & Uint32(0xFF)), Float32(1.0), Float32(0.0))
            u = cute.make_rmem_tensor(fragL, Float32)
            if cutlass.const_expr(fast):
                r7 = cute.make_rmem_tensor(fragL, Float32)
                nn7 = cute.make_rmem_tensor(fragL, Float32)
                if cutlass.const_expr(self.bcast7):
                    rv = frcp_refined(n7[0])
                    for e in cutlass.range_constexpr(4):
                        r7[e] = rv
                        nn7[e] = -n7[0]
                else:
                    for e in cutlass.range_constexpr(4):
                        r7[e] = frcp_refined(n7[e])
                        nn7[e] = -n7[e]
                for p in cutlass.range_constexpr(2):
                    c0, c1 = 2 * p, 2 * p + 1
                    v20p = p_mul((v17[c0], v17[c1]), (m5f, m5f))
                    nnp = (nn7[c0], nn7[c1])
                    rp = (r7[c0], r7[c1])
                    v23p = p_div_fast(p_div_fast((x6[c0], x6[c1]), nnp, rp), nnp, rp)
                    u[c0], u[c1] = p_mul(v20p, v23p)
                    v25[c0, j], v25[c1, j] = p_div_fast(v20p, nnp, rp)
            else:
                for e in cutlass.range_constexpr(4):
                    v20 = fmul(v17[e], m5f)
                    u[e] = fmul(v20, fdiv(fdiv(x6[e], n7[e]), n7[e]))
                    v25[e, j] = fdiv(v20, n7[e])
            tpart.append(fadd(fadd(fadd(u[0], u[1]), u[2]), u[3]))

        # ---- P2: v26 = sum(v24) = -(sum of u), torch's tree, GR rows at once ----
        tsum = tree4(tpart, b4, b3)

        # ---- P3: v44 and tmp = v44 * out -> smem rows (for the transposed softmax sums) ----
        for j in cutlass.range_constexpr(GR):
            v29 = fsel_gt_1em12(g8r[j], -tsum[j])
            x6 = cute.make_rmem_tensor(fragL, Float32)
            y12 = cute.make_rmem_tensor(fragL, Float32)
            cute.autovec_copy(blk(0, j), x6)
            cute.autovec_copy(blk(1, j), y12)
            sV = cute.composition(sW[None, j], colL)[None, lane]
            v43 = cute.make_rmem_tensor(fragL, Float32)
            cute.autovec_copy(sV, v43)
            v31 = cute.make_rmem_tensor(fragL, Float32)
            if cutlass.const_expr(fast):
                # x6 is nonzero and finite here; a NaN x6 makes v29 NaN or +0, so its sign is moot
                for e in cutlass.range_constexpr(4):
                    v31[e] = fxorsign(v29, x6[e])
            else:
                z = fmul(Float32(0.0), v29)
                for e in cutlass.range_constexpr(4):
                    v31[e] = fsgn_mul_sel(x6[e], v29, z)
            tmp = cute.make_rmem_tensor(fragL, Float32)
            for p in cutlass.range_constexpr(2):
                c0, c1 = 2 * p, 2 * p + 1
                v32p = p_add((v25[c0, j], v25[c1, j]), (v31[c0], v31[c1]))
                v44p = p_add(v32p, (v43[c0], v43[c1]))
                tmp[c0], tmp[c1] = p_mul(v44p, (y12[c0], y12[c1]))
            cute.autovec_copy(tmp, sV)
        cute.arch.sync_warp()

        # ---- P4: softmax-backward sums: lane l adds tmp[l + 32 it], then the xor tree ----
        spart = []
        for j in cutlass.range_constexpr(GR):
            spart.append(fadd(fadd(fadd(fadd(Float32(0.0), sW[lane, j]), sW[lane + 32, j]), sW[lane + 64, j]), sW[lane + 96, j]))
        ssum = tree4(spart, b4, b3)

        # ---- P5: grad_in = fma(-out, sum, tmp) == fma(out, -sum, tmp); hi/mid/lo bf16; store ----
        for j in cutlass.range_constexpr(GR):
            y12 = cute.make_rmem_tensor(fragL, Float32)
            tmp = cute.make_rmem_tensor(fragL, Float32)
            cute.autovec_copy(blk(1, j), y12)
            cute.autovec_copy(cute.composition(sW[None, j], colL)[None, lane], tmp)
            nsm = -ssum[j]
            oFrag = cute.make_rmem_tensor(cute.make_layout((2, 3), stride=(1, 2)), Uint32)
            for p in cutlass.range_constexpr(2):
                c0, c1 = 2 * p, 2 * p + 1
                hi, mid, lo = bf16_split_pair(p_fma((y12[c0], y12[c1]), (nsm, nsm), (tmp[c0], tmp[c1])))
                oFrag[p, 0] = hi
                oFrag[p, 1] = mid
                oFrag[p, 2] = lo
            cute.autovec_copy(oFrag, cute.composition(mOutW[r0 + j, None], outL)[None, lane, None])

    @cute.kernel
    def kernel(self, mIn0, mIn1, mIn3, mIn4, mIn5, mIn6, mIn7, mIn8, mIn9, mIn10, mIn11, mIn12, mOut):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        g0 = bidx * NGW  # this warp's first group; it owns groups g0 .. g0 + NGW - 1

        smem = cutlass.utils.SmemAllocator()
        sBlk = smem.allocate_tensor(Float32, cute.make_layout((GR * NE, self.nblk, NGW)), byte_alignment=128)
        sScr = smem.allocate_tensor(Float32, cute.make_layout((NE, GR)), byte_alignment=128)
        mbar = smem.allocate_array(Int64, NGW, byte_alignment=8)

        if lane == 0:
            for q in cutlass.range_constexpr(NGW):
                cute.arch.mbarrier_init(mbar + q, 1)
            cute.arch.mbarrier_init_fence()
            for q in cutlass.range_constexpr(NGW):
                rq = (g0 + q) * GR
                cute.arch.mbarrier_arrive_and_expect_tx(mbar + q, self.nblk * GR * NE * 4)
                _bulk_g2s(sBlk[None, 0, q].iterator, mIn6.iterator + rq * NE, GR * NE * 4, mbar + q)
                _bulk_g2s(sBlk[None, 1, q].iterator, mIn12.iterator + rq * NE, GR * NE * 4, mbar + q)
                if cutlass.const_expr(not self.bcast7):
                    _bulk_g2s(sBlk[None, 2, q].iterator, mIn7.iterator + rq * NE, GR * NE * 4, mbar + q)
        cute.arch.sync_warp()  # mbarrier inits visible to every lane before they wait on them

        colL = cute.make_layout((4, 32), stride=(1, 4))  # (4 cols, 32 lanes) view of a 128-wide row
        fragL = cute.make_layout(4)

        # ---- small operands of every group (plain loads, in flight with the bulk copies) ----
        jt = lane // TOPK
        mIn5w = cute.make_tensor(cute.recast_ptr(mIn5.iterator, dtype=Uint32), cute.make_layout(NGROUPS))
        a0l, a9l, idxl, d10l, m1l, m5l, g8l, n7l = [], [], [], [], [], [], [], []
        for q in cutlass.range_constexpr(NGW):
            rq = (g0 + q) * GR
            a0l.append(mIn0[rq + jt, lane % TOPK])
            a9l.append(mIn9[rq + jt, lane % TOPK])
            idxl.append(Int32(mIn11[rq + jt, lane % TOPK]))
            d10l.append(mIn10[rq + jt, 0])
            m1l.append(Int32(mIn1[rq + jt, 0]))
            m5l.append(mIn5w[g0 + q])
            g8r = cute.make_rmem_tensor(cute.make_layout(GR), Float32)
            cute.autovec_copy(cute.make_tensor((mIn8.iterator + rq).align(16), cute.make_layout(GR)), g8r)
            g8l.append(g8r)
            n7r = cute.make_rmem_tensor(cute.make_layout(GR), Float32)
            if cutlass.const_expr(self.bcast7):
                cute.autovec_copy(cute.make_tensor((mIn7.iterator + rq).align(16), cute.make_layout(GR)), n7r)
            n7l.append(n7r)
        w4 = cute.make_rmem_tensor(fragL, Float32)
        cute.autovec_copy(cute.composition(mIn4, colL)[None, lane], w4)
        s3 = mIn3[0]
        v17 = cute.make_rmem_tensor(fragL, Float32)
        ok17 = Int32(1)
        for e in cutlass.range_constexpr(4):
            v17[e] = fmul(s3, w4[e])
            ok17 = ok17 & ok_dividend(v17[e])
        sW = sScr

        for q in cutlass.range_constexpr(NGW):
            rq = (g0 + q) * GR
            a0, a9, idx, d10, m1, m5w = a0l[q], a9l[q], idxl[q], d10l[q], m1l[q], m5l[q]
            g8r, n7r = g8l[q], n7l[q]
            # ---- fast-path eligibility of every division operand of this group ----
            # a9 joins the x6 range (both divided twice), d10 the divisor range
            ok = ok17 & ok_dividend(fsel_nz(m1, Float32(0.0), a0))
            nmax = fabsmax(Float32(0.0), d10)
            nmin = fabsmin(Float32(1.0), d10)
            if cutlass.const_expr(self.bcast7):
                nmax = fabsmax(fabsmax(fabsmax(fabsmax(nmax, n7r[0]), n7r[1]), n7r[2]), n7r[3])
                nmin = fabsmin(fabsmin(fabsmin(fabsmin(nmin, n7r[0]), n7r[1]), n7r[2]), n7r[3])
            mbar_wait(mbar + q, 0)
            xmax = fabsmax(Float32(0.0), a9)
            xmin = fabsmin(Float32(1.0), a9)
            for j in cutlass.range_constexpr(GR):
                x6 = cute.make_rmem_tensor(fragL, Float32)
                cute.autovec_copy(cute.composition(cute.make_tensor(sBlk[None, 0, q].iterator + j * NE, cute.make_layout(NE)), colL)[None, lane], x6)
                for e in cutlass.range_constexpr(4):
                    xmax = fabsmax(xmax, x6[e])
                    xmin = fabsmin(xmin, x6[e])
                if cutlass.const_expr(not self.bcast7):
                    n7 = cute.make_rmem_tensor(fragL, Float32)
                    cute.autovec_copy(cute.composition(cute.make_tensor(sBlk[None, 2, q].iterator + j * NE, cute.make_layout(NE)), colL)[None, lane], n7)
                    for e in cutlass.range_constexpr(4):
                        nmax = fabsmax(nmax, n7[e])
                        nmin = fabsmin(nmin, n7[e])
            ok = ok & in_range(xmin, xmax, _X6_LO, _X6_HI) & in_range(nmin, nmax, _N_LO, _N_HI)
            if cutlass.const_expr(q > 0):
                cute.arch.sync_warp()  # previous group's reads of the scratch rows are done
            if cute.arch.vote_all_sync(ok != Int32(0)):
                self.group(True, lane, q, rq, sBlk, sW, a0, a9, idx, d10, m1, m5w, g8r, n7r, v17, mOut)
            else:
                self.group(False, lane, q, rq, sBlk, sW, a0, a9, idx, d10, m1, m5w, g8r, n7r, v17, mOut)


_DT = {torch.float32: cutlass.Float32, torch.uint8: cutlass.Uint8, torch.int64: cutlass.Int64, torch.bfloat16: cutlass.BFloat16}
_CACHE = {}

# strides the compiled kernels assume (in7 either broadcast (1, 0) or contiguous)
_STRIDES = ((8, 1), (1, 1), (1,), (1,), (1, 1), (128, 1), None, (1, 1), (8, 1), (1, 1), (8, 1), (128, 1))


def _fake(t):
    return cute.runtime.make_fake_tensor(_DT[t.dtype], tuple(t.shape), stride=tuple(t.stride()), assumed_align=16)


def run(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10, in11, in12):
    out = torch.empty((ROWS, 3 * NE), dtype=torch.bfloat16, device=in6.device)
    args = [in0, in1.view(torch.uint8), in3.view(1), in4, in5.view(torch.uint8), in6, in7, in8, in9, in10, in11, in12]
    bcast7 = in7.stride() == (1, 0)
    for i, (t, st) in enumerate(zip(args, _STRIDES)):
        if st is None:  # in7
            if bcast7:
                if t.data_ptr() % 16:
                    # keep the broadcast layout: aligned copy of the row values, expanded again
                    col = torch.empty(ROWS, dtype=t.dtype, device=t.device)
                    col.copy_(t[:, 0])
                    t = col[:, None].expand(ROWS, NE)
                args[i] = t
                continue
            st = (NE, 1)
        # bulk copies and vector loads need 16-byte aligned operands with the expected strides
        if tuple(t.stride()) != st:
            t = t.contiguous()
        if t.data_ptr() % 16:
            t = t.clone()
        args[i] = t
    args.append(out)
    # the compiled variant is keyed by the layouts actually passed, so a mismatched one is never cached
    key = (bcast7, in6.device.index, tuple(tuple(t.stride()) for t in args))
    fn = _CACHE.get(key)
    if fn is None:
        fn = cute.compile(
            RouterBwd(bcast7),
            *[_fake(t) for t in args],
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[key] = fn
    fn(*args)
    return (out, out[:, 0:NE].t(), out[:, NE : 2 * NE].t(), out[:, 2 * NE :].t())
