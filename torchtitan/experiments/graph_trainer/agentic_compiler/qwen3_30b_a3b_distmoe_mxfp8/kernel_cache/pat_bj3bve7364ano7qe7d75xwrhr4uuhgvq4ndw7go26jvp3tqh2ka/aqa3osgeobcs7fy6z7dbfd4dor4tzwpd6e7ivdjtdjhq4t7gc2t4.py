"""Router aux-loss forward (G-91): L1 row norm, clamp_min, div, mul by a token mask, column sum.

    v2 = linalg_vector_norm(in0, 1, [-1], keepdim)        [32768, 1]
    v4 = clamp_min(v2, 1e-12).expand(32768, 128)          [32768, 128] stride (1, 0)
    v7 = sum((in0 / v4) * in1, 0)                         [128]

Two CuTe DSL kernels; K2 is a programmatic dependent launch of K1.

Reduction orders reproduced bitwise (torch Reduce.cuh on this build, checked by emu.py):
  * row L1 norm (vectorized input, block (32, 16)): lane l of a row's warp sums |x| of its float4
    (cols 4l..4l+3) in slot order from 0, then shfl_down 16, 8, 4, 2, 1.
  * clamp_min: NaN passes through (it is the canonical 0x7FFFFFFF after the adds), else
    max(v, fl32(1e-12)); computed as max.NaN.f32.
  * column sum over 32768 rows (vectorize-along-output 4, block (32, 4), 512 CTAs, global
    reduce): reference CTA b, warp w, lane x owns cols 4x..4x+3 and rows 4b + w + 2048 k,
    k = 0..15. Accumulator i (k % 4 == i) sums its rows in k order from 0; the thread value is
    ((a0 + a1) + a2) + a3; the CTA value (T0 + T2) + (T1 + T3); the last CTA's warp y sums the
    staged CTA values b = y, y + 4, ..., y + 508 in order from 0; the result (U0 + U2) + (U1 + U3).

K1 (512 CTAs x 4 warps): CTA b, warp w, lane l are the reference's column-sum CTA b, warp w, lane l. A warp loads
its 16 rows (a float4 per lane: exactly the reference norm's per-lane slice, and the lane's column-sum columns).
Row norms: the lanes' partial sums go through a butterfly reduce-scatter over the 16 rows (offsets 16, 8, 4, 2 halve
the rows a lane keeps, offset 1 finishes them; every add pairs the lanes the reference's shfl_down tree level pairs,
so each row gets lane 0's tree value), leaving every row in two lanes. Those lanes store v2 / v3 (through smem as one
16 B store per CTA and row group) and publish (d, r * mask) to the warp through smem.
Division: when every divisor of the warp is in [2^-20, 2^20] and every |x| >= 2^-60, z = x / d uses ptxas's
div.rn.f32 fast path with the divisor's refined reciprocal r hoisted (q0 = x * r; q = q0 + r * (x - d * q0)), which
equals div.rn.f32 in that range (checked by divtest.py; FCHK passes there). The mask is folded into r: for a masked
row r * 0 gives a zero z, as (x / d) * 0 does for the finite quotients of that range. Otherwise the warp uses
div.rn.f32 times the mask. Zero signs may differ from torch's, but none can reach the output: K2's in-order sums
start from +0, and a sum is -0 only if every addend is.
K2 (32 CTAs x 4 columns) does the final in-order reduction of the 512 CTA values.
All fp32 arithmetic is explicit round-to-nearest PTX without ftz (scalar or packed f32x2, which rounds each element
exactly like the scalar op), so no contraction or flushing.
"""
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NROW = 32768
NCOL = 128
NCTA = 512  # reference column-sum CTAs (ctas_per_output)
SLAB = NROW // 16  # 2048: rows of one reference thread are r0 + SLAB * k
_EPS = "0f2B8CBCCC"  # fl32(1e-12)


def _vals(args, loc, ip):
    return [a.ir_value(loc=loc, ip=ip) for a in args]


def _asm1(ty, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ty, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


def _asmN(n, ty, args, text, cons, loc, ip, side=False):
    tys = ty if isinstance(ty, (list, tuple)) else [ty] * n
    st = ir.Type.parse("!llvm.struct<(" + ", ".join(str(t) for t in tys) + ")>")
    res = llvm.inline_asm(
        st, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return [llvm.extractvalue(t, res, [k], loc=loc, ip=ip) for k, t in enumerate(tys)]


@dsl_user_op
def ld_x4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.f32(), [Int64(addr)], "ld.global.nc.L1::no_allocate.v4.f32 {$0, $1, $2, $3}, [$4];",
              "=f,=f,=f,=f,l", loc, ip, side=True)
    return [Float32(v) for v in r]


@dsl_user_op
def ld_cg4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.f32(), [Int64(addr)], "ld.global.cg.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,l", loc, ip,
              side=True)
    return [Float32(v) for v in r]


@dsl_user_op
def ld_u8(addr, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int64(addr)], "ld.global.nc.u8 $0, [$1];", "=r,l", loc, ip, side=True))


@dsl_user_op
def st_f32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Float32(a)], "st.global.f32 [$0], $1;", "l,f", loc, ip, side=True)


@dsl_user_op
def st_f32x4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Float32(a), Float32(b), Float32(c), Float32(d)],
          "st.global.v4.f32 [$0], {$1, $2, $3, $4};", "l,f,f,f,f", loc, ip, side=True)


@dsl_user_op
def sts_f32x4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Float32(a), Float32(b), Float32(c), Float32(d)],
          "st.shared.v4.f32 [$0], {$1, $2, $3, $4};", "r,f,f,f,f", loc, ip, side=True)


@dsl_user_op
def lds_f32x4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.f32(), [Int32(addr)], "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,r", loc, ip,
              side=True)
    return [Float32(v) for v in r]


@dsl_user_op
def sts_f32x2(addr, a, b, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Float32(a), Float32(b)], "st.shared.v2.f32 [$0], {$1, $2};", "r,f,f", loc, ip, side=True)


@dsl_user_op
def sts_f32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Float32(a)], "st.shared.f32 [$0], $1;", "r,f", loc, ip, side=True)


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int32(addr)], "ld.shared.f32 $0, [$1];", "=f,r", loc, ip, side=True))


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def clamp_eps(a, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a)], f"max.NaN.f32 $0, $1, {_EPS};", "=f,f", loc, ip))


@dsl_user_op
def ballot_ne0(v, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(v)],
                        "{ .reg .pred p; setp.ne.u32 p, $1, 0; vote.sync.ballot.b32 $0, p, 0xffffffff; }",
                        "=r,r", loc, ip, side=True))


@dsl_user_op
def bit_f32(mask, k: int, *, loc=None, ip=None):
    """1.0f if bit k of mask is set else 0.0f (the bool -> float cast of the mask)."""
    return Float32(_asm1(T.f32(), [Uint32(mask)],
                         f"{{ .reg .b32 t; bfe.u32 t, $1, {k}, 1; cvt.rn.f32.u32 $0, t; }}", "=f,r", loc, ip))


@dsl_user_op
def u8_f32(v, *, loc=None, ip=None):
    """1.0f if the mask byte is nonzero else 0.0f (torch loads bool as byte != 0)."""
    return Float32(_asm1(T.f32(), [Uint32(v)],
                         "{ .reg .pred p; setp.ne.u32 p, $1, 0; selp.f32 $0, 0f3F800000, 0f00000000, p; }", "=f,r",
                         loc, ip))


@dsl_user_op
def fmul(a, b, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a), Float32(b)], "mul.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def acc4(v0, v1, v2, v3, x0, x1, x2, x3, d, mf, *, loc=None, ip=None):
    """v_j + (div.rn(x_j, d) * mf) for j = 0..3."""
    r = _asmN(4, T.f32(), [Float32(v0), Float32(v1), Float32(v2), Float32(v3), Float32(x0), Float32(x1),
                           Float32(x2), Float32(x3), Float32(d), Float32(mf)],
              "{ .reg .f32 q0, q1, q2, q3;"
              " div.rn.f32 q0, $8, $12; div.rn.f32 q1, $9, $12; div.rn.f32 q2, $10, $12; div.rn.f32 q3, $11, $12;"
              " mul.rn.f32 q0, q0, $13; mul.rn.f32 q1, q1, $13; mul.rn.f32 q2, q2, $13; mul.rn.f32 q3, q3, $13;"
              " add.rn.f32 $0, $4, q0; add.rn.f32 $1, $5, q1; add.rn.f32 $2, $6, q2; add.rn.f32 $3, $7, q3; }",
              "=f,=f,=f,=f,f,f,f,f,f,f,f,f,f,f", loc, ip)
    return [Float32(v) for v in r]


@dsl_user_op
def shfl_down(a, off: int, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a)], f"shfl.sync.down.b32 $0, $1, {off}, 0x1f, 0xffffffff;", "=f,f",
                         loc, ip, side=True))


@dsl_user_op
def l1_min(a, b, c, d, mn, *, loc=None, ip=None):
    """(((|a| + |b|) + |c|) + |d|, min(mn, |a|, |b|, |c|, |d|))."""
    r = _asmN(2, T.f32(), [Float32(a), Float32(b), Float32(c), Float32(d), Float32(mn)],
              "{ .reg .f32 a, b, c, d, s; abs.f32 a, $2; abs.f32 b, $3; abs.f32 c, $4; abs.f32 d, $5;"
              " add.rn.f32 s, a, b; add.rn.f32 s, s, c; add.rn.f32 $0, s, d;"
              " min.f32 s, $6, a, b; min.f32 $1, s, c, d; }",
              "=f,=f,f,f,f,f,f", loc, ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def rs_sel(h, lo, hi, *, loc=None, ip=None):
    """(send, keep) = (lo, hi) if h else (hi, lo): reduce-scatter step that keeps row bit h."""
    r = _asmN(2, T.f32(), [Int32(h), Float32(lo), Float32(hi)],
              "{ .reg .pred p; setp.ne.s32 p, $2, 0; selp.f32 $0, $3, $4, p; selp.f32 $1, $4, $3, p; }",
              "=f,=f,r,f,f", loc, ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def fast_ok(d, mn, *, loc=None, ip=None):
    """1 if every lane has 2^-20 <= d <= 2^20 and min|x| >= 2^-60 (div.rn.f32 fast-path range), else 0."""
    return Int32(_asm1(T.i32(), [Float32(d), Float32(mn)],
                       "{ .reg .pred p, q; setp.ge.f32 p, $1, 0f35800000; setp.le.and.f32 p, $1, 0f49800000, p;"
                       " setp.ge.and.f32 p, $2, 0f21800000, p; vote.sync.all.pred q, p, 0xffffffff;"
                       " selp.s32 $0, 1, 0, q; }",
                       "=r,f,f", loc, ip, side=True))


@dsl_user_op
def frcp_refined(n, *, loc=None, ip=None):
    """Divisor-only part of ptxas's div.rn.f32 fast path: r0 = MUFU.RCP(n); r = r0 + r0 * (1 - n * r0)."""
    return Float32(_asm1(T.f32(), [Float32(n)],
                         "{ .reg .f32 r0, e, nn; neg.f32 nn, $1; rcp.approx.ftz.f32 r0, $1;"
                         " fma.rn.f32 e, nn, r0, 0f3F800000; fma.rn.f32 $0, r0, e, r0; }",
                         "=f,f", loc, ip))


@dsl_user_op
def fneg(a, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a)], "neg.f32 $0, $1;", "=f,f", loc, ip))


def p_add(a, b):
    return cute.arch.add_packed_f32x2(a, b, rnd="rn")


def p_mul(a, b):
    return cute.arch.mul_packed_f32x2(a, b, rnd="rn")


def p_fma(a, b, c):
    return cute.arch.fma_packed_f32x2(a, b, c, rnd="rn")


def p_div_fast(x, nn, r):
    """ptxas's div.rn.f32 fast path (valid when its FCHK passes) with the reciprocal hoisted: q0 = x * r;
    q = q0 + r * (x - n * q0). Equals div.rn.f32 except for the sign of a zero quotient."""
    q0 = p_mul(x, r)
    return p_fma(r, p_fma(nn, q0, x), q0)


def _src_lane(k):
    """Lane holding row k after the reduce-scatter: row bits k0..k3 sit in lane bits 4, 3, 2, 1."""
    return ((k & 1) << 4) | (((k >> 1) & 1) << 3) | (((k >> 2) & 1) << 2) | (((k >> 3) & 1) << 1)


class RowColKernel:
    """K1: CTA b = reference column-sum CTA b; warp w = its warp; lane l = its lane (cols 4l..4l+3)."""

    @cute.jit
    def __call__(self, x, m, o2, o3, part, stream):
        self.kernel(x, m, o2, o3, part).launch(grid=[NCTA, 1, 1], block=[128, 1, 1], stream=stream)

    @cute.jit
    def accum(self, fast: cutlass.Constexpr, xs, ds, rs, nds, mask, sw):
        """Column accumulators of the warp's 16 rows (k order), combined to the thread value, into smem.

        fast: z = div.rn fast path with the reciprocal times the row's mask value (rs = r * m: for m = 0 this gives a
        zero z, as (x / d) * 0 does for the finite quotients the fast-path range guarantees; zero signs cannot reach
        the output). Accumulator i starts as z_i (torch seeds it with 0 + z_i: again only a zero sign differs, and the
        final in-order sums in K2 start from +0). Otherwise div.rn.f32 times the mask, from 0 seeds.
        """
        zero = Float32(0.0)
        vl = [[zero, zero, zero, zero] for _ in range(4)]
        for k in cutlass.range_constexpr(16):
            x0, x1, x2, x3 = xs[k]
            a = vl[k % 4]
            if cutlass.const_expr(fast):
                q01 = p_div_fast((x0, x1), (nds[k], nds[k]), (rs[k], rs[k]))
                q23 = p_div_fast((x2, x3), (nds[k], nds[k]), (rs[k], rs[k]))
                if cutlass.const_expr(k < 4):
                    vl[k % 4] = [q01[0], q01[1], q23[0], q23[1]]
                else:
                    lo = p_add((a[0], a[1]), q01)
                    hi = p_add((a[2], a[3]), q23)
                    vl[k % 4] = [lo[0], lo[1], hi[0], hi[1]]
            else:
                vl[k % 4] = acc4(a[0], a[1], a[2], a[3], x0, x1, x2, x3, ds[k], bit_f32(mask, _src_lane(k)))
        t01 = p_add(p_add(p_add((vl[0][0], vl[0][1]), (vl[1][0], vl[1][1])), (vl[2][0], vl[2][1])),
                    (vl[3][0], vl[3][1]))
        t23 = p_add(p_add(p_add((vl[0][2], vl[0][3]), (vl[1][2], vl[1][3])), (vl[2][2], vl[2][3])),
                    (vl[3][2], vl[3][3]))
        sts_f32x4(sw, t01[0], t01[1], t23[0], t23[1])

    @cute.kernel
    def kernel(self, x, m, o2, o3, part):
        tid, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        w = tid // 32
        lane = tid % 32
        smem = cutlass.utils.SmemAllocator()
        # [4 w][32 lanes] float4 thread values; v2 / v3 staging [2][16 k][4 w] floats; per-warp (d, r) [4 w][16 k]
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(4 * 32 * 16 + 2 * 16 * 16 + 4 * 16 * 8),
                                    byte_alignment=16)
        sb = sAll.iterator.toint()
        r0 = b * 4 + w
        px = x.iterator.toint() + Int64(r0) * (NCOL * 4) + Int64(lane * 16)
        # lane L fetches the mask byte of the row it will own after the norm's reduce-scatter (row bits k0..k3 =
        # lane bits 4..1); bit _src_lane(k) of the ballot is then row k's mask
        k_own = ((lane >> 4) & 1) | (((lane >> 3) & 1) << 1) | (((lane >> 2) & 1) << 2) | (((lane >> 1) & 1) << 3)
        mb = ld_u8(m.iterator.toint() + Int64(r0 + k_own * SLAB))
        mask = ballot_ne0(mb)
        xs = []
        for k in cutlass.range_constexpr(16):
            xs.append(ld_x4(px + Int64(k * SLAB * NCOL * 4)))
        # per-lane partial L1 sums of the 16 rows (and the lane's min |x| for the fast-division check)
        mn = Float32(float("inf"))
        vals = []
        for k in cutlass.range_constexpr(16):
            x0, x1, x2, x3 = xs[k]
            a, mn = l1_min(x0, x1, x2, x3, mn)
            vals.append(a)
        # butterfly reduce-scatter over the 16 rows: offsets 16, 8, 4, 2 each halve the rows a lane keeps (row bit
        # t kept = lane bit 4 - t), each add pairs the same lanes as the reference's shfl_down tree level; offset 1
        # finishes the row in both lanes of a pair
        for t in cutlass.range_constexpr(4):
            h = (lane >> (4 - t)) & 1
            nv = []
            for i in cutlass.range_constexpr(len(vals) // 2):
                send, keep = rs_sel(h, vals[2 * i], vals[2 * i + 1])
                nv.append(fadd(keep, cute.arch.shuffle_sync_bfly(send, offset=16 >> t)))
            vals = nv
        s = fadd(vals[0], cute.arch.shuffle_sync_bfly(vals[0], offset=1))
        d = clamp_eps(s)
        # v2 / v3 of the CTA's rows 4b + w + SLAB k go out as one 16 B store per k (via smem): fewer, wider writes
        if lane % 2 == 0:
            sv = sb + 4 * 32 * 16 + (k_own * 4 + w) * 4
            sts_f32(sv, s)
            sts_f32(sv + 16 * 16, d)
        ok = fast_ok(d, mn)
        r = fmul(frcp_refined(d), u8_f32(mb))
        # every lane needs (d, r * mask) of all 16 rows: the owners write them to smem, every lane reads them back as
        # broadcast loads (cheaper on the shared shuffle / smem pipe than 32 shuffles)
        sdr = sb + 4 * 32 * 16 + 2 * 16 * 16 + w * 16 * 8
        if lane % 2 == 0:
            sts_f32x2(sdr + k_own * 8, d, r)
        cute.arch.sync_warp()
        ds = []
        rs = []
        nds = []
        for k2 in cutlass.range_constexpr(8):
            v = lds_f32x4(sdr + k2 * 16)
            ds.append(v[0])
            rs.append(v[1])
            nds.append(fneg(v[0]))
            ds.append(v[2])
            rs.append(v[3])
            nds.append(fneg(v[2]))
        sw = sb + (w * 32 + lane) * 16
        if ok != 0:
            self.accum(True, xs, ds, rs, nds, mask, sw)
        else:
            self.accum(False, xs, ds, rs, nds, mask, sw)
        cute.arch.barrier()
        if w == 1:
            v = lds_f32x4(sb + 4 * 32 * 16 + lane * 16)  # lanes 0..15: v2 row group k = lane; 16..31: v3, k = lane - 16
            kk = lane % 16
            dst = o2.iterator.toint()
            if lane >= 16:
                dst = o3.iterator.toint()
            st_f32x4(dst + Int64(b * 4 + kk * SLAB) * 4, v[0], v[1], v[2], v[3])
        if w == 0:
            t0 = lds_f32x4(sb + lane * 16)
            t1 = lds_f32x4(sb + (32 + lane) * 16)
            t2 = lds_f32x4(sb + (64 + lane) * 16)
            t3 = lds_f32x4(sb + (96 + lane) * 16)
            bv = [fadd(fadd(t0[j], t2[j]), fadd(t1[j], t3[j])) for j in range(4)]
            st_f32x4(part.iterator.toint() + Int64(b * NCOL + lane * 4) * 4, bv[0], bv[1], bv[2], bv[3])
        cute.arch.griddepcontrol_launch_dependents()


class ColFinalKernel:
    """K2: CTA c4 reduces cols 4 c4..4 c4+3 over the 512 K1 CTA values in the reference's last-CTA order."""

    @cute.jit
    def __call__(self, part, out, stream):
        self.kernel(part, out).launch(grid=[NCOL // 4, 1, 1], block=[128, 1, 1], stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(self, part, out):
        tid, _, _ = cute.arch.thread_idx()
        c4, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(NCTA * 16), byte_alignment=16)
        sb = sAll.iterator.toint()
        cute.arch.griddepcontrol_wait()
        pp = part.iterator.toint() + Int64(c4 * 16)
        vs = []
        for i in cutlass.range_constexpr(NCTA // 128):
            vs.append(ld_cg4(pp + Int64((tid + 128 * i) * NCOL * 4)))
        for i in cutlass.range_constexpr(NCTA // 128):
            v = vs[i]
            sts_f32x4(sb + (tid + 128 * i) * 16, v[0], v[1], v[2], v[3])
        cute.arch.barrier()
        if tid < 32:
            y = (tid // 4) % 4
            j = tid % 4
            u = Float32(0.0)
            base = sb + y * 16 + j * 4
            for jj in cutlass.range_constexpr(NCTA // 4):
                u = fadd(u, lds_f32(base + jj * 64))
            a = fadd(u, shfl_down(u, 8))  # lanes j: U0 + U2, lanes 4 + j: U1 + U3
            f = fadd(a, shfl_down(a, 4))
            if tid < 4:
                st_f32(out.iterator.toint() + Int64(c4 * 4 + j) * 4, f)


_DT = {torch.float32: cutlass.Float32, torch.uint8: cutlass.Uint8}


def _fake(dtype, shape, align=16):
    stride, a = [], 1
    for s in reversed(shape):
        stride.append(a)
        a *= s
    return cute.runtime.make_fake_tensor(_DT[dtype], tuple(shape), stride=tuple(reversed(stride)), assumed_align=align)


_ST = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
_k1 = cute.compile(
    RowColKernel(),
    _fake(torch.float32, (NROW, NCOL)),
    _fake(torch.uint8, (NROW,), align=1),
    _fake(torch.float32, (NROW,), align=4),
    _fake(torch.float32, (NROW,), align=4),
    _fake(torch.float32, (NCTA, NCOL)),
    _ST,
    options="--enable-tvm-ffi",
)
_k2 = cute.compile(
    ColFinalKernel(),
    _fake(torch.float32, (NCTA, NCOL)),
    _fake(torch.float32, (NCOL,), align=4),
    _ST,
    options="--enable-tvm-ffi",
)


def _reference(in0, in1):
    v2 = torch.ops.aten.linalg_vector_norm.default(in0, 1, [-1], True)
    v3 = torch.ops.aten.clamp_min.default(v2, 1e-12)
    v4 = torch.ops.aten.expand.default(v3, [32768, 128])
    v5 = torch.ops.aten.div.Tensor(in0, v4)
    v6 = torch.ops.aten.mul.Tensor(v5, in1)
    v7 = torch.ops.aten.sum.dim_IntList(v6, [0])
    return (v2, v4, v7)


def _supported(in0, in1):
    if not (isinstance(in0, torch.Tensor) and isinstance(in1, torch.Tensor) and in0.is_cuda
            and in1.device == in0.device):
        return False
    if (in0.dtype != torch.float32 or tuple(in0.shape) != (NROW, NCOL) or tuple(in0.stride()) != (NCOL, 1)
            or in0.data_ptr() % 16):
        return False
    if in1.dtype != torch.bool or tuple(in1.shape) != (NROW, 1) or in1.stride(0) != 1:
        return False
    # The reference column sum runs 512 CTAs when num_SMs * (max threads per SM / 128) >= 512.
    p = torch.cuda.get_device_properties(in0.device)
    return p.multi_processor_count * (p.max_threads_per_multi_processor // 128) >= NCTA


def run(in0, in1):
    if not _supported(in0, in1):
        return _reference(in0, in1)
    dev = in0.device
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        v2 = torch.empty((NROW, 1), dtype=torch.float32, device=dev)
        v3 = torch.empty((NROW, 1), dtype=torch.float32, device=dev)
        part = torch.empty((NCTA, NCOL), dtype=torch.float32, device=dev)
        v7 = torch.empty((NCOL,), dtype=torch.float32, device=dev)
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = fill
    _k1(in0, in1.view(torch.uint8).view(NROW), v2.view(NROW), v3.view(NROW), part)
    _k2(part, v7)
    return (v2, v3.expand(NROW, NCOL), v7)
