"""Qwen3 235B-A22B attention forward glue (pat_xfyh / azk7g7, fusion region G102), one CuTe DSL kernel.

reference: in2 [T=8192, 9216] bf16 (wqkv output) viewed [T, 4, 18, 128]; per kv group g: heads 0..15 -> q,
           head 16 -> k, head 17 -> v. in0/in1 [256] bf16: q/k norm weights are their first 128 elements.
           in3/in4 [T,1,128] fp32: cos/sin rows (row stride 256 in the spec, 128 in the real capture).
           Outputs (in order):
           wq = in0[:128], wk = in1[:128] (views of the inputs),
           q [T,64,128] bf16 (contiguous clone), q_rope [T,64,128] bf16, rstd_q [T,64,1] f32,
           k [T,4,128] bf16, k_rope [T,4,128] bf16, rstd_k [T,4,1] f32, v [T,4,128] bf16.
           (aten._fused_rms_norm over 128, eps 1e-6; RoPE of the bf16 rms-norm outputs in fp32.)

Schedule (from the 30B G02 kernel, ada t5_dyn_final): persistent, NWARP warps per CTA, one CTA per SM, every
warp independent with a private ring of NST smem stages. The work unit is half a token row (u = 2 tok + half:
kv groups 2 half, 2 half + 1 = 9216 contiguous bytes at in2 + u * 9216) plus that token's cos and sin rows
(512 B each), so a stage is 10 KB (the 30B stage was one 10 KB token row + 1 KB table row) and units are short
enough for a balanced tail at T=8192. The first NST units of every warp are assigned statically
(s * num_warps + warp); later units are claimed dynamically by lane 0 with an atomic add on a per-call zeroed
counter, prefetched two refills ahead. Lane 0 bulk-loads a stage (cp.async.bulk, mbarrier complete_tx; L2
evict_first for in2); the warp copies the 2 v heads, computes 4 heads per step (8 lanes per head, lane t owns
head elements 8t..8t+7 and 64+8t..64+8t+7, so RoPE partners are lane-local): 8 steps for the 32 q heads, one
step for the 2 k heads (lane groups 2, 3 duplicate groups 0, 1 and do not store), and stores q/k (raw and
RoPE), v and rstd from registers.

Numerics (explicit PTX, nothing left for the compiler to contract), unchanged from the 30B kernel:
* RMSNorm reproduces the kernel aten._fused_rms_norm dispatches to on this build (torch._native override ->
  vendored quack RMSNorm; its forward config depends only on N, so N=128 bf16 is as in 30B: 16 threads per
  row, 8 elements per thread):
  s_c = sequential fma.rn.f32.bf16 chain over elements 8c..8c+7 from +0 (FHFMA.BF16); butterfly FADD with
  offsets 8, 4, 2, 1 (offset 8 pairs chunk t with chunk t+8, which this kernel holds in one lane);
  rstd = rsqrt.approx.ftz(fma.rn(sum, 1/128, fl32(1e-6))); y = bf16_rn((x *rn rstd) *rn w).
* RoPE as the eager reference (separate kernels, every op rounded to fp32):
  out[d]    = bf16_rn( (y[d] *rn cos[d])       +rn (-y[d+64] *rn sin[d]) )      d < 64
  out[d+64] = bf16_rn( (y[d+64] *rn cos[d+64]) +rn ( y[d]    *rn sin[d+64]) )
  (-a) * s is computed as a * (-s) (sign flip of s, exact; NaN results are canonical either way).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NTOK = 8192
NU = 2 * NTOK  # work units: half token rows
ROW_ELEMS = 9216
GRP_B = 18 * 256  # bytes of one kv group (16 q heads, k, v) in a row
UNIT_B = 2 * GRP_B  # 9216 bytes of in2 per unit
CSH_B = 128 * 4  # bytes of one cos (or sin) row
STAGE_B = UNIT_B + 2 * CSH_B  # 10240: [x half row | cos row | sin row]
NWARP = 4
NST = 3
NTHR = NWARP * 32
BAR_OFF = NWARP * NST * STAGE_B
TOK_OFF = BAR_OFF + NWARP * NST * 8
SMEM_B = TOK_OFF + NWARP * NST * 4
EPS_HEX = "0f358637BD"  # fl32(1e-6)


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


# ---------------------------------------------------------------- memory / sync
@dsl_user_op
def ldg_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int32(addr)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,r", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def lds_v2b64(addr, *, loc=None, ip=None):
    r = _asmN(2, T.i64(), [Int32(addr)], "ld.shared.v2.b64 {$0, $1}, [$2];", "=l,=l,r", loc, ip, side=True)
    return [Int64(v) for v in r]


@dsl_user_op
def lds_b32(addr, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int32(addr)], "ld.shared.b32 $0, [$1];", "=r,r", loc, ip, side=True))


@dsl_user_op
def sts_b32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Int32(a)], "st.shared.b32 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def st_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def st_f32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Float32(a)], "st.global.f32 [$0], $1;", "l,f", loc, ip, side=True)


@dsl_user_op
def mbar_init(addr, *, loc=None, ip=None):
    _asm1(None, [Int32(addr)], "mbarrier.init.shared::cta.b64 [$0], 1;", "r", loc, ip, side=True)


@dsl_user_op
def mbar_init_fence(*, loc=None, ip=None):
    _asm1(None, [], "fence.mbarrier_init.release.cluster;", "", loc, ip, side=True)


@dsl_user_op
def mbar_wait(addr, parity, *, loc=None, ip=None):
    _asm1(
        None, [Int32(addr), Int32(parity)],
        "{ .reg .pred p;\n"
        "WAIT_${:uid}:\n"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [$0], $1;\n"
        "@!p bra WAIT_${:uid}; }",
        "r,r", loc, ip, side=True,
    )


@dsl_user_op
def syncwarp(*, loc=None, ip=None):
    _asm1(None, [], "bar.warp.sync -1;", "", loc, ip, side=True)


# ---------------------------------------------------------------- math
@dsl_user_op
def sumsq16(a0, a1, a2, a3, b0, b1, b2, b3, *, loc=None, ip=None):
    """fl32 s_A + s_B, s_X = fma.rn.f32.bf16 chain over the 8 bf16 of words X0..X3 (lo half first) from +0."""
    return Float32(_asm1(
        T.f32(), [Uint32(a0), Uint32(a1), Uint32(a2), Uint32(a3), Uint32(b0), Uint32(b1), Uint32(b2), Uint32(b3)],
        "{ .reg .f32 a, b; .reg .b16 l, h, m, n;\n"
        "mov.b32 {l, h}, $1; mov.b32 {m, n}, $5;\n"
        "fma.rn.f32.bf16 a, l, l, 0f00000000; fma.rn.f32.bf16 b, m, m, 0f00000000;\n"
        "fma.rn.f32.bf16 a, h, h, a; fma.rn.f32.bf16 b, n, n, b;\n"
        "mov.b32 {l, h}, $2; mov.b32 {m, n}, $6;\n"
        "fma.rn.f32.bf16 a, l, l, a; fma.rn.f32.bf16 b, m, m, b;\n"
        "fma.rn.f32.bf16 a, h, h, a; fma.rn.f32.bf16 b, n, n, b;\n"
        "mov.b32 {l, h}, $3; mov.b32 {m, n}, $7;\n"
        "fma.rn.f32.bf16 a, l, l, a; fma.rn.f32.bf16 b, m, m, b;\n"
        "fma.rn.f32.bf16 a, h, h, a; fma.rn.f32.bf16 b, n, n, b;\n"
        "mov.b32 {l, h}, $4; mov.b32 {m, n}, $8;\n"
        "fma.rn.f32.bf16 a, l, l, a; fma.rn.f32.bf16 b, m, m, b;\n"
        "fma.rn.f32.bf16 a, h, h, a; fma.rn.f32.bf16 b, n, n, b;\n"
        "add.rn.f32 $0, a, b; }",
        "=f,r,r,r,r,r,r,r,r", loc, ip))


@dsl_user_op
def bfly_add(v, off: int, *, loc=None, ip=None):
    return Float32(_asm1(
        T.f32(), [Float32(v)],
        "{ .reg .f32 o;\n"
        f"shfl.sync.bfly.b32 o, $1, {off}, 31, -1;\n"
        "add.rn.f32 $0, $1, o; }",
        "=f,f", loc, ip, side=True))


@dsl_user_op
def rstd_of(total, *, loc=None, ip=None):
    return Float32(_asm1(
        T.f32(), [Float32(total)],
        "{ .reg .f32 m;\n"
        f"fma.rn.f32 m, $1, 0f3C000000, {EPS_HEX};\n"
        "rsqrt.approx.ftz.f32 $0, m; }",
        "=f,f", loc, ip))


@dsl_user_op
def unpack_bf16x2(w, *, loc=None, ip=None):
    """bf16x2 word -> f32x2 (lo element in the low 32 bits)."""
    return Int64(_asm1(
        T.i64(), [Uint32(w)],
        "{ .reg .b32 l, h;\n"
        "shl.b32 l, $1, 16; and.b32 h, $1, 0xFFFF0000;\n"
        "mov.b64 $0, {l, h}; }",
        "=l,r", loc, ip))


@dsl_user_op
def neg_f32x2(v, *, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [Int64(v)], "xor.b64 $0, $1, 0x8000000080000000;", "=l,l", loc, ip))


@dsl_user_op
def norm_rope(xa, xb, wa, wb, r, ca, cb, nsa, sb, *, loc=None, ip=None):
    """One bf16x2 word of chunk A (elements d, d+1 < 64) and the same word of chunk B (d+64, d+65).

    With y = bf16_rn((x *rn r) *rn w) (the rms-norm output) returns the RoPE outputs
    oa = bf16_rn(ya*cos_a + yb*(-sin_a)), ob = bf16_rn(yb*cos_b + ya*sin_b), every op rn fp32.
    wa, wb, ca, cb, nsa, sb are f32x2 pairs.
    """
    res = _asmN(
        2, T.i32(),
        [Uint32(xa), Uint32(xb), Int64(wa), Int64(wb), Float32(r), Int64(ca), Int64(cb), Int64(nsa), Int64(sb)],
        "{ .reg .b32 l, h, ya, yb; .reg .b64 a, b, rr, t, u;\n"
        "mov.b64 rr, {$6, $6};\n"
        "shl.b32 l, $2, 16; and.b32 h, $2, 0xFFFF0000; mov.b64 a, {l, h};\n"
        "shl.b32 l, $3, 16; and.b32 h, $3, 0xFFFF0000; mov.b64 b, {l, h};\n"
        "mul.rn.f32x2 a, a, rr; mul.rn.f32x2 a, a, $4;\n"
        "mul.rn.f32x2 b, b, rr; mul.rn.f32x2 b, b, $5;\n"
        "mov.b64 {l, h}, a; cvt.rn.bf16x2.f32 ya, h, l;\n"
        "mov.b64 {l, h}, b; cvt.rn.bf16x2.f32 yb, h, l;\n"
        "shl.b32 l, ya, 16; and.b32 h, ya, 0xFFFF0000; mov.b64 a, {l, h};\n"
        "shl.b32 l, yb, 16; and.b32 h, yb, 0xFFFF0000; mov.b64 b, {l, h};\n"
        "mul.rn.f32x2 t, a, $7; mul.rn.f32x2 u, b, $9; add.rn.f32x2 t, t, u;\n"
        "mov.b64 {l, h}, t; cvt.rn.bf16x2.f32 $0, h, l;\n"
        "mul.rn.f32x2 t, b, $8; mul.rn.f32x2 u, a, $10; add.rn.f32x2 t, t, u;\n"
        "mov.b64 {l, h}, t; cvt.rn.bf16x2.f32 $1, h, l; }",
        "=r,=r,r,r,l,l,f,l,l,l,l", loc, ip,
    )
    return Uint32(res[0]), Uint32(res[1])


@dsl_user_op
def claim(ctr, base, *, loc=None, ip=None):
    """base + the counter's previous value (counter zeroed per call)."""
    return Int32(_asm1(T.i32(), [Int64(ctr), Int32(base)],
                       "{ .reg .u32 c;\n"
                       "atom.global.add.u32 c, [$1], 1;\n"
                       "add.u32 $0, c, $2; }", "=r,l,r", loc, ip, side=True))


@dsl_user_op
def load_stage(bar, dst, x0, u, cos0, sin0, cs_b, sn_b, *, loc=None, ip=None):
    """If u < NU: arm bar for STAGE_B bytes and bulk-load in2 bytes [u * UNIT_B, +UNIT_B) (L2 evict_first),
    the cos row and the sin row of token u / 2 (row strides cs_b, sn_b bytes) into stage dst."""
    _asm1(
        None, [Int32(bar), Int32(dst), Int64(x0), Int32(u), Int64(cos0), Int64(sin0), Int32(cs_b), Int32(sn_b)],
        "{ .reg .b32 d, tk; .reg .b64 a, c, s, pf; .reg .pred v;\n"
        f"setp.lt.u32 v, $3, {NU};\n"
        "@!v bra SKIP_${:uid};\n"
        "createpolicy.fractional.L2::evict_first.b64 pf, 1.0;\n"
        f"mad.wide.u32 a, $3, {UNIT_B}, $2;\n"
        "shr.u32 tk, $3, 1;\n"
        "mad.wide.u32 c, tk, $6, $4;\n"
        "mad.wide.u32 s, tk, $7, $5;\n"
        f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {STAGE_B};\n"
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$1], [a], {UNIT_B}, [$0], pf;\n"
        f"add.u32 d, $1, {UNIT_B};\n"
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [d], [c], {CSH_B}, [$0];\n"
        f"add.u32 d, d, {CSH_B};\n"
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [d], [s], {CSH_B}, [$0];\n"
        "SKIP_${:uid}: }",
        "r,r,l,r,l,l,r,r", loc, ip, side=True,
    )


class AttnGlueFwd:
    def __init__(self, num_ctas: int):
        self.num_ctas = num_ctas

    @cute.jit
    def __call__(self, x, wq, wk, cos, sin, ctr, qraw, qrope, rq, kraw, krope, rk, vraw, stream):
        self.kernel(x, wq, wk, cos, sin, ctr, qraw, qrope, rq, kraw, krope, rk, vraw).launch(
            grid=[self.num_ctas, 1, 1], block=[NTHR, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, x, wq, wk, cos, sin, ctr, qraw, qrope, rq, kraw, krope, rk, vraw):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        wid = tid // 32
        lane = tid % 32
        grp = lane // 8
        t = lane % 8
        nwt = self.num_ctas * NWARP  # warps in the grid
        wg = bid * NWARP + wid

        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(SMEM_B), byte_alignment=128)
        sb = sAll.iterator.toint()
        stage0 = sb + wid * (NST * STAGE_B)
        bar0 = sb + BAR_OFF + wid * (NST * 8)
        stok0 = sb + TOK_OFF + wid * (NST * 4)

        x0 = x.iterator.toint()
        cos0 = cos.iterator.toint()
        sin0 = sin.iterator.toint()
        cs_b = Int32(cos.stride[0] * 4)
        sn_b = Int32(sin.stride[0] * 4)
        ctr0 = ctr.iterator.toint()
        qraw0 = qraw.iterator.toint()
        kraw0 = kraw.iterator.toint()
        vraw0 = vraw.iterator.toint()
        qrope0 = qrope.iterator.toint()
        krope0 = krope.iterator.toint()
        rq0 = rq.iterator.toint()
        rk0 = rk.iterator.toint()

        # weights (first 128 elements of in0 / in1) at this lane's columns (A: 8t..8t+7, B: 64+8t..64+8t+7)
        wq0 = wq.iterator.toint() + Int64(t * 16)
        wk0 = wk.iterator.toint() + Int64(t * 16)
        wqa = [unpack_bf16x2(v) for v in ldg_v4(wq0)]
        wqb = [unpack_bf16x2(v) for v in ldg_v4(wq0 + 128)]
        wka = [unpack_bf16x2(v) for v in ldg_v4(wk0)]
        wkb = [unpack_bf16x2(v) for v in ldg_v4(wk0 + 128)]

        # first NST units static (s * nwt + wg); later units claimed: n1 is the next stage's unit, n2 the one after
        base = Int32(NST * nwt)
        n1 = Int32(0)
        n2 = Int32(0)
        if lane == 0:
            for s in cutlass.range_constexpr(NST):
                mbar_init(bar0 + s * 8)
            mbar_init_fence()
            for s in cutlass.range_constexpr(NST):
                n0 = Int32(s * nwt) + wg
                load_stage(bar0 + s * 8, stage0 + s * STAGE_B, x0, n0, cos0, sin0, cs_b, sn_b)
                sts_b32(stok0 + s * 4, n0)
            n1 = claim(ctr0, base)
            n2 = claim(ctr0, base)
        syncwarp()

        k = Int32(0)
        u = lds_b32(stok0)
        while u < NU:
            s = k % NST
            st = stage0 + s * STAGE_B
            u64 = Int64(u)
            mbar_wait(bar0 + s * 8, (k // NST) % 2)

            # the unit's 2 v heads (512 B): lane -> group lane // 16, 16 B chunk lane % 16
            vv = lds_v4(st + (lane // 16) * GRP_B + 17 * 256 + (lane % 16) * 16)
            st_v4(vraw0 + u64 * 512 + Int64(lane * 16), vv[0], vv[1], vv[2], vv[3])

            # cos/sin of this lane's columns: [cos A | cos B | sin A | sin B], 8 floats each (4 f32x2)
            csb = st + UNIT_B + t * 32
            cA = lds_v2b64(csb) + lds_v2b64(csb + 16)
            cB = lds_v2b64(csb + 256) + lds_v2b64(csb + 272)
            sA = lds_v2b64(csb + 512) + lds_v2b64(csb + 528)
            sB = lds_v2b64(csb + 768) + lds_v2b64(csb + 784)
            nsA = [neg_f32x2(v) for v in sA]

            for it in cutlass.range_constexpr(9):
                is_q = it < 8
                if cutlass.const_expr(is_q):
                    # unit-local q head hl = 4 it + grp (group hl // 16, slot hl % 16); global head 32 half + hl
                    hoff = st + (it // 4) * GRP_B + (it % 4) * 1024 + grp * 256 + t * 16
                    raw_addr = qraw0 + u64 * 8192 + Int64(it * 1024 + grp * 256 + t * 16)
                    rope_addr = qrope0 + u64 * 8192 + Int64(it * 1024 + grp * 256 + t * 16)
                    r_addr = rq0 + u64 * 128 + Int64((it * 4 + grp) * 4)
                else:
                    # k head of group g = grp % 2 (global kv head 2 half + g); groups 2, 3 duplicate, no store
                    g = grp % 2
                    hoff = st + g * GRP_B + 16 * 256 + t * 16
                    raw_addr = kraw0 + u64 * 512 + Int64(g * 256 + t * 16)
                    rope_addr = krope0 + u64 * 512 + Int64(g * 256 + t * 16)
                    r_addr = rk0 + u64 * 8 + Int64(g * 4)
                xa = lds_v4(hoff)
                xb = lds_v4(hoff + 128)
                tot = sumsq16(xa[0], xa[1], xa[2], xa[3], xb[0], xb[1], xb[2], xb[3])
                tot = bfly_add(tot, 4)
                tot = bfly_add(tot, 2)
                tot = bfly_add(tot, 1)
                r = rstd_of(tot)
                wa = wqa if cutlass.const_expr(is_q) else wka
                wb = wqb if cutlass.const_expr(is_q) else wkb
                oa = [None] * 4
                ob = [None] * 4
                for i in cutlass.range_constexpr(4):
                    oa[i], ob[i] = norm_rope(xa[i], xb[i], wa[i], wb[i], r, cA[i], cB[i], nsA[i], sB[i])
                if cutlass.const_expr(is_q):
                    st_v4(raw_addr, xa[0], xa[1], xa[2], xa[3])
                    st_v4(raw_addr + 128, xb[0], xb[1], xb[2], xb[3])
                    st_v4(rope_addr, oa[0], oa[1], oa[2], oa[3])
                    st_v4(rope_addr + 128, ob[0], ob[1], ob[2], ob[3])
                    if t == 0:
                        st_f32(r_addr, r)
                else:
                    if lane < 16:
                        st_v4(raw_addr, xa[0], xa[1], xa[2], xa[3])
                        st_v4(raw_addr + 128, xb[0], xb[1], xb[2], xb[3])
                        st_v4(rope_addr, oa[0], oa[1], oa[2], oa[3])
                        st_v4(rope_addr + 128, ob[0], ob[1], ob[2], ob[3])
                        if t == 0:
                            st_f32(r_addr, r)

            # every lane is done reading the stage: refill it with unit n1, advance the claim pipeline
            syncwarp()
            if lane == 0:
                load_stage(bar0 + s * 8, st, x0, n1, cos0, sin0, cs_b, sn_b)
                sts_b32(stok0 + s * 4, n1)
                n1 = n2
                n2 = claim(ctr0, base)
            syncwarp()
            k = k + 1
            u = lds_b32(stok0 + (k % NST) * 4)


_DT = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32, torch.int32: cutlass.Int32}


def _fake(dtype, shape, stride=None):
    if stride is None:
        stride, acc = [], 1
        for s in reversed(shape):
            stride.append(acc)
            acc *= s
        stride = tuple(reversed(stride))
    return cute.runtime.make_fake_tensor(_DT[dtype], tuple(shape), stride=tuple(stride), assumed_align=16)


def _fake_cs():
    # cos/sin [T, 1, 128] fp32 with a runtime row stride (multiple of 4 elements: 16 B aligned rows)
    return _fake(torch.float32, (NTOK, 1, 128), (cute.sym_int(divisibility=4), cute.sym_int(), 1))


_SIG = [
    _fake(torch.bfloat16, (NTOK, ROW_ELEMS)),  # x = in2
    _fake(torch.bfloat16, (256,)),  # in0 (wq = first 128)
    _fake(torch.bfloat16, (256,)),  # in1 (wk = first 128)
    _fake_cs(),  # cos = in3
    _fake_cs(),  # sin = in4
    _fake(torch.int32, (1,)),  # unit counter (zeroed per call)
    _fake(torch.bfloat16, (NTOK, 8192)),  # q
    _fake(torch.bfloat16, (NTOK, 8192)),  # q_rope
    _fake(torch.float32, (NTOK, 64)),  # rstd_q
    _fake(torch.bfloat16, (NTOK, 512)),  # k
    _fake(torch.bfloat16, (NTOK, 512)),  # k_rope
    _fake(torch.float32, (NTOK, 4)),  # rstd_k
    _fake(torch.bfloat16, (NTOK, 512)),  # v
]

_NUM_CTAS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

_fn = cute.compile(
    AttnGlueFwd(_NUM_CTAS),
    *_SIG,
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def _check(t, name, shape, dtype):
    if tuple(t.shape) != shape or t.dtype != dtype or not t.is_cuda:
        raise ValueError(f"{name}: expected {shape} {dtype} on cuda, got {tuple(t.shape)} {t.dtype} {t.device}")


def _dense(t):
    """16-byte vector and bulk accesses need contiguous, 16 B aligned rows."""
    return t if t.is_contiguous() and t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.contiguous_format)


def _rows(t):
    """cos/sin [T, 1, 128]: any row stride that keeps the 512 B rows contiguous and 16 B aligned."""
    if t.stride(2) == 1 and t.stride(0) % 4 == 0 and t.data_ptr() % 16 == 0:
        return t
    return t.contiguous()


def run(in0, in1, in2, in3, in4):
    _check(in0, "in0", (256,), torch.bfloat16)
    _check(in1, "in1", (256,), torch.bfloat16)
    _check(in2, "in2", (NTOK, ROW_ELEMS), torch.bfloat16)
    _check(in3, "in3", (NTOK, 1, 128), torch.float32)
    _check(in4, "in4", (NTOK, 1, 128), torch.float32)
    x, w0, w1, cos, sin = _dense(in2), _dense(in0), _dense(in1), _rows(in3), _rows(in4)
    dev = x.device
    bf, f32 = torch.bfloat16, torch.float32
    ctr = torch.zeros(1, dtype=torch.int32, device=dev)
    q = torch.empty((NTOK, 64, 128), dtype=bf, device=dev)
    q_rope = torch.empty((NTOK, 64, 128), dtype=bf, device=dev)
    rq = torch.empty((NTOK, 64, 1), dtype=f32, device=dev)
    k = torch.empty((NTOK, 4, 128), dtype=bf, device=dev)
    k_rope = torch.empty((NTOK, 4, 128), dtype=bf, device=dev)
    rk = torch.empty((NTOK, 4, 1), dtype=f32, device=dev)
    v = torch.empty((NTOK, 4, 128), dtype=bf, device=dev)
    _fn(x, w0, w1, cos, sin, ctr, q.view(NTOK, 8192), q_rope.view(NTOK, 8192), rq.view(NTOK, 64),
        k.view(NTOK, 512), k_rope.view(NTOK, 512), rk.view(NTOK, 4), v.view(NTOK, 512))
    return (in0.view(256)[:128], in1.view(256)[:128], q, q_rope, rq, k, k_rope, rk, v)
