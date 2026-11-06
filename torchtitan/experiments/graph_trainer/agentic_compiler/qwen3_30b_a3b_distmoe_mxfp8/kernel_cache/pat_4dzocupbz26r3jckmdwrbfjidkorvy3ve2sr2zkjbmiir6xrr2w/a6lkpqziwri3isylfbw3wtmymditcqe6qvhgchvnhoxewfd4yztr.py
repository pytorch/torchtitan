"""Qwen3 30B-A3B attention forward glue (pat_4dzoc / a6lkpq, fusion region G02), one CuTe DSL kernel.

reference: in0 [T=32768, 5120] bf16 (wqkv output) viewed [T, 4, 10, 128]; per kv group g: heads 0..7 -> q,
           head 8 -> k, head 9 -> v. Outputs (in order):
           q [T,32,128] bf16 (contiguous clone), k [T,4,128], v [T,4,128],
           rstd_q [T,32,1] f32, rstd_k [T,4,1] f32  (aten._fused_rms_norm over 128, eps 1e-6, weights in1/in2),
           cos [T,1,128], sin [T,1,128] f32: views (stride 256,256,1) of one gathered [T,256] buffer in3[in4],
           q_rope [T,32,128] bf16, k_rope [T,4,128] bf16 (RoPE of the bf16 rms-norm outputs, fp32 math).

Schedule: persistent, NWARP warps per CTA, one CTA per SM, every warp independent with a private ring of NST
smem stages (token row 10 KB + cos/sin row 1 KB). Tokens are handed out dynamically: lane 0 claims the next
token with an atomic add on a per-call zeroed counter (SMs drain HBM at different rates; a static split left
the slowest SMs finishing ~30% after the fastest). Claims are prefetched two refills ahead (claim -> position
load -> stage load) so neither latency is on the critical path. Lane 0 bulk-loads a stage (cp.async.bulk,
mbarrier complete_tx; L2 evict_first for in0, evict_last for the cos/sin table); the warp computes 4 heads per
step (8 lanes per head, lane t owns head elements 8t..8t+7 and 64+8t..64+8t+7, so RoPE partners are
lane-local) and stores q/k (raw and RoPE), v, rstd and the cos/sin row from registers.

Numerics (explicit PTX, nothing left for the compiler to contract):
* RMSNorm reproduces the kernel aten._fused_rms_norm dispatches to on this build (torch._native override ->
  vendored quack RMSNorm, N=128 bf16: 16 threads per row, 8 elements per thread; decoded from its SASS):
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

NTOK = 32768
ROW_B = 5120 * 2  # bytes of one in0 row
CS_B = 256 * 4  # bytes of one cos/sin table row
STAGE_B = ROW_B + CS_B  # 11264
NWARP = 4  # measured: 4 warps x 2-3 stages per SM is the best of 2..16 warps x 1..8 stages
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
def claim(ctr, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int64(ctr)], "atom.global.add.u32 $0, [$1], 1;", "=r,l", loc, ip, side=True))


@dsl_user_op
def ld_pos(pos, tok, *, loc=None, ip=None):
    """Table row of position pos[tok] (aten.index wraps negative indices by the 4096 table rows); 0 if
    tok >= NTOK (no token left)."""
    return Int32(_asm1(
        T.i32(), [Int64(pos), Int32(tok)],
        "{ .reg .s64 p; .reg .pred v, n;\n"
        "mov.s64 p, 0;\n"
        f"setp.lt.u32 v, $2, {NTOK};\n"
        "mad.wide.u32 p, $2, 8, $1;\n"
        "@v ld.global.nc.s64 p, [p];\n"
        "@!v mov.s64 p, 0;\n"
        "setp.lt.s64 n, p, 0;\n"
        "@n add.s64 p, p, 4096;\n"
        "cvt.u32.u64 $0, p; }",
        "=r,l,r", loc, ip, side=True))


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
def load_stage(bar, dst, x0, tok, tab0, prow, *, loc=None, ip=None):
    """If tok < NTOK: arm bar for STAGE_B bytes and bulk-load in0 row tok (L2 evict_first) and cos/sin table row
    prow (L2 evict_last) into stage dst."""
    _asm1(
        None, [Int32(bar), Int32(dst), Int64(x0), Int32(tok), Int64(tab0), Int32(prow)],
        "{ .reg .b32 d; .reg .b64 a, c, pf, pl; .reg .pred v;\n"
        f"setp.lt.u32 v, $3, {NTOK};\n"
        "@!v bra SKIP_${:uid};\n"
        "createpolicy.fractional.L2::evict_first.b64 pf, 1.0;\n"
        "createpolicy.fractional.L2::evict_last.b64 pl, 1.0;\n"
        f"mad.wide.u32 a, $3, {ROW_B}, $2;\n"
        f"mad.wide.u32 c, $5, {CS_B}, $4;\n"
        f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {STAGE_B};\n"
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$1], [a], {ROW_B}, [$0], pf;\n"
        f"add.u32 d, $1, {ROW_B};\n"
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [d], [c], {CS_B}, [$0], pl;\n"
        "SKIP_${:uid}: }",
        "r,r,l,r,l,r", loc, ip, side=True,
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


class AttnGlueFwd:
    def __init__(self, num_ctas: int):
        self.num_ctas = num_ctas

    @cute.jit
    def __call__(self, x, wq, wk, tab, pos, ctr, qraw, kraw, vraw, rq, rk, cs, qrope, krope, stream):
        self.kernel(x, wq, wk, tab, pos, ctr, qraw, kraw, vraw, rq, rk, cs, qrope, krope).launch(
            grid=[self.num_ctas, 1, 1], block=[NTHR, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, x, wq, wk, tab, pos, ctr, qraw, kraw, vraw, rq, rk, cs, qrope, krope):
        tid, _, _ = cute.arch.thread_idx()
        wid = tid // 32
        lane = tid % 32
        grp = lane // 8
        t = lane % 8

        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(SMEM_B), byte_alignment=128)
        sb = sAll.iterator.toint()
        stage0 = sb + wid * (NST * STAGE_B)
        bar0 = sb + BAR_OFF + wid * (NST * 8)
        stok0 = sb + TOK_OFF + wid * (NST * 4)

        x0 = x.iterator.toint()
        tab0 = tab.iterator.toint()
        pos0 = pos.iterator.toint()
        ctr0 = ctr.iterator.toint()
        qraw0 = qraw.iterator.toint()
        kraw0 = kraw.iterator.toint()
        vraw0 = vraw.iterator.toint()
        cs0 = cs.iterator.toint()
        qrope0 = qrope.iterator.toint()
        krope0 = krope.iterator.toint()
        rq0 = rq.iterator.toint()
        rk0 = rk.iterator.toint()

        # weights at this lane's columns (A: 8t..8t+7, B: 64+8t..64+8t+7) as f32x2 pairs
        wq0 = wq.iterator.toint() + Int64(t * 16)
        wk0 = wk.iterator.toint() + Int64(t * 16)
        wqa = [unpack_bf16x2(v) for v in ldg_v4(wq0)]
        wqb = [unpack_bf16x2(v) for v in ldg_v4(wq0 + 128)]
        wka = [unpack_bf16x2(v) for v in ldg_v4(wk0)]
        wkb = [unpack_bf16x2(v) for v in ldg_v4(wk0 + 128)]

        # lane 0 claims tokens; n1 (with its table row p1) is the next stage's token, n2 the one after
        n1 = Int32(0)
        p1 = Int32(0)
        n2 = Int32(0)
        if lane == 0:
            for s in cutlass.range_constexpr(NST):
                mbar_init(bar0 + s * 8)
            mbar_init_fence()
            for s in cutlass.range_constexpr(NST):
                n0 = claim(ctr0)
                load_stage(bar0 + s * 8, stage0 + s * STAGE_B, x0, n0, tab0, ld_pos(pos0, n0))
                sts_b32(stok0 + s * 4, n0)
            n1 = claim(ctr0)
            p1 = ld_pos(pos0, n1)
            n2 = claim(ctr0)
        syncwarp()

        k = Int32(0)
        tok = lds_b32(stok0)
        while tok < NTOK:
            s = k % NST
            st = stage0 + s * STAGE_B
            tok64 = Int64(tok)
            mbar_wait(bar0 + s * 8, (k // NST) % 2)

            # v heads and the cos/sin row: smem -> registers -> global
            vsrc = st + grp * 2560 + 2304 + t * 16
            vdst = vraw0 + tok64 * 1024 + Int64(grp * 256 + t * 16)
            v0 = lds_v4(vsrc)
            v1 = lds_v4(vsrc + 128)
            c0 = lds_v4(st + ROW_B + lane * 16)
            c1 = lds_v4(st + ROW_B + 512 + lane * 16)
            st_v4(vdst, v0[0], v0[1], v0[2], v0[3])
            st_v4(vdst + 128, v1[0], v1[1], v1[2], v1[3])
            cdst = cs0 + tok64 * CS_B + Int64(lane * 16)
            st_v4(cdst, c0[0], c0[1], c0[2], c0[3])
            st_v4(cdst + 512, c1[0], c1[1], c1[2], c1[3])

            # cos/sin of this lane's columns: [cos A | cos B | sin A | sin B], 8 floats each (4 f32x2)
            csb = st + ROW_B + t * 32
            cA = lds_v2b64(csb) + lds_v2b64(csb + 16)
            cB = lds_v2b64(csb + 256) + lds_v2b64(csb + 272)
            sA = lds_v2b64(csb + 512) + lds_v2b64(csb + 528)
            sB = lds_v2b64(csb + 768) + lds_v2b64(csb + 784)
            nsA = [neg_f32x2(v) for v in sA]

            for it in cutlass.range_constexpr(9):
                is_q = it < 8
                if cutlass.const_expr(is_q):
                    # q head h = 4 it + grp: kv group h // 8, slot h % 8
                    head = it * 4 + grp
                    hoff = st + grp * 256 + t * 16 + ((it // 2) * 2560 + (it % 2) * 1024)
                    raw_addr = qraw0 + tok64 * 8192 + Int64(head * 256 + t * 16)
                    rope_addr = qrope0 + tok64 * 8192 + Int64(head * 256 + t * 16)
                    r_addr = rq0 + tok64 * 128 + Int64(head * 4)
                else:
                    hoff = st + grp * 2560 + 2048 + t * 16
                    raw_addr = kraw0 + tok64 * 1024 + Int64(grp * 256 + t * 16)
                    rope_addr = krope0 + tok64 * 1024 + Int64(grp * 256 + t * 16)
                    r_addr = rk0 + tok64 * 16 + Int64(grp * 4)
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
                st_v4(raw_addr, xa[0], xa[1], xa[2], xa[3])
                st_v4(raw_addr + 128, xb[0], xb[1], xb[2], xb[3])
                st_v4(rope_addr, oa[0], oa[1], oa[2], oa[3])
                st_v4(rope_addr + 128, ob[0], ob[1], ob[2], ob[3])
                if t == 0:
                    st_f32(r_addr, r)

            # every lane is done reading the stage: refill it with token n1, advance the claim pipeline
            syncwarp()
            if lane == 0:
                load_stage(bar0 + s * 8, st, x0, n1, tab0, p1)
                sts_b32(stok0 + s * 4, n1)
                n1 = n2
                p1 = ld_pos(pos0, n2)
                n2 = claim(ctr0)
            syncwarp()
            k = k + 1
            tok = lds_b32(stok0 + (k % NST) * 4)


_DT = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32, torch.int64: cutlass.Int64,
       torch.int32: cutlass.Int32}


def _fake(dtype, shape):
    stride, acc = [], 1
    for s in reversed(shape):
        stride.append(acc)
        acc *= s
    return cute.runtime.make_fake_tensor(_DT[dtype], tuple(shape), stride=tuple(reversed(stride)), assumed_align=16)


_SIG = [
    (torch.bfloat16, (NTOK, 5120)),  # x
    (torch.bfloat16, (128,)),  # wq
    (torch.bfloat16, (128,)),  # wk
    (torch.float32, (4096, 256)),  # cos/sin table
    (torch.int64, (NTOK,)),  # positions
    (torch.int32, (1,)),  # token counter (zeroed per call)
    (torch.bfloat16, (NTOK, 4096)),  # q
    (torch.bfloat16, (NTOK, 512)),  # k
    (torch.bfloat16, (NTOK, 512)),  # v
    (torch.float32, (NTOK, 32)),  # rstd_q
    (torch.float32, (NTOK, 4)),  # rstd_k
    (torch.float32, (NTOK, 256)),  # gathered cos/sin
    (torch.bfloat16, (NTOK, 4096)),  # q_rope
    (torch.bfloat16, (NTOK, 512)),  # k_rope
]

_NUM_CTAS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count

_fn = cute.compile(
    AttnGlueFwd(_NUM_CTAS),
    *[_fake(d, s) for d, s in _SIG],
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def _dense(t):
    """The kernel uses 16-byte vector and bulk accesses on contiguous rows."""
    return t if t.is_contiguous() and t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.contiguous_format)


def run(in0, in1, in2, in3, in4):
    x, wq, wk, tab, pos = _dense(in0), _dense(in1), _dense(in2), _dense(in3), in4.contiguous()
    dev = x.device
    bf, f32 = torch.bfloat16, torch.float32
    ctr = torch.zeros(1, dtype=torch.int32, device=dev)
    q = torch.empty((NTOK, 32, 128), dtype=bf, device=dev)
    k = torch.empty((NTOK, 4, 128), dtype=bf, device=dev)
    v = torch.empty((NTOK, 4, 128), dtype=bf, device=dev)
    rq = torch.empty((NTOK, 32, 1), dtype=f32, device=dev)
    rk = torch.empty((NTOK, 4, 1), dtype=f32, device=dev)
    cs = torch.empty((NTOK, 256), dtype=f32, device=dev)
    q_rope = torch.empty((NTOK, 32, 128), dtype=bf, device=dev)
    k_rope = torch.empty((NTOK, 4, 128), dtype=bf, device=dev)
    _fn(x, wq, wk, tab, pos, ctr, q.view(NTOK, 4096), k.view(NTOK, 512), v.view(NTOK, 512), rq.view(NTOK, 32),
        rk.view(NTOK, 4), cs, q_rope.view(NTOK, 4096), k_rope.view(NTOK, 512))
    cs3 = cs.view(NTOK, 1, 256)
    return (q, k, v, rq, rk, cs3[:, :, :128], cs3[:, :, 128:], q_rope, k_rope)
