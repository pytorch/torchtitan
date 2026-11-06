"""Qwen3 30B-A3B attention backward glue (pat_qnfera / armbn25, fusion region G01), CuTe DSL.

reference: RoPE backward of the k grad in0 [T,4,128] and the q grad in1 [T,32,128] (fp32 in2 / in3 [T,1,128]),
           aten._fused_rms_norm_backward for k (x in5, rstd in4, w in6) and q (x in8, rstd in7, w in9) over 128,
           cat per kv group g of [q heads 8g..8g+7 | k head g | v head g (in10)] -> [T, 5120], torchao
           mxfp8_quantize (rowwise + colwise, 32-blocks, e4m3, rceil) and triton_mx_block_rearrange of both scales.
           Returns (dw_k, dw_q, qrow, srow_blocked, scol_blocked, qcol.t()).

Work units (32 tokens x 4 heads x 128 cols): q units (block T, head group hg = 4 heads), k units (block T, the 4
k heads) and v units. CTA c (one per SM) owns the q units of (G = c % 19, hg = c // 19) -- the blocks
T = G + 19 k -- because aten's dw_q is a sum of quack per-CTA accumulators (B = 16 * 152 = 2432 persistent CTAs:
quack CTA b = 128 G + 4 i + q' accumulates exactly token i of the blocks G, G + 19, ... in that order), and one k
class G' (blocks T = G' + 152 m, quack CTAs b = 16 G' + i // 2). v units are spread to balance the load. A CTA
runs its k units, then its q units, then its v units, each in block order (static per-CTA table).

Warp specialization (384 threads, setmaxnreg): warpgroups 0-1 are 8 math warps (216 registers), warpgroup 2 is 4
quant warps (64 registers), so each SM sub-partition runs 2 math warps + 1 quant warp.
* Math thread (warp w, lane = 8 (i % 4) + l): token i = 4 w + lane // 8 of the unit, columns 8l..8l+7 (chunk A)
  and 64+8l..64+8l+7 (chunk B) of each head row (RoPE partners and quack's first butterfly step are
  thread-local), all 4 heads. Each math warp loads its own g / x rows into a private 2-stage smem ring (q: one
  TMA tile box per tensor, 4 tokens x 1 KB; k / v: 4 KB bulk copies; mbarrier complete_tx); cos/sin/rstd go
  straight to registers. It writes bf16 dx into a double-buffered swizzled smem tile, computes the rowwise MXFP8
  scales (amax reduce-scatter over 4 lanes), quantizes its own dx chunks rowwise and stores qrow / srow.
* Quant warp q: colwise MXFP8 of tile columns 128 q .. 128 q + 127 (ldmatrix.trans: 8 columns x 32 tokens per
  instruction, lane = column, 8 consecutive tokens) and the qcol / scol stores.
  Tile handoff: tfull (256 math thread arrivals) / tempty (128 quant thread arrivals) mbarriers, no CTA barrier in the loop.
* dw: each math thread's k and q accumulators are written as partial data; a second kernel (one CTA per
  (kind, column), bulk-copied into smem) rebuilds quack's per-CTA partial rows and runs reduce_rows' exact order.

Numerics (all explicit PTX, rn, nothing contractible):
* RoPE bwd: d[c] = bf16(add0(rn(s_c rn(g[c^64] in2[c^64]) + rn(g[c] in3[c])))), s_c = +1 (c < 64) / -1; the
  reference's slice_backward +0 is applied in fp32 before the rounding (rn(add0(a) + b) == add0(rn(a + b))).
* RMSNorm bwd = the vendored quack kernel aten dispatches to (N=128 bf16 on sm_103, from its SASS):
  w' = w + 0; xh = rn(x rstd); wdy = rn(w' d); S = fma chain over a lane's 8 cols from +0, butterfly 8,4,2,1;
  mean = rn(S 2^-7); dx = bf16(rn(rstd fma(-xh, mean, wdy))); acc[b][r] = fma(xh, d, acc) over CTA b's tiles,
  partial[b] = ((acc0 + acc1) + ...) + acc7; dw = reduce_rows(partial, cols=4, threads=256) (torch/_native).
* MXFP8 (torchao rceil): amax = NaN-propagating max |dx| per 32-block; byte = 255 if a >= 0x7f80 else
  max(((a + 31) >> 7) - 8, 0) (a = bf16 bits of amax); q = cvt.rn.satfinite.e4m3(dx * 2^(127 - byte)) computed as
  a bf16x2 multiply (the product of a bf16 by a power of two is exact down to the bf16 normal range; anything
  smaller is an e4m3 zero of the same sign either way) and cvt.e4m3x2.bf16x2; byte 255 -> recip NaN.
* triton_mx_block_rearrange: scale (r, c) of an [R, C/32] scale matrix -> (c // 4) * 512 + (r // 128) * 128 * C/32
  + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T, dsl_user_op

NTOK = 32768
NBLK = NTOK // 32
NCOL = 5120
NSM = 152
NQB = 16 * NSM  # quack persistent CTAs (2432)
NTHR = 384  # warpgroups 0, 1: math warps (RoPE / norm bwd / dw -> bf16 dx tile); warpgroup 2: MXFP8 quant warps
NMATH = 256
MREG = 216  # setmaxnreg budgets: 2 math warps + 1 quant warp per SM sub-partition (<= 512 registers per lane slot)
QREG = 64
MAXU = 80
STAGE_B = 8192  # per warp: g [4 tok][1 KB], x [4 tok][1 KB]
OFF_TILE = 8 * 2 * STAGE_B
TILE_B = 32768  # [32 tok][512 cols] bf16
OFF_ACC = OFF_TILE + 2 * TILE_B  # dw accumulators of heads 2, 3: [16-byte unit j][thread]
OFF_BAR = OFF_ACC + 8 * NMATH * 16
OFF_UT = OFF_BAR + 160
SMEM_B = OFF_UT + 4 * (MAXU + 1)
KQ, KK, KV = 0, 1, 2


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


# ---------------------------------------------------------------- memory
@dsl_user_op
def ldg_u32(addr, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int64(addr)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip, side=True))


@dsl_user_op
def ldg_f32(addr, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int64(addr)], "ld.global.f32 $0, [$1];", "=f,l", loc, ip, side=True))


@dsl_user_op
def ldg_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc, ip,
              side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def ldg_p2(addr, *, loc=None, ip=None):
    """Two f32x2 pairs (16 bytes)."""
    r = _asmN(2, T.i64(), [Int64(addr)], "ld.global.v2.b64 {$0, $1}, [$2];", "=l,=l,l", loc, ip, side=True)
    return [Int64(v) for v in r]


@dsl_user_op
def ldcg_p2(addr, *, loc=None, ip=None):
    r = _asmN(2, T.i64(), [Int64(addr)], "ld.global.cg.v2.b64 {$0, $1}, [$2];", "=l,=l,l", loc, ip, side=True)
    return [Int64(v) for v in r]


@dsl_user_op
def stg_p2(addr, a, b, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Int64(a), Int64(b)], "st.global.v2.b64 [$0], {$1, $2};", "l,l,l", loc, ip, side=True)


@dsl_user_op
def stg_v2(addr, a, b, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b)], "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", loc, ip, side=True)


@dsl_user_op
def stg_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)], "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
          "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def prmt(a, b, sel_: int, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], f"prmt.b32 $0, $1, $2, {sel_:#x};", "=r,r,r", loc, ip))


@dsl_user_op
def stg_b32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a)], "st.global.b32 [$0], $1;", "l,r", loc, ip, side=True)


@dsl_user_op
def stg_u8(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a)], "st.global.b8 [$0], $1;", "l,r", loc, ip, side=True)


@dsl_user_op
def stg_f32(addr, a, *, loc=None, ip=None):
    """L2 evict_last store (partial data is re-read by the reduce kernel)."""
    _asm1(None, [Int64(addr), Float32(a)],
          "{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0;\n"
          "st.global.L2::cache_hint.f32 [$0], $1, pl; }", "l,f", loc, ip, side=True)


@dsl_user_op
def stg_f4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Float32(a), Float32(b), Float32(c), Float32(d)],
          "{ .reg .b64 pl; createpolicy.fractional.L2::evict_last.b64 pl, 1.0;\n"
          "st.global.L2::cache_hint.v4.f32 [$0], {$1, $2, $3, $4}, pl; }", "l,f,f,f,f", loc, ip, side=True)


@dsl_user_op
def add_fv(a, b, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip, side=True))


@dsl_user_op
def ldg_f4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.f32(), [Int64(addr)], "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];", "=f,=f,=f,=f,l", loc, ip,
              side=True)
    return [Float32(v) for v in r]


@dsl_user_op
def ldg_f2(addr, *, loc=None, ip=None):
    r = _asmN(2, T.f32(), [Int64(addr)], "ld.global.v2.f32 {$0, $1}, [$2];", "=f,=f,l", loc, ip, side=True)
    return [Float32(v) for v in r]


@dsl_user_op
def split_p(v, *, loc=None, ip=None):
    r = _asmN(2, T.f32(), [Int64(v)], "mov.b64 {$0, $1}, $2;", "=f,=f,l", loc, ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def stg_b16(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a)], "{ .reg .b16 h; cvt.u16.u32 h, $1; st.global.b16 [$0], h; }", "l,r", loc,
          ip, side=True)


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int32(addr)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", loc, ip,
              side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def lds_p2(addr, *, loc=None, ip=None):
    r = _asmN(2, T.i64(), [Int32(addr)], "ld.shared.v2.b64 {$0, $1}, [$2];", "=l,=l,r", loc, ip, side=True)
    return [Int64(v) for v in r]


@dsl_user_op
def sts_p2(addr, a, b, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Int64(a), Int64(b)], "st.shared.v2.b64 [$0], {$1, $2};", "r,l,l", loc, ip, side=True)


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int32(addr)], "ld.shared.f32 $0, [$1];", "=f,r", loc, ip, side=True))


@dsl_user_op
def lds_b32(addr, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int32(addr)], "ld.shared.b32 $0, [$1];", "=r,r", loc, ip, side=True))


@dsl_user_op
def sts_b32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Int32(a)], "st.shared.b32 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def sts_f32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Float32(a)], "st.shared.f32 [$0], $1;", "r,f", loc, ip, side=True)


@dsl_user_op
def sts_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
          "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def ldsm_x4t(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int32(addr)], "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,r", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def mbar_init(addr, cnt: int, *, loc=None, ip=None):
    _asm1(None, [Int32(addr)], f"mbarrier.init.shared::cta.b64 [$0], {cnt};", "r", loc, ip, side=True)


@dsl_user_op
def mbar_init_fence(*, loc=None, ip=None):
    _asm1(None, [], "fence.mbarrier_init.release.cluster;", "", loc, ip, side=True)


@dsl_user_op
def mbar_arrive(addr, *, loc=None, ip=None):
    _asm1(None, [Int32(addr)], "mbarrier.arrive.shared::cta.b64 _, [$0];", "r", loc, ip, side=True)


@dsl_user_op
def mbar_expect(addr, nbytes, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Int32(nbytes)], "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;", "r,r", loc,
          ip, side=True)


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
def bulk_1k(dst, src, bar, *, loc=None, ip=None):
    """1 KB global -> shared bulk copy completing on bar (L2 evict_first)."""
    _asm1(
        None, [Int32(dst), Int64(src), Int32(bar)],
        "{ .reg .b64 pf;\n"
        "createpolicy.fractional.L2::evict_first.b64 pf, 1.0;\n"
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], 1024, [$2], pf; }",
        "r,l,r", loc, ip, side=True,
    )


@dsl_user_op
def stg_v2_if(addr, a, b, ok, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Int32(ok)],
          "{ .reg .pred p; setp.ne.s32 p, $3, 0; @p st.global.v2.b32 [$0], {$1, $2}; }", "l,r,r,r", loc, ip, side=True)


@dsl_user_op
def stg_u8_if(addr, a, ok, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Int32(ok)],
          "{ .reg .pred p; setp.ne.s32 p, $2, 0; @p st.global.b8 [$0], $1; }", "l,r,r", loc, ip, side=True)


@dsl_user_op
def mbar_arrive_if(addr, ok, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Int32(ok)],
          "{ .reg .pred p; setp.ne.s32 p, $1, 0; @p mbarrier.arrive.shared::cta.b64 _, [$0]; }", "r,r", loc, ip,
          side=True)


@dsl_user_op
def bulk_cp(dst, src, bar, nbytes: int, *, loc=None, ip=None):
    """nbytes global -> shared bulk copy completing on bar (L2 evict_first)."""
    _asm1(
        None, [Int32(dst), Int64(src), Int32(bar)],
        "{ .reg .b64 pf;\n"
        "createpolicy.fractional.L2::evict_first.b64 pf, 1.0;\n"
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], {nbytes}, [$2], pf; }}",
        "r,l,r", loc, ip, side=True,
    )


@dsl_user_op
def fence_proxy_async(*, loc=None, ip=None):
    _asm1(None, [], "fence.proxy.async.shared::cta;", "", loc, ip, side=True)


@dsl_user_op
def syncwarp(*, loc=None, ip=None):
    _asm1(None, [], "bar.warp.sync -1;", "", loc, ip, side=True)


@dsl_user_op
def fence_gpu(*, loc=None, ip=None):
    _asm1(None, [], "fence.acq_rel.gpu;", "", loc, ip, side=True)


@dsl_user_op
def st_release(addr, v, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Int32(v)], "st.release.gpu.global.b32 [$0], $1;", "l,r", loc, ip, side=True)


@dsl_user_op
def ld_acquire(addr, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int64(addr)], "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l", loc, ip, side=True))


@dsl_user_op
def shfl_xor(v, off: int, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(v)], f"shfl.sync.bfly.b32 $0, $1, {off}, 31, -1;", "=r,r", loc, ip,
                        side=True))


@dsl_user_op
def shfl_idx(v, src, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(v), Int32(src)], "shfl.sync.idx.b32 $0, $1, $2, 31, -1;", "=r,r,r", loc,
                        ip, side=True))


# ---------------------------------------------------------------- math
@dsl_user_op
def wprime(wv, *, loc=None, ip=None):
    """bf16x2 weight word -> f32x2 of (w + 0)."""
    return Int64(_asm1(
        T.i64(), [Uint32(wv)],
        "{ .reg .b32 l, h; .reg .b64 p, z;\n"
        "shl.b32 l, $1, 16; and.b32 h, $1, 0xFFFF0000; mov.b64 p, {l, h};\n"
        "mov.b64 z, 0; add.rn.f32x2 $0, p, z; }",
        "=l,r", loc, ip))


@dsl_user_op
def neg_p(v, *, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [Int64(v)], "xor.b64 $0, $1, 0x8000000080000000;", "=l,l", loc, ip))


@dsl_user_op
def pair(a, *, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [Float32(a)], "mov.b64 $0, {$1, $1};", "=l,f", loc, ip))


@dsl_user_op
def rope4(ga, gb, c3a, c3b, c2b, c2a, *, loc=None, ip=None):
    """Words ga (cols j, j+1) and gb (64+j, 65+j) of the grad -> f32x2 of the bf16-rounded RoPE-bwd outputs:
    A = add0(rn(rn(gB in2B) + rn(gA in3A))), B = add0(rn(rn(gB in3B) - rn(gA in2A)))."""
    r = _asmN(
        2, T.i64(), [Uint32(ga), Uint32(gb), Int64(c3a), Int64(c3b), Int64(c2b), Int64(c2a)],
        "{ .reg .b32 l, h, z; .reg .b64 a, b, oa, ob, pa, pb, zz;\n"
        "shl.b32 l, $2, 16; and.b32 h, $2, 0xFFFF0000; mov.b64 a, {l, h};\n"
        "shl.b32 l, $3, 16; and.b32 h, $3, 0xFFFF0000; mov.b64 b, {l, h};\n"
        "mul.rn.f32x2 oa, a, $4;\n"
        "mul.rn.f32x2 ob, b, $5;\n"
        "mul.rn.f32x2 pa, b, $6;\n"
        "mul.rn.f32x2 pb, a, $7;\n"
        "mov.b64 zz, 0;\n"
        "add.rn.f32x2 pa, pa, oa; add.rn.f32x2 pa, pa, zz;\n"
        "sub.rn.f32x2 pb, ob, pb; add.rn.f32x2 pb, pb, zz;\n"
        "mov.b32 z, 0;\n"
        "mov.b64 {l, h}, pa; cvt.rn.bf16x2.f32 l, l, z; cvt.rn.bf16x2.f32 h, h, z; mov.b64 $0, {l, h};\n"
        "mov.b64 {l, h}, pb; cvt.rn.bf16x2.f32 l, l, z; cvt.rn.bf16x2.f32 h, h, z; mov.b64 $1, {l, h}; }",
        "=l,=l,r,r,l,l,l,l", loc, ip,
    )
    return Int64(r[0]), Int64(r[1])


@dsl_user_op
def norm4(xa, xb, rr, wa, wb, da, db, acca, accb, sa, sb, *, loc=None, ip=None):
    """xh = rn(x rstd), wdy = rn(w' d), acc = fma(xh, d, acc), S = fma(xh, wdy, S) (lo then hi), chunks A / B."""
    tys = [T.i64()] * 6 + [T.f32()] * 2
    r = _asmN(
        8, tys,
        [Uint32(xa), Uint32(xb), Int64(rr), Int64(wa), Int64(wb), Int64(da), Int64(db), Int64(acca), Int64(accb),
         Float32(sa), Float32(sb)],
        "{ .reg .b32 l, h; .reg .f32 p, q, u, v, s, t; .reg .b64 a, b;\n"
        "shl.b32 l, $8, 16; and.b32 h, $8, 0xFFFF0000; mov.b64 a, {l, h};\n"
        "shl.b32 l, $9, 16; and.b32 h, $9, 0xFFFF0000; mov.b64 b, {l, h};\n"
        "mul.rn.f32x2 $0, a, $10;\n"
        "mul.rn.f32x2 $1, b, $10;\n"
        "mul.rn.f32x2 $2, $11, $13;\n"
        "mul.rn.f32x2 $3, $12, $14;\n"
        "fma.rn.f32x2 $4, $0, $13, $15;\n"
        "fma.rn.f32x2 $5, $1, $14, $16;\n"
        "mov.b64 {p, q}, $0; mov.b64 {u, v}, $2;\n"
        "fma.rn.f32 s, p, u, $17; fma.rn.f32 $6, q, v, s;\n"
        "mov.b64 {p, q}, $1; mov.b64 {u, v}, $3;\n"
        "fma.rn.f32 t, p, u, $18; fma.rn.f32 $7, q, v, t; }",
        "=l,=l,=l,=l,=l,=l,=f,=f,r,r,l,l,l,l,l,l,l,f,f", loc, ip,
    )
    return [Int64(v) for v in r[:6]] + [Float32(r[6]), Float32(r[7])]


@dsl_user_op
def rope4b(ga, gb, c3a, c3b, c2b, c2a, *, loc=None, ip=None):
    """rope4 with scalar outputs (dA_j, dA_j+1, dB_j, dB_j+1), each the f32 value of the bf16-rounded result."""
    r = _asmN(
        4, T.i32(), [Uint32(ga), Uint32(gb), Int64(c3a), Int64(c3b), Int64(c2b), Int64(c2a)],
        "{ .reg .b32 l, h, z; .reg .b64 a, b, oa, ob, pa, pb, zz;\n"
        "shl.b32 l, $4, 16; and.b32 h, $4, 0xFFFF0000; mov.b64 a, {l, h};\n"
        "shl.b32 l, $5, 16; and.b32 h, $5, 0xFFFF0000; mov.b64 b, {l, h};\n"
        "mul.rn.f32x2 oa, a, $6;\n"
        "mul.rn.f32x2 ob, b, $7;\n"
        "mul.rn.f32x2 pa, b, $8;\n"
        "mul.rn.f32x2 pb, a, $9;\n"
        "mov.b64 zz, 0;\n"
        "add.rn.f32x2 pa, pa, oa; add.rn.f32x2 pa, pa, zz;\n"
        "sub.rn.f32x2 pb, ob, pb; add.rn.f32x2 pb, pb, zz;\n"
        "mov.b32 z, 0;\n"
        "mov.b64 {l, h}, pa; cvt.rn.bf16x2.f32 $0, l, z; cvt.rn.bf16x2.f32 $1, h, z;\n"
        "mov.b64 {l, h}, pb; cvt.rn.bf16x2.f32 $2, l, z; cvt.rn.bf16x2.f32 $3, h, z; }",
        "=r,=r,=r,=r,r,r,l,l,l,l", loc, ip,
    )
    return [Uint32(v) for v in r]


@dsl_user_op
def pair_u(a, b, *, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [Uint32(a), Uint32(b)], "mov.b64 $0, {$1, $2};", "=l,r,r", loc, ip))


@dsl_user_op
def norm4b(xa, xb, rr, we, wo, de, do, acce, acco, ss, *, loc=None, ip=None):
    """(A_j, B_j)-paired norm step for j = 2k (e) and 2k + 1 (o): x words xa (A cols 2k, 2k+1), xb (B cols).
    xh = rn(x rstd), wdy = rn(w' d), acc = fma(xh, d, acc), S = fma(xh, wdy, S) (e then o; S = (S_A, S_B))."""
    r = _asmN(
        7, T.i64(),
        [Uint32(xa), Uint32(xb), Int64(rr), Int64(we), Int64(wo), Int64(de), Int64(do), Int64(acce), Int64(acco),
         Int64(ss)],
        "{ .reg .b32 l, h, m, n; .reg .b64 xe, xo, t;\n"
        "shl.b32 l, $7, 16; and.b32 h, $7, 0xFFFF0000;\n"
        "shl.b32 m, $8, 16; and.b32 n, $8, 0xFFFF0000;\n"
        "mov.b64 xe, {l, m}; mov.b64 xo, {h, n};\n"
        "mul.rn.f32x2 $0, xe, $9;\n"
        "mul.rn.f32x2 $1, xo, $9;\n"
        "mul.rn.f32x2 $2, $10, $12;\n"
        "mul.rn.f32x2 $3, $11, $13;\n"
        "fma.rn.f32x2 $4, $0, $12, $14;\n"
        "fma.rn.f32x2 $5, $1, $13, $15;\n"
        "fma.rn.f32x2 t, $0, $2, $16;\n"
        "fma.rn.f32x2 $6, $1, $3, t; }",
        "=l,=l,=l,=l,=l,=l,=l,r,r,l,l,l,l,l,l,l,l", loc, ip,
    )
    return [Int64(v) for v in r]


@dsl_user_op
def dx2b(xe, xo, we, wo, nm, rr, *, loc=None, ip=None):
    """Words (A cols 2k, 2k+1) and (B cols 2k, 2k+1) of dx = bf16(rn(rstd fma(-xh, mean, wdy)))."""
    r = _asmN(
        2, T.i32(), [Int64(xe), Int64(xo), Int64(we), Int64(wo), Int64(nm), Int64(rr)],
        "{ .reg .b64 te, to; .reg .f32 p, q, u, v;\n"
        "fma.rn.f32x2 te, $2, $6, $4; mul.rn.f32x2 te, $7, te;\n"
        "fma.rn.f32x2 to, $3, $6, $5; mul.rn.f32x2 to, $7, to;\n"
        "mov.b64 {p, q}, te; mov.b64 {u, v}, to;\n"
        "cvt.rn.bf16x2.f32 $0, u, p; cvt.rn.bf16x2.f32 $1, v, q; }",
        "=r,=r,l,l,l,l,l,l", loc, ip,
    )
    return Uint32(r[0]), Uint32(r[1])


@dsl_user_op
def sum_halves(v, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int64(v)], "{ .reg .f32 a, b; mov.b64 {a, b}, $1; add.rn.f32 $0, a, b; }",
                         "=f,l", loc, ip))


@dsl_user_op
def wpair(wa, wb, hi: int, *, loc=None, ip=None):
    """(w'[A col], w'[B col]) of the even (hi=0) or odd (hi=1) element of the bf16x2 words wa, wb; w' = w + 0."""
    ext = "shl.b32 l, $1, 16; shl.b32 m, $2, 16;" if hi == 0 else "and.b32 l, $1, 0xFFFF0000; and.b32 m, $2, 0xFFFF0000;"
    return Int64(_asm1(
        T.i64(), [Uint32(wa), Uint32(wb)],
        "{ .reg .b32 l, m; .reg .b64 p, z;\n" + ext + "\n"
        "mov.b64 p, {l, m}; mov.b64 z, 0; add.rn.f32x2 $0, p, z; }",
        "=l,r,r", loc, ip))


@dsl_user_op
def bfly_add(v, off: int, *, loc=None, ip=None):
    return Float32(_asm1(
        T.f32(), [Float32(v)],
        "{ .reg .f32 o;\n"
        f"shfl.sync.bfly.b32 o, $1, {off}, 31, -1;\n"
        "add.rn.f32 $0, $1, o; }",
        "=f,f", loc, ip, side=True))


@dsl_user_op
def add_f(a, b, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def neg_mean(s, *, loc=None, ip=None):
    """f32x2 (-m, -m), m = rn(S * 2^-7)."""
    return Int64(_asm1(
        T.i64(), [Float32(s)],
        "{ .reg .f32 m; mul.rn.f32 m, $1, 0f3C000000; neg.f32 m, m; mov.b64 $0, {m, m}; }",
        "=l,f", loc, ip))


@dsl_user_op
def dx2(xh, wd, nm, rr, *, loc=None, ip=None):
    """bf16x2 word of dx = rn(rstd fma(-xh, mean, wdy)) for one f32x2 pair."""
    return Uint32(_asm1(
        T.i32(), [Int64(xh), Int64(wd), Int64(nm), Int64(rr)],
        "{ .reg .b64 t; .reg .f32 p, q;\n"
        "fma.rn.f32x2 t, $1, $3, $2;\n"
        "mul.rn.f32x2 t, $4, t;\n"
        "mov.b64 {p, q}, t; cvt.rn.bf16x2.f32 $0, q, p; }",
        "=r,l,l,l,l", loc, ip))


@dsl_user_op
def acc_add(a, b, *, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [Int64(a), Int64(b)], "add.rn.f32x2 $0, $1, $2;", "=l,l,l", loc, ip))


@dsl_user_op
def hmax4(a, b, c, d, *, loc=None, ip=None):
    return Uint32(_asm1(
        T.i32(), [Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
        "{ .reg .b32 x, y;\n"
        "max.NaN.xorsign.abs.bf16x2 x, $1, $2;\n"
        "max.NaN.xorsign.abs.bf16x2 y, $3, $4;\n"
        "max.NaN.xorsign.abs.bf16x2 $0, x, y; }",
        "=r,r,r,r,r", loc, ip))


@dsl_user_op
def hmax2(a, b, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "max.NaN.xorsign.abs.bf16x2 $0, $1, $2;", "=r,r,r", loc,
                        ip))


@dsl_user_op
def merge_halves(a, b, *, loc=None, ip=None):
    """(max of a's halves) in the low half, (max of b's halves) in the high half (sign bits garbage)."""
    return Uint32(_asm1(
        T.i32(), [Uint32(a), Uint32(b)],
        "{ .reg .b32 x, y;\n"
        "prmt.b32 x, $1, $2, 0x5410;\n"
        "prmt.b32 y, $1, $2, 0x7632;\n"
        "max.NaN.xorsign.abs.bf16x2 $0, x, y; }",
        "=r,r,r", loc, ip))


@dsl_user_op
def scale2(m, *, loc=None, ip=None):
    """m: two block |max| (bf16 bits in each half, sign garbage). Returns (e8m0 bytes in bits 0-7 / 16-23,
    bf16x2 recip 2^(127 - byte), NaN for byte 255)."""
    r = _asmN(
        2, T.i32(), [Uint32(m)],
        "{ .reg .b32 a, e, n, f, b, r, c8;\n"
        "mov.b32 c8, 0x00080008;\n"
        "and.b32 a, $2, 0x7FFF7FFF;\n"
        "add.u32 e, a, 0x001F001F;\n"
        "shr.u32 e, e, 7;\n"
        "and.b32 e, e, 0x01FF01FF;\n"
        "max.u16x2 e, e, c8;\n"
        "sub.u32 b, e, 0x00080008;\n"
        "add.u32 n, a, 0x00800080;\n"
        "prmt.b32 f, n, 0, 0x4B49;\n"
        "or.b32 b, b, f;\n"
        "sub.u32 r, 0x01FE01FE, b;\n"
        "shl.b32 r, r, 7;\n"
        "and.b32 r, r, 0x7F807F80;\n"
        "or.b32 $1, r, f;\n"
        "mov.b32 $0, b; }",
        "=r,=r,r", loc, ip,
    )
    return Uint32(r[0]), Uint32(r[1])


@dsl_user_op
def bcast_lo(r, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(r)], "prmt.b32 $0, $1, $1, 0x1010;", "=r,r", loc, ip))


@dsl_user_op
def bcast_hi(r, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(r)], "prmt.b32 $0, $1, $1, 0x3232;", "=r,r", loc, ip))


@dsl_user_op
def quant4(y0, y1, rr, *, loc=None, ip=None):
    """4 e4m3 bytes (byte k = element k) of the bf16x2 words y0, y1 times the bf16x2 recip rr."""
    return Uint32(_asm1(
        T.i32(), [Uint32(y0), Uint32(y1), Uint32(rr)],
        "{ .reg .b32 p, q; .reg .b16 h0, h1;\n"
        "mul.rn.bf16x2 p, $1, $3;\n"
        "mul.rn.bf16x2 q, $2, $3;\n"
        "cvt.rn.satfinite.e4m3x2.bf16x2 h0, p;\n"
        "cvt.rn.satfinite.e4m3x2.bf16x2 h1, q;\n"
        "mov.b32 $0, {h0, h1}; }",
        "=r,r,r,r", loc, ip))


@dsl_user_op
def sel(msk, a, b, *, loc=None, ip=None):
    """a where msk is all ones, b where msk is 0."""
    return Uint32(_asm1(
        T.i32(), [Uint32(msk), Uint32(a), Uint32(b)],
        "{ .reg .b32 t; xor.b32 t, $2, $3; and.b32 t, t, $1; xor.b32 $0, $3, t; }",
        "=r,r,r,r", loc, ip))


def _bits(v, lo, n):
    return (v >> lo) & ((1 << n) - 1)


class AttnGlueBwd:
    @cute.jit
    def __call__(self, g_k, g_q, c2, c3, r_k, x_k, w_k, r_q, x_q, w_q, g_v,
                 qrow, srow, scol, qcol, units, ucnt, kcls, pe, po,
                 s2: cutlass.Constexpr, s3: cutlass.Constexpr, stream):
        # q g / x rows of a warp (4 tokens x one head group = 4 x 1 KB, 8 KB apart): one TMA box each over the u32
        # view [T][2048]
        box = cute.make_layout((256, 4), stride=(1, 256))
        op = cpasync.CopyBulkTensorTileG2SOp()
        tq = [cpasync.make_tiled_tma_atom(op, cute.make_tensor(cute.recast_ptr(t.iterator, dtype=Uint32),
                                                               cute.make_layout((2048, NTOK), stride=(1, 2048))),
                                          box, (256, 4)) for t in (g_q, x_q)]
        self.kernel(g_k, g_q, c2, c3, r_k, x_k, w_k, r_q, x_q, w_q, g_v, qrow, srow, scol, qcol, units, ucnt, kcls,
                    pe, po, tq[0][0], tq[0][1], tq[1][0], tq[1][1], s2, s3).launch(
            grid=[NSM, 1, 1], block=[NTHR, 1, 1], smem=SMEM_B, stream=stream, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, g_k, g_q, c2, c3, r_k, x_k, w_k, r_q, x_q, w_q, g_v,
               qrow, srow, scol, qcol, units, ucnt, kcls, pe, po, a_gq, m_gq, a_xq, m_xq,
               s2: cutlass.Constexpr, s3: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        w = tid // 32
        lane = tid % 32
        l = lane % 8
        m = l % 4
        i = w * 4 + lane // 8
        swz_i = (i & 1) | (((i >> 3) & 3) << 1)

        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(SMEM_B), byte_alignment=1024)
        sb = Int32(sAll.iterator.toint())
        ring = sb + w * (2 * STAGE_B)
        wbar = sb + OFF_BAR + w * 16
        tfull = sb + OFF_BAR + 128
        tempty = sb + OFF_BAR + 144

        # TMA partitions: smem side indexed by 2 w + stage (the warp rings), gmem side by (hg, 4-token block)
        lay = cute.make_layout((256, 4, 16), stride=(1, 256, STAGE_B // 4))
        tparts = []
        for atom, mt, off in ((a_gq, m_gq, 0), (a_xq, m_xq, 4096)):
            sd = cute.make_tensor(cute.recast_ptr(sAll.iterator + off, dtype=Uint32), lay)
            gt = cute.local_tile(mt, (256, 4), (None, None))
            tparts.append(cpasync.tma_partition(atom, 0, cute.make_layout(1), cute.group_modes(sd, 0, 2),
                                                cute.group_modes(gt, 0, 2)))
        TM = {"a_gq": a_gq, "a_xq": a_xq, "sg": tparts[0][0], "gg": tparts[0][1], "sx": tparts[1][0], "gx": tparts[1][1]}
        P = {
            "gk": g_k.iterator.toint(), "gq": g_q.iterator.toint(), "c2": c2.iterator.toint(),
            "c3": c3.iterator.toint(), "rk": r_k.iterator.toint(), "xk": x_k.iterator.toint(),
            "rq": r_q.iterator.toint(), "xq": x_q.iterator.toint(), "gv": g_v.iterator.toint(),
            "qrow": qrow.iterator.toint(), "srow": srow.iterator.toint(), "scol": scol.iterator.toint(),
            "qcol": qcol.iterator.toint(),
        }
        p_units = units.iterator.toint() + Int64(cta * MAXU * 4)
        p_pe = pe.iterator.toint()
        p_po = po.iterator.toint()
        p_wq = w_q.iterator.toint() + Int64(l * 16)
        p_wk = w_k.iterator.toint() + Int64(l * 16)

        nu = ldg_u32(ucnt.iterator.toint() + Int64(cta * 4))
        kc = ldg_u32(kcls.iterator.toint() + Int64(cta * 4))
        nk = ldg_u32(kcls.iterator.toint() + Int64((NSM + cta) * 4))
        nkq = ldg_u32(kcls.iterator.toint() + Int64((2 * NSM + cta) * 4))
        hgc = cta // 19
        cbq = (hgc >> 1) * 1280 + (hgc & 1) * 512  # first output column of this CTA's q heads

        # per-thread constants
        C = {
            "i": i, "l": l, "m": m, "lane": lane, "w": w,
            "msk1": Uint32(0) - Uint32(m & 1), "msk2": Uint32(0) - Uint32((m >> 1) & 1),
            "so": (i % 4) * 1024 + l * 16,  # this thread's 16-byte chunk A of head 0 in a stage
            "to": i * 1024 + (l ^ swz_i) * 16,  # ... in a dx tile (swizzled)
            "cbq": cbq,
        }
        tr = 8 * ((lane % 8) // 2) + 2 * (lane // 8) + (lane % 2)  # colwise: token row this lane addresses
        C["swc"] = (tr & 1) | (((tr >> 3) & 3) << 1)
        C["crow"] = tr * 1024 + (w // 2) * 256 + (w % 2) * 128
        C["q4"] = lane % 4
        C["cc"] = lane // 4

        if tid < 8:
            mbar_init(sb + OFF_BAR + tid * 16, 1)
            mbar_init(sb + OFF_BAR + tid * 16 + 8, 1)
        if tid == 8:
            mbar_init(tfull, NMATH)  # every math thread arrives after writing its tile chunks
            mbar_init(tfull + 8, NMATH)
            mbar_init(tempty, 128)  # every quant thread arrives after its last tile read
            mbar_init(tempty + 8, 128)
        if tid < MAXU:
            sts_b32(sb + OFF_UT + tid * 4, ldg_u32(p_units + Int64(tid * 4)))
        mbar_init_fence()
        cute.arch.barrier()
        B = {"sb": sb, "ring": ring, "wbar": wbar, "tfull": tfull, "tempty": tempty}

        if w >= 8:
            cute.arch.setmaxregister_decrease(QREG)
            self.quant_loop(nu, w - 8, lane, B, P)
        else:
            cute.arch.setmaxregister_increase(MREG)
            # register state
            acc = cute.make_rmem_tensor((16,), Int64)  # heads 0, 1: [head][pair j = (A_j, B_j)] dw accumulators
            accs = sb + OFF_ACC + tid * 16  # heads 2, 3: unit 4 (head - 2) + jj (pairs 2 jj, 2 jj + 1) at accs + 4096 unit
            wf = cute.make_rmem_tensor((8,), Int64)  # w' pairs (A_j, B_j)
            csf = cute.make_rmem_tensor((16,), Int64)  # in3 A, in3 B, in2 B, in2 A pairs (cols 2k, 2k + 1)
            rsf = cute.make_rmem_tensor((4,), Float32)  # rstd per head
            amx = cute.make_rmem_tensor((8,), Uint32)  # per-head amax words [A h0..h3, B h0..h3]
            R = {"acc": acc, "accs": accs, "wf": wf, "csf": csf, "rsf": rsf, "amx": amx}
            self.zero_acc(acc, accs)
            self.load_w(p_wk, wf)

            # ---- prologue: unit 0 -> stage 0, its cos/sin/rstd -> registers
            uw0 = lds_b32(sb + OFF_UT)
            self.issue_unit(uw0, Int32(0), ring, wbar, lane, w, P, TM)
            self.load_regs(uw0, i, l, csf, rsf, P, s2, s3)

            u = Int32(0)
            while u < nk:
                self.step_math(KK, u, nu, C, B, R, P, TM, s2, s3)
                u = u + 1
            # k accumulators: rows r = 4 (i % 2) + head of quack CTA b = 16 G' + i // 2 -> partial data; q from 0
            self.write_partial(0, 16 * kc + i // 2, i % 2, l, acc, accs, p_pe, p_po)
            self.zero_acc(acc, accs)
            self.load_w(p_wq, wf)
            while u < nkq:
                self.step_math(KQ, u, nu, C, B, R, P, TM, s2, s3)
                u = u + 1
            while u < nu:
                self.step_math(KV, u, nu, C, B, R, P, TM, s2, s3)
                u = u + 1
            # ---- q accumulators: rows r = 4 (hg & 1) + head of quack CTA b = 128 G + 4 i + hg // 2 -> partial data
            self.write_partial(1, 128 * (cta % 19) + 4 * i + (hgc >> 1), hgc & 1, l, acc, accs, p_pe, p_po)

    @cute.jit
    def step_math(self, kind: cutlass.Constexpr, u, nu, C, B, R, P, TM, s2: cutlass.Constexpr, s3: cutlass.Constexpr):
        """Math warp, unit u: prefetch unit u + 1, then RoPE bwd + norm bwd (+ dw) of unit u (or the v copy) into the
        bf16 dx tile u % 2."""
        sb = B["sb"]
        i = C["i"]
        l = C["l"]
        lane = C["lane"]
        uwn = lds_b32(sb + OFF_UT + u * 4 + 4)
        if u + 1 < nu:
            # stage (u + 1) % 2 was last read (into registers) by this warp during step u - 1
            syncwarp()
            self.issue_unit(uwn, (u + 1) % 2, B["ring"] + ((u + 1) % 2) * STAGE_B, B["wbar"] + ((u + 1) % 2) * 8, lane,
                            C["w"], P, TM)
        # the tile u % 2 was last read by the quant warps for unit u - 2
        if u >= 2:
            mbar_wait(B["tempty"] + (u % 2) * 8, ((u // 2) + 1) % 2)
        tile = sb + OFF_TILE + (u % 2) * TILE_B + C["to"]
        mbar_wait(B["wbar"] + (u % 2) * 8, (u // 2) % 2)
        ld_off = B["ring"] + (u % 2) * STAGE_B + C["so"]
        amx = R["amx"]
        if cutlass.const_expr(kind == KV):
            for s in cutlass.range_constexpr(4):
                ga = lds_v4(ld_off + s * 256)
                gb = lds_v4(ld_off + s * 256 + 128)
                sts_v4(tile + s * 256, ga[0], ga[1], ga[2], ga[3])
                sts_v4(tile + s * 256 + 128, gb[0], gb[1], gb[2], gb[3])
                amx[s] = hmax4(ga[0], ga[1], ga[2], ga[3])
                amx[4 + s] = hmax4(gb[0], gb[1], gb[2], gb[3])
        else:
            self.rows(ld_off, tile, R["acc"], R["accs"], R["wf"], R["csf"], R["rsf"], amx)
        mbar_arrive(B["tfull"] + (u % 2) * 8)
        # next unit's cos/sin/rstd (the registers are free now)
        if u + 1 < nu:
            self.load_regs(uwn, i, l, R["csf"], R["rsf"], P, s2, s3)
        # rowwise MXFP8 of unit u: reduce-scatter of the per-head amax words over the 4 lanes l % 4 (lane m ends
        # with head m's blocks), scales, then quantize this thread's dx chunks (re-read from the tile)
        if cutlass.const_expr(kind == KQ):
            cb0 = C["cbq"]
            hstr = 128
        elif cutlass.const_expr(kind == KK):
            cb0 = Int32(1024)
            hstr = 1280
        else:
            cb0 = Int32(1152)
            hstr = 1280
        m = C["m"]
        tb = lds_b32(sb + OFF_UT + u * 4) & 1023
        tok = tb * 32 + i
        msk1 = C["msk1"]
        msk2 = C["msk2"]
        nA = [None] * 2
        nB = [None] * 2
        for p in cutlass.range_constexpr(2):
            nA[p] = hmax2(sel(msk1, amx[2 * p + 1], amx[2 * p]), shfl_xor(sel(msk1, amx[2 * p], amx[2 * p + 1]), 1))
            nB[p] = hmax2(sel(msk1, amx[5 + 2 * p], amx[4 + 2 * p]),
                          shfl_xor(sel(msk1, amx[4 + 2 * p], amx[5 + 2 * p]), 1))
        wA0 = hmax2(sel(msk2, nA[1], nA[0]), shfl_xor(sel(msk2, nA[0], nA[1]), 2))
        wB0 = hmax2(sel(msk2, nB[1], nB[0]), shfl_xor(sel(msk2, nB[0], nB[1]), 2))
        byt, rcp = scale2(merge_halves(wA0, wB0))
        oth = shfl_xor(byt, 4)
        rows = [shfl_idx(rcp, (lane & 28) + hh) for hh in range(4)]
        if l < 4:
            sadr = P["srow"] + Int64(((cb0 + m * hstr) >> 7) * 512 + (tb >> 2) * 20480 + i * 16 + (tb & 3) * 4)
            stg_b32(sadr, byt | (oth << Uint32(8)))
        qbase = P["qrow"] + Int64(tok) * NCOL + Int64(cb0 + l * 8)
        for s in cutlass.range_constexpr(4):
            ya = lds_v4(tile + s * 256)
            yb = lds_v4(tile + s * 256 + 128)
            ra = bcast_lo(rows[s])
            rb = bcast_hi(rows[s])
            qadr = qbase + s * hstr
            stg_v2(qadr, quant4(ya[0], ya[1], ra), quant4(ya[2], ya[3], ra))
            stg_v2(qadr + 64, quant4(yb[0], yb[1], rb), quant4(yb[2], yb[3], rb))

    @cute.jit
    def quant_loop(self, nu, q, lane, B, P):
        """Quant warp q (0..3), every unit: colwise MXFP8 of the two 64-column slabs 2 q, 2 q + 1 of the dx tile
        (ldmatrix.trans: 8 columns x 32 tokens per instruction, lane = column, 8 consecutive tokens), then release
        the tile."""
        sb = B["sb"]
        m = lane % 4
        tr = 8 * ((lane % 8) // 2) + 2 * (lane // 8) + (lane % 2)  # colwise: token row this lane addresses
        Cq = {
            "lane": lane, "msk1": Uint32(0) - Uint32(m & 1), "msk2": Uint32(0) - Uint32((m >> 1) & 1),
            "swc": (tr & 1) | (((tr >> 3) & 3) << 1), "q4": lane % 4, "cc": lane // 4,
        }
        crow0 = tr * 1024 + q * 256
        u = Int32(0)
        while u < nu:
            uw = lds_b32(sb + OFF_UT + u * 4)
            kind = (uw >> 13) & 3
            tb = uw & 1023
            hg = (uw >> 10) & 7
            cb0 = Int32(1024)
            hstr = Int32(1280)
            if kind == KQ:
                cb0 = (hg >> 1) * 1280 + (hg & 1) * 512
                hstr = Int32(128)
            elif kind == KV:
                cb0 = Int32(1152)
            mbar_wait(B["tfull"] + (u % 2) * 8, (u // 2) % 2)
            tile = sb + OFF_TILE + (u % 2) * TILE_B
            # colwise: slabs 2 q + h (head q, columns 64 h .. 64 h + 63)
            for h in cutlass.range_constexpr(2):
                rowa = tile + crow0 + h * 128
                swc = Cq["swc"]
                d = [ldsm_x4t(rowa + (j ^ swc) * 16) for j in range(8)]
                mw = [hmax4(d[j][0], d[j][1], d[j][2], d[j][3]) for j in range(8)]
                msk1 = Cq["msk1"]
                msk2 = Cq["msk2"]
                n = [None] * 4
                for p_ in cutlass.range_constexpr(4):
                    n[p_] = hmax2(sel(msk1, mw[2 * p_ + 1], mw[2 * p_]), shfl_xor(sel(msk1, mw[2 * p_], mw[2 * p_ + 1]), 1))
                f = [None] * 2
                for pp in cutlass.range_constexpr(2):
                    f[pp] = hmax2(sel(msk2, n[2 * pp + 1], n[2 * pp]), shfl_xor(sel(msk2, n[2 * pp], n[2 * pp + 1]), 2))
                cbyt, crcp = scale2(merge_halves(f[0], f[1]))
                Rc = [shfl_idx(crcp, (lane & 28) + jj) for jj in range(4)]
                Cw = {"w": 2 * q + h, "q4": Cq["q4"], "cc": Cq["cc"]}
                self.colwise_store(d, Rc, cbyt, Int32(1), cb0, hstr, tb, Cw, P)
            mbar_arrive(B["tempty"] + (u % 2) * 8)
            u = u + 1

    # ------------------------------------------------------------ helpers (inlined)
    @cute.jit
    def zero_acc(self, acc, accs):
        for k in cutlass.range_constexpr(16):
            acc[k] = Int64(0)
        for j in cutlass.range_constexpr(8):
            sts_p2(accs + j * 4096, Int64(0), Int64(0))

    @cute.jit
    def load_w(self, pw, wf):
        wa = ldg_v4(pw)
        wb = ldg_v4(pw + 128)
        for k in cutlass.range_constexpr(4):
            wf[2 * k] = wpair(wa[k], wb[k], 0)
            wf[2 * k + 1] = wpair(wa[k], wb[k], 1)

    @cute.jit
    def colwise_store(self, d, Rc, cbyt, vp, pcb, phs, ptb, C, P):
        """Quantize and store the colwise e4m3 bytes (32 per column: 4 lanes x 8 tokens) and the two scale bytes
        (chunks q4, q4 + 4) of this lane."""
        w = C["w"]
        q4 = C["q4"]
        colb = pcb + (w // 2) * phs + 64 * (w % 2) + C["cc"]
        cbase = P["qcol"] + Int64(colb) * NTOK + Int64(ptb * 32 + q4 * 8)
        for j in cutlass.range_constexpr(8):
            rr = bcast_lo(Rc[j % 4]) if j < 4 else bcast_hi(Rc[j % 4])
            stg_v2_if(cbase + 8 * j * NTOK, quant4(d[j][0], d[j][1], rr), quant4(d[j][2], d[j][3], rr), vp)
        for e in cutlass.range_constexpr(2):
            col = colb + 8 * (q4 + 4 * e)
            sadr = (ptb >> 2) * 512 + (col >> 7) * 131072 + (col & 31) * 16 + ((col & 127) >> 5) * 4 + (ptb & 3)
            stg_u8_if(P["scol"] + Int64(sadr), (cbyt >> Uint32(16 * e)) & Uint32(255), vp)

    @cute.jit
    def rows(self, ld_off, tile, acc, accs, wf, csf, rsf, amx):
        """RoPE bwd + RMSNorm bwd of the thread's 4 head rows (16 columns each) from the smem stage; norm math on
        (A_j, B_j) column pairs (the two quack lanes' chains in one f32x2); dx goes to the tile (head s chunk A at
        tile + 256 s, chunk B at + 128)."""
        c3a = [csf[k] for k in range(4)]
        c3b = [csf[4 + k] for k in range(4)]
        c2b = [csf[8 + k] for k in range(4)]
        c2a = [csf[12 + k] for k in range(4)]
        wfp = [wf[j] for j in range(8)]
        for pp in cutlass.range_constexpr(2):
            st = {}
            for s in (2 * pp, 2 * pp + 1):
                ga = lds_v4(ld_off + s * 256)
                gb = lds_v4(ld_off + s * 256 + 128)
                xa = lds_v4(ld_off + 4096 + s * 256)
                xb = lds_v4(ld_off + 4096 + s * 256 + 128)
                rr = pair(rsf[s])
                ss = Int64(0)
                xh = [None] * 8
                wd = [None] * 8
                if cutlass.const_expr(s < 2):
                    av = [acc[s * 8 + j] for j in range(8)]
                else:
                    av = []
                    for jj in cutlass.range_constexpr(4):
                        av += lds_p2(accs + ((s - 2) * 4 + jj) * 4096)
                for k in cutlass.range_constexpr(4):
                    d0, d1, d2, d3 = rope4b(ga[k], gb[k], c3a[k], c3b[k], c2b[k], c2a[k])
                    de = pair_u(d0, d2)
                    do = pair_u(d1, d3)
                    xh[2 * k], xh[2 * k + 1], wd[2 * k], wd[2 * k + 1], av[2 * k], av[2 * k + 1], ss = norm4b(
                        xa[k], xb[k], rr, wfp[2 * k], wfp[2 * k + 1], de, do, av[2 * k], av[2 * k + 1], ss)
                if cutlass.const_expr(s < 2):
                    for j in cutlass.range_constexpr(8):
                        acc[s * 8 + j] = av[j]
                else:
                    for jj in cutlass.range_constexpr(4):
                        sts_p2(accs + ((s - 2) * 4 + jj) * 4096, av[2 * jj], av[2 * jj + 1])
                st[s] = [rr, sum_halves(ss), xh, wd]
            for off in (4, 2, 1):
                for s in st:
                    st[s][1] = bfly_add(st[s][1], off)
            for s in st:
                rr, tot, xh, wd = st[s]
                nm = neg_mean(tot)
                ya = [None] * 4
                yb = [None] * 4
                for k in cutlass.range_constexpr(4):
                    ya[k], yb[k] = dx2b(xh[2 * k], xh[2 * k + 1], wd[2 * k], wd[2 * k + 1], nm, rr)
                sts_v4(tile + s * 256, ya[0], ya[1], ya[2], ya[3])
                sts_v4(tile + s * 256 + 128, yb[0], yb[1], yb[2], yb[3])
                amx[s] = hmax4(ya[0], ya[1], ya[2], ya[3])
                amx[4 + s] = hmax4(yb[0], yb[1], yb[2], yb[3])

    @cute.jit
    def issue_unit(self, uw, st, stage, bar, lane, w, P, TM):
        """Loads unit uw's g / x rows of this warp's 4 tokens into stage st of its ring (g at 0, x at 4096): q by two
        TMA boxes (whole warp: the copy elects its issuing lane), k / v by lane 0 bulk copies."""
        kind = (uw >> 13) & 3
        tb = uw & 1023
        hg = (uw >> 10) & 7
        if kind == KQ:
            if lane == 0:
                mbar_expect(bar, Int32(8192))
            syncwarp()
            bptr = cute.make_ptr(Int64, bar, cute.AddressSpace.smem, assumed_align=8)
            cute.copy(TM["a_gq"], TM["gg"][(None, hg, tb * 8 + w)], TM["sg"][(None, 2 * w + st)], tma_bar_ptr=bptr)
            cute.copy(TM["a_xq"], TM["gx"][(None, hg, tb * 8 + w)], TM["sx"][(None, 2 * w + st)], tma_bar_ptr=bptr)
        elif kind == KK:
            if lane == 0:
                tk = tb * 32 + w * 4
                mbar_expect(bar, Int32(8192))
                bulk_cp(stage, P["gk"] + Int64(tk) * 1024, bar, 4096)
                bulk_cp(stage + 4096, P["xk"] + Int64(tk) * 1024, bar, 4096)
        else:
            if lane == 0:
                tk = tb * 32 + w * 4
                mbar_expect(bar, Int32(4096))
                bulk_cp(stage, P["gv"] + Int64(tk) * 1024, bar, 4096)

    @cute.jit
    def load_regs(self, uw, i, l, csf, rsf, P, s2: cutlass.Constexpr, s3: cutlass.Constexpr):
        kind = (uw >> 13) & 3
        tb = uw & 1023
        hg = (uw >> 10) & 7
        if kind != KV:
            tk = Int64(tb * 32 + i)
            a3 = P["c3"] + tk * (s3 * 4) + Int64(l * 32)
            a2 = P["c2"] + tk * (s2 * 4) + Int64(l * 32)
            for t, (base, off) in enumerate(((a3, 0), (a3, 16), (a3, 256), (a3, 272), (a2, 256), (a2, 272), (a2, 0),
                                             (a2, 16))):
                v = ldg_p2(base + off)
                csf[2 * t] = v[0]
                csf[2 * t + 1] = v[1]
            ra = P["rk"] + tk * 16
            if kind == KQ:
                ra = P["rq"] + tk * 128 + Int64(hg * 16)
            rv = ldg_v4(ra)
            for s in cutlass.range_constexpr(4):
                rsf[s] = _u2f(rv[s])

    @cute.jit
    def write_partial(self, kd: cutlass.Constexpr, b, odd, l, acc, accs, p_pe, p_po):
        """Quack CTA b's rows of this thread's 4 heads at its 16 columns. odd == 0: E = ((a0 + a1) + a2) + a3 into
        PE[kd][c][b // 64][b % 64]; odd == 1: the four rows into PO[kd][c][b // 64][b % 64][0..3]."""
        av = []
        for h in cutlass.range_constexpr(4):
            if cutlass.const_expr(h < 2):
                av.append([acc[h * 8 + k] for k in range(8)])
            else:
                t = []
                for j in cutlass.range_constexpr(4):
                    t += lds_p2(accs + ((h - 2) * 4 + j) * 4096)
                av.append(t)
        u = b % 64
        n = b // 64
        cols = [(l * 8 + j) if half == 0 else (64 + l * 8 + j) for j in range(8) for half in range(2)]
        if odd == 0:
            e = [acc_add(acc_add(acc_add(av[0][k], av[1][k]), av[2][k]), av[3][k]) for k in range(8)]
            for k in cutlass.range_constexpr(8):
                lo, hi = split_p(e[k])
                for half in cutlass.range_constexpr(2):
                    c = cols[2 * k + half]
                    stg_f32(p_pe + Int64((((kd * 128 + c) * 38 + n) * 64 + u) * 4), lo if half == 0 else hi)
        else:
            for k in cutlass.range_constexpr(8):
                parts = [split_p(av[h][k]) for h in range(4)]
                for half in cutlass.range_constexpr(2):
                    c = cols[2 * k + half]
                    stg_f4(p_po + Int64((((kd * 128 + c) * 38 + n) * 64 + u) * 16),
                           parts[0][half], parts[1][half], parts[2][half], parts[3][half])

    @cute.jit
    def store_acc_row(self, adr, s: cutlass.Constexpr, acc, accs):
        """Global row (16 columns of this thread) of head s's accumulators."""
        if cutlass.const_expr(s < 2):
            av = [acc[s * 8 + k] for k in range(8)]
        else:
            av = []
            for j in cutlass.range_constexpr(4):
                av += lds_p2(accs + ((s - 2) * 4 + j) * 4096)
        stg_p2(adr, av[0], av[1])
        stg_p2(adr + 16, av[2], av[3])
        stg_p2(adr + 256, av[4], av[5])
        stg_p2(adr + 272, av[6], av[7])

    @cute.jit
    def sum_rows(self, p_acc, p_out, b, l):
        """out[b][cols] = ((a0 + a1) + ...) + a7 over the 8 rows a_r = acc[b][r] at this thread's 16 columns."""
        base = p_acc + Int64((b * 8 * 128 + l * 8) * 4)
        tot = [None] * 8
        for r in cutlass.range_constexpr(8):
            v = [ldcg_p2(base + r * 512 + off) for off in (0, 16, 256, 272)]
            vals = [v[0][0], v[0][1], v[1][0], v[1][1], v[2][0], v[2][1], v[3][0], v[3][1]]
            for k in cutlass.range_constexpr(8):
                tot[k] = vals[k] if r == 0 else acc_add(tot[k], vals[k])
        ob = p_out + Int64((b * 128 + l * 8) * 4)
        stg_p2(ob, tot[0], tot[1])
        stg_p2(ob + 16, tot[2], tot[3])
        stg_p2(ob + 256, tot[4], tot[5])
        stg_p2(ob + 272, tot[6], tot[7])


@dsl_user_op
def _u2f(a, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Uint32(a)], "mov.b32 $0, $1;", "=f,r", loc, ip))


class ReduceRows:
    """dw[kind][c] = bf16(reduce_rows(P_kind[2432, 128], cols=4, threads=256)) in the exact order of
    torch/_native/cutedsl/reduce.py (chain over rows u + 64 n from 0, butterfly over u ^ 1, ^ 2, ^ 4, then the 8
    groups in order), with P[b][c] = (((E + O0) + O1) + O2) + O3 rebuilt from the kernel's partial data.
    CTA = (kind, column), thread = chain u."""

    @cute.jit
    def __call__(self, pe, po, dwk, dwq, stream):
        self.kernel(pe, po, dwk, dwq).launch(grid=[256, 1, 1], block=[64, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pe, po, dwk, dwq):
        u, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        kd = bidx // 128
        c = bidx % 128
        dst = dwk.iterator.toint()
        if kd == 1:
            dst = dwq.iterator.toint()
        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(_K2_SMEM), byte_alignment=128)
        sb = Int32(sAll.iterator.toint())
        bar = sb + _K2_BAR
        # this (kind, column)'s partial data: E [38][64] f32 and O [38][64][4] f32, both contiguous -> two bulk copies
        if u == 0:
            mbar_init(bar, 1)
            mbar_init_fence()
            mbar_expect(bar, Int32(38 * 64 * 20))
            blk = kd * 128 + c
            bulk_cp(sb, pe.iterator.toint() + Int64(blk * 38 * 64 * 4), bar, 38 * 64 * 4)
            bulk_cp(sb + 38 * 64 * 4, po.iterator.toint() + Int64(blk * 38 * 64 * 16), bar, 38 * 64 * 16)
        cute.arch.barrier()
        mbar_wait(bar, 0)
        acc = Float32(0.0)
        for n in cutlass.range_constexpr(38):
            e = lds_f32(sb + (n * 64 + u) * 4)
            o = lds_v4(sb + 38 * 64 * 4 + (n * 64 + u) * 16)
            pn = add_fv(add_fv(add_fv(add_fv(e, _u2f(o[0])), _u2f(o[1])), _u2f(o[2])), _u2f(o[3]))
            acc = add_fv(acc, pn)
        acc = bfly_add(acc, 1)
        acc = bfly_add(acc, 2)
        acc = bfly_add(acc, 4)
        sp = sb + _K2_BAR + 16
        if u % 8 == 0:
            sts_f32(sp + (u // 8) * 4, acc)
        cute.arch.barrier()
        if u == 0:
            for k in cutlass.range_constexpr(1, 8):
                acc = add_f(acc, lds_f32(sp + k * 4))
            stg_b16(dst + Int64(c * 2), _bf16_bits(acc))


_K2_BAR = 38 * 64 * 20
_K2_SMEM = _K2_BAR + 16 + 32


@dsl_user_op
def _bf16_bits(a, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Float32(a)], "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.u32.u16 $0, h; }",
                        "=r,f", loc, ip))


# ---------------------------------------------------------------- host side
_DT = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32, torch.uint8: cutlass.Uint8,
       torch.int32: cutlass.Int32}


def _fake(dtype, n):
    return cute.runtime.make_fake_tensor(_DT[dtype], (n,), stride=(1,), assumed_align=16)


def _build_units():
    """Per-CTA unit lists: the k units first, then q and v units in block order. Unit word: T | hg << 10 | kind << 13.
    Also returns [k class per CTA, number of k units per CTA]."""
    nq = [len(range(c % 19, NBLK, 19)) for c in range(NSM)]
    load = [float(n) for n in nq]
    lists = [[(t, KQ, c // 19) for t in range(c % 19, NBLK, 19)] for c in range(NSM)]
    kcls = [0] * NSM
    classes = sorted(range(NSM), key=lambda g: -len(range(g, NBLK, NSM)))
    for g in classes:
        c = min(range(NSM), key=lambda c: (load[c], c))
        kcls[c] = g
        load[c] += len(range(g, NBLK, NSM))
        lists[c] += [(t, KK, 0) for t in range(g, NBLK, NSM)]
    for t in range(NBLK):
        c = min(range(NSM), key=lambda c: (load[c], c))
        load[c] += 0.64
        lists[c].append((t, KV, 0))
    table = torch.full((NSM, MAXU), -1, dtype=torch.int32)
    cnt = torch.zeros(NSM, dtype=torch.int32)
    nk = [0] * NSM
    nkq = [0] * NSM
    for c in range(NSM):
        ks = sorted(x for x in lists[c] if x[1] == KK)
        qs = sorted(x for x in lists[c] if x[1] == KQ)
        lst = ks + qs + sorted(x for x in lists[c] if x[1] == KV)
        assert len(lst) <= MAXU
        nk[c] = len(ks)
        nkq[c] = len(ks) + len(qs)
        for j, (t, kind, hg) in enumerate(lst):
            table[c, j] = t | (hg << 10) | (kind << 13)
        cnt[c] = len(lst)
    return table, cnt, torch.tensor(kcls + nk + nkq, dtype=torch.int32)


_CACHE = {}


def _compiled(s2, s3):
    key = (s2, s3)
    if key not in _CACHE:
        bf, f32, u8, i32 = torch.bfloat16, torch.float32, torch.uint8, torch.int32
        sig = [
            (bf, NTOK * 512), (bf, NTOK * 4096), (f32, (NTOK - 1) * s2 + 128), (f32, (NTOK - 1) * s3 + 128),
            (f32, NTOK * 4), (bf, NTOK * 512), (bf, 128), (f32, NTOK * 32), (bf, NTOK * 4096), (bf, 128),
            (bf, NTOK * 512),
            (u8, NTOK * NCOL), (u8, NTOK * 160), (u8, NCOL * 1024), (u8, NCOL * NTOK),
            (i32, NSM * MAXU), (i32, NSM), (i32, 3 * NSM), (f32, 2 * 128 * 64 * 38), (f32, 2 * 128 * 64 * 38 * 4),
        ]
        k1 = cute.compile(AttnGlueBwd(), *[_fake(d, n) for d, n in sig], s2, s3,
                          cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
        _CACHE[key] = k1
    return _CACHE[key]


_K2 = cute.compile(ReduceRows(), _fake(torch.float32, 2 * 128 * 64 * 38), _fake(torch.float32, 2 * 128 * 64 * 38 * 4),
                   _fake(torch.bfloat16, 128), _fake(torch.bfloat16, 128),
                   cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")

_TABLES = {}


def _tables(dev):
    if dev not in _TABLES:
        t, c, k = _build_units()
        _TABLES[dev] = (t.to(dev).view(-1), c.to(dev), k.to(dev))
    return _TABLES[dev]


# the static unit tables are uploaded at import, so a first call inside a CUDA-graph capture does no H2D copy
_tables(torch.device("cuda", torch.cuda.current_device()))


def _flat(t):
    t = t if t.is_contiguous() and t.data_ptr() % 16 == 0 else t.contiguous()
    return t.view(-1)


def _cs_view(t):
    """[T, 1, 128] fp32 view with token stride S -> flat [(T - 1) S + 128] view and S."""
    if t.stride(2) != 1 or t.stride(0) % 4 != 0 or t.data_ptr() % 16 != 0:
        t = t.contiguous()
    s = t.stride(0)
    return t.as_strided(((NTOK - 1) * s + 128,), (1,)), s


assert torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count == NSM


def run(in0, in1, in2, in3, in4, in5, in6, in7, in8, in9, in10):
    dev = in0.device
    c2, s2 = _cs_view(in2)
    c3, s3 = _cs_view(in3)
    k1 = _compiled(s2, s3)
    units, ucnt, kcls = _tables(dev)
    u8, f32 = torch.uint8, torch.float32
    qrow = torch.empty((NTOK, NCOL), dtype=u8, device=dev)
    srow = torch.empty((NTOK, 160), dtype=u8, device=dev)
    scol = torch.empty((NCOL, 1024), dtype=u8, device=dev)
    qcol = torch.empty((NCOL, NTOK), dtype=u8, device=dev)
    pe = torch.empty(2 * 128 * 64 * 38, dtype=f32, device=dev)
    po = torch.empty(2 * 128 * 64 * 38 * 4, dtype=f32, device=dev)
    dwk = torch.empty(128, dtype=torch.bfloat16, device=dev)
    dwq = torch.empty(128, dtype=torch.bfloat16, device=dev)
    k1(_flat(in0), _flat(in1), c2, c3, _flat(in4), _flat(in5), _flat(in6), _flat(in7), _flat(in8), _flat(in9),
       _flat(in10), qrow.view(-1), srow.view(-1), scol.view(-1), qcol.view(-1), units, ucnt, kcls, pe, po)
    _K2(pe, po, dwk, dwq)
    return (dwk, dwq, qrow.view(torch.float8_e4m3fn), srow.view(torch.float8_e8m0fnu),
            scol.view(torch.float8_e8m0fnu), qcol.view(torch.float8_e4m3fn))
