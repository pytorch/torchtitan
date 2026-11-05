"""Fused MLA KV backward + MXFP8 quantize (DSv3 671B, DistMoE MXFP8) -- one persistent CuTe DSL kernel.

Reference: torchtitan::fused_mla_kv_backward (Triton, BLOCK_H=128, num_warps=4) produces
grad_kv [4096,128,256] (pure copy of grad_k nope dims + grad_v) and grad_k_pe; grad_kv is
viewed as [4096, 32768] and quantized by torchao::mxfp8_quantize (rowwise 1x32 + colwise 32x1,
e4m3, rceil), and both scale tensors go through torchao::triton_mx_block_rearrange.
grad_kv itself never touches HBM here.

Tiles: 128 tokens x 1 head (grad_kv columns [256 h, 256 h + 256)) plus the grad_k_pe row of token
128 tt + h; 4096 tiles, ordered so that concurrently running tiles cover 16 token tiles x ~10 heads
(DRAM locality of the transposed colwise output). One CTA (8 warps) per SM loops over tiles
k = blockIdx.x + n * gridDim.x.

Pipeline: every thread cp.async-prefetches exactly the shared-memory bytes it later reads itself
(16 rows x 16 B), so the 2-stage ring needs no cross-thread synchronization: tile n + 2 is
issued into stage n % 2 right after the thread copied tile n into registers. Warps 0..3 prefetch
the token's 128 x 64 rope dims (Triton thread layout) into registers one tile ahead.
Compute: warp w owns tile rows 16w..16w+15, lane l owns columns 8l..8l+7 (l < 16 grad_k nope,
else grad_v). Rowwise 32-col amax: in-thread + shfl_xor 1, 2. Colwise 32-row amax: in-thread
16 rows + partner warp (2i, 2i+1) through shared memory (64-thread named barrier). Rowwise e4m3
rows are stored directly (8 B per lane, 256 B per warp row); the colwise e4m3 tile is transposed
through a XOR-swizzled shared-memory tile and both swizzled scale tiles (exactly the
triton_mx_block_rearrange byte layout) are staged in shared memory, then written with full
128 B lines after one CTA barrier per tile.

Numerics (all explicit PTX, no contraction):
* amax: max of |bf16| values (exact; max is order independent; .NaN propagates NaN).
* scale byte: cvt.rp.ue8m0x2.f32(amax *rn fl(1/448)) (torchao calculate_scale<RCEIL>);
  reciprocal = 2^(127 - byte) as fp32 bits (254 - byte) << 23, byte 255 -> NaN
  (torchao reciprocal_scale; byte 254 would need 2^-127 but is unreachable from bf16 input:
  the largest finite amax gives byte 247).
* data: cvt.rn.satfinite.e4m3x2.f32(x *rn reciprocal) (mul.rn.f32x2).
* grad_k_pe: per pair, sum over heads h = 16 i + 4 w + r in Triton's order:
  in-thread ((i0+i1)+(i2+i3))+((i4+i5)+(i6+i7)); shfl_xor 16 (r^2), 8 (r^1);
  across warps (w0+w2)+(w1+w3); then +0.0, bf16 round trip;
  out_even = fma(e, cos, rn(o*sin)); out_odd = fma(o, cos, -rn(e*sin)); bf16 rn.

run(): every output byte is written by the kernel, so the outputs are allocated with the
deterministic-mode NaN fill of torch.empty disabled (it would add ~264 MiB of writes).
"""

import os

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NT = 4096  # tokens
NH = 128  # heads
DK = 192  # grad_k head dim (128 nope + 64 rope)
DV = 128  # grad_v head dim
NCOL = NH * 256  # 32768 grad_kv columns
TT = 128  # tokens per tile
NTILES = (NT // TT) * NH  # 4096
NCONS = 256  # consumer threads (8 warps)
NTHR = NCONS

OFF_CS = 65536
STAGE_BYTES = 65792  # one stage: 64 KB data rows [128][512 B] + 256 B rope cos/sin row
STAGES = 2
OFF_COLT = STAGES * STAGE_BYTES  # [2 buf][256 cols][128 B] colwise e4m3 tile (16 B chunks XOR-swizzled by (col // 8) % 8)
OFF_SCL = OFF_COLT + 2 * 32768  # [2 buf][rowwise 1 KB | colwise 1 KB] swizzled scale tiles
OFF_PART = OFF_SCL + 2 * 2048  # [2 buf][256 thr][16 B] colwise partial amax
OFF_RSUM = OFF_PART + 2 * NCONS * 16  # [2 buf][2 par][4 w][32] f32 rope partial sums
OFF_POS = OFF_RSUM + 2 * 2 * 4 * 32 * 4  # [32] int64 positions of this CTA's tiles
SMEM_BYTES = OFF_POS + 32 * 8

_INV448 = "0f3B124925"  # fl32(1/448)


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


# ---------------------------------------------------------------- memory / sync primitives
@dsl_user_op
def ld_i64(addr, *, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [Int64(addr)], "ld.global.nc.b64 $0, [$1];", "=l,l", loc, ip, side=True))


@dsl_user_op
def st_b32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a)], "st.global.b32 [$0], $1;", "l,r", loc, ip, side=True)


@dsl_user_op
def st_u8(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a)], "st.global.u8 [$0], $1;", "l,r", loc, ip, side=True)


@dsl_user_op
def st_v2(addr, a, b, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b)], "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", loc, ip, side=True)


@dsl_user_op
def st_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def sts_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
          "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def sts_f32x4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Float32(a), Float32(b), Float32(c), Float32(d)],
          "st.shared.v4.f32 [$0], {$1, $2, $3, $4};", "r,f,f,f,f", loc, ip, side=True)


@dsl_user_op
def sts_u8(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Uint32(a)], "st.shared.u8 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int32(addr)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def lds_b32(addr, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int32(addr)], "ld.shared.b32 $0, [$1];", "=r,r", loc, ip, side=True))


@dsl_user_op
def lds_f32(addr, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Int32(addr)], "ld.shared.f32 $0, [$1];", "=f,r", loc, ip, side=True))


@dsl_user_op
def lds_f32x2(addr, *, loc=None, ip=None):
    r = _asmN(2, T.f32(), [Int32(addr)], "ld.shared.v2.f32 {$0, $1}, [$2];", "=f,=f,r", loc, ip, side=True)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def mbar_init(addr, count: int, *, loc=None, ip=None):
    _asm1(None, [Int32(addr)], f"mbarrier.init.shared::cta.b64 [$0], {count};", "r", loc, ip, side=True)


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
def mbar_arrive(addr, *, loc=None, ip=None):
    _asm1(None, [Int32(addr)], "mbarrier.arrive.shared::cta.b64 _, [$0];", "r", loc, ip, side=True)


@dsl_user_op
def mbar_arrive_tx(addr, tx: int, *, loc=None, ip=None):
    _asm1(None, [Int32(addr)], f"mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], {tx};", "r", loc, ip, side=True)


@dsl_user_op
def bulk_g2s(dst, src, size: int, bar, *, loc=None, ip=None):
    _asm1(
        None, [Int32(dst), Int64(src), Int32(bar)],
        f"cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], {size}, [$2];",
        "r,l,r", loc, ip, side=True,
    )


@dsl_user_op
def named_bar(bid, nthreads: int, *, loc=None, ip=None):
    _asm1(None, [Int32(bid)], f"bar.sync $0, {nthreads};", "r", loc, ip, side=True)


@dsl_user_op
def syncwarp(*, loc=None, ip=None):
    _asm1(None, [], "bar.warp.sync -1;", "", loc, ip, side=True)


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc, ip)
    return [Uint32(v) for v in r]


@dsl_user_op
def cp_wait1(*, loc=None, ip=None):
    _asm1(None, [], "cp.async.wait_group 1;", "", loc, ip, side=True)


@dsl_user_op
def cp16(dst, src, *, loc=None, ip=None):
    _asm1(None, [Int32(dst), Int64(src)], "cp.async.cg.shared.global [$0], [$1], 16;", "r,l", loc, ip, side=True)


@dsl_user_op
def cp8(dst, src, *, loc=None, ip=None):
    _asm1(None, [Int32(dst), Int64(src)], "cp.async.ca.shared.global [$0], [$1], 8;", "r,l", loc, ip, side=True)


@dsl_user_op
def cp_commit(*, loc=None, ip=None):
    _asm1(None, [], "cp.async.commit_group;", "", loc, ip, side=True)


@dsl_user_op
def cp_wait0(*, loc=None, ip=None):
    _asm1(None, [], "cp.async.wait_group 0;", "", loc, ip, side=True)


@dsl_user_op
def sts_i64(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Int64(a)], "st.shared.b64 [$0], $1;", "r,l", loc, ip, side=True)


@dsl_user_op
def lds_i64(addr, *, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [Int32(addr)], "ld.shared.b64 $0, [$1];", "=l,r", loc, ip, side=True))


# ---------------------------------------------------------------- math primitives
@dsl_user_op
def hmax(a, b, *, loc=None, ip=None):
    """Per 16-bit half: max(|a|, |b|) with NaN propagation; sign bit is garbage (xor of signs)."""
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "max.NaN.xorsign.abs.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


@dsl_user_op
def prmt(a, b, sel: int, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], f"prmt.b32 $0, $1, $2, {sel};", "=r,r,r", loc, ip))


@dsl_user_op
def unpack2(w, *, loc=None, ip=None):
    """bf16x2 word -> 64-bit register pair {f32(lo half), f32(hi half)} (exact)."""
    return Int64(_asm1(
        T.i64(), [Uint32(w)],
        "{ .reg .b32 a, b; shl.b32 a, $1, 16; and.b32 b, $1, 0xFFFF0000; mov.b64 $0, {a, b}; }",
        "=l,r", loc, ip))


@dsl_user_op
def scale2(p, *, loc=None, ip=None):
    """p = two masked bf16 amax values (lo, hi).

    Returns (bytes16 = lo byte | hi byte << 8, recip pair {recip_lo, recip_hi} as a 64-bit register pair).
    recip bits = (254 - b) << 23, byte 255 -> 0x7FC00000 (NaN).
    """
    r = _asmN(
        2, [T.i32(), T.i64()], [Uint32(p)],
        "{ .reg .b32 a0, a1, b0, b1, e0, e1, t0, t1; .reg .b64 x, s; .reg .b16 h;\n"
        "shl.b32 a0, $2, 16;\n"
        "and.b32 a1, $2, 0xFFFF0000;\n"
        "mov.b64 x, {a0, a1};\n"
        f"mov.b64 s, {{{_INV448}, {_INV448}}};\n"
        "mul.rn.f32x2 x, x, s;\n"
        "mov.b64 {a0, a1}, x;\n"
        "cvt.rp.ue8m0x2.f32 h, a1, a0;\n"
        "cvt.u32.u16 $0, h;\n"
        "and.b32 b0, $0, 255;\n"
        "shr.u32 b1, $0, 8;\n"
        "mad.lo.s32 e0, b0, -8388608, 0x7F000000;\n"
        "mad.lo.s32 e1, b1, -8388608, 0x7F000000;\n"
        "mad.lo.s32 t0, b0, 16384, 16384;\n"
        "mad.lo.s32 t1, b1, 16384, 16384;\n"
        "or.b32 e0, e0, t0;\n"
        "or.b32 e1, e1, t1;\n"
        "and.b32 e0, e0, 0x7FC00000;\n"
        "and.b32 e1, e1, 0x7FC00000;\n"
        "mov.b64 $1, {e0, e1}; }",
        "=r,=l,r", loc, ip,
    )
    return Uint32(r[0]), Int64(r[1])


@dsl_user_op
def recip_lo(rp, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int64(rp)], "{ .reg .b32 a, b; mov.b64 {a, b}, $1; mov.b32 $0, a; }", "=r,l", loc, ip))


@dsl_user_op
def recip_hi(rp, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int64(rp)], "{ .reg .b32 a, b; mov.b64 {a, b}, $1; mov.b32 $0, b; }", "=r,l", loc, ip))


@dsl_user_op
def qrow4(p0, p1, r, *, loc=None, ip=None):
    """4 bytes e4m3(p * r) for two f32 pairs p0 = {c0, c1}, p1 = {c2, c3}; byte k = column k."""
    return Uint32(_asm1(
        T.i32(), [Int64(p0), Int64(p1), Uint32(r)],
        "{ .reg .b64 s, y0, y1; .reg .b32 g0, g1, g2, g3; .reg .b16 h0, h1;\n"
        "mov.b64 s, {$3, $3};\n"
        "mul.rn.f32x2 y0, $1, s;\n"
        "mul.rn.f32x2 y1, $2, s;\n"
        "mov.b64 {g0, g1}, y0;\n"
        "mov.b64 {g2, g3}, y1;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h0, g1, g0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h1, g3, g2;\n"
        "mov.b32 $0, {h0, h1}; }",
        "=r,l,l,r", loc, ip,
    ))


@dsl_user_op
def qcol4(p0, p1, p2, p3, rp, *, loc=None, ip=None):
    """Rows j..j+3 (pairs p0..p3 = {col a, col b}) scaled by column recips rp = {ra, rb}.

    Returns (word for column a: bytes = rows j..j+3, word for column b)."""
    r = _asmN(
        2, T.i32(), [Int64(p0), Int64(p1), Int64(p2), Int64(p3), Int64(rp)],
        "{ .reg .b64 y0, y1, y2, y3; .reg .b32 a0, b0, a1, b1, a2, b2, a3, b3; .reg .b16 h0, h1, h2, h3;\n"
        "mul.rn.f32x2 y0, $2, $6;\n"
        "mul.rn.f32x2 y1, $3, $6;\n"
        "mul.rn.f32x2 y2, $4, $6;\n"
        "mul.rn.f32x2 y3, $5, $6;\n"
        "mov.b64 {a0, b0}, y0;\n"
        "mov.b64 {a1, b1}, y1;\n"
        "mov.b64 {a2, b2}, y2;\n"
        "mov.b64 {a3, b3}, y3;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h0, a1, a0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h1, a3, a2;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h2, b1, b0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h3, b3, b2;\n"
        "mov.b32 $0, {h0, h1};\n"
        "mov.b32 $1, {h2, h3}; }",
        "=r,=r,l,l,l,l,l", loc, ip,
    )
    return Uint32(r[0]), Uint32(r[1])


@dsl_user_op
def unpack(w, *, loc=None, ip=None):
    r = _asmN(2, T.f32(), [Uint32(w)], "shl.b32 $0, $2, 16;\nand.b32 $1, $2, 0xFFFF0000;", "=f,=f,r", loc, ip)
    return Float32(r[0]), Float32(r[1])


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return Float32(_asm1(T.f32(), [Float32(a), Float32(b)], "add.rn.f32 $0, $1, $2;", "=f,f,f", loc, ip))


@dsl_user_op
def bf16_rt(a, *, loc=None, ip=None):
    """float(bf16_rn(a))."""
    return Float32(_asm1(
        T.f32(), [Float32(a)],
        "{ .reg .b16 h; cvt.rn.bf16.f32 h, $1; cvt.f32.bf16 $0, h; }", "=f,f", loc, ip))


@dsl_user_op
def rope_out(e, o, c, s, *, loc=None, ip=None):
    """bf16x2 word {lo: bf16(fma(e, c, o*s)), hi: bf16(fma(o, c, -(e*s)))}."""
    return Uint32(_asm1(
        T.i32(), [Float32(e), Float32(o), Float32(c), Float32(s)],
        "{ .reg .f32 p, q, oe, oo;\n"
        "mul.rn.f32 p, $2, $4;\n"
        "fma.rn.f32 oe, $1, $3, p;\n"
        "mul.rn.f32 q, $1, $4;\n"
        "neg.f32 q, q;\n"
        "fma.rn.f32 oo, $2, $3, q;\n"
        "cvt.rn.bf16x2.f32 $0, oo, oe; }",
        "=r,f,f,f,f", loc, ip,
    ))


class FusedKVBwdQuant:
    def __init__(self, num_ctas: int):
        self.num_ctas = num_ctas

    @cute.jit
    def __call__(self, gk, gv, cache, pos, kpe, qrow, srow, scol, qcol_t, stream):
        self.kernel(gk, gv, cache, pos, kpe, qrow, srow, scol, qcol_t).launch(
            grid=[self.num_ctas, 1, 1], block=[NTHR, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(self, gk, gv, cache, pos, kpe, qrow, srow, scol, qcol_t):
        G = self.num_ctas
        tid, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        w = tid // 32
        l = tid % 32

        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(SMEM_BYTES), byte_alignment=128)
        sb = sAll.iterator.toint()

        gk0 = gk.iterator.toint()
        gv0 = gv.iterator.toint()
        cache0 = cache.iterator.toint()
        cnt = (NTILES - 1 - b) // G + 1

        if w == 0:
            if l < cnt:
                kq0 = b + l * G
                tok0 = ((kq0 // 2048) * 16 + (kq0 % 16)) * NH + (kq0 % 2048) // 16
                sts_i64(sb + OFF_POS + l * 8, ld_i64(pos.iterator.toint() + Int64(tok0) * 8))
        cute.arch.barrier()

        qrow0 = qrow.iterator.toint()
        qcol0 = qcol_t.iterator.toint()
        srow0 = srow.iterator.toint()
        scol0 = scol.iterator.toint()
        kpe0 = kpe.iterator.toint()
        lr = l // 8
        lc = l % 8
        i_blk = w // 2

        # per-thread source of its 16 x 16 B data chunks: lane < 16 -> grad_k nope, else grad_v
        src0 = Int64(0)
        relem = Int64(0)
        if l < 16:
            src0 = gk0 + Int64(l * 16)
            relem = Int64(DK)
        else:
            src0 = gv0 + Int64((l - 16) * 16)
            relem = Int64(DV)

        def issue(n, G, b, sb, w, l, lr, lc, src0, relem, gk0, cache0):
            # Each thread prefetches exactly the stage bytes it reads itself (release-after-read):
            # tile n + 2 is issued into stage n % 2 right after this thread read tile n from it.
            k = b + n * G
            tt = (k // 2048) * 16 + (k % 16)
            h = (k % 2048) // 16
            t0 = Int64(tt * TT + w * 16)
            so = (n % STAGES) * STAGE_BYTES
            src = src0 + (t0 * NH + Int64(h)) * relem * 2
            for j in cutlass.range_constexpr(16):
                cp16(sb + so + (w * 16 + j) * 512 + l * 16, src + relem * (NH * 2 * j))
            if w == 0:
                pv = lds_i64(sb + OFF_POS + n * 8)
                cp8(sb + so + OFF_CS + l * 8, cache0 + (pv * 32 + Int64(l)) * 8)

        def rope_load(n, G, b, w, lr, lc, gk0, rbuf):
            # grad_k rope rows of this tile's token, Triton layout (warps 0..3): heads 16 i + 4 w + lr
            k = b + n * G
            tt = (k // 2048) * 16 + (k % 16)
            h = (k % 2048) // 16
            rb = gk0 + ((Int64(tt * NH + h) * NH + Int64(4 * w + lr)) * DK + 128 + Int64(lc * 8)) * 2
            for i in cutlass.range_constexpr(8):
                v4 = ld_v4(rb + i * (16 * DK * 2))
                for c in cutlass.range_constexpr(4):
                    rbuf[i * 4 + c] = v4[c]

        rbuf = cute.make_rmem_tensor(cute.make_layout(32), Uint32)
        if w < 4:
            rope_load(Int32(0), G, b, w, lr, lc, gk0, rbuf)
        issue(Int32(0), G, b, sb, w, l, lr, lc, src0, relem, gk0, cache0)
        cp_commit()
        if cnt > 1:
            issue(Int32(1), G, b, sb, w, l, lr, lc, src0, relem, gk0, cache0)
        cp_commit()
        for n in cutlass.range(cnt, unroll=1):
            k = b + n * G
            tt = (k // 2048) * 16 + (k % 16)
            h = (k % 2048) // 16
            h64 = Int64(h)
            pb = n % 2
            t0 = Int64(tt * TT + w * 16)
            sColT = sb + OFF_COLT + pb * 32768
            sScl = sb + OFF_SCL + pb * 2048

            so = (n % STAGES) * STAGE_BYTES
            cp_wait1()
            x = []
            for j in cutlass.range_constexpr(16):
                x.append(lds_v4(sb + so + (w * 16 + j) * 512 + l * 16))

            # ---- grad_k_pe (warps 0..3, Triton layout); warp 0 finalizes ----
            if w < 4:
                rv = []
                for i in cutlass.range_constexpr(8):
                    rv.append([rbuf[i * 4 + c] for c in range(4)])
                if n + 1 < cnt:
                    rope_load(n + 1, G, b, w, lr, lc, gk0, rbuf)
                sums = []
                for par in cutlass.range_constexpr(2):
                    for p in cutlass.range_constexpr(4):
                        v = []
                        for i in cutlass.range_constexpr(8):
                            lo, hi = unpack(rv[i][p])
                            v.append(lo if par == 0 else hi)
                        sm = fadd(fadd(fadd(v[0], v[1]), fadd(v[2], v[3])),
                                  fadd(fadd(v[4], v[5]), fadd(v[6], v[7])))
                        sm = fadd(sm, cute.arch.shuffle_sync_bfly(sm, offset=16))
                        sm = fadd(sm, cute.arch.shuffle_sync_bfly(sm, offset=8))
                        sums.append(sm)
                if l < 8:
                    for par in cutlass.range_constexpr(2):
                        sts_f32x4(sb + OFF_RSUM + (((pb * 2 + par) * 4 + w) * 32 + lc * 4) * 4,
                                  sums[par * 4 + 0], sums[par * 4 + 1], sums[par * 4 + 2], sums[par * 4 + 3])
                named_bar(5, 128)
                if w == 0:
                    cs_c, cs_s = lds_f32x2(sb + so + OFF_CS + l * 8)
                    a = []
                    for par in cutlass.range_constexpr(2):
                        for ww in cutlass.range_constexpr(4):
                            a.append(lds_f32(sb + OFF_RSUM + (((pb * 2 + par) * 4 + ww) * 32 + l) * 4))
                    ev = fadd(fadd(fadd(a[0], a[2]), fadd(a[1], a[3])), Float32(0.0))
                    od = fadd(fadd(fadd(a[4], a[6]), fadd(a[5], a[7])), Float32(0.0))
                    st_b32(kpe0 + (Int64(tt * NH + h) * 64 + Int64(l) * 2) * 2,
                           rope_out(bf16_rt(ev), bf16_rt(od), cs_c, cs_s))
            # this thread has read all of its bytes of stage n % 2: prefetch tile n + 2 into it
            if n + 2 < cnt:
                issue(n + 2, G, b, sb, w, l, lr, lc, src0, relem, gk0, cache0)
            cp_commit()

            # ---- colwise amax over the warp pair's 32 rows ----
            cpart = []
            for q in cutlass.range_constexpr(4):
                m = hmax(x[0][q], x[1][q])
                for j in cutlass.range_constexpr(2, 16):
                    m = hmax(m, x[j][q])
                cpart.append(m)
            sts_v4(sb + OFF_PART + (pb * NCONS + tid) * 16, cpart[0], cpart[1], cpart[2], cpart[3])
            named_bar(1 + i_blk, 64)
            other = lds_v4(sb + OFF_PART + (pb * NCONS + (tid ^ 32)) * 16)
            crp = []
            cby = []
            for q in cutlass.range_constexpr(4):
                cm = hmax(cpart[q], other[q]) & Uint32(0x7FFF7FFF)
                b16, rp = scale2(cm)
                cby.append(b16)
                crp.append(rp)
            # colwise scale bytes (even warp: this lane's columns 0..3, odd warp: 4..7) -> smem tile
            for kk in cutlass.range_constexpr(4):
                cbyte = Uint32(0)
                if w % 2 == 0:
                    cbyte = cby[kk // 2] if kk % 2 == 0 else cby[kk // 2] >> Uint32(8)
                else:
                    cbyte = cby[2 + kk // 2] if kk % 2 == 0 else cby[2 + kk // 2] >> Uint32(8)
                cc = l * 8 + (w % 2) * 4 + kk
                sts_u8(sScl + 1024 + (cc // 128) * 512 + (cc % 32) * 16 + ((cc % 128) // 32) * 4 + i_blk, cbyte)

            # ---- rowwise amax/scales + quantize ----
            ccol = [[None] * 4 for _ in range(8)]
            for quad in cutlass.range_constexpr(4):
                j0 = quad * 4
                rr = []
                for pr in cutlass.range_constexpr(2):
                    ja = j0 + pr * 2
                    ma = hmax(hmax(x[ja][0], x[ja][1]), hmax(x[ja][2], x[ja][3]))
                    mb = hmax(hmax(x[ja + 1][0], x[ja + 1][1]), hmax(x[ja + 1][2], x[ja + 1][3]))
                    pp = hmax(prmt(ma, mb, 0x5410), prmt(ma, mb, 0x7632))
                    pp = hmax(pp, cute.arch.shuffle_sync_bfly(pp, offset=1))
                    pp = hmax(pp, cute.arch.shuffle_sync_bfly(pp, offset=2))
                    pp = pp & Uint32(0x7FFF7FFF)
                    b16, rp = scale2(pp)
                    rr.append(recip_lo(rp))
                    rr.append(recip_hi(rp))
                    if l % 4 == 0:
                        g = l // 4
                        rbase = sScl + (g // 4) * 512 + ((w % 2) * 16 + ja) * 16 + i_blk * 4 + (g % 4)
                        sts_u8(rbase, b16)
                        sts_u8(rbase + 16, b16 >> Uint32(8))
                P = []
                for jj in cutlass.range_constexpr(4):
                    P.append([unpack2(x[j0 + jj][q]) for q in range(4)])
                for jj in cutlass.range_constexpr(4):
                    y0 = qrow4(P[jj][0], P[jj][1], rr[jj])
                    y1 = qrow4(P[jj][2], P[jj][3], rr[jj])
                    st_v2(qrow0 + (t0 + (j0 + jj)) * NCOL + h64 * 256 + Int64(l) * 8, y0, y1)
                for q in cutlass.range_constexpr(4):
                    ca, cb = qcol4(P[0][q], P[1][q], P[2][q], P[3][q], crp[q])
                    ccol[2 * q][quad] = ca
                    ccol[2 * q + 1][quad] = cb
            # colwise bytes -> transposed smem tile: column c = 8l + cidx, 16 B chunk = this warp's 16 tokens
            for cidx in cutlass.range_constexpr(8):
                sts_v4(sColT + (l * 8 + cidx) * 128 + ((w ^ (l % 8)) * 16),
                       ccol[cidx][0], ccol[cidx][1], ccol[cidx][2], ccol[cidx][3])

            cute.arch.barrier()

            # ---- coalesced write-out: colwise tile rows (128 B each), scale tiles ----
            for mm in cutlass.range_constexpr(8):
                row = w * 32 + mm * 4 + l // 8
                ch = l % 8
                vt = lds_v4(sColT + row * 128 + ((ch ^ ((row // 8) % 8)) * 16))
                st_v4(qcol0 + (h64 * 256 + Int64(row)) * NT + Int64(tt * TT + ch * 16), vt[0], vt[1], vt[2], vt[3])
            if tid < 64:
                vsr = lds_v4(sScl + tid * 16)
                st_v4(srow0 + Int64(tt) * 131072 + h64 * 1024 + Int64(tid) * 16, vsr[0], vsr[1], vsr[2], vsr[3])
            elif tid < 128:
                kq = tid - 64
                vsc = lds_v4(sScl + 1024 + kq * 16)
                st_v4(scol0 + Int64(tt) * 512 + (h64 * 2 + Int64(kq // 32)) * 16384 + Int64(kq % 32) * 16,
                      vsc[0], vsc[1], vsc[2], vsc[3])


_DT = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32, torch.int64: cutlass.Int64,
       torch.uint8: cutlass.Uint8}


def _fake(dtype, shape):
    stride = []
    acc = 1
    for s in reversed(shape):
        stride.append(acc)
        acc *= s
    return cute.runtime.make_fake_tensor(_DT[dtype], tuple(shape), stride=tuple(reversed(stride)), assumed_align=16)


_SIG = [
    (torch.bfloat16, (NT, NH, DK)),
    (torch.bfloat16, (NT, NH, DV)),
    (torch.float32, (NT, 32, 2)),
    (torch.int64, (1, NT)),
    (torch.bfloat16, (NT, 64)),
    (torch.uint8, (NT, NCOL)),
    (torch.uint8, (NT, NCOL // 32)),
    (torch.uint8, (NCOL, NT // 32)),
    (torch.uint8, (NCOL, NT)),
]

_NUM_CTAS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
if NTILES > 32 * _NUM_CTAS:  # the per-CTA position table holds 32 tiles
    raise ValueError(f"needs >= {NTILES // 32} SMs, got {_NUM_CTAS}")

_fn = cute.compile(
    FusedKVBwdQuant(_NUM_CTAS),
    *[_fake(d, s) for d, s in _SIG],
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi"
    + (f" --keep-ptx --keep-cubin --dump-dir={os.environ['DEX_DUMP']}" if os.environ.get("DEX_DUMP") else ""),
)


def _check(t, dtype, shape):
    if t.dtype != dtype or tuple(t.shape) != shape or not t.is_contiguous() or t.data_ptr() % 16:
        raise ValueError(f"expected contiguous 16 B aligned {dtype} {shape}, got {t.dtype} {tuple(t.shape)} "
                         f"stride {t.stride()}")


def run(in0, in1, in2, in3):
    _check(in0, torch.bfloat16, (NT, NH, DV))
    _check(in1, torch.bfloat16, (NT, NH, DK))
    _check(in2, torch.float32, (NT, 32, 2))
    _check(in3, torch.int64, (1, NT))
    dev = in0.device
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.utils.deterministic.fill_uninitialized_memory = False
    try:
        kpe = torch.empty((NT, 64), dtype=torch.bfloat16, device=dev)
        qrow = torch.empty((NT, NCOL), dtype=torch.uint8, device=dev)
        srow = torch.empty((NT, NCOL // 32), dtype=torch.uint8, device=dev)
        scol = torch.empty((NCOL, NT // 32), dtype=torch.uint8, device=dev)
        qcol = torch.empty_strided((NT, NCOL), (1, NT), dtype=torch.uint8, device=dev)  # column major
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = fill
    _fn(in1, in0, in2, in3, kpe, qrow, srow, scol, qcol.t())
    return (
        kpe,
        qrow.view(torch.float8_e4m3fn),
        srow.view(torch.float8_e8m0fnu),
        scol.view(torch.float8_e8m0fnu),
        qcol.view(torch.float8_e4m3fn).t(),
    )
