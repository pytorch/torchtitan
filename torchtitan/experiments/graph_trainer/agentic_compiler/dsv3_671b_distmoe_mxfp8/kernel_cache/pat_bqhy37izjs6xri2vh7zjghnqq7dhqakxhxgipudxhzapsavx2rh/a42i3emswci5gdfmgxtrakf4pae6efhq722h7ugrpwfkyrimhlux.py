"""pat_bqhy... / a42i3emswci5gdfmgxtrakf4pae6efhq722h7ugrpwfkyrimhlux.py: rowwise+colwise mxfp8_quantize(in0) -> _scaled_mm_v2(qrow, in1, [rearranged srow], [3], [1], [in2], [3], [1]); returns (qcol.t(), gemm, rearranged colwise scales).
in0 [4096, 4096] bf16, in1 [4096, 7168] float8_e4m3fn, in2 [56, 32, 32, 16] float8_e8m0fnu.

The GEMM is the reference's own torch.ops.aten._scaled_mm_v2 call with identical arguments on
tensors with identical shapes, strides, dtypes and (allocator) alignment; only the quantize +
triton_mx_block_rearrange part is replaced by one CuTe DSL kernel that writes the scales directly
in the rearranged layout.

Quantize numerics (torchao mxfp8_quantize, e4m3, rceil; emulation checked against torchao for
colwise-only and rowwise+colwise on every bf16 amax pattern and x sweeps):
  amax = NaN-propagating max |x| over the 32-element block; for bf16 amax bits a:
  byte = 255 if a >= 0x7f80 else max(((a + 31) >> 7) - 8, 0);
  recip = 0x7fffffff if byte == 255 else (254 - byte) << 23; q = cvt.rn.satfinite.e4m3(x *rn recip).
triton_mx_block_rearrange of a logical [rows, cols] scale tensor: (r, c) -> (c // 4) * 512
  + (r // 128) * 128 * cols_pad + (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4, zero padding
  (rows to a multiple of 128, cols to a multiple of 4). Each 128x128 tile of x owns exactly one
  512-byte block of each rearranged scale tensor.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# ---- specialization: x [4096, 4096] ----
R = 4096
C = 4096
ROWWISE = True
# schedule (measured per shape on GB300): cp.async prefetch depth, CTAs per SM, tile order
STAGES = 1
CPS = 2
COL_FAST = True

NRB = R // 32  # colwise row blocks
NCB = C // 32  # rowwise column blocks
NCB_PAD = -(-NCB // 4) * 4  # triton_mx_block_rearrange output cols of the rowwise scales
C_PAD = -(-C // 128) * 128  # triton_mx_block_rearrange output rows of the colwise scales
NCT = -(-C // 128)  # 128-column tiles
NTHR = 128
COL_EVEN = C % 128 == 0


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
def ld_stream_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def ld_stream_v4_pred(addr, p, *, loc=None, ip=None):
    """16-byte load if p != 0, else zeros (zero columns give amax 0 -> scale byte 0 == rearrange padding)."""
    r = _asmN(4, T.i32(), [Int64(addr), Int32(p)],
              "{ .reg .pred q; setp.ne.s32 q, $5, 0;\n"
              "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;\n"
              "@q ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4]; }",
              "=r,=r,=r,=r,l,r", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
          "st.global.L1::no_allocate.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def st_v2(addr, a, b, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b)], "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", loc, ip, side=True)


@dsl_user_op
def sts_u8(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Uint32(a)], "st.shared.u8 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int32(addr)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def cp16(dst, src, nbytes, *, loc=None, ip=None):
    """cp.async 16 bytes global -> shared; nbytes = 0 zero-fills the destination."""
    _asm1(None, [Int32(dst), Int64(src), Int32(nbytes)], "cp.async.cg.shared.global [$0], [$1], 16, $2;", "r,l,r", loc, ip, side=True)


@dsl_user_op
def cp_commit(*, loc=None, ip=None):
    _asm1(None, [], "cp.async.commit_group;", "", loc, ip, side=True)


@dsl_user_op
def cp_wait(n: int, *, loc=None, ip=None):
    _asm1(None, [], f"cp.async.wait_group {n};", "", loc, ip, side=True)


# ---------------------------------------------------------------- math
@dsl_user_op
def hmax(a, b, *, loc=None, ip=None):
    """Per 16-bit half: NaN-propagating max |.| of bf16 (sign bit is garbage)."""
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "max.NaN.xorsign.abs.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


@dsl_user_op
def hmax4(y0, y1, y2, y3, *, loc=None, ip=None):
    return Uint32(_asm1(
        T.i32(), [Uint32(y0), Uint32(y1), Uint32(y2), Uint32(y3)],
        "{ .reg .b32 a, b;\n"
        "max.NaN.xorsign.abs.bf16x2 a, $1, $2;\n"
        "max.NaN.xorsign.abs.bf16x2 b, $3, $4;\n"
        "max.NaN.xorsign.abs.bf16x2 $0, a, b; }",
        "=r,r,r,r,r", loc, ip))


@dsl_user_op
def scale_of(m, *, loc=None, ip=None):
    """m = packed per-half block |max|; the block amax is the max of both halves.
    Returns (e8m0 byte, recip f32 bits): a = bf16 bits of amax (> 0x7f80 is NaN);
    byte = 255 if a >= 0x7f80 else max(((a + 31) >> 7) - 8, 0) (== torchao rceil for all 2^16 a);
    recip = 0x7fffffff if byte == 255 else (254 - byte) << 23."""
    r = _asmN(
        2, T.i32(), [Uint32(m)],
        "{ .reg .b32 a, b, e, r; .reg .pred p;\n"
        "and.b32 a, $2, 0x7FFF7FFF;\n"
        "shr.u32 b, a, 16;\n"
        "and.b32 a, a, 0xFFFF;\n"
        "max.u32 a, a, b;\n"
        "add.s32 e, a, -993;\n"
        "shr.s32 e, e, 7;\n"
        "max.s32 e, e, 0;\n"
        "setp.ge.u32 p, a, 0x7F80;\n"
        "mad.lo.s32 r, e, -8388608, 0x7F000000;\n"
        "selp.b32 $0, 255, e, p;\n"
        "selp.b32 $1, 0x7FFFFFFF, r, p; }",
        "=r,=r,r", loc, ip,
    )
    return Uint32(r[0]), Uint32(r[1])


@dsl_user_op
def scale2(m, *, loc=None, ip=None):
    """Same as scale_of for each 16-bit half of m separately (two columns).
    Returns (byte_lo, byte_hi, recip_lo, recip_hi)."""
    r = _asmN(
        4, T.i32(), [Uint32(m)],
        "{ .reg .b32 a, b, e, f, r, s; .reg .pred p, q;\n"
        "and.b32 a, $4, 0x7FFF;\n"
        "shr.u32 b, $4, 16; and.b32 b, b, 0x7FFF;\n"
        "add.s32 e, a, -993; shr.s32 e, e, 7; max.s32 e, e, 0;\n"
        "add.s32 f, b, -993; shr.s32 f, f, 7; max.s32 f, f, 0;\n"
        "setp.ge.u32 p, a, 0x7F80;\n"
        "setp.ge.u32 q, b, 0x7F80;\n"
        "mad.lo.s32 r, e, -8388608, 0x7F000000;\n"
        "mad.lo.s32 s, f, -8388608, 0x7F000000;\n"
        "selp.b32 $0, 255, e, p;\n"
        "selp.b32 $1, 255, f, q;\n"
        "selp.b32 $2, 0x7FFFFFFF, r, p;\n"
        "selp.b32 $3, 0x7FFFFFFF, s, q; }",
        "=r,=r,=r,=r,r", loc, ip,
    )
    return Uint32(r[0]), Uint32(r[1]), Uint32(r[2]), Uint32(r[3])


@dsl_user_op
def quant_row4(y0, y1, r, *, loc=None, ip=None):
    """4 e4m3 bytes (byte k = column k) of the bf16 pairs y0, y1 scaled by recip r (f32 bits)."""
    return Uint32(_asm1(
        T.i32(), [Uint32(y0), Uint32(y1), Uint32(r)],
        "{ .reg .b32 a, b, c, d; .reg .b64 p, q, s; .reg .b16 h0, h1;\n"
        "shl.b32 a, $1, 16; and.b32 b, $1, 0xFFFF0000;\n"
        "shl.b32 c, $2, 16; and.b32 d, $2, 0xFFFF0000;\n"
        "mov.b64 p, {a, b}; mov.b64 q, {c, d}; mov.b64 s, {$3, $3};\n"
        "mul.rn.f32x2 p, p, s;\n"
        "mul.rn.f32x2 q, q, s;\n"
        "mov.b64 {a, b}, p; mov.b64 {c, d}, q;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h0, b, a;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h1, d, c;\n"
        "mov.b32 $0, {h0, h1}; }",
        "=r,r,r,r", loc, ip))


@dsl_user_op
def quant_col4(w0, w1, w2, w3, rlo, rhi, *, loc=None, ip=None):
    """Words w0..w3 = rows t..t+3 of one bf16x2 column pair (lo = column a, hi = column b).
    Returns (column-a word, column-b word): byte i = e4m3(row t+i *rn recip of that column)."""
    r = _asmN(
        2, T.i32(), [Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3), Uint32(rlo), Uint32(rhi)],
        "{ .reg .b32 a0, a1, a2, a3, b0, b1, b2, b3; .reg .b64 p, q, u, v, sa, sb; .reg .b16 h0, h1, h2, h3;\n"
        "shl.b32 a0, $2, 16; and.b32 b0, $2, 0xFFFF0000;\n"
        "shl.b32 a1, $3, 16; and.b32 b1, $3, 0xFFFF0000;\n"
        "shl.b32 a2, $4, 16; and.b32 b2, $4, 0xFFFF0000;\n"
        "shl.b32 a3, $5, 16; and.b32 b3, $5, 0xFFFF0000;\n"
        "mov.b64 sa, {$6, $6}; mov.b64 sb, {$7, $7};\n"
        "mov.b64 p, {a0, a1}; mov.b64 q, {a2, a3}; mov.b64 u, {b0, b1}; mov.b64 v, {b2, b3};\n"
        "mul.rn.f32x2 p, p, sa; mul.rn.f32x2 q, q, sa;\n"
        "mul.rn.f32x2 u, u, sb; mul.rn.f32x2 v, v, sb;\n"
        "mov.b64 {a0, a1}, p; mov.b64 {a2, a3}, q; mov.b64 {b0, b1}, u; mov.b64 {b2, b3}, v;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h0, a1, a0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h1, a3, a2;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h2, b1, b0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h3, b3, b2;\n"
        "mov.b32 $0, {h0, h1};\n"
        "mov.b32 $1, {h2, h3}; }",
        "=r,=r,r,r,r,r,r,r", loc, ip,
    )
    return Uint32(r[0]), Uint32(r[1])


NTILES = (R // 128) * NCT
STAGE_BYTES = 16 * NTHR * 16  # each thread's 16 x 16 B of one tile
OFF_SCL = STAGES * STAGE_BYTES  # [2][1024] scale staging, double-buffered by tile parity
SMEM_BYTES = OFF_SCL + 2048


class MxQuant:
    """Persistent: CTA b handles tiles b, b + G, ... (tile id = tj * (R / 128) + ti, row tiles fastest).
    Tile = 128 rows x 128 columns, 4 warps; warp w owns rows 32w..32w+31 (one colwise block), lane l
    owns the 16-byte column chunk l % 16 (8 columns) of rows 16 (l // 16) .. + 15. Every thread
    cp.async-prefetches exactly the shared-memory bytes it later reads itself into a STAGES-deep
    ring, so the ring needs no cross-thread synchronization."""

    def __init__(self, num_ctas: int):
        self.num_ctas = num_ctas

    @cute.jit
    def __call__(self, x, qrow, srow, qcol, scol, stream):
        self.kernel(x, qrow, srow, qcol, scol).launch(grid=[self.num_ctas, 1, 1], block=[NTHR, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x, qrow, srow, qcol, scol):
        G = self.num_ctas
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        w = tid // 32
        l = tid % 32
        cq = l % 16
        hh = l // 16

        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(SMEM_BYTES), byte_alignment=128)
        sb = sAll.iterator.toint()
        x0 = x.iterator.toint()
        cnt = (NTILES - 1 - bid) // G + 1
        myslot = sb + tid * 16

        def issue(n, cnt, G, bid, x0, myslot, w, hh, cq):
            if n < cnt:
                tn = bid + n * G
                pti = tn // NCT if COL_FAST else tn % (R // 128)
                ptj = tn % NCT if COL_FAST else tn // (R // 128)
                c0 = ptj * 128 + cq * 8
                nb = Int32(16)
                if cutlass.const_expr(not COL_EVEN):
                    nb = Int32(c0 < C) * 16
                src = x0 + (Int64(pti * 128 + w * 32 + hh * 16) * C + Int64(c0)) * 2
                dst = myslot + (n % STAGES) * STAGE_BYTES
                for t in cutlass.range_constexpr(16):
                    cp16(dst + t * (NTHR * 16), src + t * (C * 2), nb)
            cp_commit()

        for n0 in cutlass.range_constexpr(STAGES):
            issue(Int32(n0), cnt, G, bid, x0, myslot, w, hh, cq)

        for n in cutlass.range(cnt, unroll=1):
            tn = bid + n * G
            ti = tn // NCT if COL_FAST else tn % (R // 128)
            tj = tn % NCT if COL_FAST else tn // (R // 128)
            ssc = sb + OFF_SCL + (n % 2) * 1024
            cp_wait(STAGES - 1)
            src_s = myslot + (n % STAGES) * STAGE_BYTES
            xv = []
            for t in cutlass.range_constexpr(16):
                xv.append(lds_v4(src_s + t * (NTHR * 16)))
            r0 = ti * 128 + w * 32 + hh * 16
            c0 = tj * 128 + cq * 8
            valid = Int32(1)
            if cutlass.const_expr(not COL_EVEN):
                valid = Int32(c0 < C)

            # colwise amax first (consumes every loaded word), then refill this stage
            ms = []
            for k in cutlass.range_constexpr(4):
                m = hmax(xv[0][k], xv[1][k])
                for t in cutlass.range_constexpr(2, 16):
                    m = hmax(m, xv[t][k])
                ms.append(hmax(m, cute.arch.shuffle_sync_bfly(m, offset=16)))
            issue(n + STAGES, cnt, G, bid, x0, myslot, w, hh, cq)

            qca = qcol.iterator.toint() + Int64(c0) * R + Int64(r0)
            for k in cutlass.range_constexpr(4):
                blo, bhi, rlo, rhi = scale2(ms[k])
                colw_a = []
                colw_b = []
                for t4 in cutlass.range_constexpr(4):
                    ca, cb = quant_col4(xv[4 * t4][k], xv[4 * t4 + 1][k], xv[4 * t4 + 2][k], xv[4 * t4 + 3][k], rlo, rhi)
                    colw_a.append(ca)
                    colw_b.append(cb)
                if valid != 0:
                    st_v4(qca + (2 * k) * R, colw_a[0], colw_a[1], colw_a[2], colw_a[3])
                    st_v4(qca + (2 * k + 1) * R, colw_b[0], colw_b[1], colw_b[2], colw_b[3])
                if hh == 0:
                    for half in cutlass.range_constexpr(2):
                        cc = cq * 8 + 2 * k + half
                        sts_u8(ssc + (cc % 32) * 16 + (cc // 32) * 4 + w, blo if half == 0 else bhi)

            if cutlass.const_expr(ROWWISE):
                qra = qrow.iterator.toint() + Int64(r0) * C + Int64(c0)
                for t in cutlass.range_constexpr(16):
                    m = hmax4(xv[t][0], xv[t][1], xv[t][2], xv[t][3])
                    m = hmax(m, cute.arch.shuffle_sync_bfly(m, offset=1))
                    m = hmax(m, cute.arch.shuffle_sync_bfly(m, offset=2))
                    byte, rcp = scale_of(m)
                    lo = quant_row4(xv[t][0], xv[t][1], rcp)
                    hi = quant_row4(xv[t][2], xv[t][3], rcp)
                    if valid != 0:
                        st_v2(qra + t * C, lo, hi)
                    if cq % 4 == 0:
                        sts_u8(ssc + 512 + (hh * 16 + t) * 16 + w * 4 + cq // 4, byte)

            cute.arch.barrier()
            if tid < 32:
                sv = lds_v4(ssc + tid * 16)
                st_v4(scol.iterator.toint() + Int64(tj * (NRB * 128) + ti * 512 + tid * 16), sv[0], sv[1], sv[2], sv[3])
            if cutlass.const_expr(ROWWISE):
                if tid >= 32:
                    if tid < 64:
                        sv2 = lds_v4(ssc + 512 + (tid - 32) * 16)
                        st_v4(srow.iterator.toint() + Int64(ti * (NCB_PAD * 128) + tj * 512 + (tid - 32) * 16),
                              sv2[0], sv2[1], sv2[2], sv2[3])


def _fake(dtype, shape):
    stride, acc = [], 1
    for s in reversed(shape):
        stride.append(acc)
        acc *= s
    return cute.runtime.make_fake_tensor(dtype, tuple(shape), stride=tuple(reversed(stride)), assumed_align=16)


_NUM_SMS = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
_quant = cute.compile(
    MxQuant(min(NTILES, _NUM_SMS * CPS)),
    _fake(cutlass.BFloat16, (R, C)),
    _fake(cutlass.Uint8, (R, C)) if ROWWISE else None,
    _fake(cutlass.Uint8, (R, NCB_PAD)) if ROWWISE else None,
    _fake(cutlass.Uint8, (C, R)),
    _fake(cutlass.Uint8, (C_PAD, NRB)),
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)


def _quantize(x):
    """Returns the reference's quantize + rearrange outputs with identical shapes and strides:
    (rowwise q [R, C] row-major, rearranged rowwise scales [R, NCB_PAD]) if ROWWISE else (None, None),
    colwise q [R, C] with strides (1, R), rearranged colwise scales [C_PAD, NRB]."""
    dev = x.device
    qcol = torch.empty_strided((R, C), (1, R), dtype=torch.float8_e4m3fn, device=dev)
    scol = torch.empty((C_PAD, NRB), dtype=torch.float8_e8m0fnu, device=dev)
    qrow = srow = None
    if ROWWISE:
        qrow = torch.empty((R, C), dtype=torch.float8_e4m3fn, device=dev)
        srow = torch.empty((R, NCB_PAD), dtype=torch.float8_e8m0fnu, device=dev)
    _quant(x, qrow.view(torch.uint8) if ROWWISE else None, srow.view(torch.uint8) if ROWWISE else None,
           qcol.t().view(torch.uint8), scol.view(torch.uint8))
    return qrow, srow, qcol, scol


def _dense(t):
    """The quantize kernel uses 16-byte vector loads on contiguous rows."""
    return t if t.is_contiguous() and t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.contiguous_format)


def run(in0, in1, in2):
    qrow, srow, qcol, scol = _quantize(_dense(in0))
    out = torch.ops.aten._scaled_mm_v2.default(qrow, in1, [srow], [3], [1], [in2], [3], [1], None, torch.bfloat16)
    return (qcol.t(), out, scol)
