"""dist_moe::prepare_mxfp8_weight (training: MXFP8 E4M3, 32x32 blocks) as one fused CuTe DSL kernel.

Reference: dist_moe _prepare_mxfp8_weight_op -> _prepare_blockscaled_weight_impl ->
quantize_block_scaled_axes(TWO_D, CUBLAS_BLOCKED) -> _QuantizeBlockScaledRowCol in mode
TILE2D_FP8_VECTOR (one kernel, one warp per 32x32 tile, 64-byte loads per lane from 32 different
rows, byte stores for the scales). The weight [G, RG, C] is flattened to [R = G * RG, C]. Outputs
(three separate buffers): qdata e4m3 [G, RG, C] row major (FPROP and DGRAD share it), FPROP scales
[R, C / 32] and DGRAD scales [G * C, RG / 32], both cuBLAS-blocked e8m0 without padding.

Numerics, decoded from the reference PTX and reproduced with explicit PTX (no ftz, no contraction):
* tile amax = max |x| over the 32x32 block, where any NaN or inf gives +inf (reference: per-row
  max.NaN.f32 chain of abs.f32, NaN -> +inf, then max.f32 butterfly over the 32 rows). Here:
  max.NaN.xorsign.abs.bf16x2 chain (exact max of |bf16|, NaN carried, sign bit garbage), & 0x7FFF,
  NaN -> 0x7F80, << 16.
* s = amax *rn fl32(1/448) (0x3B124925); byte = cvt.rp.ue8m0x2.f32(0, s) low byte;
  recip = s != +inf ? bits(0x7F000000 - ((byte << 23) & 0x7F800000)) : NaN (0x7FC00000).
* q = cvt.rn.satfinite.e4m3x2.f32(x *rn recip) (mul.rn.f32x2).
Scale layouts: a 128-row x 128-col region holds 4 x 4 tiles (rbq, kbq). Its FPROP atom is the
16-byte pattern [rbq * 4 + kbq] repeated 32 times at atom index (row // 128) * (C / 128) + col // 128;
its DGRAD atom is the pattern [kbq * 4 + rbq] repeated 32 times at atom index
g * (C / 128) * (RG / 128) + (col // 128) * (RG / 128) + (row_in_group // 128) (512 bytes per atom).

Work split: CTA = 128 rows x 128 cols. Lane l owns VB columns (VB = 8: 16-byte ld.global.v4, VB = 16:
32-byte ld.global.v8) of RPT rows kept in registers; warp w covers RPW = RPT * 32 / (128 / VB) rows.
Tile amax: in-thread max, shfl_xor over the lanes of a tile row and over the row lanes, then (if a
row block spans WPB > 1 warps) the partial maxima of those warps through shared memory (named
barrier per row block). qdata goes straight from registers to global (full 128-byte lines per row).
The scale patterns are assembled in shared memory and written as two full 512-byte atoms after
one CTA barrier.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

G = 4
RG = 7168
C = 2048
# (columns per lane, rows per thread) of the kernel used for 32-byte aligned inputs, and of the
# one used for inputs that are only 16-byte aligned (None: 16-byte aligned inputs use PRIMARY).
PRIMARY = (8, 16)
FALLBACK_16B = None

R = G * RG
NUMEL = R * C
TR = 128  # CTA rows
TC = 128  # CTA cols
NCTA = (R // TR) * (C // TC)
FP_ATOMS_PER_ROW = C // 128  # FPROP atoms per 128-row band
DG_ATOMS_PER_COL = RG // 128  # DGRAD atoms per 128-col band of one group
assert R % TR == 0 and C % TC == 0 and RG % TR == 0


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
def ld_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip)
    return [Uint32(v) for v in r]


@dsl_user_op
def ld_v8(addr, *, loc=None, ip=None):
    r = _asmN(8, T.i32(), [Int64(addr)],
              "ld.global.nc.L1::no_allocate.v8.b32 {$0, $1, $2, $3, $4, $5, $6, $7}, [$8];",
              "=r,=r,=r,=r,=r,=r,=r,=r,l", loc, ip)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_v2(addr, a, b, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b)], "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", loc, ip, side=True)


@dsl_user_op
def st_v4(addr, a, b, c, d, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b), Uint32(c), Uint32(d)],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def sts_u8(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Uint32(a)], "st.shared.u8 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def sts_b32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Uint32(a)], "st.shared.b32 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def lds_b32(addr, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int32(addr)], "ld.shared.b32 $0, [$1];", "=r,r", loc, ip, side=True))


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int32(addr)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def named_bar(bid, nthreads: int, *, loc=None, ip=None):
    _asm1(None, [Int32(bid)], f"bar.sync $0, {nthreads};", "r", loc, ip, side=True)


@dsl_user_op
def hmax(a, b, *, loc=None, ip=None):
    """Per 16-bit half: max(|a|, |b|), NaN -> canonical NaN; sign bit garbage (xor of signs)."""
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "max.NaN.xorsign.abs.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


@dsl_user_op
def shfl_xor(a, off: int, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(a)], f"shfl.sync.bfly.b32 $0, $1, {off}, 31, -1;", "=r,r", loc, ip))


@dsl_user_op
def scale_of(m, *, loc=None, ip=None):
    """m: bf16x2 word with the NaN-carrying |max| in either half (sign bits garbage).

    Returns (scale byte, recip f32 bits), bit-identical to the reference."""
    r = _asmN(
        2, T.i32(), [Uint32(m)],
        "{ .reg .b32 a, h, t, b; .reg .f32 f, s; .reg .b16 c; .reg .pred p, q;\n"
        "prmt.b32 h, $2, $2, 0x1032;\n"
        "max.NaN.xorsign.abs.bf16x2 a, $2, h;\n"
        "and.b32 a, a, 0x7FFF;\n"
        "setp.gt.u32 q, a, 0x7F80;\n"
        "selp.b32 a, 0x7F80, a, q;\n"
        "shl.b32 a, a, 16;\n"
        "mov.b32 f, a;\n"
        "mul.rn.f32 s, f, 0f3B124925;\n"
        "mov.f32 f, 0f00000000;\n"
        "cvt.rp.ue8m0x2.f32 c, f, s;\n"
        "cvt.u32.u16 b, c;\n"
        "and.b32 $0, b, 255;\n"
        "shl.b32 t, b, 23;\n"
        "and.b32 t, t, 0x7F800000;\n"
        "sub.s32 t, 0x7F000000, t;\n"
        "setp.ne.f32 p, s, 0f7F800000;\n"
        "selp.b32 $1, t, 0x7FC00000, p; }",
        "=r,=r,r", loc, ip,
    )
    return Uint32(r[0]), Uint32(r[1])


@dsl_user_op
def quant8(w0, w1, w2, w3, rc, *, loc=None, ip=None):
    """8 bf16 (4 words) -> 8 e4m3 bytes (2 words): cvt.rn.satfinite.e4m3x2(x *rn recip)."""
    r = _asmN(
        2, T.i32(), [Uint32(w0), Uint32(w1), Uint32(w2), Uint32(w3), Uint32(rc)],
        "{ .reg .b32 a0, b0, a1, b1, a2, b2, a3, b3; .reg .b64 x0, x1, x2, x3, s; .reg .b16 h0, h1, h2, h3;\n"
        "shl.b32 a0, $2, 16; and.b32 b0, $2, 0xFFFF0000;\n"
        "shl.b32 a1, $3, 16; and.b32 b1, $3, 0xFFFF0000;\n"
        "shl.b32 a2, $4, 16; and.b32 b2, $4, 0xFFFF0000;\n"
        "shl.b32 a3, $5, 16; and.b32 b3, $5, 0xFFFF0000;\n"
        "mov.b64 s, {$6, $6};\n"
        "mov.b64 x0, {a0, b0}; mov.b64 x1, {a1, b1}; mov.b64 x2, {a2, b2}; mov.b64 x3, {a3, b3};\n"
        "mul.rn.f32x2 x0, x0, s;\n"
        "mul.rn.f32x2 x1, x1, s;\n"
        "mul.rn.f32x2 x2, x2, s;\n"
        "mul.rn.f32x2 x3, x3, s;\n"
        "mov.b64 {a0, b0}, x0; mov.b64 {a1, b1}, x1; mov.b64 {a2, b2}, x2; mov.b64 {a3, b3}, x3;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h0, b0, a0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h1, b1, a1;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h2, b2, a2;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h3, b3, a3;\n"
        "mov.b32 $0, {h0, h1};\n"
        "mov.b32 $1, {h2, h3}; }",
        "=r,=r,r,r,r,r,r", loc, ip,
    )
    return Uint32(r[0]), Uint32(r[1])


def _tree_hmax(words):
    """Balanced max.NaN tree over one load's words (plain Python: unrolled at trace time; max is exact,
    so the order is free)."""
    t = list(words)
    while len(t) > 1:
        t = [hmax(t[k], t[k + 1]) for k in range(0, len(t), 2)]
    return t[0]


class PrepWeight:
    def __init__(self, vb: int, rpt: int):
        self.vb = vb  # columns per lane: 8 (16-byte loads) or 16 (32-byte loads)
        self.rpt = rpt  # rows per thread
        self.lpr = TC // vb  # lanes per row
        self.rpi = 32 // self.lpr  # rows per warp load instruction
        self.rpw = rpt * self.rpi  # rows per warp
        assert vb in (8, 16) and self.rpw <= 32 and 32 % self.rpw == 0, "a warp must stay inside one row block"
        self.wpb = 32 // self.rpw  # warps per 32-row block
        self.nw = TR // self.rpw
        self.lpt = 32 // vb  # lanes per tile row

    @cute.jit
    def __call__(self, x, q, s1, s0, stream):
        self.kernel(x, q, s1, s0).launch(grid=[NCTA, 1, 1], block=[self.nw * 32, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x, q, s1, s0):
        VB, RPT, LPR, RPI, RPW, WPB, LPT = self.vb, self.rpt, self.lpr, self.rpi, self.rpw, self.wpb, self.lpt
        tid, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        w = tid // 32
        l = tid % 32
        smem = cutlass.utils.SmemAllocator()
        sAll = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(32 + self.nw * 4 * 4), byte_alignment=16)
        sb = sAll.iterator.toint()  # [0, 16): FPROP pattern, [16, 32): DGRAD pattern
        sp = sb + 32  # [NW][4 tile cols] u32 partial amax

        tr = b // (C // TC)
        tc = b - tr * (C // TC)
        c = l % LPR
        sub = l // LPR
        row0 = Int64(tr * TR + w * RPW + sub)
        col0 = Int64(tc * TC + c * VB)
        src = x.iterator.toint() + (row0 * C + col0) * 2
        v = []
        for i in cutlass.range_constexpr(RPT):
            if cutlass.const_expr(VB == 16):
                v.append(ld_v8(src + i * (RPI * C * 2)))
            else:
                v.append(ld_v4(src + i * (RPI * C * 2)))
        m = _tree_hmax(v[0])
        for i in cutlass.range_constexpr(1, RPT):
            m = hmax(m, _tree_hmax(v[i]))
        for k in cutlass.range_constexpr(LPT.bit_length() - 1):
            m = hmax(m, shfl_xor(m, 1 << k))
        for k in cutlass.range_constexpr(RPI.bit_length() - 1):
            m = hmax(m, shfl_xor(m, LPR << k))
        kq = c // LPT
        rb = w // WPB
        if cutlass.const_expr(WPB > 1):
            if l < LPR:
                if l % LPT == 0:
                    sts_b32(sp + (w * 4 + kq) * 4, m)
            named_bar(1 + rb, 32 * WPB)
            for j in cutlass.range_constexpr(WPB):
                m = hmax(m, lds_b32(sp + ((rb * WPB + j) * 4 + kq) * 4))
        sbyte, rc = scale_of(m)
        dst = q.iterator.toint() + row0 * C + col0
        for i in cutlass.range_constexpr(RPT):
            y = []
            for k in cutlass.range_constexpr(0, VB // 2, 4):
                y0, y1 = quant8(v[i][k], v[i][k + 1], v[i][k + 2], v[i][k + 3], rc)
                y.append(y0)
                y.append(y1)
            if cutlass.const_expr(VB == 16):
                st_v4(dst + i * (RPI * C), y[0], y[1], y[2], y[3])
            else:
                st_v2(dst + i * (RPI * C), y[0], y[1])
        if w % WPB == 0:
            if l < LPR:
                if l % LPT == 0:
                    sts_u8(sb + rb * 4 + kq, sbyte)
                    sts_u8(sb + 16 + kq * 4 + rb, sbyte)
        cute.arch.barrier()
        if tid < 64:
            chunk = Int64(tid % 32)
            g = tr // DG_ATOMS_PER_COL
            trl = tr - g * DG_ATOMS_PER_COL
            if tid < 32:
                pat = lds_v4(sb)
                atom = Int64(tr * FP_ATOMS_PER_ROW + tc)
                st_v4(s1.iterator.toint() + atom * 512 + chunk * 16, pat[0], pat[1], pat[2], pat[3])
            else:
                pat = lds_v4(sb + 16)
                atom = Int64(g * (FP_ATOMS_PER_ROW * DG_ATOMS_PER_COL) + tc * DG_ATOMS_PER_COL + trl)
                st_v4(s0.iterator.toint() + atom * 512 + chunk * 16, pat[0], pat[1], pat[2], pat[3])


def _compile(vb, rpt):
    def fake(dtype, n, align):
        return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=align)

    return cute.compile(
        PrepWeight(vb, rpt),
        fake(cutlass.BFloat16, NUMEL, 2 * vb),
        fake(cutlass.Uint8, NUMEL, 16),
        fake(cutlass.Uint8, R * (C // 32), 16),
        fake(cutlass.Uint8, G * C * (RG // 32), 16),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


_fn = _compile(*PRIMARY)
_fn16 = _compile(*FALLBACK_16B) if FALLBACK_16B is not None else _fn
_ALIGN = 2 * PRIMARY[0]  # input alignment the primary kernel's vector loads need


def run(in0):
    if (in0.dtype != torch.bfloat16 or in0.dim() != 1 or in0.numel() != NUMEL or in0.stride(0) != 1
            or in0.data_ptr() % 16 or in0.device.type != "cuda"):
        raise ValueError(f"expected contiguous 16 B aligned cuda bf16 [{NUMEL}], got {in0.dtype} "
                         f"{tuple(in0.shape)} stride {in0.stride()} ptr % 16 = {in0.data_ptr() % 16}")
    fn = _fn if in0.data_ptr() % _ALIGN == 0 else _fn16
    dev = in0.device
    q = torch.empty((G, RG, C), dtype=torch.float8_e4m3fn, device=dev)
    s1 = torch.empty((R, C // 32), dtype=torch.float8_e8m0fnu, device=dev)
    s0 = torch.empty((G * C, RG // 32), dtype=torch.float8_e8m0fnu, device=dev)
    fn(in0, q.view(-1).view(torch.uint8), s1.view(-1).view(torch.uint8), s0.view(-1).view(torch.uint8))
    return q, s1, s0
