"""GPT-OSS attention-sink backward (G-028), two CuTe DSL kernels, bitwise equal to reference.py.

reference (T = 16384 tokens, H = 64 heads, D = 64; in0 = do [T, H*D] bf16, in1 = o [T, H, D] bf16,
in2 = s [T, H, 1] bf16 (stride [1, T, 1]), in3 = sig [T, H] fp32 (stride [1, T])):
    v6  = bf16(do * s)                                   output 0, [T, H, D] bf16
    v5  = bf16(do * o)
    v7  = sum(v5, -1) -> bf16      aten Reduce.cuh: block (32, 16), one warp per (t, h); lane l
                                   accumulates (0 + x_l) + (0 + x_{l+32}) (+ two empty
                                   accumulators), then shfl_down 16, 8, 4, 2, 1
    v12 = -sigmoid_backward(fp32(v7), sig) = -((v9 * (1 - sig)) * sig)       (fp32, no FMA)
    v15 = bf16(sum(v12, 0))        aten Reduce.cuh on [T, H] fp32: output_vec 4, block (16, 8),
                                   128 CTAs (global reduce), see kernel 2
Kernel 1 (NT / TT CTAs x 256 threads, TT tokens per CTA): v6, v5, v7 (same add tree via lane
shuffles: 8 lanes per (t, h), lane c holds dims 8c..8c+7), v12 -> scratch [16][T][4] fp32.
mul.rn.bf16x2 equals torch's bf16 mul (fp32 mul + RNE) on all 2^32 bf16 pairs. (0 + a) + (0 + b)
then + 0 + 0 equals (a + b) + 0 for every a, b (only -0 handling differs, and it does not).
Kernel 2 (16 CTAs x 1024 threads, one 4-column output vector per CTA) replays Reduce.cuh's column
reduction DAG exactly: row r = y + 8c + 1024 i + 4096 k, per thread (y, c)
    vl_i = (((0 + R[r_i]) + R[r_i + 4096]) + R[r_i + 8192]) + R[r_i + 12288], r_i = y + 8c + 1024 i
    v = ((vl_0 + vl_1) + vl_2) + vl_3;  P[c] = block_y tree over y (offsets 4, 2, 1)
    Q_y = (((0 + P[y]) + P[y + 8]) + ...) + P[y + 120];  out = block_y tree over Q (4, 2, 1)
All fp32 arithmetic is explicit .rn PTX; bf16 rounding is cvt.rn (what __float2bfloat16 emits).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NT = 16384  # tokens
H = 64
D = 64
TT = 32  # tokens per CTA (kernel 1)
NB = 8  # rows per load batch per 8-lane group
THREADS = 256


def _vals(args, loc, ip):
    return [a.ir_value(loc=loc, ip=ip) for a in args]


def _asm1(ty, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ty, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


def _asmN(tys, args, text, cons, loc, ip, side=False):
    st = ir.Type.parse("!llvm.struct<(" + ", ".join(str(t) for t in tys) + ")>")
    res = llvm.inline_asm(
        st, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return [llvm.extractvalue(t, res, [k], loc=loc, ip=ip) for k, t in enumerate(tys)]


@dsl_user_op
def ld_stream_v4(addr, *, loc=None, ip=None):
    r = _asmN([T.i32()] * 4, [Int64(addr)], "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    r = _asmN([T.i32()] * 4, [Int64(addr)], "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def ld_v4_cg(addr, *, loc=None, ip=None):
    """L2-coherent load (data written by the previous kernel)."""
    r = _asmN([T.i32()] * 4, [Int64(addr)], "ld.global.cg.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_v4(addr, w, *, loc=None, ip=None):
    _asm1(None, [Int64(addr)] + [Uint32(x) for x in w],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def st_v4_na(addr, w, *, loc=None, ip=None):
    _asm1(None, [Int64(addr)] + [Uint32(x) for x in w],
          "st.global.L1::no_allocate.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def st_b64(addr, a, b, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(a), Uint32(b)], "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", loc, ip, side=True)


@dsl_user_op
def sts_v4(addr, w, *, loc=None, ip=None):
    _asm1(None, [Int32(addr)] + [Uint32(x) for x in w],
          "st.shared.v4.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def sts_b32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Uint32(a)], "st.shared.b32 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def lds_b32(addr, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int32(addr)], "ld.shared.b32 $0, [$1];", "=r,r", loc, ip, side=True))


@dsl_user_op
def lds_u16x2(addr, *, loc=None, ip=None):
    """bf16 at addr replicated into both halves of a word."""
    return Uint32(_asm1(T.i32(), [Int32(addr)],
                        "{ .reg .b16 s; ld.shared.b16 s, [$1]; mov.b32 $0, {s, s}; }", "=r,r", loc, ip, side=True))


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    r = _asmN([T.i32()] * 4, [Int32(addr)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", loc, ip,
              side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def bmul(a, b, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "mul.rn.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


# level 1 of the sum tree on one bf16x2 word pair: own (p) and partner (q) products ->
# a = (p + q) + 0 for both halves (fp32 pair as two words lo, hi)
@dsl_user_op
def tree_l1(p, q, *, loc=None, ip=None):
    r = _asmN([T.i32()] * 2, [Uint32(p), Uint32(q)],
              "{ .reg .b32 a, b, c, d; .reg .b64 x, y, z;\n"
              "shl.b32 a, $2, 16; and.b32 b, $2, 0xFFFF0000; mov.b64 x, {a, b};\n"
              "shl.b32 c, $3, 16; and.b32 d, $3, 0xFFFF0000; mov.b64 y, {c, d};\n"
              "mov.b64 z, 0;\n"
              "add.rn.f32x2 x, x, y; add.rn.f32x2 x, x, z;\n"
              "mov.b64 {$0, $1}, x; }",
              "=r,=r,r,r", loc, ip)
    return Uint32(r[0]), Uint32(r[1])


@dsl_user_op
def add2(a0, a1, b0, b1, *, loc=None, ip=None):
    r = _asmN([T.i32()] * 2, [Uint32(a0), Uint32(a1), Uint32(b0), Uint32(b1)],
              "{ .reg .b64 x, y; mov.b64 x, {$2, $3}; mov.b64 y, {$4, $5}; add.rn.f32x2 x, x, y; mov.b64 {$0, $1}, x; }",
              "=r,=r,r,r,r,r", loc, ip)
    return Uint32(r[0]), Uint32(r[1])


@dsl_user_op
def add1(a, b, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "add.rn.f32 $0, $1, $2;", "=r,r,r", loc, ip))


# f (fp32 bits), sig (fp32 bits) -> v12 = -((fp32(bf16(f)) * (1 - sig)) * sig)
@dsl_user_op
def sink_grad(f, sg, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(f), Uint32(sg)],
                        "{ .reg .b16 h; .reg .b32 v, t;\n"
                        "cvt.rn.bf16.f32 h, $1;\n"
                        "cvt.u32.u16 v, h; shl.b32 v, v, 16;\n"
                        "sub.rn.f32 t, 0f3F800000, $2;\n"
                        "mul.rn.f32 t, v, t;\n"
                        "mul.rn.f32 t, t, $2;\n"
                        "neg.f32 $0, t; }",
                        "=r,r,r", loc, ip))


@dsl_user_op
def cvt_bf16x2(lo, hi, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(lo), Uint32(hi)], "cvt.rn.bf16x2.f32 $0, $2, $1;", "=r,r,r", loc, ip))


def _shfl_xor(v, m):
    return Uint32(cute.arch.shuffle_sync_bfly(v, offset=m))


def _shfl_down(v, m):
    return Uint32(cute.arch.shuffle_sync_down(v, offset=m))


class SinkBwdMain:
    @cute.jit
    def __call__(self, x0, x1, sc, sg, v6, scr, stream):
        self.kernel(x0, x1, sc, sg, v6, scr).launch(grid=[NT // TT, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, x0, x1, sc, sg, v6, scr):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        t0 = bid * TT
        smem = cutlass.utils.SmemAllocator()
        sS = smem.allocate_tensor(cutlass.BFloat16, cute.make_layout(H * TT), byte_alignment=16)
        sB = smem.allocate_tensor(cutlass.Float32, cute.make_layout(H * TT), byte_alignment=16)
        sV = smem.allocate_tensor(cutlass.Float32, cute.make_layout(H * TT), byte_alignment=16)
        aS = sS.iterator.toint()
        aB = sB.iterator.toint()
        aV = sV.iterator.toint()
        # ---- per-CTA tiles of s (bf16) and sig (fp32): [h][t_local] in smem
        for r in cutlass.range_constexpr((H * TT // 4 + THREADS - 1) // THREADS):
            idx = tid + r * THREADS
            if idx < H * TT // 4:
                hB = idx // (TT // 4)
                pB = idx % (TT // 4)
                wB = ld_v4(sg.iterator.toint() + (Int64(hB) * NT + Int64(t0 + 4 * pB)) * 4)
                sts_v4(aB + (hB * TT + 4 * pB) * 4, wB)
        for r in cutlass.range_constexpr((H * TT // 8 + THREADS - 1) // THREADS):
            idx = tid + r * THREADS
            if idx < H * TT // 8:
                hS = idx // (TT // 8)
                pS = idx % (TT // 8)
                wS = ld_v4(sc.iterator.toint() + (Int64(hS) * NT + Int64(t0 + 8 * pS)) * 2)
                sts_v4(aS + (hS * TT + 8 * pS) * 2, wS)
        cute.arch.barrier()

        g = tid // 8
        c = tid % 8
        tile = Int64(t0) * (H * 128) + Int64(c * 16)  # byte offset of (t0, h=0, chunk c)
        p0 = x0.iterator.toint() + tile
        p1 = x1.iterator.toint() + tile
        p6 = v6.iterator.toint() + tile
        for jb in cutlass.range_constexpr(0, 2 * TT, NB):
            xs = []
            ys = []
            for u in cutlass.range_constexpr(NB):
                roff = Int64((32 * (jb + u) + g) * 128)
                xs.append(ld_stream_v4(p0 + roff))
                ys.append(ld_stream_v4(p1 + roff))
            for u in cutlass.range_constexpr(NB):
                j = jb + u
                tl = j // 2
                h = 32 * (j % 2) + g
                roff = Int64((32 * j + g) * 128)
                ss = lds_u16x2(aS + (h * TT + tl) * 2)
                st_v4_na(p6 + roff, [bmul(xs[u][w], ss) for w in range(4)])
                pw = [bmul(xs[u][w], ys[u][w]) for w in range(4)]
                # level 1: lanes c and c ^ 4 hold dims l and l + 32
                a = []
                for w in cutlass.range_constexpr(4):
                    lo, hi = tree_l1(pw[w], _shfl_xor(pw[w], 4))
                    a.append(lo)
                    a.append(hi)
                # level 2: + dims l + 16 (lane c ^ 2); level 3: + dims l + 8 (lane c ^ 1)
                for m in (2, 1):
                    q = [_shfl_xor(a[e], m) for e in range(8)]
                    b = []
                    for w in cutlass.range_constexpr(4):
                        lo, hi = add2(a[2 * w], a[2 * w + 1], q[2 * w], q[2 * w + 1])
                        b.append(lo)
                        b.append(hi)
                    a = b
                # levels 4..6 in registers: + dims l + 4, l + 2, l + 1
                d01 = add2(a[0], a[1], a[4], a[5])
                d23 = add2(a[2], a[3], a[6], a[7])
                e01 = add2(d01[0], d01[1], d23[0], d23[1])
                f = add1(e01[0], e01[1])
                v12 = sink_grad(f, lds_b32(aB + (h * TT + tl) * 4))
                if c == 0:
                    sts_b32(aV + (h * TT + tl) * 4, v12)
        cute.arch.barrier()
        # ---- v12 tile -> scratch [H / 4][NT][4]
        for r in cutlass.range_constexpr((16 * TT + THREADS - 1) // THREADS):
            idx = tid + r * THREADS
            if idx < 16 * TT:
                cg = idx // TT
                tl2 = idx % TT
                w4 = [lds_b32(aV + ((4 * cg + q) * TT + tl2) * 4) for q in range(4)]
                st_v4(scr.iterator.toint() + (Int64(cg) * NT + Int64(t0 + tl2)) * 16, w4)


class SinkBwdColSum:
    @cute.jit
    def __call__(self, scr, out, stream):
        self.kernel(scr, out).launch(grid=[H // 4, 1, 1], block=[1024, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, scr, out):
        tid, _, _ = cute.arch.thread_idx()
        cg, _, _ = cute.arch.block_idx()
        smem = cutlass.utils.SmemAllocator()
        sP = smem.allocate_tensor(cutlass.Float32, cute.make_layout(128 * 4), byte_alignment=16)
        aP = sP.iterator.toint()
        c = tid // 8
        y = tid % 8
        base = scr.iterator.toint() + Int64(cg) * (NT * 16)
        z = Uint32(0)
        vl = []
        for i in cutlass.range_constexpr(4):
            acc = [z, z, z, z]
            for k in cutlass.range_constexpr(4):
                rv = ld_v4_cg(base + Int64(y + 8 * c + 1024 * i + 4096 * k) * 16)
                acc = [add1(acc[e], rv[e]) for e in range(4)]
            vl.append(acc)
        v = [add1(add1(add1(vl[0][e], vl[1][e]), vl[2][e]), vl[3][e]) for e in range(4)]
        for off in (4, 2, 1):
            v = [add1(v[e], _shfl_down(v[e], off)) for e in range(4)]
        if y == 0:
            sts_v4(aP + c * 16, v)
        cute.arch.barrier()
        if tid < 32:
            q = [z, z, z, z]
            for k in cutlass.range_constexpr(16):
                pv = lds_v4(aP + ((tid % 8) + 8 * k) * 16)
                q = [add1(q[e], pv[e]) for e in range(4)]
            for off in (4, 2, 1):
                q = [add1(q[e], _shfl_down(q[e], off)) for e in range(4)]
            if tid == 0:
                st_b64(out.iterator.toint() + Int64(cg) * 8, cvt_bf16x2(q[0], q[1]), cvt_bf16x2(q[2], q[3]))


_bf = cutlass.BFloat16
_f32 = cutlass.Float32
_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
_main = cute.compile(
    SinkBwdMain(),
    cute.runtime.make_fake_tensor(_bf, (NT, H * D), stride=(H * D, 1), assumed_align=16),
    cute.runtime.make_fake_tensor(_bf, (NT, H, D), stride=(H * D, D, 1), assumed_align=16),
    cute.runtime.make_fake_tensor(_bf, (NT, H, 1), stride=(1, NT, 1), assumed_align=16),
    cute.runtime.make_fake_tensor(_f32, (NT, H), stride=(1, NT), assumed_align=16),
    cute.runtime.make_fake_tensor(_bf, (NT, H, D), stride=(H * D, D, 1), assumed_align=16),
    cute.runtime.make_fake_tensor(_f32, (H // 4, NT, 4), stride=(NT * 4, 4, 1), assumed_align=16),
    _stream,
    options="--enable-tvm-ffi",
)
_colsum = cute.compile(
    SinkBwdColSum(),
    cute.runtime.make_fake_tensor(_f32, (H // 4, NT, 4), stride=(NT * 4, 4, 1), assumed_align=16),
    cute.runtime.make_fake_tensor(_bf, (H,), stride=(1,), assumed_align=16),
    _stream,
    options="--enable-tvm-ffi",
)


def _relayout(t, shape, stride):
    out = torch.empty_strided(shape, stride, dtype=t.dtype, device=t.device)
    out.copy_(t)
    return out


def run(in0, in1, in2, in3):
    dev = in0.device
    v6 = torch.empty((NT, H, D), dtype=torch.bfloat16, device=dev)
    v15 = torch.empty((H,), dtype=torch.bfloat16, device=dev)
    scr = torch.empty((H // 4, NT, 4), dtype=torch.float32, device=dev)
    try:
        # The compiled entry point validates dtypes, devices, shapes, strides and alignment.
        _main(in0, in1, in2, in3, v6, scr)
    except ValueError:
        # Unexpected layouts: copies in the spec layouts.
        _main(_relayout(in0, (NT, H * D), (H * D, 1)), _relayout(in1, (NT, H, D), (H * D, D, 1)),
              _relayout(in2, (NT, H, 1), (1, NT, 1)), _relayout(in3, (NT, H), (1, NT)), v6, scr)
    _colsum(scr, v15)
    return (v6, v15)
