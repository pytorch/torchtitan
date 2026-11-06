"""GPT-OSS router backward (G-041), one CuTe DSL kernel, bitwise equal to reference.py.

reference (T = 16384 tokens, K = 4, E = 128; all fp32 except in3 int64):
    v6 = in0 * 1.0;  v7 = -v6;  v9 = (in1 / in2) / in2;  v10 = v7 * v9;  v11 = v6 / in2
    v12 = sum(v10, 1)    Reduce.cuh on [T, 4]: block (4, 128), one input per lane x, lane value
                         r_x = (((0 + v10_x) + 0) + 0) + 0 (vt0 = 4 accumulators), shfl_down 2, 1:
                         v12 = (r0 + r2) + (r1 + r3)
    v14 = v11 + v12
    v16 = scatter_add(zeros [T, E], 1, in3, v14)   deterministic -> index_put_ with two index
                         tensors (rows, in3) -> indexing_backward_kernel_stride_1: per destination
                         gradient = 0 + v_a (+ v_b ...) over its sources in ascending k (stable sort),
                         dst = 0 + gradient; no source -> +0. in3 < 0 wraps (+E); out of range
                         device-asserts (here: trap).
    v17 = in4 + v16
    out = _softmax_backward_data(v17, in5): tmp = v17 * in5 (separate aten mul), then
          softmax_warp_backward<float, 7>: lane l holds tmp[l + 32 it], it = 0..3,
          sum = (((0 + tmp_0) + tmp_1) + tmp_2) + tmp_3, butterfly sum += shfl_xor(sum, 16, 8, 4,
          2, 1), out = tmp - in5 * sum (nvcc contracts to fma(-in5, sum, tmp)).
Every fp32 op is explicit .rn PTX (div.rn.f32 = IEEE division, as nvcc's default -prec-div), FMA
only where aten's kernel has it. One warp per token row, 2 rows per warp, 4 warps per CTA.

Host path: raw data pointers and an explicit stream go to the compiled function; run() checks
dtypes, shapes, strides, devices and alignment itself. Layouts: the reference's v12 reduction order
depends on the layout of in0 / in1 / in2 (TensorIterator gives v10 their layout, and a column-major
v10 takes another Reduce.cuh configuration), and its scatter on the layout of in3 (an index
expanded along rows takes a different index_put path). So if in0, in1, in3 have strides other
than [4, 1] or in2 a row stride other than 1, _run_general replays the reference's aten ops;
otherwise (spec strides, any alignment, any in4 / in5 layout: softmax backward and the adds are
layout-independent) it runs the kernel on aligned dense copies.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NT = 16384  # tokens
K = 4
E = 128
ROWS = 2  # rows per warp
WARPS = 4  # warps per CTA


def _vals(args, loc, ip):
    return [a.ir_value(loc=loc, ip=ip) for a in args]


def _asm1(ty, args, text, cons, loc, ip, side=False):
    return llvm.inline_asm(
        ty, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


def _asmN(n, ty, args, text, cons, loc, ip, side=False):
    st = ir.Type.parse("!llvm.struct<(" + ", ".join([str(ty)] * n) + ")>")
    res = llvm.inline_asm(
        st, _vals(args, loc, ip), text, cons,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return [llvm.extractvalue(ty, res, [k], loc=loc, ip=ip) for k in range(n)]


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def ld_b32(addr, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int64(addr)], "ld.global.nc.b32 $0, [$1];", "=r,l", loc, ip, side=True))


@dsl_user_op
def st_b32(addr, v, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(v)], "st.global.b32 [$0], $1;", "l,r", loc, ip, side=True)


# Per-row scalars -> the 4 scattered values v14_k (fp32 bits).
#   in: a_k = in0[t, k], b_k = in1[t, k], d = in2[t]
_ROW = """{
.reg .f32 a<4>, b<4>, d, u<4>, w<4>, q<4>, r<4>, sa, sb, s, z;
mov.b32 a0, $4; mov.b32 a1, $5; mov.b32 a2, $6; mov.b32 a3, $7;
mov.b32 b0, $8; mov.b32 b1, $9; mov.b32 b2, $10; mov.b32 b3, $11;
mov.b32 d, $12; mov.f32 z, 0f00000000;
"""
for _k in range(4):
    _ROW += (f"mul.rn.f32 u{_k}, a{_k}, 0f3F800000;\n"
             f"div.rn.f32 b{_k}, b{_k}, d; div.rn.f32 b{_k}, b{_k}, d;\n"
             f"neg.f32 w{_k}, u{_k}; mul.rn.f32 w{_k}, w{_k}, b{_k};\n"
             f"div.rn.f32 q{_k}, u{_k}, d;\n"
             f"add.rn.f32 r{_k}, z, w{_k}; add.rn.f32 r{_k}, r{_k}, z; add.rn.f32 r{_k}, r{_k}, z;"
             f" add.rn.f32 r{_k}, r{_k}, z;\n")
_ROW += """add.rn.f32 sa, r0, r2; add.rn.f32 sb, r1, r3; add.rn.f32 s, sa, sb;
add.rn.f32 q0, q0, s; add.rn.f32 q1, q1, s; add.rn.f32 q2, q2, s; add.rn.f32 q3, q3, s;
mov.b32 $0, q0; mov.b32 $1, q1; mov.b32 $2, q2; mov.b32 $3, q3;
}"""


@dsl_user_op
def row_vals(a, b, d, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Uint32(x) for x in a] + [Uint32(x) for x in b] + [Uint32(d)], _ROW,
              "=r,=r,=r,=r" + ",r" * 9, loc, ip)
    return [Uint32(v) for v in r]


@dsl_user_op
def wrap_check(lo, hi, *, loc=None, ip=None):
    """int64 index (lo, hi words) in [-E, E) -> wrapped Int32 in [0, E); otherwise trap."""
    return Int32(_asm1(T.i32(), [Uint32(lo), Uint32(hi)],
                       "{ .reg .b64 x, y; .reg .pred p, q; mov.b64 x, {$1, $2};\n"
                       "setp.lt.s64 p, x, -%d; setp.ge.s64 q, x, %d; or.pred p, p, q; @p trap;\n"
                       "add.s64 y, x, %d; setp.lt.s64 p, x, 0; selp.b64 x, y, x, p; cvt.u32.u64 $0, x; }"
                       % (E, E, E),
                       "=r,r,r", loc, ip, side=True))


# One element: g = in4 + (0 + acc), acc = 0 (+ v_k for each k with idx_k == e, ascending);
# tmp = g * y.
_ELEM = """{
.reg .f32 acc, t, v, z, x, y; .reg .pred p;
mov.f32 z, 0f00000000; mov.f32 acc, z;
"""
for _k in range(4):
    _ELEM += (f"setp.eq.s32 p, ${_k + 5}, $1; mov.b32 v, ${_k + 9}; add.rn.f32 t, acc, v; @p mov.f32 acc, t;\n")
_ELEM += """add.rn.f32 acc, z, acc;
mov.b32 x, $2; mov.b32 y, $3; add.rn.f32 x, x, acc; mul.rn.f32 x, x, y; mov.b32 $0, x;
}"""


@dsl_user_op
def elem_tmp(e, g, y, idx, v, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int32(e), Uint32(g), Uint32(y), Uint32(0)] + [Int32(x) for x in idx]
                        + [Uint32(x) for x in v], _ELEM, "=r,r,r,r,r" + ",r" * 8, loc, ip))


@dsl_user_op
def fadd(a, b, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)],
                        "{ .reg .f32 x, y; mov.b32 x, $1; mov.b32 y, $2; add.rn.f32 x, x, y; mov.b32 $0, x; }",
                        "=r,r,r", loc, ip))


@dsl_user_op
def out_val(tmp, y, s, *, loc=None, ip=None):
    """tmp - y * s as fma.rn(-y, s, tmp)."""
    return Uint32(_asm1(T.i32(), [Uint32(tmp), Uint32(y), Uint32(s)],
                        "{ .reg .f32 t, y, s; mov.b32 t, $1; mov.b32 y, $2; mov.b32 s, $3;\n"
                        "neg.f32 y, y; fma.rn.f32 t, y, s, t; mov.b32 $0, t; }",
                        "=r,r,r,r", loc, ip))


class RouterBwd:
    @cute.jit
    def __call__(self, p0: Int64, p1: Int64, p2: Int64, p3: Int64, p4: Int64, p5: Int64, pout: Int64, stream):
        self.kernel(p0, p1, p2, p3, p4, p5, pout).launch(
            grid=[NT // (ROWS * WARPS), 1, 1], block=[32 * WARPS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, p0: Int64, p1: Int64, p2: Int64, p3: Int64, p4: Int64, p5: Int64, pout: Int64):
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        t0 = (bid * WARPS + warp) * ROWS
        # all loads first
        A = []
        B = []
        Dv = []
        I = []
        G = []
        Y = []
        for r in cutlass.range_constexpr(ROWS):
            t = Int64(t0 + r)
            A.append(ld_v4(p0 + t * 16))
            B.append(ld_v4(p1 + t * 16))
            Dv.append(ld_b32(p2 + t * 4))
            I.append(ld_v4(p3 + t * 32) + ld_v4(p3 + t * 32 + 16))
            G.append([ld_b32(p4 + (t * E + lane + 32 * it) * 4) for it in range(4)])
            Y.append([ld_b32(p5 + (t * E + lane + 32 * it) * 4) for it in range(4)])
        for r in cutlass.range_constexpr(ROWS):
            t = Int64(t0 + r)
            v = row_vals(A[r], B[r], Dv[r])
            idx = [wrap_check(I[r][2 * k], I[r][2 * k + 1]) for k in range(4)]
            tmp = [elem_tmp(lane + 32 * it, G[r][it], Y[r][it], idx, v) for it in range(4)]
            s = Uint32(0)
            for it in cutlass.range_constexpr(4):
                s = fadd(s, tmp[it])
            for off in (16, 8, 4, 2, 1):
                s = fadd(s, Uint32(cute.arch.shuffle_sync_bfly(s, offset=off)))
            for it in cutlass.range_constexpr(4):
                st_b32(pout + (t * E + lane + 32 * it) * 4, out_val(tmp[it], Y[r][it], s))


_fn = cute.compile(RouterBwd(), *[Int64(0)] * 7, cute.runtime.make_fake_stream(), options="--enable-tvm-ffi")

_F32 = torch.float32
_I64 = torch.int64
_raw_stream = torch._C._cuda_getCurrentRawStream
_cur_dev = torch.cuda.current_device
_SPEC = ((NT, K), (NT, K), (NT, 1), (NT, K), (NT, E), (NT, E))
_DT = (_F32, _F32, _F32, _I64, _F32, _F32)


def _reference_ops(in0, in1, in2, in3, in4, in5):
    """reference.py's aten op sequence (for layouts whose reduction order the kernel does not model)."""
    aten = torch.ops.aten
    v6 = aten.mul.Tensor(in0, 1.0)
    v7 = aten.neg.default(v6)
    v9 = aten.div.Tensor(aten.div.Tensor(in1, in2), in2)
    v10 = aten.mul.Tensor(v7, v9)
    v11 = aten.div.Tensor(v6, in2)
    v12 = aten.sum.dim_IntList(v10, [1], True)
    v14 = aten.add.Tensor(v11, aten.expand.default(v12, [NT, K]))
    v15 = aten.new_zeros.default(v14, [NT, E])
    v16 = aten.scatter_add.default(v15, -1, in3, v14)
    v17 = aten.add.Tensor(in4, v16)
    return aten._softmax_backward_data.default(v17, aten.alias.default(in5), -1, torch.float32)


def _run_general(ins, out):
    dev = out.device
    for x, shape, dt in zip(ins, _SPEC, _DT):
        if x.dtype != dt or tuple(x.shape) != shape or x.device != dev:
            raise ValueError("G-041: inputs do not match the spec (dtypes, shapes, devices)")
    in0, in1, in2, in3 = ins[:4]
    if in0.stride() != (K, 1) or in1.stride() != (K, 1) or in3.stride() != (K, 1) or in2.stride(0) != 1:
        with torch.cuda.device(dev):
            return _reference_ops(*ins)
    ins = [x if x.is_contiguous() and x.data_ptr() % 16 == 0 else x.clone(memory_format=torch.contiguous_format)
           for x in ins]
    with torch.cuda.device(dev):
        _fn(*[x.data_ptr() for x in ins], out.data_ptr(), _raw_stream(dev.index))
    return out


def run(in0, in1, in2, in3, in4, in5):
    dev = in4.device
    out = torch.empty((NT, E), dtype=_F32, device=dev)
    p = (in0.data_ptr(), in1.data_ptr(), in2.data_ptr(), in3.data_ptr(), in4.data_ptr(), in5.data_ptr())
    di = dev.index
    if (not ((p[0] | p[1] | p[3] | p[4] | p[5]) & 15) and not (p[2] & 3)
            and in0.dtype is _F32 and in1.dtype is _F32 and in2.dtype is _F32 and in3.dtype is _I64
            and in4.dtype is _F32 and in5.dtype is _F32
            and in0.shape == _SPEC[0] and in1.shape == _SPEC[1] and in2.shape == _SPEC[2] and in3.shape == _SPEC[3]
            and in4.shape == _SPEC[4] and in5.shape == _SPEC[5]
            and in0.is_contiguous() and in1.is_contiguous() and in2.is_contiguous() and in3.is_contiguous()
            and in4.is_contiguous() and in5.is_contiguous()
            and in0.device == dev and in1.device == dev and in2.device == dev and in3.device == dev
            and in5.device == dev and di == _cur_dev()):
        _fn(*p, out.data_ptr(), _raw_stream(di))
        return out
    return _run_general((in0, in1, in2, in3, in4, in5), out)


def _warm():
    """Load the module and run it once on the current device."""
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda", _cur_dev())
    ins = [torch.zeros(s, dtype=dt, device=dev) for s, dt in zip(_SPEC, _DT)]
    run(*ins)
    torch.cuda.synchronize(dev)


_warm()
