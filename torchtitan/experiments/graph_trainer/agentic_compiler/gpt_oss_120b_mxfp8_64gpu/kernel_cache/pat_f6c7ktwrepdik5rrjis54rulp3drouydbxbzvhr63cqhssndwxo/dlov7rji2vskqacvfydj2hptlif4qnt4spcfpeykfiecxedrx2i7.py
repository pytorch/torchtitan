"""GPT-OSS routed-experts mul.Scalar + mul (G-048), one CuTe DSL kernel, bitwise equal to reference.py.

reference (in0, in1 = [R, 2880] bf16, contiguous):
    v2 = bf16(in0 * 1)      aten mul with a wrapped scalar: fp32 opmath, RNE to bf16
    v3 = bf16(in1 * v2)
Both are mul.rn.bf16x2 (== torch's bf16 mul: fp32 product, RNE to bf16, NaN -> 0x7fff, denormals
kept, on all 2^32 pairs, sol's exhaustive test), in the reference's operand order.
One flat grid over the R * 360 16-byte vectors (256 threads x 4 vectors).

Host path: raw data pointers and an explicit stream go to the compiled function; run() checks
dtypes, shapes, strides, devices and alignment itself. Anything else (misaligned or strided views,
other current device, R too large for Int32 vector indices) takes _run_general: dense copies, the
right device, row chunks. The kernel is loaded and run once at import so the first training call
does not block on module loading.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

D = 2880
NV = D // 8  # 16-byte vectors per row
THREADS = 256
U = 4  # vectors per thread
CHUNK = (1 << 31) // (NV * 2)  # rows per launch: every Int32 vector index stays < 2^31


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
def st_v4(addr, w, *, loc=None, ip=None):
    _asm1(None, [Int64(addr)] + [Uint32(x) for x in w],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def mul_word(a, b, *, loc=None, ip=None):
    """bf16x2: b * (a * 1)."""
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)],
                        "{ .reg .b32 one, t; mov.b32 one, 0x3F803F80;\n"
                        "mul.rn.bf16x2 t, $1, one; mul.rn.bf16x2 $0, $2, t; }",
                        "=r,r,r", loc, ip))


class ScaledMul:
    @cute.jit
    def __call__(self, p0: Int64, p1: Int64, pout: Int64, nrows: Int32, stream):
        nvec = nrows * NV
        nblk = (nvec + THREADS * U - 1) // (THREADS * U)
        self.kernel(p0, p1, pout, nvec).launch(grid=[nblk, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, p0: Int64, p1: Int64, pout: Int64, nvec: Int32):
        """p0 = in0, p1 = in1, pout = out: dense [nvec / NV, D] bf16, 16-byte aligned."""
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        base = bid * (THREADS * U) + tid
        # every load of this thread before any store (side-effecting asm keeps program order)
        xs = []
        ys = []
        for u in cutlass.range_constexpr(U):
            i = base + u * THREADS
            xv = [Uint32(0)] * 4
            yv = [Uint32(0)] * 4
            if i < nvec:
                xv = ld_v4(p0 + Int64(i) * 16)
                yv = ld_v4(p1 + Int64(i) * 16)
            xs.append(xv)
            ys.append(yv)
        for u in cutlass.range_constexpr(U):
            i = base + u * THREADS
            if i < nvec:
                st_v4(pout + Int64(i) * 16, [mul_word(xs[u][j], ys[u][j]) for j in range(4)])


_fn = cute.compile(ScaledMul(), Int64(0), Int64(0), Int64(0), Int32(0), cute.runtime.make_fake_stream(),
                   options="--enable-tvm-ffi")

_BF = torch.bfloat16
_raw_stream = torch._C._cuda_getCurrentRawStream
_cur_dev = torch.cuda.current_device


def _dense(t):
    return t if t.is_contiguous() and t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.contiguous_format)


def _run_general(in0, in1, out):
    dev = in0.device
    if not (dev.type == "cuda" and in1.device == dev and in0.dtype == _BF and in1.dtype == _BF
            and in0.dim() == 2 and tuple(in0.shape) == tuple(in1.shape) and in0.shape[1] == D):
        raise ValueError("G-048: inputs do not match the spec (bf16 [R, 2880] x 2 on one device)")
    rows = in0.shape[0]
    if rows == 0:
        return
    in0, in1 = _dense(in0), _dense(in1)
    with torch.cuda.device(dev):
        s = _raw_stream(dev.index)
        for r0 in range(0, rows, CHUNK):
            off = r0 * D * 2
            _fn(in0.data_ptr() + off, in1.data_ptr() + off, out.data_ptr() + off, min(CHUNK, rows - r0), s)


def run(in0, in1):
    dev = in0.device
    shape = in0.shape
    out = torch.empty(shape, dtype=_BF, device=dev)
    p0 = in0.data_ptr()
    p1 = in1.data_ptr()
    di = dev.index
    if (len(shape) == 2 and 0 < shape[0] <= CHUNK and shape[1] == D and not ((p0 | p1) & 15)
            and in0.dtype is _BF and in1.dtype is _BF and in0.stride() == (D, 1) and in1.stride() == (D, 1)
            and in1.shape == shape and in1.device == dev and di == _cur_dev()):
        _fn(p0, p1, out.data_ptr(), shape[0], _raw_stream(di))
    else:
        _run_general(in0, in1, out)
    return out


def _warm():
    """Load the module and run it once on the current device (one row)."""
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda", _cur_dev())
    x = torch.zeros((1, D), dtype=_BF, device=dev)
    out = torch.empty((1, D), dtype=_BF, device=dev)
    _fn(x.data_ptr(), x.data_ptr(), out.data_ptr(), 1, _raw_stream(dev.index))
    torch.cuda.synchronize(dev)


_warm()
