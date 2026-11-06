"""GPT-OSS w2 expert-bias expand + add (G-039), one CuTe DSL kernel, bitwise equal to reference.py.

reference (R routed rows, E = 8 local experts, O = 2880):
    in0 = offs [E] int32 (cumulative rows per expert), in1 = R, in2 = bias [E, O] bf16,
    in3 = x [R, O] bf16; outputs (index, out [R, O])
    counts = [offs[0], offs[1] - offs[0], ..., offs[7] - offs[6], R - offs[7]]   (int32 -> int64)
    index = repeat_interleave(counts, output_size=R)      [R] int64 (output 0). The reference
            device-asserts unless every count >= 0, i.e. 0 <= offs[0] <= ... <= offs[7] <= R;
            for those inputs index[r] = #{k : offs[k] <= r} (8 = the zero-bias tail padding)
    out[r] = bf16(x[r] + cat(bias, zeros)[index[r]])      (output 1)
The bf16 add is add.rn.bf16x2 (== torch's bf16 add: fp32 add, RNE to bf16, NaN -> 0x7fff, on all
2^32 pairs, sol's exhaustive test); tail rows add +0, as the reference adds its zeros row.
One flat grid over the R * 360 16-byte vectors (256 threads x 4 vectors); each row's first vector
also stores index[r]. No host sync: R is the Python int argument.

Host path: raw data pointers and an explicit stream go to the compiled function (TVM-FFI tensor
arguments cost ~1.5-3 us each); run() checks dtypes, shapes, strides, devices and alignment
itself. Anything else (misaligned or strided views, other current device, R too large for Int32
vector indices) takes _run_general: dense copies, the right device, row chunks. The kernel is
loaded and run once at import so the first training call does not block on module loading.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

E = 8
O = 2880
NV = O // 8  # 16-byte vectors per row
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
def ld_s32(addr, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int64(addr)], "ld.global.nc.u32 $0, [$1];", "=r,l", loc, ip, side=True))


@dsl_user_op
def ld_v4(addr, *, loc=None, ip=None):
    r = _asmN(4, T.i32(), [Int64(addr)], "ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def ld_bias_v4(addr, e, *, loc=None, ip=None):
    """16 B of bias row e, or +0 for the tail row e == E (no load)."""
    r = _asmN(4, T.i32(), [Int64(addr), Int32(e)],
              "{ .reg .pred p; setp.lt.s32 p, $5, %d;\n"
              "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;\n"
              "@p ld.global.nc.v4.b32 {$0, $1, $2, $3}, [$4]; }" % E,
              "=r,=r,=r,=r,l,r", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_v4(addr, w, *, loc=None, ip=None):
    _asm1(None, [Int64(addr)] + [Uint32(x) for x in w],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def st_i64(addr, v, *, loc=None, ip=None):
    """int64 store of a non-negative Int32."""
    _asm1(None, [Int64(addr), Int32(v)], "st.global.v2.b32 [$0], {$1, 0};", "l,r", loc, ip, side=True)


@dsl_user_op
def badd(a, b, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "add.rn.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


@dsl_user_op
def expert_of(row, o, *, loc=None, ip=None):
    """#{k : o[k] <= row}."""
    text = "{ .reg .pred p; .reg .b32 c;\nmov.b32 $0, 0;\n"
    for k in range(E):
        text += "setp.le.s32 p, $%d, $1; selp.b32 c, 1, 0, p; add.s32 $0, $0, c;\n" % (k + 2)
    text += "}"
    return Int32(_asm1(T.i32(), [Int32(row)] + [Int32(x) for x in o], text, "=r,r" + ",r" * E, loc, ip))


class BiasExpand:
    @cute.jit
    def __call__(self, pb: Int64, px: Int64, po: Int64, pidx: Int64, pout: Int64, row0: Int32, nrows: Int32,
                 stream):
        nvec = nrows * NV
        nblk = (nvec + THREADS * U - 1) // (THREADS * U)
        self.kernel(pb, px, po, pidx, pout, row0, nvec).launch(
            grid=[nblk, 1, 1], block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, pb: Int64, px: Int64, po: Int64, pidx: Int64, pout: Int64, row0: Int32, nvec: Int32):
        """pb: bias [E, O] bf16, px: x [R, O] bf16, po: offsets [E] int32, pidx: index [R] int64,
        pout: out [R, O] bf16 (all dense, 16-byte aligned except po: 4); rows row0 .. row0 + nvec / NV."""
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        o = [ld_s32(po + 4 * k) for k in range(E)]
        r0 = Int64(row0)
        px = px + r0 * (O * 2)
        pout = pout + r0 * (O * 2)
        pidx = pidx + r0 * 8
        base = bid * (THREADS * U) + tid
        # every load of this thread before any store (side-effecting asm keeps program order)
        xs = []
        bs = []
        for u in cutlass.range_constexpr(U):
            i = base + u * THREADS
            xv = [Uint32(0)] * 4
            bv = [Uint32(0)] * 4
            if i < nvec:
                row = i // NV
                c = i - row * NV
                e = expert_of(row + row0, o)
                xv = ld_v4(px + Int64(i) * 16)
                bv = ld_bias_v4(pb + Int64(e) * (O * 2) + Int64(c) * 16, e)
            xs.append(xv)
            bs.append(bv)
        for u in cutlass.range_constexpr(U):
            i = base + u * THREADS
            if i < nvec:
                st_v4(pout + Int64(i) * 16, [badd(xs[u][j], bs[u][j]) for j in range(4)])
                row = i // NV
                if i - row * NV == 0:
                    st_i64(pidx + Int64(row) * 8, expert_of(row + row0, o))


_fn = cute.compile(BiasExpand(), *[Int64(0)] * 5, Int32(0), Int32(0), cute.runtime.make_fake_stream(),
                   options="--enable-tvm-ffi")

_BF = torch.bfloat16
_I32 = torch.int32
_I64 = torch.int64
_BIAS_SHAPE = (E, O)
_BIAS_STRIDE = (O, 1)
_raw_stream = torch._C._cuda_getCurrentRawStream
_cur_dev = torch.cuda.current_device


def _dense(t):
    return t if t.is_contiguous() and t.data_ptr() % 16 == 0 else t.clone(memory_format=torch.contiguous_format)


def _run_general(bias, x, rows, offs, idx, out):
    dev = x.device
    if not (isinstance(rows, int) and dev.type == "cuda" and bias.device == dev and offs.device == dev
            and bias.dtype == _BF and x.dtype == _BF and offs.dtype == _I32 and tuple(bias.shape) == _BIAS_SHAPE
            and tuple(x.shape) == (rows, O) and tuple(offs.shape) == (E,)):
        raise ValueError("G-039: inputs do not match the spec (dtypes, shapes, devices)")
    if rows == 0:
        return
    bias, x = _dense(bias), _dense(x)
    offs = offs.contiguous()
    with torch.cuda.device(dev):
        s = _raw_stream(dev.index)
        for r0 in range(0, rows, CHUNK):
            _fn(bias.data_ptr(), x.data_ptr(), offs.data_ptr(), idx.data_ptr(), out.data_ptr(), r0,
                min(CHUNK, rows - r0), s)


def run(in0, in1, in2, in3):
    rows = in1
    dev = in3.device
    idx = torch.empty((rows,), dtype=_I64, device=dev)
    out = torch.empty((rows, O), dtype=_BF, device=dev)
    pb = in2.data_ptr()
    px = in3.data_ptr()
    po = in0.data_ptr()
    di = dev.index
    if (0 < rows <= CHUNK and not ((pb | px) & 15) and not (po & 3)
            and in2.dtype is _BF and in3.dtype is _BF and in0.dtype is _I32
            and in2.shape == _BIAS_SHAPE and in2.stride() == _BIAS_STRIDE
            and in3.shape == (rows, O) and in3.stride() == (O, 1)
            and in0.shape == (E,) and in0.stride() == (1,)
            and in2.device == dev and in0.device == dev and di == _cur_dev()):
        _fn(pb, px, po, idx.data_ptr(), out.data_ptr(), 0, rows, _raw_stream(di))
    else:
        _run_general(in2, in3, rows, in0, idx, out)
    return idx, out


def _warm():
    """Load the module and run it once on the current device (one row)."""
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda", _cur_dev())
    bias = torch.zeros(_BIAS_SHAPE, dtype=_BF, device=dev)
    x = torch.zeros((1, O), dtype=_BF, device=dev)
    offs = torch.zeros((E,), dtype=_I32, device=dev)
    idx = torch.empty((1,), dtype=_I64, device=dev)
    out = torch.empty((1, O), dtype=_BF, device=dev)
    _fn(bias.data_ptr(), x.data_ptr(), offs.data_ptr(), idx.data_ptr(), out.data_ptr(), 0, 1, _raw_stream(dev.index))
    torch.cuda.synchronize(dev)


_warm()
