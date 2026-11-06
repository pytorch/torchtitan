"""GPT-OSS MoE aux-loss forward (G-046), one CuTe DSL kernel + the reference's D2H copy, bitwise
equal to reference.py.

reference (T = 16384 tokens, K = 4, E = 128; in0 [T, E] f32 only gives zeros_like its shape,
in1 [T, K] int64 expert ids, in2 [T] bool):
    v4  = zeros_like(in0, bool).scatter_(-1, in1, True)   (ids outside [0, E) device-assert)
    v7  = v4 & ~in2[:, None]
    v8  = sum(v7, 0)            int64 [E]   = #tokens t with e in in1[t] and not in2[t]
    v12 = sum(float(v7), 0)     f32 [E]     = float(v8) (integer partial sums < 2^24: exact
                                             in any order)
    v14 = sum(v4, 0).view(16, 8)  int64     = #tokens t with e in in1[t]
    v17 = v14.sum(1).to("cpu", non_blocking=True)   int64 [16], pinned (as _to_copy does:
                                             empty_strided(pin_memory=True) + copy_(non_blocking))
A token whose ids repeat counts once per distinct expert (scatter_ of True is idempotent).
Kernel: one cluster of 8 CTAs x 512 threads, 2048 tokens per CTA. Lane loads are 16 B and fully
coalesced: chunk c holds ids 2h, 2h + 1 of token c // 2 (h = c % 2), and the partner lane (c ^ 1)
holds the other half, read with one shuffle for the distinctness test. Each distinct id adds
1 + (not masked) << 16 to a packed per-expert shared-memory word (both counts <= 16384 < 2^16).
After a cluster barrier, CTA 0 sums the 8 histograms through distributed shared memory and writes
v8, v12, v14 and v17 (8-lane shuffle sums); a second cluster barrier keeps the other CTAs' shared
memory alive until then. Integer counts: no ordering question.
v17 goes to the CPU through torch's copy_ exactly as in the reference, so the pinned block
gets the caching host allocator's stream event; the copy is asynchronous (no host sync).

Host path: raw data pointers and an explicit stream go to the compiled function; run() checks
dtypes, shapes, strides, devices and alignment itself; other layouts take dense copies.
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
GROUP = 8  # experts per EP rank (v14 rows)
NCTA = 8  # CTAs in the cluster
THREADS = 512
TPC = NT // NCTA  # tokens per CTA
CPT = TPC * 2 // THREADS  # 16-byte id chunks per thread


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
def sts_b32(addr, v, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Uint32(v)], "st.shared.b32 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def lds_b32(addr, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int32(addr)], "ld.shared.b32 $0, [$1];", "=r,r", loc, ip, side=True))


@dsl_user_op
def st_i64(addr, v, *, loc=None, ip=None):
    """int64 store of a non-negative 32-bit value."""
    _asm1(None, [Int64(addr), Uint32(v)], "st.global.v2.b32 [$0], {$1, 0};", "l,r", loc, ip, side=True)


@dsl_user_op
def st_f32_of_u32(addr, v, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Uint32(v)],
          "{ .reg .f32 f; cvt.rn.f32.u32 f, $1; st.global.f32 [$0], f; }", "l,r", loc, ip, side=True)


# One 16-byte chunk: ids a, b of one token (lo/hi words $1..$4), the partner chunk's lo words
# ($5, $6: ids of the other half), half h ($7), bool byte m ($8), histogram base ($0).
# Half 0 holds ids 0, 1: count id0; id1 if != id0. Half 1 holds ids 2, 3: id2 if not in {id0, id1};
# id3 if not in {id0, id1, id2}. Trap unless 0 <= id < E (the reference's scatter_ device-asserts).
_CHUNK = """{
.reg .pred p, q, h1, ca, cb; .reg .b32 inc, x, aa, ab;
or.b32 x, $2, $4; setp.ne.b32 p, x, 0;
setp.ge.u32 q, $1, %d; or.pred p, p, q; setp.ge.u32 q, $3, %d; or.pred p, p, q;
@p trap;
setp.eq.b32 p, $8, 0; selp.b32 inc, 0x10001, 1, p;
setp.ne.b32 h1, $7, 0;
// half 0: ca = true, cb = (b != a); half 1: ca = a not in {p0, p1}, cb = b not in {p0, p1, a}
setp.ne.b32 cb, $3, $1;
setp.ne.b32 ca, $1, $5; setp.ne.and.b32 ca, $1, $6, ca; @!h1 setp.eq.b32 ca, 0, 0;
@h1 setp.ne.and.b32 cb, $3, $5, cb; @h1 setp.ne.and.b32 cb, $3, $6, cb;
shl.b32 aa, $1, 2; add.s32 aa, aa, $0;
shl.b32 ab, $3, 2; add.s32 ab, ab, $0;
@ca red.shared.add.u32 [aa], inc;
@cb red.shared.add.u32 [ab], inc;
}""" % (E, E)


@dsl_user_op
def count_chunk(hist, w, pa, pb, half, m, *, loc=None, ip=None):
    _asm1(None, [Int32(hist)] + [Uint32(x) for x in w] + [Uint32(pa), Uint32(pb), Int32(half), Uint32(m)],
          _CHUNK, "r" + ",r" * 8, loc, ip, side=True)


@dsl_user_op
def ld_u8(addr, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int64(addr)], "ld.global.nc.u8 $0, [$1];", "=r,l", loc, ip, side=True))


@dsl_user_op
def ld_dsmem(addr, rank, *, loc=None, ip=None):
    """32-bit word at shared address addr of cluster CTA rank."""
    return Uint32(_asm1(T.i32(), [Int32(addr), Int32(rank)],
                        "{ .reg .b32 ra; mapa.shared::cluster.u32 ra, $1, $2; ld.shared::cluster.u32 $0, [ra]; }",
                        "=r,r,r", loc, ip, side=True))


class AuxCounts:
    @cute.jit
    def __call__(self, p1: Int64, p2: Int64, p8: Int64, p12: Int64, p14: Int64, p17: Int64, stream):
        self.kernel(p1, p2, p8, p12, p14, p17).launch(
            grid=[NCTA, 1, 1], block=[THREADS, 1, 1], cluster=[NCTA, 1, 1], stream=stream)

    @cute.kernel
    def kernel(self, p1: Int64, p2: Int64, p8: Int64, p12: Int64, p14: Int64, p17: Int64):
        tid, _, _ = cute.arch.thread_idx()
        rank = cute.arch.block_idx_in_cluster()
        smem = cutlass.utils.SmemAllocator()
        sH = smem.allocate_tensor(cutlass.Int32, cute.make_layout(E), byte_alignment=16)
        hist = sH.iterator.toint()
        if tid < E:
            sts_b32(hist + tid * 4, Uint32(0))
        base = Int64(rank * (TPC * 2))  # first chunk of this CTA
        ws = [ld_v4(p1 + (base + j * THREADS + tid) * 16) for j in range(CPT)]
        tok = [Int64(rank * TPC) + (j * THREADS + tid) // 2 for j in range(CPT)]
        ms = [ld_u8(p2 + tok[j]) for j in range(CPT)]
        cute.arch.barrier()
        half = tid % 2
        for j in cutlass.range_constexpr(CPT):
            pa = Uint32(cute.arch.shuffle_sync_bfly(ws[j][0], offset=1))
            pb = Uint32(cute.arch.shuffle_sync_bfly(ws[j][2], offset=1))
            count_chunk(hist, ws[j], pa, pb, half, ms[j])
        cute.arch.barrier()
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        if rank == 0:
            if tid < E:
                h = Uint32(0)
                for r in cutlass.range_constexpr(NCTA):
                    h = h + ld_dsmem(hist + tid * 4, r)
                c_all = h & Uint32(0xFFFF)
                c_un = h >> Uint32(16)
                st_i64(p8 + Int64(tid) * 8, c_un)
                st_f32_of_u32(p12 + Int64(tid) * 4, c_un)
                st_i64(p14 + Int64(tid) * 8, c_all)
                s = c_all
                for off in (1, 2, 4):
                    s = s + Uint32(cute.arch.shuffle_sync_bfly(s, offset=off))
                if tid % GROUP == 0:
                    st_i64(p17 + Int64(tid // GROUP) * 8, s)
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()


_fn = cute.compile(AuxCounts(), *[Int64(0)] * 6, cute.runtime.make_fake_stream(), options="--enable-tvm-ffi")

_F32 = torch.float32
_I64 = torch.int64
_BOOL = torch.bool
_CPU = torch.device("cpu")
_raw_stream = torch._C._cuda_getCurrentRawStream
_cur_dev = torch.cuda.current_device


def _outputs(dev):
    v8 = torch.empty((E,), dtype=_I64, device=dev)
    v12 = torch.empty((E,), dtype=_F32, device=dev)
    v14 = torch.empty((E // GROUP, GROUP), dtype=_I64, device=dev)
    v16 = torch.empty((E // GROUP,), dtype=_I64, device=dev)
    return v8, v12, v14, v16


def _finish(v8, v12, v14, v16):
    # _to_copy(v16, int64, strided, cpu, non_blocking=True): pinned empty_strided + copy_
    v17 = torch.empty((E // GROUP,), dtype=_I64, pin_memory=True)
    v17.copy_(v16, non_blocking=True)
    return v8, v12, v14, v17


def _run_general(in0, in1, in2):
    dev = in1.device
    if not (dev.type == "cuda" and in0.device == dev and in2.device == dev and tuple(in0.shape) == (NT, E)
            and in1.dtype == _I64 and tuple(in1.shape) == (NT, K) and in2.dtype == _BOOL
            and tuple(in2.shape) == (NT,)):
        raise ValueError("G-046: inputs do not match the spec (shapes, dtypes, devices)")
    in1 = in1 if in1.is_contiguous() and in1.data_ptr() % 16 == 0 else in1.clone(memory_format=torch.contiguous_format)
    in2 = in2 if in2.is_contiguous() else in2.clone(memory_format=torch.contiguous_format)
    with torch.cuda.device(dev):
        outs = _outputs(dev)
        _fn(in1.data_ptr(), in2.data_ptr(), *[o.data_ptr() for o in outs], _raw_stream(dev.index))
        return _finish(*outs)


def run(in0, in1, in2):
    dev = in1.device
    p1 = in1.data_ptr()
    p2 = in2.data_ptr()
    di = dev.index
    if (not (p1 & 15) and in1.dtype is _I64 and in2.dtype is _BOOL
            and in1.shape == (NT, K) and in1.stride() == (K, 1) and in2.shape == (NT,) and in2.stride() == (1,)
            and in0.shape == (NT, E) and in0.device == dev and in2.device == dev and di == _cur_dev()):
        outs = _outputs(dev)
        _fn(p1, p2, *[o.data_ptr() for o in outs], _raw_stream(di))
        return _finish(*outs)
    return _run_general(in0, in1, in2)


def _warm():
    """Load the module and run it once on the current device."""
    if not torch.cuda.is_available():
        return
    dev = torch.device("cuda", _cur_dev())
    run(torch.empty((NT, E), dtype=_F32, device=dev), torch.zeros((NT, K), dtype=_I64, device=dev),
        torch.zeros((NT,), dtype=_BOOL, device=dev))
    torch.cuda.synchronize(dev)


_warm()
