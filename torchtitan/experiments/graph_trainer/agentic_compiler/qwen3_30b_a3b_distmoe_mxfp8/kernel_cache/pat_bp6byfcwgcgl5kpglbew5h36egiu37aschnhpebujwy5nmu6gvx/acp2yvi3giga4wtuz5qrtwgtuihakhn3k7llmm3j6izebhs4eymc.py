"""MoE router forward: per-expert token counts (Qwen3 30B-A3B, 32768 tokens, 128 experts, top-8).

reference: v7 = scatter(zeros_bool, idx, True) & ~mask[:, None]; counts = v7.sum(0) (int64) and
v7.float().sum(0) (fp32). Both are exact integer counts: every partial sum of 0/1 values is an
integer below 2^24, so the fp32 sum is exact in any order. A row counts each distinct expert
once (scatter of True is idempotent), and rows whose mask byte is nonzero count nothing.

Two CuTe DSL kernels, no state kept across calls and no zero-initialized buffers:
  1. histogram: CTA c histograms rows [256 c, 256 c + 256) in shared memory (one row per thread,
     duplicate indices within a row skipped) and writes its 128 counts to a per-call scratch
     buffer (every entry is written, so it needs no initialization).
  2. reduce: one CTA sums the 128 partial histograms and writes both outputs.
Kernel 2 is launched with programmatic dependent launch (PDL): kernel 1 lets it launch early,
and kernel 2 waits (griddepcontrol.wait) for kernel 1 to complete before reading the partials.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

ROWS = 32768
NE = 128
TOPK = 8
THREADS = 256  # one row per thread
NCTA = ROWS // THREADS
RTHREADS = 1024  # reduce kernel: thread (p, e) sums partials p * 16 .. p * 16 + 15 of expert e
PDL = True


@dsl_user_op
def ld_gpu_u32(addr, *, loc=None, ip=None):
    """Coherent (L2) load of a partial count written by kernel 1."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _tree_sum(v):
    while len(v) > 1:
        v = [v[i] + v[i + 1] for i in range(0, len(v), 2)]
    return v[0]


@cute.kernel
def _hist_kernel(mIdx: cute.Tensor, mMask: cute.Tensor, mPart: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    if cutlass.const_expr(PDL):
        cute.arch.griddepcontrol_launch_dependents()  # the reduce kernel may launch (it waits for us)

    smem = cutlass.utils.SmemAllocator()
    sHist = smem.allocate_tensor(Int32, cute.make_layout(NE), byte_alignment=16)

    row = bidx * THREADS + tidx
    ids = cute.make_rmem_tensor(cute.make_layout(2 * TOPK), Int32)  # (lo, hi) words of the int64s
    cute.autovec_copy(cute.make_tensor(cute.recast_ptr(mIdx.iterator + row * TOPK, dtype=Int32).align(16), cute.make_layout(2 * TOPK)), ids)
    m = mMask[row]
    if tidx < NE:
        sHist[tidx] = Int32(0)
    cute.arch.sync_threads()

    # an index outside [0, NE) makes the reference raise: such a row is skipped instead of letting
    # it touch memory outside the histogram (reading the high words also keeps the loads 16 B wide)
    hi = ids[1] | ids[3] | ids[5] | ids[7] | ids[9] | ids[11] | ids[13] | ids[15]
    lo = ids[0] | ids[2] | ids[4] | ids[6] | ids[8] | ids[10] | ids[12] | ids[14]
    if (m == Uint8(0)) & ((hi | (lo & Int32(-NE))) == Int32(0)):
        v = [ids[2 * k] for k in range(TOPK)]
        for k in cutlass.range_constexpr(TOPK):
            fresh = Int32(1)
            for j in cutlass.range_constexpr(k):
                if v[j] == v[k]:
                    fresh = Int32(0)
            if fresh != Int32(0):
                cute.arch.atomic_add(sHist.iterator + v[k], Int32(1))
    cute.arch.sync_threads()
    if tidx < NE:
        mPart[bidx * NE + tidx] = sHist[tidx]


@cute.kernel
def _reduce_kernel(mPart: cute.Tensor, mCnt: cute.Tensor, mCntF: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    if cutlass.const_expr(PDL):
        cute.arch.griddepcontrol_wait()  # kernel 1 has completed and its partials are visible
    smem = cutlass.utils.SmemAllocator()
    sRed = smem.allocate_tensor(Int32, cute.make_layout(RTHREADS), byte_alignment=16)
    e = tidx % NE
    p = tidx // NE
    per = NCTA // (RTHREADS // NE)
    vals = [ld_gpu_u32((mPart.iterator + (p * per + c) * NE + e).toint()) for c in range(per)]
    sRed[tidx] = _tree_sum(vals)
    cute.arch.sync_threads()
    if tidx < NE:
        tot = Int32(0)
        for q in cutlass.range_constexpr(RTHREADS // NE):
            tot = tot + sRed[q * NE + tidx]
        mCnt[tidx] = Int64(tot)
        mCntF[tidx] = Float32(tot)


@cute.jit
def _launch(mIdx: cute.Tensor, mMask: cute.Tensor, mPart: cute.Tensor, mCnt: cute.Tensor, mCntF: cute.Tensor, stream):
    _hist_kernel(mIdx, mMask, mPart).launch(grid=[NCTA, 1, 1], block=[THREADS, 1, 1], stream=stream)
    _reduce_kernel(mPart, mCnt, mCntF).launch(grid=[1, 1, 1], block=[RTHREADS, 1, 1], stream=stream, use_pdl=PDL)


_CACHE = {}


def _fake(dtype, n):
    return cute.runtime.make_fake_tensor(dtype, (n,), stride=(1,), assumed_align=16)


def run(in0, in1, in2):
    dev = in1.device
    idx = in1 if in1.is_contiguous() else in1.contiguous()
    if idx.data_ptr() % 16:
        idx = idx.clone()
    msk = in2.view(torch.uint8)
    if not msk.is_contiguous():
        msk = msk.contiguous()
    part = torch.empty(NCTA * NE, dtype=torch.int32, device=dev)  # per-call scratch, fully written by kernel 1
    cnt = torch.empty(NE, dtype=torch.int64, device=dev)
    cntf = torch.empty(NE, dtype=torch.float32, device=dev)
    fn = _CACHE.get(dev.index)
    if fn is None:
        fn = cute.compile(
            _launch,
            _fake(Int64, ROWS * TOPK),
            cute.runtime.make_fake_tensor(Uint8, (ROWS,), stride=(1,), assumed_align=1),
            _fake(Int32, NCTA * NE),
            _fake(Int64, NE),
            _fake(Float32, NE),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _CACHE[dev.index] = fn
    fn(idx.view(-1), msk.view(-1), part, cnt, cntf)
    return (cnt, cntf)
