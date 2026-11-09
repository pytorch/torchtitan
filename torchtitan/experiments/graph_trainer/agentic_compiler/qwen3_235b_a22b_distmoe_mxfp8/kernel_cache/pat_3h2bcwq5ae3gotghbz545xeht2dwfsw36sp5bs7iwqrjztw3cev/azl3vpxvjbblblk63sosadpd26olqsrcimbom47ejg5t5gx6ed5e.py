"""FSDP reduce-scatter bucket pack of 7 dense-layer grads (Qwen3 235B-A22B, 256-way FSDP) as one CuTe DSL copy.

reference: bucketing._pre_bucket_reduce_scatter([in0..in6], 256, [2, 3, 4]) = torch._chunk_cat(ins, 0, 256).flatten():
every input is split into 256 chunks of ceil(dim0 / 256) rows along dim 0, chunks past the end are zero, and
chunk c of the output is [chunk c of in0, ..., chunk c of in6].flatten(). Per chunk (282658 bf16):
  in0 wqkv  [9216,4096]: 36 rows   elements [0, 147456)
  in1 wo    [4096,8192]: 16 rows   [147456, 278528)
  in2 q_norm [128]:     1 element  278528        (zero for c >= 128)
  in3 k_norm [128]:     1 element  278529        (zero for c >= 128)
  in4 gate  [128,4096]: 1 row      [278530, 282626) (zero for c >= 128)
  in5 attention_norm [4096]: 16    [282626, 282642)
  in6 ffn_norm [4096]: 16          [282642, 282658)
The chunk stride (565316 bytes) is only 4-byte aligned and every segment starts on a 4-byte boundary, so the
output is copied as 32-bit words (in2[c], in3[c] form one word). Chunk c in words (141329 per chunk):
  [0, 73728) in0, [73728, 139264) in1, 139264 = in2[c] | in3[c] << 16, [139265, 141313) in4,
  [141313, 141321) in5, [141321, 141329) in6.
Grid (69, 256): CTA k < 36 copies 2048 words of in0, 36 <= k < 68 2048 words of in1, k = 68 the 2065-word tail
(in2/in3/in4/in5/in6). Each thread issues 8 coalesced 4-byte loads (stride 256 words) before its 8 stores.
Bitwise exact by construction (pure copy; the padding is +0.0 = zero bytes).
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

NCHUNK = 256
SHAPES = [(9216, 4096), (4096, 8192), (128,), (128,), (128, 4096), (4096,), (4096,)]
OUT_NUMEL = 72360448
CW = OUT_NUMEL // 2 // NCHUNK  # words per chunk
W0 = 36 * 4096 // 2  # in0 words per chunk
W1 = 16 * 8192 // 2  # in1 words per chunk
W4 = 4096 // 2  # in4 words per chunk (c < 128)
W5 = 16 // 2  # in5 / in6 words per chunk
OFF1 = W0
OFF23 = W0 + W1
OFF4 = OFF23 + 1
OFF5 = OFF4 + W4
OFF6 = OFF5 + W5
assert OFF6 + W5 == CW and CW * NCHUNK * 2 == OUT_NUMEL
THREADS = 256
UNROLL = 8
TILE = THREADS * UNROLL  # words per CTA
K0 = W0 // TILE  # in0 CTAs per chunk
K1 = W1 // TILE
assert W0 % TILE == 0 and W1 % TILE == 0 and W4 == TILE
NK = K0 + K1 + 1


def _vals(args, loc, ip):
    return [a.ir_value(loc=loc, ip=ip) for a in args]


@dsl_user_op
def ld_b32(addr, *, loc=None, ip=None):
    return Uint32(llvm.inline_asm(
        T.i32(), _vals([Int64(addr)], loc, ip), "ld.global.nc.L1::no_allocate.b32 $0, [$1];", "=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def ld_u16(addr, *, loc=None, ip=None):
    return Uint32(llvm.inline_asm(
        T.i32(), _vals([Int64(addr)], loc, ip), "ld.global.nc.u16 $0, [$1];", "=r,l",
        has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def st_b32(addr, v, *, loc=None, ip=None):
    llvm.inline_asm(
        None, _vals([Int64(addr), Uint32(v)], loc, ip), "st.global.b32 [$0], $1;", "l,r",
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


def _copy_tile(src, dst, tidx):
    """TILE words src -> dst (byte addresses), thread tidx moves words tidx + u * THREADS."""
    vals = [ld_b32(src + (u * THREADS + tidx) * 4) for u in range(UNROLL)]
    for u in range(UNROLL):
        st_b32(dst + (u * THREADS + tidx) * 4, vals[u])


@cute.kernel
def _pack_kernel(m0: cute.Tensor, m1: cute.Tensor, m2: cute.Tensor, m3: cute.Tensor, m4: cute.Tensor,
                 m5: cute.Tensor, m6: cute.Tensor, mOut: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    k, c, _ = cute.arch.block_idx()
    t = Int64(tidx)
    cb = mOut.iterator.toint() + Int64(c) * (CW * 4)  # chunk base (byte address)
    if k < K0:
        _copy_tile(m0.iterator.toint() + (Int64(c) * W0 + Int64(k) * TILE) * 4, cb + Int64(k) * (TILE * 4), t)
    elif k < K0 + K1:
        kk = k - K0
        _copy_tile(m1.iterator.toint() + (Int64(c) * W1 + Int64(kk) * TILE) * 4,
                   cb + (OFF1 + Int64(kk) * TILE) * 4, t)
    else:
        if c < 128:
            _copy_tile(m4.iterator.toint() + Int64(c) * (W4 * 4), cb + OFF4 * 4, t)
            if tidx == 0:
                lo = ld_u16(m2.iterator.toint() + Int64(c) * 2)
                hi = ld_u16(m3.iterator.toint() + Int64(c) * 2)
                st_b32(cb + OFF23 * 4, lo | (hi << Uint32(16)))
        else:
            for u in cutlass.range_constexpr(UNROLL):
                st_b32(cb + (OFF4 + u * THREADS + t) * 4, Uint32(0))
            if tidx == 0:
                st_b32(cb + OFF23 * 4, Uint32(0))
        if tidx < W5:
            st_b32(cb + (OFF5 + t) * 4, ld_b32(m5.iterator.toint() + (Int64(c) * W5 + t) * 4))
        elif tidx < 2 * W5:
            st_b32(cb + (OFF6 - W5 + t) * 4, ld_b32(m6.iterator.toint() + (Int64(c) * W5 - W5 + t) * 4))


@cute.jit
def _launch(m0: cute.Tensor, m1: cute.Tensor, m2: cute.Tensor, m3: cute.Tensor, m4: cute.Tensor,
            m5: cute.Tensor, m6: cute.Tensor, mOut: cute.Tensor, stream):
    _pack_kernel(m0, m1, m2, m3, m4, m5, m6, mOut).launch(grid=[NK, NCHUNK, 1], block=[THREADS, 1, 1], stream=stream)


_CACHE = {}


def _fake(n, align):
    return cute.runtime.make_fake_tensor(cutlass.BFloat16, (n,), stride=(1,), assumed_align=align)


def run(in0, in1, in2, in3, in4, in5, in6):
    ins = (in0, in1, in2, in3, in4, in5, in6)
    for i, (t, shape) in enumerate(zip(ins, SHAPES)):
        align = 2 if i in (2, 3) else 4
        if not (t.is_cuda and t.dtype == torch.bfloat16 and tuple(t.shape) == shape and t.is_contiguous()
                and t.data_ptr() % align == 0 and t.device == in0.device):
            raise ValueError(f"in{i}: expected contiguous {align}B-aligned bf16 {shape} on {in0.device}, got "
                             f"{t.dtype} {tuple(t.shape)} stride {t.stride()} on {t.device}")
    dev = in0.device
    out = torch.empty(OUT_NUMEL, dtype=torch.bfloat16, device=dev)
    fn = _CACHE.get(dev.index)
    if fn is None:
        fakes = [_fake(t.numel(), 2 if i in (2, 3) else 4) for i, t in enumerate(ins)]
        fn = cute.compile(_launch, *fakes, _fake(OUT_NUMEL, 16),
                          cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True), options="--enable-tvm-ffi")
        _CACHE[dev.index] = fn
    with torch.cuda.device(dev):
        fn(*[t.view(-1) for t in ins], out)
    return out
