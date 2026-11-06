"""G-025 (GPT-OSS 120B MXFP8): new_zeros [in1, 2880] + index_put(accumulate=True) of in0
[R, 2880] at in2 [R]; bitwise equal to reference.py.

Reference semantics (deterministic algorithms; aten index_put_with_sort_kernel ->
indexing_backward_kernel<bf16, 4>, slice 2880 > 32): cub radix sort of the wrapped index
(stable), duplicates in ascending source position, each duplicate does
w = bf16(float(w) + float(x)) from w = +0; rows without a source stay +0. In training in2 is a
permutation, so out[in2[i]] = 0 + in0[i] (-0 -> +0, NaN -> 0x7fff), but any index works.
Here: acc = add.rn.bf16x2(acc, x) per source row in position order from +0 (add.rn.bf16x2 ==
torch's bf16 add for all 2^32 pairs: sol's exhaustive test).

Kernel (one cooperative CuTe DSL launch, one CTA of 8 warps per SM, CTA o owns destination
rows [o*B, o*B + B), B = ceil(M / #SM)):
  phase A: CTA c takes index slice [c*S, c*S + S) (S = ceil(N / #SM)), range-checks it (out of
    range -> trap, as the reference's device assert), wraps negatives, and stable-sorts the
    slice by owner CTA (match.any ranks per warp, block scan) into a per-call workspace.
  grid barrier: persistent per-device monotonic 64-bit counter (one barrier per launch, #SM
    CTAs per launch, so it is a multiple of #SM between launches; no per-call zeroing). The
    launch generation (counter / #SM, the same for every CTA of a launch) selects one of two
    persistent workspaces: launch k+2 can only start once every CTA of launch k has exited
    (one CTA per SM, cooperative), so a workspace is never shared by running launches.
  phase B: CTA o gathers its entries slice by slice (= ascending source position), builds
    per-row source streams (stable counting sort by row: match.any ranks per warp, block scan),
    then each warp streams its rows' source rows through a 4-deep cp.async.bulk ring (5760 B
    rows into smem, mbarrier completion) and stores from registers.
  If a CTA's entries exceed the smem list (heavy duplicates), it works in windows of slices and
  rows carry their bf16 partial through the output (exact: the state is bf16 after every add).
No host syncs (the per-device state is zeroed once, at the first call on a device). Limits (else the reference ops
run, which is exact by definition): N < 2^22, N <= 160 * 1024, M <= 512 * #SM.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

D = 2880
ROWB = D * 2  # bytes per row
NV = D // 8  # 16-byte vectors per row (360)
NJ = (NV + 31) // 32  # vectors per lane (12, the last one for lanes < 8 only)

W = 8  # warps per CTA
K2T = 32 * W
STAGES = 4
GMAX = 160  # CTAs (one per SM)
PMAX = GMAX  # scanner slices
SA_MAX = 1024  # index positions per scanner slice (n <= GMAX * SA_MAX)
IT_MAX = SA_MAX // K2T  # 32-position iterations per warp in phase A
BSZMAX = 512  # destination rows per CTA
LCAP = 1024  # smem list entries (>= SA_MAX)
WS_BYTES = (GMAX * SA_MAX + GMAX * (GMAX + 1) + 64) * 4  # one per-launch workspace
STATE_BYTES = 256 + 2 * WS_BYTES  # barrier word, then two workspaces
GB = 8  # batched gather loads per chunk

MODE_ACC_ADD, MODE_ACC, MODE_LAST = 0, 1, 2


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
def ld_v4(addr, *, loc=None, ip=None):
    r = _asmN([T.i32()] * 4, [Int64(addr)], "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,l", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def st_v4(addr, w, *, loc=None, ip=None):
    _asm1(None, [Int64(addr)] + [Uint32(x) for x in w],
          "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def st_v4_pred(addr, w, pred, *, loc=None, ip=None):
    _asm1(None, [Int64(addr)] + [Uint32(x) for x in w] + [Int32(pred)],
          "{ .reg .pred q; setp.ne.s32 q, $5, 0; @q st.global.v4.b32 [$0], {$1, $2, $3, $4}; }",
          "l,r,r,r,r,r", loc, ip, side=True)


@dsl_user_op
def lds_v4(addr, *, loc=None, ip=None):
    r = _asmN([T.i32()] * 4, [Int32(addr)], "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
              "=r,=r,=r,=r,r", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def ld_pair(saddr, gaddr, mode, *, loc=None, ip=None):
    """Index pair (q, q + 1) as (lo0, hi0, lo1, hi1): mode 2 -> 16 B from smem, 1 -> element q
    from global (q + 1 invalid), 0 -> zeros."""
    r = _asmN([T.i32()] * 4, [Int32(saddr), Int64(gaddr), Int32(mode)],
              "{ .reg .pred p2, p1; setp.eq.s32 p2, $6, 2; setp.eq.s32 p1, $6, 1;\n"
              "mov.b32 $0, 0; mov.b32 $1, 0; mov.b32 $2, 0; mov.b32 $3, 0;\n"
              "@p2 ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];\n"
              "@p1 ld.global.nc.v2.b32 {$0, $1}, [$5]; }",
              "=r,=r,=r,=r,r,l,r", loc, ip, side=True)
    return [Uint32(v) for v in r]


@dsl_user_op
def classify(lo, hi, inr, m, row0, nrows, *, loc=None, ip=None):
    """int64 index (lo, hi) -> (hit, local row, bad): wrap negatives by m; bad = in range of
    the scan but outside [-m, m); hit = valid and row0 <= wrapped < row0 + nrows."""
    r = _asmN([T.i32()] * 3, [Uint32(lo), Uint32(hi), Int32(inr), Int32(m), Int32(row0), Int32(nrows)],
              "{ .reg .pred pin, pv, ph, pb; .reg .b32 sg, t, r;\n"
              "setp.ne.s32 pin, $5, 0;\n"
              "shr.s32 sg, $4, 31;\n"
              "and.b32 t, sg, $6; add.u32 t, t, $3;\n"
              "setp.eq.s32 pv, $4, sg;\n"
              "setp.lt.and.u32 pv, t, $6, pv;\n"
              "sub.u32 r, t, $7;\n"
              "setp.lt.and.u32 ph, r, $8, pv;\n"
              "and.pred ph, ph, pin;\n"
              "not.pred pb, pv; and.pred pb, pb, pin;\n"
              "selp.u32 $0, 1, 0, ph; mov.b32 $1, r; selp.u32 $2, 1, 0, pb; }",
              "=r,=r,=r,r,r,r,r,r,r", loc, ip)
    return Int32(r[0]), Int32(r[1]), Int32(r[2])


@dsl_user_op
def trap_if(pred, *, loc=None, ip=None):
    _asm1(None, [Int32(pred)], "{ .reg .pred q; setp.ne.s32 q, $0, 0; @q trap; }", "r", loc, ip, side=True)


@dsl_user_op
def lanemask_lt(*, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [], "mov.u32 $0, %lanemask_lt;", "=r", loc, ip))


@dsl_user_op
def match_any(key, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Int32(key)], "match.any.sync.b32 $0, $1, 0xffffffff;", "=r,r", loc, ip,
                        side=True))


@dsl_user_op
def bfind(x, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Uint32(x)], "bfind.u32 $0, $1;", "=r,r", loc, ip))


@dsl_user_op
def sts_b32(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Int32(a)], "st.shared.b32 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def lds_b32(addr, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int32(addr)], "ld.shared.b32 $0, [$1];", "=r,r", loc, ip, side=True))


@dsl_user_op
def sts_u16(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Int32(a)], "{ .reg .b16 s; cvt.u16.u32 s, $1; st.shared.b16 [$0], s; }",
          "r,r", loc, ip, side=True)


@dsl_user_op
def lds_u16(addr, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int32(addr)], "{ .reg .b16 s; ld.shared.b16 s, [$1]; cvt.u32.u16 $0, s; }",
                       "=r,r", loc, ip, side=True))


@dsl_user_op
def atom_add_s(addr, v, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Int32(v)], "red.shared.add.u32 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def atom_max_s(addr, v, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Int32(v)], "red.shared.max.s32 [$0], $1;", "r,r", loc, ip, side=True)


@dsl_user_op
def sts_u8(addr, a, *, loc=None, ip=None):
    _asm1(None, [Int32(addr), Int32(a)], "{ .reg .b16 s; cvt.u16.u32 s, $1; st.shared.b8 [$0], s; }",
          "r,r", loc, ip, side=True)


@dsl_user_op
def lds_u8(addr, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int32(addr)], "{ .reg .b16 s; ld.shared.u8 s, [$1]; cvt.u32.u16 $0, s; }",
                       "=r,r", loc, ip, side=True))


@dsl_user_op
def syncwarp(*, loc=None, ip=None):
    _asm1(None, [], "bar.warp.sync 0xffffffff;", "", loc, ip, side=True)


@dsl_user_op
def mbar_init(addr, *, loc=None, ip=None):
    _asm1(None, [Int32(addr)], "mbarrier.init.shared::cta.b64 [$0], 1;", "r", loc, ip, side=True)


@dsl_user_op
def fence_init(*, loc=None, ip=None):
    _asm1(None, [], "fence.mbarrier_init.release.cluster;\nfence.proxy.async.shared::cta;", "", loc, ip,
          side=True)


@dsl_user_op
def bulk_load(mbar, sdst, gsrc, nbytes, *, loc=None, ip=None):
    """Expect nbytes on mbar and copy nbytes global -> shared (16 B granules)."""
    _asm1(None, [Int32(mbar), Int32(sdst), Int64(gsrc), Int32(nbytes)],
          "fence.proxy.async.shared::cta;\n"
          "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $3;\n"
          "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [$1], [$2], $3, [$0];",
          "r,r,l,r", loc, ip, side=True)


@dsl_user_op
def mbar_wait(addr, parity, *, loc=None, ip=None):
    """Wait for the mbarrier phase; ~2^30 polls trap instead of hanging (protocol error)."""
    _asm1(None, [Int32(addr), Int32(parity)],
          "{ .reg .pred P1; .reg .u32 n; mov.u32 n, 0;\nLAB_WAIT:\n"
          "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1;\n"
          "@P1 bra DONE;\nadd.u32 n, n, 1; setp.gt.u32 P1, n, 1073741824; @P1 trap;\nbra LAB_WAIT;\nDONE: }",
          "r,r", loc, ip, side=True)


@dsl_user_op
def badd(a, b, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(a), Uint32(b)], "add.rn.bf16x2 $0, $1, $2;", "=r,r,r", loc, ip))


@dsl_user_op
def bmul1(a, *, loc=None, ip=None):
    """bf16(float(x) * 1.0f) per half: NaN -> 0x7fff, everything else unchanged."""
    return Uint32(_asm1(T.i32(), [Uint32(a)],
                        "{ .reg .b32 one; mov.b32 one, 0x3F803F80; mul.rn.bf16x2 $0, $1, one; }", "=r,r", loc, ip))



def _hitbits_text(k, checked):
    """Asm text: update mask ($0, tied to the last input) with the hit bits of index pair k
    (inputs lo0, hi0, lo1, hi1, m, row0, nrows[, lim], mask)."""
    lines = ["{ .reg .b32 sg, t; .reg .pred p, v;", "mov.b32 $0, $%d;" % (9 if checked else 8)]
    for h in range(2):
        lo, hi = 1 + 2 * h, 2 + 2 * h
        lines.append(f"shr.s32 sg, ${hi}, 31; and.b32 t, sg, $5; add.u32 t, t, ${lo}; sub.u32 t, t, $6;")
        lines.append("setp.lt.u32 p, t, $7;")
        if checked:
            lines.append(f"setp.gt.s32 v, $8, {2 * k + h}; and.pred p, p, v;")
        lines.append(f"@p or.b32 $0, $0, {1 << (2 * k + h)};")
    lines.append("}")
    return "\n".join(lines)


@dsl_user_op
def hitbits(k, w4, m, row0, nrows, mask, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(x) for x in w4] + [Int32(m), Int32(row0), Int32(nrows), Uint32(mask)],
                        _hitbits_text(k, False), "=r,r,r,r,r,r,r,r,r", loc, ip))


@dsl_user_op
def hitbits_lim(k, w4, m, row0, nrows, lim, mask, *, loc=None, ip=None):
    return Uint32(_asm1(T.i32(), [Uint32(x) for x in w4] + [Int32(m), Int32(row0), Int32(nrows), Int32(lim),
                                                           Uint32(mask)],
                        _hitbits_text(k, True), "=r,r,r,r,r,r,r,r,r,r", loc, ip))


@dsl_user_op
def ld_elem(saddr, gaddr, from_smem, *, loc=None, ip=None):
    """One int64 index as (lo, hi), from smem or global."""
    r = _asmN([T.i32()] * 2, [Int32(saddr), Int64(gaddr), Int32(from_smem)],
              "{ .reg .pred p; setp.ne.s32 p, $4, 0;\n"
              "@p ld.shared.v2.b32 {$0, $1}, [$2];\n"
              "@!p ld.global.nc.v2.b32 {$0, $1}, [$3]; }",
              "=r,=r,r,l,r", loc, ip, side=True)
    return Uint32(r[0]), Uint32(r[1])


@dsl_user_op
def wrap_local(lo, hi, m, row0, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Uint32(lo), Uint32(hi), Int32(m), Int32(row0)],
                       "{ .reg .b32 sg, t; shr.s32 sg, $2, 31; and.b32 t, sg, $3; add.u32 t, t, $1;"
                       " sub.u32 $0, t, $4; }", "=r,r,r,r,r", loc, ip))


@dsl_user_op
def check_index(addr, m, *, loc=None, ip=None):
    """Trap unless the int64 index at addr is in [-m, m) (the reference device-asserts)."""
    _asm1(None, [Int64(addr), Int32(m)],
          "{ .reg .b32 lo, hi, sg, t; .reg .pred pv;\n"
          "ld.global.nc.v2.b32 {lo, hi}, [$0];\n"
          "shr.s32 sg, hi, 31; and.b32 t, sg, $1; add.u32 t, t, lo;\n"
          "setp.eq.s32 pv, hi, sg; setp.lt.and.u32 pv, t, $1, pv;\n"
          "@!pv trap; }", "l,r", loc, ip, side=True)


@dsl_user_op
def ctz(x, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Uint32(x)], "{ .reg .b32 t; brev.b32 t, $1; clz.b32 $0, t; }", "=r,r", loc, ip))

def _ballot(p):
    return Uint32(cute.arch.vote_ballot_sync(p))


def _popc(x):
    return Int32(cute.arch.popc(x))




@dsl_user_op
def ld_idx_pred(addr, pred, *, loc=None, ip=None):
    """int64 index as (lo, hi), zeros when pred == 0."""
    r = _asmN([T.i32()] * 2, [Int64(addr), Int32(pred)],
              "{ .reg .pred p; setp.ne.s32 p, $3, 0; mov.b32 $0, 0; mov.b32 $1, 0;\n"
              "@p ld.global.nc.v2.b32 {$0, $1}, [$2]; }", "=r,=r,l,r", loc, ip, side=True)
    return Uint32(r[0]), Uint32(r[1])


@dsl_user_op
def check_wrap(lo, hi, valid, m, *, loc=None, ip=None):
    """Trap if valid and the index is outside [-m, m); returns the wrapped index."""
    return Int32(_asm1(T.i32(), [Uint32(lo), Uint32(hi), Int32(valid), Int32(m)],
                       "{ .reg .b32 sg, t; .reg .pred pv, pin;\n"
                       "setp.ne.s32 pin, $3, 0;\n"
                       "shr.s32 sg, $2, 31; and.b32 t, sg, $4; add.u32 t, t, $1;\n"
                       "setp.eq.s32 pv, $2, sg; setp.lt.and.u32 pv, t, $4, pv;\n"
                       "not.pred pv, pv; and.pred pv, pv, pin;\n"
                       "@pv trap; mov.b32 $0, t; }", "=r,r,r,r,r", loc, ip, side=True))


@dsl_user_op
def ld_b32(addr, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int64(addr)], "ld.global.b32 $0, [$1];", "=r,l", loc, ip, side=True))


@dsl_user_op
def st_b32(addr, v, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Int32(v)], "st.global.b32 [$0], $1;", "l,r", loc, ip, side=True)




@dsl_user_op
def grid_sync(word, nctas, *, loc=None, ip=None):
    """Grid barrier on a persistent monotonic 64-bit counter (one thread per CTA, between CTA
    barriers). Every launch has exactly nctas CTAs and one barrier, so the counter is a
    multiple of nctas between launches; a CTA waits until the counter reaches the next
    multiple. Launches on one device cannot overlap (cooperative, one CTA per SM by smem).
    A wait of ~2^30 polls traps instead of hanging."""
    _asm1(None, [Int64(word), Int32(nctas)],
          "{ .reg .u64 old, cur, tgt, g; .reg .u32 n; .reg .pred p;\n"
          "cvt.u64.u32 g, $1;\n"
          "fence.acq_rel.gpu;\n"
          "atom.add.release.gpu.u64 old, [$0], 1;\n"
          "div.u64 tgt, old, g; add.u64 tgt, tgt, 1; mul.lo.u64 tgt, tgt, g;\n"
          "mov.u32 n, 0;\n"
          "LAB_SPIN:\n"
          "ld.acquire.gpu.u64 cur, [$0];\n"
          "add.u32 n, n, 1; setp.gt.u32 p, n, 1073741824; @p trap;\n"
          "setp.lt.u64 p, cur, tgt;\n"
          "@p bra LAB_SPIN;\n"
          "fence.acq_rel.gpu; }", "l,r", loc, ip, side=True)


@dsl_user_op
def ld_b32_pred(addr, pred, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int64(addr), Int32(pred)],
                       "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.b32 $0, 0; @p ld.global.b32 $0, [$1]; }",
                       "=r,l,r", loc, ip, side=True))


@dsl_user_op
def ld_acquire_u64(addr, *, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [Int64(addr)], "ld.acquire.gpu.u64 $0, [$1];", "=l,l", loc, ip, side=True))


class RowScatterCoop:
    """Phase A: CTA c stable-sorts its index slice by owner CTA (rows [o*B, o*B + B)) into the
    workspace; grid barrier; phase B: CTA o gathers its entries slice by slice (position
    order), builds per-row source streams and streams rows through a cp.async.bulk ring."""

    def __init__(self, mode, nsm):
        self.mode = mode
        self.nsm = nsm

    @cute.jit
    def __call__(self, anchor, p_src: Int64, p_idx: Int64, p_add: Int64, p_out: Int64, p_state: Int64,
                 m: Int32, n: Int32, stream):
        self.kernel(p_src, p_idx, p_add, p_out, p_state, m, n).launch(
            grid=[self.nsm, 1, 1], block=[K2T, 1, 1], stream=stream, cooperative=True)

    @cute.kernel
    def kernel(self, p_src: Int64, p_idx: Int64, p_add: Int64, p_out: Int64, p_state: Int64, m: Int32, n: Int32):
        tid, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = tid % 32
        warp = tid // 32
        npch = gdim
        bcta = cutlass.max((m + npch - 1) // npch, 1)
        # launch generation = barrier count / #CTAs (every CTA of this launch reads the same value,
        # see grid_sync); workspaces alternate by generation, so a launch never shares its
        # workspace with a still-running previous launch
        gen = ld_acquire_u64(p_state) // Int64(self.nsm)
        p_ws = p_state + 256 + (gen % 2) * WS_BYTES
        sa = (n + npch - 1) // npch
        row0 = b * bcta
        nrows = cutlass.max(cutlass.min(bcta, m - row0), 0)
        p_starts = p_ws + Int64(npch * sa) * 4

        smem = cutlass.utils.SmemAllocator()
        s_ring = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(W * STAGES * ROWB), byte_alignment=128)
        s_list = smem.allocate_tensor(cutlass.Int32, cute.make_layout(LCAP), byte_alignment=16)
        s_stream = smem.allocate_tensor(cutlass.Int32, cute.make_layout(LCAP + BSZMAX), byte_alignment=16)
        s_ccnt = smem.allocate_tensor(cutlass.Int32, cute.make_layout(PMAX), byte_alignment=16)
        s_cbase = smem.allocate_tensor(cutlass.Int32, cute.make_layout(PMAX), byte_alignment=16)
        s_coffs = smem.allocate_tensor(cutlass.Int32, cute.make_layout(PMAX + 4), byte_alignment=16)
        s_rcnt = smem.allocate_tensor(cutlass.Int32, cute.make_layout(BSZMAX), byte_alignment=16)
        s_roffs = smem.allocate_tensor(cutlass.Int32, cute.make_layout(BSZMAX + 4), byte_alignment=16)
        s_hp = smem.allocate_tensor(cutlass.Int8, cute.make_layout(BSZMAX), byte_alignment=16)
        s_mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(W * STAGES), byte_alignment=16)
        s_h = smem.allocate_tensor(cutlass.Int32, cute.make_layout(max(W * GMAX, W * BSZMAX // 2)), byte_alignment=16)
        s_rank = smem.allocate_tensor(cutlass.Int32, cute.make_layout(LCAP), byte_alignment=16)
        s_t = smem.allocate_tensor(cutlass.Int32, cute.make_layout(W + 4), byte_alignment=16)
        a_ring = s_ring.iterator.toint() + warp * (STAGES * ROWB)
        a_list = s_list.iterator.toint()
        a_stream = s_stream.iterator.toint()
        a_ccnt = s_ccnt.iterator.toint()
        a_cbase = s_cbase.iterator.toint()
        a_coffs = s_coffs.iterator.toint()
        a_rcnt = s_rcnt.iterator.toint()
        a_roffs = s_roffs.iterator.toint()
        a_hp = s_hp.iterator.toint()
        a_mb = s_mbar.iterator.toint() + warp * (STAGES * 8)
        a_h = s_h.iterator.toint()
        a_rank = s_rank.iterator.toint()
        a_wc = a_h  # phase B per-(warp, row) u16 counters / bases, aliasing the phase A histogram
        a_t = s_t.iterator.toint()
        lmask = lanemask_lt()

        # ================= phase A: this CTA's slice [b*sa, b*sa + sa) by owner CTA
        for w in cutlass.range_constexpr(W):
            for o in cutlass.range(tid, npch, K2T):
                sts_b32(a_h + (w * GMAX + o) * 4, 0)
        sw = (sa + W - 1) // W
        send = cutlass.min(b * sa + sa, n)
        base = b * sa + warp * sw
        os_ = []
        rls = []
        valids = []
        for it in cutlass.range_constexpr(IT_MAX):
            pos = base + it * 32 + lane  # noqa
            vld = Int32((it * 32 + lane < sw) & (pos < send))
            ilo, ihi = ld_idx_pred(p_idx + Int64(pos) * 8, vld)
            t = check_wrap(ilo, ihi, vld, m)
            o = t // bcta
            os_.append(o)
            rls.append(t - o * bcta)
            valids.append(vld)
        cute.arch.barrier()
        a_hw = a_h + warp * (GMAX * 4)
        ranks = []
        for it in cutlass.range_constexpr(IT_MAX):
            key = cutlass.select_(valids[it] != 0, os_[it], Int32(0x10000) + lane)
            peers = match_any(key)
            cur = Int32(0)
            if valids[it] != 0:
                cur = lds_b32(a_hw + key * 4)
            ranks.append(cur + _popc(peers & lmask))
            syncwarp()
            if (valids[it] != 0) & (bfind(peers) == lane):
                sts_b32(a_hw + key * 4, cur + _popc(peers))
            syncwarp()
        cute.arch.barrier()
        # exclusive scan over (owner, warp), owner-major; thread t owns owner t
        tot = Int32(0)
        if tid < npch:
            for w in cutlass.range_constexpr(W):
                tot = tot + lds_b32(a_h + (w * GMAX + tid) * 4)
        incl = tot
        for off in (1, 2, 4, 8, 16):
            y = Int32(cute.arch.shuffle_sync_up(incl, offset=off))
            if lane >= off:
                incl = incl + y
        if lane == 31:
            sts_b32(a_t + warp * 4, incl)
        cute.arch.barrier()
        run = incl - tot
        for w in cutlass.range_constexpr(W):
            tw = lds_b32(a_t + w * 4)
            if w < warp:
                run = run + tw
        if tid < npch:
            st_b32(p_starts + (Int64(b) * (npch + 1) + tid) * 4, run)
            for w in cutlass.range_constexpr(W):
                hv = lds_b32(a_h + (w * GMAX + tid) * 4)
                sts_b32(a_h + (w * GMAX + tid) * 4, run)
                run = run + hv
        if tid == 0:
            st_b32(p_starts + (Int64(b) * (npch + 1) + npch) * 4, cutlass.max(send - b * sa, 0))
        cute.arch.barrier()
        for it in cutlass.range_constexpr(IT_MAX):
            if valids[it] != 0:
                slot = lds_b32(a_hw + os_[it] * 4) + ranks[it]
                st_b32(p_ws + (Int64(b) * sa + slot) * 4, (rls[it] << 22) | (base + it * 32 + lane))
        cute.arch.barrier()
        if tid == 0:
            grid_sync(p_state, npch)
        cute.arch.barrier()

        # ================= phase B: gather this CTA's entries, per-row streams, rows
        if lane == 0:
            for k in cutlass.range_constexpr(STAGES):
                mbar_init(a_mb + k * 8)
            fence_init()
        # chunk counts and their exclusive prefix (thread c owns chunk c; npch <= K2T)
        ccv = Int32(0)
        if tid < npch:
            s0 = ld_b32(p_starts + (Int64(tid) * (npch + 1) + b) * 4)
            s1 = ld_b32(p_starts + (Int64(tid) * (npch + 1) + b + 1) * 4)
            ccv = s1 - s0
            sts_b32(a_ccnt + tid * 4, ccv)
            sts_b32(a_cbase + tid * 4, tid * sa + s0)
        for k in cutlass.range_constexpr(BSZMAX // K2T):
            sts_u8(a_hp + tid + k * K2T, 0)
        incl = ccv
        for off in (1, 2, 4, 8, 16):
            y = Int32(cute.arch.shuffle_sync_up(incl, offset=off))
            if lane >= off:
                incl = incl + y
        if lane == 31:
            sts_b32(a_t + warp * 4, incl)
        cute.arch.barrier()
        run = incl - ccv
        ctot = Int32(0)
        for w in cutlass.range_constexpr(W):
            tw = lds_b32(a_t + w * 4)
            if w < warp:
                run = run + tw
            ctot = ctot + tw
        if tid < npch:
            sts_b32(a_coffs + tid * 4, run)
        if tid == 0:
            sts_b32(a_coffs + npch * 4, ctot)
        cute.arch.barrier()

        last_ok = Int32(lane < NV - 32 * (NJ - 1))
        voff_last = cutlass.min(lane + 32 * (NJ - 1), NV - 1) * 16
        q3 = Int32(0)
        c0 = Int32(0)
        done = Int32(0)
        while done == 0:
            # window [c0, c1): as many chunks as fit in the list (one chunk always fits)
            base0 = lds_b32(a_coffs + c0 * 4)
            lo = cutlass.min(c0 + 1, npch)
            hi = npch
            if ctot - base0 <= LCAP:
                lo = npch
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if lds_b32(a_coffs + mid * 4) - base0 <= LCAP:
                    lo = mid
                else:
                    hi = mid - 1
            c1 = lo
            final = Int32(c1 == npch)
            nl = lds_b32(a_coffs + c1 * 4) - base0
            # gather (position order): entry e of the window comes from the chunk c with
            # coffs[c] <= base0 + e < coffs[c + 1] (binary search); all loads in flight at once
            gvs = []
            for k in cutlass.range_constexpr(LCAP // K2T):
                e = tid + k * K2T
                ge = base0 + e
                clo = c0
                chi = c1 - 1
                while clo < chi:
                    cm = (clo + chi + 1) // 2
                    if lds_b32(a_coffs + cm * 4) <= ge:
                        clo = cm
                    else:
                        chi = cm - 1
                gsrc_e = lds_b32(a_cbase + clo * 4) + ge - lds_b32(a_coffs + clo * 4)
                gvs.append(ld_b32_pred(p_ws + Int64(gsrc_e) * 4, Int32(e < nl)))
            for k in cutlass.range_constexpr(LCAP // K2T):
                e = tid + k * K2T
                if e < nl:
                    sts_b32(a_list + e * 4, gvs[k])
            # zero the per-(warp, row) counters (u16)
            for k in cutlass.range(tid, W * BSZMAX // 2, K2T):
                sts_b32(a_wc + k * 4, 0)
            cute.arch.barrier()
            # pass 1: warp w ranks its list range within (warp, row), stable
            lw = (nl + W - 1) // W
            e_lo = cutlass.min(warp * lw, nl)
            e_hi = cutlass.min(e_lo + lw, nl)
            a_wcw = a_wc + warp * (BSZMAX * 2)
            for j in cutlass.range(e_lo, e_hi, 32):
                ee = j + lane
                valid = ee < e_hi
                key = Int32(0x10000) + lane
                if valid:
                    key = (lds_b32(a_list + ee * 4) >> 22) & 0x3FF
                peers = match_any(key)
                cur = Int32(0)
                if valid:
                    cur = lds_u16(a_wcw + key * 2)
                    sts_b32(a_rank + ee * 4, cur + _popc(peers & lmask))
                syncwarp()
                if valid & (bfind(peers) == lane):
                    sts_u16(a_wcw + key * 2, cur + _popc(peers))
                syncwarp()
            cute.arch.barrier()
            # pass 2: per-row stream lengths, exclusive scan, per-(row, warp) bases
            rpt = (nrows + K2T - 1) // K2T
            rr0 = cutlass.min(tid * rpt, nrows)
            rr1 = cutlass.min(rr0 + rpt, nrows)
            lsum = Int32(0)
            for r in cutlass.range(rr0, rr1):
                tr = Int32(0)
                for w in cutlass.range_constexpr(W):
                    tr = tr + lds_u16(a_wc + (w * BSZMAX + r) * 2)
                if cutlass.const_expr(self.mode == MODE_LAST):
                    tr = Int32(tr > 0)
                elif cutlass.const_expr(self.mode == MODE_ACC_ADD):
                    tr = tr + final
                lsum = lsum + tr
            incl = lsum
            for off in (1, 2, 4, 8, 16):
                y = Int32(cute.arch.shuffle_sync_up(incl, offset=off))
                if lane >= off:
                    incl = incl + y
            if lane == 31:
                sts_b32(a_t + warp * 4, incl)
            cute.arch.barrier()
            run = incl - lsum
            rtot = Int32(0)
            for w in cutlass.range_constexpr(W):
                tw = lds_b32(a_t + w * 4)
                if w < warp:
                    run = run + tw
                rtot = rtot + tw
            for r in cutlass.range(rr0, rr1):
                sts_b32(a_roffs + r * 4, run)
                tr = Int32(0)
                if cutlass.const_expr(self.mode == MODE_LAST):
                    # only the row's last entry is streamed: keep per-warp prefixes and the total
                    wrun = Int32(0)
                    for w in cutlass.range_constexpr(W):
                        hv = lds_u16(a_wc + (w * BSZMAX + r) * 2)
                        sts_u16(a_wc + (w * BSZMAX + r) * 2, wrun)
                        wrun = wrun + hv
                    sts_b32(a_rcnt + r * 4, wrun)
                    run = run + Int32(wrun > 0)
                else:
                    wrun = run
                    for w in cutlass.range_constexpr(W):
                        hv = lds_u16(a_wc + (w * BSZMAX + r) * 2)
                        sts_u16(a_wc + (w * BSZMAX + r) * 2, wrun)
                        wrun = wrun + hv
                        tr = tr + hv
                    if cutlass.const_expr(self.mode == MODE_ACC_ADD):
                        if final != 0:
                            sts_b32(a_stream + wrun * 4, -1 - r)
                            tr = tr + 1
                    run = run + tr
            if tid == 0:
                sts_b32(a_roffs + nrows * 4, rtot)
            cute.arch.barrier()
            # pass 3: scatter positions into the per-row streams
            for j in cutlass.range(e_lo, e_hi, 32):
                ee = j + lane
                if ee < e_hi:
                    ent = lds_b32(a_list + ee * 4)
                    rl_ = (ent >> 22) & 0x3FF
                    slot = lds_u16(a_wcw + rl_ * 2) + lds_b32(a_rank + ee * 4)
                    if cutlass.const_expr(self.mode == MODE_LAST):
                        # row rank of this entry; only the last duplicate is written
                        if slot == lds_b32(a_rcnt + rl_ * 4) - 1:
                            sts_b32(a_stream + lds_b32(a_roffs + rl_ * 4) * 4, ent & 0x3FFFFF)
                    else:
                        sts_b32(a_stream + slot * 4, ent & 0x3FFFFF)
            cute.arch.barrier()

            # rows: warp w owns a contiguous block, bulk-copy ring
            bw = (nrows + W - 1) // W
            r0 = cutlass.min(warp * bw, nrows)
            r1 = cutlass.min(r0 + bw, nrows)
            e0 = lds_b32(a_roffs + r0 * 4)
            e1 = lds_b32(a_roffs + r1 * 4)
            if lane == 0:
                for k in cutlass.range_constexpr(STAGES):
                    if e0 + k < e1:
                        code = lds_b32(a_stream + (e0 + k) * 4)
                        gsrc = cutlass.select_(code >= 0, p_src + Int64(code) * ROWB,
                                               p_add + Int64(row0 - 1 - code) * ROWB)
                        sl = (q3 + k) % STAGES
                        bulk_load(a_mb + sl * 8, a_ring + sl * ROWB, gsrc, ROWB)
            kp = cutlass.min(e0 + STAGES, e1)
            for r in cutlass.range(r0, r1):
                t = row0 + r
                obase = p_out + Int64(t) * ROWB
                nrow = lds_b32(a_roffs + (r + 1) * 4) - lds_b32(a_roffs + r * 4)
                acc = cute.make_rmem_tensor((4 * NJ,), Uint32)
                hp = Int32(0)
                if cutlass.const_expr(self.mode != MODE_LAST):
                    hp = lds_u8(a_hp + r)
                    for jj in cutlass.range_constexpr(NJ):
                        for k in cutlass.range_constexpr(4):
                            acc[4 * jj + k] = Uint32(0)
                    if hp != 0:
                        for jj in cutlass.range_constexpr(NJ):
                            off = voff_last if jj == NJ - 1 else (lane + 32 * jj) * 16
                            w4 = ld_v4(obase + Int64(off))
                            for k in cutlass.range_constexpr(4):
                                acc[4 * jj + k] = w4[k]
                for kk in cutlass.range(nrow):
                    sl = q3 % STAGES
                    mbar_wait(a_mb + sl * 8, (q3 // STAGES) % 2)
                    sbase = a_ring + sl * ROWB
                    for jj in cutlass.range_constexpr(NJ):
                        off = voff_last if jj == NJ - 1 else (lane + 32 * jj) * 16
                        w4 = lds_v4(sbase + off)
                        for k in cutlass.range_constexpr(4):
                            if cutlass.const_expr(self.mode == MODE_LAST):
                                acc[4 * jj + k] = bmul1(w4[k])
                            else:
                                acc[4 * jj + k] = badd(acc[4 * jj + k], w4[k])
                    syncwarp()
                    if lane == 0:
                        if kp < e1:
                            code = lds_b32(a_stream + kp * 4)
                            gsrc = cutlass.select_(code >= 0, p_src + Int64(code) * ROWB,
                                                   p_add + Int64(row0 - 1 - code) * ROWB)
                            bulk_load(a_mb + sl * 8, sbase, gsrc, ROWB)
                    kp = cutlass.min(kp + 1, e1)
                    q3 = q3 + 1
                if cutlass.const_expr(self.mode == MODE_LAST):
                    do_store = nrow > 0
                elif cutlass.const_expr(self.mode == MODE_ACC_ADD):
                    do_store = (nrow > 0) | (final != 0)
                else:
                    do_store = (nrow > 0) | ((final != 0) & (hp == 0))
                if do_store:
                    for jj in cutlass.range_constexpr(NJ):
                        w4 = [acc[4 * jj + k] for k in range(4)]
                        if cutlass.const_expr(jj < NJ - 1):
                            st_v4(obase + Int64((lane + 32 * jj) * 16), w4)
                        else:
                            st_v4_pred(obase + Int64(voff_last), w4, last_ok)
                if cutlass.const_expr(self.mode != MODE_LAST):
                    if (final == 0) & (nrow > 0) & (lane == 0):
                        sts_u8(a_hp + r, 1)
            c0 = c1
            done = final
            cute.arch.barrier()




_I64 = cutlass.Int64


def _compile(mode, nsm):
    return cute.compile(
        RowScatterCoop(mode, nsm),
        cute.runtime.make_fake_tensor(cutlass.Uint8, (STATE_BYTES,), stride=(1,), assumed_align=16),
        _I64(0), _I64(0), _I64(0), _I64(0), _I64(0), Int32(1), Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def fits(m, n, g):
    """Whether (m rows, n indices) on g SMs is within the kernel's limits."""
    return g <= GMAX and n <= g * SA_MAX and n < (1 << 22) and (m + g - 1) // g <= BSZMAX

_MODE = MODE_ACC


_TBF = torch.bfloat16
_TI64 = torch.int64
_FNS = {}
_STATE = {}


def _fn_for(nsm):
    fn = _FNS.get(nsm)
    if fn is None:
        fn = _FNS[nsm] = _compile(_MODE, nsm)
    return fn


def _setup(dev):
    """Per-device (state tensor, its address, SM count, compiled launcher); the state (barrier
    word + two workspaces) is zeroed once, with one host sync, at the first call on a device."""
    st = _STATE.get(dev)
    if st is None:
        nsm = torch.cuda.get_device_properties(dev).multi_processor_count
        state = torch.zeros((STATE_BYTES,), dtype=torch.uint8, device=dev)
        torch.cuda.synchronize(dev)
        st = _STATE[dev] = (state, state.data_ptr(), nsm, _fn_for(nsm))
    return st


def _empty(shape, dev):
    """torch.empty without the deterministic NaN fill (the kernel writes every element)."""
    if torch.utils.deterministic.fill_uninitialized_memory:
        torch.utils.deterministic.fill_uninitialized_memory = False
        try:
            return torch.empty(shape, dtype=_TBF, device=dev)
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = True
    return torch.empty(shape, dtype=_TBF, device=dev)


def _rows16(x):
    """Rows of 2880 bf16 at stride 2880 starting 16 B aligned (cp.async.bulk); else a dense copy."""
    if x.stride(1) == 1 and (x.shape[0] <= 1 or x.stride(0) == D) and x.data_ptr() % 16 == 0:
        return x
    return x.clone(memory_format=torch.contiguous_format)


def _idx1(x):
    return x if x.shape[0] <= 1 or x.stride(0) == 1 else x.contiguous()


_fn_for(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)




def _reference(in0, in1, in2):
    v3 = torch.ops.aten.new_zeros.default(in0, [in1, D], dtype=torch.bfloat16, layout=torch.strided,
                                          device=torch.device("cuda"))
    return torch.ops.aten.index_put.default(v3, [in2], in0, True)


def run(in0, in1, in2):
    st = _STATE.get(in0.device)
    n = in2.shape[0]
    if (st is not None and in0.dtype == _TBF and in2.dtype == _TI64 and in0.shape == (n, D) and in0.stride() == (D, 1)
            and in2.stride() == (1,) and in2.device == in0.device and in0.data_ptr() & 15 == 0
            and type(in1) is int and in1 > 0 and n > 0 and in0.device.index == torch.cuda.current_device()
            and fits(in1, n, st[2])):
        out = _empty((in1, D), in0.device)  # every row is written (rows without a source: +0)
        st[3](st[0], in0.data_ptr(), in2.data_ptr(), out.data_ptr(), out.data_ptr(), st[1], in1, n)
        return out
    return _general(in0, in1, in2)


def _general(in0, in1, in2):
    """First call on a device, unusual layouts, empty inputs, and out-of-limit sizes."""
    if not (isinstance(in0, torch.Tensor) and isinstance(in2, torch.Tensor) and isinstance(in1, int)
            and in0.dtype == _TBF and in2.dtype == _TI64 and in0.is_cuda and in0.dim() == 2 and in0.shape[1] == D
            and in2.dim() == 1 and in2.shape[0] == in0.shape[0] and in1 >= 0 and in2.device == in0.device
            and in0.device.index == torch.cuda.current_device()):
        return _reference(in0, in1, in2)
    n = in2.shape[0]
    st = _setup(in0.device)
    if not fits(in1, n, st[2]):
        return _reference(in0, in1, in2)
    out = _empty((in1, D), in0.device)
    if n == 0 and in1 == 0:
        return out
    in0, in2 = _rows16(in0), _idx1(in2)
    st[3](st[0], in0.data_ptr(), in2.data_ptr(), out.data_ptr(), out.data_ptr(), st[1], in1, n)
    return out
