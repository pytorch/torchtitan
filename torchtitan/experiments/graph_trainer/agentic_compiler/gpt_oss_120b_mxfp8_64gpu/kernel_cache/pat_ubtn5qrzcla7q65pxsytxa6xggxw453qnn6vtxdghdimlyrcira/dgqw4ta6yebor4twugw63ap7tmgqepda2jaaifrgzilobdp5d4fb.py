"""G-014 (GPT-OSS 120B MXFP8) MoE combine forward: gather of the top-k scores, scale of the
routed rows, zeros_like + torchtitan::deterministic_scatter_add; bitwise equal to reference.py.

Reference semantics:
  v7 = in0.reshape(-1)[in1].reshape(-1, 1): aten.index (negatives wrap, out of range
    device-asserts), raw fp32 bits; output [N, 1] with stride [1, 1].
  v9 = bf16(float(in2) * v7): mul.Tensor(bf16, f32) promotes to f32 (one RNE multiply, denormals
    kept), then _to_copy RNE to bf16.
  v11 = deterministic_scatter_add(zeros_like(in3), in4, v9) runs scatter_add under
    use_deterministic_algorithms(True). in4 is stride 0 along dim 1 (an expanded token index), so
    _scatter_via_index_put takes its broadcast path, index_put_(out, [in4[:, 0]], v9,
    accumulate=True) -> index_put_with_sort -> indexing_backward_kernel<bf16, 4> (slice 2880 >
    32): stable radix sort of the wrapped row index, so duplicates come in ascending source
    position, and each duplicate does w = bf16(float(w) + float(v9)) from w = +0 (rows with no
    source stay +0). A dense in4 (stride(1) != 0) takes the two-index path with different
    numerics; that layout runs the reference ops here.
Here, per source row in position order: acc = add.rn.bf16x2(acc, cvt.rn.bf16x2.f32(
mul.rn.f32(float(x), s))) from +0 (add.rn.bf16x2 == torch's bf16 add for all 2^32 pairs: sol's
exhaustive test; -0 -> +0 and NaN -> 0x7fff on the first add, so v9's NaN payload never
matters).

Kernel (one cooperative CuTe DSL launch, one CTA of 8 warps per SM, CTA o owns destination
rows [o*B, o*B + B), B = ceil(M / #SM)):
  phase A: CTA c takes positions [c*S, c*S + S) (S = ceil(N / #SM)): range-checks and wraps
    in1 and in4[:, 0] (out of range -> trap, as the reference's device asserts), writes
    v7[p] = in0.flat[in1[p]], and stable-sorts the slice by owner CTA (match.any ranks per warp,
    block scan) into a per-call workspace as (row << 22 | position, score) pairs.
  grid barrier: persistent per-device monotonic 64-bit counter (one barrier per launch, #SM
    CTAs per launch, so it is a multiple of #SM between launches; no per-call zeroing). The
    launch generation (counter / #SM, the same for every CTA of a launch) selects one of two
    persistent workspaces: launch k+2 can only start once every CTA of launch k has exited
    (one CTA per SM, cooperative), so a workspace is never shared by running launches.
  phase B: CTA o gathers its entries slice by slice (= ascending source position), builds
    per-row source streams with their scores (stable counting sort by row: match.any ranks per
    warp, block scan), then each warp streams its rows' in2 rows through a 4-deep
    cp.async.bulk ring (5760 B rows into smem, mbarrier completion), scales and accumulates in
    registers and stores each destination row once.
  If a CTA's entries exceed the smem list (heavy duplicates), it works in windows of slices and
  rows carry their bf16 partial through the output (exact: the state is bf16 after every add).
No host syncs (the per-device state is zeroed once, at the first call on a device). Limits (else
the reference ops run, which is exact by definition): N < 2^22, N <= 160 * 1024,
M <= 512 * #SM, in0.numel() < 2^31.
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
PMAX = GMAX  # index slices (one per CTA)
SA_MAX = 1024  # index positions per slice (n <= GMAX * SA_MAX)
IT_MAX = SA_MAX // K2T  # positions per thread in phase A
BSZMAX = 512  # destination rows per CTA
LCAP = 1024  # smem list entries (>= SA_MAX)
WS_BYTES = (GMAX * SA_MAX * 2 + GMAX * (GMAX + 1) + 64) * 4  # one per-launch workspace
STATE_BYTES = 256 + 2 * WS_BYTES  # barrier word, then two workspaces


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
def ld_acquire_u64(addr, *, loc=None, ip=None):
    return Int64(_asm1(T.i64(), [Int64(addr)], "ld.acquire.gpu.u64 $0, [$1];", "=l,l", loc, ip, side=True))


@dsl_user_op
def st_v2(addr, a, b, *, loc=None, ip=None):
    _asm1(None, [Int64(addr), Int32(a), Int32(b)], "st.global.v2.b32 [$0], {$1, $2};", "l,r,r", loc, ip,
          side=True)


@dsl_user_op
def ld_v2_pred(addr, pred, *, loc=None, ip=None):
    r = _asmN([T.i32()] * 2, [Int64(addr), Int32(pred)],
              "{ .reg .pred p; setp.ne.s32 p, $3, 0; mov.b32 $0, 0; mov.b32 $1, 0;\n"
              "@p ld.global.v2.b32 {$0, $1}, [$2]; }", "=r,=r,l,r", loc, ip, side=True)
    return Int32(r[0]), Int32(r[1])


@dsl_user_op
def ld_nc_b32_pred(addr, pred, *, loc=None, ip=None):
    return Int32(_asm1(T.i32(), [Int64(addr), Int32(pred)],
                       "{ .reg .pred p; setp.ne.s32 p, $2, 0; mov.b32 $0, 0; @p ld.global.nc.b32 $0, [$1]; }",
                       "=r,l,r", loc, ip, side=True))


@dsl_user_op
def bscale_add(acc, x, s, *, loc=None, ip=None):
    """Per bf16 half: acc + bf16(float(x) * s), fp32 product (RNE), RNE to bf16, bf16 add (RNE;
    == bf16(float(acc) + float(v)), -0 -> +0 and NaN -> 0x7fff on the first add)."""
    return Uint32(_asm1(T.i32(), [Uint32(acc), Uint32(x), Uint32(s)],
                        "{ .reg .b32 lo, hi, v;\n"
                        "shl.b32 lo, $2, 16; and.b32 hi, $2, 0xFFFF0000;\n"
                        "mul.rn.f32 lo, lo, $3; mul.rn.f32 hi, hi, $3;\n"
                        "cvt.rn.bf16x2.f32 v, hi, lo;\n"
                        "add.rn.bf16x2 $0, $1, v; }", "=r,r,r,r", loc, ip))



class CombineFwdCoop:
    """Phase A: CTA c gathers the scores of its index slice (v7) and stable-sorts the slice by
    owner CTA (rows [o*B, o*B + B)) into the workspace as (entry, score) pairs; grid barrier;
    phase B: CTA o gathers its entries slice by slice (position order), builds per-row source
    streams and streams source rows through a cp.async.bulk ring, scaling and accumulating."""

    def __init__(self, nsm):
        self.nsm = nsm

    @cute.jit
    def __call__(self, anchor, p_src: Int64, p_idx: Int64, s_idx: Int64, p_sc: Int64, p_i1: Int64, p_o0: Int64,
                 p_out: Int64, p_state: Int64, m: Int32, n: Int32, ns: Int32, stream):
        self.kernel(p_src, p_idx, s_idx, p_sc, p_i1, p_o0, p_out, p_state, m, n, ns).launch(
            grid=[self.nsm, 1, 1], block=[K2T, 1, 1], stream=stream, cooperative=True)

    @cute.kernel
    def kernel(self, p_src: Int64, p_idx: Int64, s_idx: Int64, p_sc: Int64, p_i1: Int64, p_o0: Int64,
               p_out: Int64, p_state: Int64, m: Int32, n: Int32, ns: Int32):
        tid, _, _ = cute.arch.thread_idx()
        b, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        lane = tid % 32
        warp = tid // 32
        npch = gdim
        bcta = cutlass.max((m + npch - 1) // npch, 1)
        # launch generation = barrier count / #CTAs (the same for every CTA of a launch, see
        # grid_sync); workspaces alternate by generation, so a launch never shares its workspace
        # with a still-running previous launch
        gen = ld_acquire_u64(p_state) // Int64(self.nsm)
        p_ws = p_state + 256 + (gen % 2) * WS_BYTES
        sa = (n + npch - 1) // npch
        row0 = b * bcta
        nrows = cutlass.max(cutlass.min(bcta, m - row0), 0)
        p_starts = p_ws + Int64(npch * sa) * 8

        smem = cutlass.utils.SmemAllocator()
        s_ring = smem.allocate_tensor(cutlass.Uint8, cute.make_layout(W * STAGES * ROWB), byte_alignment=128)
        s_list = smem.allocate_tensor(cutlass.Int32, cute.make_layout(LCAP), byte_alignment=16)
        s_lsc = smem.allocate_tensor(cutlass.Int32, cute.make_layout(LCAP), byte_alignment=16)
        s_stream = smem.allocate_tensor(cutlass.Int32, cute.make_layout(LCAP), byte_alignment=16)
        s_ssc = smem.allocate_tensor(cutlass.Int32, cute.make_layout(LCAP), byte_alignment=16)
        s_ccnt = smem.allocate_tensor(cutlass.Int32, cute.make_layout(PMAX), byte_alignment=16)
        s_cbase = smem.allocate_tensor(cutlass.Int32, cute.make_layout(PMAX), byte_alignment=16)
        s_coffs = smem.allocate_tensor(cutlass.Int32, cute.make_layout(PMAX + 4), byte_alignment=16)
        s_roffs = smem.allocate_tensor(cutlass.Int32, cute.make_layout(BSZMAX + 4), byte_alignment=16)
        s_hp = smem.allocate_tensor(cutlass.Int8, cute.make_layout(BSZMAX), byte_alignment=16)
        s_mbar = smem.allocate_tensor(cutlass.Int64, cute.make_layout(W * STAGES), byte_alignment=16)
        s_h = smem.allocate_tensor(cutlass.Int32, cute.make_layout(max(W * GMAX, W * BSZMAX // 2)), byte_alignment=16)
        s_rank = smem.allocate_tensor(cutlass.Int32, cute.make_layout(LCAP), byte_alignment=16)
        s_t = smem.allocate_tensor(cutlass.Int32, cute.make_layout(W + 4), byte_alignment=16)
        a_ring = s_ring.iterator.toint() + warp * (STAGES * ROWB)
        a_list = s_list.iterator.toint()
        a_lsc = s_lsc.iterator.toint()
        a_stream = s_stream.iterator.toint()
        a_ssc = s_ssc.iterator.toint()
        a_ccnt = s_ccnt.iterator.toint()
        a_cbase = s_cbase.iterator.toint()
        a_coffs = s_coffs.iterator.toint()
        a_roffs = s_roffs.iterator.toint()
        a_hp = s_hp.iterator.toint()
        a_mb = s_mbar.iterator.toint() + warp * (STAGES * 8)
        a_h = s_h.iterator.toint()
        a_rank = s_rank.iterator.toint()
        a_wc = a_h  # phase B per-(warp, row) u16 counters / bases, aliasing the phase A histogram
        a_t = s_t.iterator.toint()
        lmask = lanemask_lt()

        # ================= phase A: this CTA's slice [b*sa, b*sa + sa)
        for w in cutlass.range_constexpr(W):
            for o in cutlass.range(tid, npch, K2T):
                sts_b32(a_h + (w * GMAX + o) * 4, 0)
        sw = (sa + W - 1) // W
        send = cutlass.min(b * sa + sa, n)
        base = b * sa + warp * sw
        valids = []
        ilos = []
        ihis = []
        jlos = []
        jhis = []
        # all index loads in flight before the first use
        for it in cutlass.range_constexpr(IT_MAX):
            pos = base + it * 32 + lane  # noqa
            vld = Int32((it * 32 + lane < sw) & (pos < send))
            ilo, ihi = ld_idx_pred(p_idx + Int64(pos) * s_idx, vld)
            jlo, jhi = ld_idx_pred(p_i1 + Int64(pos) * 8, vld)
            valids.append(vld)
            ilos.append(ilo)
            ihis.append(ihi)
            jlos.append(jlo)
            jhis.append(jhi)
        os_ = []
        rls = []
        js = []
        for it in cutlass.range_constexpr(IT_MAX):
            t = check_wrap(ilos[it], ihis[it], valids[it], m)
            js.append(check_wrap(jlos[it], jhis[it], valids[it], ns))
            o = t // bcta
            os_.append(o)
            rls.append(t - o * bcta)
        scs = []
        for it in cutlass.range_constexpr(IT_MAX):
            scs.append(ld_nc_b32_pred(p_sc + Int64(js[it]) * 4, valids[it]))
        for it in cutlass.range_constexpr(IT_MAX):
            if valids[it] != 0:
                st_b32(p_o0 + Int64(base + it * 32 + lane) * 4, scs[it])
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
                st_v2(p_ws + (Int64(b) * sa + slot) * 8, (rls[it] << 22) | (base + it * 32 + lane), scs[it])
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
                gvs.append(ld_v2_pred(p_ws + Int64(gsrc_e) * 8, Int32(e < nl)))
            for k in cutlass.range_constexpr(LCAP // K2T):
                e = tid + k * K2T
                if e < nl:
                    sts_b32(a_list + e * 4, gvs[k][0])
                    sts_b32(a_lsc + e * 4, gvs[k][1])
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
                for w in cutlass.range_constexpr(W):
                    hv = lds_u16(a_wc + (w * BSZMAX + r) * 2)
                    sts_u16(a_wc + (w * BSZMAX + r) * 2, run)
                    run = run + hv
            if tid == 0:
                sts_b32(a_roffs + nrows * 4, rtot)
            cute.arch.barrier()
            # pass 3: scatter positions and scores into the per-row streams
            for j in cutlass.range(e_lo, e_hi, 32):
                ee = j + lane
                if ee < e_hi:
                    ent = lds_b32(a_list + ee * 4)
                    rl_ = (ent >> 22) & 0x3FF
                    slot = lds_u16(a_wcw + rl_ * 2) + lds_b32(a_rank + ee * 4)
                    sts_b32(a_stream + slot * 4, ent & 0x3FFFFF)
                    sts_b32(a_ssc + slot * 4, lds_b32(a_lsc + ee * 4))
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
                        sl = (q3 + k) % STAGES
                        bulk_load(a_mb + sl * 8, a_ring + sl * ROWB, p_src + Int64(code) * ROWB, ROWB)
            kp = cutlass.min(e0 + STAGES, e1)
            for r in cutlass.range(r0, r1):
                t = row0 + r
                obase = p_out + Int64(t) * ROWB
                rs = lds_b32(a_roffs + r * 4)
                nrow = lds_b32(a_roffs + (r + 1) * 4) - rs
                acc = cute.make_rmem_tensor((4 * NJ,), Uint32)
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
                    sc = Uint32(lds_b32(a_ssc + (rs + kk) * 4))
                    mbar_wait(a_mb + sl * 8, (q3 // STAGES) % 2)
                    sbase = a_ring + sl * ROWB
                    for jj in cutlass.range_constexpr(NJ):
                        off = voff_last if jj == NJ - 1 else (lane + 32 * jj) * 16
                        w4 = lds_v4(sbase + off)
                        for k in cutlass.range_constexpr(4):
                            acc[4 * jj + k] = bscale_add(acc[4 * jj + k], w4[k], sc)
                    syncwarp()
                    if lane == 0:
                        if kp < e1:
                            code = lds_b32(a_stream + kp * 4)
                            bulk_load(a_mb + sl * 8, sbase, p_src + Int64(code) * ROWB, ROWB)
                    kp = cutlass.min(kp + 1, e1)
                    q3 = q3 + 1
                if (nrow > 0) | ((final != 0) & (hp == 0)):
                    for jj in cutlass.range_constexpr(NJ):
                        w4 = [acc[4 * jj + k] for k in range(4)]
                        if cutlass.const_expr(jj < NJ - 1):
                            st_v4(obase + Int64((lane + 32 * jj) * 16), w4)
                        else:
                            st_v4_pred(obase + Int64(voff_last), w4, last_ok)
                if (final == 0) & (nrow > 0) & (lane == 0):
                    sts_u8(a_hp + r, 1)
            c0 = c1
            done = final
            cute.arch.barrier()


_I64 = cutlass.Int64


def _compile(nsm):
    return cute.compile(
        CombineFwdCoop(nsm),
        cute.runtime.make_fake_tensor(cutlass.Uint8, (STATE_BYTES,), stride=(1,), assumed_align=16),
        _I64(0), _I64(0), _I64(0), _I64(0), _I64(0), _I64(0), _I64(0), _I64(0), Int32(1), Int32(1), Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def fits(m, n, g):
    """Whether (m rows, n indices) on g SMs is within the kernel's limits."""
    return g <= GMAX and n <= g * SA_MAX and n < (1 << 22) and (m + g - 1) // g <= BSZMAX


_TBF = torch.bfloat16
_TF32 = torch.float32
_TI64 = torch.int64
_FNS = {}
_STATE = {}


def _fn_for(nsm):
    fn = _FNS.get(nsm)
    if fn is None:
        fn = _FNS[nsm] = _compile(nsm)
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


def _empty(shape, stride, dtype, dev):
    """torch.empty_strided without the deterministic NaN fill (the kernel writes every element)."""
    if torch.utils.deterministic.fill_uninitialized_memory:
        torch.utils.deterministic.fill_uninitialized_memory = False
        try:
            return torch.empty_strided(shape, stride, dtype=dtype, device=dev)
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = True
    return torch.empty_strided(shape, stride, dtype=dtype, device=dev)


def _reference(in0, in1, in2, in3, in4):
    import torchtitan.ops.scatter_add  # noqa: F401  (torchtitan::deterministic_scatter_add)

    v5 = torch.ops.aten.reshape.default(in0, [-1])
    v6 = torch.ops.aten.index.Tensor(v5, [in1])
    v7 = torch.ops.aten.reshape.default(v6, [-1, 1])
    v8 = torch.ops.aten.mul.Tensor(in2, v7)
    v9 = torch.ops.aten._to_copy.default(v8, dtype=torch.bfloat16)
    v10 = torch.ops.aten.zeros_like.default(in3, pin_memory=False)
    v11 = torch.ops.torchtitan.deterministic_scatter_add.default(v10, in4, v9)
    return (v7, v11)


def run(in0, in1, in2, in3, in4):
    st = _STATE.get(in0.device)
    n = in1.shape[0]
    if (st is not None and in0.dtype == _TF32 and in1.dtype == _TI64 and in2.dtype == _TBF and in3.dtype == _TBF
            and in4.dtype == _TI64 and in0.is_contiguous() and in1.stride() == (1,) and in2.shape == (n, D)
            and in2.stride() == (D, 1) and in3.dim() == 2 and in3.shape[1] == D and in3.is_contiguous()
            and in4.shape == (n, D) and in4.stride(1) == 0 and n > 0 and 0 < in0.numel() < (1 << 31)
            and in1.device == in0.device and in2.device == in0.device and in4.device == in0.device
            and in3.device == in0.device and in2.data_ptr() & 15 == 0
            and in0.device.index == torch.cuda.current_device() and fits(in3.shape[0], n, st[2])):
        return _launch(st, in0, in1, in2, in3, in4)
    return _general(in0, in1, in2, in3, in4)


def _launch(st, in0, in1, in2, in3, in4):
    n, m = in1.shape[0], in3.shape[0]
    dev = in0.device
    v7 = _empty((n, 1), (1, 1), _TF32, dev)
    v11 = _empty((m, D), (D, 1), _TBF, dev)
    st[3](st[0], in2.data_ptr(), in4.data_ptr(), in4.stride(0) * 8, in0.data_ptr(), in1.data_ptr(), v7.data_ptr(),
          v11.data_ptr(), st[1], m, n, in0.numel())
    return (v7, v11)


def _general(in0, in1, in2, in3, in4):
    """First call on a device, unusual layouts, broadcasting, empty and out-of-limit sizes."""
    if not (all(isinstance(x, torch.Tensor) for x in (in0, in1, in2, in3, in4)) and in0.is_cuda
            and in0.dtype == _TF32 and in1.dtype == _TI64 and in2.dtype == _TBF and in3.dtype == _TBF
            and in4.dtype == _TI64 and in1.dim() == 1 and in2.dim() == 2 and in3.dim() == 2 and in4.dim() == 2
            and in1.shape[0] > 0 and tuple(in2.shape) == (in1.shape[0], D) and in3.shape[1] == D
            and in3.is_contiguous() and tuple(in4.shape) == (in1.shape[0], D) and in4.stride(1) == 0
            and 0 < in0.numel() < (1 << 31)
            and all(x.device == in0.device for x in (in1, in2, in3, in4))
            and in0.device.index == torch.cuda.current_device()):
        return _reference(in0, in1, in2, in3, in4)
    st = _setup(in0.device)
    if not fits(in3.shape[0], in1.shape[0], st[2]):
        return _reference(in0, in1, in2, in3, in4)
    in0 = in0.contiguous()
    in1 = in1.contiguous()
    if not (in2.stride() == (D, 1) and in2.data_ptr() % 16 == 0):
        in2 = in2.clone(memory_format=torch.contiguous_format)
    return _launch(st, in0, in1, in2, in3, in4)


_fn_for(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
