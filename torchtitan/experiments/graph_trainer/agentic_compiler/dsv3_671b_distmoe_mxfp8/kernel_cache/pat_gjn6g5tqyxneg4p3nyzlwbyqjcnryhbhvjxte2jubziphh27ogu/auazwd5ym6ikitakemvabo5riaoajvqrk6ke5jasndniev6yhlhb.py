"""Fused MoE router forward (DSv3 671B group-limited routing, 4096 tokens x 256 experts, top-8).

CuTe DSL implementation of reference.py, bitwise identical to torch's CUDA kernels.

Two launches (both with programmatic dependent launch):
  * main kernel (128 CTAs x 512 threads, 32 token rows per CTA):
      phase 1a (all 16 warps, 2 rows each; lane l owns columns [4l, 4l+4) and [128+4l, 128+4l+4)):
        sigmoid, bias, topk keys and sigmoid values -> smem.
      phase 1b (warps 8-15, 4 rows each, same lane mapping): row L1 norm, row-normalized masked
        scores accumulated into per-warp column partials in torch's order.
      phase 2 (warps 0-7, concurrent with 1b; 8 lanes per row, lane g owns expert group g =
        columns [32g, 32g+32)): group top-2 sums, top-4 groups, masked top-8 (torch's sbtopk set
        and order), gather, normalization, dispatch mask, per-row outputs, per-CTA routing counts.
      epilogue: per-CTA column partials and packed counts -> workspace.
  * tail kernel (one cluster of 8 CTAs x 256 threads): cross-CTA column sums (v39), counts (v29),
    v44/v45, the v46 sum gathered into rank 0 over DSMEM, and the scalar aux loss v50.

topk semantics (torch sbtopk::gatherTopK, sorted=False for the two index-producing calls):
keys are TopKTypeConfig<float>::convert(v) (NaN -> 0xffffffff); the selected set is the first k
in (key desc, index asc) order; the output order is the elements with key > kth-key in index
order, then the elements with key == kth-key in index order.

Reduction orders reproduce torch's Reduce.cuh on these shapes (GB300, fp32):
  * sum(v19, -1) over 8: ((g0+g4)+(g2+g6))+((g1+g5)+(g3+g7)).
  * vector_norm(v4, 1, -1) over 256: per lane a_k = x[4l+k] + x[128+4l+k], ((a0+a1)+a2)+a3,
    then a 32-lane tree (16..1).
  * sum(v38, 0) over 4096 rows: row = w + 4 b + 256 m. A_i = sum_j over m = i + 4 j (sequential),
    P = ((A0+A1)+A2)+A3, C_b = (P_w0 + P_w2) + (P_w1 + P_w3), T_y = sequential over b = y + 4 k,
    result (T0+T2)+(T1+T3).
  * sum(v45) over 256: s_c = sequential over 4c..4c+3, u_l = s_l + s_{l+32}, 32-lane tree.
Counts of booleans are exact in any order.
Every float add/mul/div is an explicit round-to-nearest op (no FMA contraction); exp is the
non-fastmath expansion; 1/(1+e) is rcp.rn (== div.rn(1, .), both correctly rounded).
"""

import cutlass
import cutlass.cute as cute
import numpy as np
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir_helpers.math import RoundingMode

_RN = RoundingMode.NEAREST_EVEN
_T = 4096  # tokens
_E = 256  # experts
_K = 8  # top-k
_NB = 64  # torch's ctas_per_output for sum(v38, 0)
_NCTA = 2 * _NB  # CTA (b, p) handles row classes w in {p, p + 2}
_NW = 16  # warps per CTA
_NWP = 8  # warps 0-7 run topk (phase 2); warps 8-15 run the L1/column partials (phase 1b)
_ROWS = 32  # rows per CTA
_GS = 36  # smem words per expert group (32 keys + 4 pad: conflict-free 16B reads per octet)

_EPS20 = float(np.float32(1e-20))
_EPS12 = float(np.float32(1e-12))
_NEG_INF_KEY = 0x007FFFFF  # convert(-inf)
_X_FAST_MIN = -69.0  # x >= -69: 1 + exp(-x) < 2^100 and sigmoid(x) > 2^-100
_B_FAST_MIN = 2.0**-100  # divisor range where the div.rn fast-path sequence is exact
_B_FAST_MAX = 2.0**20
_S_FAST_MIN = float(np.array([229595136], dtype=np.int32).view(np.float32)[0])  # fp32 sigmoid(-69)
# 19-comparator sorting network for 8 keys (descending: max to the lower index).
_SORT8 = (
    ((0, 2), (1, 3), (4, 6), (5, 7)),
    ((0, 4), (1, 5), (2, 6), (3, 7)),
    ((0, 1), (2, 3), (4, 5), (6, 7)),
    ((2, 4), (3, 5)),
    ((1, 4), (3, 6)),
    ((1, 2), (3, 4), (5, 6)),
)


def _add(a, b):
    return cute.math.add(a, b, rounding=_RN)


def _mul(a, b):
    return cute.math.mul(a, b, rounding=_RN)


def _div(a, b):
    return cute.math.div(a, b, rounding=_RN)


def _fma(a, b, c):
    return cute.math.fma(a, b, c, rounding=_RN)


def _sigmoid(x):
    e = cute.math.exp(-x, fastmath=False)
    return cute.math.rcp(_add(Float32(1.0), e), rounding=_RN)


def _sel(c, a, b, ty):
    return ty(cutlass.select_(c, ty(a), ty(b)))


def _umax(a, b):
    return _sel(a > b, a, b, Uint32)


def _umin(a, b):
    return _sel(a < b, a, b, Uint32)


def _conv(v):
    """torch TopKTypeConfig<float>::convert."""
    x = v.bitcast(Uint32)
    mask = _sel((x & Uint32(0x80000000)) != Uint32(0), 0xFFFFFFFF, 0x80000000, Uint32)
    return _sel(v != v, 0xFFFFFFFF, x ^ mask, Uint32)


def _deconv(k):
    mask = _sel((k & Uint32(0x80000000)) != Uint32(0), 0x80000000, 0xFFFFFFFF, Uint32)
    return (k ^ mask).bitcast(Float32)


def _vec4(ptr, off):
    return cute.make_tensor((ptr + off).align(16), cute.make_layout(4))


def _sort8(v):
    # plain Python loops: helpers are not processed by the DSL preprocessor (trace-time unrolled)
    v = list(v)
    for st in range(len(_SORT8)):
        for ci in range(len(_SORT8[st])):
            i0, i1 = _SORT8[st][ci]
            hi = _umax(v[i0], v[i1])
            lo = _umin(v[i0], v[i1])
            v[i0] = hi
            v[i1] = lo
    return v


def _bitonic8(c):
    """Sort a bitonic sequence of 8 keys descending."""
    c = list(c)
    for si in range(3):
        sd = 4 >> si
        for i in range(8):
            if (i & sd) == 0:
                hi = _umax(c[i], c[i + sd])
                lo = _umin(c[i], c[i + sd])
                c[i] = hi
                c[i + sd] = lo
    return c


def _merge8(a, b):
    """Top-8 (sorted descending) of two descending-sorted 8-lists (multiset)."""
    return _bitonic8([_umax(a[i], b[7 - i]) for i in range(8)])


def _kth_set_bit(m, k):
    """0-based position of the k-th (1-based) set bit of m; requires 1 <= k <= popc(m)."""
    pos = Int32(0)
    for width in (16, 8, 4, 2, 1):
        c = Int32(cute.arch.popc(m & Uint32((1 << width) - 1)))
        go = c < k
        k = _sel(go, k - c, k, Int32)
        m = _sel(go, m >> Uint32(width), m, Uint32)
        pos = _sel(go, pos + width, pos, Int32)
    return pos


def _ld_dsmem_f32(addr):
    return cute.arch.inline_ptx(
        "ld.shared::cluster.f32 {$w0}, [{$r0}];", write_only_types=[Float32], read_only_args=[addr]
    )


def _st_async_f32(val, remote_addr, remote_mbar_addr):
    """st.async to a cluster peer's smem, completing 4 tx bytes on the peer's mbarrier."""
    cute.arch.inline_ptx(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [{$r0}], {$r1}, [{$r2}];",
        read_only_args=[remote_addr, val, remote_mbar_addr],
    )


def _row_of(b, p, q):
    # CTA-local row slot q in [0, 32): w = p + 2 (q // 16), m = q % 16
    return 4 * b + p + 2 * (q // 16) + 256 * (q % 16)


@cute.kernel
def _main_kernel(
    mX: cute.Tensor,
    mBias: cute.Tensor,
    mPad: cute.Tensor,
    mV18: cute.Tensor,
    mV28w: cute.Tensor,
    mV52: cute.Tensor,
    mV53: cute.Tensor,
    mWsT: cute.Tensor,
    mWsCnt: cute.Tensor,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    lane = tidx % 32
    warp = tidx // 32
    b = bidx // 2
    p = bidx % 2

    smem = cutlass.utils.SmemAllocator()
    sKey = smem.allocate_tensor(Uint32, cute.make_layout(_ROWS * 8 * _GS), byte_alignment=16)
    sS = smem.allocate_tensor(Float32, cute.make_layout((_ROWS, _E), stride=(_E, 1)), byte_alignment=16)
    sA = smem.allocate_tensor(Float32, cute.make_layout((_NWP, _E), stride=(_E, 1)), byte_alignment=16)
    sCnt = smem.allocate_tensor(Int32, cute.make_layout(_E), byte_alignment=16)

    if tidx < _E:
        sCnt[tidx] = Int32(0)

    # Trigger only after the wait: the tail reads in2/in3 before its own wait, which is safe only
    # if the tail cannot launch before this grid's predecessors have completed.
    cute.arch.griddepcontrol_wait()
    cute.arch.griddepcontrol_launch_dependents()
    cute.arch.sync_threads()  # sCnt zeroed

    px = mX.iterator
    ba = cute.make_rmem_tensor(cute.make_layout(4), Float32)
    bb = cute.make_rmem_tensor(cute.make_layout(4), Float32)
    cute.autovec_copy(_vec4(mBias.iterator, lane * 4), ba)
    cute.autovec_copy(_vec4(mBias.iterator, 128 + lane * 4), bb)
    bl = [ba[0], ba[1], ba[2], ba[3], bb[0], bb[1], bb[2], bb[3]]

    # ------------- phase 1a (all 16 warps, 2 rows each): sigmoid and topk keys -> smem -------------
    kofs_a = (lane // 8) * _GS + 4 * (lane % 8)  # group lane//8, offset within group
    kofs_b = (4 + lane // 8) * _GS + 4 * (lane % 8)
    xs = []
    for rr in cutlass.range_constexpr(2):
        q = warp + 16 * rr
        row = _row_of(b, p, q)
        fa = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        fb = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        cute.autovec_copy(_vec4(px, row * _E + lane * 4), fa)
        cute.autovec_copy(_vec4(px, row * _E + 128 + lane * 4), fb)
        xs.append((fa, fb))
    # Fast path (whole warp): every x >= -69 and not NaN, where sigmoid's rcp.rn equals the
    # MUFU.RCP + one Newton step sequence; otherwise rcp.rn.
    ok = (xs[0][0][0] >= Float32(_X_FAST_MIN))
    for rr in cutlass.range_constexpr(2):
        for k in cutlass.range_constexpr(4):
            ok = ok & (xs[rr][0][k] >= Float32(_X_FAST_MIN)) & (xs[rr][1][k] >= Float32(_X_FAST_MIN))
    allok = cute.arch.vote_all_sync(ok)
    for rr in cutlass.range_constexpr(2):
        q = warp + 16 * rr
        fa, fb = xs[rr]
        y = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        if allok:
            for k in cutlass.range_constexpr(8):
                xv = fa[k] if cutlass.const_expr(k < 4) else fb[k - 4]
                d = _add(Float32(1.0), cute.math.exp(-xv, fastmath=False))
                r = cute.arch.rcp_approx(d)
                e = _fma(-d, r, Float32(1.0))
                y[k] = _fma(r, e, r)
        else:
            for k in cutlass.range_constexpr(4):
                y[k] = _sigmoid(fa[k])
                y[4 + k] = _sigmoid(fb[k])
        sa = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        sb = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        ka = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
        kb = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
        for k in cutlass.range_constexpr(4):
            sa[k] = y[k]
            sb[k] = y[4 + k]
            ka[k] = _conv(_add(y[k], bl[k]))
            kb[k] = _conv(_add(y[4 + k], bl[4 + k]))
        cute.autovec_copy(ka, _vec4(sKey.iterator, q * (8 * _GS) + kofs_a))
        cute.autovec_copy(kb, _vec4(sKey.iterator, q * (8 * _GS) + kofs_b))
        cute.autovec_copy(sa, _vec4(sS.iterator, q * _E + lane * 4))
        cute.autovec_copy(sb, _vec4(sS.iterator, q * _E + 128 + lane * 4))
    cute.arch.sync_threads()

    # ------------- phase 1b (warps 8-15): warp (ww, i) handles rows m = i + 4 j, j = 0..3: row L1
    # norm and row-normalized masked scores accumulated in torch's column-sum order -------------
    if warp >= _NWP:
        w8 = warp - _NWP
        ww = w8 // 4
        ii = w8 % 4
        ys = []
        pads = []
        for jj in cutlass.range_constexpr(4):
            q = ww * 16 + ii + 4 * jj
            pads.append(mPad[_row_of(b, p, q)])
            y = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            cute.autovec_copy(_vec4(sS.iterator, q * _E + lane * 4), cute.make_tensor(y.iterator, cute.make_layout(4)))
            cute.autovec_copy(_vec4(sS.iterator, q * _E + 128 + lane * 4), cute.make_tensor(y.iterator + 4, cute.make_layout(4)))
            ys.append(y)
        l1 = []
        for jj in cutlass.range_constexpr(4):
            y = ys[jj]
            a0 = _add(y[0], y[4])
            a1 = _add(y[1], y[5])
            a2 = _add(y[2], y[6])
            a3 = _add(y[3], y[7])
            l1.append(_add(_add(_add(a0, a1), a2), a3))
        for off in cutlass.range_constexpr(5):
            for jj in cutlass.range_constexpr(4):
                l1[jj] = _add(l1[jj], cute.arch.shuffle_sync_bfly(l1[jj], 16 >> off))
        # v38 = (v4 / v33) * notpad. div.rn's fast-path sequence with the row's refined reciprocal
        # is exact for dividends >= sigmoid(-69) and divisors in [2^-100, 2^20] (same operand set
        # as validated for router recompute); otherwise div.rn.
        acc = [None] * 8
        for jj in cutlass.range_constexpr(4):
            y = ys[jj]
            v33 = l1[jj]
            if l1[jj] < Float32(_EPS12):
                v33 = Float32(_EPS12)
            fok = (v33 >= Float32(_B_FAST_MIN)) & (v33 <= Float32(_B_FAST_MAX))
            for k in cutlass.range_constexpr(8):
                fok = fok & (y[k] >= Float32(_S_FAST_MIN))
            qv = cute.make_rmem_tensor(cute.make_layout(8), Float32)
            if cute.arch.vote_all_sync(fok):
                rcp = cute.arch.rcp_approx(v33)
                rt = _fma(-v33, rcp, Float32(1.0))
                rcp = _fma(rcp, rt, rcp)
                for k in cutlass.range_constexpr(8):
                    qq = _mul(y[k], rcp)
                    ee = _fma(-v33, qq, y[k])
                    qv[k] = _fma(rcp, ee, qq)
            else:
                for k in cutlass.range_constexpr(8):
                    qv[k] = _div(y[k], v33)
            notpad = _sel(pads[jj] != Uint8(0), 0.0, 1.0, Float32)
            for k in cutlass.range_constexpr(8):
                t = _mul(qv[k], notpad)
                if cutlass.const_expr(jj == 0):
                    acc[k] = t
                else:
                    acc[k] = _add(acc[k], t)
        fa2 = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        fb2 = cute.make_rmem_tensor(cute.make_layout(4), Float32)
        for k in cutlass.range_constexpr(4):
            fa2[k] = acc[k]
            fb2[k] = acc[4 + k]
        cute.autovec_copy(fa2, _vec4(sA.iterator, w8 * _E + lane * 4))
        cute.autovec_copy(fb2, _vec4(sA.iterator, w8 * _E + 128 + lane * 4))
    else:
        # ------------- phase 2 (warps 0-7): octet o of warp handles row slot q2 = 4 warp + o,
        # lane g = expert group g -------------
        g = lane % 8
        obase = lane - g
        q2 = 4 * warp + lane // 8
        row = _row_of(b, p, q2)
        notpadded = mPad[row] == Uint8(0)
        kbase = q2 * (8 * _GS) + g * _GS
        kr = []
        for c4 in cutlass.range_constexpr(8):
            t4 = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            cute.autovec_copy(_vec4(sKey.iterator, kbase + 4 * c4), t4)
            for k in cutlass.range_constexpr(4):
                kr.append(t4[k])

        # group top-2 (values only, multiset) -> group score key
        m1s = []
        m2s = []
        for i in cutlass.range_constexpr(16):
            m1s.append(_umax(kr[2 * i], kr[2 * i + 1]))
            m2s.append(_umin(kr[2 * i], kr[2 * i + 1]))
        for lvl in cutlass.range_constexpr(4):
            h = 8 >> lvl
            nm1 = []
            nm2 = []
            for i in cutlass.range_constexpr(h):
                a1, a2, c1, c2 = m1s[i], m2s[i], m1s[i + h], m2s[i + h]
                nm1.append(_umax(a1, c1))
                nm2.append(_umax(_umin(a1, c1), _umax(a2, c2)))
            m1s = nm1
            m2s = nm2
        sk = _conv(_add(_deconv(m1s[0]), _deconv(m2s[0])))

        # top-4 groups (set only): rank by (key desc, group asc)
        rank = Int32(0)
        for hh in cutlass.range_constexpr(8):
            G = cute.arch.shuffle_sync(sk, obase + hh)
            rank = rank + _sel((G > sk) | ((G == sk) & (Int32(hh) < g)), 1, 0, Int32)
        sel = rank < 4

        # k-th largest key T: the 128 keys of the 4 selected groups (masked keys add only -inf ties,
        # which cannot change the top-8 multiset) are split over the octet, 16 per lane: lane g takes
        # half (g % 2) of the (g // 2)-th selected group in group order.
        om = (Uint32(cute.arch.vote_ballot_sync(sel)) >> Uint32(obase)) & Uint32(0xFF)
        sg = _kth_set_bit(om, g // 2 + 1)
        cb = 32 * sg + 16 * (g % 2)  # first column of this lane's 16-key slice
        hbase = q2 * (8 * _GS) + sg * _GS + 16 * (g % 2)
        hk = []
        for c4 in cutlass.range_constexpr(4):
            t4 = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            cute.autovec_copy(_vec4(sKey.iterator, hbase + 4 * c4), t4)
            for k in cutlass.range_constexpr(4):
                hk.append(t4[k])
        lst = _merge8(_sort8(hk[0:8]), _sort8(hk[8:16]))
        for off in cutlass.range_constexpr(3):
            o = [cute.arch.shuffle_sync_bfly(lst[7 - i], 1 << off) for i in range(8)]
            lst = _bitonic8([_umax(lst[i], o[i]) for i in range(8)])
        T = lst[7]
        Gc = Int32(0)
        for i in cutlass.range_constexpr(7):
            Gc = Gc + _sel(lst[i] > T, 1, 0, Int32)
        need = _K - Gc
        jeq = g - Gc
        isg = g < Gc

        gi = Int32(0)
        if cute.arch.vote_any_sync(T == Uint32(_NEG_INF_KEY)):
            # Degenerate rows (k-th key is -inf): the == T set includes masked groups, so build
            # the masks over every group (lane g = group g, all 256 columns).
            kr2 = []
            for c4 in cutlass.range_constexpr(8):
                t4 = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
                cute.autovec_copy(_vec4(sKey.iterator, kbase + 4 * c4), t4)
                for k in cutlass.range_constexpr(4):
                    kr2.append(t4[k])
            gtb = [_sel(kr2[i] > T, 1 << i, 0, Uint32) for i in range(32)]
            eqb = [_sel(kr2[i] == T, 1 << i, 0, Uint32) for i in range(32)]
            for lvl in cutlass.range_constexpr(5):
                h = 16 >> lvl
                gtb = [gtb[i] | gtb[i + h] for i in range(h)]
                eqb = [eqb[i] | eqb[i + h] for i in range(h)]
            gtm = _sel(sel, gtb[0], 0, Uint32)
            eqm = _sel(sel, eqb[0], _sel(T == Uint32(_NEG_INF_KEY), 0xFFFFFFFF, 0, Uint32), Uint32)
            ngt = Int32(cute.arch.popc(gtm))
            neq = Int32(cute.arch.popc(eqm))
            packed = ngt | (neq << 16)
            incl = packed
            for off in cutlass.range_constexpr(3):
                dsh = 1 << off
                osh = cute.arch.shuffle_sync_up(incl, dsh)
                incl = _sel(g >= dsh, incl + osh, incl, Int32)
            excl = incl - packed
            rq = need - (excl >> 16)
            eqcut = _sel(rq < neq, Uint32(1) << Uint32(_kth_set_bit(eqm, rq + 1)), 0, Uint32) - Uint32(1)
            eqk = _sel(rq >= neq, eqm, _sel(rq <= 0, 0, eqm & eqcut, Uint32), Uint32)
            selm = gtm | eqk
            hg = Int32(0)
            he = Int32(0)
            for hh in cutlass.range_constexpr(8):
                Ih = cute.arch.shuffle_sync(incl, obase + hh)
                hg = hg + _sel((Ih & 0xFFFF) <= g, 1, 0, Int32)
                he = he + _sel((Ih >> 16) <= jeq, 1, 0, Int32)
            hsrc = _sel(isg, hg, he, Int32)
            gm = cute.arch.shuffle_sync(gtm, obase + hsrc)
            emk = cute.arch.shuffle_sync(eqm, obase + hsrc)
            ex = cute.arch.shuffle_sync(excl, obase + hsrc)
            msrc = _sel(isg, gm, emk, Uint32)
            kk = _sel(isg, g - (ex & 0xFFFF), jeq - (ex >> 16), Int32)
            gi = 32 * hsrc + _kth_set_bit(msrc, kk + 1)
            pm = _sel(notpadded, selm, 0, Uint32)
            w0 = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            w1 = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            for wq in cutlass.range_constexpr(8):
                word = (((pm >> (4 * wq)) & Uint32(0xF)) * Uint32(0x204081)) & Uint32(0x01010101)
                if cutlass.const_expr(wq < 4):
                    w0[wq] = word
                else:
                    w1[wq - 4] = word
            pw = mV28w.iterator + row * (_E // 4) + 8 * g
            cute.autovec_copy(w0, cute.make_tensor(pw.align(16), cute.make_layout(4)))
            cute.autovec_copy(w1, cute.make_tensor((pw + 4).align(16), cute.make_layout(4)))
        else:
            # Normal rows: every selected element lies in the redistributed slices, which are in
            # increasing column order over the octet's lanes.
            gtb = [_sel(hk[i] > T, 1 << i, 0, Uint32) for i in range(16)]
            eqb = [_sel(hk[i] == T, 1 << i, 0, Uint32) for i in range(16)]
            for lvl in cutlass.range_constexpr(4):
                h = 8 >> lvl
                gtb = [gtb[i] | gtb[i + h] for i in range(h)]
                eqb = [eqb[i] | eqb[i + h] for i in range(h)]
            gtm = gtb[0]
            eqm = eqb[0]
            ngt = Int32(cute.arch.popc(gtm))
            neq = Int32(cute.arch.popc(eqm))
            packed = ngt | (neq << 16)
            incl = packed
            for off in cutlass.range_constexpr(3):
                dsh = 1 << off
                osh = cute.arch.shuffle_sync_up(incl, dsh)
                incl = _sel(g >= dsh, incl + osh, incl, Int32)
            excl = incl - packed
            rq = need - (excl >> 16)
            eqcut = _sel(rq < neq, Uint32(1) << Uint32(_kth_set_bit(eqm, rq + 1)), 0, Uint32) - Uint32(1)
            eqk = _sel(rq >= neq, eqm, _sel(rq <= 0, 0, eqm & eqcut, Uint32), Uint32)
            selm = gtm | eqk
            hg = Int32(0)
            he = Int32(0)
            for hh in cutlass.range_constexpr(8):
                Ih = cute.arch.shuffle_sync(incl, obase + hh)
                hg = hg + _sel((Ih & 0xFFFF) <= g, 1, 0, Int32)
                he = he + _sel((Ih >> 16) <= jeq, 1, 0, Int32)
            hsrc = _sel(isg, hg, he, Int32)
            gm = cute.arch.shuffle_sync(gtm, obase + hsrc)
            emk = cute.arch.shuffle_sync(eqm, obase + hsrc)
            ex = cute.arch.shuffle_sync(excl, obase + hsrc)
            cbh = cute.arch.shuffle_sync(cb, obase + hsrc)
            msrc = _sel(isg, gm, emk, Uint32)
            kk = _sel(isg, g - (ex & 0xFFFF), jeq - (ex >> 16), Int32)
            gi = cbh + _kth_set_bit(msrc, kk + 1)
            # dispatch mask: this lane's 16-column slice; unselected own groups are all False
            pm = _sel(notpadded, selm, 0, Uint32)
            w0 = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
            for wq in cutlass.range_constexpr(4):
                w0[wq] = (((pm >> (4 * wq)) & Uint32(0xF)) * Uint32(0x204081)) & Uint32(0x01010101)
            cute.autovec_copy(w0, cute.make_tensor((mV28w.iterator + row * (_E // 4) + cb // 4).align(16), cute.make_layout(4)))
            if not sel:
                z = cute.make_rmem_tensor(cute.make_layout(4), Uint32)
                for wq in cutlass.range_constexpr(4):
                    z[wq] = Uint32(0)
                pz = mV28w.iterator + row * (_E // 4) + 8 * g
                cute.autovec_copy(z, cute.make_tensor(pz.align(16), cute.make_layout(4)))
                cute.autovec_copy(z, cute.make_tensor((pz + 4).align(16), cute.make_layout(4)))
        if notpadded:
            cute.arch.atomic_add(sCnt.iterator + gi, Int32(1))

        # gathered scores: normalize and write per-row outputs (lane g -> output slot g)
        gv = sS[q2, gi]
        tsum = gv
        tsum = _add(tsum, cute.arch.shuffle_sync_bfly(tsum, 4))
        tsum = _add(tsum, cute.arch.shuffle_sync_bfly(tsum, 2))
        tsum = _add(tsum, cute.arch.shuffle_sync_bfly(tsum, 1))
        v21 = _add(tsum, Float32(_EPS20))
        v23 = _mul(_div(gv, v21), Float32(2.5))
        mV18[row, g] = Int64(gi)
        if notpadded:
            mV52[row, g] = Int64(gi)
            mV53[row, g] = v23
        else:
            mV52[row, g] = Int64(-1)
            mV53[row, g] = Float32(0.0)


    cute.arch.sync_threads()

    # CTA column partial: P_ww = ((A0+A1)+A2)+A3, X = P_0 + P_1
    if tidx < _E:
        col = tidx
        P0 = _add(_add(_add(sA[0, col], sA[1, col]), sA[2, col]), sA[3, col])
        P1 = _add(_add(_add(sA[4, col], sA[5, col]), sA[6, col]), sA[7, col])
        mWsT[bidx, col] = _add(P0, P1)
    if tidx < _E // 4:
        w = sCnt[4 * tidx] | (sCnt[4 * tidx + 1] << 8) | (sCnt[4 * tidx + 2] << 16) | (sCnt[4 * tidx + 3] << 24)
        mWsCnt[bidx, tidx] = w


@cute.kernel
def _tail_kernel(
    mWsT: cute.Tensor,
    mWsCnt: cute.Tensor,
    mPadw: cute.Tensor,
    mN: cute.Tensor,
    mV29: cute.Tensor,
    mV50: cute.Tensor,
):
    """One cluster of 8 CTAs; CTA c owns columns [32c, 32c+32)."""
    tidx, _, _ = cute.arch.thread_idx()
    crank = cute.arch.block_idx_in_cluster()
    lane = tidx % 32
    warp = tidx // 32

    smem = cutlass.utils.SmemAllocator()
    sTy = smem.allocate_tensor(Float32, cute.make_layout((4, 32), stride=(32, 1)), byte_alignment=16)
    sCp = smem.allocate_tensor(Int32, cute.make_layout((4, 32), stride=(32, 1)), byte_alignment=16)
    sXall = smem.allocate_tensor(Float32, cute.make_layout(_E), byte_alignment=16)
    sNp = smem.allocate_tensor(Int32, cute.make_layout(8), byte_alignment=16)
    mbar = smem.allocate_array(Int64, 1, byte_alignment=8)

    # Off the critical path (before waiting on the main kernel): rank 0's mbarrier for the v45
    # gather, and the inputs the main kernel does not produce.
    if tidx == 0:
        cute.arch.mbarrier_init(mbar, 1)
        cute.arch.mbarrier_init_fence()
    cute.arch.cluster_arrive_relaxed()
    n = mN[0]
    pw = cute.make_rmem_tensor(cute.make_layout(4), Int32)
    cute.autovec_copy(cute.make_tensor((mPadw.iterator + 4 * tidx).align(16), cute.make_layout(4)), pw)
    s4 = pw[0] + pw[1] + pw[2] + pw[3]  # bytes are 0/1: per-byte sums <= 4
    npad = Int32((Uint32(s4) * Uint32(0x01010101)) >> Uint32(24))
    npad = cute.arch.warp_redux_sync(npad, "add")
    if lane == 0:
        sNp[warp] = npad
    if n < Int64(1):
        n = Int64(1)
    v49 = _mul(_div(Float32(1.0), Float32(n)), Float32(1.0))
    cute.arch.sync_threads()  # sNp
    # v41 = clamp_min(v40, 1e-12) with v40 = sum of the per-expert counts = 8 per non-padded row
    # (exact integers); its refined reciprocal feeds div.rn's fast-path sequence below, which is
    # exact for every (count in [0, 4096], v41 in {1e-12} U 8 * [1, 4096]) pair (checked
    # exhaustively against div.rn: cyd/scratch/micro/fastdiv.py).
    nonpad = Int32(_T)
    for k in cutlass.range_constexpr(8):
        nonpad = nonpad - sNp[k]
    v41 = Float32(_K * nonpad)
    if v41 < Float32(_EPS12):
        v41 = Float32(_EPS12)
    r41 = cute.arch.rcp_approx(v41)
    r41 = _fma(r41, _fma(-v41, r41, Float32(1.0)), r41)
    cute.arch.cluster_wait()

    cute.arch.griddepcontrol_wait()

    if tidx < 128:
        # column chains: thread (y, cc) -> T_y for column 32 c + cc
        y = tidx // 32
        col = 32 * crank + lane
        xv = cute.make_rmem_tensor(cute.make_layout(32), Float32)
        for k in cutlass.range_constexpr(16):
            bq = y + 4 * k
            xv[2 * k] = mWsT[2 * bq, col]
            xv[2 * k + 1] = mWsT[2 * bq + 1, col]
        acc = _add(xv[0], xv[1])
        for k in cutlass.range_constexpr(1, 16):
            acc = _add(acc, _add(xv[2 * k], xv[2 * k + 1]))
        sTy[y, lane] = acc
    else:
        # routing counts: thread (part, cc) sums CTAs [32 part, 32 part + 32) for column 32 c + cc
        part = (tidx - 128) // 32
        col = 32 * crank + lane
        cw = cute.make_rmem_tensor(cute.make_layout(32), Int32)
        for k in cutlass.range_constexpr(32):
            cw[k] = mWsCnt[32 * part + k, col // 4]
        sh = 8 * (lane % 4)
        cnt = Int32(0)
        for k in cutlass.range_constexpr(32):
            cnt = cnt + ((cw[k] >> sh) & 0xFF)
        sCp[part, lane] = cnt
    cute.arch.sync_threads()

    if tidx < 32:
        col = 32 * crank + lane
        v39 = _add(_add(sTy[0, lane], sTy[2, lane]), _add(sTy[1, lane], sTy[3, lane]))
        c = sCp[0, lane] + sCp[1, lane] + sCp[2, lane] + sCp[3, lane]
        cf = Float32(c)
        q43 = _mul(cf, r41)
        q43 = _fma(r41, _fma(-v41, q43, cf), q43)
        v44 = _mul(q43, Float32(256.0))
        # push v45 into rank 0's smem; rank 0's mbarrier counts the bytes
        _st_async_f32(
            _mul(v44, v39),
            Int32(cute.arch.map_dsmem_ptr(sXall.iterator + col, 0).toint()),
            Int32(cute.arch.map_dsmem_ptr(mbar, 0).toint()),
        )
        mV29[col] = Int64(c)
    if crank == 0:
        if warp == 0:
            if lane == 0:
                cute.arch.mbarrier_arrive_and_expect_tx(mbar, _E * 4)
            cute.arch.mbarrier_wait(mbar, 0)
            s0 = _add(_add(_add(sXall[4 * lane], sXall[4 * lane + 1]), sXall[4 * lane + 2]), sXall[4 * lane + 3])
            s1 = _add(
                _add(_add(sXall[128 + 4 * lane], sXall[128 + 4 * lane + 1]), sXall[128 + 4 * lane + 2]),
                sXall[128 + 4 * lane + 3],
            )
            u = _add(s0, s1)
            for off in cutlass.range_constexpr(5):
                u = _add(u, cute.arch.shuffle_sync_bfly(u, 16 >> off))
            if lane == 0:
                mV50[0] = _mul(u, v49)


@cute.jit
def _launch(
    mX: cute.Tensor,
    mBias: cute.Tensor,
    mPad: cute.Tensor,
    mPadw: cute.Tensor,
    mN: cute.Tensor,
    mV18: cute.Tensor,
    mV28w: cute.Tensor,
    mV29: cute.Tensor,
    mV50: cute.Tensor,
    mV52: cute.Tensor,
    mV53: cute.Tensor,
    mWsT: cute.Tensor,
    mWsCnt: cute.Tensor,
    stream,
):
    _main_kernel(mX, mBias, mPad, mV18, mV28w, mV52, mV53, mWsT, mWsCnt).launch(
        grid=[_NCTA, 1, 1], block=[_NW * 32, 1, 1], stream=stream, use_pdl=True
    )
    _tail_kernel(mWsT, mWsCnt, mPadw, mN, mV29, mV50).launch(
        grid=[8, 1, 1], block=[256, 1, 1], cluster=[8, 1, 1], stream=stream, use_pdl=True
    )


def _fake(dtype, shape, align=16):
    stride = []
    acc = 1
    for s in reversed(shape):
        stride.append(acc)
        acc *= s
    return cute.runtime.make_fake_tensor(
        dtype, tuple(shape), stride=tuple(reversed(stride)), assumed_align=align
    )


_COMPILED = None


def _get_compiled():
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = cute.compile(
            _launch,
            _fake(Float32, (_T, _E)),
            _fake(Float32, (_E,)),
            _fake(Uint8, (_T,)),
            _fake(Int32, (_T // 4,)),
            _fake(Int64, (1,), 8),
            _fake(Int64, (_T, _K)),
            _fake(Int32, (_T, _E // 4)),
            _fake(Int64, (_E,)),
            _fake(Float32, (1,), 4),
            _fake(Int64, (_T, _K)),
            _fake(Float32, (_T, _K)),
            _fake(Float32, (_NCTA, _E)),
            _fake(Int32, (_NCTA, _E // 4)),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    return _COMPILED


def run(in0, in1, in2, in3):
    dev = in0.device
    v18 = torch.empty((_T, _K), dtype=torch.int64, device=dev)
    v28 = torch.empty((_T, _E), dtype=torch.bool, device=dev)
    v29 = torch.empty((_E,), dtype=torch.int64, device=dev)
    v50 = torch.empty((), dtype=torch.float32, device=dev)
    v52 = torch.empty((_T, _K), dtype=torch.int64, device=dev)
    v53 = torch.empty((_T, _K), dtype=torch.float32, device=dev)
    ws_t = torch.empty((_NCTA, _E), dtype=torch.float32, device=dev)
    ws_cnt = torch.empty((_NCTA, _E // 4), dtype=torch.int32, device=dev)
    _get_compiled()(
        in0,
        in1,
        in2.view(torch.uint8),
        in2.view(torch.uint8).view(torch.int32),
        in3.view(1),
        v18,
        v28.view(torch.uint8).view(torch.int32),
        v29,
        v50.view(1),
        v52,
        v53,
        ws_t,
        ws_cnt,
    )
    return (v18, v28, v29, v50, v52, v53)
