# FP8 All-Gather on GroupedRaggedShard (block-wise quantization)

## Goal

Halve the all-gather bandwidth of a FlexShard `GroupedRaggedShard` bucket by
moving **fp8 bytes + tiny scales** across the wire instead of bf16, **without
changing numerics**. The enabling trick is to make the byte-balanced shard cut
land on **128×128 block-wise-quantization tile boundaries**, so every rank owns
*whole* tiles and can quantize its shard **locally** (no cross-rank `amax`),
producing a gathered fp8 weight that is **bit-identical** to "all-gather bf16,
then quantize."

This is a **communication** optimization for the forward/`grad_x` compute weight.
The high-precision master weight stays bf16-sharded for the optimizer; fp8 is only
materialized transiently by the unshard collective (the torchao/float8 FSDP model,
ported to FlexShard's placement contract).

Scope: weights under the **DeepSeek-V3 block-wise recipe** (128×128 weight tiles).
Activations (1×128) are quantized in the GEMM as today and are out of scope here.

## Background

**GroupedRaggedShard** (`example/ragged_shard.py:422`) plans a bucket as one
param-major flat buffer and cuts it into `world_size` byte-balanced ranges
(`_bucket_rank_layout`). Today the cut granularity is
`alignment_numel = lcm(suffix_numel)` (`_param_alignment_numel`), i.e. **whole
rows** (`suffix_numel = in_features` for a 2D weight under `dims=(0,)`). Unshard
is `prepare_unshard_bucket` → `run_prepared_unshard` (`all_gather`) →
`finish_prepared_unshard` (slice each param at `param_offset`).

**Block-wise weight quant** (torchao `triton_fp8_blockwise_weight_quant_rhs`,
`kernels.py:773`): for each **128×128 tile**, one scalar `amax = max(|tile|)` →
`scale = FP8_MAX/amax` → store **reciprocal** `1/scale` (fp32). Scale tensor shape
`(out/128, in/128)`. The GEMM consumes fp8 + reciprocal scales directly
(`dot(a,b)·a_s·b_s`), so **no dequant** is needed after the gather.

The single fact that governs everything: **a weight scale is `amax` over a full
128×128 tile**, so to compute it locally a rank must own the whole tile.

## Core invariant: align the cut to the tile

A `GroupedRaggedShard` weight is `(out, in)` sharded on **dim 0** (rows / `out`).
A 128×128 tile spans 128 rows × 128 cols. Therefore:

- **`in` (cols) is never sharded** → the `in/128` tiling is always local. Only
  the row dimension interacts with the cut.
- **Invariant:** every rank's row range must be a whole multiple of `block_m=128`
  (whole tile-rows). Then each rank owns complete 128×128 tiles → local quant →
  **identical tiles to single-device** → bit-identical numerics, zero `amax`
  communication.
- If violated, a tile straddles two ranks → its `amax` needs a cross-rank
  reduction *before* quant (the "precompute amax" pass tensorwise float8 FSDP
  needs), which both adds a collective and **changes numerics**. We **align, not
  reduce.**

## One gather serves forward and backward (square-tile transpose-invariance)

The weight is needed in **both** orientations, yet we all-gather it only **once**:

- **Forward** `y = x @ Wᵀ` → RHS = `Wᵀ` `(in, out)` column-major.
- **Backward dgrad** `grad_x = grad_output @ W` → RHS = `W` `(out, in)` column-major.
- (`grad_weight = grad_outputᵀ @ x` does not touch `W`.)

This works because the weight is quantized in **square** `128×128` tiles, which makes
quantization **commute with transpose**: tile `(i,j)` of `W` and tile `(j,i)` of `Wᵀ`
are the *same* 128×128 elements transposed, so their `amax` (hence scale) is identical.
Therefore **`fp8(Wᵀ) = fp8(W)ᵀ`** and **`scale(Wᵀ) = scale(W)ᵀ`**, *exactly*. From the
single gathered `W` (fp8 `(out, in)` row-major + scale `(out/128, in/128)`):

| pass | RHS data | RHS scale | cost |
|---|---|---|---|
| forward (`Wᵀ` col-major) | `gathered.t()` | `scale.t()` | **free** — a row-major `(out,in)` reinterpreted *is* col-major `(in,out)` = `Wᵀ` |
| backward dgrad (`W` col-major) | `gathered` transpose-copy | `scale` (direct) | one **local fp8 copy** (1 byte/elem) |

No re-quantization, **no second collective** — just a stride reinterpret (forward) and
a cheap local fp8 transpose (backward dgrad). Square is the *unique* tiling with this
property (derived in [§ Non-square tiles](#non-square-tiles-no-free-reuse)), so the
all-gathered, reused-across-passes weight should be square. torchao's non-all-gather
prototype instead re-quantizes the local bf16 weight twice
(`..._weight_quant_transposed_rhs` for forward, `..._weight_quant_rhs` for backward) —
with fp8 all-gather we gather once and transpose.

> **Attribution (corrected):** transpose-reuse is a *derivable property* of square tiles,
> **not** DeepSeek-V3's stated rationale. The DeepSeek-V3 report §3.3.2 motivates the
> 128×128 weight block (and 1×128 activation tile) by FP8's *limited dynamic range* and
> *activation outliers* (*"better accommodate outliers by adapting the scale according
> to smaller groups"*), and does not mention forward/backward reuse. It does, in §3.5.2,
> describe the cost a square weight *avoids*: a 1×128 activation must be *"dequantized,
> transposed, re-quantized into 128×1 tiles"* in backward — corroborating the mechanics
> below, but framed as a hardware ask, not a weight-tiling reason.

**Guarded in code:** `blockwise_transpose(fp8, scale, block)` returns the transposed
`(Wᵀ, scale.t())` views and **rejects any non-square tiling** (a `scale` whose shape is
not `(out/block, in/block)`), because the `fp8(Wᵀ)=fp8(W)ᵀ` identity holds *only* for
square tiles — a `1×128` (activation) or non-square weight tiling would regroup elements
under transpose and **silently corrupt numerics**.

## Non-square tiles: no free reuse

The single-gather trick is special to **square** tiles. The constraint: a scaled fp8
GEMM dequantizes **per K-block** — `Σ_k (a_k·s_a)(b_k·s_b)` only factors if the scales
are constant over each K-block — so the weight's scale must be 128-blocked along the
GEMM's contraction dim:

- forward contracts over `N = in` → weight scales must be blocked along **N**;
- backward dgrad contracts over `M = out` → weight scales must be blocked along **M**.

`128×128` is the **only** tiling blocked along *both* M and N, so one fp8 buffer is
K-correct for both passes (and transpose-reusable). Any non-square tiling is K-aligned
for one orientation only — `1×128` (along N) is forward-correct / backward-wrong;
`128×1` (along M) is backward-correct / forward-wrong. And fp8 is lossy and
tiling-specific: one fp8 buffer cannot be re-derived to another tiling without returning
to high precision. So a non-square weight **cannot serve both passes from one fp8
buffer**.

### Options for non-square (ranked)

1. **Promote the gathered weight to square 128×128 (recommended).** One gather,
   transpose-reusable; `blockwise_transpose` enforces it. `1×128` belongs to tensors
   quantized fresh per matmul (activations), which are not all-gathered as weights.
   ✅ **Implemented:** `promote_to_square_block(block_m, block_n)` (= `lcm`, the smallest
   square that tiles both — coarser scale, transpose-reusable) feeding the square
   `Fp8BlockwiseGroupedRaggedShard`.
2. **Two fp8 buffers, one per orientation.** Gather `fp8(W)` tiled `(1, block)` (forward,
   scale grouped along `in`) and `fp8(W)` tiled `(block, 1)` (backward, scale grouped
   along `out`). Correct, but needs **two** gathered buffers (see bandwidth) and two
   quant recipes. ✅ **Implemented:** `Fp8TwoOrientationGroupedRaggedShard` /
   `make_fp8_two_orientation_grouped_ragged_placement_fn` — aligns the cut to `block`
   rows, gathers both fp8 buffers + both scales, and `finish` returns the forward weight
   with `_fp8_backward` / `_scale_forward` `(out, in/block)` / `_scale_backward`
   `(out/block, in)` attached.
3. **bf16 all-gather + local per-orientation quant** (torchao prototype; universal
   fallback). Any tiling, no fp8-gather savings — route such params to plain
   `GroupedRaggedShard`.
4. **Dequant → requant** the gathered fp8 to the other tiling — avoid (double-quant
   error; needs the high-precision value to be meaningful anyway).

### Bandwidth: square vs non-square vs bf16

Bytes/elem moved across fwd+bwd. The durable fact is **square needs one gathered buffer,
non-square needs two** (the two orientations are distinct fp8 buffers that can't share):

| scheme | reshard off (gather once, keep) | reshard on (re-gather in backward) |
|---|---|---|
| bf16 | 2 B (1 gather, both passes via local re-quant) | 4 B (2 gathers) |
| fp8 **square** | **1 B** (transpose serves both) | **2 B** (2 gathers) |
| fp8 **non-square** (2 buffers) | **2 B** (both orientations) | **2 B** (1 orientation/pass) |

So non-square fp8 **is** a 2× win versus a per-pass re-gathering (**reshard-on**) bf16
baseline, but only **ties** the gather-once-and-keep (**reshard-off**) bf16 baseline.
Square wins 2× vs bf16 in *both* regimes, is ≤ non-square always (strictly 2× cheaper
reshard-off; tie reshard-on), and its smaller footprint can keep it reshard-off where
bf16 would force reshard-on (up to ~4× vs the bf16 you'd actually run). **Net: prefer
square because `square ≤ non-square`, not because non-square is bandwidth-useless.**

## Design

### 1. Alignment — the one-line change

Set the bucket alignment to a **full tile-row** instead of a row:

```
alignment_numel = block_m * suffix_numel          # = 128 * in_features
# (vs GroupedRaggedShard's current alignment_numel = suffix_numel)
```

`_bucket_rank_layout` already pads `global_numel` up to
`alignment_numel * total_units`, so every rank range becomes a multiple of
`128 * in_features` = whole tile-rows. Also require `in_features % 128 == 0`
(tile width; already a torchao kernel assertion). For mixed-shape buckets,
`alignment = lcm_i(128 * in_features_i)`; in practice keep **one weight per
bucket** (or same-`in` group) so alignment stays small.

### 2. Storage & master weight — unchanged (bf16, sharded)

`GroupedRaggedShard`'s byte storage stays **bf16** and byte-balanced. The master
weight is what the optimizer updates (reduce-scatter of grads is unchanged,
high-precision). fp8 is **never stored** — only produced in the send buffer of
the unshard. So `bucket_storage_layout`, `copy_param_to_storage`,
`prepare_reduce_grad`, `reduce_prepared_grad` are **untouched**.

### 3. fp8 unshard hooks (the core change)

A subclass `Fp8BlockwiseGroupedRaggedShard(GroupedRaggedShard)` overriding the
alignment and the three unshard methods:

- **`prepare_unshard_bucket`**: for each param, take the local bf16 shard (whole
  tile-rows) and quantize → `shard_fp8` (1 byte/elem) + `shard_scale`
  (`rows/128 × in/128` fp32 reciprocals), reusing torchao's
  `triton_fp8_blockwise_weight_quant_rhs`. Allocate **two** gathered buffers: an
  fp8 param-major bucket (`global_numel` fp8 bytes) and a scale param-major bucket
  (`(out/128)·(in/128)` fp32). Stage both into bucket-flat send views (the same
  `_make_local_bucket_view` trick, once for fp8, once for scales).
- **`run_prepared_unshard`**: `all_gather` the fp8 bucket **and** the scale
  bucket. Two collectives, or pack `[fp8 bytes | scale bytes]` per rank into one
  byte buffer and do a single `all_gather` (sizes are equal per rank under even
  units). Recommended: **two** for clarity — fp8 dominates, scales are ~0.03%.
- **`finish_prepared_unshard`**: slice each param's full fp8 at `param_offset`
  (fp8 space) and its full scales at the proportional tile offset; return a
  **Float8 pair** `(weight_fp8, weight_scale)` per param.

### 4. Scales — a parallel ragged gather

The scale tensor `(out/128, in/128)` is itself **dim-0 ragged-sharded the same
way** (each rank owns `rows/128` scale-rows). So the scale all-gather is the
*identical* `GroupedRaggedShard` layout at `1/128²` size — reuse the same
machinery on a second (tiny) bucket. No new collective logic.

### 5. Consumer — fp8-aware linear / Float8 wrapper

FlexShard's unsharded-param getter hands the module the unsharded weight (gathered
`W` fp8 + `_blockwise_scale`). A block-wise-fp8 linear (mirroring torchao
`Float8BlockwiseLinear` / `blockwise_scaled_mm`) consumes it without dequant, using
the same gathered tensor for both passes: **forward** RHS = `blockwise_transpose(W,
scale)` (the free `Wᵀ` col-major view), **backward dgrad** RHS = `W` (one local fp8
transpose-copy to col-major) — see the square-tile section above. This is the one
place FlexShard couples to an fp8 consumer; everything else is placement-internal.

### 6. Backward & reshard-after-forward

- `grad_x = grad_output @ weight` reuses the gathered fp8 weight → backward needs
  the fp8 unshard too. With `reshard_after_forward`, free the fp8 buffer after
  forward and **re-quantize+all-gather** the bf16 master in backward; tag the fp8
  `all_gather` op `MUST_RECOMPUTE` in `reshard_after_forward.py`'s
  `_FLEX_SHARD_COLLECTIVE_OPS` (same pattern as the existing collectives).
- `grad_weight` does not need the weight; the **grad reduce-scatter stays bf16/
  fp32** (master-precision). fp8 gradient reduce-scatter is **future work**.

### 7. Fusing the scale calculation (precompute + horizontal fusion)

Lesson from the FSDP/TP work (*Float8 All-Gather in FSDP and TP*, PyTorch
Composability and Scale): computing the scale **inside** the all-gather hook,
**per parameter**, is the perf trap. In per-tensor dynamic scaling each weight
needs an `amax` all-reduce — *"each float8 parameter requires 1 all-reduce"* — and
those small per-param all-reduces inside the **untraced, uncompiled** FSDP hooks
wiped out most of the 50%-bandwidth win (**1.40x → only 1.42x**). The fix was to
**hoist amax/scale out of the hooks into one pass after `optimizer.step`**
(`precompute_float8_amax`, aka `precompute_float8_dynamic_scale_for_fsdp`): a
single all-reduce for *all* float8 params at once → **horizontal fusion** of the
amax across params → **1.40x → 1.48x**.

Two things transfer to block-wise + `GroupedRaggedShard`:

1. **Precompute scales once per step, outside the unshard hooks.** FlexShard's
   eager unshard hooks are likewise not compiled; quantizing **per bucket inside**
   `prepare_unshard_bucket` means many small, uncompiled kernel launches. Instead,
   add a `precompute_blockwise_fp8(model)` called after `optimizer.step` that runs
   **one fused pass** over *all* bucket params — batched `amax` over every param's
   128×128 tiles — block-quantizes each rank's **local** shard to fp8 + reciprocal
   scales, and **caches** them. The unshard hook then only **reads the cached fp8
   + scale** and gathers: no per-bucket quant, one horizontally-fused precompute.

2. **The alignment invariant removes the all-reduce entirely — the bigger win.**
   The per-tensor recipe *needs* the amax all-reduce because the scale is global.
   Block-wise + the [Core invariant](#core-invariant-align-the-cut-to-the-tile)
   makes every 128×128 tile **owned whole by one rank**, so each tile's `amax` is
   **local** — there is **no amax collective at all** when aligned. So for
   block-wise the "fusion" is a *local* batched quant (zero communication),
   strictly better than the per-tensor case it is borrowed from.

   *Fallback (unaligned only):* if a config truly cannot align (a tile straddles
   ranks), batch the straddling-tile `amax` all-reduces into **one** collective for
   all params in the same post-`optimizer.step` precompute (the direct
   `precompute_float8_amax` port) — never per-tile inside the hook.

This keeps the **cast-before-gather ≡ gather-then-cast** equivalence the post
relies on: per-tensor it holds because amax is all-reduced to replicated;
block-wise it holds because each tile is owned whole (local amax = the true tile
amax). Either way the gathered fp8 is bit-identical to gathering bf16 then
quantizing — the precompute only changes *when/where* the (identical) scales are
computed, not their value.

## Alignment math (worked example)

Weight `(out=4096, in=2048)`, `block=128`, `N=4` ranks:

| quantity | plain GroupedRaggedShard | fp8 variant |
|---|---|---|
| `alignment_numel` | `in = 2048` (whole row) | `128·in = 262144` (whole tile-row) |
| padded global | `8,388,608` | `align_up(8388608, 262144·4) = 8,388,608` |
| rows/rank | `1024` | `1024` = **8 tile-rows** (1024/128 ✓) |
| tiles/rank | — | `8 × (2048/128=16) = 128` whole 128×128 tiles |
| shard moved | 1024·2048·**2 B** = 4 MB | 1024·2048·**1 B** = 2 MB + scales `8·16·4 B = 512 B` |

→ ~**2×** less all-gather traffic; every tile owned whole → local quant. (Non-
dividing `out`/`N` pads the last rows to a full tile-row with zeros — handled by
the existing pad, amax taken over real data.)

## Numerics & validation

- **Reference:** plain `GroupedRaggedShard` (gather full bf16) → torchao
  `triton_fp8_blockwise_weight_quant_rhs` (128×128) → `blockwise_scaled_mm`.
- **Test:** the fp8-all-gather path. Aligned ⇒ per-rank tiles == global tiles ⇒
  **bit-identical** gathered fp8 + scales ⇒ identical GEMM output.
- **CPU structural:** assert the cut lands on tile-rows (`rank_numels` multiple of
  `128·in`), and that an unaligned config raises a clear error.
- **GPU parity (world_size=2, SM90+):** end-to-end linear output + `grad_x` match
  the reference within fp8 tolerance (`atol/rtol≈2e-2`, as in
  `test/prototype/blockwise_fp8_training/_distributed_test_utils.py`).
- **Comm/byte check:** assert the gathered fp8 buffer is `0.5×` the bf16 bytes and
  the step issues the fp8 `all_gather` (not bf16).

## Memory & bandwidth

- **All-gather:** ~`0.5×` (fp8) + ~`0.03%` (scales: 1 fp32 per `128²=16384`
  elems) of the bf16 traffic.
- **At-rest:** master stays bf16-sharded, byte-balanced (unchanged) — this plan
  saves **comm, not at-rest memory**.
- **Transient:** the unsharded fp8 buffer is `0.5×` the bf16 unshard buffer.
- **Compute:** the quant moves *earlier* (pre-gather, on the smaller shard) — the
  forward would quantize the weight anyway, so it is net ~free, not extra.

## Implementation phases

1. **Alignment** — `Fp8BlockwiseGroupedRaggedShard._param_alignment_numel =
   block_m * suffix`; validate `in % 128 == 0`; CPU structural test.
2. **fp8 unshard** — quantize-in-`prepare`, fp8 + scale gathers in `run`, Float8
   pair in `finish`; reuse `triton_fp8_blockwise_weight_quant_rhs`.
3. **Scale precompute / fusion** (§7) — hoist block-quant out of the unshard hook
   into one fused `precompute_blockwise_fp8(model)` after `optimizer.step` (batched
   amax over all params' tiles, **local** — no collective when aligned); cache fp8
   + scales; the unshard hook reads the cache.
4. **Consumer** — block-wise-fp8 linear / Float8 wrapper reading `(fp8, scale)`.
5. **reshard-after-forward** — tag fp8 `all_gather` `MUST_RECOMPUTE`; the backward
   recompute reads the cached scales (no re-quant).
6. **Tests** — CPU structural + GPU parity vs bf16-gather-then-quant + byte check
   + assert the per-param amax/quant happens once/step outside the hooks.
7. **(future)** fp8 gradient reduce-scatter; per-param vs packed scale gather;
   MoE `Shard(0)` experts (each expert `(out,in)` already tile-local).

## API sketch

```python
class Fp8BlockwiseGroupedRaggedShard(GroupedRaggedShard):
    def __init__(self, dims=(0,), local_units=(1,), block_size=128):
        super().__init__(dims, local_units); self.block_size = block_size
    def _param_alignment_numel(self, named_params):
        # whole tile-rows so the byte cut never splits a 128x128 tile
        return math.lcm(*[self.block_size * self._suffix_numel(p.shape)
                          for _, p in named_params])
    # override prepare/run/finish_unshard_bucket: quantize -> gather fp8+scales
    #   -> return Float8(weight_fp8, weight_scale)

def make_fp8_blockwise_grouped_ragged_placement_fn(*, block_size=128, local_units):
    ...  # mirrors make_grouped_ragged_placement_fn
```

Plugs into `flex_shard(model, mesh, buckets)` like any other `BucketSpec`
placement; the linear that owns the param must be fp8-aware.

## Open questions & risks

- **Scale layout / major-order.** torchao stores reciprocal scales in specific
  row/col-major strides for `_scaled_mm` (`kernels.py:831,542`); preserve them on
  reassembly (don't `.contiguous()` into the wrong order).
- **Partial last tile.** `out % 128 != 0` ⇒ padded rows are zeros; amax over real
  data only — verify the padded tile's scale is harmless.
- **`reshard_after_forward` recompute** of the fp8 all-gather must be replayable
  (one bucket → one execution unit), like the Owned/Shard paths.
- **Interaction with Owned/Muon.** Orthogonal: this is a fp8 *communication*
  placement; Muon placements (`Owned`, `RaggedShard`) are about the optimizer.
- **e4m3 vs e5m2; bf16 vs fp32 master.** Use `e4m3` for weights (torchao default);
  master bf16 is sufficient for fp8 forward.
- **DTensor-free.** torchao's path is DTensor + FSDP2 hooks + `register_sharding`;
  FlexShard reimplements the equivalent in the placement contract (no DTensor).

## References

- torchao block-wise fp8 training: `torchao/prototype/blockwise_fp8_training/`
  — `kernels.py` (`triton_fp8_blockwise_weight_quant_rhs` 128×128 weight scale
  `:773`, `triton_fp8_blockwise_act_quant_lhs` 1×128 `:481`, `blockwise_scaled_mm`
  `:135`), `linear.py` (`fp8_blockwise_mm` `:98`, `Float8BlockwiseLinear` `:214`).
  Note: that prototype keeps the weight **bf16** and quantizes in-forward — **no**
  fp8 all-gather; it only supplies the scale recipe.
- `GroupedRaggedShard`: `example/ragged_shard.py:422` (`_param_alignment_numel`
  `:493`, `_bucket_rank_layout` `:517`, `prepare/run/finish_unshard` `:719-788`).
- Placement contract: `flex_shard/placement_contract.py` (unshard lifecycle),
  `flex_shard/bucket_storage.py` (`ParamInfo`, `BucketLayout`).
- Comparison context: `muon_flex_shard_placement_strategies.md` (the placement
  ladder; this fp8 work rides the same `GroupedRaggedShard` all-gather).
- *Float8 All-Gather in FSDP and TP* (Wei Feng et al., PyTorch Composability and
  Scale Workplace group, 2024) — the `fsdp_pre_all_gather` cast-before-gather
  pattern, the cast-before-gather ≡ gather-then-cast equivalence, and
  `precompute_float8_amax` (single post-`optimizer.step` all-reduce + horizontal
  amax fusion; 1.40x→1.48x). Motivates §7.
  `fb.workplace.com/groups/2038476226487590/permalink/2241628622839015/`
