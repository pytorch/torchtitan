# DeepSeek-V3 671B (MXFP8, Dist-MoE, FSDP 256 x EP 64): adjudicated fusion opportunities

All 64 cards are grouped into **47 opportunities: 47 real, 0 not real**. Total real saving: **218.9 ms per step**.
Details are in `judgement.json`.

## Where the saving is

| Area | Opportunities | ms |
|---|---|---|
| FSDP weight prep: copy-out + MXFP8 weight quantize, fwd and bwd re-unshard, temp->persistent copy_ | O-43, O-44, O-45 | 70.2 |
| FSDP and embedding glue: copy-in casts, reduce-scatter pack, RS cast + grad clone, embedding table copies | O-39..O-42, O-46, O-47 | 37.2 |
| Router fwd/bwd + aux loss + HiMidLo split (launch-bound, top-k dominated) | O-14, O-15, O-24, O-25 | 30.2 |
| Attention glue (FA epilogues/postprocess, MLA KV bwd + quantize) | O-10, O-31, O-33, O-34 | 18.9 |
| Norm + quantize prologues/epilogues (fwd and bwd) | O-01, O-04..O-07, O-27, O-28, O-35, O-36 | 17.5 |
| Dist-MoE combine/dispatch/staging | O-02, O-16, O-17, O-30 | 15.9 |
| Everything else (rope epilogues, colwise-in-forward, CE, SwiGLU, horizontal GEMM, ...) | rest | 28.9 |

Split by tier: T0 63.3 ms, T1 103.9 ms, T2 18.3 ms, T3 33.5 ms.

## How I computed savings

- Saving = measured time of the replaced ops - ideal of the fused kernel. The ideal is `bytes / 6921 GB/s` or
  `flops / GEMM peak`, whichever is larger, and I recomputed the bytes myself. Where two cards measured the same
  ops, I used the smaller time.
- For GEMM-epilogue fusions (O-11, O-12) I counted only the removed glue kernel minus the extra epilogue traffic.
  I did not count the GEMM's own gap to peak, because fusing does not recover it. For O-31 the GEMM gap can't be
  separated from the card's numbers, so that saving is an upper bound (at most ~0.7 ms too high).
- Where a horizontal or merged GEMM is the whole point (O-20, O-26), the GEMM efficiency gain is counted. It is
  measured against peak, so it is an upper bound.
- Launch-bound opportunities (router, Dist-MoE planning, kv_norm) use the roofline ideal as instructed. In practice
  each will keep a few microseconds of launch floor per layer.

## How I grouped the tricky cases

- **Cross-layer residual kernel (C-7968).** I split it into four parts:
  - attention_norm + one shared quantize (O-01, together with C-6651);
  - MoE combine + adds (O-02, together with C-6773);
  - the cross-boundary extra: the norm stops re-reading the residual stream (O-03, 0.49 ms, C-7968 only);
  - final norm (O-04, together with C-1508).
- **Fusions that cross into a neighbouring kernel are recorded as separate extras.** This applies to:
  - the FA-forward epilogue (O-10 on top of the quantize+swizzle in O-09);
  - the FA dQ postprocess (O-33 on top of q-rope-bwd + quantize in O-32);
  - single-pass cross-entropy (O-23 on top of O-22);
  - merged wq_a/wkv_a dgrad (O-21 on top of O-20 and O-28);
  - embedding backward writing straight into the reduce-scatter buffer (O-39 on top of O-38).
- **w2 dY quantize.** C-1047 fuses it into the producer (attention_norm backward). C-1543 fuses it into the consumer
  (broadcast_and_scale). Both only remove the same dy read, so they share one opportunity (O-29) as alternatives.
  C-1543 also covers 15% of O-30, the broadcast_and_scale elimination that C-4163 proposes.
- **Router.**
  - Router-only time is C-9469 minus C-9256 = 14.44 ms (O-14). The aux loss is O-15.
  - Router backward (O-24, C-5502) and the HiMidLo split (O-25 = C-1640 minus C-5502) are separate.
  - The gate wgrad accumulation is its own opportunity (O-26, C-9194; C-1260 covers part of it).
- **Dist-MoE staging.** I split it into planning plus padding glue (O-16) and the activation save/restore copies
  (O-17). Per-kernel times come from the card arithmetic: the union of C-4525 and C-2777 minus C-2793 and C-8100.
- **FSDP.**
  - Forward and backward re-unshard are separate opportunities, because C-7051 and C-2650 cover only the forward.
  - C-1888 lists temp->persistent `copy_` and clone kernels that C-7051 lacks. I verified the `copy_` refill in
    `torchtitan/quantization/mxfp8/tensor.py`, so those kernels became their own opportunity (O-45, the remainder of
    C-1888's 92.8 ms).
  - Backward re-unshard costs are scaled from the smaller forward report. This is conservative: C-7411's own number
    would give ~1.7 ms more on O-42.
- **Colwise in forward.** C-2076 and C-9441 propose the same change, so they form one opportunity (O-19). The fused
  ideal is just the colwise write that moves into forward (1.205 ms).

## Card claims I corrected

- **Ideals too high:**
  - C-6651: 214 MB per layer, should be ~154 MB.
  - C-7051 and C-1888: 2.1 and 3.0 GB per block. A fused copy-out + quantize kernel reads the gathered buffer once
    and writes 1-byte qdata, ~1.23 GB per MoE block. These two corrections more than double the FSDP weight-prep
    saving relative to the cards.
  - C-8089: 22.9 ms, while a one-pass pack is 15.3 ms.
  - C-9958: 7.53 ms, should be 7.03 ms.
  - C-7582: its ideal also keeps the full kv write.
- **Gaps that include a GEMM's own inefficiency:** C-1332 (+0.39 ms), C-7582 (+1.1 ms), and C-5659 (can't be
  separated).
- **Measured times:**
  - C-2076 is ~0.8 ms above C-9441's backward requantize time; I used C-9441.
  - The w2 quantize time implied by C-1543 (1.80 ms) is higher than the one implied by C-1047 minus C-7467
    (1.59 ms); I used the smaller.
  - Duplicate cards with higher measured times (C-4713, C-3838, C-8154, C-9717, C-5526, C-5686, C-3040, C-9619,
    C-2330, C-1217) gave way to their smaller twins.
- **Tiers:**
  - C-1640: T2 -> T1. The gather backward has no collisions.
  - C-1888: T2 -> T1. Max-based block scales don't depend on reduction order.
  - C-4192: the postprocess part is T3, because it replaces flash-attention's own kernel.
- **Caveats, not corrections:**
  - C-8973's prepare-kernel folding can't be verified; Dist-MoE is external and there may be a cross-rank ordering
    dependency.
  - C-2793's planning kernels are split by EP barriers, so the ~0 ideal is optimistic.
  - C-9655's param.grad clone runs outside the program, at the edge of the backward scope.
  - O-19 changes the saved-activation format. It costs memory for the attention output and the ffn_norm output.
