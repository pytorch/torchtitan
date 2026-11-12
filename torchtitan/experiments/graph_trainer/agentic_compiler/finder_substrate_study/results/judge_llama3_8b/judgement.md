# Llama 3 8B (FSDP 8, seq 8192, fake PG) -- adjudicated fusion opportunities

- 55 cards -> **33 opportunities: 32 real, 1 not real**.
- **Total real saving: 118.06 ms/step.** Of that, 4.28 ms (O-28, O-29, O-31) exists only in the
  per-module FSDP realization, the one that re-gathers weights in backward.
- By category (ms): activation 35.83, communication-glue 33.45, gemm-adjacent 23.57, rope 9.52,
  loss 8.33, attention-glue 4.23, norm 3.13.
- Largest opportunities:
  - FSDP reduce-scatter pack (O-24): 24.95
  - SwiGLU backward (O-02): 20.16
  - SwiGLU forward (O-01): 13.51
  - chunked cross-entropy (O-18): 7.68
  - wgrad-into-bucket epilogue (O-25): 6.05
  - SwiGLU backward into the w2 dgrad epilogue (O-05): 5.43

## Method

- Ideal = max(bytes / 6921 GB/s, flops / 1929.5 TFLOPS), recomputed from each card's bytes, shapes
  and instance counts.
- When cards measure the same ops differently, I used the smaller value and said so in
  `saving_basis`.
- **GEMM prologue/epilogue opportunities (T3)** are recorded as the *extra* beyond the standalone
  glue fusion.
  - In this fake-PG run the large GEMMs measure *below* their bf16 roofline (low-entropy data):
    - w13: 28.35 vs 31.91 ms
    - w2 dgrad: 14.85 vs 15.96 ms
    - wqkv: 6.49 vs 6.84 ms
    - lm_head: 8.59 vs 8.92 ms
  - The strict formula would therefore charge a fused kernel more than the plain GEMM costs today.
  - Instead I used max(bytes/BW, min(flops/peak, measured GEMM)). In every case the epilogue bytes
    fit under the GEMM time, so the extra equals the ideal of the standalone fused glue kernel that
    disappears.
  - These extras sum to 22.99 ms. With the strict formula they would be:
    - O-04: ~0 (w13 route; it is -0.30)
    - O-05: 4.33
    - O-08: 0.60
    - O-19: 0.27
    - O-25: 5.1-6.4

## Grouping of tricky cases

- **SwiGLU.** Split into five additive pieces:
  - forward fusion with silu still saved (O-01)
  - backward fusion reading the saved silu (O-02)
  - fwd/bwd pairing: recompute silu instead of saving it (O-03, 2.17 ms)
  - forward folded into a GEMM: w13 epilogue *or* w2 prologue, which are alternatives (O-04)
  - backward folded into the w2 dgrad epilogue (O-05)

  GEMM-fold cards get 0.5 on O-03, because the GEMM hides the silu write or read on their side.
- **RoPE.** The per-layer rotary gather, bounds check and conj clones are one cross-layer
  opportunity (O-07). The forward and backward rotation fusions exclude them (O-06, O-09). Fused
  RoPE cards that gather inline get partial credit on O-07.
- **Attention backward:**
  - postprocess folded into the RoPE-backward pass (O-10)
  - memsets folded into the preprocess kernel (O-11)
  - preprocess folded into the wo dgrad epilogue (O-12, C-9764 only)
- **Residual add.** add+norm (O-13, O-14) and the wo/w2 epilogue compete for the same add. The
  epilogue route removes the add entirely, so the epilogue cards cover O-13 and O-14 fully plus the
  extra O-15.
- **Two FSDP realizations.** The cards come from two runtimes of the same step:
  - per-module FSDP: two-kernel copy-in, backward re-gather, fp32 addmm accumulation for lm_head
    with one reduce-scatter
  - traced/bucketed program: one pack kernel per bucket, no re-gather, per-chunk lm_head
    reduce-scatter

  I grouped by logical stage and took the smaller measured time:
  - reduce-scatter pack (C-7328 / C-9055)
  - copy-in (C-6658 / C-2543)
  - copy-out (C-6460 / C-6650)
  - lm_head wgrad (C-3808 / C-6983)
  - embedding backward (C-7174 / C-5781)

  Backward re-gather instances are separate opportunities, flagged as realization-dependent.
- **Not real: O-22 (C-8387).** Folding the per-chunk lm_head shard accumulation into the
  reduce-scatter means fusing into the collective itself.

## Card claims corrected (main ones)

- **C-7770, C-4671:** their ideals drop the saved silu write, but the unchanged backward still
  needs silu. The standalone forward saving is 13.51 ms, not 14.59.
- **C-5925, C-4671, C-6148, C-6097, C-9529, C-2562, C-5541, C-8340, C-9095, C-1135:** the claimed
  GEMM+glue gaps are mostly the standalone glue fusion. They are also distorted by GEMMs that run
  faster than the roofline. Each has been re-split into base and extra.
- **C-8256:** includes 0.27 ms of w2 forward GEMM inefficiency, which is not a fusion saving.
- **C-9264:** used the larger measurements. The recomputed saving is 35.83 ms, not 36.07.
- **C-9878, C-6419:** 1.60 / 1.71 ms raised to 1.94 ms. The add kernel goes away entirely, and the
  GEMMs are at or below roofline.
- **C-7034:** includes the final norm backward, which has no residual add; its measured time is
  higher than C-3593's (used 4.91 ms).
- **C-3987:** the final norm backward has no fusion partner. Only the gradient-buffer fill/copies
  (O-20) and the dw cast (half of O-23) are saved.
- **C-2560:** covers only the final 4.2 GB copy, 0.48 of the lm_head wgrad stage.
- **C-6658:** in the traced realization its cast+copy fusion already exists. The forward saving uses
  that realization's 1.11 ms.
