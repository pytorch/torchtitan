# Qwen3 30B-A3B step: adjudicated fusion opportunities

55 cards -> **41 opportunities: 38 real, 3 not real. Total real saving: 369.1 ms per step.**
Every saving was recomputed as measured minus the fused kernel's ideal, where ideal = max(bytes / 6921 GB/s, flops / matching GEMM peak). Where two cards measured the same ops differently, the smaller number was used. `out/scratch/build_judgement.py` regenerates `judgement.json`.

## Where the time is

| Group | Opportunities | Saving (ms) |
| --- | --- | --- |
| RoPE forward + backward (fp32 rotate-half chains; the `rope` local_compile region is off in this capture) | O-01, O-05 | 196.5 |
| Rest of the attention glue: QKV split, QK-norm fwd/bwd, cross-fusions, dQKV quantize, FA pre/postprocess | O-02..O-04, O-06..O-10 | 44.1 |
| Chunked cross-entropy fwd+bwd + dgrad staging | O-39, O-40 | 32.5 |
| Router fwd/bwd glue, HiMidLo split, gate wgrad | O-25..O-29 | 33.3 |
| Expert and dense weight prep (cast + all-gather copies + MXFP8 32x32) | O-32, O-33 | 19.1 |
| MoE combine / dispatch / backward chains, Dist-MoE metadata | O-18, O-22..O-24, O-30, O-31 | 20.9 |
| Residual + norm (+ quantize) chains, MXFP8 activation epilogues | O-11..O-14, O-17, O-19..O-21 | 14.2 |
| FSDP reduce-scatter and root-group glue, embedding backward | O-34..O-38 | 8.4 |

## How the tricky cases were grouped

- **Nested attention chains.** Forward: split+norm (O-02), RoPE (O-01), the norm->RoPE increment (O-03) and the wqkv-epilogue increment (O-04). Backward: RoPE (O-05), norm+cat (O-06), RoPE->norm increment (O-07), the dQKV quantize folded in (O-08) and the FA postprocess folded in (O-09).
  - The quantize and postprocess times were derived by subtraction: C-5618 minus C-2144 minus C-8971, and C-7296 minus C-5618. The kernel counts agree (3 kernels each).
  - The parts sum to each umbrella card's own total.
- **Residual / norm / combine chains** were cut at the shared ops. Increments go only to the cards that fuse across the cut:
  - O-13 / O-14: the add + ffn_norm + dispatch copy, and the wo-epilogue add.
  - O-17 / O-18: the add + attention_norm + quantize, and the combine fused into the add.
  - O-21 / O-22 / O-23: the norm backward, the broadcast, and their cross-fusion.
- **Partial coverage.** Cards that remove only one op of a chain got the fraction of the saving they would capture:
  - C-9768: 0.48 of O-13.
  - C-3721: 0.25 of O-24.
  - C-3535: about half of the router forward and of the router backward.
- **FSDP glue.** The cards use two naming schemes (`fsdp.*` vs `bucketing.*`) for the same logical copies, so they were grouped by parameter group and direction: expert prep, dense prep, root groups, dense reduce-scatter, expert grads.
- **Dist-MoE metadata** was split into a forward opportunity (O-30) and a backward one (O-31). C-9301 lists only 9 of the 18 kernels, so it covers about 0.63 of each.

## Not real

- **O-16, the standalone MoE combine (C-1552).** It is already at roofline once you count the 1 GiB [T,8,D] copy of the route outputs that must be saved for backward:
  - The combine reads the [T,8,D] buffer and writes y plus that copy, 2.28 GB in total, for an ideal of 329.8 us. The measured time is 326.49 us.
  - The evidence for the copy: C-8332 lists it, and C-1212/C-2351 read saved [T,8,D] route outputs. The combine buffer is symmetric memory that every layer reuses, so the outputs have to be copied out to survive until backward.
  - What is left is the cross-fusion O-18.
- **O-15, writing the ffn_norm output only into the dispatch buffer (C-5765's increment).** That buffer is transient symmetric memory, but the normed tensor has to persist to backward, so one of the two stores must stay.
- **O-41, linear cross-entropy (C-5562's increment over O-39).** The bf16 lm_head GEMMs are already compute bound, measuring below the roofline. The gradient also needs the final logsumexp, which forces either a recompute GEMM or materialized logits.

## Card claims corrected

- **C-9374:** its 11.68 ms ignores the saved 1 GiB copy. The real value is O-17 + O-18 = 4.4 ms, which matches C-8332.
- **C-5562:** its ideal assumes the logits are computed once. It realistically captures about 67% of O-39.
- **C-1835 / C-3475:** the single-read ideal needs the 304 KB bf16 row on-chip, which takes a CTA cluster. A two-pass kernel would save 31.0 ms instead of 32.4 ms.
- **C-6432:** its total includes the emulated size-1 collective copy. C-6450's smaller number was used for the expert-grad conversion.
- **C-8981:** it measured only the adds and cast (0.28 ms), but its single-GEMM proposal is worth 3.79 ms (O-29).
- **Tiers:**
  - C-3845: T1 should be T3, because the wo half needs the FA epilogue.
  - C-4049: T2 should be T1, because the round-robin router's ids are distinct within each row, so its scatter has no collisions.
  - C-4174: its single K-stacked GEMM is not bitwise.
- **Smaller measurements used:** C-1422 over C-4698, C-7280 over C-5765, C-3653 over C-7500, C-5490 over C-9447, C-3845 over C-7359, C-1835 over C-3475, and C-8055 over C-4730 for the forward unshard.
  - C-1704's lower chain total comes from omitting the position bound-check kernels, not from measuring the same ops lower.
- **GEMM-epilogue cards measure their GEMMs below the random-data roofline** (low-entropy fake-PG data): C-4735, C-9768, C-7967, C-3721. Their roofline savings are therefore conservative for the glue they remove.
