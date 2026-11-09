# Effect of reuse: Qwen3 30B-A3B -> Qwen3 235B-A22B

The question (item 2 of `../TODO.md`): if a verified kernel pack exists for one model, how much cheaper is the
agentic compiler loop for a sibling model? Qwen3 235B-A22B is the same architecture family as Qwen3 30B-A3B, at
different sizes and with a different parallel layout. The loop was run once more, starting from everything the
30B loop produced. The audit report of the run is `README.md`.

## Bottom line

| | Qwen3 30B-A3B, from scratch | Qwen3 235B-A22B, seeded from 30B |
|---|---|---|
| Wall clock to a validated result | ~13 h | **2 h 41 min** |
| Fusion plans / training iterations | 4 plans, 7 iterations | 2 plans, 1 lift-only + 1 interim + 1 final |
| Kernels accepted | 15 | 19 (15 ported, 4 new) |
| Trials per accepted kernel | up to 30 (G01: t30), typically 2-14 | **1 for every kernel** |
| Verifier FAILs | at least 2 (G06 t13, G15 t3) | **0** |
| GPU time saved per step (excl. fake-PG collectives) | -29.8% | -28.8% |
| Numerics | bitwise identical | bitwise identical |

The 30B duration comes from its autokernel board, which spans 09:54 -> 22:43 on 2026-11-05 with finder iterations
interleaved; that board's timestamps were written by agents and are approximate. The 235B times are the
coordinator's wall clock (`timeline.log`).

**Reuse cut the loop by about 5x without losing kernel quality.** It removed most of the writing and all of the
design search. It did not reduce verification, which is now about 75% of the wall clock.

## What was reused, and how it carried over

| 30B artifact | How the 235B run used it | Outcome |
|---|---|---|
| Lift passes (topology matchers, sizes as wildcards) | replayed unchanged on the 235B graph dump | 13 of 16 lifted the expected sites (94 per layer region); 3 replaced by the 235B finder's regions, no hand edits |
| Fusion plan | used as plan_iter1, extended with the finder's coverage of the 235B profile | no plan iterations; the 30B loop needed 4 |
| Kernel source and design notes | writer prompt: the 30B kernel, its board row, its verifier's checklist | 15 of 19 kernels ported from a 30B kernel |
| Kernel files as is | run unchanged on the 235B specializations | **0 of 13 usable**: every kernel hard-codes its shapes |
| Verifier checklists (30B board, README sec. 7) | verifier brief: use them as the checklist, adapted to the new shapes | every verifier reused the sibling's test design; each run still took 40-100 min |
| Harness, briefs, promotion procedure | copied | no setup time |
| Cache patterns (l1 keys) | the pass keys the lifted code | 6 of 13 patterns had the same l1 key and `reference.py` as 30B; only the specialization differed |

### Regions

The matchers transferred well. Of the three that did not:
- **G02 (attention forward glue):** the 235B graph decomposes the k/v split as a single `reshape` where 30B had
  `clone` + `view`. This is a new failure mode for item 1 of `../TODO.md`.
- **G33 and G42 (expert weight prep):** the expert weights now come from the FSDP all-gather instead of an fp32
  cast (R1).

All 20 lift passes also match under the default pipeline with CUDA graphs, not only under the fusion_finder
pipeline they were generated on.

The finder still mattered. Four regions exist only because 235B FSDP-shards the experts and uses 256-way
buckets, and one of them (G-105, the expert-gradient bucket pack) saves 31.7 ms/step. Reusing the 30B plan alone
would have missed about 43 ms/step.

### Kernels

The writers started from the final 30B design instead of searching for one. The ported kernels stay close to
their sources: difflib similarity 0.76-1.00, 4-331 changed lines. Item 2 of `../TODO.md` records 0.30-0.39 for
near-identical patterns that were rewritten from scratch.

| Kind of change | Regions | Changed lines | Writer time |
|---|---|---|---|
| Constants only (row count, grid split) | G06, G07, G17, G28, G47 | 4-25 | 2-5 min each |
| Reduction order re-derived for the new N or T | G19, G15, G45, G41 (quack RMSNorm at N=4096); G91 (column sum over 8192 rows); G90 (nll over 1024 rows) | 31-119 | 3-13 min each |
| New input contract | G-115, G-129 (bf16 input from the all-gather, FSDP unshard permutation) | ~150 | 1-6 min each |
| Work decomposition re-laid out | G01 (dw ownership no longer aligns with the MXFP8 column blocks: cross-CTA scale exchange); G-102 (64 q heads, half-token work units) | ~330 | 6-28 min |
| No 30B counterpart | G-105, G-111, G-125, G-133 (layout copies) | new | 1-7 min each |

Writer times are the intervals between a writer's consecutive handoffs (writers worked through their problems in
order). All 18 non-G01 candidates were handed off within 26 minutes of the writers starting.

**The real porting cost was reduction orders that depend on shape.** A bitwise gate requires the kernel to add
in the same order as the reference, and that order is chosen by the backend from the sizes:
- quack's RMSNorm configuration depends on N. At 4096 the forward splits a row over 2 warps, and the backward
  uses 304 blocks and 4 chunks per thread and evaluates a different dx formula.
- torch's reduction config depends on the reduced size: 128 CTAs instead of 512 for G91, 64 threads instead of
  256 for G90.

The writers re-derived these from source and SASS. Each verifier re-derived them independently and showed that
the 30B order gives different bytes.

### Quality

| Region | 30B speedup | 235B speedup |
|---|---:|---:|
| G01 attention bwd | 13.4x | 10.6x |
| G02 / G-102 attention fwd | 14.3x | 14.2x |
| G06 router bwd | 31.7x | 36x |
| G07 loss bwd | 4.9x | 4.9x |
| G90 loss fwd | 3.75x | 3.86x |
| G19 norm bwd + quant | 1.76x | 1.84x |
| G15 dgrad + norm bwd | 1.23x | 1.19x |
| G41 norm + quant | 1.47x | 1.48x |
| G45 add + norm | 1.27x | 1.33x |
| G33 / G-115 weight prep | 2.15x | 2.90x |
| G42 / G-129 weight prep | 2.23x | 2.90x |
| G47 router softmax | 3.2x | 4.9x |
| G28 quant + GEMM | 1.09x | 1.05x |
| G17 router counts | 35x | 18x |
| G91 aux-loss fwd | 11.2x | 7.1x |

The ported kernels keep their speedups. The two launch-bound kernels (G17, G91) have 4x fewer tokens per call, so
fixed launch cost is a larger share. G01 is latency-bound in a structure inherited from 30B; its writer judged a
redesign not worth it. Kernels the 30B loop had already pushed to the HBM floor (G07, G90, G-102) are again at
or near it. At step level the saving is the same fraction as for 30B.

## Where the time went

| Phase | Wall clock | Effect of reuse |
|---|---|---|
| Baseline, replay of the 30B plan, finder, problem capture | 28 min | replay and coverage replaced the plan iterations |
| Writing (6 writers in parallel) | ~30 min to the last handoff | large: one trial per kernel |
| Verification (one fresh verifier per candidate) | ~2 h, the critical path | small: verifiers reused the checklists but ran every check in full |
| Final runs | 11 min | none |

About 40 minutes of verification were lost to coordination, not to the work itself:
- a single GPU lock that verifiers held through JIT compiles;
- the 20-agent session cap, which delayed G01's verifier by 17 minutes.

## Where reuse fell short

1. **Kernel files are not reusable as is.** They encode M, N, head counts, grid splits and hex constants such as
   1/N.
2. **Silent fallbacks hide that.** The 30B G07 and G90 kernels fall back to a copy of the reference ops for other
   shapes. They passed the bitwise gate on the 235B specializations with no speedup, which looks like successful
   reuse. Only a timing check exposed it. This run required kernels to raise on unsupported inputs instead.
3. **Verification did not get cheaper.** Five ports changed only constants (4-25 lines), yet each verifier ran the
   full suite: sanitizers, exhaustive sweeps, graph tests. The checks are reusable per family, but nothing let a
   verifier rely on the sibling's verdict for the unchanged code.
4. **Shape-dependent orders are rediscovered each time.** Nothing records "this kernel's order is quack's N=2048
   configuration", so the writer had to find that out again for N=4096.
5. **Matchers can still break** on decomposition changes (G02) and parameter-producer chains (G33/G42). The finder
   rerun caught them here.

## Caveats of the comparison

- **No controlled from-scratch arm.** There is no from-scratch 235B run, so the 5x mixes the effect of reuse with
  a more mature process: briefs, tools and the coordinator's experience from three earlier loops.
- **Different hosts and stacks:** 30B ran on CUDA 13.2 with cutlass-dsl 4.8.1; 235B on CUDA 13.0 with
  cutlass-dsl 4.8.0.
- The 30B duration is approximate (agent-written board timestamps) and includes building the 30B loop's own
  tooling.
- Per-kernel speedups compare different shapes and must not be read as like-for-like kernel quality.

## Recommendations

1. **Ship a port manifest with each kernel:**
   - the shape constants it assumes;
   - the backend configuration its reduction order copies (e.g. "quack RMSNorm fwd, N=2048: 32 threads/row";
     "torch reduce, 32768 rows: 512 CTAs");
   - a raise for anything else.

   Porting then becomes a lookup plus a constant change for most kernels.
2. **Delta verification for constants-only ports.** Verify the diff against the verified sibling, rerun the
   sibling's suite at the new shapes, and re-time. This could take a verifier from about an hour to minutes for a
   third of the problems.
3. **Key retrieval by l1.** 6 of 13 patterns kept their l1 key, so a cache that offers the existing pattern's
   kernel for a new specialization makes reuse automatic for those.
4. **Plan the verification capacity before spawning:** GPU-sharing rules (lock only timing) and enough agent slots
   for writers and verifiers.
5. **Keep running the finder on the new model.** One layout-specific region (G-105, 31.7 ms/step) saved about as
   much as the six ported router and loss kernels together (34 ms/step).
