# Eager vs Graph Fusion-Region Finders: Full Comparison Report

DeepSeek-V3 671B, one training step, fake process group, one GB300 per side.
Commit `c5c112d82`. Date 2026-11-04.

Sources:
- `eager/report.md` and `eager/regions.json` (30 regions);
- `graph/report.md` and `graph/regions.json` (146 regions);
- the raw per-op table `eager/analysis/op_table.parquet` (66,303 rows) and
  per-node table `graph/analysis/node_table.csv.gz` (126,413 rows), which
  every cross-check in this report is computed from.

## 0. Executive summary

1. **Same hot spots, same measurements.** Both finders rank the MoE token
   path first: the combine forward and backward, and the two deterministic
   `index_put` calls in the dispatch backward. Then routed-expert SwiGLU and
   query RoPE + concat. Where the two programs run the same code, measured
   time agrees to about 1%: RoPE 741 vs 729 ms, MLA K/V glue 279.5 vs
   279.5 ms, shared experts 193 vs 196 ms.
2. **The two programs are not identical.** GPU busy is 41.3 s (eager) vs
   37.1 s (graph). About 3.0 s of that is recompute: eager `torch_remat`
   replays the whole MoE dispatch/combine path, while graph remat
   re-materializes only what backward consumes. Another 0.5 s is eager-only
   forward glue (the `spmd_types` all-to-all concat). Step-level totals
   therefore can't be compared directly; per-family and per-instance numbers
   can.
3. **Graph finds more kinds of opportunity:**
   - 61 GEMM prologue/epilogue candidates (eager: 1);
   - 14 recompute->backward regions (eager: 3);
   - 8 cross-layer regions (eager: 1);
   - SimpleFSDP gradient-pack glue;
   - a long tail of 28 small regions (174 ms in total).

   Its candidate generator enumerates these systematically: per-phase
   components, then cross-module, cross-layer, recompute->backward, and
   GEMM-adjacent joins.
4. **Eager finds better-explained, more actionable opportunities:**
   - the kernel that actually ran and why it is slow;
   - the `Configurable`/method to override for each region;
   - eager-only waste (2.85 s per step of dead combine recompute);
   - three standalone, order-free opportunities that graph only lists inside
     GEMM-fused (T3) candidates: forward un-permute E-06 (1,040 ms),
     post-all-to-all permute gather E-07 (551 ms), and combine backward gather
     E-12 (267 ms).
5. **Bitwise tiering.** Eager reasons about semantics: a permutation
   `index_put` is order-free (T0), and the combine backward needs only one
   matched reduction (T1). Graph tiers by op type ("`index_put` present ->
   T2"). Eager is right on both. But one of eager's tier notes is wrong:
   aten's deterministic `index_put` at this row width rounds to bf16 after
   **each** duplicate. It does not accumulate in fp32 (verified from source,
   section 9.1).
6. **Neither finder found horizontal (sibling) fusions.** Both name this as a
   blind spot.
7. **Recall check:** both flagged all six spans that eager normally compiles
   with `local_compile` (6/6 each). The eager result is weaker evidence,
   because the eager agent read a plan doc that names them.
8. **Side findings worth follow-up (not verified here):**
   - GraphTrainer `memory_policy="full"` runs out of memory at this config on
     HEAD;
   - `ExecutionTraceObserver` crashes on `torch_remat` placeholder tensors;
   - eager full recompute replays the MoE combine for nothing;
   - host syncs cost about 275 ms of idle per step;
   - fake-PG low-entropy data runs GEMMs about 25% faster than real data,
     which overstates glue's share of the step.

Bottom line: the union of both inventories is clearly better than either
alone. The graph substrate is the stronger discovery tool. The eager
substrate is the stronger diagnosis and integration tool.

## 1. Experiment design

**Question:** given the same model, config, hardware, and output schema, how
does the substrate change which fusion opportunities an agent finds, and how
well it characterizes them? The substrates are eager code plus runtime traces
vs the GraphTrainer FX graph plus per-node traces.

**Fixed setup, identical for both:**
- DSv3 671B (`deepseek_v3_671b(seq_len=4096)`), fake PG with `NGPU=256`,
  FSDP 256, EP 64;
- 98,304 tokens per microbatch, no gradient accumulation;
- 10 steps, profile step 10 after 3 warmup steps;
- `seed=42`, `deterministic=True`;
- full recompute (eager `FullActivationRematConfig`, graph
  `memory_policy="full"`);
- `model.local_compile.regions = []`;
- CUDA graphs off;
- GPU 0 for eager and GPU 2 for graph; 2 spare GPUs.

**Agent rules:**
- each agent used only its own substrate. The eager agent could not trace to
  FX or read graph dumps. The graph agent could not read model source to find
  candidates, only to name what it had already found;
- no subagents;
- no guidance on where regions are, beyond "find all, large or small,
  including cross-module, cross-layer, backward, and recompute->backward";
- the same `regions.json` schema and `report.md` sections.

**Confound:** one agent run per substrate. Differences in method between the
two agent instances are mixed in with substrate effects. Section 11
separates the two where the evidence allows.

### 1.1 Leakage audit

Keyword audit of both transcripts: Read/Grep/Glob paths and shell commands.

| Check | Eager agent | Graph agent |
|---|---|---|
| Read the other side's outputs | no | no |
| Read the other side's prompt or plan | no (saw filenames once via `ls`) | no |
| Read its own side's plan doc | **yes**: `cat EAGER_FUSION_PLAN.md` (names the six `local_compile` regions; uses router and `index_put` as examples) | no |
| Broke its substrate rule | no FX tracing, no `graph_trainer` reads, no tlparse | no model-source reads; listed `torchtitan_recipes/models/` once to find the recipe |

Impact: small. The six region names are also visible as `@local_compile`
decorators in the model code the eager agent was allowed to read. It does
make the recall check (section 10) weaker evidence for eager.

## 2. Run outcomes side by side

| | Eager | Graph |
|---|---|---|
| Wall time | ~50 min | ~56 min |
| Tool calls | 168 | 147 |
| Capture attempts | 1 (plus a debug-model probe) | 3: dump crash, out-of-memory, good run |
| Deviation from the fixed setup | none | extra pass `recompute_fresh_buffer_index_put_pass`, which retags 58 combine chains to recompute (HEAD runs out of memory otherwise) |
| Instrumentation | module pre/post hooks with `record_function`; patched `torch_remat` phase context (recompute markers; execution trace paused in recompute) | pass pipeline registered via `compile.pass_pipeline` (graph dumps + `custom_codegen_pass`); no tracked edits |
| Loss / grad_norm, step 1 | 12.24438 / 1.4762 | 12.24438 / 1.4762 |
| Loss / grad_norm, step 10 | 12.05490 / 5.9943 | 12.05693 / 5.9956 |
| Peak memory, step 10 | 257.35 GiB | 235.06 GiB |
| tps, step 10 | 2,407 | 2,603 |
| GPU busy / idle in the profiled step | 41,277 / 820 ms | 37,142 / 901 ms |
| Attribution coverage | 98.56% of busy (595.6 ms of the previous step's tail unowned) | 99.93% of kernel time to FX nodes; 99.996% of busy including eager optimizer ops |
| HBM peak used | 7,216 GB/s (elementwise add; copy measured 6,876) | 6,876 GB/s (copy) |
| GEMM peak used | 1,786 TFLOP/s (random data) | 2,360 TFLOP/s (zero operands; trace GEMMs reach 2.34) |
| Regions / no-opportunity entries | 30 / 7 | 146 / 16 |
| Report size | 518 lines | 1,680 lines |

## 3. Did both measure the same program?

### 3.1 Numerics

- Step 1 loss and grad_norm are identical to the printed precision, so
  forward and backward compute the same values on both substrates.
- By step 10, loss differs in the 4th decimal. Not investigated.
  Hypotheses:
  - fake collectives behave differently under FSDP2 (eager) and SimpleFSDP
    (graph), so the "reduced" gradients and the optimizer update differ;
  - the MoE expert-bias hook.

  This doesn't affect the kernel-time comparison. It does mean fake-PG
  multi-step loss isn't a valid eager-vs-graph bitwise check.

### 3.2 GPU time by phase and category (ms)

| Phase | Eager GEMM | Eager attn | Eager fake comm | Eager glue | Eager total | Graph GEMM | Graph attn | Graph fake comm | Graph glue | Graph total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| forward | 4,285 | 481 | 680 | 5,815 | 11,262 | 4,156 | 478 | 700 | 5,293 | 10,627 |
| recompute | 4,191 | 488 | 399 | 5,444 | 10,523 | 3,950 | 489 | 615 | 2,421 | 7,475 |
| backward | 8,608 | 2,248 | 684 | 7,919 | 19,460 | 8,357 | 2,152 | 735 | 7,773 | 19,017 |
| total | 17,085 | 3,218 | 1,763 | 19,179 | 41,264 | 16,463 | 3,119 | 2,050 | 15,487 | 37,119 |

Kernel launches: eager 92,074 (forward 36,508, recompute 17,974, backward
36,866). Graph 81,274 (forward 32,124, recompute 10,783, backward 38,367).

### 3.3 Memory-bound glue by module and phase (ms)

Selected rows; rows under 10 ms on both sides are omitted.

| Module (layers collapsed) | E fwd | E rec | E bwd | E total | G fwd | G rec | G bwd | G total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `feed_forward.routed_experts` (dispatch, combine, SwiGLU) | 4,788 | 4,813 | 6,834 | 16,434 | 4,294 | 1,695 | 6,646 | 12,635 |
| `attention.rope` | 251 | 248 | 242 | 741 | 242 | 243 | 243 | 729 |
| `feed_forward.router` | 258 | 6 | 27 | 291 | 257 | 113 | 25 | 395 |
| `attention.inner_attention` (K/V materialization) | 73 | 73 | 134 | 280 | 73 | 73 | 134 | 280 |
| `attention` (q split/cat) | 87 | 87 | 89 | 264 | 87 | 87 | 89 | 264 |
| `feed_forward.shared_experts` | 55 | 55 | 83 | 193 | 55 | 56 | 84 | 196 |
| `feed_forward` (dense layers, MoE wrapper) | 64 | 31 | 45 | 140 | 61 | 29 | 42 | 133 |
| `attention_norm` | 28 | 34 | 75 | 137 | 24 | 25 | 40 | 88 |
| `ffn_norm` | 24 | 24 | 77 | 125 | 24 | 24 | 40 | 89 |
| block (residual adds) | 72 | 36 | 0 | 107 | 72 | 36 | 0 | 107 |
| `router.aux_loss` | 28 | 27 | 34 | 90 | 28 | 29 | 31 | 88 |
| `router.gate` (HiMidLo split) | 0 | 0 | 55 | 55 | 0 | 0 | 20 | 20 |
| no module (accumulation adds, FSDP, loss) | 76 | 0 | 172 | 249 | 42 | 0 | 231 | 273 |
| GEMM-module glue (wgrad casts, bucket pack) | 0 | 0 | 36 | 36 | 23 | 2 | 122 | 147 |

The rows that run identical code agree to within a few percent. The
differences come from:

1. **Recompute scope (-3.1 s glue in graph, routed experts).**
   - Eager `torch_remat` replays everything in the block except saved regions,
     including the full combine path (score multiply, cast, deterministic
     scatter-add, all-to-all concat), whose outputs are dead in recompute.
   - Graph `selective_activation_remat_pass` duplicates only nodes that
     backward consumes.
   - Opposite direction: graph recomputes the router selection (113 ms); eager
     keeps `routing_decision` (a `recompute=False` region).
2. **All-to-all implementation (+0.5 s forward glue in eager).** Eager runs
   `spmd_types._AllToAllUneven`, which builds a list of chunks and calls
   `torch.cat` (about 3.3 ms per call, 4 per MoE layer). The traced graph uses
   `all_to_all_single`, with no concat.
3. **Attribution, not cost.**
   - The norm-backward rows differ because eager charges input-gradient
     accumulation adds to the norm; graph leaves 474 accumulation nodes
     without an FQN.
   - The GEMM-module rows differ because graph charges weight-grad casts and
     bucket packs to the GEMM module (SimpleFSDP), while eager's FSDP2 copies
     sit at the root.
4. **The graph out-of-memory workaround** adds one w2 grouped GEMM +
   `index_put_` + combine all-to-all per MoE layer to recompute. It shows in
   GEMM recompute (graph's is similar to eager's) and fake-comm recompute
   (615 vs 399 ms).

## 4. Methodology side by side

| Aspect | Eager | Graph |
|---|---|---|
| Unit of analysis | aten op instance (66,303 rows owning GPU work) | FX node (126,413 rows; 128,357 nodes in the graph) |
| Time source | Kineto; kernel -> owning op via External id | Kineto; kernel -> enclosing `_RecordFunctionFast` node label via correlation id |
| Kernel granularity | kernels split across child aten ops (e.g. `index_put` 11 kernels + `arange`/`div_`/`remainder` children) | every kernel under a node counted at the node (e.g. `index_put` 21-22 kernels) |
| Source / module | `with_stack` file:line (function granularity for operator expressions) + module hooks (FQN, class) | `custom.module_fqn` / `stack_trace` meta; 474 accumulation nodes without FQN; bucketing nodes mislabelled |
| Backward provenance | `evaluate_function` + `Sequence number` -> forward op (98.4% of backward time linked) | `autograd_backward` meta (exact) |
| Recompute provenance | patched `torch_remat` phase markers + module hooks under `is_recomputing()` | `_recomputed` node-name suffix plus `recompute` tag (cross-checked: 25,835 nodes) |
| Dataflow | Execution Trace tensor/storage IDs; address-keyed, no frees; **crashes under `torch_remat`** so it is paused in recompute; 31.5k backward edges unverified | FX args/users (exact); false connectivity from out-variant buffers and the bucket pack, cut by rules |
| Tensor sizes | Kineto input dims + execution trace | `val` meta; unbacked routed sizes resolved from runtime input dims (232 expressions) |
| Legality rule | code span within one owner (judgment), plus remat / host-sync / collective blockers | "fusion stage" = number of GEMM/attention/collective nodes on the longest path; equal-stage glue components are legal |
| Candidate generation | op table + dataflow components as discovery aids, then **manual curation by code span** | **algorithmic**: per-phase L1 components, then XM (cross-module), XL (cross-layer), RB (recompute->backward), T3 prologue/epilogue joins, then curation |
| Region granularity | one region per code span; forward and recompute merged ("forward+recompute") | one region per phase; supersets kept as separate overlapping regions (`overlaps_with`) |
| Tiering | by semantics (permutation, reduction order, -0.0 handling) | by op-type rule (`index_put`/`scatter`/`sort` -> T2; GEMM present -> T3) |
| Coverage accounting | every memory-bound row claimed once (18,963 ms in regions + 216 ms no-opportunity = 19,179 ms) | every memory-bound node class in a region or no-opportunity (including via T3 regions) |
| Integration hint | owning `Configurable` + method (override point) | node names + module FQN (graph-pass match) |

Discussion:

- **Discovery.** Graph discovery is systematic and repeatable; its pipeline
  reruns in about 2 minutes. Eager discovery is analyst-driven. The eager
  agent tried automatic components, found they merged router, dispatch, and
  combine into 50-100-op blobs, and fell back to code spans.
- **Dataflow.** The graph's exactness is substrate-inherent. Eager's weakness
  comes from tooling (Execution Trace limitations), not from the substrate
  itself (section 11).

## 5. Inventory overview

| | Eager | Graph (all) | Graph (greedy non-overlapping) |
|---|---|---|---|
| Regions | 30 | 146 | 69 (40 of them T3) |
| Total gap (ms) | 16,342 | n/a (overlaps) | 17,058 |
| Non-T3 regions / gap (ms) | 29 / 16,062 | 85 / n/a | 29 / 10,161 (agent's own figure: 11.1 s) |
| T3 regions / gap (ms) | 1 / 279 | 61 / 11,506 (with overlaps) | 40 / 6,897 |
| Tiers T0 / T1 / T2 / T3 | 13 / 12 / 4 / 1 | 33 / 30 / 22 / 61 | |
| Scope intra / cross-module / cross-layer / recompute->bwd | 19 / 8 / 1 / 3 | 76 / 62 / 8 / 14 | |
| Phase combos | fwd+rec 10, bwd 10, fwd 5, rec+bwd 3, fwd+rec+bwd 1, fwd+bwd 1 | bwd 64, fwd 47, rec 21, bwd+rec 14 | |

Why the non-T3 gaps differ (16.06 s vs 10.16 s, a 5.9 s difference):
- 3.0 s is E-01's recompute replay plus the all-to-all concat (E-01 minus
  G-005), which is eager-only work;
- 1.9 s is opportunities graph listed only as T3: E-06, E-07, E-12;
- the remaining ~1.0 s is smaller differences: the routed-SwiGLU pairing, byte
  accounting, and the HBM peak used (7,216 vs 6,876 GB/s).

## 6. Family-by-family comparison

Graph "primary" = the non-overlapping, non-T3 regions that best match the
eager regions. "Variants" = graph supersets or T3 framings of the same
family.

| Family | Eager regions (gap ms) | Eager gap | Graph primary (gap ms) | Graph gap | Graph overlapping variants (gap ms) |
|---|---|---:|---|---:|---|
| MoE combine forward (score multiply, cast, scatter-add) | E-01 (5,489) | 5,489 | G-005 (2,490) | 2,490 | G-004 (2,512) |
| MoE combine backward | E-02 (2,662) | 2,662 | G-002 (2,824) | 2,824 | G-001 (2,902), G-003 (2,801) |
| Dispatch backward: permute bwd + local-reorder bwd (two index_put per layer) | E-04 (1,166), E-03 (1,263) | 2,429 | G-006 (2,337) | 2,337 | G-008 (1,041) |
| Forward un-permute (index_put scatter) | E-06 (1,040) | 1,040 | - | 0 | G-010 (987), G-011 (972) |
| Post-a2a permute gather + index math | E-07 (551) | 551 | - | 0 | G-013 (848), G-012 (861) |
| Combine-a2a backward + un-permute backward gather | E-12 (267) | 267 | - | 0 | G-016 (553) |
| Local reorder (argsort, gathers) | E-10 (347) | 347 | G-033 (158), G-032 (165) | 322 | - |
| Routed-expert SwiGLU (fwd, recompute, bwd) | E-09 (434), E-05 (1,154) | 1,588 | G-019 (398), G-020 (387), G-015 (566) | 1,351 | - |
| Shared + dense SwiGLU | E-15 (118), E-18 (67), E-22 (46), E-24 (26) | 257 | G-050 (64), G-059 (45), G-061 (43), G-073 (33), G-091 (23), G-093 (22) | 229 | - |
| Query RoPE + concat (fwd, recompute, bwd) | E-08 (506), E-13 (245) | 751 | G-027 (241), G-026 (241), G-025 (243) | 725 | G-028 (216), G-029 (214), G-030 (213), G-031 (212) |
| MLA K/V layout glue | E-11 (279) | 279 | - | 0 | G-044 (86), G-045 (86), G-042 (108) |
| Router selection and scoring | E-14 (240), E-26 (19) | 259 | G-023 (255), G-041 (111) | 366 | G-018 (444), G-021 (304) |
| Router / aux-loss backward | E-19 (59) | 59 | G-087 (25), G-075 (32) | 57 | G-034 (143), G-035 (137) |
| Aux-loss statistics (fwd, recompute) | E-20 (53) | 53 | G-085 (26), G-082 (28) | 54 | G-052 (60) |
| Residual add + RMSNorm (fwd, incl. cross-layer) | E-25 (25), E-23 (39) | 63 | G-114 (10), G-112 (10), G-113 (10), G-096 (21) | 51 | - |
| RMSNorm bwd + gradient accumulation | E-17 (75), E-21 (51) | 126 | G-054 (57), G-068 (36) | 93 | G-088 (25), G-089 (25) |
| kv_norm / k_pe RoPE | E-28 (8), E-29 (7) | 15 | G-126 (5), G-125 (5), G-122 (6) | 16 | - |
| Chunked cross-entropy | E-16 (85) | 85 | G-079 (31), G-077 (31) | 62 | - |
| HiMidLo router-gate split (bwd) | E-27 (19) | 19 | G-099 (17) | 17 | G-051 (60) |
| MoE dispatch bookkeeping (counts, cumsum) | E-30 (1) | 1 | G-111 (11), G-131 (2), G-134 (2) | 14 | - |
| Weight-grad fp32 casts + reduce-scatter pack | - | 0 | G-067 (37), G-080 (30), G-086 (26), G-108 (13) | 105 | - |

Not mapped: 28 small graph regions (174 ms total, largest 47 ms). They are
mostly cross-layer (G-057 final-norm backward into the last combine
backward; G-084 aux-loss scalar chain; G-119 embedding + first norm), plus
norm-backward fragments and tiny recompute/forward bookkeeping. No eager
region is unmapped.

### 6.1 MoE token path, per instance

| Region | Phases | Instances | Kernels/inst | Measured us/inst | Ideal us/inst | Ext. GB/inst | Tier |
|---|---|---:|---:|---:|---:|---:|---|
| E-01 | for+rec | 116 | 57 | 49,080 | 1,759 | 12.69 | T2 |
| G-005 | for | 58 | 55 | 44,986 | 2,051 | 14.10 | T2 |
| E-02 | bac | 58 | 88 | 49,222 | 3,322 | 23.97 | T1 |
| G-002 | bac | 58 | 111 | 52,184 | 3,487 | 23.98 | T2 |
| E-04 | bac | 58 | 16 | 23,231 | 3,126 | 22.55 | T0 |
| E-03 | bac | 58 | 34 | 23,535 | 1,759 | 12.70 | T2 |
| G-006 | bac | 116 | 23 | 22,708 | 2,563 | 17.62 | T2 |
| G-008 | bac | 58 | 24 | 19,990 | 2,050 | 14.10 | T2 |
| E-06 | for+rec | 116 | 21 | 12,094 | 3,126 | 22.55 | T0 |
| G-010 | for | 58 | 23 | 26,807 | 9,783 | 14.62 | T3 |
| E-07 | for+rec | 116 | 32 | 7,876 | 3,126 | 22.55 | T0 |
| G-013 | for | 58 | 10 | 34,193 | 19,566 | 17.96 | T3 |
| E-12 | bac | 58 | 10 | 7,726 | 3,126 | 22.55 | T0 |
| G-016 | bac | 58 | 10 | 19,310 | 9,783 | 25.89 | T3 |
| E-10 | for+rec | 116 | 35 | 4,756 | 1,764 | 12.73 | T2 |
| G-033 | for | 58 | 32 | 4,564 | 1,847 | 12.70 | T2 |
| E-09 | for | 58 | 8 | 8,828 | 1,339 | 9.66 | T0 |
| G-019 | for | 58 | 8 | 8,260 | 1,405 | 9.66 | T0 |
| E-05 | rec+bac | 58 | 26 | 22,576 | 2,678 | 19.33 | T0 |
| G-020 | rec | 58 | 8 | 8,544 | 1,874 | 12.88 | T0 |
| G-015 | bac | 58 | 18 | 12,574 | 2,811 | 19.33 | T0 |

**Combine forward (E-01 vs G-005).**
- Same diagnosis in both: the multiply by the fp32 score materializes a
  `[786432, 7168]` fp32 tensor, followed by a cast and the deterministic
  scatter-add.
- Per instance: 49.1 ms (eager, including the all-to-all concat) vs 45.0 ms
  (graph).
- Eager counts 116 instances because recompute replays it. Graph counts 58.
  That alone doubles E-01's step-level gap (5,489 vs 2,490 ms).
- Eager adds the root cause: the multiply runs the non-vectorized
  `BinaryFunctor<float>` at 1.6 TB/s.
- Graph adds a superset, G-004, which also absorbs the routed+shared add.

**Combine backward (E-02 vs G-002).**
- Per instance: 49.2 vs 52.2 ms; external bytes 23.97 vs 23.98 GB. Close
  agreement.
- Graph adds two supersets: G-001 (plus router/aux-loss score backward, 183
  kernels per instance) and G-003 (plus the next layer's `attention_norm`
  backward and residual-gradient add, cross-layer). Eager has neither.
- Tier: eager T1 is correct (section 9.2).

**Dispatch backward (E-04 + E-03 vs G-006).**
- Both found the two backward `index_put` calls per MoE layer. In layer 10:
  - permute backward: 19.3 ms (eager) vs 23.1 ms (graph);
  - local-reorder backward: 18.0 ms (eager) vs 19.3 ms (graph).
- Eager splits them into two regions; graph combines them into one region
  with 116 instances. The gaps agree: 2,429 vs 2,337 ms.
- Both noticed that the permute backward's indices form a permutation, so a
  plain gather would do. Eager also noticed that `0 + (-0.0) = +0.0`, which a
  replacement gather must reproduce.
- Graph adds G-008, which joins the local-reorder accumulate with the
  shared-expert dgrad add.

**Forward un-permute, post-all-to-all permute gather, combine-backward
gather (E-06, E-07, E-12).**
- Eager reports them as standalone T0 regions: 1,040, 551, and 267 ms.
- Graph lists them only as GEMM-fused T3 candidates (G-010/G-011,
  G-012/G-013, G-016), and not in no-opportunity either.
- Root cause in the graph method: a single glue node whose neighbours are a
  GEMM and a collective forms no multi-node same-stage component, so it only
  surfaces through the T3 generator.
- These are the cheapest bitwise-safe wins in the inventory: a permutation
  scatter becomes a plain scatter or gather with no sort. Graph's non-T3 list
  misses about 1.9 s here.

**Routed SwiGLU (E-09 + E-05 vs G-019 + G-020 + G-015).**
- Per-pass numbers agree: forward 8.8 vs 8.3 ms per layer; backward parts
  12.6 ms (graph).
- Eager proposes one recompute+backward kernel (E-05, 22.6 ms per instance,
  1,154 ms gap) that reads `gate_up` once and emits both `h` (for the w2
  weight grad) and `d_gate_up`.
- Graph keeps recompute (G-020) and backward (G-015) separate. Its
  recompute->backward generator didn't pair them, probably because the
  recomputed SwiGLU output feeds the w2 GEMM, which puts the pair on
  different fusion stages.

### 6.2 Attention side

- **Query RoPE + concat:** same three passes in both (forward, recompute,
  backward), 751 vs 725 ms. Graph also lists RoPE-only sub-regions
  (G-028..G-031); G-028 is a recompute->backward pairing of the RoPE
  frequency gather.
- **MLA K/V layout glue:** both conclude that the K materialization copy only
  goes away by changing the attention kernel or GEMM epilogue (T3). Eager
  lists one combined region (E-11, 279 ms). Graph splits it into GEMM
  epilogue/prologue candidates around `wkv_b` (G-044, G-045, G-042).
- **kv_norm / k_pe:** small (15 vs 16 ms), found by both.

### 6.3 Router and aux loss

- **Forward selection:** found by both (E-14 240 ms; G-023 255 ms). Both say
  `topk` on 32- and 256-wide rows is the slow part (bitonic sort + gather,
  about 2 ms per call).
- **Recompute:** graph recomputes the router (G-041, 111 ms); eager doesn't.
- Graph adds recompute->backward pairings (G-034 143 ms, G-035 137 ms) and a
  forward superset with aux loss and dispatch prologue (G-018, 444 ms). Eager
  has only separate forward, backward, and aux pieces (E-14, E-19, E-20,
  E-26).

### 6.4 Norms, residuals, loss

- **Forward residual add + RMSNorm:** small in both (63 vs 51 ms).
- **Backward:** both pair the RMSNorm backward with gradient-accumulation adds
  (126 vs 93 ms). Graph also pairs the recomputed norm with its backward
  (G-088, G-089).
- **Chunked cross-entropy:** 85 vs 62 ms. Graph also lists the fused
  linear-cross-entropy T3 candidate (G-063) and the loss-backward-into-
  `lm_head`-dgrad candidate (G-053).
- Both note that standalone RMSNorm kernels already run at roofline. Eager
  identifies them as CuTeDSL kernels at 1.0-1.3x roofline.

### 6.5 FSDP glue

- **Eager:** FSDP2 copy-in/copy-out and reduce-scatter packing go in
  no-opportunity (150 ms, owned by FSDP2's flat-buffer protocol).
- **Graph:** SimpleFSDP weight-grad fp32 casts and bucket packs appear as
  ordinary nodes, reported as regions (G-067, G-080, G-086, G-108: 105 ms).
  Plus T3 candidates where the wgrad GEMM emits fp32 directly into the bucket
  (G-017 481 ms, G-024 253 ms, G-062 42 ms).

## 7. GEMM-adjacent (T3) candidates

- Graph: 61 regions (40 in the non-overlapping set, 6.9 s). Eager: 1 (E-11).
- The largest graph T3 candidates:
  - G-007 (2,164 ms): w13 dgrad grouped GEMM writes un-permuted rows,
    replacing `zeros` + `index_put`;
  - G-009 (1,026 ms): w2 dgrad grouped GEMM + SwiGLU backward epilogue;
  - G-010 / G-011 (987 / 972 ms): w2 grouped GEMM stores rows un-permuted;
  - G-012 / G-013 (861 / 848 ms): permutation gather in the w13 prologue;
  - G-014 (798 ms): SwiGLU in the w2 prologue;
  - G-016 (553 ms): permutation gather in the w2 dgrad prologue;
  - G-017 (481 ms): w13 wgrad emits fp32 into the reduce-scatter bucket.
- Caveats:
  - T3 gaps depend on the GEMM peak. Graph used the zero-operand rate
    (2,360 TFLOP/s). With the random-operand rate (about 1,870 TFLOP/s) the
    GEMM part of each ideal grows and the gaps shrink.
  - Under the current bitwise requirement (design decision D1), T3 is out of
    scope. These are inputs for the escalation tier (custom grouped-GEMM
    kernels with fused gather/scatter), not for the bitwise loop.
- Why only graph has them: graph's generator has explicit T3-prologue and
  T3-epilogue steps over the exact dataflow. The eager agent treated GEMMs as
  boundaries and noted "the eager trace gives no cost model for them". This
  is mostly a method difference that the graph substrate makes easy.

## 8. Findings unique to one side

**Graph only:**
- the GEMM-adjacent candidates (section 7);
- 11 non-T3 recompute->backward regions for the router, aux loss, norms, and
  RoPE frequencies, none overlapping eager's three SwiGLU pairings;
- cross-layer joins: G-003 (next layer's norm backward into combine
  backward), G-057, G-084, G-113, G-119;
- the host-sync idle breakdown: 58 gaps, 275 ms, from 14,848
  `_local_scalar_dense` reads after the D2H copy of the split sizes;
- the memory-policy root cause of the out-of-memory failure;
- TensorIterator splitting `[786432, 7168]` ops into 16 launches.

**Eager only:**
- E-05 routed-SwiGLU recompute+backward (1,154 ms);
- E-06 / E-07 / E-12 as standalone T0 regions (1,858 ms);
- dead combine recompute (2.85 s per step);
- the `spmd_types` all-to-all concat (real code, not a fake-PG artifact);
- kernel-level root causes for every top region;
- `-0.0` handling in the permutation replacement;
- override points (owning `Configurable` + method) for every region.

## 9. Disagreements and corrections

### 9.1 Deterministic `index_put` accumulation semantics (eager note is wrong)

E-01 and E-03 state that the deterministic sort-based `index_put` uses "fp32
accumulation, one bf16 rounding". From `aten/src/ATen/native/cuda/Indexing.cu`
at this commit:

- **Kernel selection.** `sliceSize == 1` -> `indexing_backward_kernel_stride_1`.
  `sliceSize <= warp_size` -> `indexing_backward_kernel_small_stride`, which
  accumulates duplicates in fp32 and rounds once. Otherwise ->
  `indexing_backward_kernel`.
- At row width 7168, `indexing_backward_kernel` runs. For each duplicate, in
  stable-sort order, it loads the destination as fp32, adds, and stores bf16:
  **one bf16 rounding per duplicate**.

A fused combine or dispatch-backward kernel that accumulates 8 rows in fp32
and rounds once won't be bitwise identical. It must round after each add,
in sorted original-position order. This matters for the kernel phase.

### 9.2 Tiering

| Region | Eager | Graph | Correct | Why |
|---|---|---|---|---|
| permute backward (permutation `index_put`) | T0 | T2 | T0 | each destination gets exactly one value added to zero; order-free (watch -0.0 -> +0.0) |
| forward un-permute (permutation `index_put_`, no accumulate) | T0 | T3 only | T0 | permutation scatter; no ordering |
| combine backward | T1 | T2 | T1 | the gather is order-free; the only constraint is the 7168-wide fp32 score-grad row sum |
| combine forward | T2 | T2 | T2 | 8 duplicates per token, rounded per add in sorted order (9.1) |
| local-reorder backward | T2 | T2 | T2 | same as combine forward |

### 9.3 Reconciled discrepancies

- **Dispatch backward:** G-006 has 116 instances (both `index_put` calls per
  layer); eager splits them into E-04 + E-03. Totals agree (section 6.1).
- **Combine forward:** the step-level gap differs 2.2x only because of eager's
  recompute replay and the all-to-all concat. Per instance it's 49.1 vs
  45.0 ms.
- **Kernel counts per instance** differ because of attribution granularity
  (section 4), not because different kernels run.

### 9.4 Unresolved

- External bytes per instance for the combine forward: 12.69 GB (eager) vs
  14.10 GB (graph). The two count the score tensor and the zero-initialized
  destination differently. This changes the ideal by about 0.2 ms per
  instance; the ranking is unaffected.
- The step-10 loss difference (section 3.1).

## 10. Recall check (six known opportunities)

With `local_compile.regions = []`, the six spans the eager recipe normally
compiles run as plain aten ops in both substrates. Did each finder flag them?

| `local_compile` region | Eager | Graph |
|---|---|---|
| `loss` (cross-entropy) | E-16 | G-077, G-079 |
| `fused_binary_activation` (SwiGLU) | E-05, E-09, E-15, E-18, E-22, E-24 | G-015, G-019, G-020, G-050, G-059, G-061 |
| `fp32_to_bf16_split` (HiMidLo) | E-27 | G-099 |
| `moe_aux_loss` | E-19, E-20 | G-075, G-082, G-085 (+ G-052) |
| `rope` | E-08, E-13, E-28, E-29 | G-025..G-031 |
| `mla_qk` (q RoPE + cat) | E-08, E-13 | G-025, G-026, G-027 |

Both found all six. Graph is the stronger evidence: the graph agent had no
access to the names.

## 11. Substrate assessment: inherent vs method

| Capability | Eager | Graph | Inherent to substrate, or method? |
|---|---|---|---|
| Exact dataflow | approximate (address-keyed IDs, no frees) | exact | Inherent to graph. Eager's gap is Execution Trace tooling; fixable upstream (free events, stable tensor IDs) |
| Recompute visibility | inferred (Execution Trace crashes under `torch_remat`) | explicit nodes | Inherent to graph; eager's gap is a tooling bug |
| Phase and provenance labels | good (98.4% of backward linked) | exact | Both adequate |
| Root cause at kernel level | strong | possible (graph also has kernel names) but not done | Method (agent focus) |
| Override / integration points | direct (code span -> `Configurable`) | node sets -> graph-pass match | Inherent: each substrate's integration path |
| Systematic enumeration (cross-module, cross-layer, recompute->bwd, T3) | manual | algorithmic | Mostly method, made easy by the graph's exact dataflow and stages |
| Standalone single-kernel inefficiencies | found | missed when adjacent to a GEMM or collective | Method (graph generator design) |
| Program-specific waste (dead recompute, all-to-all concat) | visible | absent from the graph program | Inherent: each sees its own program |
| Horizontal fusion | not found | not found | Method gap in both |
| Outside-the-step work (optimizer, data) | visible | outside the graph | Inherent to graph |

Takeaway: the graph substrate gives a better foundation for automated
discovery (exact structure, explicit recompute, mechanical legality). The
eager substrate gives better diagnosis and a direct path to integration.
Some of the observed gaps belong to the agents' methods and are fixable on
either side:
- eager's lack of T3 and recompute->backward enumeration;
- graph's missing standalone regions and op-type tiering.

## 12. Side findings (not verified here)

| # | Finding | Evidence | Suggested next step |
|---|---|---|---|
| 1 | GraphTrainer `memory_policy="full"` runs out of memory at this config on HEAD | graph `report.md` section 1: `tag_sac_policy` Pass 1b makes in-place mutations and their consumers MUST_SAVE; combine all_to_all output 10.5 GiB per MoE layer; 274.6 GiB at layer 22 | Reproduce with the graph agent's recipe; check whether Pass 1b should exempt fresh-buffer `index_put_` chains |
| 2 | `ExecutionTraceObserver` crashes on `torch_remat` placeholder tensors (`std::get: wrong index for variant`) | eager `report.md` section 6; reproduced on the debug model | Minimal repro; file upstream (PyTorch Execution Trace or torch_remat) |
| 3 | Eager full recompute replays the MoE combine, whose outputs are dead in recompute (2.85 s per step here) | eager `report.md` section 6; section 3.3 here | Check `torch_remat` early stop / saved-region behavior for the combine under `FullActivationRematConfig` |
| 4 | Host syncs before each MoE token-count exchange: about 275 ms idle per step (graph) | graph `report.md` section 2 idle breakdown | Known design cost of variable-size all-to-all; relevant to CUDA-graph and overlap work |
| 5 | Fake-PG data is low-entropy, so GEMMs run about 25% faster (2.34 vs 1.8-1.9 PFLOP/s) | both calibrations | Treat fake-PG glue share as an upper bound; recalibrate on real data before claiming end-to-end wins |
| 6 | TensorIterator splits `[786432, 7168]` (>2^31 elements) ops into 16 launches | graph `report.md` section 6 | Folds into the combine kernels; also an upstream TensorIterator note |

## 13. Recommendations

**For the finders:**
1. **Graph:**
   - emit a standalone (non-T3) candidate for every glue node that runs far
     from roofline, even when its neighbours are a GEMM or a collective;
   - tier by semantics: detect permutation indices (from `sort`/`cumsum`
     provenance) and reduction-only constraints;
   - report one representative per overlap group, plus per-family totals;
   - try pairing recompute->backward across a GEMM, as in the routed-SwiGLU
     case.
2. **Eager:**
   - enumerate GEMM-neighbour and recompute->backward pairs explicitly;
   - get the Execution Trace fixed or worked around (free events, recompute
     support), since its dataflow is weakest exactly there.
3. **Both:**
   - add a horizontal-fusion pass (siblings sharing an input, or back-to-back
     independent kernels);
   - record the deterministic-kernel semantics catalog (section 9.1) in
     `BITWISE_RULES.md` before the kernel phase.

**For the next round of this experiment:**
1. Align the programs: same recompute set (or report per instance-pass) and
   the same all-to-all implementation, so step totals become comparable.
2. Run at least 2 agents per substrate to separate substrate effects from
   agent variance.
3. Keep plan docs out of the agents' reach; put each prompt in its own
   folder.
4. Use a held-out list of known opportunities, never shown to the agents, as
   the recall metric.

**For the fusion workflow:**
- Use the union. Graph's systematic enumeration and eager's diagnosis and
  override points complement each other.
- Kernel priorities common to both, in bitwise-safe order:
  1. permutation `index_put` -> gather/scatter (T0): E-04, E-06, E-07, E-12,
     about 3.0 s;
  2. combine forward (T2) and backward (T1);
  3. local-reorder backward (T2);
  4. routed SwiGLU, including recompute+backward (T0).

## Appendix A: Eager regions (all 30)

| ID | Title | Phases | Scope | Inst | Measured ms | Ideal ms | Gap ms | % step | Tier |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| E-01 | MoE combine: post-all-to-all concat + score multiply + bf16 cast + deterministic scatter-add | for+rec | intra | 116 | 5,693.3 | 204.0 | 5,489.2 | 13.30 | T2 |
| E-02 | MoE combine backward: scatter-add grad gather + score-multiply backward (dgrad, score grad reduction) | bac | intra | 58 | 2,854.9 | 192.7 | 2,662.2 | 6.45 | T1 |
| E-03 | MoE dispatch all-to-all backward concat + local-reorder backward (zeros + deterministic scatter-add of K=8 rows per token) | bac | intra | 58 | 1,365.1 | 102.0 | 1,263.0 | 3.06 | T2 |
| E-04 | MoE permute backward: zeros + deterministic index_put accumulate over a permutation | bac | intra | 58 | 1,347.4 | 181.3 | 1,166.1 | 2.83 | T0 |
| E-05 | Routed-expert SwiGLU recompute + backward in one pass (h for the w2 weight grad and d_gate_up from one read of gate_up) | rec+bac | rc-bwd | 58 | 1,309.4 | 155.3 | 1,154.1 | 2.80 | T0 |
| E-06 | MoE unpermute (expert-major -> rank-major) as deterministic index_put scatter | for+rec | intra | 116 | 1,402.9 | 362.6 | 1,040.3 | 2.52 | T0 |
| E-07 | MoE dispatch all-to-all concat + rank-major -> expert-major permute gather + permute index math | for+rec | intra | 116 | 913.7 | 362.6 | 551.1 | 1.33 | T0 |
| E-08 | Query RoPE (bf16->fp32 strided copy, complex multiply, fp32->bf16) + concat with q_nope | for+rec | x-mod | 122 | 670.2 | 163.8 | 506.4 | 1.23 | T0 |
| E-09 | Routed-expert SwiGLU: silu(gate) * up on strided halves of w13 output | for | intra | 58 | 512.0 | 77.7 | 434.3 | 1.05 | T0 |
| E-10 | MoE local reorder: token counts + argsort + score gather + token gather (all-to-all send buffer) | for+rec | intra | 116 | 551.7 | 204.6 | 347.1 | 0.84 | T2 |
| E-11 | MLA K/V layout glue: K materialization concat (fwd) and dK/dV pack + dk_pe head-sum (bwd) | for+rec+bac | intra | 183 | 279.5 | 0.0 | 279.5 | 0.68 | T3 |
| E-12 | MoE combine all-to-all backward concat + unpermute backward gather | bac | intra | 58 | 448.1 | 181.3 | 266.8 | 0.65 | T0 |
| E-13 | Query RoPE backward + q split backward concat | bac | x-mod | 61 | 326.8 | 81.9 | 244.9 | 0.59 | T0 |
| E-14 | Router group-limited top-k selection (bias add, group top-2 sum, group top-4, mask, top-8) | for | intra | 58 | 240.9 | 0.9 | 240.1 | 0.58 | T2 |
| E-15 | Shared-expert SwiGLU recompute + backward in one pass | rec+bac | rc-bwd | 58 | 137.8 | 19.4 | 118.4 | 0.29 | T0 |
| E-16 | Chunked cross-entropy: fp32 upcast + log_softmax + nll (fwd) and nll/log_softmax backward + bf16 cast (bwd) | for+bac | intra | 1 | 92.1 | 7.0 | 85.1 | 0.21 | T1 |
| E-17 | ffn_norm input-grad accumulation (routed + shared + router-gate) + RMSNorm backward + residual grad add | bac | x-mod | 61 | 145.2 | 70.3 | 74.9 | 0.18 | T1 |
| E-18 | Dense-FFN SwiGLU recompute + backward in one pass (layers 0-2) | rec+bac | rc-bwd | 3 | 75.7 | 9.0 | 66.6 | 0.16 | T0 |
| E-19 | Router + aux-loss backward (route-norm backward, gather backward, sigmoid backward, aux grads) | bac | intra | 58 | 60.7 | 1.9 | 58.8 | 0.14 | T1 |
| E-20 | MoE load-balance statistics (aux loss forward) | for+rec | intra | 116 | 55.5 | 2.0 | 53.5 | 0.13 | T1 |
| E-21 | attention_norm input-grad accumulation (wq_a + wkv_a) + RMSNorm backward + residual grad add | bac | x-mod | 61 | 110.5 | 59.6 | 51.0 | 0.12 | T1 |
| E-22 | Shared-expert SwiGLU forward | for | intra | 58 | 55.2 | 9.7 | 45.5 | 0.11 | T0 |
| E-23 | MoE output add (routed + shared) + block residual add + next layer's attention_norm (or final norm) | for | x-mod,x-layer | 61 | 97.8 | 59.0 | 38.9 | 0.09 | T1 |
| E-24 | Dense-FFN SwiGLU forward (layers 0-2) | for | intra | 3 | 30.5 | 4.5 | 26.0 | 0.06 | T0 |
| E-25 | Attention residual add + ffn_norm RMSNorm | for+rec | x-mod | 122 | 119.9 | 95.3 | 24.6 | 0.06 | T1 |
| E-26 | Router scoring: sigmoid + gather top-k scores + route norm + scale + routing map + token counts | for+rec | intra | 116 | 23.4 | 4.2 | 19.2 | 0.05 | T1 |
| E-27 | HiMidLo router-gate backward: split fp32 grad into bf16 hi/mid pieces + concat | bac | intra | 58 | 20.5 | 1.7 | 18.8 | 0.05 | T0 |
| E-28 | wkv_a output epilogue: kv contiguous copy + kv_norm RMSNorm + k_pe RoPE | for+rec | x-mod | 122 | 11.4 | 3.8 | 7.6 | 0.02 | T1 |
| E-29 | kv_norm backward + k_pe RoPE backward + kv split backward concat | bac | x-mod | 61 | 9.9 | 2.8 | 7.2 | 0.02 | T1 |
| E-30 | MoE dispatch split bookkeeping (count sums before the host copy) | for+rec | intra | 116 | 0.8 | 0.0 | 0.8 | 0.00 | T1 |

## Appendix B: Graph regions, greedy non-overlapping set (69 of 146)

Selected by descending gap, skipping any region that overlaps an
already-selected one. The full list is in `graph/regions.json`.

| ID | Title | Phases | Scope | Inst | Measured ms | Ideal ms | Gap ms | % step | Tier |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| G-001 | MoE combine backward + router/aux-loss score backward (one stage, per layer) | bac | x-mod | 58 | 3,107.6 | 205.3 | 2,902.4 | 7.63 | T2 |
| G-004 | MoE combine forward + routed/shared expert sum (score mul, cast, deterministic scatter-add, add) | for | x-mod | 58 | 2,643.2 | 130.8 | 2,512.4 | 6.60 | T2 |
| G-006 | Backward un-permute: zeros + index_put(accumulate=True) of routed-token grads (dispatch backward) | bac | intra | 116 | 2,634.1 | 297.3 | 2,336.8 | 6.14 | T2 |
| G-009 | T3: w2 dgrad grouped GEMM + SwiGLU backward epilogue (routed experts) | bac | x-mod | 58 | 1,593.6 | 567.4 | 1,026.2 | 2.70 | T3 |
| G-010 | T3: w2 grouped GEMM epilogue stores rows un-permuted (replace new_empty + index_put_), forward | for | x-mod | 58 | 1,554.8 | 567.4 | 987.4 | 2.60 | T3 |
| G-011 | T3: w2 grouped GEMM + un-permute index_put_ (recompute copy) | rec | x-mod | 58 | 1,539.8 | 567.4 | 972.4 | 2.56 | T3 |
| G-012 | T3: post-all_to_all token permutation gather fused into w13 grouped GEMM prologue (recompute) | rec | x-mod | 58 | 1,995.7 | 1,134.8 | 860.9 | 2.26 | T3 |
| G-013 | T3: post-all_to_all token permutation gather fused into w13 grouped GEMM prologue (forward) | for | x-mod | 58 | 1,983.2 | 1,134.8 | 848.4 | 2.23 | T3 |
| G-017 | T3: w13 wgrad grouped GEMM epilogue emits fp32 directly into the reduce-scatter bucket | bac | intra | 58 | 1,615.9 | 1,134.8 | 481.1 | 1.26 | T3 |
| G-018 | MoE router forward: sigmoid/bias/group-limited top-k/normalize + aux-loss stats + dispatch sort | for | x-mod | 57 | 550.3 | 106.3 | 444.1 | 1.17 | T2 |
| G-019 | Routed-expert SwiGLU forward: silu + mul on strided w13 halves (1 kernel instead of 2) | for | intra | 58 | 479.1 | 81.5 | 397.5 | 1.04 | T0 |
| G-020 | Routed-expert SwiGLU forward, recompute copy: silu + mul | rec | intra | 58 | 495.5 | 108.7 | 386.9 | 1.02 | T0 |
| G-021 | MoE router + dispatch prologue, recompute copy (sigmoid/topk/normalize/sort/gather) | rec | x-mod | 58 | 413.2 | 109.2 | 304.0 | 0.80 | T2 |
| G-024 | T3: w2 wgrad grouped GEMM epilogue emits fp32 (drop _to_copy) | bac | intra | 58 | 820.4 | 567.4 | 253.0 | 0.67 | T3 |
| G-025 | Q RoPE backward + grad concat: fp32 casts, conj-mul, cast, cat (backward) | bac | x-mod | 61 | 328.7 | 86.0 | 242.7 | 0.64 | T0 |
| G-026 | Q RoPE + Q concat, recompute copy (fp32 cast, complex mul, cast, cat) | rec | x-mod | 61 | 327.3 | 86.0 | 241.4 | 0.63 | T0 |
| G-027 | Q RoPE + Q concat (forward): fp32 cast, complex mul, bf16 cast, cat(q_nope, q_pe) | for | x-mod | 61 | 326.5 | 86.0 | 240.6 | 0.63 | T0 |
| G-039 | T3: attention.wo wgrad GEMM epilogue emits fp32 (drop _to_copy) | bac | intra | 61 | 714.8 | 596.8 | 118.0 | 0.31 | T3 |
| G-042 | T3: kv-grad concat folded into the wkv_b dgrad GEMM prologue (backward) | bac | x-mod | 61 | 223.1 | 115.5 | 107.6 | 0.28 | T3 |
| G-043 | T3: shared-expert w2 dgrad GEMM + SwiGLU backward epilogue | bac | x-mod | 58 | 169.9 | 70.9 | 99.0 | 0.26 | T3 |
| G-044 | T3: wkv_b GEMM epilogue writes K = cat(k_nope, broadcast k_pe) directly | for | x-mod | 61 | 187.5 | 101.3 | 86.2 | 0.23 | T3 |
| G-045 | T3: wkv_b GEMM epilogue writes K directly, recompute copy | rec | x-mod | 61 | 187.2 | 101.3 | 85.9 | 0.23 | T3 |
| G-046 | T3: attention.wo GEMM epilogue adds the residual, recompute copy | rec | x-mod | 61 | 676.3 | 596.8 | 79.6 | 0.21 | T3 |
| G-047 | T3: attention.wo GEMM epilogue adds the residual (forward) | for | x-mod | 61 | 670.9 | 596.8 | 74.1 | 0.20 | T3 |
| G-048 | T3: shared-expert SwiGLU fused into w2 GEMM prologue | for | x-mod | 58 | 137.7 | 70.9 | 66.7 | 0.17 | T3 |
| G-049 | T3: shared-expert w13 dgrad GEMM accumulates into the token grad (epilogue add) | bac | intra | 58 | 206.1 | 141.9 | 64.3 | 0.17 | T3 |
| G-053 | T3: chunked loss backward (nll/log_softmax backward + cast) fused into lm_head dgrad GEMM prologue | bac | x-mod | 8 | 135.9 | 77.2 | 58.7 | 0.15 | T3 |
| G-055 | T3: dense FFN (layers 0-2) w2 dgrad GEMM + SwiGLU backward epilogue | bac | x-mod | 3 | 83.9 | 33.0 | 50.8 | 0.13 | T3 |
| G-056 | T3: wq_a/wkv_a dgrad GEMM accumulates into the token grad (epilogue add) | bac | x-mod | 61 | 103.1 | 55.9 | 47.1 | 0.12 | T3 |
| G-060 | T3: wq_b dgrad GEMM + q_norm RMSNorm backward epilogue | bac | x-mod | 61 | 235.3 | 191.8 | 43.5 | 0.11 | T3 |
| G-061 | Shared-expert SwiGLU forward, recompute copy | rec | intra | 58 | 56.3 | 13.6 | 42.8 | 0.11 | T0 |
| G-062 | T3: wq_a wgrad GEMM epilogue emits fp32 into the reduce-scatter bucket | bac | intra | 61 | 98.0 | 56.0 | 42.0 | 0.11 | T3 |
| G-063 | T3: chunked loss: lm_head GEMM + fp32 upcast + log_softmax + nll (fused linear cross-entropy) | for | x-mod | 1 | 52.8 | 11.1 | 41.7 | 0.11 | T3 |
| G-064 | T3: q_norm RMSNorm fused into the wq_b GEMM prologue (forward) | for | x-mod | 61 | 231.9 | 191.8 | 40.0 | 0.10 | T3 |
| G-065 | T3: router-gate dgrad GEMM accumulates into the token grad (epilogue add) | bac | intra | 58 | 64.6 | 26.6 | 38.0 | 0.10 | T3 |
| G-069 | T3: attention_norm RMSNorm fused into the wq_a GEMM prologue (forward) | for | x-mod | 61 | 91.6 | 55.9 | 35.6 | 0.09 | T3 |
| G-070 | T3 GEMM epilogue (backward): feed_forward.shared_experts.w13: mm + _to_copy + cat | bac | intra | 58 | 177.2 | 141.9 | 35.3 | 0.09 | T3 |
| G-071 | T3 GEMM prologue (forward): feed_forward, feed_forward.w2: silu + mul + mm | for | x-mod | 3 | 68.0 | 33.0 | 34.9 | 0.09 | T3 |
| G-081 | T3 GEMM epilogue (backward): attention.wq_b: mm + _to_copy | bac | intra | 61 | 221.3 | 191.8 | 29.5 | 0.08 | T3 |
| G-083 | T3 GEMM epilogue (backward): attention.inner_attention.wkv_b: mm + _to_copy | bac | intra | 61 | 112.5 | 85.3 | 27.2 | 0.07 | T3 |
| G-088 | Recomputed ffn_norm fused with its RMSNorm backward + residual-grad add | bac+rec | intra,rc-bwd | 60 | 98.8 | 73.8 | 25.0 | 0.07 | T1 |
| G-089 | Recomputed attention_norm fused with its RMSNorm backward + residual-grad add | bac+rec | intra,rc-bwd | 61 | 99.8 | 75.0 | 24.8 | 0.07 | T1 |
| G-090 | T3 GEMM epilogue (backward): feed_forward.w13: mm + _to_copy + cat | bac | intra | 3 | 89.3 | 66.0 | 23.3 | 0.06 | T3 |
| G-091 | Intra-module (recompute): feed_forward: silu + mul | rec | intra | 3 | 28.9 | 6.3 | 22.6 | 0.06 | T0 |
| G-094 | T3 GEMM prologue (forward): feed_forward.shared_experts.w13: cat + mm | for | intra | 58 | 163.4 | 141.9 | 21.5 | 0.06 | T3 |
| G-097 | T3 GEMM prologue (backward/recompute): feed_forward.w13: cat + mm | bac+rec | intra,rc-bwd | 3 | 84.9 | 66.0 | 18.9 | 0.05 | T3 |
| G-098 | T3 GEMM epilogue (backward): lm_head: mm + _to_copy | bac | intra | 8 | 95.2 | 77.2 | 18.0 | 0.05 | T3 |
| G-101 | T3 GEMM epilogue (recompute): attention.q_norm, attention.wq_a: mm + _fused_rms_norm | rec | x-mod | 61 | 72.1 | 55.9 | 16.1 | 0.04 | T3 |
| G-102 | T3 GEMM epilogue (backward): feed_forward.shared_experts.w2: mm + _to_copy | bac | intra | 58 | 85.1 | 70.9 | 14.1 | 0.04 | T3 |
| G-103 | T3 GEMM prologue (forward): feed_forward.w13, ffn_norm: _fused_rms_norm + mm | for | x-mod | 3 | 79.9 | 66.0 | 13.9 | 0.04 | T3 |
| G-104 | T3 GEMM epilogue (backward): feed_forward.w2: mm + _to_copy | bac | intra | 3 | 46.5 | 33.0 | 13.5 | 0.04 | T3 |
| G-116 | T3 GEMM epilogue (backward): attention.wkv_a: mm + _to_copy | bac | intra | 61 | 30.1 | 21.0 | 9.1 | 0.02 | T3 |
| G-117 | T3 GEMM prologue (backward): attention, attention.wkv_a: cat + mm | bac | x-mod | 61 | 30.0 | 21.0 | 9.0 | 0.02 | T3 |
| G-118 | Cross-module (forward): feed_forward.routed_experts, feed_forward.router, feed_forward.router.aux_loss: sigmoid + add + topk + sum + ones_like + scatter_.value + masked_fill.Scalar + gather + di | for | x-mod | 1 | 10.4 | 1.9 | 8.5 | 0.02 | T2 |
| G-121 | Embedding backward (dense scatter-add) + fp32 cast + bucket pack | bac | intra | 1 | 7.4 | 0.7 | 6.7 | 0.02 | T2 |
| G-122 | Recompute->backward (backward/recompute): attention.kv_norm: _fused_rms_norm + _fused_rms_norm_backward | bac+rec | intra,rc-bwd | 61 | 9.7 | 3.6 | 6.1 | 0.02 | T1 |
| G-123 | T3 GEMM epilogue (backward): feed_forward.router.gate: mm.dtype + add_ | bac | intra | 116 | 30.2 | 25.1 | 5.1 | 0.01 | T3 |
| G-128 | Intra-module (recompute): feed_forward.routed_experts: arange + repeat_interleave + index_select + index + add + sub | rec | intra | 58 | 3.9 | 0.1 | 3.8 | 0.01 | T2 |
| G-130 | Intra-module (forward): feed_forward.routed_experts: arange + repeat_interleave + index_select + index + add + sub | for | intra | 58 | 3.7 | 0.1 | 3.6 | 0.01 | T2 |
| G-131 | Intra-module (forward): feed_forward.routed_experts: sum + _to_copy + cumsum + sub + clone | for | intra | 58 | 1.8 | 0.0 | 1.8 | 0.01 | T1 |
| G-132 | Recompute->backward (backward/recompute): attention.wkv_a, attention.wq_a: _to_copy + constant_pad_nd + bkt._pre_bucket_all_gather | bac+rec | x-mod,rc-bwd | 61 | 1.9 | 0.1 | 1.8 | 0.01 | T0 |
| G-133 | Cross-module (forward): attention.wkv_a, attention.wq_a: _to_copy + constant_pad_nd + bkt._pre_bucket_all_gather | for | x-mod | 61 | 1.8 | 0.1 | 1.7 | 0.01 | T0 |
| G-134 | Intra-module (recompute): feed_forward.routed_experts: cumsum + sub + clone + sum | rec | intra | 58 | 1.6 | 0.0 | 1.6 | 0.00 | T1 |
| G-136 | Recompute->backward (backward/recompute): norm: _fused_rms_norm + _fused_rms_norm_backward | bac+rec | intra,rc-bwd | 1 | 1.1 | 0.6 | 0.5 | 0.00 | T1 |
| G-138 | Recompute->backward (backward/recompute): ffn_norm: _fused_rms_norm + _fused_rms_norm_backward | bac+rec | intra,rc-bwd | 1 | 1.0 | 0.8 | 0.2 | 0.00 | T1 |
| G-141 | Cross-layer (forward): layers.*, norm: add + _fused_rms_norm | for | x-layer | 1 | 1.0 | 0.8 | 0.2 | 0.00 | T1 |
| G-144 | Intra-module (backward): loss: add_ | bac | intra | 1 | 0.0 | 0.0 | 0.0 | 0.00 | T0 |
| G-145 | Intra-module (forward): feed_forward.router.aux_loss: clamp_min + reciprocal + mul | for | intra | 1 | 0.0 | 0.0 | 0.0 | 0.00 | T0 |
| G-146 | Intra-module (backward): norm: _to_copy + bkt._pre_bucket_reduce_scatter | bac | intra | 1 | 0.0 | 0.0 | 0.0 | 0.00 | T0 |

## Appendix C: Data and reproduction

- Eager:
  - `eager/run_capture.sh`, `eager/recipe/eager_fusion_recipe.py`;
  - `eager/scripts/{calibrate_roofline,build_op_table,dataflow,regions,validate_regions,make_report}.py`;
  - op table `eager/analysis/op_table.parquet`.
- Graph:
  - `graph/scripts/profile_dsv3_fake_graph.sh`,
    `graph/scripts/fusion_finder_passes.py`;
  - `graph/scripts/{calibrate_roofline,parse_trace,graph_lib,build_node_table,find_regions,curate_regions,validate_outputs}.py`;
  - node table `graph/analysis/node_table.csv.gz`.
- This report's cross-checks (sections 3.2, 3.3, 5, 6, appendices) are
  computed from the two tables and the two `regions.json` files. The
  `index_put` semantics (9.1) come from the PyTorch source at
  `/home/bahuang/local/pytorch`.
