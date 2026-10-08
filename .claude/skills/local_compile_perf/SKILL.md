---
name: local-compile-perf
description: Make a torchtitan model train faster with `local_compile` regions and compile-friendly rewrites: measure, find the slow glue, rewrite it, verify. Use when asked to speed up a model, layer or op with torch.compile, or to review such a change.
---

# Speeding up a model with local compile

The method: measure a model's layers against their ideal time, find the glue ops that run far slower than the bytes they move, and rewrite or regroup them so Inductor fuses them into fast kernels. Then verify numerics, graph counts and parallelism, and deliver small changes.

Terms:
- **Region:** a function decorated with `@local_compile` (`torchtitan/distributed/local_compile.py`), compiled on its own with `torch.compile(fullgraph=True)` when listed in the model's `local_compile_regions`. It traces through everything it calls (callees are inlined); `fullgraph` only forbids graph breaks.
- **Glue:** everything that isn't a GEMM or attention: norms, RoPE, residual adds, routing, permutes, casts, scores.
- **Ideal time:** minimum bytes moved / measured HBM bandwidth (or a FLOP floor for GEMMs).
- **Per-layer (block) compile:** `torch.compile` on each transformer block. Used here as a baseline, not a deliverable.
- **Graph budget:** compiled graphs per region; `recompile_limit` is 8 by default and `fullgraph` turns overflow into an error.
- **Gate:** an if-branch that picks the rewritten form only when compiling.
- **fp32 matmul precision:** e.g. BF16x9, fp32 matmuls emulated on bf16 tensor cores (`torch.backends.cuda.matmul.fp32_precision`). Trainers differ across trees; match what this tree's trainer sets.
- **SAC:** selective activation checkpointing. **GraphTrainer:** torchtitan's whole-step tracer (make_fx + Inductor).

## Mental model

```text
eager:   x -> [norm] -> HBM -> [rope] -> HBM -> [cat] -> HBM -> out   (3 kernels, 3 round trips)
region:  x -> [norm + rope + cat] -> out                              (1 kernel, inputs read once)
win available = measured glue time - ideal time
```

- GEMMs and attention dominate a step; compile rarely speeds them up. The budget is the glue.
- Compile wins when Inductor can see and fuse the ops. Complex numbers, opaque custom ops, extern GEMV, data-dependent shapes and symbolic small sizes block or degrade fusion; rewriting around them is most of the work.
- Every region costs graphs, cold-compile seconds and host launch time. At small token counts and under CUDA graphs, host time can cancel kernel wins.

## Sizing a region

Aim for the smallest region that captures the win:

```text
C() = A() + B()
only A wins compiled                                  -> region(A)
A and B win independently                             -> region(A), region(B)
fused kernels span A|B and beat A + B compiled apart  -> region(C)   (C inherits all call signatures -> more graphs)
```

Example: DSv4 q norm + RoPE went from 1497 µs eager to 1044 µs in one region, because the cat fused into the norm's reduction.

## Workflow

```text
0 plan (devlog, stop rule)
1 baselines ............. harness.md
2 ideal times ........... measurement.md
3 profile + diff ........ harness.md (profiler) -> symptoms.md
4 rank candidates
5 pick a fix ............ patterns.md, gating-and-options.md
6 graph budget .......... measurement.md#graph-budget
7 screen, then confirm .. measurement.md
8 correctness, AC, TP/EP  measurement.md, gating-and-options.md, deceiving-quirks.md
9 commit + scope
10 stack + findings doc . Output (below), pr-template.md
11 feed the skill back
Before investing in an idea: negative-results.md. If compile itself fails: compile-limits.md.
```

0. **Plan.** Set a time budget and a stop rule: stop when the remaining gap to ideal is under ~1% of the layer, or the next candidate's estimated gain is below that. Keep a devlog: one file per effort, a timestamped entry appended right after each measurement, decision and dead end (not in batches). Check how this tree declares regions (harness.md: where regions live) and what fused kernels already exist under `torchtitan_recipes/overrides/` (e.g. `fused_mla`); benchmark them as alternatives.
1. **Baselines** (harness.md):
   - 2 layers at real dims, covering each block type; packed tokens per rank from small to large (e.g. 2k, 4k, 16k, 32k+). Torchtitan has no batch dim, so long rows are normal.
   - fwd+bwd, with the recipe's activation checkpointing mode (none, full, selective).
   - three trees: eager, upstream regions (what the model's `local_compile_regions` enables on main today), per-layer compile. Rewrites plus regions beat per-layer compile by 18-29% on Kimi K3, so it's a baseline, not a ceiling.
   - fp32 matmul precision matching this tree's trainer.
   - check that the expected attention backend actually loaded (kernel names in the profile) and pin optional deps (requirements.txt); a missing FA4 or a stale attn-gym silently falls back to a slow path (deceiving-quirks.md).
   - record wall and kernel time, peak memory, kernel count; check attention against its FLOP floor. When attention is the largest item and far from its floor, it's a backend choice (flex, varlen, FA4, cuDNN): report it with the floor rather than trying a region.
   - per-layer compile caveats: attention inlined into an outer compile loses its own region options (flex attention got 25% slower), so compare glue only; `torch.compile` on a SAC `CheckpointWrapper` silently ran eager kernels.
2. **Ideal times** per glue op; build a measured / ideal table.
3. **Profile and diff** eager vs regions vs per-layer compile by kernel; look up each symptom in symptoms.md.
4. **Rank** by (measured − ideal) × share of the layer. Items under ~1% are usually not worth a PR unless they fix correctness or remove a host sync.
5. **Pick a fix**, roughly cheapest first:
   ```text
   enable an existing region
   resize a region (A / B / C above)
   rewrite the function so it lowers well (patterns.md)
   add a region-scoped option (gating-and-options.md)
   autograd.Function with a hand-written backward
   ```
6. **Graph budget, early:** count all shapes and call sites plus a `no_grad` pass. Levers: `mark_static`, a dynamic token dim marked in eager, fewer signatures per region, `triton.mix_order_reduction_non_strict_mode`, raising the limit (we used 16 in several models; how depends on the tree, see measurement.md). Details in measurement.md.
7. **Screen in one process, confirm with 3 fresh processes** on a quiet GPU.
8. **Correctness and side effects:** measurement.md (correctness) and gating-and-options.md (numerics, AC, parallelism); check deceiving-quirks.md.
9. **Commit each validated change locally** before the next. Enable new regions in the target model's config or recipe, not a shared default list. A region enabled in a model config also reaches every config that subclasses it (DSv3 -> Kimi K2.7): check subclasses or enable per recipe. List which recipes reach the change.
10. **Package** the stack and findings doc (Output).
11. **Feed the skill back:** list what this skill missed (new patterns, edge cases, negative results, wrong advice, new compile limits for compile-limits.md) as proposed additions.

## Output

- **A diff stack:** one small branch per change, each on the previous one, committed locally, ordered by impact (biggest first; state the order). Ask the user whether to open it as draft PRs, and where, or not at all. Don't open PRs without asking, and don't reference other people's PRs or issues.
- **A findings doc** (short): what was measured, the stack with deltas, rejected ideas, not tested, skill feedback. Its top block is the stack summary in every PR.
- **PR text:** TLDR, Problem and Solution as numbered baby steps, a diagram, numbers in code blocks; no paragraphs or tables. Template, filled example and rules: pr-template.md.
- **Code:** minimal diff; regions enabled per model config; a test that fails before and passes after; code rules in pr-template.md.
- **Compute:**
  - iterate on one GPU;
  - multi-GPU only for what one GPU can't show (comm overlap, EP/TP behavior, zero-token ranks): build-check the harness at the target world size (meta tensors or a fake process group), then submit one small, short job and keep working;
  - fake process groups and FakeTensor for tracing and shape checks only; above world 1, fake peers alias storage and give garbage routing;
  - parallel subagents for independent work; a fresh non-fork agent for blind checks.

## What usually saves time

Lessons from specific cases, not rules:

- **Measure the function as written under compile before rewriting.** Inductor sometimes already does well: a compiled scatter matched our gather rewrite.
- **Move work to the narrower tensor.** Applying routing scores to the 2048-wide SwiGLU output inside its region, instead of the 7168-wide w2 output in eager, saved 13% of a MoE layer and 29% peak memory.
- More in negative-results.md.

## Checklist

- [ ] Plan: budget, stop rule, devlog.
- [ ] Baselines (eager / upstream / per-layer) at 3+ token counts, recipe AC mode, trainer's fp32 matmul precision.
- [ ] Ideal-time table; kernel diff; ranked candidates (ms, % of layer).
- [ ] Region boundaries chosen by measurement.
- [ ] Graph count: all shapes + `no_grad`, headroom under the limit.
- [ ] 3 fresh processes; outputs and all grads vs eager (bf16, fp32) and fp64.
- [ ] Gate chosen from eager cost (gating-and-options.md).
- [ ] Quirks checked: TP, EP, hardware, dispatch mode, AC, compile mode.
- [ ] Regions per model config; reaching recipes listed; each change committed.
- [ ] Stack + findings doc; asked before opening any PR; skill feedback.
