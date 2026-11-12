# Protocol: is the graph a better substrate than eager for finding fusion regions?

Written 2026-11-10, before any capture or agent run. Changes after this point go to "Deviations" at the end,
with the reason.

## Question

With the agent, prompt, tooling, program and scorer held equal, does a region-finder agent working on
GraphTrainer's FX graph find more of the achievable fusion saving than one working on the eager code, as
precisely, at no higher cost?

## Arms and models

- **E (eager):** model source, eager profiler trace with module scopes and Python stacks, execution trace.
- **G (graph):** final joint FX graph (node records with module stack, stack trace, shapes, seq_nr), graph
  profiler trace with per-node labels, `graph_trainer` code. Model source only to name a region already found.
- One run per arm per model, 6 runs.

| Model | Recipe base | Layout (fake PG, one GB300) | Tokens per microbatch |
|---|---|---|---|
| DSv3 671B | `deepseek_v3_671b_dist_moe_mxfp8` | FSDP 256 x EP 64 | 4,096 |
| Qwen3 30B-A3B | `qwen3_30b_a3b_dist_moe_mxfp8_8gpu`, FA4 varlen | FSDP 8 x EP 8 | 32,768 |
| Llama3 8B | `llama3_8b(seq_len=8192)`, test tokenizer, C4 sample | FSDP 8 | 8,192 |

## Matched program

- The graph config is the eager config converted with `to_graph_trainer_config` (plus the recipe family's
  required Dist-MoE setting), so both run the same recipe.
- No recompute on either side: eager `activation_remat = None`, graph `memory_policy = "none"`.
- `local_compile.regions = []`, deterministic, seed 42, one microbatch per step, CUDA graphs off.
- Remaining differences (FSDP2 vs GraphTrainer's FSDP, functional weight-gradient accumulation) are measured
  in a program-difference table (GPU time by category) before the agents run.

## Prepared artifacts (equal for both arms)

- The experimenter captures one profiled step (step 10, 3 warmup steps) per model and arm, back to back on
  the same GPU, with the same Kineto settings (CPU + CUDA, shapes, Python stacks).
- E adds module-scope `record_function`s and an execution trace; G adds the graph dumps and per-node labels.
- One shared `hw_profile.json` (HBM bandwidth, GEMM peaks, measured with random data) for every run.
- Agents don't capture, profile or use GPUs.

## Leakage controls

- Agents work in a fresh single-commit snapshot of the repo (no history), without `agentic_compiler/`, the
  `fusion_finder` / `autokernel` / `local_compile_perf` skills, or any earlier finder output.
- Each agent sees only the snapshot, its own artifacts and its own output directory.
- Fresh subagents (not forks), no skills, no subagents of their own.
- Transcripts are audited afterwards for reads outside the allowed paths and for skill use.

## Prompt symmetry

One template; the two prompts for a model differ only in the substrate paragraph and the artifact list.
Both list every opportunity kind: within a module, across modules, across layers, forward + backward, glue
around communication and sharding, GEMM / attention prologue and epilogue, and sibling (horizontal) fusion.
Same 90-minute cap, same output schema, regions logged with timestamps as found.

## Ground truth and scoring

- **Pool** per model: every region from both arms.
- **Common key:** each region lists its footprint as (phase, module path with layer indices as `*`, layers,
  normalized op names), plus a substrate-neutral description.
- **One judge agent per model, blind to the arm:** cards are rendered from structured fields with random IDs,
  random order, and substrate vocabulary scrubbed. The judge groups cards into opportunities, decides
  whether each opportunity is real, its bitwise tier and category, and its achievable saving (ms per step)
  from the shared roofline, and records how much of each opportunity each card covers.
- **Recall (primary):** sum over real opportunities of saving x the arm's best coverage, divided by the
  total real saving. Reported overall, without GEMM-fused (T3) opportunities, and without
  communication/sharding glue.
- **Precision:** judged saving covered / claimed gap; share of cards judged not real.
- **Also:** unique-to-one-arm opportunities, wall time and tool calls, recall over time.

## Decision rule

With one run per arm there is no estimate of run-to-run variance. The graph is called better for finding
if its recall without T3 exceeds eager's by at least 10 points on at least 2 of 3 models, and eager's does
not exceed the graph's by 10 points on the third. Otherwise the result is reported as inconclusive.
Precision and cost are reported, not folded into the call.

## Deviations
- 2026-11-10, before any agent run: the snapshot also drops the "Autoresearch" paragraph of
  `graph_trainer/MANIFESTO.md` ("a better substrate for agentic optimization than eager code"), which only
  the graph agent would read. Prompts don't mention a comparison or a sibling agent. Artifact paths were
  rewritten to point at the snapshot (same commit) and the capture tool's name was replaced by "capture".
- Program differences measured before the agents ran (`program_diff.json`, ms per step, excluding fake-PG
  communication): Llama3 8B eager 344.5 / graph 353.0 (glue 134.0 / 140.8); Qwen3 30B-A3B 986.0 / 967.4
  (glue 511.2 / 492.6); DSv3 671B 546.6 / 487.2 (glue 331.5 / 271.1; eager has more copy glue). Step-1 loss
  matches between arms for Llama3 and Qwen3; DSv3 differs (12.24389 vs 12.28130), as in the earlier DSv3
  eager study.
- DSv3 finders started ~25 min after the other four, when the DSv3 graph capture finished.
- The DSv3 eager agent could not write `report.md` (harness refusal); the experimenter generated sections 1-5
  with the agent's own `write_report.py` and appended its returned sections 6-7. `regions.json`, the scored
  output, was written by the agent.
- Audit refinements after the first transcripts (no effect on any run): the installed `dist_moe` source path is
  allowed (the prompt allows installed package source), the agent's own harness-spilled outputs are allowed,
  and the eager substrate-rule regex no longer matches the recipe module path
  `torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks`.
- Post hoc (not pre-registered): robustness checks in `scoring/robustness.py` -- recall with verified eager-only
  program work removed (backward re-gather, persistent MXFP8 operand copy), and recall excluding the judge's
  `communication-glue` and `weight-prep` categories.
- Post hoc (not pre-registered): raw opportunity counts per arm without saving estimates (`scoring/counts.py`;
  found = best coverage >= 0.5), and a comparison of the six transcripts' working steps.
