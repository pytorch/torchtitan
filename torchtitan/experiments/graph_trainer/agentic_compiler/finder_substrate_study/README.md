# Eager vs graph as the substrate for finding fusion regions

A controlled study, run 2026-11-10. It asks whether a region-finder agent finds more fusion opportunities
when it works on GraphTrainer's FX graph than when it works on the eager code, with everything else held
equal. `PROTOCOL.md` was written before any run; its last section lists the deviations.

## Headline

- **Both arms found nearly the same opportunities.** Of the real opportunities in the pooled ground truth,
  eager found 31, 32 and 42 and graph 27, 31 and 38 (Llama3 8B, Qwen3 30B-A3B, DSv3 671B). Once the work that
  only the eager program does is dropped, eager leads by one opportunity on each model.
- **The pre-registered call is inconclusive.** Weighted by estimated saving, both arms cover 88-99% of the
  program they share. Neither is ahead on every model.
- **The graph made the agents cheaper, not better at finding.** Attribution was nearly free on the graph and
  the backward was explicit. Graph agents used 17-24 fewer tool calls and 2-14 fewer minutes, but both arms
  searched the same way: read one layer's ops, match chains against the prompt's list of fusion kinds, and
  hand-write each region.
- **The 2026-11-04 pilot's graph advantage came from its prompt.** There the eager finder excluded FSDP glue as
  owned by FSDP2, and GEMM-adjacent fusion was not listed. Here both prompts say library-owned glue counts and
  list every kind, and the eager finders found that glue.

## Setup

| | Llama3 8B | Qwen3 30B-A3B | DSv3 671B |
|---|---|---|---|
| Recipe | `llama3_8b(seq_len=8192)`, BF16 | `qwen3_30b_a3b_dist_moe_mxfp8_8gpu`, FA4 varlen | `deepseek_v3_671b_dist_moe_mxfp8` |
| Layout (fake PG, one GB300) | FSDP 8 | FSDP 8 x EP 8 | FSDP 256 x EP 64 |
| Tokens per microbatch | 8,192 | 32,768 | 4,096 |
| GPU ms per step excluding communication, eager / graph | 344.5 / 353.0 | 986.0 / 967.4 | 546.6 / 487.2 |
| Step-1 loss, eager vs graph | identical | identical | differs (12.24389 vs 12.28130) |

- **Same program:** both arms ran the same recipe with no recompute, `local_compile` off, deterministic mode
  and one microbatch. The graph config is the eager config converted with `to_graph_trainer_config`.
- **Same inputs:** the experimenter captured step 10 of both arms with the same profiler settings, and every
  agent used one shared `hw_profile.json`. Eager also got module-scope annotations and an execution trace;
  graph got the final joint graph and per-node trace labels.
- **No leaks:** each agent worked in a single-commit snapshot of the repo, without git history, earlier
  results, the finder skills, or the `graph_trainer` manifesto paragraph that calls the graph "a better
  substrate". Prompts come from one template (`prompts/gen.py` checks that the two differ only in the
  substrate block), list every opportunity kind, and never mention a comparison.
- **Agents:** fresh subagents, one run per arm and model, no skills, no subagents, no GPU, 90-minute cap.
- **Ground truth:** every region of both arms was rendered as a card with a random ID, in random order, with
  substrate vocabulary removed. One judge agent per model grouped the cards into opportunities, decided which
  are real, estimated each one's saving from the roofline, and recorded how much of each opportunity each
  card covers.

## Result 1: raw counts

An arm found an opportunity if one of its regions covers at least half of it. "Shared program" drops the
opportunities that exist only in the eager program (next section). Source: `scoring/counts.py`.

| | Llama3 8B | Qwen3 30B-A3B | DSv3 671B |
|---|---:|---:|---:|
| Regions submitted, eager / graph | 27 / 28 | 34 / 21 | 31 / 33 |
| Opportunities after grouping (judged not real) | 33 (1) | 41 (3) | 47 (0) |
| Real opportunities found by both | 26 | 25 | 33 |
| Found only by eager / only by graph | 5 / 1 | 7 / 6 | 9 / 5 |
| **Found, eager / graph** | **31 / 27** | **32 / 31** | **42 / 38** |
| **Found on the shared program, eager / graph** | **28 / 27** | **32 / 31** | **39 / 38** |

Backward opportunities only (found by both / only eager / only graph), shared program: Llama3 11 / 1 / 0,
Qwen3 14 / 2 / 4, DSv3 15 / 1 / 4.

## Result 2: recall weighted by estimated saving

Recall is the estimated saving an arm's regions cover, divided by the total estimated saving of the real
opportunities. The savings are roofline estimates (measured time of the ops minus the ideal time of a fused
kernel); no kernels were written in this study.

| | Llama3 8B | Qwen3 30B-A3B | DSv3 671B |
|---|---:|---:|---:|
| Total estimated saving (ms per step) | 118.1 | 369.1 | 218.9 |
| Recall, eager / graph (pre-registered) | 95.4 / 94.6 | 99.2 / 90.2 | 90.8 / 71.7 |
| Recall without GEMM-fused (T3) opportunities (decision metric) | 98.8 / 94.2 | 99.2 / 95.1 | 96.0 / 72.3 |
| Recall on the shared program | 95.2 / 98.2 | 99.2 / 91.9 | 88.2 / 92.4 |
| Recall without communication glue, weight prep and T3 | 98.1 / 98.3 | 99.7 / 99.3 | 98.6 / 98.3 |

**Decision** (pre-registered: recall without T3, 10-point threshold on 2 of 3 models): graph minus eager is
-4.6, -4.1 and -23.8 points, so the call is inconclusive. The shared-program rows were added after the runs.

## Why eager's raw numbers lead

Most opportunities only eager found are work the graph program doesn't do:
- **Backward re-gather of weights.** Eager's FSDP2 frees gathered weights after the forward and gathers them
  again in the backward (33, 49 and 119 backward all-gathers). The graph programs gather once, in the forward.
  The re-gather's copy-in, copy-out and MXFP8 re-quantization are eager-only opportunities.
- **Copy into persistent MXFP8 operand storage (DSv3).** Eager's FSDP tensor subclass quantizes on unshard and
  copies into persistent buffers; the graph quantizes inside the linear's forward.

The graph program has a small runtime-only cost too (gradient clones into `param.grad` after the graph); it
is mixed into opportunities with shared work, so it stays in.

## What stays unique

Both arms found every large family: RoPE, router, loss, norm + quantization, MoE dispatch and combine glue,
and FSDP pack and unpack. The unique finds on the shared program are mostly GEMM- or attention-epilogue
ideas, in both directions:
- **Only eager:** the flash-attention backward preprocess in the wo dgrad GEMM epilogue (Llama3, Qwen3), the
  flash-attention backward postprocess fold and a wqkv epilogue that emits normed, rotated q/k (Qwen3), the
  routed-expert `broadcast_and_scale` in the expert GEMMs (DSv3), plus a few small items.
- **Only graph:** the same flash-attention backward preprocess idea (DSv3), MXFP8 dY quantized straight from
  flash-attention's dQ accumulator, the router-gate weight-gradient GEMMs merged, the reduce-scatter output
  cast written into `param.grad` (DSv3), and small cross-module items (Qwen3).

The same idea going to eager on two models and to graph on the third looks like run-to-run variance, not
substrate.

## How the agents worked

Read from the six transcripts. All agents followed the same steps: attribute kernels to a table, print one
representative layer's forward and backward in launch order, read it to spot fusible chains, write each
region as a hand-written filter over the table, make a second pass over the list of kinds, consolidate.

| | Eager | Graph |
|---|---|---|
| Kernel -> op | correlation ID -> launch call -> call tree rebuilt from timestamps | correlation ID -> enclosing node label |
| Module of a backward op | autograd node's sequence number -> forward op -> its module annotation | `module_fqn` on the node |
| Shapes and bytes | execution trace joined by `rf_id` | node `val` |
| Attribution problems | FSDP's backward all-gather runs inside an unrelated autograd node; nodes without sequence numbers; ops before the first module | labels without `file:line`; post-graph gradient clones outside the graph; gradient-accumulation adds without `module_fqn` |
| Dataflow claims | from model source, op order and autograd knowledge | node arguments read from the graph code |
| Opaque library ops (Dist-MoE, flash-attention, FSDP2) | Python stacks + library source | the same for custom ops |
| Region bytes | formulas over model dimensions | formulas (Llama3, Qwen3); computed from graph edges (DSv3) |

| | Llama3 E / G | Qwen3 E / G | DSv3 E / G |
|---|---|---|---|
| Minute a correctly attributed one-layer listing existed | 9.5 / 3.3 | 6.0 / 2.9 | 9.7 / 5.1 |
| Wall time (min) | 22 / 20 | 26 / 24 | 40 / 26 |
| Tool calls | 76 / 59 | 81 / 64 | 87 / 63 |
| Tool calls reading model source / library source | 19 / 3 vs 0 / 2 | 19 / 24 vs 0 / 15 | 15 / 33 vs 0 / 16 |
| Subagent tokens (k) | 322 / 264 | 339 / 306 | 371 / 314 |

DSv3 eager's extra time went mostly to tracing FSDP and MXFP8 copies to their Python call sites in a very
large trace. Neither arm enumerated fusible chains mechanically from the dataflow, so the graph's exact edges
saved time but did not change what the search covered.

## Audit

Every finder and judge transcript was scanned (`scoring/audit_*.json`): no skills, subagents, web or GPU use;
no reads of the main repo, earlier results or other sandboxes; no eager reads of `graph_trainer` code and no
FX tracing; no judge reads of finder outputs. Every prompt matches its file byte for byte.

## Caveats

- **One run per arm.** Differences of a few opportunities or points can be agent variance.
- **Ground truth is the pool of what the two arms found,** judged by one LLM agent. Opportunities neither arm
  found are invisible. Savings are roofline estimates, softest for T3 and FSDP.
- **Programs matched by recipe, not by runtime:** FSDP2 vs GraphTrainer's FSDP differ (backward re-gather,
  persistent operand copies, post-graph clones).
- **Fake process group on one GPU:** real expert-parallel communication glue is invisible to both arms.

## Files

- `PROTOCOL.md`: the pre-registered protocol and deviations.
- `prompts/`: template, generator with the symmetry check, six finder prompts, three judge prompts.
- `capture/`: capture recipes and scripts; `hw_profile.json`; `program_diff.json`.
- `scoring/`: card pooling (`pool.py`), judge prompt generation, scoring (`score.py`, `robustness.py`,
  `counts.py`), transcript audit, the unblinding keys (`keys/`) and the result JSONs.
- `results/<model>_<arm>/`: each finder's `regions.json` and `report.md`.
- `results/judge_<model>/`: the blind cards, the overlap hints and the judgement.

The scripts ran in the experimenter's workspace (`outputs/finder_substrate/`, agent sandboxes under
`/home/bahuang/local/step_analysis/`) and keep those paths.
