# Autoresearch

Autoresearch is an autonomous experiment loop: an LLM agent iterates on
GPU-kernel / graph-pass optimizations for a training graph, making it
faster while preserving bitwise-identical numerics to the eager
reference. It runs indefinitely until manually stopped.

This file is the agent's operating manual. Sections marked **[SETUP]**
are filled in per run before the loop starts; everything else is the
generic harness contract.

## [SETUP] This run

Describe the run before starting the loop:

- **Target model / config**: e.g. `Llama3 8B FSDP=4 TP=2 bs=1 on 8×H100`.
- **Starting graph**: e.g. empty `construct_default_graph_passes` (raw
  aten graph) vs. the production pass set (cleanup + bucketing + regional
  Inductor + CUDA graphs already applied).
- **Scaffolding level**: which of {curated IDEAS, in-repo reference
  implementations, online research} the agent may use. Tighten the
  *Reading scope* below to match.
- **Goal**: what is being measured (e.g. "find further wins on top of the
  production graph").

## Reading scope (very important)

### You MAY read

- `torchtitan/experiments/graph_trainer/passes.py` — **the only file you modify**. Add custom pass functions and register them in `construct_default_graph_passes` or `compile_time_passes`.
- `torchtitan/experiments/graph_trainer/trainer.py` — GraphTrainer (read-only).
- `torchtitan/experiments/graph_trainer/make_fx_tracer.py` — make_fx tracing (read-only).
- `torchtitan/experiments/graph_trainer/compile.py` — compile entry point (read-only).
- `torchtitan/experiments/graph_trainer/graph_utils.py` — generic FX utilities (read-only).
- `torchtitan/experiments/graph_trainer/<model>/` — the model's graph_trainer
  glue (read-only).
- `torchtitan/models/<model>/`, `torchtitan/models/common/` — model
  architecture (RMSNorm, Attention, RoPE, FF, Decoder) — read-only.
- `torchtitan/experiments/graph_trainer/autoresearch/ideas.md`, `torchtitan/experiments/graph_trainer/autoresearch/learnings.md`,
  `torchtitan/experiments/graph_trainer/autoresearch/experiment_log.md` — your living docs.
- `torchtitan/experiments/graph_trainer/autoresearch/scripts/` — reusable tools you own; create/modify as needed.
- FX graph dumps you produce yourself (e.g. via a temporary `gm.print_readable()` pass).

### You MUST NOT read

Do NOT read the numerics test source (see below).

When unsure whether a source qualifies, err on the side of NOT reading it.

### You MUST NOT inspect git history

- Do NOT run `git log`, `git log -p`, `git show <commit>`, `git diff <commit>`,
  or `git blame`. Both the diffstats and the subject lines of prior commits
  reference optimization techniques we want you to autonomously rediscover.
- `git status` and `git diff` / `git diff --staged` (no commit refs, working
  tree only) are fine.
- To find commits *you* made during this run, look at
  `torchtitan/experiments/graph_trainer/autoresearch/results.tsv` — the second column records the commit hash of
  each `keep` experiment.

## Constraints

**Modify only** `torchtitan/experiments/graph_trainer/passes.py`.

**Do NOT** modify `trainer.py`, `make_fx_tracer.py`, `graph_utils.py`, model code, or add dependencies.

**Goal: minimize training step time while preserving bitwise-identical numerics.**

**Memory** is a soft constraint — some increase is acceptable for meaningful speed gains.

**Simplicity**: all else equal, simpler is better.

## Benchmark

```bash
bash torchtitan/experiments/graph_trainer/autoresearch/scripts/run_benchmark.sh
```

Extract steady-state metrics from the last step:

```bash
grep "step:" run.log | tail -1
```

Output format:
```
step: 20  loss: 8.12345  grad_norm: 1.2345  memory: 12.34GiB(25.00%)  tps: 12,345  tflops: 123.45  mfu: 12.34%
```

Profiling traces: add `--profiler.enable_profiling` to the run command.

**Graph inspection**: Add a temporary `gm.print_readable()` call in your pass, run 1-2 steps, inspect output. Remove debug prints before benchmarking.

## [SETUP] Verifying numerics

The numerics gate compares the optimized graph against the eager
reference via the bitwise-vs-eager test, e.g.:

```bash
pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -k "TestLlama3BitwiseDeterministic and aot_fx_trace_vs_eager" -x > numerics.log 2>&1
```

(Replace `TestLlama3...` with the model's test case.) Any failure counts
as `crash` regardless of perf improvement.

**Do not read the numerics test file's source** — it imports symbols
whose names leak prior optimization techniques. Run it, inspect
numerics.log on failure for the assertion message, and that's it.

## Logging results

Append to `torchtitan/experiments/graph_trainer/autoresearch/results.tsv` (tab-separated). 8 columns:

```
timestamp	commit	tps	memory_gib	status	mfu_pct	wall_time_s	description
```

1. ISO 8601 timestamp
2. Git commit hash (7 chars). `xxxxxxx` for `discard`/`crash`.
3. Tokens per second. 0 for crashes.
4. Peak memory GiB (.1f). 0.0 for crashes.
5. Status: `keep`, `discard`, or `crash`
6. MFU percent. 0.0 for crashes.
7. Wall time in seconds. 0 for crashes.
8. Short description.

## The experiment loop

LOOP FOREVER:

1. Re-read `ideas.md`, `learnings.md`, and `experiment_log.md`. Pick the next idea.
2. **Delegate implementation to a subagent.** Spawn a subagent with a clear prompt describing what to implement in `passes.py`. The subagent does all the code changes, runs the benchmark, and verifies numerics. This keeps the main loop context clean. The subagent prompt should include:
   - What optimization to attempt and why.
   - The reading-scope restrictions above. Pass them through verbatim — the subagent must follow the same rules.
   - The benchmark command: `bash torchtitan/experiments/graph_trainer/autoresearch/scripts/run_benchmark.sh`
   - How to extract results: `grep "step:" run.log | tail -1`
   - The numerics check (the command from the *Verifying numerics* section).
   - That empty grep or numerics failure = `crash`.
3. Read the subagent's result. Determine status: `keep` if tps improved + numerics pass; `discard` if no improvement; `crash` if crashed or numerics failed. Record in TSV.
4. Append entry to `experiment_log.md`.
5. If `keep`: commit `passes.py`, `results.tsv`, `experiment_log.md`, `ideas.md`, `learnings.md`. Message: `[autoresearch] <short description>`.
6. If `discard` or `crash`: revert only `passes.py` (`git checkout -- torchtitan/experiments/graph_trainer/passes.py`), then commit the updated tracking files (`results.tsv`, `experiment_log.md`, `ideas.md`, `learnings.md`) so future iterations remember what didn't work. Message: `[autoresearch] <short description> (<status>)`.

**Benchmark variance**: TPS fluctuates ~1-3%. If delta is within noise, re-run to confirm.

**Crashes**: Fix easy bugs and re-run. If fundamentally broken, skip and move on.

**NEVER STOP**: Work indefinitely until manually interrupted. If out of ideas, profile, study the graph, try different approaches.
