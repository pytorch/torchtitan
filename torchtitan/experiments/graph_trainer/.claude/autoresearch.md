# Autoresearch — Autonomous Graph Optimization

This is an experiment to have the LLM autonomously optimize graph passes for the
GraphTrainer to improve training performance.

## Prerequisites

- **Working directory**: All commands assume cwd is the repo root (`torchtitan/`).
- **GPU**: At least 8 GPUs available. Default benchmark uses NGPU=8.

## Setup

To set up a new experiment, work with the user to:

1. **Create the branch**: `git checkout -b graph_trainer/autoresearch_<YYYY-MM-DD>` from current commit (use today's date).
2. **Read the in-scope files**: Read these files for full context:
   - `torchtitan/experiments/graph_trainer/.claude/autoresearch.md` — this file.
   - `torchtitan/experiments/graph_trainer/passes.py` — **the primary file you modify**. Contains all graph passes (bucketing, SAC, inductor, cudagraph, etc.). Read it to understand existing pass patterns and the pass signature convention.
   - `torchtitan/experiments/graph_trainer/graph_utils.py` — pass orchestration: how passes are composed and applied. Read-only.
   - `torchtitan/experiments/graph_trainer/compile.py` — compilation dispatcher (JIT/AOT/aot_fx_trace). Read-only.
   - `torchtitan/experiments/graph_trainer/trainer.py` — GraphTrainer: the training loop. Read-only.
   - `torchtitan/experiments/graph_trainer/make_fx_tracer.py` — make_fx tracing infrastructure. Read-only.
   - `torchtitan/experiments/graph_trainer/cudagraph.py` — CUDA graph wrapper. Read for reference.
   - `torchtitan/experiments/graph_trainer/common_utils.py` — shared utilities. Read for reference.
   - `autoresearch/IDEAS.md` — optimization ideas to try, maintained by the user. Re-read at the start of each loop iteration.
   - `autoresearch/LEARNINGS.md` — high-level learnings and methodology guide. You own this file — read it, update it, and use it to inform experiment choices.
   - `autoresearch/scripts/` — reusable analysis tools. You own these scripts and can modify/extend them.
3. **Establish the baseline**: Run the benchmark as-is (see Running below) and record the result.
4. **Initialize results.tsv**: Create `autoresearch/results.tsv` with just the header row. Record the baseline after the first run.
5. **Confirm and go**: Confirm setup looks good.

Your `@identity` for IDEAS.md comments is `@claude`. Use this consistently.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment benchmarks training steps of Llama3 8B. You launch it as:

```bash
# Llama3 8B with FSDP + TP (8 GPUs)
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh \
    --compile.mode aot_fx_trace \
    --parallelism.data_parallel_shard_degree=4 \
    --parallelism.tensor_parallel_degree=2 \
    --metrics.no-enable_tensorboard \
    --profiling.no-enable_profiling \
    --comm.trace_buf_size=0 \
    --training.steps 20 \
    > run.log 2>&1
```

Use `--training.steps 20` for benchmarks (first few steps include compilation overhead — look at the last few steps for steady-state performance).

**What you CAN do:**
- Modify `torchtitan/experiments/graph_trainer/passes.py` — add new graph passes, modify existing ones, or compose them differently.
- Create new pass files in `autoresearch/` and add them to `apply_default_graph_passes` in `passes.py`.
- Optimizations include: op fusion, reordering ops for better scheduling, comm/compute overlap, custom bucketing strategies, memory planning, selective recomputation, kernel fusion hints, etc.

**What you CANNOT do:**
- Modify `trainer.py`, `compile.py`, `make_fx_tracer.py`, or `graph_utils.py` — these are read-only.
- Modify model code (`llama3/`, `deepseek_v3/`, `torchtitan/models/`).
- Break numerics. All changes must remain **bitwise-identical** to the eager reference (fwd and bwd). Verify with the bitwise deterministic test when in doubt.
- Install new packages or add dependencies.

**The goal is simple: minimize training step time while preserving bitwise-identical numerics.** Equivalently, maximize tps / tflops / MFU. The training logs report step time, tps, tflops, MFU, and peak memory.

**Memory** is a soft constraint. Some increase is acceptable for meaningful speed gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small speedup that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better perf is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the speedup magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the benchmark as-is.

**Graph inspection**: Before attempting optimizations, dump and study the FX graph to understand what ops, collectives, and patterns are present. Use the `dump_gm` helper documented in `.claude/CLAUDE.md` ("Dumping Graph Modules for Debugging") to dump the graph before/after passes in `construct_default_graph_passes` in `passes.py`. Run 1-2 steps and inspect the output. Remove the debug dumps before benchmarking. Understanding the graph structure is essential for designing effective passes.

## Output format

The training logs print per-step metrics like:

```
step: 20  loss: 8.12345  grad_norm: 1.2345  memory: 12.34GiB(25.00%)  tps: 12,345  tflops: 123.45  mfu: 12.34%
```

You can extract the key metrics from the log (use the last logged step for steady-state):

```bash
# Get the last step's metrics
grep "step:" run.log | tail -1

# Parse individual values:
# tps (e.g. 12345)
grep "step:" run.log | tail -1 | grep -oP 'tps:\s*\K[0-9,]+' | tr -d ','
# tflops (e.g. 123.45)
grep "step:" run.log | tail -1 | grep -oP 'tflops:\s*\K[0-9.,]+'
# mfu_pct (e.g. 12.34)
grep "step:" run.log | tail -1 | grep -oP 'mfu:\s*\K[0-9.]+'
# memory_gib (e.g. 12.34)
grep "step:" run.log | tail -1 | grep -oP 'memory:\s*\K[0-9.]+'
```

Profiling traces can be collected by adding `--profiling.enable_profiling` to the run command. Traces go to `outputs/profile_traces/`. Inspect these with Chrome's `chrome://tracing` to understand kernel scheduling, identify gaps, and find optimization opportunities.

**Profile the baseline first**: After establishing the baseline, run a profiling pass to understand where time is spent (kernel launch overhead? comm latency? compute-bound? memory-bound?). This informs which optimizations are worth pursuing rather than trying ideas blind.

## Verifying numerics

After any change, run the bitwise deterministic test **first**, before any other tests:

```bash
pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -x > numerics.log 2>&1
```

This verifies that the aot_fx_trace path produces bitwise identical losses and gradients across runs, and matches eager numerics exactly. Any change that breaks this test must be investigated and fixed before proceeding. Treat any failure as a `crash` regardless of perf improvement.


## Logging results

When an experiment is done, log it to `autoresearch/results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 6 columns:

```
commit	tps	memory_gib	status	mfu_pct	description
```

1. git commit hash (short, 7 chars). Use `xxxxxxx` for `discard` and `crash` entries (no commit to reference).
2. tokens per second (e.g. 12345) — use 0 for crashes
3. peak memory in GiB, round to .1f (e.g. 12.3) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. MFU percent (e.g. 12.34) — use 0.0 for crashes
6. short text description of what this experiment tried

Example:

```
commit	tps	memory_gib	status	mfu_pct	description
a1b2c3d	12345	12.3	keep	12.34	baseline
b2c3d4e	13000	12.5	keep	13.01	fuse rmsnorm + quant in SAC pass
xxxxxxx	12100	12.3	discard	12.10	reorder bucketing (no improvement)
xxxxxxx	0	0.0	crash	0.0	custom overlap pass (numerics mismatch)
```

## The experiment loop

The experiment runs on the `graph_trainer/autoresearch` branch.

LOOP FOREVER:

1. Re-read `autoresearch/IDEAS.md`, `autoresearch/LEARNINGS.md`, and `autoresearch/EXPERIMENT_LOG.md` to pick the next idea.
2. Modify `torchtitan/experiments/graph_trainer/passes.py` (or create new pass files in `autoresearch/`) with an optimization idea.
3. Run the benchmark: `NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh --compile.mode aot_fx_trace --parallelism.data_parallel_shard_degree=4 --parallelism.tensor_parallel_degree=2 --metrics.no-enable_tensorboard --profiling.no-enable_profiling --comm.trace_buf_size=0 --training.steps 20 > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
4. Read out the results: `grep "step:" run.log | tail -1`
5. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on that idea.
6. If the run succeeded, verify numerics: `pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -x > numerics.log 2>&1`. Check the exit code and look for any `FAILED` lines. If numerics are broken, treat as a `crash` regardless of perf improvement.
7. Determine the status: `keep` if tps improved and numerics pass, `discard` if tps did not improve, `crash` if the run crashed or numerics failed. Record the results in the TSV.
8. Append a report entry to `autoresearch/EXPERIMENT_LOG.md` for every experiment (see format below).
9. If status is `keep`:
   - git commit the tracked files (`passes.py`, `autoresearch/results.tsv`, `autoresearch/EXPERIMENT_LOG.md`, `autoresearch/IDEAS.md`, and `autoresearch/LEARNINGS.md` if updated, plus any new pass files). Use message format: `[autoresearch] <short description>`. This commit becomes the new "last keep" — note its hash.
10. If status is `discard` or `crash`:
   - Restore the working tree: `git checkout -- .` to revert all source changes back to the last committed (keep) state. Do not commit failed experiments — the experiment log already preserves the history of what was tried.

The branch HEAD always points to the best-so-far state. The full history of experiments (including failures) is preserved in `autoresearch/EXPERIMENT_LOG.md`.

### Experiment report format

After every experiment, append an entry to `autoresearch/EXPERIMENT_LOG.md`. This file is cumulative — never overwrite previous entries. Re-read it at the start of each loop iteration to learn from past experiments and avoid repeating failed approaches.

Each entry follows this format:

```markdown
## <short title> — <status> (<commit hash>)

- **Idea**: What optimization was attempted and why it was expected to help.
- **Changes**: What was actually modified (brief summary, not a full diff).
- **Result**: Perf numbers (tps, MFU, memory_gib) or crash/error description.
- **Analysis**: Why it worked, why it didn't help, or why it broke.
- **Lessons**: Key takeaways — what to build on, what to avoid, or what to try differently.
```

**Benchmark runs**: Each benchmark takes ~5-10 minutes (compilation + 20 training steps). Budget ~10-15 minutes per experiment including numerics checks.

**Benchmark variance**: TPS can fluctuate ~1-3% between runs due to GPU thermals, system load, and NCCL timing. If the delta from baseline is within this noise range, re-run to confirm before calling it `keep` or `discard`. Do not chase phantom improvements.

**Crashes**: If a run crashes (OOM, bug, numerics mismatch), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the TSV, and move on.

**DON'T STOP PREMATURELY**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working autonomously. If you run out of ideas, think harder — read the model code for new angles, study the graph structure, try combining previous near-misses, try more radical graph transformations.

**Stopping criterion**: Stop the loop if the last 10 consecutive experiments all failed to improve TPS (i.e., 10 straight `discard` or `crash` results with no `keep`). When this happens, write a final summary to `autoresearch/EXPERIMENT_LOG.md` explaining what was tried, what worked overall, and what avenues remain unexplored, then stop.

## Learnings

You own `autoresearch/LEARNINGS.md`. This is your high-level guide — a living document of what works, what doesn't, and how to approach optimization effectively.

- **Re-read at loop start**: At the start of each loop iteration, re-read `LEARNINGS.md` alongside `IDEAS.md` and `EXPERIMENT_LOG.md`. Use it to inform your next experiment choice.
- **Update after experiments**: After meaningful experiments (especially surprising results — both positive and negative), update `LEARNINGS.md` with new insights.
- **Include in commits**: Include `LEARNINGS.md` in `keep` commits when it was updated.
- **Keep it concise and actionable**: Focus on patterns and principles, not per-experiment details (those go in `EXPERIMENT_LOG.md`).

## Scripts and tools

You own `autoresearch/scripts/`. These are reusable analysis scripts that support the experiment loop. You can create, modify, and extend these tools as needed during experimentation.

Use profiling (`--profiling.enable_profiling`) and memory snapshots (`--profiling.enable_memory_snapshot`) as you see fit. Traces go to `outputs/profile_traces/`, memory snapshots to `outputs/memory_snapshot/`.

## Optimization ideas

`autoresearch/IDEAS.md` is a **shared, crowd-sourced** document. Multiple humans and agents can add ideas, post findings, or comment on existing entries. At the start of each loop iteration, re-read it to check for new ideas. Prioritize untried ideas (marked `[ ]`) before generating your own.

### Format

Each idea is a top-level bullet with a status checkbox:
- `[ ]` — open, not yet explored
- `[~]` — partially explored, more work possible
- `[x]` — fully explored or no further opportunity

Comments and findings go as **indented sub-bullets** under the idea, with identity and timestamp at the **beginning**:
```
- [ ] **Idea name**: Description.
  - @identity, YYYY-MM-DD HH:MM — Finding or comment here.
```

### Rules
- **Always tag your edits**: Put `@your_identity, YYYY-MM-DD HH:MM —` at the start of each comment.
- **Don't modify other people's comments**. Add a new sub-bullet instead.
- **Don't delete ideas**, even if fully explored. The history is valuable.
- Use `[~]` when an idea is partially explored but more work is possible.
- Include `IDEAS.md` in the commit for `keep` results.

### Graph pass convention

All graph passes must follow the signature documented in `.claude/CLAUDE.md`:
```python
def my_pass(gm: torch.fx.GraphModule, example_inputs, *, other_kwargs) -> torch.fx.GraphModule:
```

To add a new pass, include it in `construct_default_graph_passes` in `passes.py`.
