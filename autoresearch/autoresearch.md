# Autoresearch

## This branch: Run 1 — Pure Autonomy Baseline

The goal is to see what an LLM agent can autonomously discover about
optimizing a Llama3 8B training graph **with no human-provided ideas and
no access to prior optimization work**. Ideas must come from inspecting
the live FX graph and the model architecture.

## Reading scope (very important)

### You MAY read

- `torchtitan/experiments/graph_trainer/passes.py` — **the only file you modify**. Add custom pass functions and register them in `construct_default_graph_passes`.
- `torchtitan/experiments/graph_trainer/trainer.py` — GraphTrainer (read-only).
- `torchtitan/experiments/graph_trainer/make_fx_tracer.py` — make_fx tracing (read-only).
- `torchtitan/experiments/graph_trainer/compile.py` — compile entry point (read-only).
- `torchtitan/experiments/graph_trainer/graph_utils.py` — generic FX utilities (read-only).
- `torchtitan/experiments/graph_trainer/llama3/` — the Llama3 graph_trainer
  glue (read-only).
- `torchtitan/models/llama3/`, `torchtitan/models/common/` — model
  architecture (RMSNorm, Attention, RoPE, FF, Decoder) — read-only.
- `autoresearch/IDEAS.md`, `autoresearch/LEARNINGS.md`,
  `autoresearch/EXPERIMENT_LOG.md` — your living docs.
- `autoresearch/scripts/` — reusable tools you own; create/modify as needed.
- FX graph dumps you produce yourself (e.g. via a temporary `gm.print_readable()` pass).

### You MUST NOT read

- **Any file outside the reading scope above**, including any file in
  `torchtitan/experiments/graph_trainer/` not enumerated above (e.g. config
  modules, test files, stub modules, registry modules). Their docstrings,
  field names, and comments leak prior optimization techniques.
- `autoresearch/EXPERIMENT_PLAN.md` — meta-document for the experimenter,
  not for the agent.
- Upstream PyTorch source (`torch/_inductor/`, `torch/distributed/_composable`, etc.).
- `torchao`, FlashAttention, xformers, Triton tutorials, or any other
  external kernel library.
- Web content, blog posts, papers, tutorials.

### You MUST NOT inspect git history

- Do NOT run `git log`, `git log -p`, `git show <commit>`, `git diff <commit>`,
  or `git blame`. Both the diffstats and the subject lines of prior commits
  reference optimization techniques we want you to autonomously rediscover.
- `git status` and `git diff` / `git diff --staged` (no commit refs, working
  tree only) are fine.
- To find commits *you* made during this run, look at
  `autoresearch/results.tsv` — the second column records the commit hash of
  each `keep` experiment.

When unsure whether a source qualifies, err on the side of NOT reading it.

## Constraints

**Modify only** `torchtitan/experiments/graph_trainer/passes.py`.

**Do NOT** modify `trainer.py`, `make_fx_tracer.py`, `graph_utils.py`, model code, or add dependencies.

**Goal: minimize training step time while preserving bitwise-identical numerics.**

**Memory** is a soft constraint — some increase is acceptable for meaningful speed gains.

**Simplicity**: all else equal, simpler is better.

## Benchmark

```bash
bash autoresearch/scripts/run_benchmark.sh
```

Extract steady-state metrics from the last step:

```bash
grep "step:" run.log | tail -1
```

Output format:
```
step: 20  loss: 8.12345  grad_norm: 1.2345  memory: 12.34GiB(25.00%)  tps: 12,345  tflops: 123.45  mfu: 12.34%
```

Profiling traces: add `--profiling.enable_profiling` to the run command.

**Graph inspection**: Add a temporary `gm.print_readable()` call in your pass, run 1-2 steps, inspect output. Remove debug prints before benchmarking.

## Verifying numerics

```bash
pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -k "TestLlama3BitwiseDeterministic and aot_fx_trace_vs_eager" -x > numerics.log 2>&1
```

Any failure counts as `crash` regardless of perf improvement.

**Do not read this test file's source** — it imports symbols whose names
leak prior optimization techniques. Run it, inspect numerics.log on
failure for the assertion message, and that's it.

## Logging results

Append to `autoresearch/results.tsv` (tab-separated). 8 columns:

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

1. Re-read `IDEAS.md`, `LEARNINGS.md`, and `EXPERIMENT_LOG.md`. Pick the next idea.
2. **Delegate implementation to a subagent.** Spawn a subagent with a clear prompt describing what to implement in `passes.py`. The subagent does all the code changes, runs the benchmark, and verifies numerics. This keeps the main loop context clean. The subagent prompt should include:
   - What optimization to attempt and why.
   - The reading-scope restrictions above. Pass them through verbatim — the subagent must follow the same rules.
   - The benchmark command: `bash autoresearch/scripts/run_benchmark.sh`
   - How to extract results: `grep "step:" run.log | tail -1`
   - The numerics check: `pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -k "TestLlama3BitwiseDeterministic and aot_fx_trace_vs_eager" -x > numerics.log 2>&1`
   - That empty grep or numerics failure = `crash`.
3. Read the subagent's result. Determine status: `keep` if tps improved + numerics pass; `discard` if no improvement; `crash` if crashed or numerics failed. Record in TSV.
4. Append entry to `EXPERIMENT_LOG.md`.
5. If `keep`: commit `passes.py`, `results.tsv`, `EXPERIMENT_LOG.md`, `IDEAS.md`, `LEARNINGS.md`. Message: `[autoresearch] <short description>`.
6. If `discard` or `crash`: revert only `passes.py` (`git checkout -- torchtitan/experiments/graph_trainer/passes.py`), then commit the updated tracking files (`results.tsv`, `EXPERIMENT_LOG.md`, `IDEAS.md`, `LEARNINGS.md`) so future iterations remember what didn't work. Message: `[autoresearch] <short description> (<status>)`.

**Benchmark variance**: TPS fluctuates ~1-3%. If delta is within noise, re-run to confirm.

**Crashes**: Fix easy bugs and re-run. If fundamentally broken, skip and move on.

**NEVER STOP**: Work indefinitely until manually interrupted. If out of ideas, profile, study the graph, try different approaches.
