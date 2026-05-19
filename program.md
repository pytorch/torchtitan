# TorchTitan Qwen3 parallelism autoresearch

This is an experiment to have an agent synthesize and optimize the Qwen3
parallelism implementation for one specific train command on one specific
machine or cluster. The generated implementation includes both runtime
orchestration in `parallelize.py` and DTensor placement contracts in
`sharding.py`.

The goal is not to write a universal upstream implementation. The goal is to
produce the fastest correct parallelization strategy for the exact workload and
hardware being tested.

## Setup

When autoresearch is started from `aditvenk/autoresearch-parallelize`, it must
always start FRESH by creating a brand-new experiment branch. It must never
resume an existing experiment, and must never CHEAT by inspecting, checking out,
diffing against, reading logs from, or otherwise using any existing
`autoresearch-parallelize/*` experiment branch.

To set up a new experiment:

1. **Infer and record the baseline train command**: the human will not provide
   a baseline command. The Worker/Executor must determine the command that
   defines the target workload and starting point from this program, the
   experiment-loop rules, the Qwen3 14B config, TorchTitan launcher patterns,
   and the available machine or cluster environment. This includes launcher,
   config file, overrides, world size, model flavor, sequence length, precision,
   compile flags, parallelism flags, and initial training settings. If several
   commands are plausible, choose one conservative command that exercises the
   Qwen3 14B workload on the available hardware, record the evidence and
   reasoning in `learnings.md`, and proceed. Do not ask the human to provide a
   baseline command. The agent may tune allowed `qwen3_14b()` config fields
   during search as long as the model, data source, checkpoint behavior, and
   hardware target remain the same.
2. **Agree on a run tag**: propose a tag based on today's date and workload
   name, for example `may14-qwen3-8xh100`. The branch
   `autoresearch-parallelize/<tag>` must not already exist.
3. **Create the branch**: branch from the current commit.
4. **Read the in-scope files**:
   - `program.md` -- this operating guide.
   - `torchtitan/models/qwen3/model.py` -- model structure and config update.
   - `torchtitan/models/qwen3/parallelize.py` -- runtime parallelism
     orchestration to synthesize.
   - `torchtitan/models/qwen3/sharding.py` -- DTensor sharding contracts to
     synthesize.
   - `torchtitan/models/qwen3/config_registry.py` -- `qwen3_14b()` config
     values to tune within the allowed scope.
   - `torchtitan/distributed/parallel_dims.py` -- mesh names and constraints.
   - `torchtitan/distributed/pipeline_parallel.py` -- PP composition rules.
   - `torchtitan/protocols/module.py` and `torchtitan/protocols/sharding.py`
     -- config-based DTensor sharding mechanics.
5. **Check environment**: confirm GPUs, world size, CUDA, PyTorch, and data paths
   required by the train command are available. Record hardware properties
   needed for roofline reasoning: GPU model, GPU count, available memory,
   expected peak compute, expected memory bandwidth, and any visible
   interconnect/topology information. If data or checkpoints are missing, stop
   and ask the human to provide them.
6. **Initialize committed experiment logs**: create `ideas.md`,
   `learnings.md`, and `results.tsv`. `ideas.md` records ideas to try next and
   ideas that have already been tried. It must include exactly two top-level
   sections: `## Human Generated Ideas` and `## Manager Generated Ideas`.
   `learnings.md` records detailed learnings from the experiment loop so far,
   including hardware properties, profile notes, roofline conclusions, and
   experiment interpretation.
   `results.tsv` records compact run results with the header specified below.
   Commit these three files as the first commit on the experiment branch before
   any implementation experiment.
7. **Record the starting point**: identify the current source commit, current
   best commit if one exists, and any measured throughput/MFU row already in
   `results.tsv`. The Qwen3 scaffold may not yet be the best implementation;
   that is expected. Do not ask the human for a runnable baseline or a baseline
   command during setup.
8. **Summarize setup**: record the train command, hardware, editable files,
   current source commit, current best if known, and objective in `learnings.md`
   before starting the autonomous loops.
9. **Lock the first experiment's baseline**: before any source/config edit,
   write down which parts of the recorded train command are fixed baseline
   requirements and which single part will be changed or implemented first.
   The first experiment must not bundle multiple optimization knobs under a
   vague "make it runnable" or "bootstrap" label. If the target command needs
   several missing capabilities, split them into separate ordered bootstrap
   ideas and run them one at a time.

Begin experimentation after setup is recorded.

Setup is read-only for source files. Reading in-scope files, checking the
environment, creating the experiment branch, and initializing committed run
logs are setup work. Editing `parallelize.py`, `sharding.py`,
`config_registry.py`, or launch knobs is experiment work and must follow the
one-idea rule below.

The inferred and recorded baseline command is a contract. Do not silently add or
remove compile, FSDP, activation checkpointing, precision, quantization, batch
size, resharding, TP, CP, PP, EP, optimizer, metrics, or communication knobs
during the first runnable attempt. Each such change is an experiment idea unless
it is already present in the recorded baseline command.

## Editable Scope

**What you CAN edit:**

- `torchtitan/models/qwen3/parallelize.py`
- `torchtitan/models/qwen3/sharding.py`
- `torchtitan/models/qwen3/config_registry.py`, but only inside
  `qwen3_14b()` and only for allowed `Trainer.Config` values
- committed experiment logs: `ideas.md`, `learnings.md`, and `results.tsv`
- local untracked run artifacts such as `run.log`, profiler traces, and
  temporary generated command files
- command-line config knobs used to launch experiments

Both source implementation and config knobs are part of the search space. You
may change how `parallelize_qwen3()` orchestrates runtime parallelism, how
`set_qwen3_sharding_config()` defines DTensor placements, and launch/config
settings such as parallelism degrees, batch sizes, microbatch sizes, activation
checkpointing, compile settings, FSDP reshard policy, pipeline schedule, and
other CLI/config overrides, as long as the resulting command still represents
the same target model/workload class inferred and recorded during setup.

Within `qwen3_14b()`, you may tune any `Trainer.Config` values except these
fixed fields:

- `loss`
- `hf_assets_path`
- `dataloader`
- `checkpoint`

`model_spec` is partially editable: keep the Qwen3 14B flavor, but you may
change only the `model_registry(...)` keyword choices `attn_backend`,
`moe_comm_backend`, and `converters`. Do not change the model flavor passed to
`model_registry`.

Everything else in that `Trainer.Config` is fair game, including optimizer,
learning-rate schedule, training settings such as sequence length and batch
size, parallelism, activation checkpointing, compile, metrics, communication,
validation, and other non-fixed config sections. Keep the model flavor fixed as
Qwen3 14B.

**What you CANNOT edit without explicit human approval:**

- model implementation files such as `torchtitan/models/qwen3/model.py`
- `loss`, `hf_assets_path`, `dataloader`, or `checkpoint` inside
  `qwen3_14b()`
- `model_spec` changes other than `attn_backend`, `moe_comm_backend`, and
  `converters` kwargs to `model_registry("14B", ...)`
- shared distributed infrastructure, including files under
  `torchtitan/distributed/`, `torchtitan/protocols/`, and
  `torchtitan/models/common/`
- trainer, metrics, data loader, loss, or evaluation code
- dependency files or environment setup

If a candidate needs broader source changes, log the idea and move on unless the
human expands the scope.

## Agent Roles

The experiment is split between the main Codex agent and a Manager sub-agent
with strict ownership:

- The main Codex agent is the **Worker/Executor**, not the Manager. During
  setup, the main Codex agent must spawn a **Manager** sub-agent and give the
  Manager ownership of research, idea generation, and experiment review.
- **Worker/Executor** owns setup, branch creation, source/config changes,
  experiment execution, commits for candidate implementations, `results.tsv`,
  and all git commits on the branch.
- **Manager** owns research, idea generation, and experiment review. The
  Manager may update only `ideas.md` and `learnings.md`, or may return proposed
  updates for the Worker/Executor to apply. The Manager must not commit.
- The Manager reads the Worker/Executor's transcript, `results.tsv`, logs,
  profiler summaries, structured metrics, hardware properties, and relevant
  documentation. After each experiment, the Manager reviews what happened and
  updates or proposes updates to `learnings.md` with detailed conclusions and
  `ideas.md` with newly researched ideas, reprioritized ideas, and crossed-out
  discarded ideas.
- The Worker/Executor reads `ideas.md` and `learnings.md` before each
  iteration, chooses exactly one idea to execute, updates source/config as
  needed, runs the experiment, appends the result to `results.tsv`, and commits
  all changes that should land on the branch.
- The Manager must not edit source/config files, `results.tsv`, logs, dumps, or
  generated artifacts. The Manager must not run `git commit`.
- The Worker/Executor is the only agent that may run `git add`, `git commit`,
  rebase, or any other command that changes branch history.
- The Worker/Executor and Manager should communicate through the three
  committed experiment files: `ideas.md`, `learnings.md`, and `results.tsv`,
  plus concise handoff messages when a Manager sub-agent cannot directly edit
  the shared workspace.

## Objective

Maximize TorchTitan's reported steady-state tokens/sec for the exact train
command on this machine or cluster while staying correct.

Primary objective:

```
maximize reported tps / throughput(tps)
```

Use the value reported by the train command. Do not invent a separate
throughput formula. Console logs usually call this `tps`; structured metrics
usually call this `throughput(tps)`.

Tie-breakers and diagnostics, in order:

1. peak memory closer to the target utilization without OOM
2. higher reported `mfu` / `mfu(%)` when TorchTitan reports it
3. simpler `parallelize.py` and `sharding.py`

Memory target:

- Prefer candidates that use substantial available memory.
- Reject OOMs.
- Treat sustained peak memory above roughly 95% of available GPU memory as risky
  unless the run is clearly stable.
- Memory savings are valuable when they can be converted into larger batch
  size, larger microbatches, less recomputation, or a faster parallelism layout
  that improves reported tokens/sec. Do not treat low memory use as a win by
  itself.

## Correctness And Measurement

A candidate uses one real training run for both correctness and performance.
Do not run a separate short correctness job followed by a separate performance
job unless the human explicitly asks for it or the candidate is failing before
the train loop and needs a small diagnostic rerun.

Each experiment-loop training run is capped at exactly 10 training steps. Add a
command-line override such as `--training.steps=10` to every candidate command,
regardless of the default configured in `qwen3_14b()`. If a candidate changes
`training.steps` in config, the executed experiment command must still run 10
steps unless the human explicitly changes this program.

At minimum:

1. Python import/syntax succeeds.
2. The model builds on meta and reaches `parallelize_qwen3`.
3. The redirected 10-step training run completes forward/backward and logs
   enough steps to judge both convergence and steady-state tokens/sec.
4. Use the early logged steps from that same run as the convergence sanity
   check. The loss must stay finite and should trend downward. A candidate that
   is fast but produces NaNs, exploding loss, or a flat/increasing loss curve
   should be discarded or fixed before accepting its performance result.
   TorchTitan logs this as per-step `loss: ...`; for example a healthy short
   debug run may show loss falling across steps 1-10.

Exact numerical equivalence with an unsharded or differently sharded baseline is
not required. Different valid parallel layouts can change reduction order and
therefore exact numerics. The correctness goal is that the generated
parallelization trains normally for the target workload.

## Implementation Guidance

`parallelize_qwen3()` is allowed to be machine-specific. It may make narrow
assumptions about:

- the exact model flavor
- mesh shape and world size
- GPU type and topology
- sequence length and batch sizes selected by `qwen3_14b()` or CLI overrides
- whether PP/TP/CP/EP/FSDP are enabled
- compile and activation checkpointing settings
- attention backend, MoE communication backend, and config converters selected
  through allowed `model_spec` kwargs
- whether the command is training or inference

TorchTitan and PyTorch-native primitives are useful starting points, not
requirements:

- `model.parallelize(tp_mesh)` for config-based DTensor sharding
- `parallelize_module` and `PrepareModuleInputOutput` for local custom plans
- `apply_cp_to_forward` for context parallel attention wrapping
- `apply_ac` for activation checkpointing
- `apply_compile` for per-block compile
- PyTorch FSDP and DTensor APIs when the candidate needs them
- `ExpertParallel`, `NoParallel`, or model-specific expert plans if the target
  Qwen3 flavor and command actually exercise MoE paths
- quantization converters such as `Float8LinearConverter`,
  `Float8GroupedExpertsConverter`, `MXFP8LinearConverter`, or
  `MXFP8GroupedExpertsConverter` when supported by the hardware and installed
  dependencies

It is also fine to write custom logic inside the editable files when the
existing helpers are too generic or leave performance on the table for this
machine. Custom code must still be explainable, correct for the target command,
limited to `parallelize.py` / `sharding.py`, and small enough that the measured
tokens/sec change can be attributed to the stated hypothesis.

The generated implementation does not have to follow TorchTitan's existing
helper implementation for activation checkpointing, compile, FSDP wrapping, or
their ordering. Alternative local wrapping strategies are allowed inside the
editable Qwen3 files when they are part of one measured hypothesis. For example,
the agent may try a different per-block compile boundary, a different
checkpoint wrapper placement, a different FSDP wrapping granularity, or a
different ordering between compile/checkpoint/FSDP, as long as the code remains
scoped, explainable, and correct for the target command.

Profile and roofline analysis should drive the search. Use TorchTitan profiler
traces, structured metrics, and hardware properties to decide whether the next
idea is attacking compute, HBM bandwidth, communication, launch overhead,
pipeline bubbles, data loading, or memory headroom. A good idea should name the
suspected bottleneck and why the proposed change should improve reported
tokens/sec on this hardware.

Explore freely from the current best runnable implementation. Do not inspect or
copy non-Qwen3 model `parallelize.py` or `sharding.py` files as references; they
are intentionally scaffolded for this experiment. Each source or config change
must be justified as a concrete attempt to improve reported tokens/sec or
memory headroom that will be converted into tokens/sec improvement in a
follow-up candidate.

Consider quantization as another performance lever. It is allowed through
`model_spec` converters, and it may require compatible compile settings,
hardware support, and converter filtering to be worthwhile. Quantized
candidates are compared with the same primary objective: TorchTitan-reported
`tps` / `throughput(tps)`. If TorchTitan omits MFU for a quantized candidate,
record MFU as `N/A` and treat it as a diagnostic gap, not a ranking blocker.

Do not add clever distributed code unless you can explain the tensor placements,
residual placement compatibility, gradient placement, and FSDP interaction.

Use profiler output to identify concrete follow-up hypotheses. When the current
best is runnable, when a candidate is unexpectedly slow, or when ideas are
running low, run a profiled 10-step experiment on the current best or candidate
command. Inspect the generated Kineto traces for GPU idle gaps, dominant CUDA
kernels, collective time, data-loading stalls, compile overhead, and memory
pressure. Convert profiler observations into one narrow next experiment; for
example, a communication hotspot can motivate one TP/FSDP layout change, an
attention hotspot can motivate one attention-backend change, and unused memory
can motivate one batch-size or checkpointing change.

Use roofline reasoning to avoid local search traps. If a profile shows the run
is compute-bound relative to the hardware peak, look for kernel, compile,
precision, attention backend, or quantization changes. If it is memory-bound,
look for layout, activation checkpointing, batch size, sequence length, and
memory format changes that improve useful work per byte. If it is
communication-bound, look at mesh shape, overlap, sharding granularity,
resharding policy, and collective placement. If the bottleneck is unclear,
prefer a profiling or roofline-disambiguation idea over another blind knob
sweep.

Based on profile and roofline analysis, feel free to undo prior optimizations
and move in a different direction. Maintain a healthy explore/exploit tradeoff;
do not keep searching around a local maximum by trying nearby values of the same
knob unless profile or roofline evidence says that knob is still the bottleneck.

## Measurement

Redirect full output to logs. Do not let distributed training spam the context.

Example:

```
<train command> --training.steps=10 > run.log 2>&1
```

Every experiment-loop run must include `--training.steps=10` or an equivalent
CLI override that makes TorchTitan report `total steps 10` at startup. Treat a
run with any other step count as invalid for comparison and rerun with the
10-step cap.

For profiler-driven idea generation, enable TorchTitan's profiler on a normal
10-step run, for example:

```
<train command> --training.steps=10 --profiler.enable_profiling --profiler.profile_freq=10 --profiler.profiler_warmup=2 --profiler.profiler_active=1 > run.log 2>&1
```

Profiler traces are written under the dump folder's `profiling/traces`
directory. The manager uses those traces to explain the next hypothesis in
`ideas.md` or the latest conclusion in `learnings.md`. Because profiling adds
overhead, do not compare a profiled run's tokens/sec against unprofiled runs as
the primary performance result unless every candidate being compared was
measured with the same profiler settings. Enable
`--profiler.enable_memory_snapshot` only when memory behavior is the bottleneck
or when diagnosing OOMs.

For roofline notes, combine hardware properties with measured metrics:

- hardware peak compute from TorchTitan's `Peak FLOPS used for computing MFU`
  line or vendor/H100 documentation when TorchTitan does not print it
- observed per-step `tflops`, `tps`, step time, and memory use
- profiler evidence for dominant kernels, memory stalls, GPU idle time, and
  collective latency
- known memory bandwidth and interconnect/topology limits when available

The manager writes a short roofline conclusion in `learnings.md` when it guides
the next idea: compute-bound, memory-bound, communication-bound,
launch/overhead-bound, or unclear. If it is unclear, state the measurement
needed to disambiguate it.

Extract metrics from both console logs and the structured JSONL file printed at
startup:

```
Structured logging -> JSONL: <path>
```

Useful setup fields from console logs:

- `CUDA capacity: <gpu> with <memory>GiB memory`
- `Model <name> <flavor> size: <n> total parameters`
- `Building device mesh with parallelism: pp=..., dp_replicate=..., dp_shard=..., cp=..., tp=..., ep=...`
- `Successfully created meshes with active dimensions: [...]`
- applied techniques such as `Applied FSDP`, `Applied HSDP`, `Applied Context Parallel`, `Applied selective activation checkpointing`, `Applied CPU Offloading`
- `Peak FLOPS used for computing MFU: <value>`
- `CUDA memory usage for model: <GiB>(<pct>%)`
- `Trainer is initialized with local batch size <n>, global batch size <n>, gradient accumulation steps <n>, sequence length <n>, total steps <n>`

Useful per-step console fields:

- `step: <n>`
- `loss: <value>`
- `grad_norm: <value>`
- `memory: <GiB>GiB(<pct>%)`
- `tps: <tokens/sec per device>`
- `tflops: <value>`
- `mfu: <percent>`

TorchTitan's console step line looks like:

```
step: 10  loss: 4.17040  grad_norm: 1.8422  memory: 0.69GiB(0.72%)  tps: 221,070  tflops: 15.83  mfu: 1.60%
```

Use `tps` as reported by TorchTitan. In the current metrics implementation it
is tokens/sec per device, normalized by non-data-parallel size, and `mfu` is
computed from that same value. For comparing candidates on the same world size
and workload, reported `tps` / `throughput(tps)` is the primary performance
field. Reported `mfu` / `mfu(%)` is a useful diagnostic when present.

Useful success fields:

- `Training completed`
- `Process group destroyed`

The same metrics are also logged with structured names:

- `loss_metrics/global_avg_loss`
- `loss_metrics/global_max_loss`
- `grad_norm`
- `throughput(tps)`
- `tflops`
- `mfu(%)`
- `time_metrics/end_to_end(s)`
- `time_metrics/data_loading(s)`
- `time_metrics/data_loading(%)`
- `memory/max_active(GiB)`
- `memory/max_active(%)`
- `memory/max_reserved(GiB)`
- `memory/max_reserved(%)`
- `memory/num_alloc_retries`
- `memory/num_ooms`

Useful failure fields:

- TorchElastic `FAILED` summary
- `Root Cause (first observed failure)`
- `rank`, `local_rank`, `exitcode`, `pid`, and `error_file`
- Python exception type and message, for example `RuntimeError` or
  `torch.OutOfMemoryError`
- stack frame where the failure happened, especially whether it failed during
  model build, `parallelize_qwen3`, dataloading, forward, backward, optimizer,
  checkpointing, or metric logging

Use TorchTitan's reported `tps` / `throughput(tps)` from logged training steps
as the performance objective. Do not invent a separate throughput formula. Step
1 is often noisy because it includes first-iteration overheads, so compare
candidates using later logged steps after warmup. If a run does not report
tokens/sec, treat it as insufficient for performance comparison and fix logging
or rerun. Record reported MFU when present, but do not require MFU for ranking.

## Experiment Logs

`ideas.md` is the manager-owned idea queue and idea history. It is checked in.
It must contain exactly two top-level sections:

- `## Human Generated Ideas`
- `## Manager Generated Ideas`

Each idea should include:

- idea name
- current best source commit
- source of the idea: human, profile, roofline, metric regression, or
  agent-generated
- expected mechanism for improving reported tokens/sec
- supporting evidence from logs, profiler traces, or roofline notes
- planned source/config changes
- planned command or config overrides
- success criteria and expected risk

This can be more detailed than `results.tsv`; use it to make the reasoning and
expectations auditable. The manager keeps this file current.

The human may add ideas to `## Human Generated Ideas` while the loop is running.
The manager should consider any uncrossed human-generated ideas before adding
new manager-generated ideas. If the worker tries a human-generated idea and
discards it, the manager crosses it out in `ideas.md` using Markdown
strikethrough and adds a short note pointing to the relevant `results.tsv` row
or commit. If an idea is kept, the manager marks it as kept with the winning
commit and measured tokens/sec.

`learnings.md` is the manager-owned research memory. After reviewing each
completed experiment, the manager updates it with detailed learnings from the
run: what changed, what the logs/profiler/roofline evidence showed, why the
result likely happened, what risks or confounders remain, and what this implies
for future ideas. Keep profile notes, roofline notes, and hardware analysis in
`learnings.md`, not in `ideas.md`.

When an experiment finishes, the worker appends one row to `results.tsv`.
`results.tsv` is checked in and worker-owned.

The TSV has this header:

```
commit	tokens_per_sec	mfu_percent	peak_memory_gb	status	description	command
```

Columns:

1. short commit hash, 7 chars
2. steady-state tokens/sec, use `0` for crashes
3. steady-state MFU percent when reported, use `0.00` for crashes and `N/A`
   when TorchTitan omits MFU
4. peak memory in GB rounded to one decimal, use `0.0` for crashes
5. status: `keep`, `discard`, or `crash`
6. short description of the implementation/config idea
7. exact command used

Example:

```
commit	tokens_per_sec	mfu_percent	peak_memory_gb	status	description	command
abc1234	1800000	41.20	73.4	keep	TP=2 attention and FFN sharding	torchrun ...
def5678	1730000	39.80	70.1	discard	disable sequence parallel	torchrun ...
012abcd	0	0.00	0.0	crash	TP=8 with bad qk_norm placement	torchrun ...
```

The Manager may edit `ideas.md` and `learnings.md` or propose edits for the
Worker/Executor to apply, but the Worker/Executor commits those edits. The
Worker/Executor also commits `results.tsv` updates. Do not commit large logs,
profiling traces, dumps, or temporary generated command files unless the human
explicitly asks.

## Experiment Loop

The Worker/Executor loop and Manager sub-agent loop run continuously after
setup is confirmed.

Worker/Executor loop:

1. Inspect git state and record the current best commit.
2. Read `ideas.md` and `learnings.md`, especially `## Human Generated Ideas`,
   `## Manager Generated Ideas`, and the latest profile/roofline conclusions in
   `learnings.md`.
3. Choose exactly one concrete implementation/config hypothesis that is expected
   to improve reported tokens/sec over the current best. Prefer a relevant
   uncrossed human-provided or manager-proposed idea when one exists; otherwise
   choose a hypothesis backed by a recent profiler observation, roofline
   conclusion, or documented hardware/software behavior. State the expected
   mechanism before editing.
   For the first experiment on a new branch, also state why this is exactly one
   change from the locked baseline and list any important knobs intentionally
   left unchanged.
4. Edit the smallest source/config surface needed for that one hypothesis. Edit
   only `torchtitan/models/qwen3/parallelize.py` and
   `torchtitan/models/qwen3/sharding.py` unless changing launch config knobs in
   the command or an allowed `qwen3_14b()` field. The diff must contain only
   code that is actually exercised by the candidate command.
5. Run import/syntax checks.
6. Commit the candidate source/config changes. Include the selected idea name
   or ID in the commit message.
7. Run the training command once with `--training.steps=10` and output
   redirected to `run.log`.
8. Use that one run for both correctness and performance. If correctness fails,
   inspect logs, fix obvious bugs, and retry only as needed for diagnosis. If
   the idea is fundamentally broken, log `crash` or `discard`.
9. Extract tokens/sec, MFU, peak memory, convergence, and failure signals.
10. Append a row to `results.tsv`.
11. Commit `results.tsv` plus any Manager-made or Manager-proposed `ideas.md`
    and `learnings.md` updates that should be recorded with this experiment
    result.
12. If the candidate improves the objective and passes correctness, keep the
    commit and make it the new best.
13. If it is worse, restore source/config files back to the previous best
    source state while preserving committed `results.tsv`, `ideas.md`, and
    `learnings.md` history.

Manager loop:

1. Watch for the worker's transcript, new `results.tsv` rows, worker commits,
   run logs, profiler traces, and structured metrics.
2. Review each completed experiment and update `learnings.md` with detailed
   conclusions, including profile/roofline interpretation and any confounders.
3. Update `ideas.md`: cross out tried-and-discarded ideas, mark kept ideas with
   commit and measured tokens/sec, reprioritize remaining ideas, and add newly
   researched ideas under `## Manager Generated Ideas`.
4. Keep researching. Use recent results, profiler traces, roofline analysis,
   hardware properties, and relevant documentation to generate more ideas for
   the worker.
5. Do not commit. Leave `ideas.md` and `learnings.md` edits for the worker to
   review and commit.

Profiling And Roofline:

- The profiler is a primary idea-generation tool. Profile computation,
  communication latency, overlap, GPU idle time, and data loading to decide
  what to improve next.
- Use roofline analysis based on the actual hardware properties recorded during
  setup. Do not assume the trace is at hardware roofline; estimate whether
  kernels are leaving compute, bandwidth, or communication headroom.

One-idea rule:

- Each loop iteration tests one idea only.
- The first experiment is not exempt. "Initial runnable baseline", "bootstrap",
  "port the scaffold", or "match historical setup" may not combine independent
  changes such as enabling FSDP, enabling compile, changing activation
  checkpointing, changing dtype/precision, changing batch size, changing FSDP
  resharding, adding quantization, or changing NCCL/runtime knobs. Pick one
  such change, record it as the first idea, run it, and only then proceed to
  the next.
- Broad baseline ports are not a valid idea. Do not copy an entire reference
  `parallelize.py`, FSDP helper, sharding module, or generic runnable baseline
  and treat it as one experiment.
- Do not add broad code merely because the scaffold is incomplete. If the
  current best cannot run, the first worker iteration may be a narrow bootstrap
  idea that makes the target command runnable. Record that idea in `ideas.md`
  and `learnings.md`; keep the diff tiny and auditable.
- A bootstrap idea may implement only the single missing capability named in
  the idea. Examples of valid first bootstrap ideas are "add the minimal Qwen3
  FSDP wrapper required by the already-recorded baseline command" or "add the
  missing Qwen3 TP placement for the already-recorded TP=2 command". Invalid
  bootstrap ideas include "add FSDP, turn on compile, raise batch size, and
  switch activation checkpointing", or "copy the Llama parallelization stack
  and test the whole bundle".
- An idea may include the minimum coupled source and command/config changes
  required to make that hypothesis valid. For example, trying TP=2 with the
  necessary Qwen3 tensor placement changes is one idea.
- Coupled changes must be inseparable for correctness, not merely convenient
  or historically associated. If two choices could each be toggled
  independently in a command line or a small source diff, they are two ideas.
- Do not add unused or inactive code in an experiment step. This includes
  unused imports, helper functions that are not called, dormant branches for
  parallelisms or config modes that the candidate command does not enable,
  compile/quantization/checkpointing plumbing that is not part of the stated
  hypothesis, and future-looking scaffolding for later ideas. If the code is
  not required for the current candidate to run and measure the stated
  hypothesis, leave it out.
- Do not bundle unrelated optimizations in one iteration. For example, changing
  TP degree, activation checkpointing mode, compile settings, and batch size in
  the same iteration is not allowed unless the hypothesis explicitly depends on
  that full combination.
- Prefer deleting, reusing, or narrowly adapting existing code over adding new
  copied helpers. If the diff is large enough that several independent choices
  could explain the result, split it before running.
- If a result improves, attribute the improvement to the tested hypothesis and
  use the next loop iteration for the next idea.

Crashes:

- OOMs are useful data. Log them and reduce memory pressure or change strategy.
- Simple coding bugs may be fixed in-place and rerun.
- Repeated failures for the same idea should be abandoned quickly.

Timeout:

- Use a timeout appropriate to the train command and cluster queue.
- If a run hangs, kill it, log `crash`, and move on.

## Stop Condition

NEVER STOP!

Do not stop after one successful run. Continue until the human interrupts you or
provides a new instruction. If ideas run low, run a profiled 10-step pass on
the current best, update the roofline notes, and choose the next narrow change
from the observed bottleneck.
