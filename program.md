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

To set up a new experiment, work with the user to:

1. **Record the baseline train command**: get the command that defines the
   target workload and starting point. This includes launcher, config file,
   overrides, world size, model flavor, sequence length, precision, compile
   flags, parallelism flags, and initial training settings. The agent may tune
   allowed `qwen3_1_7b()` config fields during search as long as the model,
   data source, checkpoint behavior, and hardware target remain the same.
2. **Agree on a run tag**: propose a tag based on today's date and workload
   name, for example `may14-qwen3-8xh100`. The branch
   `autoresearch-parallelize/<tag>` must not already exist.
3. **Create the branch**: branch from the current commit that contains the
   human-approved starting implementation.
4. **Read the in-scope files**:
   - `program.md` -- this operating guide.
   - `torchtitan/models/qwen3/model.py` -- model structure and config update.
   - `torchtitan/models/qwen3/parallelize.py` -- runtime parallelism
     orchestration to synthesize.
   - `torchtitan/models/qwen3/sharding.py` -- DTensor sharding contracts to
     synthesize.
   - `torchtitan/models/qwen3/config_registry.py` -- `qwen3_1_7b()` config
     values to tune within the allowed scope.
   - `torchtitan/models/llama3/parallelize.py` -- dense decoder reference.
   - `torchtitan/models/llama4/parallelize.py` -- MoE/FSDP/EP reference.
   - `torchtitan/models/gpt_oss/parallelize.py` -- GPT-OSS MoE reference.
   - `torchtitan/distributed/parallel_dims.py` -- mesh names and constraints.
   - `torchtitan/distributed/pipeline_parallel.py` -- PP composition rules.
   - `torchtitan/protocols/module.py` and `torchtitan/protocols/sharding.py`
     -- config-based DTensor sharding mechanics.
   - `/home/avenkataraman/fbsource/genai/llama4x` -- optional inspiration for
     parallelization ideas and performance hypotheses. Read-only; do not copy
     code blindly.
5. **Check environment**: confirm GPUs, world size, CUDA, PyTorch, and data paths
   required by the train command are available. If data or checkpoints are
   missing, stop and ask the human to provide them.
6. **Initialize `parallelize_results.tsv`** with just the header row. Do not
   commit this file.
7. **Confirm the starting point**: identify the current best source commit and,
   if available, its measured MFU/throughput row. If the starting commit cannot
   run the target command because Qwen3 parallelization is unimplemented, stop
   and ask the human for a runnable starting commit or explicit approval for a
   narrowly scoped bootstrap candidate. Do not create a generic baseline
   implementation during setup.
8. **Confirm setup**: summarize the train command, hardware, editable files,
   current best, and objective before starting the autonomous loop.

Once the human confirms, begin experimentation.

Setup is read-only for source files. Reading references, checking the
environment, creating the experiment branch, and initializing local run
artifacts are setup work. Editing `parallelize.py`, `sharding.py`,
`config_registry.py`, or launch knobs is experiment work and must follow the
one-idea rule below.

## Editable Scope

**What you CAN edit:**

- `torchtitan/models/qwen3/parallelize.py`
- `torchtitan/models/qwen3/sharding.py`
- `torchtitan/models/qwen3/config_registry.py`, but only inside
  `qwen3_1_7b()` and only for allowed `Trainer.Config` values
- local untracked run artifacts such as `run.log`, `correctness.log`,
  `parallelize_results.tsv`, and temporary generated command files
- command-line config knobs used to launch experiments

Both source implementation and config knobs are part of the search space. You
may change how `parallelize_qwen3()` orchestrates runtime parallelism, how
`set_qwen3_sharding_config()` defines DTensor placements, and launch/config
settings such as parallelism degrees, batch sizes, microbatch sizes, activation
checkpointing, compile settings, FSDP reshard policy, pipeline schedule, and
other CLI/config overrides, as long as the resulting command still represents
the same target model/workload class agreed with the human.

Within `qwen3_1_7b()`, you may tune any `Trainer.Config` values except these
fixed fields:

- `loss`
- `hf_assets_path`
- `dataloader`
- `checkpoint`

`model_spec` is partially editable: keep the Qwen3 1.7B flavor, but you may
change only the `model_registry(...)` keyword choices `attn_backend`,
`moe_comm_backend`, and `converters`. Do not change the model flavor passed to
`model_registry`.

Everything else in that `Trainer.Config` is fair game, including optimizer,
learning-rate schedule, training settings such as sequence length and batch
size, parallelism, activation checkpointing, compile, metrics, communication,
validation, and other non-fixed config sections. Keep the model flavor fixed as
Qwen3 1.7B.

**What you CANNOT edit without explicit human approval:**

- model implementation files such as `torchtitan/models/qwen3/model.py`
- `loss`, `hf_assets_path`, `dataloader`, or `checkpoint` inside
  `qwen3_1_7b()`
- `model_spec` changes other than `attn_backend`, `moe_comm_backend`, and
  `converters` kwargs to `model_registry("1.7B", ...)`
- common distributed utilities
- trainer, metrics, data loader, loss, or evaluation code
- dependency files or environment setup

If a candidate needs broader source changes, log the idea and move on unless the
human expands the scope.

## Objective

Maximize TorchTitan's reported steady-state MFU for the exact train command on
this machine or cluster while staying correct.

Primary objective:

```
maximize reported mfu / mfu(%)
```

Quantization exception:

TorchTitan may intentionally omit `mfu` / `mfu(%)` for quantized runs because
the usual BF16 peak-FLOP denominator is not the right comparison for those
kernels. If a quantized candidate omits MFU for that reason, compare it against
other quantized or BF16 candidates using TorchTitan-reported `tps` /
`throughput(tps)` and record MFU as `N/A`. Do not compute a custom MFU formula.

Tie-breakers and diagnostics, in order:

1. higher reported `tps` / `throughput(tps)`
2. peak memory closer to the target utilization without OOM
3. simpler `parallelize.py` and `sharding.py`

Memory target:

- Prefer candidates that use substantial available memory.
- Reject OOMs.
- Treat sustained peak memory above roughly 95% of available GPU memory as risky
  unless the run is clearly stable.
- Memory savings are valuable when they can be converted into larger batch
  size, larger microbatches, less recomputation, or a faster parallelism layout
  that improves reported MFU. Do not treat low memory use as a win by itself.

## Correctness And Measurement

A candidate uses one real training run for both correctness and performance.
Do not run a separate short correctness job followed by a separate performance
job unless the human explicitly asks for it or the candidate is failing before
the train loop and needs a small diagnostic rerun.

At minimum:

1. Python import/syntax succeeds.
2. The model builds on meta and reaches `parallelize_qwen3`.
3. The redirected training run completes forward/backward and logs enough steps
   to judge both convergence and steady-state MFU.
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
- sequence length and batch sizes selected by `qwen3_1_7b()` or CLI overrides
- whether PP/TP/CP/EP/FSDP are enabled
- compile and activation checkpointing settings
- attention backend, MoE communication backend, and config converters selected
  through allowed `model_spec` kwargs
- whether the command is training or inference

TorchTitan and PyTorch-native primitives are good starting points:

- `model.parallelize(tp_mesh)` for config-based DTensor sharding
- `parallelize_module` and `PrepareModuleInputOutput` for local custom plans
- `apply_cp_to_forward` for context parallel attention wrapping
- `apply_ac` for activation checkpointing
- `apply_compile` for per-block compile
- an existing FSDP helper when it can be reused directly; do not copy a full
  reference implementation just to make a baseline runnable
- `apply_moe_ep_tp`, `ExpertParallel`, `NoParallel`, or model-specific expert
  plans for MoE paths
- quantization converters such as `Float8LinearConverter`,
  `Float8GroupedExpertsConverter`, `MXFP8LinearConverter`, or
  `MXFP8GroupedExpertsConverter` when supported by the hardware and installed
  dependencies

It is also fine to write custom logic inside the editable files when the
existing helpers are too generic or leave performance on the table for this
machine. Custom code must still be explainable, correct for the target command,
limited to `parallelize.py` / `sharding.py`, and small enough that the measured
MFU change can be attributed to the stated hypothesis.

Use `/home/avenkataraman/fbsource/genai/llama4x` as read-only inspiration for
parallelization ideas, performance hypotheses, and machine-specific tricks when
useful. Adapt concepts to TorchTitan's APIs and the target command instead of
copying code blindly.

Explore freely from the current best runnable implementation. Do not start by
copying over a reference model's baseline orchestration. Each source or config
change must be justified as a concrete attempt to improve MFU, throughput, or
memory headroom that will be converted into MFU improvement in a follow-up
candidate.

Consider quantization as another performance lever. It is allowed through
`model_spec` converters, and it may require compatible compile settings,
hardware support, and converter filtering to be worthwhile. If TorchTitan
reports MFU for a quantized candidate, use it normally. If TorchTitan omits MFU
for a quantized candidate, use the quantization exception in the Objective
section.

Do not add clever distributed code unless you can explain the tensor placements,
residual placement compatibility, gradient placement, and FSDP interaction.

## Measurement

Redirect full output to logs. Do not let distributed training spam the context.

Example:

```
<train command> > run.log 2>&1
```

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
and workload, reported `mfu` and `tps` are the primary performance fields.

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

Use TorchTitan's reported `mfu` / `mfu(%)` from logged training steps as the
performance objective. Do not invent a separate MFU formula. Step 1 is often
noisy because it includes first-iteration overheads, so compare candidates using
later logged steps after warmup. If a non-quantized run does not report MFU,
treat it as insufficient for performance comparison and fix logging or rerun. If
a quantized run intentionally omits MFU, compare it using reported `tps` /
`throughput(tps)` and record MFU as `N/A`.

## Logging Results

When an experiment finishes, append one row to `parallelize_results.tsv`.

The TSV has this header:

```
commit	mfu_percent	tokens_per_sec	peak_memory_gb	status	description	command
```

Columns:

1. short commit hash, 7 chars
2. steady-state MFU percent, use `0.00` for crashes and `N/A` for quantized
   runs where TorchTitan intentionally omits MFU
3. steady-state tokens/sec, use `0` for crashes
4. peak memory in GB rounded to one decimal, use `0.0` for crashes
5. status: `keep`, `discard`, or `crash`
6. short description of the implementation/config idea
7. exact command used

Example:

```
commit	mfu_percent	tokens_per_sec	peak_memory_gb	status	description	command
abc1234	41.20	1800000	73.4	keep	TP=2 attention and FFN sharding	torchrun ...
def5678	39.80	1730000	70.1	discard	disable sequence parallel	torchrun ...
012abcd	0.00	0	0.0	crash	TP=8 with bad qk_norm placement	torchrun ...
```

Do not commit `parallelize_results.tsv`.

## Experiment Loop

LOOP FOREVER after setup is confirmed:

1. Inspect git state and record the current best commit.
2. Choose exactly one concrete implementation/config hypothesis that is expected
   to improve MFU over the current best. State the expected mechanism before
   editing.
3. Edit the smallest source/config surface needed for that one hypothesis. Edit
   only `torchtitan/models/qwen3/parallelize.py` and
   `torchtitan/models/qwen3/sharding.py` unless changing launch config knobs in
   the command or an allowed `qwen3_1_7b()` field.
4. Run import/syntax checks.
5. Commit the candidate.
6. Run the training command once with output redirected to `run.log`.
7. Use that one run for both correctness and performance. If correctness fails,
   inspect logs, fix obvious bugs, and retry only as needed for diagnosis. If
   the idea is fundamentally broken, log `crash` or `discard`.
8. Extract MFU, tokens/sec, peak memory, convergence, and failure signals.
9. Append a row to `parallelize_results.tsv`.
10. If the candidate improves the objective and passes correctness, keep the
    commit and make it the new best.
11. If it is worse, reset back to the previous best source commit. Keep the TSV
    row uncommitted.

One-idea rule:

- Each loop iteration tests one idea only.
- Broad baseline ports are not a valid idea. Do not copy an entire reference
  `parallelize.py`, FSDP helper, sharding module, or generic runnable baseline
  and treat it as one experiment.
- Do not add code merely because the scaffold is incomplete. If the current best
  cannot run, stop for a human-approved runnable starting point or an explicitly
  approved bootstrap candidate with a tiny, auditable diff.
- An idea may include the minimum coupled source and command/config changes
  required to make that hypothesis valid. For example, trying TP=2 with the
  necessary Qwen3 tensor placement changes is one idea.
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

Do not stop after one successful run. Continue until the human interrupts you or
provides a new instruction. If ideas run low, re-read the reference
parallelization files and try narrower changes around the best implementation.
