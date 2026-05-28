# Qwen3 Autoresearch Travelogue

This is a subjective account of the Qwen3 14B throughput search on the
`autoresearch-parallelize/may19-qwen3-14b` branch. It covers the recorded ledger
through `run656`.

The objective was narrow: maximize TorchTitan-reported Qwen3 14B training
throughput on this 8x B200 machine for the local benchmark setup, while keeping
the run plausibly correct and inside a usable memory envelope. I treated
tokens/sec as the primary score, peak memory as a hard practical constraint once
allocator retries appeared, and profiler/MFU numbers as diagnostics rather than
the ranking source.

Correctness validation was intentionally lightweight but consistent with the
benchmark nature of the task. A candidate had to complete the requested training
window, keep loss finite, avoid obvious bad grad-norm behavior, and generally
show a sane short-run loss trend. Computation-changing ideas such as MXFP8 and
custom Triton kernels were screened with local numerical smoke tests when
possible, then judged by short training runs. This was not a convergence proof;
it was a throughput search with loss sanity checks.

Performance was measured from TorchTitan's printed `tps` line. Most candidates
used 10-step runs for screening. Serious contenders were re-run for 20 steps and
ranked by sustained throughput over steps 11-20, because many step-10 spikes did
not validate. Profiler runs were used to choose the next hypothesis, not to rank
throughput, since profiler overhead changes the schedule.

The search was allowed to use command/config changes, environment flags, source
changes under the Qwen3 experiment path, MXFP8/TorchAO plumbing changes, and
custom Triton kernels or operator replacements where I could justify them from
the trace. I also treated dependency work for Flex/CUTE as allowed, but core
TorchTitan behavior was kept experiment-specific rather than generalized into
unrelated model paths.

The short version: the first runnable FSDP baseline was `7,254` tps at
`173.90 GiB`. The durable winner is `run580`, which sustained `14,340` tps over
steps 11-20 at `168.89 GiB`. That is `+7,086` tps, about `+97.7%`, or `1.98x`
over the first experiment while also using about `5 GiB` less peak memory.

The final command I would carry forward is:

```bash
TORCH_NCCL_CUDA_EVENT_CACHE=0 NCCL_NVLS_ENABLE=1 NCCL_CTA_POLICY=2 \
NGPU=8 LOG_RANK=0 MODULE=qwen3 CONFIG=qwen3_14b ./run_train.sh \
  --training.steps=20 \
  --compile.enable \
  --compile.components=model,loss,qk_norm_rope,qkv_grad_input_concat,pairwise_rope \
  --training.dtype=bfloat16 \
  --training.seq_len=128 \
  --training.local_batch_size=176 \
  --loss.num_chunks=4 \
  --optimizer.weight_decay=0.0 \
  --dataloader.num_workers=2 \
  --dataloader.persistent_workers \
  --dataloader.prefetch_factor=2 \
  --metrics.log_freq=1 \
  --comm.trace_buf_size=0
```

In that command, `--compile.enable` turns on the experiment's compile/component
dispatcher. The `components` list is not uniformly "things passed to
`torch.compile`". In this branch it became a compact feature switch:

- `model` means TorchTitan's standard model compile path: compile each repeated
  `TransformerBlock` with `torch.compile(..., fullgraph=True)`. It is not a
  single top-level `torch.compile(model)` call.
- `loss` compiles the chunked cross-entropy loss path, which mattered for memory.
- `qk_norm_rope` is a small `torch.compile` boundary around Q/K RMSNorm plus
  RoPE inside attention.
- `qkv_grad_input_concat` and `pairwise_rope` select experiment-specific
  operator replacements/custom kernels; they are carried in `compile.components`
  because that was the existing switchboard, not because they are themselves
  `torch.compile` calls.
- Earlier experiments also tried nested compile boundaries such as `qkv_linear`,
  `feed_forward`, and `norm_modules`, which compile smaller modules inside each
  block. Those were useful at some points but did not compose with the final
  `model` recipe.

## The Route

![Qwen3 throughput vs experiment number](qwen3_throughput_progression.svg)

The chart shows every recorded candidate as a dot colored by its inferred
experiment category, and the dotted dark line is the best-so-far short-run
screen. Categories are inferred from the run note and command, so mixed
experiments are assigned to their dominant tested idea. Some short-run spikes sit
above the final sustained recommendation; the narrative below calls out where
those spikes failed 20-step validation.

### 1. Making The Baseline Real

The first part of the search was not glamorous. Qwen3's parallelization scaffold
did not yet give me the FSDP baseline the config expected, so the first useful
source change was simply to wrap the transformer blocks, `lm_head`, and root
model with composable FSDP on the `fsdp` mesh. That produced `run001`: `7,254`
tps at `173.90 GiB`.

That result was useful precisely because it was uncomfortable. It trained, but
it was already close to full HBM. I tried full activation checkpointing next
because `qwen3_14b()` sets `activation_checkpoint.mode="full"`, and my first
FSDP-enabling patch had made the model runnable before I had verified that this
configured checkpointing path was actually being honored. It worked as a memory
reducer, but not as a throughput idea: memory dropped dramatically and
throughput fell with it. That taught an early rule for this workload: memory
savings are only valuable if I can spend them on more tokens, less communication
exposure, or a better compiled graph.

I then tried the obvious early levers:

- `torch.compile` on the block path improved the baseline and lowered memory.
- BF16 training dtype became useful once compile was active.
- A larger local batch converted the recovered memory into throughput.
- Full AC, memory-budget AC, TP, CP, and other parallelism routes either lost
  too much time or solved the wrong problem.

By `run010`, the search had moved from `7,254` to `8,391` tps. The gain was not
from one exotic trick. It was compile reducing the step time, BF16 making memory
less tight, and local batch size spending that memory.

### 2. Communication Overlap And Attention Detours

The first profiler trace of the baseline was not a pure GEMM story. GEMMs were
large, but FSDP reduce-scatter and all-gather were close enough that I could not
ignore overlap. That pushed me toward FSDP prefetch experiments.

One-module-ahead FSDP prefetch was the first communication-side change that
clearly paid off: `run059` reached `8,835` tps. Variants around it were less
convincing. Forward-only prefetch, wider windows, endpoint tweaks, and no-reshard
experiments often looked plausible in isolation but either raised memory or lost
throughput. The useful version was narrow: preserve per-layer FSDP and prefetch
just enough to overlap nearby all-gathers.

I also spent time on attention. FlexAttention had some promising moments early,
especially compared with the first FP8 variants, but it did not age well. SDPA
became the practical winner because it was fast enough, stable, and did not
depend on a fragile local CUTE stack. Later, after `cutlass` and `flash-attn`
were installed, Flex FLASH/CUTE got farther, but the local `flash_attn.cute`
interfaces still did not line up with what Inductor expected. I shimmed missing
pieces and got deeper failures, but never a competitive run.

The lesson from this leg was that a backend can be interesting and still be the
wrong use of time for a benchmark search. SDPA was less exciting, but it let the
rest of the system make progress.

### 3. Changing The Shape

The biggest command-level move was the sequence-length sweep. I tested
constant-token shapes: longer sequence with smaller batch versus shorter sequence
with larger batch. `seq_len=128` with a large local batch won for this benchmark.
`run084` reached `9,709` tps, and SDPA at the same shape reached `10,005` tps in
`run099`.

This was one of the most important pivots in the whole search. Once the workload
shape moved to `seq_len=128`, many previous assumptions changed. Loss chunking,
MXFP8 row tiling, attention backend cost, FSDP overlap, and local batch memory
all had to be re-evaluated at the new shape.

Around this point I accumulated a set of small but durable environment and
runtime choices:

- `NCCL_CTA_POLICY=2` means NCCL's `ZERO` CTA policy. When a collective is
  eligible, NCCL can use a copy-engine path instead of spending SM CTAs on the
  communication kernel. That fit this profile because FSDP collectives were
  overlapping with GEMMs, so reducing NCCL's SM footprint was useful.
- `NCCL_NVLS_ENABLE=1` forces NVLink SHARP / NVLink multicast support rather
  than leaving it purely to autodetect. On this NVSwitch B200 box it helped the
  FSDP collective path enough to keep; on a system with broken fabric-manager or
  NVSwitch setup, this is exactly the kind of flag I would revalidate or disable.
- Two persistent dataloader workers were better than the default path.
- `--comm.trace_buf_size=0` removed avoidable flight-recorder overhead.
- Loss chunking mattered, but the right chunk count changed once MXFP8 entered.

Most neighboring knobs did not matter. NCCL protocol forcing, channel forcing,
algorithm lists, high-priority streams, CTA floors, CGA size 4, and many logging
knobs either regressed or produced short-run noise that did not validate.

### 4. Making MXFP8 Work Rather Than Just Turning It On

MXFP8 was the first really messy section. The naive hope was simple: B200 should
like MXFP8, so enabling MXFP8 should help. The actual work was debugging why it
crashed, why some shapes violated TorchAO's dim1 row-tiling assumptions, and why
some "successful" runs collapsed into allocator retries.

The main fixes and constraints were:

- Force problematic MXFP8 dim0/dim1 conversion paths onto Triton instead of the
  crashing CUDA path.
- Respect the dim1 row constraint: the loss chunking shape had to produce row
  counts divisible by the 128-row tile.
- Keep enough loss compile to prevent the output/loss path from blowing up
  memory.
- Avoid batch/chunk combinations that technically ran but lived in allocator
  retry territory.

The first clean MXFP8 runs were in the 11k range. That was a real improvement,
but it also changed the bottleneck. The profile now showed MXFP8 scaled matmuls,
MXFP8 casts, FSDP collectives, and materialization overhead as the main buckets.
Attention and loss were still visible, but they were no longer the first thing
to attack.

This is where a lot of tempting ideas failed. Excluding small projections from
MXFP8 did not beat full coverage. Leaving `lm_head` out of MXFP8 did not save the
loss path enough. BF16 optimizer state variants and optimizer-in-backward had
caveats or did not validate as durable winners. The memory cliff was real: a run
could look close on step 1 and then spend the rest of the window fighting the
allocator.

### 5. Shared Casts And Projection Boundaries

The MXFP8 profile made duplicate input casts look expensive enough to target.
That led to shared MXFP8 input casts for FFN gate/up and attention Q/K/V. This
was one of the first source changes that felt guided rather than speculative:
the trace showed repeated cast work, and the model structure had adjacent
linears consuming the same input.

The FFN and QKV compile experiments then tested where to put compiler boundaries
without sending the whole MXFP8 backward graph through unstable paths. Compiling
feed-forward and QKV projection boundaries helped. Combining that with
`TORCH_NCCL_CUDA_EVENT_CACHE=0` pushed the active recipe into the 12k range.

Not every fusion generalized:

- Shared QKV input work helped.
- QKV grad-input concat helped later.
- QKV grad-weight concat was worse.
- Full QKV backward concat gave attractive short samples but did not sustain.
- FFN gate/up concat and gate/up grad-input concat usually added memory or worse
  scheduling.

The pattern was clear in hindsight: fuse the expensive part the trace points at.
Do not fuse every adjacent thing just because it can be concatenated.

### 6. Q/K Norm, RoPE, And Custom Kernels

After the projection work, the trace still had visible Q/K normalization and
RoPE overhead. This led to compiling just the Q/K RMSNorm plus RoPE helper inside
attention: after `qkv_linear` produced `xq`, `xk`, and `xv`, the helper took
`xq`, `xk`, the Q/K norm weights and epsilons, and the RoPE cache, then returned
normalized/position-encoded `xq` and `xk`. I called it "narrow" because it did
not compile the whole attention block, the projection, or the model. That was
the late-stage jump: `run528` reached `14,003` tps.

I then tried to refine that region. Some ideas helped, some did not:

- A pairwise Triton RoPE kernel helped because Qwen3's RoPE layout made the
  pairwise half-dimension operation natural.
- Different RoPE block sizes gave strong short samples, but sustained validation
  often fell back below the simpler setting.
- Fusing Q and K RoPE into a single launch did not help.
- A fused Q/K RMSNorm plus RoPE forward looked promising briefly but did not
  sustain as the durable path.
- Direct `aten._fused_rms_norm` at the norm boundary was slower than the compiled
  Q/K norm-plus-RoPE helper.

This is also where custom-kernel caution became obvious. Pairwise RoPE was worth
it because it removed real repeated work with a cheap boundary. Custom Triton
SwiGLU was numerically valid, but much slower: the autograd boundary and
materialization cost were larger than the elementwise work it saved.

### 7. The Compile Boundary Reversal

For much of the search, the `model` compile component was a trap. In this code
path, `model` does not mean one giant top-level graph. It calls TorchTitan's
`apply_compile`, which walks the repeated transformer blocks and compiles each
block with `torch.compile(..., fullgraph=True)`. That is still broader than the
smaller boundaries I had been using, because each compiled block includes the
attention path, residuals, norms, and feed-forward path.

Early in the search, that block-level compile either crashed, used too much
memory, or lost to narrower compiles such as `qkv_linear`, `feed_forward`,
`norm_modules`, and the Q/K norm-plus-RoPE helper. At that point the better
strategy was to compile only the pieces whose eager overhead or fusion boundary
showed up in the trace.

After MXFP8 input casts, QKV grad-input concat, and pairwise RoPE were in place,
I revisited the block-level `model` compile because the graph inside each block
was materially different. This time the result depended on restraint:

- `run578` added block-level `model` compile on top of nested `qkv_linear`,
  `feed_forward`, and `norm_modules` compiles. It ran, but lost throughput.
- `run579` removed those nested compiles and kept the `model` compile stack lean:
  `model,loss,qk_norm_rope,qkv_grad_input_concat,pairwise_rope`.
- `run580` validated that lean recipe for 20 steps and sustained `14,340` tps at
  `168.89 GiB`.

That reversal is one of the main lessons of the whole run. "`model` compile is
bad" was true for the old block graph and false for the later block graph. The
correct rule is to revisit compile granularity after changing the operator mix,
but avoid piling nested compile boundaries inside an already compiled block
unless a profiler trace gives a specific reason.

The follow-up closure mostly confirmed the shape of the winner. Nested
`norm_modules`, `qkv_linear`, and `feed_forward` compiles did not improve the
sustained run. `dynamic=False` gave short spikes but lost over 20 steps.
Compiled autograd lost. Nested attention/FFN FSDP under fullgraph compile
crashed, and the non-fullgraph workaround lowered memory but lost too much
throughput.

Removing loss compile was the cleanest negative control. `run656` hit allocator
retries, peaked at `172.38 GiB`, and ended at only `1,702` tps. The compiled
chunked loss is part of why the final recipe fits.

## How The Profiler Guided The Search

The useful traces were not the ones I used to admire a single kernel. The useful
traces answered a ranking question: where is enough time concentrated that a
source change could plausibly move the total step?

Early rank traces showed a mixed bottleneck: GEMM was largest, but FSDP
collectives were close enough to matter. That is what led to FSDP prefetch
experiments and to being careful with no-reshard ideas. Later MXFP8 traces showed
that dynamic quantization changed the profile: MXFP8 matmuls, dim1 casts, dim0
casts, copy/cat work, and NCCL dominated. Attention and loss were visible, but
not first-order.

The best profiler practice was to aggregate by buckets I could act on:

- Matmul/GEMM and MXFP8 nvjet kernels.
- NCCL reduce-scatter and all-gather separately.
- MXFP8 dim1 casts versus dim0 casts.
- Copy/cat/chunk and other materialization overhead.
- RMSNorm/RoPE/attention small-kernel regions.
- CPU launch or logging overhead only after ruling out GPU-dominant work.

I also learned to treat rank skew as signal. If one rank has much larger exposed
NCCL time, a scheduling or overlap idea may help. If all ranks are dominated by
the same GEMM/cast bucket, an NCCL flag is unlikely to rescue the run.

Profiled throughput itself was not used for ranking. It is frequently lower or
occasionally oddly high because profiler overhead changes the schedule. I used
profile runs to choose the next idea, then judged candidates with unprofiled
runs. Serious candidates needed 20-step validation over steps 11-20 because many
10-step "wins" disappeared.

## Best Practices For Future Autoresearch

Keep a ledger and believe it. `keep` should mean "useful or best at the time",
not "permanent ingredient". Several earlier kept ideas were later dropped because
they did not compose with MXFP8 or block-level `model` compile.

Do not rerun the current best by habit. Rerun only when variance blocks a
decision or when a short-run winner needs sustained validation. Otherwise run a
new, controlled delta.

Change one meaningful variable at a time. This search had enough variance that
compound changes were hard to interpret unless they were explicitly staged.

Watch memory as a first-class metric. Above roughly 95% HBM, many runs still
"complete" but collapse from allocator retries or step-time stalls. A slower
lower-memory run can be the better branch if it unlocks the next useful source
change.

Treat loss compile and loss chunking as part of the memory model. They affect
whether backward fits, and MXFP8 dim1 tiling means not every chunk count is legal.

Prefer narrow source changes around profiler-visible repeated work. The wins came
from shared MXFP8 input casts, QKV grad-input concat, Q/K norm+RoPE, pairwise
RoPE, and finally lean block-level `model` compile. Broad rewrites usually made
memory or compiler behavior worse.

For custom kernels, first ask whether the saved work is large enough to pay for a
new autograd boundary. RoPE was worth it. SwiGLU was not.

When trying compile, revisit old failures after the operator mix changes. Model
compile was a loser early and the winner late, where "model compile" specifically
means compiling each repeated transformer block through TorchTitan's `model`
component. The correct conclusion was not "`model` compile is bad"; it was
"`model` compile was bad for the old block graph."

Do not over-index on backend availability. Flex FLASH/CUTE consumed time because
it was blocked by local dependency/API issues. It remains an interesting backend
in principle, but for this branch the evidence says SDPA plus targeted source
changes was the productive path.

## Final Interpretation

The final run is still not at an obvious roofline. `run581` and `run617` profiles
show the remaining time split across MXFP8 scaled matmuls, FSDP collectives,
MXFP8 casts, and some materialization overhead. That means the next large gain is
unlikely to come from another small logging flag or a generic compile option.

The next credible directions are deeper:

- Reduce MXFP8 cast overhead, especially dim1 conversion and repeated materialization.
- Improve overlap between FSDP collectives and GEMM without raising memory.
- Revisit custom kernels only where the trace shows repeated work and the
  autograd boundary can be kept cheap.
- Revisit Flex FLASH/CUTE only with a FlashAttention/CUTE interface that matches
  the PyTorch Inductor expectations.

The main lesson I would hand to future autoresearch is this: use the profiler to
select hypotheses, use short runs to screen them, use sustained runs to believe
them, and keep the winning recipe lean. The final 14.3k result came from a stack
of narrow, explainable changes, not from one magic flag.
