# Ideas — advisory guidance for the Qwen3-14B efficiency search

Human-authored, **advisory** guidance (format: `ARCHITECTURE.md` section 4.2).
The Agent reads these via `observe().ideas` and may use, reprioritize, or ignore
them; nothing here changes rules or verdicts. Acknowledge an item by putting its
`id` in `candidate.addresses`; outcomes are then visible in the ledger.

These priors are distilled from a prior search on this model/hardware. **Two
things changed since then and they matter:** (a) `seq_len` is now LOCKED at 4096,
so short-sequence tricks do not apply and attention is a larger cost fraction;
(b) quality is now floored against a golden, so anything that touched the model's
math must clear the eval — it is no longer a free throughput knob.

## Front-load these (high expected-value, low cost)

- id: compile-blocks
  kind: prior
  target: compile
  weight: 1.0
  text: Per-block `torch.compile(fullgraph=True)` on the repeated transformer
    block is a reliable early win (lower step time and memory). Usually
    quality-neutral, so verify should clear it cheaply.

- id: batch-fill-hbm
  kind: prior
  target: batch
  weight: 0.9
  text: Converting freed HBM into a larger local batch was a large early
    throughput win. NOTE: batch size is now quality-affecting — change it
    COUPLED with an LR rescale as one candidate, and expect to pay the held-out
    eval. Do not treat a bigger batch as a free win.

- id: fsdp-prefetch-overlap
  kind: prior
  target: fsdp/overlap
  weight: 0.8
  text: One-module-ahead FSDP prefetch / collective overlap paid off once the
    profile showed reduce-scatter and all-gather close to the GEMM cost. Keep
    per-layer FSDP; prefetch just enough to overlap nearby all-gathers.

- id: mxfp8-precision
  kind: prior
  target: precision
  weight: 1.0
  text: On Blackwell/B200-class hardware MXFP8 was the single largest lever, but
    it is messy and quality-affecting (skip on hardware without MXFP8 support).
    Expect: TorchAO dim1 cast requires per-chunk row counts
    divisible by 128 (tie loss chunking to seq x batch); force problematic
    dim0/dim1 conversions onto Triton if the CUDA path crashes; keep the chunked
    loss compiled or backward memory blows up; watch for allocator-retry cliffs
    near full HBM. Gate every MXFP8 candidate on the eval.

- id: profiler-guided-fusion
  kind: prior
  target: kernels/fusion
  weight: 0.8
  text: Late wins came from fusing the repeated work the trace actually points
    at: shared input casts for adjacent linears (QKV, FFN gate/up), QKV
    grad-input concat, and a compiled Q/K norm-plus-RoPE helper. Fuse what the
    profile flags, not every adjacent op — gate/up grad-weight concat and full
    QKV backward concat did NOT sustain.

- id: loss-chunk-compile
  kind: prior
  target: loss/memory
  weight: 0.7
  text: Chunked cross-entropy plus a compiled loss path materially reduced
    backward memory and was part of why large batches fit. Treat loss chunking
    and loss compile as part of the memory model; chunk count also constrains
    MXFP8 dim1 row tiling.

## Screen as one batched sweep (graph-invariant runtime flags)

- id: runtime-flags
  kind: hint
  target: runtime flags
  weight: 0.5
  text: A handful of env/runtime flags each gave small wins; screen them as ONE
    factorial mini-sweep rather than one run each: `NCCL_CTA_POLICY=2` (zero-CTA),
    `NCCL_NVLS_ENABLE=1`, `TORCH_NCCL_CUDA_EVENT_CACHE=0`, `--comm.trace_buf_size=0`,
    and 2 persistent dataloader workers with prefetch. Re-confirm any keeper in
    isolation. Most neighboring NCCL knobs (protocol/channel/algorithm forcing,
    high-priority streams, CTA floors) were noise — do not sweep them serially.

## Meta-priors

- id: revisit-compile-granularity
  kind: prior
  target: compile
  weight: 0.7
  text: Compile granularity is not fixed: block-level `model` compile was a loser
    early and a winner late, after the operator mix (MXFP8 casts, fused RoPE)
    changed. Revisit compile boundaries after any change to the operator mix; but
    avoid piling nested compiles inside an already-compiled block without a
    profiler reason.

## Cautions (deprioritize unless conditions are met)

- id: attention-backend-at-4096
  kind: soft_constraint
  target: attention backend
  weight: 0.6
  text: A prior search found SDPA "good enough" and treated Flex/FlashAttention-
    CUTE as a dependency rabbit hole (local `flash_attn.cute` did not match
    Inductor's expectations; many runs, no competitive result). BUT that was at
    seq_len=128 where attention was negligible. At seq_len=4096 attention is a
    real cost fraction, so a WORKING flash/flex backend is worth more here —
    invest only if the local cutlass/flash-attn deps are clean; otherwise stay on
    SDPA and spend the time elsewhere. Time-box dependency debugging hard.

- id: custom-kernel-cost
  kind: soft_constraint
  target: kernels
  weight: 0.6
  text: Only add a custom kernel when the trace shows large repeated work AND the
    new autograd boundary stays cheap. A pairwise RoPE kernel paid off; a custom
    Triton SwiGLU was numerically valid but slower because the autograd boundary
    and materialization cost exceeded the elementwise work saved.

- id: optimizer-reg-is-quality
  kind: soft_constraint
  target: optimizer
  weight: 0.8
  text: Optimizer/regularization changes are quality-affecting now. A prior
    search dropped `weight_decay` as a "throughput win"; that is a regularization
    change and must clear the eval floor. Do not score weight_decay, betas, or
    optimizer swaps as free.

## Do not pursue (no longer applicable)

- id: na-seq128
  kind: soft_constraint
  target: seq_len
  weight: 1.0
  text: The prior search's single biggest win was a constant-token shape at
    seq_len=128. `seq_len` is LOCKED at 4096 and the Harness rejects changes to
    it. Do not pursue short-sequence shapes; the at-4096 cost profile is
    different (attention matters far more).

- id: na-logfreq
  kind: soft_constraint
  target: measurement
  weight: 1.0
  text: A prior search "improved" reported tps by lowering `metrics.log_freq` —
    that was a measurement-window artifact, not real speed. `log_freq` is
    Harness-pinned now; do not touch it.

## References (grounding facts)

- id: ref-roofline
  kind: reference
  target: roofline
  text: Hardware is detected at run start, not assumed. Use TorchTitan's printed
    "Peak FLOPS used for computing MFU" and "CUDA capacity" lines for the exact
    per-device roofline. MXFP8/FP8 levers apply only on hardware with native
    support (Blackwell/B200-class); on other GPUs they will not help or run.

- id: ref-mxfp8-tiling
  kind: reference
  target: precision
  text: TorchAO MXFP8 dim1 cast asserts row counts divisible by 128. With
    seq_len 4096 and N loss chunks, each chunk's row count is (local_batch x
    4096 / N); choose batch/chunk so that is a multiple of 128.
