# Ideas — advisory guidance (llama3-8B efficiency search)

Human-authored, **advisory** (format: `ARCHITECTURE.md` section 4.2). The Agent
reads these via `observe().ideas`; they never bind a verdict. Acknowledge an item
by putting its `id` in `candidate.addresses`.

**v1 is faithfulness-only:** math-changing levers (precision/fp8, batch, LR,
weight_decay, optimizer) are REJECTED, so favor *math-preserving* throughput
knobs. **Optimize the measured bottleneck, not a guess.**

## Method — do this first

- id: profile-first
  kind: prior
  target: method
  weight: 1.0
  text: Read the profiler summary and the printed MFU/memory before proposing.
    Decide what the run is bound by, then attack that: comm-bound (all-gather /
    reduce-scatter exposed) -> reshard policy + prefetch/overlap; compute-bound
    (high MFU) -> fusion/compile; memory-bound -> activation-checkpoint mode.
    Propose the smallest change that targets the top cost in the trace. Do NOT
    blind-sweep knobs that the profile says are off the critical path.

## Faithful throughput levers (v1 can promote these)

- id: fsdp-comm-overlap
  kind: prior
  target: fsdp
  weight: 0.9
  text: At small batch the win is usually FSDP comm scheduling: reshard_after_
    forward policy and one-module-ahead prefetch to overlap all-gather with
    compute. Pure scheduling, so faithful. Justify it from the trace's exposed
    collective time.

- id: activation-checkpointing
  kind: prior
  target: ac
  weight: 0.8
  text: AC mode (none / selective / full) trades recompute for memory. With
    memory headroom, less recompute is faster; selective-op keeps the expensive
    matmuls and recomputes cheap ops. Faithful (recompute reproduces values).

- id: compile-granularity
  kind: prior
  target: compile
  weight: 0.6
  text: Per-block torch.compile fuses pointwise/norm and cuts launch overhead,
    but only pays when compute-bound (check MFU first), and on llama3 it is
    hampered by complex-valued RoPE (Inductor cannot codegen complex ops).
    Revisit only after the operator mix changes.

- id: attention-backend
  kind: hint
  target: attention
  weight: 0.6
  text: At long seq_len attention is a real cost fraction, so a working flash/
    flex backend can beat SDPA. Invest only if the local deps are clean; time-box
    dependency debugging hard.

- id: runtime-flags
  kind: hint
  target: runtime
  weight: 0.5
  text: Screen graph-invariant env/runtime flags as ONE factorial mini-sweep,
    not one run each (NCCL CTA/NVLS knobs, dataloader workers + prefetch).
    Re-confirm any keeper in isolation; most neighboring NCCL knobs are noise.

## Guardrails — do not

- id: na-logfreq
  kind: soft_constraint
  target: measurement
  weight: 1.0
  text: Do not touch metrics.log_freq to "improve" tps; that is a measurement
    artifact and it is Harness-pinned anyway.

- id: affecting-is-rejected
  kind: soft_constraint
  target: precision/batch/optimizer
  weight: 1.0
  text: precision (fp8/mxfp8/dtype), batch size, LR, weight_decay, and optimizer
    changes move the math and are REJECTED in v1. Do not propose them; they
    cannot pass faithfulness.

## Reference

- id: ref-roofline
  kind: reference
  target: roofline
  text: Hardware is detected at run start. Use TorchTitan's printed "Peak FLOPS
    used for computing MFU", MFU, and memory lines for the per-device roofline
    and to tell compute-bound from comm/memory-bound.
