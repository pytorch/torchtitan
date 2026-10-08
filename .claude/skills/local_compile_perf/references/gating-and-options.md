# Gating, options, numerics and parallelism

## Which code path runs where

A gate is an if-branch that picks the rewritten form only when compiling. A rewrite that helps compile can hurt eager; the eager cost (measured with inputs requiring grad) was our most useful guide. Defaults, not rules:

```text
new form slower in eager?
  no  -> one path, no gate (simplest; also helps GraphTrainer)
  yes -> does it also win under GraphTrainer full Inductor?
           yes -> compile_friendly()     (if this tree has it; else is_compiling())
           no / model-dependent -> torch.compiler.is_compiling()
Dynamo-only hints (mark_static, maybe_mark_dynamic) -> is_compiling()
```

Example of `compile_friendly()`: MoE dispatch/combine as a gather-sum costs +3.8 to +4.1 ms per 2 layers in eager (screen) but wins under GraphTrainer. The verdict can flip per model and Inductor config: the ComplexRoPE rewrite was slower under GraphTrainer on two models with default cat lowering, and faster on one of them with ConcatKernel.

## Region options

`@local_compile(name, batch_invariant=..., options={...})` passes options to that region's own `torch.compile`. They're dropped when the region is inlined under an outer compile (block compile, GraphTrainer).

| Option | What it does | Caveat |
|---|---|---|
| `max_pointwise_cat_inputs=0`, `max_complex_pointwise_cat_inputs=0` | per-input cat copies (ConcatKernel) | wins on some shapes, loses on others (patterns.md 2) |
| `unroll_reductions_threshold=K+1` | unrolls a static small reduction | no effect on symbolic sizes |
| `triton.mix_order_reduction_non_strict_mode=True` | no guards from the mix-order heuristic | time-neutral in our cases |
| `coordinate_descent_tuning=True` | autotunes reductions (HcPost backward 739 -> 594 µs) | doubles cold compile, can slow decode shapes; turn off when `torch.are_deterministic_algorithms_enabled()` (Inductor only skips it on its own `config.deterministic`) |
| `triton.prefer_nd_tiling=True` | tiles `[T, N, D]` as rows x D so a symbolic small N isn't divided into every index | Kimi dynamic-width backward 1409 -> 571 µs |
| `assume_32bit_indexing=True` | int32 index math | hard compile failure once T*N*D > int32; we rejected it |

## Numerics

- Compiled fused pointwise math is not bitwise vs eager: Triton contracts into FMAs, so ~1 bf16 ulp differences appear. Tests should allow ~1 ulp; keep bitwise checks only for batch invariance.
- Routing-dependent tensors (router gate grads) can differ by 0.1 relative under any compile because ties in top-k flip with rounding. Check that eager-vs-eager variants and block compile show the same pattern before blaming your change.
- Report rel L2 vs fp64; fused fp32 accumulation often makes compiled closer to fp64 than eager.
- Correctness procedure: measurement.md.

## Activation checkpointing

It matters, so measure with the AC mode the recipe uses.

- **RegionAC (current main, `torch_remat`):** code outside declared remat regions is always replayed in the backward, including eager glue around `local_compile` regions. A bare consumer of a remat region's output needs `remat.recompute_needs_tensor`. Check how your compile regions line up with the remat regions: a compile region that falls outside a remat region is replayed.

Older trees (eager SelectiveAC):

- **Full AC:** each block's forward runs twice. Region forwards run twice too, so a region's forward win counts double, and so does any eager glue left outside regions.
- **Selective AC (eager):** saves a policy-chosen set of op outputs and recomputes the rest. A `local_compile` region's outputs are not on the save list by default (`torch._inductor.config.wrap_inductor_compiled_regions` is False), so regions are replayed in the backward. Turning that flag on crashed MoE models after a recompile, because SAC keys saved outputs per compiled cache entry.
- **SAC host overhead:** SAC's dispatch mode costs ~20 µs of CPU per eager op. With many small eager ops (e.g. 4k tokens per rank), the step becomes host-bound: on DSv3 at 4k, regions under SAC were 48.6 ms vs 34.3 ms for block compile + SAC, mostly host time.
- **Region-level SAC:** calling `torch.utils.checkpoint` with the SAC policy inside a compiled half-block applies the policy at trace time with no runtime dispatch mode; it matched block compile + SAC in our prototype (needs care with side effects and thread-local state inside the checkpoint).

## Parallelism checklist

- **FSDP/HSDP:** regions see local shards; no change unless a region reads parameters that FSDP hooks touch.
- **TP:** is the rewrite exact per shard? A per-row scale before w2 is exact when w2 is replicated over TP (routed experts under EP); a residual add folded into a TP-partial output is wrong before the all-reduce.
- **EP:** shapes are rank-local; zero-token ranks happen at large EP. Check fakes and backward for empty inputs, and that collectives take the same code path on every rank.
- **PP:** stage metadata is static; regions are fine.
- **CUDA graphs:** blocking ops with host syncs or data-dependent sizes cannot be captured; check capture and replay bitwise.
- **GraphTrainer / make_fx:** class-attribute tensors become constants (update a persistent tensor in place); Python ints specialize; wrong fakes record wrong shapes.
- **Thread-local state:** storing any new attribute on an object defined outside a checkpointed region (e.g. lazily creating a list on a module-level `threading.local`) fails under Dynamo with a generic "Observed exception"; create the state outside (e.g. in the thread-local's `__init__`).
