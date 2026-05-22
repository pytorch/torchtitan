# Autoresearch Ideas

This file is intentionally empty for Run 1.

Run 1 is the **pure autonomy baseline**: the agent generates its own ideas
from FX graph inspection, without any human-curated seeds.

## Format

Each idea is a top-level bullet with a status checkbox:
- `[ ]` — open, not yet explored
- `[~]` — partially explored, more work possible
- `[x]` — fully explored or no further opportunity

Comments and findings go as **indented sub-bullets** with a timestamp at
the beginning:

```
- [ ] **Idea name**: Description.
  - YYYY-MM-DD HH:MM — Finding or comment.
```

## Ideas

- [x] **Identity slice + double-transpose elimination** (Exp 7 keep, +2.4%): extend Exp 1's peephole-cleanup story. `aten.slice.Tensor(x, dim, 0, end, 1)` is a no-op when `end >= size(dim)`; `aten.t(aten.t(x))` cancels to `x`; `aten._to_copy(x, dtype=x.dtype)` is identity. All numerics-safe.
  - 2026-05-22 — Exp 7 keep: 421/421 slices (100% — every slice was a full-range DTensor local-extraction), 450/1125 double-`t` (40%, mostly backward grad-accumulation `t(t(W))`), 0/841 identity casts. tps 4,569 (best of 2, +2.4% vs 4,460). Memory unchanged (slice/t are metadata-only; win is launch/dispatch overhead, not deallocation as in Exp 1). Pair with `remove_identity_views` as the standard cleanup prologue.

- [ ] **Fuse FSDP weight all-gather + dtype cast** (forward + backward weight re-gather): every parameter is materialized as `view -> view -> _to_copy(f32->bf16) -> all_gather_into_tensor(group=4 '42') -> wait_tensor -> slice -> view -> view` before the matmul that consumes it. Replace the per-param `_to_copy + all_gather` with a single fused all-gather that emits bf16 directly (or shifts the cast onto the smaller pre-AG shard), and drop the dead identity `view`/`slice` chain. Skips ~291 copies and many launches per step.
  - 2026-05-21 — From recon: 291 fwd+bwd FSDP all-gathers on mesh '42', each preceded by a `_to_copy` to bf16. AG shapes are roughly `[64128, 4096]`, `[7168, 4096]` x2, `[1024, 7168]`, `[2048, 4096]`, `[512, 4096]` x2, `[1024, 2048]`, plus 3 RMSNorm weight `[1024]` gathers per layer x32 layers.

- [ ] **Rewrite TP `split([N,N],1) -> cat([·,·],0) -> reduce_scatter('sum', 2, '21')` as direct reduce_scatter on dim 1**: this pattern fires after every TP-sharded matmul (wo, w2, lm_head, and once after `tok_embeddings`). The split+cat is a "rearrange-then-RS-on-dim-0" trick that wastes a full activation-sized concat allocation. A custom `reduce_scatter_v` on the sequence dim, or a `reduce_scatter` on a non-contiguous strided view, would avoid the intermediate `[2, 4096, 4096]` tensor entirely.
  - 2026-05-21 — From recon: 130 `reduce_scatter_tensor(..., 2, '21')` calls, each immediately preceded by `split.Tensor(x, 4096, 1); getitem[0]; getitem[1]; cat([·, ·])`. Activation shape per layer is bf16 [1,8192,4096]; the cat allocates 64MB per occurrence.

- [x] **Eliminate identity DTensor view chains**: every DTensor op materializes as `view_as(input)` + `view_as(local_tensor)` pairs that pass identical shapes/strides through. The graph has 715 `input.view_as(input)` and 1228 `local_tensor.view_as(local_tensor)` calls — ~1.9K nodes that compile/launch for no reason. A peephole pass that replaces such no-op `view.default(x, list(x.shape))` with the input should be very cheap to write and shrink the graph substantially.
  - 2026-05-21 — From recon: 3358 `aten.view.default` nodes total; the vast majority annotated with the DTensor `_api.py:112 / _api.py:229` source lines are zero-cost in eager but still emit FX nodes that take pass/compile time and can confuse downstream fusion.
  - 2026-05-21 — Exp 1 keep: pass removed 2006/3358 view nodes (more than estimated). tps 4,161→4,460 (+7.2%), peak memory 49.0→47.0 GiB (-2.0). Numerics bitwise identical. 1222 genuine reshapes preserved. Surprise: memory dropped — identity views were anchoring tensor references and delaying deallocation post-AOT.

- [x] **Overlap FSDP all-gather with prior layer's compute (prefetch)**: the forward executes `AG_layer_i -> wait -> matmul_i -> ...` strictly serially. Reordering / scheduling so that `AG_layer_{i+1}` issues before `matmul_i` completes (and `wait` is delayed until just before the matmul that needs it) would hide the all-gather latency behind compute. A pass that walks the graph, finds (`AG_w`, `wait_w`, ..., `mm`) chains, and moves the AG earlier (subject to dependency) is feasible at the FX level.
  - 2026-05-21 — From recon: 291 weight AGs all use group '42' (FSDP dp_shard, size 4). Each is followed by a `wait_tensor` and a single `mm`. Currently the AG and the mm sit on the same chain with no overlapping op between them, so the wait blocks compute.
  - 2026-05-21 — Exp 3 discard (Strategy A only): pure-reorder pass moved 130/421 AGs by avg 3.5 slots; tps 4,371 (slight regression vs 4,460). Each AG's *input* (`_to_copy` of the shard) is produced layer-locally, so "earliest legal position" is still inside the same layer — no cross-layer overlap won. To get real prefetch we'd need to move the AG's producer chain (cast+view+AG together) above the previous layer's wait — bigger surgery. Strategy B not attempted in this iteration.
  - 2026-05-21 — Exp 5 discard (Strategy B, first attempt): hoisted producer chains above the *immediately preceding AG's wait*. 290/421 AGs hoisted, avg 10.6 slots / 1.2 wait barriers crossed. tps 4,481 (best of 2; +0.5%, below threshold). Movement turned out to be *intra-layer*, not cross-layer, because the "preceding AG wait" is the same layer's previous weight wait. To get genuine cross-layer prefetch, anchor `prev_wait` to the last `wait_tensor` of the **previous layer's residual reduce_scatter** (the layer's true compute barrier), not just the previous AG. Memory unchanged. Numerics OK. Worth another try with the corrected anchor.
  - 2026-05-22 — Exp 6 discard (Strategy B v2, "most recent TP-RS wait" anchor): 290/291 weight AGs hoisted, avg 20.5 slots / **1.0 barriers** crossed each. tps 4,467 (+0.16%, well below threshold). Memory bump only +0.2 GiB (vs the larger bump expected if real prefetch were happening). Root cause: 130 RS-waits exist (forward + backward) — fwd has 2 per layer (after `wo` mid-layer, after `w2` end-of-layer), so "most recent RS-wait" usually lands at a mid-layer barrier, still intra-layer. To get true cross-layer prefetch, must filter to **post-`w2` (end-of-layer) RS-waits only**, OR anchor on the *previous-layer*'s last RS-wait. Also learned: NCCL group-name tags (`'42'`, `'21'`, `'19'`) are **unstable across runs** because they depend on PG creation order. Match by group **size** (FSDP=4, TP=2) instead.
  - 2026-05-22 — Exp 8 discard (Strategy B v3, end-of-layer-only RS-wait anchor): filtered the 130 RS-waits to 66 layer-end barriers (every-other in fwd/bwd node order), confirmed structurally by gap analysis (attention block ~210 lines, FFN block ~127 lines). 290/291 weight AGs hoisted, avg 47 slots / **1.44 barriers** crossed each (close to target ≥1.5). Memory +0.2 GiB. **tps 4,549 (-0.4% vs 4,569)** — actually regressed. Marking `[x]`: 4 prefetch attempts, all null-or-negative. Strong evidence that either (a) NCCL stream is already serialized at the runtime level so launching the AG earlier in node order doesn't create real overlap, or (b) AOT-trace runtime doesn't issue collectives onto a separate stream as expected. Either way, FX-level node reordering is not the right lever for this scale. Stop chasing prefetch via reordering.

- [~] **Bucket consecutive small-weight all-gathers (Q/K/V, w1/w3, RMSNorm γ)**: per layer we issue separate AGs for Q [2048,4096], K [512,4096], V [512,4096], wo [4096,2048], w1 [7168,4096], w2 [4096,7168], w3 [7168,4096], attention_norm γ [4096], ffn_norm γ [4096]. Q/K/V share the input (post attention_norm) and could be fused into a single `[3072,4096]` AG; w1/w3 likewise into `[14336,4096]`. The two RMSNorm γ AGs per layer (just 4096 bf16 each) are pure launch overhead and could be merged with a sibling AG.
  - 2026-05-21 — From recon: Q/K/V AGs land at lines 680/706/732 (etc.) back-to-back with identical group/mesh. w1/w3 likewise at 890/919. The K and V AGs are 8KB each — well below NCCL's bandwidth-bound threshold.
  - 2026-05-21 — Exp 4 discard (Q/K/V only): bucketed 32 triplets (64 AG launches saved/step). Numerics bitwise-identical. tps 4,490 (best of 2) vs 4,460 baseline = +0.7%, below the +1% keep threshold. Conclusion: Q/K/V AGs at 4-rank NVLink are *bandwidth-bound*, not launch-bound — the Q shard alone is 64KB and dominates. cat+split/reshape overhead in the rewrite cancels most of the launch win. Likely-better targets: the truly tiny RMSNorm γ AGs (4KB at FSDP=4) or full per-layer bucketing using a coalesced AG op if one exists.

- [ ] **Move bf16 cast from full weight to shard before AG**: `_to_copy(f32 -> bf16)` runs on the **pre-AG sharded** weight (e.g. `[1792,4096]`), which is correct shape-wise but still ~4x more data than needed if the weight already lived in bf16. If the FSDP parameter could be kept in bf16 on the shard and AG'd directly, the `_to_copy.default` (420 occurrences, all fp32->bf16) disappears and AG bandwidth halves on average. Alternatively, fuse the cast into the AG buffer prep.
  - 2026-05-21 — From recon: 842 `_to_copy.default` ops, almost evenly split between fp32->bf16 (FSDP weight cast, 420) and bf16->fp32 (RoPE/loss, ~420). Removing the FSDP-cast half is 420 fewer kernel launches.

- [ ] **Fuse RoPE complex-mul block into a single bf16 RoPE kernel**: per layer the graph emits `_to_copy(bf16->f32) -> view -> view_as_complex -> mul (complex) -> view_as_real -> view -> _to_copy(f32->bf16)` for both Q and K. That's ~12 ops per layer x32 = 384 nodes for what should be a single elementwise fused kernel. A pass that rewrites this subgraph into a single `aten.fused_rope` (or a custom HOP) reduces launches and avoids materializing the fp32 intermediate.
  - 2026-05-21 — From recon: 128 `view_as_complex` + 128 `view_as_real` + 64 `_conj` (backward) + ~256 surrounding `_to_copy` ops. Each layer has exactly 4 view_as_complex (Q, K) and 4 view_as_real, plus the two casts.

- [x] **Drop unused SDPA outputs (`getitem_2..5`)**: `_scaled_dot_product_flash_attention.default` returns 9 outputs; only `[0]` (out), `[1]` (logsumexp), `[6]` (philox seed), `[7]` (philox offset) are used. `getitem_6/7/8/9` are bound and then immediately set to None. Trim them at the FX level so the kernel/return tuple can be specialized; also let the partitioner skip materializing them in backward.
  - 2026-05-21 — From recon: 32 SDPA forward calls (one per layer). Every one shows `getitem_6 = ...; getitem_6 = None` pattern through `getitem_9`. Detach of `getitem_4` produces an unused `detach_2` on the forward output that is only used for backward graph wiring.
  - 2026-05-21 — Exp 2 discard: pass removed 0 nodes. The dead getitems shown in `/tmp/autoresearch_graph.txt` (pre-passes) are already DCE'd by `eliminate_dead_code()` inside `remove_identity_views`. tps 4,432 (run-to-run noise vs 4,460). No further upside here without changing the SDPA op's output schema (out of scope — only passes.py is modifiable).

- [ ] **Hoist DTensor `_to_copy` of constants out of the per-step graph**: RMSNorm weight casts (`f32->bf16`, shape `[1024]`) and parameter view chains execute every step despite operating on parameters whose stored dtype is static. If the FSDP MixedPrecisionPolicy cast is amortized into the optimizer step (or skipped when `param_dtype == storage_dtype`), the forward graph drops one cast + 2 views per RMSNorm γ and per Linear weight.
  - 2026-05-21 — From recon: 65 RMSNorm `_to_copy` casts on `[1024]` tensors (32 attention_norm + 32 ffn_norm + 1 final norm). Each cast is followed by an AG on `[4096]` — both are launch-bound, not bandwidth-bound.

- [ ] **Async grad reduce-scatter overlap with backward compute**: backward emits 130 RS-sum-mesh-21 (activation grads) and 161 RS-sum-mesh-42 (parameter grads). Each is `op -> reduce_scatter -> wait` strictly serial with the next matmul. Reorder the wait so the next matmul issues before this RS finishes; or batch multiple parameter-grad RSes into a single bucketed call (FSDP-style "grad bucketing" at graph level).
  - 2026-05-21 — From recon: 161 `reduce_scatter_tensor(..., 'sum', 4, '42')` (parameter grads, one per FSDP-sharded param per step) and 130 `reduce_scatter_tensor(..., 'sum', 2, '21')` (TP-sharded activation grads). Both currently followed immediately by `wait_tensor` in graph order.
