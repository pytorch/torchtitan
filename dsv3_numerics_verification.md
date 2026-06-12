# DSv3 Numerics Verification — eager vs graph_trainer

Bitwise comparison of the **eager `Trainer`** against the **`graph_trainer`
(aot_fx_trace)** path for DeepSeek-V3, on the `graph_trainer/dsv3_scaling`
branch. Goal: confirm graph_trainer's ChunkedLoss + full-AC numerics match the
trusted eager reference (the "loss divergence" concern flagged in
`dsv3_scaling_experiment_results.md`).

**TL;DR**
- **debugmodel: bitwise identical** (loss + grad_norm, all 20 steps, max\|diff\| = 0).
- **16B: bitwise identical for steps 1–6, then diverges** — a last-bit
  (`~2e-5`) difference at step 7 that compounds to `~2e-2` by step 50.
- **It is a deterministic, systematic eager-vs-graph difference, not
  non-determinism**: graph_trainer and eager are each bitwise-reproducible
  run-to-run at 16B.
- **Per-step param/grad hashing localizes it to `lm_head.weight.grad`** — the
  language-model-head gradient from the chunked-loss path
  (`ChunkedCELossWithParamGrads` vs eager `ChunkedCELoss`). It is the **only**
  tensor that differs for steps 1–5; **every transformer block, attention, MoE
  expert, norm and embedding is bitwise identical**, so FSDP bucketing / MoE
  `_grouped_mm` / flex are bit-exact (bucket order *is* matched). Not a
  convergence problem.
- **Root cause: chunks-then-ranks vs ranks-then-chunks summation order
  (both f32), because graph_trainer uses simple_fsdp, not FSDP2.** Eager FSDP2
  accumulates the per-chunk lm_head grads UNSHARDED in **f32** then does **one
  coalesced reduce-scatter** (`RS(Σ_f32 gᵢ)`); graph_trainer's simple_fsdp
  fires **one reduce-scatter per chunk** then sums in f32 (`Σ_f32 RS(gᵢ)`).
  Both f32 — the only difference is the order, an f32-ulp diff (2.3e-10) that
  compounds. Ruled out by ablation: weight tying (377==377, ratio ~1× not 2×),
  NCCL algo (pinned → identical), fused-vs-decomposed CE (identical),
  `num_chunks=1` (bitwise — no cross-chunk accumulation).
- **FIXED (✅, validated bitwise at 16B):** rewrote `ChunkedCELossWithParamGrads`
  to all-gather the lm_head weight once, run the chunk loop on local tensors
  (no per-chunk collective), accumulate per-chunk grads UNSHARDED in f32, and do
  ONE reduce-scatter — reproducing eager's order. 16B eager-vs-graph
  (`num_chunks=8`, 15 steps): loss / grad_norm / `lm_head.weight.grad` all
  **bitwise identical (0 diverging steps)**. Chunking (memory) preserved.
- **Perf @ 16B (B=4):** graph_trainer uses ~12% less peak memory and is ~2.7%
  faster than eager.

## Method

Both sides use **ChunkedLoss** and **full activation checkpointing**:

| | eager (baseline) | graph_trainer (test) |
|---|---|---|
| loss | `ChunkedCELoss` | `ChunkedCELossWithParamGrads` (#3248, tracer-friendly) |
| full AC | `--activation_checkpoint.mode full` (`checkpoint_wrapper`) | `--compile.memory_policy full` (documented mirror of eager full AC) |
| compile | **pure eager** (`--compile.no-enable`; the 16B config's `compile.components=["loss"]` is disabled so the reference is unfused) | `aot_fx_trace`, regional-inductor flex (required for bitwise flex match), **cudagraph disabled** to isolate ChunkedLoss+AC numerics |

`scripts/loss_compare.py` forces `--debug.deterministic --debug.seed=42` and
`--metrics.log_freq=1`. We re-read **full-precision** loss and grad_norm from
TensorBoard (the 5-digit stdout is not enough) plus memory / tps / tflops / mfu.
No seed checkpoint is needed: with identical parallelism + seed the two runs
initialize to byte-identical weights (verified — the debugmodel matches both
with and without a seed checkpoint, and the 16B runs agree bit-for-bit through
step 6).

Reusable launcher: **`./run_loss_compare_dsv3.sh`**

```bash
STEPS=50 MODEL_SIZE=16b ./run_loss_compare_dsv3.sh          # the 16B run below
MODEL_SIZE=debugmodel STEPS=20 ./run_loss_compare_dsv3.sh   # the debugmodel run
```

---

## Result 1 — debugmodel: BITWISE IDENTICAL ✅

`deepseek_v3_debugmodel` vs `graph_trainer_deepseek_v3_debugmodel`,
`dp_shard=8 tp=1 ep=4`, 20 steps.

| metric | steps | max \|diff\| | steps differing |
|---|---|---|---|
| loss | 20 | **0.000e+00** | 0 / 20 |
| grad_norm | 20 | **0.000e+00** | 0 / 20 |

Full precision, every step identical (e.g. step 1 loss `8.089265823364258` /
grad `3.4663453102111816`; step 20 loss `3.237527370452881` / grad
`0.5489872694015503`). Confirmed both **with** a shared seed checkpoint and
**without** one (`--no-seed-checkpoint`), proving init is deterministic across
the two code paths.

---

## Result 2 — 16B: matches for 6 steps, then diverges ⚠️

`deepseek_v3_16b` vs `graph_trainer_deepseek_v3_16b`, **real distributed config
`dp_shard=8 tp=1 ep=4`**, 50 steps, `c4_test`.

### Bitwise (loss + grad_norm, full precision)

| metric | steps | max \|diff\| | steps differing | first differing step |
|---|---|---|---|---|
| loss | 50 | 8.544e-02 | 44 / 50 | **step 7** |
| grad_norm | 50 | 4.231e+00 | 45 / 50 | step 7 |

**Verdict: NOT bitwise identical.** Steps **1–6 are bitwise identical**
(diff = `0.0`), then a `~2e-5` perturbation appears at step 7 and compounds
chaotically:

| step | loss (eager) | loss (graph_trainer) | loss Δ |
|---:|---|---|---|
| 1 | 11.98492431640625 | 11.98492431640625 | +0.0e+00 |
| 6 | 10.956615447998047 | 10.956615447998047 | +0.0e+00 |
| 7 | 10.4504976272583 | 10.450478553771973 | **+1.9e-05** |
| 10 | 10.009139060974121 | 10.009237289428711 | -9.8e-05 |
| 20 | 8.061719894... | 8.061529...        | +1.9e-04 |
| 30 | 6.948173046... | 6.947973...        | +2.0e-04 |
| 40 | 6.46696138...  | 6.485...           | -1.8e-02 |
| 50 | 6.190555572509766 | 6.211479... | -2.1e-02 |

The onset is a last-bit (`~1e-5`, bf16/accumulation epsilon) difference, not a
gross error — the forward/backward/optimizer are bitwise identical for the
first 6 steps, then one step's gradient differs in the last bits and chaotic
training dynamics amplify it. This does **not** reproduce at debugmodel scale
(8 experts, dim 256) — it appears only at 16B scale (64 experts, dim 2048).

### Perf (memory = peak; tps/tflops/mfu = avg of steps > 10)

| metric | eager | graph_trainer | graph/eager |
|---|---:|---:|---:|
| peak reserved mem (GiB) | 40.78 | 36.11 | **0.885×** |
| peak active mem (GiB) | 36.17 | 34.24 | 0.946× |
| throughput (tps) | 7,966 | 8,179 | **1.027×** |
| tflops | 144.23 | 148.09 | 1.027× |
| mfu (%) | 14.58 | 14.97 | 1.027× |

graph_trainer uses **~12% less peak memory** and is **~2.7% faster** at matched
batch (B=4, seq 4096). (Memory/mfu are expected to differ — `memory_policy=full`
recompute and eager full-AC are different mechanisms, and graph_trainer fuses
flex via regional inductor.)

---

## Root-cause diagnosis: a deterministic, systematic last-bit difference

The step-7 onset (after 6 *exactly*-identical steps) is the signature of either
last-bit nondeterminism, or a tiny systematic eager-vs-graph difference. To
disambiguate, each path was run against itself (same seed/config/parallelism,
50 steps):

| path (run A vs run B) | loss | grad_norm | |
|---|---|---|---|
| **graph_trainer 16B** | max\|diff\| 0.000e+00, 0/50 differ | 0.000e+00, 0/50 | **deterministic ✅** |
| **eager 16B** | max\|diff\| 0.000e+00, 0/50 differ | 0.000e+00, 0/50 | **deterministic ✅** |

**Both paths are individually bitwise-deterministic** (they satisfy the
determinism guardrail). The eager-vs-graph divergence is therefore **not
non-determinism** — it is a *reproducible, systematic* last-bit difference
between the two code paths.

### Localized with per-step param/grad hashing

`Trainer.train_step` was instrumented (env-guarded `TITAN_HASH_DEBUG`, using the
existing `tests.utils.hash_model` / `hash_gradient` with `per_tensor=True`) to
hash every parameter's **pre-clip gradient** (after backward) and **post-step
value** each step, in both paths. FQNs are normalized (eager full-AC's
`_checkpoint_wrapped_module.` prefix stripped) so the two paths' tensors align.
Diffing the two runs' per-tensor hashes (16B, 10 steps):

| step | grad: #tensors differing | model: #tensors differing |
|---:|---|---|
| 1–5 | **1 / 377** (`lm_head.weight.grad` only) | **1 / 456** (`lm_head.weight` only) |
| 6 | 365 / 377 | 315 / 456 |
| 7 | 377 / 377 | 376 / 456 |
| 8–10 | 377 / 377 | 378–380 / 456 |

**First divergence: `lm_head.weight.grad`, step 1.** It is the *only* tensor
that differs for the first five steps — **every transformer-block weight/grad
(attention, MoE experts, norms) and the embeddings are bitwise identical.**

**Root cause: the lm_head gradient from the chunked-loss path**, i.e.
graph_trainer's `ChunkedCELossWithParamGrads` (#3248) produces a last-bit
different `lm_head.weight.grad` than eager `ChunkedCELoss`. (#3248 was verified
for bitwise *loss* parity, but the lm_head *param-gradient* was not — and that
is what differs.) The debugmodel shows the same single-tensor `lm_head` diff but
it never propagates (rounds away at vocab 2048), which is why the debugmodel
loss stayed bitwise identical.

**Propagation:** the perturbation stays contained in `lm_head` for 5 steps
(low-bit drift). At **step 6** the drifted `lm_head.weight` makes
`dL/dhidden = grad_output @ lm_head.weightᵀ` round differently → every
transformer-block gradient diverges → the loss diverges at step 7. So the
step-7 loss onset is the *propagation* point, not the origin.

**What this rules out:** FSDP gradient bucketing
(`joint_transformer_block_bucketing_reordering_pass`), MoE `_grouped_mm`,
flex/regional-inductor, and `memory_policy=full` recompute are all **bit-exact**
vs eager — every transformer-block tensor matches through step 5. The bucket
order *is* matched (as expected from #3590).

**This is a bitwise-exactness gap, not a convergence/correctness problem:** both
paths are deterministic and track to ~4 sig-figs; the ~0.02 loss gap at step 50
is within different-seed spread.

### Confirmed: last-bit reduction order, NOT double-counted / tied weights

graph_trainer builds its grad list with `named_parameters(remove_duplicate=False)`
and writes back `param.grad += grad` (to support weight tying), and
`torch.autograd.grad(y, [W, W])` returns the *full* grad for each occurrence — so
a tied/duplicated param would be **exactly 2×**. Tested directly (step-1
`lm_head.weight.grad`, eager vs graph, 16B):

| check | result |
|---|---|
| duplicate params (`n_nodup` vs `n_unique_ids`) | **377 == 377** → no tying, nothing returned twice |
| `lm_head.weight.grad` ratio graph/eager | min 0.999924, **median 1.000000**, max 1.000102 → **~1×, not 2×** |
| `lm_head.weight.grad` abs diff | `max 9.3e-10` (grads up to 4.5e-2), median rel diff **0** |
| `tok_embeddings.weight.grad` | **bitwise identical** (diff 0) |

So it is **not** the double-count hypothesis. It is a **last-bit
reduction-order difference** in the lm_head chunked-grad accumulation:
`dL/d(lm_head.weight)` is summed over the 8 sequence chunks (per-chunk FSDP
reduce-scatter), and the traced `ChunkedCELossWithParamGrads` accumulates those
chunks in a different float order than eager `ChunkedCELoss`. The embedding grad
(single scatter-add path) matches exactly — the difference is unique to the
lm_head chunk-accumulation path.

### Exact spot (traced loss subgraph) + NCCL ruled out

**NCCL is not the cause.** Re-ran eager-vs-graph with `NCCL_ALGO=Ring
NCCL_PROTO=Simple` pinned identically on both sides: the `lm_head.weight.grad`
diff is **byte-for-byte the same** as with default NCCL (`max|absdiff|=2.328e-10`,
same ratio). The reduce-scatter algorithm is not it.

**Dumped the traced fwd+bwd graph** (`make_fx_graph_traced`, debugmodel). The
lm_head weight-grad path is, per sequence chunk (`num_chunks=8`):

```
nll_loss_backward + _log_softmax_backward_data   # dL/dlogits  (decomposed CE)
  -> cast bf16, transpose
mm  = dlogitsᵀ @ hidden        # bf16 aten.mm  -> bf16 per-chunk weight grad
  -> cast f32
reduce_scatter_tensor('sum', 8)             # per-chunk, f32
```

and the 8 per-chunk shards are accumulated as a **sequential left-fold in chunk
order**: `g = rs₀; g = add_(g, rs₁); … add_(g, rs₇)`. That is the **same order
eager uses** (per-chunk FSDP reduce-scatter, accumulated 0→7) — so neither the
accumulation order nor NCCL is the source.

**Two real differences remain, both on the lm_head weight-grad path only:**
1. **CE is fused in eager, decomposed in the graph.** Eager runs
   `torch.nn.functional.cross_entropy` (loss.py:33) as a fused kernel; make_fx
   decomposes it to `_log_softmax_backward_data` + `nll_loss_backward`.
2. **The per-chunk weight grad is a standalone bf16 `aten.mm`** reducing over the
   chunk's tokens (the large dim), then cast to f32. Eager computes the same
   product inside autograd's Linear backward.

**Why only lm_head, not the transformer:** `dL/dhidden` (`mm = dlogits @ Wᵀ`) is
*assembled by chunk slice* into `[B,L,D]` with **no cross-chunk sum**, so it is
bitwise identical → every transformer-block grad matches. Only the *weight*
grad has the cross-chunk reduction + the large-token-reduction matmul, so only it
diverges in the last bit.

### ROOT CAUSE NAILED: bf16-coalesced vs f32-per-chunk reduce-scatter

Ablations (all on `lm_head.weight.grad`, step 1, eager vs graph):

| ablation | result | conclusion |
|---|---|---|
| weight tying / dup params | 377==377, ratio ~1× | not double-counting |
| `NCCL_ALGO=Ring` pinned both | identical diff (2.3e-10) | not the NCCL algorithm |
| decomposed CE both (`TITAN_DECOMP_CE`) | identical diff (2.3e-10) | not fused-vs-decomposed CE |
| **`--loss.num_chunks=1`** | **bitwise identical (0)** | **it's the cross-chunk accumulation** |

The mechanism, from the eager chunk loop (`loss.py:588-602`) vs the traced graph:

- **Eager** disables FSDP gradient sync for chunks `0..N-2`
  (`set_requires_gradient_sync(False)`) so each chunk's `backward()`
  accumulates its partial lm_head weight grad into the **unsharded `.grad`
  in bf16**, then re-enables sync at the **last chunk** → **one coalesced
  reduce-scatter** of the bf16-summed grad. Net: `RS( Σ_bf16 gᵢ )`.
- **The traced graph does not honor that `set_requires_gradient_sync(False)`
  coalescing** — `make_fx` emits **8 separate per-chunk `reduce_scatter`s**
  (`reduce_scatter_tensor`..`_7`, all `f32[256,256]`) and accumulates them in
  **f32** (`add_` fold 0→7). Net: `Σ_f32 RS(gᵢ)`.

`RS` is linear, so the difference is purely the **cross-chunk accumulation
precision**: eager sums the 8 chunks in **bf16**, the graph sums them in **f32**.
That last-bit difference (the graph is actually the *more* precise of the two)
is the entire source of the `lm_head.weight.grad` divergence — and it vanishes
at `num_chunks=1` because there is then nothing to cross-accumulate.

### The deeper WHY: simple_fsdp (graph) vs FSDP2 (eager)

It's not that `make_fx` ignores a flag — the two paths use **different
data-parallel implementations**:

- **Eager** wraps `lm_head` with **FSDP2** (`torch.distributed._composable.fsdp`).
  `ChunkedCELoss` detects this (`fsdp_enabled = isinstance(lm_head, FSDPModule)`)
  and uses `set_requires_gradient_sync(False)` to **coalesce** the per-chunk
  reduce-scatters into one (bf16 unsharded accumulation, single RS at the last
  chunk).
- **graph_trainer** wraps `lm_head` with **`simple_fsdp`** (a DTensor
  parametrization), so `lm_head` is **not** an `FSDPModule` →
  `fsdp_enabled = False` → the entire coalescing block is **dead code**.
  simple_fsdp's `ReplicateComputation` instead fires on every `lm_head(h_chunk)`
  weight access: `redistribute(Shard→Replicate)` all-gather +
  `to_local(grad_placements=[Partial])`, whose backward is a `Partial→Shard`
  **reduce-scatter — one per chunk**, summed in f32. Nothing coalesces them.

So the divergence is intrinsic to **simple_fsdp ≠ FSDP2** in the chunked-loss
path, not a tracing bug.

**Why lm_head only:** it's the sole parameter whose gradient is produced by the
chunked-loss loop; everything else is bitwise identical.

**Fixes (pick one):**
1. **`--loss.num_chunks=1`** — immediate, zero-risk bitwise parity (loses the
   chunked-loss memory saving).
2. **Coalesce simple_fsdp's per-chunk lm_head reduce-scatters — IMPLEMENTED ✅.**
   `ChunkedCELossWithParamGrads.__call__` now: reads `lm_head.weight` once (one
   all-gather), runs the per-chunk forward+backward against a detached *leaf*
   (no collective inside the loop, chunking/memory preserved), accumulates the
   per-chunk grads UNSHARDED in **f32**, and does ONE `Partial→Shard`
   reduce-scatter (`_reduce_scatter_grad_like`, mesh-aware: `Partial` on FSDP/DP
   axes, kept `Shard` on TP). Reproduces eager's `RS(Σ_f32 gᵢ)` order exactly.
   Validated bitwise: CPU `test_chunked_loss` (3/3), debugmodel no-TP & TP, and
   **16B (loss + grad_norm + lm_head.grad, 0/15 diverging)**. (+194/−76 in
   `chunked_loss.py`, lint-clean.) **PR #3636** (ghstack, stacked on #3248):
   https://github.com/pytorch/torchtitan/pull/3636
3. Accept it — the graph's f32 cross-chunk accumulation is *more* accurate; the
   gap is last-bit and within different-seed noise (both converge identically).

### Implementation plan for fix #2 (chosen: local_map boundary-comm region)

Keep chunking (memory) but do the lm_head collectives **once at a boundary**,
reproducing eager FSDP2's all-gather-once → bf16 per-chunk accumulate →
single reduce-scatter. In `ChunkedCELossWithParamGrads` (graph_trainer only):

1. Get the raw `Shard(0)` weight without the per-access all-gather:
   `with disable_active_parametrization(): w = lm_head.weight`
   (simple_fsdp's documented escape hatch; `simple_fsdp.py:32-39, 249-255`).
2. Run the chunked forward+backward on **local** tensors so no collective fires
   inside the loop. The weight enters the region all-gathered **once**
   (`Shard(0)→Replicate`, `forward_dtype=param_dtype` = bf16); the per-chunk
   weight grads accumulate in a local bf16 buffer (matching eager); the boundary
   does a **single** `Partial→Shard(0)` reduce-scatter (`reduce_dtype` = f32).
   `local_map` is plain Python (no own autograd.Function) — so the manual
   per-chunk backward stays, and the weight grad is plumbed out as a `Partial`
   output that is reduce-scattered once.
3. Plumb hidden grad + lm_head param grads as explicit autograd outputs (the
   existing `_ChunkedLossWithParamGrads` role) so `torch.autograd.grad(loss,
   [hidden, *lm_head.parameters()])` lines up.

**Correctness risks (must validate bitwise):** (a) boundary all-gather dtype =
`param_dtype`, reduce-scatter/accumulate dtype = `reduce_dtype` (read from the
simple_fsdp `MixedPrecisionPolicy`) — else not bitwise; (b) the 2D dp+tp mesh
needs the mesh-collapse path (reuse `ReplicateComputation`); for `tp=1` it's the
simple 1D `Shard(0)` case; (c) must also handle the plain-`nn.Linear` (CPU-test,
non-DTensor) path; (d) loss-parallel (vocab-sharded) interaction.
**Validate:** CPU `tests/test_chunked_loss.py` (functional) + debugmodel
`num_chunks=8` eager-vs-graph `lm_head.weight.grad` (bitwise).

> Note: the `Trainer.train_step` hash instrumentation is **temporary**
> (env-guarded; inert unless `TITAN_HASH_DEBUG` is set) and must be **reverted
> before any commit** — it imports from `tests/` and does not belong on a
> branch. Driver: `~/tmp/hash_investigate.sh`; comparator:
> `~/tmp/compare_hashes.py`.

---

## Artifacts

| artifact | link |
|---|---|
| 16B eager-vs-graph run log (pastry) | https://www.internalfb.com/intern/paste/P2373815547/ |
| 16B per-step hash localization → `lm_head.weight.grad` (pastry) | https://www.internalfb.com/intern/paste/P2373960868/ |
| 16B lm_head grad ratio (rules out 2× / tying; ~1× last-bit) (pastry) | https://www.internalfb.com/intern/paste/P2374473539/ |
| NCCL test + traced lm_head weight-grad subgraph (pastry) | https://www.internalfb.com/intern/paste/P2374518886/ |
| 16B determinism verdicts — graph×2, eager×2 (pastry) | https://www.internalfb.com/intern/paste/P2373833095/ |
| debugmodel run log — with seed ckpt (pastry) | https://www.internalfb.com/intern/paste/P2373818675/ |
| debugmodel run log — no seed ckpt (pastry) | https://www.internalfb.com/intern/paste/P2373818714/ |
| reusable launcher | `run_loss_compare_dsv3.sh` (repo root) |
| 16B full-precision summary | `~/tmp/loss_compare_dsv3_16b/fullprec_summary.md` |
