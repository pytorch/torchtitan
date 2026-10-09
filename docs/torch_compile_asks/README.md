# torch.compile asks from TorchTitan's regional compile

[Felipe Mello](mailto:felipemello@meta.com)

## TL;DR

- TorchTitan compiles small functions ("regions") with `fullgraph=True`, `recompile_limit=8` and no eager fallback; the rest runs eager.
- Rewrites around those regions take a 2-layer 16k fwd+bwd step from 304.6 to 192.2 ms on Kimi K3 (-37%) and 240.1 to 75.1 ms on DeepSeek-V4 (-69%), beating per-layer `torch.compile`. Several were lowering choices Inductor could make itself.
- 13 asks, each with numbers at real shapes, our workaround, and (for 12) a pure-torch repro in `repros/`:
  - hard failures: graph budgets ending in `FailOnRecompileLimitHit`; an FX graph cache serving code from a stale custom-op fake;
  - biggest perf gaps: opaque deterministic scatter-add (~2.0% of a Kimi K3 microbatch), symbolic strides on packed views (~2%), small reductions not unrolled (~1.6%).
- One GB300, torch 2.15.0.dev20260926; every repro re-run on 2026-10-09 (within ~1%).

**Terms.**
- Region: a function decorated with TorchTitan's `@local_compile`, i.e. `torch.compile(fn, fullgraph=True, **options)`. Block compile: `torch.compile` of a whole transformer layer.
- T: tokens per microbatch per rank. Packed training concatenates documents into a fixed T per step; in RL, T varies per step, so it goes dynamic.
- GraphTrainer: TorchTitan's experimental trainer that traces the whole training step with non-strict `make_fx` and compiles tagged regions with `regional_inductor`.
- Kimi residual: Kimi K3's block-level residual, a softmax-weighted sum over a stack of N <= 8 earlier block outputs `[T, N, 7168]`; 186 calls per forward.
- EP8: expert parallelism over 8 ranks; "rank-local shapes" are one rank's shapes.
- Ideal: minimum HBM bytes / 7.1 TB/s (measured).

**Scale.** Every percentage is against TorchTitan main's default regions: 2-layer harness wall at 16k, no activation checkpointing, times the layer count. Kimi K3 304.6 ms -> ~14 s (93 layers, EP8 rank-local shapes); DeepSeek-V4 flash 240.1 ms -> ~5.2 s (43 layers); DeepSeek-V3 671B 184.5 ms -> ~5.6 s (61 layers).

```text
#   kind                ask                                                           impact (per call; share of a 16k microbatch)
A. hard failures and stale results
1   hard failure        graph budgets under fullgraph (grad mode, guards, call sites) FailOnRecompileLimitHit
2   silently stale      FX graph cache ignores a custom op's fake metadata            old code after a fake fix
3   crash               AOTAutograd AttributeError on a parameter view's ._base       BackendCompilerFailed
B. performance
4   codegen             traceable fixed-order segmented sum (deterministic scatter)   3.1 ms/MoE layer; ~2.0% Kimi K3
5   codegen             static strides/offsets on views of a packed projection        2.6-3.3 ms/MoE layer; ~1.7-2.1% Kimi K3
6   lowering            unroll small reductions (+ GEMV decomposition, int32 index)   up to 1.2 ms/call; up to ~1.6% Kimi K3
7   lowering            cost model for cat/stack lowering                             1.37 ms/layer; ~1.1% DSv4
8   codegen             complex-multiply codegen                                      0.25-0.42 ms/call; ~0.2% DSv4, ~0.5% DSv3
9   scheduling          fuse tiny-block reductions across steps (Sinkhorn)            0.2 ms/call; ~0.4% DSv4
10  region semantics    views and in-place gradients from regions                     0.38 ms/layer; ~0.4% DSv3
11  lowering            don't inline a producer into a high-broadcast consumer        0.95 ms/call; no_grad forward only
12  Triton              masked loads halve a many-load kernel on SM103                0.35 ms/call; dynamic T (RL) only
C. compile time
13  partitioning        regional_inductor partitioning scales superlinearly           382 s per process start, DSv4 4 layers
```

Repros: `./repros/run_all.sh cpu` and `./repros/run_all.sh gpu`. Raw outputs: `repros/logs/`.

---

## A. Hard failures and stale results

### 1. Graph budgets under fullgraph

**Ask:**
- (a) let a `no_grad` call reuse the grad-mode graph's forward, or count signatures per grad mode. The forward ops are identical; the training graph only adds saved tensors and the backward. Today a `no_grad` validation or RL logprob pass costs a full extra graph per shape class against the same limit of 8;
- (b) performance heuristics should never install guards (`statically_known_true` / `guard_or_false`, falling back to the unfused schedule);
- (c) per-call-site budgets, or a limit that doesn't count shape-class x grad-mode products against one code object.

**Impact:** hard failures. Under `fullgraph=True` every guard failure is a new graph against `recompile_limit=8`, ending in `FailOnRecompileLimitHit` with no eager fallback. TorchTitan main already carries the workaround for one region: `fused_binary_activation` uses `recompile_limit=16` because "DeepSeek-V3 at TP8 needs 10" (`torchtitan/models/common/activation.py:31-32`).

Four sources of graphs stack up:

```text
(a) grad mode is a global-state guard (`ask01b`):
    after training at 3 token counts:   graphs = 2   (static T, then dynamic T)
    after one no_grad validation pass:  graphs = 3   guard failure "GLOBAL_STATE changed: grad_mode"
    5 input ranks x 2 grad modes:       FailOnRecompileLimitHit at signature 9

(b) the mix-order reduction heuristic installs a shape guard (`ask01`, Kimi residual fwd+bwd, T = 16k, 8k, 2k):
    default:                                     3 graphs; guard "4096 <= stack.size()[0]"  (scheduler.py:492, can_fuse)
    triton.mix_order_reduction_non_strict_mode:  2 graphs

(c) one norm region shared by call sites of different rank/layout (`ask01c`, Qwen3.5 OffsetRMSNorm:
    2D layer norms + 3D q/k norms, q a strided view; training 16k/8k/2k then no_grad 16k/2k):
    as written 9 graphs | non_strict option 7 | contiguous q 7 | eager-side marks 8 | marks + option 6

(d) size-0/1 dims always specialize (`ask01d`):
    widths 1,2,3,5,8,1:                      2 graphs (N=1 static, then dynamic N)
    maybe_mark_dynamic on the size-1 dim:    2 graphs
    mark_unbacked on the width:              1 graph  (must be applied at every call site)
```

**Why:**
1. (b) is deliberate: the scheduler calls `evaluate_expr(..., size_oblivious=True)` on `nrow >= 4096` "since nrow can have dynamic shape in real models" (`torch/_inductor/scheduler.py:470-494`). Under fullgraph that heuristic becomes a budget item.
2. In the Kimi residual, two token counts use 4 graphs ({N=1, dynamic N} x {static T, dynamic T}); a `no_grad` pass adds 2 and the mix-order guard up to 2 more: exactly 8. A routed MoE call with 0 or 1 rows (an EP rank receiving no tokens) also costs a graph per grad mode.
3. Qwen3.5's `offset_rmsnorm` region hit `FailOnRecompileLimitHit` in a budget probe; the option alone or eager-side marks alone still leave it at the limit (8 of 8); both together give 6.

**Workaround:** the variants in the blocks above, plus raising the limit.

**Repros:** `repros/ask01_mix_order_guards.py`, `ask01b_grad_mode_guard.py`, `ask01c_shared_norm_call_sites.py`, `ask01d_size1_specialization.py` (GPU); `ask01c_shared_norm_call_sites_cpu.py` (CPU, Dynamo guards only: strided q 7 graphs).

### 2. The FX graph cache ignores a custom op's fake output metadata

**Ask:** include the fake's output metadata (or the op library's version) in the FX graph cache key.

**Impact:** after fixing a custom op's fake, code compiled from the broken fake keeps running until the cache directory changes. We lost time to this fixing attention-gym's GDN fakes.

```text
fake v1 (wrong strides), fresh cache A  -> AssertionError: expected size 32==32, stride 64==1 at dim=0 ...
fake v2 (fixed),         same cache A   -> the same AssertionError (code compiled from v1 is reused)
fake v2,                 fresh cache B  -> ok
CPU: FX graph cache on -> stale; TORCHINDUCTOR_FX_GRAPH_CACHE=0 -> ok
```

**Workaround:** a fresh `TORCHINDUCTOR_CACHE_DIR` per variant when developing fakes.

**Repro:** `repros/ask02_cache_ignores_fake.py` (GPU; the cache behavior also reproduces on CPU).

### 3. AOTAutograd crashes with an internal AttributeError when an `autograd.Function` inspects a parameter view's `._base`

**Ask:** at minimum a clear error instead of an internal `AttributeError`; ideally guard on or specialize `._base` from the input's source, or graph-break on it.

**Impact:** a crash, even with `fullgraph=False`. It breaks `torch.compile` around a torch.library MoE backend that walks `weight._base` to accumulate weight gradients in place (TorchTitan's default).

```text
with grad_dtype read            eager     ok
with grad_dtype read            aot_eager FAIL BackendCompilerFailed | AttributeError: 'NoneType' object has no attribute 'is_leaf'
with grad_dtype read            inductor  FAIL (same)
without grad_dtype read         all backends ok   (1 graph, autograd_function_apply HOP)
with grad_dtype read, no view   all backends ok
explain: 3 graphs, 2 breaks; graph 1 = l_w_._base ; param.is_leaf ; return ()
```

**Why:** reading `grad_dtype` graph-breaks, so the Function's prefix becomes its own graph with the view as input and `l_w_._base` as a node. Under AOTAutograd that input is a fresh functionalized fake with no view link, so `_base` is `None` and the next read raises in `run_functionalized_fw_and_collect_metadata` (`torch/_functorch/aot_autograd.py:596`).

**Workaround:** turn off in-place weight-gradient accumulation (then one graph, 0 breaks, bitwise vs eager).

**Repro:** `repros/ask03_autograd_fn_param_base.py` (CPU).

---

## B. Performance

### 4. A traceable fixed-order segmented sum (deterministic scatter-add)

**Ask:** a traceable deterministic scatter-add lowering when each output row has a fixed set of contributors, or a fixed-order segmented-sum primitive Inductor can fuse.

**Impact:** 3.1 ms per Kimi K3 MoE layer, ~290 ms per 16k microbatch (~2.0%).

```text
Kimi K3 MoE combine: 262144 expert-sorted rows x 3584 bf16 -> 16384 tokens, fwd+bwd
eager as written             13555 us   42 kernels
compiled as written           4220 us   27 kernels   (indexing_backward_kernel 2705 us survives inside the custom op)
compiled per-token rewrite    1095 us    2 kernels   ideal 827 us
```

**Why:** the custom op exists to force deterministic accumulation. Once it is opaque, compile only fuses the scaling around it, and the op's sort-based deterministic `index_put` dominates.

**Workaround** (pytorch/torchtitan#5157): every token owns exactly K rows, so combine is an `autograd.Function` whose forward is an unrolled per-token fp32 sum and whose backward is a gather by token. Deterministic, no atomics, 2-3x more accurate vs fp64.

**Repro:** `repros/ask04_opaque_custom_op.py` (GPU).

### 5. Static strides and offsets on views of a packed projection

**Ask:**
- (a) keep a view's stride and offset static when they are fixed multiples of a static or specialized size;
- (b) a way to mark strides and offsets static alongside sizes;
- (c) prefer a static graph that accepts the inputs over the newest symbolic one.

**Impact:** 2.6-3.3 ms per Kimi K3 MoE layer, ~240-300 ms per 16k microbatch (~1.7-2.1%). TorchTitan main now works around it with `mark_static` ("~1.9x slower fwd+bwd" without it, `torchtitan/models/common/activation.py:51-54`), at ~3 graphs per F.

```text
SwiGLU on gate, up = unbind(-2) of a packed [16384, 2, 6144] bf16 projection, fwd+bwd
F static (compiled at 6144 only)       361 us
F dynamic (3072 first, then 6144)      602 us
Kimi K3 SiTUGLU, shared expert T x 6144:
  static 462.9 us | dynamic F 845.6 us | mark_static on F 555.1 us | + torch._check on stride and offset 470.6 us
```

**Why:**
1. Once F is symbolic, the views' row stride (2F) and `up`'s storage offset (F) get their own symbols, and kernels index with `xindex % ks0` instead of a constant.
2. `mark_static` pins sizes only; the stride and offset need `torch._check` too. Separately allocated gate/up with dynamic F show no slowdown.
3. For (c): Dynamo checks the most recently added cache entry first, so once one call site creates the symbolic graph, every call site it accepts runs it, including shapes that had a static graph.

**Workaround:** `torch._dynamo.mark_static` on the hidden dim inside the region (on TorchTitan main).

**Repro:** `repros/ask05_symbolic_hidden_dim.py` (GPU).

### 6. Unroll small reductions

**Ask:** unroll when a symbol's upper bound is below `unroll_reductions_threshold`; raise or autotune the threshold (8, strict `<`) for reductions over a non-innermost dim.

**Impact:** up to 1.2 ms per Kimi residual call, up to ~230 ms per Kimi K3 16k microbatch (~1.6%); separately ~1 ms per MoE layer for dispatch at K = 16 (~0.6%).

```text
(a) Kimi residual weighted sum [16384, N, 7168] + partial, forward:
N=7 static                    314 us   pointwise (unrolled)     ideal 298 us
N=8 static                   1572 us   persistent reduction     ideal 331 us
N=8 static, threshold 9       351 us   pointwise
N=7 dynamic                  1245 us   looped reduction
N=7 dynamic, threshold 64    1245 us   the threshold never applies to symbolic sizes

sum of each token's K=16 rows of [262144, 3584] bf16 (MoE dispatch backward):
gather + .sum over K                  1354 us   persistent reduction
same, threshold 16                    1354 us   (strict <)
same, threshold 17                     290 us   pointwise              ideal 281 us
contiguous [T,16,D] .sum over K       1001 us   (3.6x off bandwidth without any gather)
```

**Why:** Inductor unrolls only static sizes below the threshold (`torch/_inductor/ir.py:1903-1908`). In production N varies 1-8 per layer, so after the second width N is symbolic and every call takes the looped reduction (Kimi residual fwd+bwd 2402 us static vs 3642 us dynamic). `split_reductions`, cooperative reductions and multi-kernel don't help.

Two related asks hit the same op:

- **(b) decompose memory-bound batched GEMVs by default.** TorchTitan main writes the Kimi residual as `softmax(scores)[T,1,8] @ values.float()[T,8,7168]`. As a bmm it goes to cuBLAS with a separate materialized fp32 upcast (1.9 GB at 16k). `decompose_mm_pass` already handles the shape but is opt-in, has a 10240-row floor, and skips fp32 bmm under BF16x9 (`torch/_inductor/fx_passes/decompose_mem_bound_mm.py:62-90`). Repro `ask06b_extern_gemv.py`:

  ```text
  fwd+bwd                     T=16384    T=2048
  bmm as written              3478 us    460 us   (10 kernels, extern gemv)
  bmm + decompose_mm_pass     2356 us    460 us   (pass doesn't fire below 10240 rows)
  explicit sum                2157 us    275 us   ideal 794 / 99 us
  ```

- **(c) int32 indexing for bounded symbolic sizes.** `_decide_tl_dtype` types every symbolic size argument as `tl.int64` (`torch/_inductor/codegen/triton_utils.py:259-278`) and ignores `torch._check` bounds: dynamic-N residual fwd+bwd 1919 us default, 1915 us with `torch._check(N <= 8)`, 1817 us with `assume_32bit_indexing`. That knob fails to compile once T*N*D > int32 max (T=65536, N=8). 0.14 ms per call, up to ~0.2%. Repro `ask06c_int64_indexing_symbolic.py`.

**Workaround:** an `autograd.Function` with a hand-written backward and an explicit weighted sum (pytorch/torchtitan#5156: 16k fwd+bwd 3642 -> 2652 us vs 2403 static); a Python loop over `top_k` for dispatch (pytorch/torchtitan#5157). A region-scoped `unroll_reductions_threshold=17` gives the same dispatch kernel bitwise.

**Repros:** `repros/ask06a_symbolic_small_dim.py`, `ask06a_small_k_in_tile.py`, `ask06b_extern_gemv.py`, `ask06c_int64_indexing_symbolic.py` (GPU).

### 7. A cost model for cat/stack lowering

**Ask:** a cost model (or autotuning) between pointwise-cat and ConcatKernel lowering that accounts for input layouts and downstream reductions; and lower an interleave/cat consumed by a reduction without re-loading the row per branch.

**Impact:** 1.37 ms per DeepSeek-V4 flash layer for inverse RoPE alone, ~59 ms per 16k microbatch (~1.1%).

```text
DeepSeek-V4 attention shapes [16384, 64, 512] bf16, fwd+bwd
(a) inverse rope on o          default cat 2158 us (2 kernels) | ConcatKernel  781 us (5 kernels)   ideal 605 us
(b) per-head RMS norm + rope   default cat 1027 us (2 kernels) | ConcatKernel 1502 us (7 kernels)   ideal 756 us
```

**Why:**
1. The default fuses the cat into one masked pointwise kernel with per-element index math: 2.8x slower than per-input copies for (a), but faster for (b), where it lets the cat fuse into the norm's persistent reduction. No global setting is right: on H100 the same ConcatKernel option made Qwen3.5's partial RoPE 24-26% slower in training.
2. Inside a persistent reduction (DeepSeek-V4 q norm + RoPE, 455 us vs ideal 302 us), the split + interleaved stack + cat re-loads the q row 4 times under different masks, with int64 `%` / `//` index math.
3. It interacts with rewrites: on a DeepSeek-V3 attention layer, real-arithmetic RoPE alone was slower (46.95 ms) and ConcatKernel alone neutral (46.37), both together 46.34 -> 45.85 ms.

**Workaround:** `options={"max_pointwise_cat_inputs": 0, "max_complex_pointwise_cat_inputs": 0}` per region, chosen by measurement (pytorch/torchtitan#5197, #5161).

**Repro:** `repros/ask07_cat_lowering.py` (GPU).

### 8. Complex-multiply codegen

**Ask:** lower complex `mul` (with `view_as_complex` / `view_as_real`) to real arithmetic.

**Impact:** 0.25 ms per DeepSeek-V4 RoPE call, ~11 ms per 16k microbatch (~0.2%); 2 x 212 us per DeepSeek-V3 attention layer, ~26 ms per microbatch (~0.5%).

```text
DeepSeek-V4 RoPE on q [16384, 64, 64] bf16, complex64 cache, fwd+bwd
eager complex              446 us   7 kernels
compiled complex           444 us   7 kernels   (ATen complex-mul fallback, 214 us, stays)
compiled real arithmetic   199 us   2 kernels
```

**Why:** Inductor keeps the ATen fallback ("does not support code generation for complex operators", printed only on a cold compile), which also blocks fusing the split/cat glue around it.

**Workaround:** a real-arithmetic branch under `torch.compiler.is_compiling()` (pytorch/torchtitan#5196).

**Repro:** `repros/ask08_complex_ops.py` (GPU).

### 9. Fuse tiny-block reductions across steps (Sinkhorn)

**Ask:** fuse reductions over a tiny trailing block (4 x 4 per token) across steps, e.g. by keeping the block in registers.

**Impact:** 0.20-0.23 ms per call, ~20 ms per DeepSeek-V4 flash 16k microbatch (~0.4%).

```text
DeepSeek-V4 hyper-connection Sinkhorn: 20 row/column normalizations of [16384, 4, 4] fp32, fwd+bwd
[T,4,4] tensor form         287-316 us   159 kernels   rel vs fp64 8.8e-8
16 per-entry [T] tensors     84 us        32 kernels   rel vs fp64 8.8e-8
```

**Why:** each column normalization reads the previous step at transposed indices, so consecutive steps don't fuse and every intermediate is realized. With one `[T]` tensor per entry every read is same-index.

**Workaround:** 16 per-entry tensors in the compiled branch (pytorch/torchtitan#5199), at ~42 s more cold compile for a 2-layer harness.

**Repro:** `repros/ask09_sinkhorn_transposed_reads.py` (GPU).

### 10. Views and in-place gradients from regions

**Ask:** let regions return views of their inputs without a gradient copy, and allow in-place updates of an incoming gradient used nowhere else.

**Impact:** these block replacing hand-written Triton overrides with compile: 0.38 ms per DeepSeek-V3 attention layer, ~23 ms per 16k microbatch (~0.4%); 0.42 ms per Kimi K3 MLA layer (~10 ms, ~0.07%).

```text
one DeepSeek-V3 attention layer, 16k fwd+bwd, bitwise vs eager, 3 processes
mla_qk region (dense v, complex RoPE fallback)                46.34 ms
+ real-arithmetic RoPE + ConcatKernel                         45.85 ms
+ RoPE in flip form                                           45.77 ms
+ v as a view, via an autograd.Function around two regions    45.48 ms
fused Triton override (v a view, grad_q rotated in place)     45.10 ms   (q RoPE bwd 120 us in place vs 303 us fresh)
Kimi K3 MLA k/v: a region returning v as a view of kv leaves a 419 us eager copy of v's gradient
```

**Why:** AOTAutograd copies the gradient of an output that aliases an input into the view's strides, and its backward always writes a fresh input gradient. The view half has a workaround (return the view from an `autograd.Function` outside the regions); the in-place half has none.

**Workaround:** the fused MLA override on TorchTitan main for DeepSeek-V3.

**Repro:** none standalone (numbers from the comparison script in pytorch/torchtitan#5161).

### 11. Don't inline a producer into a high-broadcast consumer

**Ask:** account for the consumer's broadcast factor when deciding to inline a producer with expensive per-element ops (exp, div).

**Impact:** 0.95 ms per Kimi residual call under `no_grad` (evaluation, RL prefill), ~177 ms per Kimi K3 16k no_grad forward. Training is unaffected.

```text
softmax + zero-pad of [T, N+1] scores feeding an 8-way weighted sum over [16384, 8, 7168], no_grad forward
default                      1225 us   one kernel recomputes 8 exp/div per output element
realize_reads_threshold=1     277 us   the [T, 8] probs are realized first        ideal 265 us
```

**Why:** in training the probs are saved for backward, so they are realized and the kernel is fast. Under `no_grad` Inductor inlines the softmax into the consumer, multiplying its cost by D = 7168. `realize_reads_threshold=1` region-wide slows the backward (fwd+bwd 3106 us).

**Workaround:** none for `no_grad`.

**Repro:** `repros/ask11_inline_recompute.py` (GPU).

### 12. Triton (not Inductor): masked loads halve a many-load kernel on SM103

Dynamic T (RL) only; packed training is unaffected.

**Ask:** Triton triage with the two standalone kernels. On the Inductor side, the mask is provably unnecessary only when XBLOCK divides 7168.

**Impact:** 0.35 ms per Kimi residual call, ~65 ms per Kimi K3 16k forward (~0.5%). Filed here because Inductor emits the mask; standalone Triton kernels attached.

```text
Inductor, forward:   residual weighted sum   T static 314 us | T dynamic 663 us
                     swiglu (control)        T static 134 us | T dynamic 134 us
standalone generated kernels, us by XBLOCK/num_warps:
  unmasked  512/4 316, 512/8 564, 1024/4 322, 1024/8 317, 2048/4 309, 2048/8 329
  masked    512/4 597, 512/8 644, 1024/4 699, 1024/8 607, 2048/4 834, 2048/8 746
```

**Why:** the two sources differ only by `xmask = xindex < xnumel` on 18 loads (8 broadcast loads of per-token softmax values); both carry `tt.divisibility=16`. The slowdown holds at every config, so it is not autotuning. The mechanism is not established.

**Repros:** `repros/ask12_masked_loads_symbolic_T.py`, `ask12_standalone_masked_kernels.py` (GPU).

---

## C. Compile time

### 13. `regional_inductor` partitioning scales superlinearly

**Ask:** build the dependency map once and share it across regions (or partition all region ids in one pass), and bound the merge loop's cycle checks.

**Impact:** DeepSeek-V4 regional compile under GraphTrainer took 382 s at 4 layers on every process start; a warm cache doesn't help.

```text
repro: scoop_regions only, synthetic graph (untagged matmul + 8 tagged pointwise ops per layer)
tagged nodes                  400     800    1600    3200
one shared region id         0.24    1.43   10.91      -   s
one region id per layer      0.18    1.17    7.69   62.10  s   (~N^2.9)
```

**Why:** with one shared id (closest to production), 45,448 `maybe_merge_partition` attempts each run a cycle check (`dfs_iter_find_cycle`). With many ids, each region builds its own `CapabilityBasedPartitioner` over the whole graph, and each builds a transitive-downstream set per node (`torch/fx/passes/infra/partitioner.py:50-58`).

**Workaround:** none.

**Repro:** `repros/ask13_regional_partition_scaling.py` (CPU).

---

## Known behaviors (one line each)

- **FP contraction:** compiled is not bitwise vs eager for fused pointwise math (RoPE: 81 of 4.2M elements, 1 bf16 ulp); `emulate_precision_casts=True` makes it bitwise, `TRITON_DEFAULT_FP_FUSION=0` has no effect. TorchTitan's `test_rope_compile` (`rtol=atol=0`) fails on main on GB300. Repro `kb_fma_bitwise.py`.
- **Region options are dropped when inlined:** a nested `torch.compile(fn, options=...)` inside an outer compile keeps only `fullgraph` (`torch/_dynamo/variables/functions.py:3095-3110`), so region-scoped Inductor options (e.g. `unroll_reductions_threshold=17`, bitwise equal to a hand-unrolled loop) vanish under block compile; we ship hand rewrites instead.
- **`maybe_mark_dynamic` is forbidden in graphs** (unlike `mark_static`): a helper that calls it crashes once it is itself compiled; guard with `is_compiling()`. Repro `kb_maybe_mark_dynamic_traced.py`.
- **make_fx captures an attribute tensor as a constant:** in-place updates are seen, rebinding is silently ignored (TorchTitan's aux-loss denominator hit this under GraphTrainer; main now passes it as an argument). Repro `kb_make_fx_tensor_constant.py`.
- **make_fx specializes Python ints silently:** a graph traced with `n=3` returns 3 values when called with `n=5`. Repro `kb_make_fx_int_specialization.py`.
- **`functools.cache` under fake tensors:** a cached factory first called under `FakeTensorMode` caches a FakeTensor and breaks the next eager call. Repro `kb_cache_under_fake_tensors.py`.
- **`regional_inductor` region tagging:** `fx.traceback.annotate` tags every node created inside the context, so under SimpleFSDP the parameter all-gathers (and, via forward-to-backward metadata copy, the reduce-scatters) join the compiled partition and can lose overlap; annotated nodes without an `inductor_region` key also merge into one partition (`torch/fx/passes/regional_inductor.py:143-160`, `torch/_functorch/_aot_autograd/utils.py:660`).
- **Storing a new attribute on an outer-scope object inside a HOP** fails with a generic 'Observed exception'; a 'Non-nullified side effect' error would point at the cause. Repro `kb_hop_outer_attr_store.py`.

## No longer applies to TorchTitan

- **SelectiveAC + `wrap_inductor_compiled_regions` crashes after a recompile** in plain torch (`repros/nla_sac_wrapped_region_recompile.py cold`: recompute pops another graph's saved outputs), but TorchTitan main compiles regions inside `torch_remat` checkpoint blocks and that path passes (`repros/nla_remat_checkpoint_wrapped_regions.py`). Rule: compile inside a remat region is fine; avoid remat inside a compiled region.

## Observations without a minimal repro

- **Small-k `topk` falls back to ATen.** DeepSeek-V3 router glue at 4k: compiled 159.7 us, of which ATen `topk` 96.9 us; with `triton.decompose_sort_ops` 106.4 us. ~0.05 ms per MoE layer.
- **Gradient-accumulation adds fold into GEMM epilogues only under whole-graph compile.** With regions, grouped-GEMM weight gradients keep autograd's standalone bf16 AccumulateGrad adds: ~1.3 ms per DeepSeek-V3 MoE layer at any token count (8.5 GB per add).
- **Multi-output reductions are partitioned by output shape.** DeepSeek-V4's hyper-connection output backward (16k): `[T,4,4]`, `[T,4,D]` and `[T,D]` gradients become 3 kernels (558 us, ideal ~265), reading the gradient 3 times.
- **Whole-layer Inductor can be slower than the region boundary** (DeepSeek-V3, 4k, GraphTrainer full Inductor vs regions; `repros/obs_whole_graph_vs_region.py`): the routed SwiGLU backward lowers `unbind` as a masked cat (0.298 vs 0.113 ms with ConcatKernel), and the MoE dispatch backward recomputes the router's sigmoid through 4 dependent index loads (0.42 vs 0.075 ms).

## How we measured

- One GB300 (SM103), torch 2.15.0.dev20260926+cu130, triton 3.8.0. Kernel time = sum of profiler CUDA kernel durations per call. Ideal = minimum HBM bytes / 7.1 TB/s (measured).
- Repro numbers: `repros/logs/` (one log per script from a fresh `run_all.sh cpu` + `gpu` on 2026-10-09; GPU timings are kernel time, matching earlier runs within ~1%).
- Per-microbatch estimates multiply a measured per-call saving by the number of calls in the model; percentages use the Scale line above (main's default regions). Model-level numbers without a repro come from TorchTitan layer harnesses (2 layers at native dims, medians of 3 processes).
- TorchTitan citations are at main 01b1b69e3; torch citations at the torch version above.
