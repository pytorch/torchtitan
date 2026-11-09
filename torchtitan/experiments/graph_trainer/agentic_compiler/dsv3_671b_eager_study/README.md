# Graph-found optimizations in eager: DeepSeek-V3 671B

The question, as for Qwen3 30B-A3B (`../qwen3_30b_eager_study/`): optimizations can be applied in both
eager and graph mode; what fraction of the optimizations the agentic compiler found in the graph can eager
mode achieve?

Subject: the DSv3 671B DistMoE MXFP8 loop (`../dsv3_671b_distmoe_mxfp8/`): 20 verified CuTe DSL kernels
(18 patterns) for 13 of its 15 lifted regions. In that loop's setup (FSDP 256 x EP 64 without PP on the fake
PG, one 4096-token microbatch per step, fused MLA and SwiGLU, deterministic), the per-kernel savings add up
to 75.7 ms/step (its README section 7). DSv3 differs from Qwen3 in ways that matter here:
- **GraphTrainer recomputes.** Its default memory policy recomputes forward ops in backward, down to the
  attention projections' GEMMs. Eager DSv3 runs without activation remat.
- **MLA attention, shared experts, and a sigmoid router with group-limited top-k.**
- **One region crosses a layer**, and three regions are FSDP gradient-bucket glue.

Short answer (step 4): eager reaches 93% of the graph's GPU-time saving and the same throughput gain,
bitwise, with 16 of the 20 kernels. The other four exist only because of the graph's recompute and its
gradient-bucket layout.

## Step 1: classification of the patterns

### Method

The Qwen3 study's `classify.py` and `sites.py` (`../qwen3_30b_eager_study/`) replay the loop's final plan
(`../dsv3_671b_distmoe_mxfp8/plans/plan_iter9.txt`) on the graph the lift passes see. The graph is the
matcher-robustness experiment's reference dump of the same recipe and layout (a local workspace file, not
committed); `classify.json` holds the per-region output. The phase now distinguishes recompute: a forward op
that the memory policy duplicated into the backward (`*_recomputed` nodes).

Every region has its expected site count: 61 per layer-wide region (3 dense + 58 MoE layers), 58 for the
MoE-only ones, and twice that for the regions whose forward ops are recomputed. Findings:

- **Half the forward sites are recompute sites.** G-047 and G-082 each match 61 forward and 61 recompute
  sites; G-020 (router) and the backward half of G-021 / G-061 (expert weight preparation, redone after
  the backward all-gather) are recompute only. The forward and recompute sites of one region need
  different kernels: the forward copy drops what backward will recompute (G-047's forward kernel `z47c`
  returns neither the norm output nor `rstd`; its recompute kernel `use7` returns both).
- **One region crosses a layer.** G-049 (rmsnorm backward + residual-gradient add + row and column MXFP8
  quantization) has two forms: at 61 sites ffn_norm's backward feeds the same layer's `wo` (kernel `a6rv`);
  at 60 sites layer L's attention_norm backward feeds layer L-1's FFN down projection (`3ybb`): the
  quantization belongs to the previous layer's `w2` backward.
- **Autograd-engine adds** appear in two regions, as in Qwen3: the router scores' consumers (3 per G-006
  site) and the residual-stream gradient (1 per G-049 site).
- **Three regions are FSDP gradient-bucket glue** (G-003, G-028, G-007's permutation): copies that build
  the reduce-scatter input.

### Classification

Saving = the loop's verified per-kernel saving (its README section 7), ms/step.

| Region | Kernel(s) | What | Phase (sites) | Scope | Saving |
|---|---|---|---|---|---:|
| G-001 | gjn6 | router: sigmoid, group-limited top-8, normalized and scaled scores, routing map and counts, aux-loss value, padding masks | F (58) | cross-module: router, aux_loss, routed_experts | 13.8 |
| G-006 | uu33 | router backward: masks, aux-loss gradient, normalization, sigmoid backward, split of the fp32 gate gradient into bf16 pieces | B (58) | cross-module: routed_experts, aux_loss, router, gate (HiMidLo) | 11.3 |
| G-007 | 5nj2 | shared-expert w13 WGRAD GEMM (unchanged) + FSDP's row-block reorder of a dim-1-sharded gradient | B (61; 58 with a kernel) | FSDP gradient glue | 9.0 |
| G-021, G-061 | x6t6, 5uzo | Dist-MoE `prepare_mxfp8_weight` of the gathered w13 / w2 | F (58 + 58), R (58 + 58) | FSDP parameter path | 7.8 |
| G-003 | ponk | wo WGRAD GEMM (unchanged) + reduce-scatter bucket pack of the block's attention / shared-expert gradients | B (58) | FSDP gradient glue | 5.7 |
| G-035 | k6pn | fused MLA KV backward + row and column quantization of the wkv_b gradient | B (61) | cross-module: fused_mla's KV Function, wkv_b | 5.6 |
| G-020 | h4xd | router recompute: the tensors G-006 reads | R (58) | cross-module: router, aux_loss | 4.7 |
| G-049 | a6rv, 3ybb | rmsnorm backward + residual-gradient add + row and column quantization | B (61 + 60) | 61 cross-module (ffn_norm, wo); 60 cross-layer (attention_norm, previous layer's w2) | 4.4 |
| G-028 | czaf | routed-expert reduce-scatter bucket pack | B (58) | FSDP gradient glue | 4.1 |
| G-047 | z47c (F), use7 (R) | attention_norm + two rowwise quantizations (wq_a, wkv_a) | F (61), R (61) | cross-module: attention_norm, wq_a, wkv_a | 2.1 + 1.7 |
| G-082 | c2qd, 5yin (F), dre3 (R) | residual add + ffn_norm + rowwise quantization (shared or dense w13) | F (61), R (61) | cross-module: residual, ffn_norm, w13 | 1.6 + 1.5 |
| G-015 | lfu5 (3 shapes), bqhy | gradient row and column quantization + DGRAD GEMM (unchanged) | B (245; 241 with a kernel) | one MXFP8 linear's backward | 2.4 |
| G-031, G-073 | - | no promoted kernel | | | 0 |

By scope (of 75.7 ms/step):

| Scope | Regions | Saving | Share |
|---|---|---:|---:|
| Cross-layer | G-049's 60 cross-layer sites | 2.2 | 3% |
| Cross-module within a layer | G-001, G-006, G-020, G-035, G-047, G-082, G-049 (same-layer sites) | 44.5 | 59% |
| One linear's backward | G-015 | 2.4 | 3% |
| FSDP parameter and gradient paths | G-021, G-061, G-003, G-028, G-007 | 26.6 | 35% |

By phase: forward 21.4 ms (28%), recompute 11.8 ms (16%), backward 42.5 ms (56%); the weight preparation
splits evenly between forward and recompute.

### What this means for eager

The Qwen3 study's rules carry over: a forward kernel needs its backward in the same Function, a backward
region needs one Function owning every forward op it differentiates, engine adds are fused by returning a
tensor once per consumer, and the saved tensors are the kernel's contract. DSv3 adds three cases:

1. **Recompute regions have no eager counterpart.** Eager saves the tensors G-020 recomputes, so it never
   pays G-020's 86.7 us per call. But its forward must produce what backward reads, which is the recompute
   kernel's contract, not the forward kernel's: eager's attention_norm forward returns the norm output and
   `rstd` (saved for backward) plus both quantizations, exactly `use7`, so `z47c` has no use in eager; the
   same holds for `dre3` against `c2qd` / `5yin`. For the router, G-001's kernel gives the routing outputs
   and G-020's kernel, run in the forward, gives the tensors to save.
2. **A cross-layer region needs a handoff between layers.** The quantized gradient that G-049 produces in
   layer L's backward is an input of layer L-1's `w2` backward, but autograd passes only the bf16 gradient
   between them. The handoff: layer L stores the quantized operands keyed by the gradient tensor it
   returns, and layer L-1's FFN backward uses them when it receives that same tensor (the engine passes it
   through the residual and the MoE output adds unchanged).
3. **FSDP glue lives in FSDP2.** Eager FSDP2 builds the same reduce-scatter input with the same ops: it
   reorders a dim-1-sharded gradient with `torch.chunk` + a 256-way `torch.cat` (G-007's cat), then copies
   all gradients into the flat buffer with `torch._chunk_cat` (what GraphTrainer's
   `bucketing._pre_bucket_reduce_scatter` lowers to, G-003 and G-028). FSDP2 has an experimental hook for
   this copy-in, `FSDPModule.set_reduce_scatter_input_fn`, so these regions are reachable when a kernel's
   contract fits FSDP2's buffer layout.

### Overlap with what eager already does

- **The recipe's `local_compile` regions** (`loss`, `fused_binary_activation`, `fp32_to_bf16_split`,
  `moe_aux_loss`, `rope`, `mla_qk`; GraphTrainer runs without them). `fp32_to_bf16_split` covers the gate
  gradient split inside G-006 and `moe_aux_loss` the aux-loss statistics inside G-001; step 2 measures them.
- **PyTorch's experimental `reduce_scatter_input_fn_with_native_copy`** (`torch.distributed.fsdp.experimental`)
  copies dim-1-sharded gradients straight into the reduce-scatter buffer, removing G-007's cat without a
  custom kernel. torchtitan does not install it.
- **Already fused in eager, nothing for the graph to find:** FA4 attention, Dist-MoE's grouped GEMMs, the
  fused MLA RoPE and SwiGLU kernels of the recipe's overrides, quack's RMSNorm.


## Step 2: eager in the loop's setup

`eager_recipe.py` is the eager twin of the loop's recipe: `deepseek_v3_671b_dist_moe_mxfp8` with the loop's
layout changes (`example_recipe.dsv3_671b_base`: FSDP 256 x EP 64 without PP, gradient reduction not
deferred, one 4096-token microbatch per step, Dist-MoE scratch capacity 64), deterministic, seed 42, fake PG
with NGPU=256 on one GB300. `graph_recipe.py` runs the loop's recipe without and with its plan and kernel
cache in the same setup. Measurements as in the Qwen3 study:
- **GPU time per step** (`busy.py`): the union of kernel intervals in profiled step 10, CUDA graphs off,
  excluding kernels launched under a c10d collective op (the fake PG's emulated collectives, about 180 ms
  here). FSDP2's copy-in / copy-out and GraphTrainer's bucket packing count.
- **Throughput:** tokens/s at step 20, CUDA graphs as the recipe sets them, no profiler.

GPU clocks make run-to-run GPU time vary by up to 2% (about 10 ms) when the host's other GPUs are busy, as a
uniform slowdown of every kernel. Step 4's GPU times are therefore from alternating runs on one GPU with no
other run of this study on the host, and per-Function savings are counted from kernel call counts
(`account.py`), not from differences of totals.

**Numerics.** Unlike Qwen3, eager and GraphTrainer differ from step 1: loss 12.243885 eager vs 12.281303
graph, grad_norm 64.78094 vs 63.61986 (at step 10: loss 12.99760 vs 12.47923). Both are deterministic. The difference is already in the step-1
forward, so it is in the initial state or the forward, not in gradient reduction; it was not investigated
further (one candidate: the fake PG's emulated collectives during initialization, which FSDP2 and
SimpleFSDP use differently). Eager variants are checked against eager with `local_compile` off.

**Memory.** Eager peaks at 205.3 GiB, the graph at 139.5 GiB: eager saves what the graph recomputes.

**Eager's own `local_compile` regions** (each alone against `local_compile` off; saving counted from kernel
call counts):

| Region | Saving ms | Numerics | Overlaps |
|---|---:|---|---|
| `moe_aux_loss` | 3.1 | bitwise | G-001's aux-loss statistics |
| `loss` | 1.9 | differs | none (the plan has no loss region) |
| `fp32_to_bf16_split` | 0.9 | bitwise | G-006's gate-gradient split |
| `rope`, `mla_qk`, `fused_binary_activation` | 0 | bitwise | none: the recipe's `fused_mla` and `fused_swiglu` overrides bypass them |
| all six (the recipe) | 7.4 | differs (loss) | |


## Step 3: the graph kernels in eager

Sixteen of the loop's 20 kernels run from six autograd.Functions, an FSDP2 copy-in function and a
weight-prep patch (`graph_kernels.py`), loaded from the loop's `kernel_cache/`: 15 unchanged, and `ponk`
without its GEMM (its pack function only). A pattern with several specializations
(G-015's `lfu5` has one per linear shape) is picked by its arguments' shapes and dtypes; a shape without a
promoted kernel runs the stock ops (e.g. the dense layers' w13 gradient, whose `bqhy` specialization was not
promoted). Activation:
- **`override_dsv3.graph_kernel_block`**, a `@override` of `DeepSeekV3TransformerBlock.Config` with one
  flag per Function. Its forward runs the attention, router, routed experts and FFNs itself. Overrides of
  nested nodes cannot be combined, so it also applies the recipe's `fused_mla` and `fused_swiglu` overrides
  to its own attention and SwiGLU configs, and `eager_recipe.py` drops them from `override.imports`.
- **`fsdp_copy_in`**, a flag of the same override: on its first forward the block (an FSDP2 module) installs
  `reduce_scatter_input_fn` with `set_reduce_scatter_input_fn`.
- **`install_expert_weight_prep()`** patches the Dist-MoE weights' FSDP2 extension hook, as for Qwen3.

| Function | Owns (forward) | Kernels (forward / backward) | Lines |
|---|---|---|---:|
| `AttentionInput` | attention_norm, wq_a and wkv_a (MXFP8); returns `x` for the residual | use7 / lfu5 x2, 3ybb (+ handoff) | 55 |
| `Linear` | wq_b (MXFP8) | - / lfu5 | 55 + 12 |
| `MLAKeyValue` | wkv_b (MXFP8), per-head K with RoPE, V view | - / k6pn | 59 |
| `AttentionOutput` | wo (MXFP8), residual add, ffn_norm; returns `h`, the norm output and its rowwise operands | dre3 / a6rv | 46 |
| `Router` | gate (HiMidLo), routing, counters, aux loss and its injection, Dist-MoE's padding masks | gjn6, h4xd / uu33 | 77 + 30 |
| `SwiGLUFeedForward` | shared or dense w13, SwiGLU, w2 (MXFP8) | - / bqhy, 5nj2 (+ handoff) | 85 |
| `reduce_scatter_input_fn` | FSDP2's reduce-scatter copy-in for one block | ponk's pack | 49 + 25 |
| expert weight prep | `_build_operands` of the Dist-MoE w13 / w2 FSDP2 extension | x6t6, 5uzo | 51 |

The block override is 239 lines; the kernel loader, quantization helpers and the handoff about 120. The
MXFP8 linear helpers (`mxfp8_operands`, `linear_meta`, `scaled_mm`, `wgrad`, `wgrad_from_col`) are imported
from the Qwen3 port.

How the regions map:
- **G-047, G-082 (forward + recompute).** `use7` and `dre3`, the recompute kernels, at the 61 forward sites.
  The forward kernels `z47c`, `c2qd` and `5yin` have no use.
- **G-001, G-020, G-006 (router).** The gate GEMM, `gjn6`, the in-place counter and metric adds of the
  stock router, then `h4xd` for the tensors backward reads. `gjn6` also applies Dist-MoE's padding masks to
  the expert ids and scores, so the routed experts are called without a padding mask. Backward: `uu33` and
  the gate's HiMidLo GEMMs (the same code as the Qwen3 router).
- **G-049 (cross-layer).** `AttentionInput`'s backward runs `3ybb` and returns the summed gradient of `x`;
  it stores the gradient's quantized operands keyed by that tensor (`_GRAD_QUANT`, one entry). The previous
  layer's `SwiGLUFeedForward` backward takes them when its incoming gradient is that same tensor, otherwise
  it quantizes as usual. Per step, 60 of 61 lookups hit: the last layer's FFN receives the final norm's
  gradient. Layer 0's `3ybb` quantization goes unused (one call; the graph runs stock ops there).
- **G-015.** `lfu5` in `AttentionInput` (wq_a, wkv_a) and `Linear` (wq_b), `bqhy` in `SwiGLUFeedForward`.
- **G-035.** `MLAKeyValue` owns wkv_b and the fused MLA K/V assembly, so its backward is `k6pn` plus the
  wkv_b GEMMs.
- **G-007.** `SwiGLUFeedForward`'s backward computes the shared-expert w13 WGRAD with `5nj2` (the same
  GEMM, then the rows in FSDP2's dim-1 shard order) and returns it in the parameter's shape;
  `reduce_scatter_input_fn` recognizes it by data pointer and skips FSDP2's chunk + 256-way cat. If such a
  gradient does not reach the copy-in (AccumulateGrad copied it), the function raises instead of reordering
  it a second time.
- **G-003.** The block's attention / shared-expert gradients are, in FSDP2's parameter order, exactly
  `ponk`'s 12 pack sources (`wo`'s gradient being its GEMM output), and FSDP2's buffer has the pack's size
  and layout (`torch._chunk_cat(grads, 0, 256)`). So the copy-in calls `ponk`'s pack (its module's `_fn`,
  which `run` calls after the GEMM) on eager's gradients; eager has already done the GEMM.
- **G-028 is not used.** `czaf` writes each chunk as w2 then w13; FSDP2's routed-expert group is ordered w13
  then w2, so the layouts differ.
- **G-021, G-061.** As for Qwen3, but on the gathered bf16 weight (the expert FSDP mesh has 4 ranks here),
  refilling FSDP's operand tensors through the kernel module's internal function.

**Verification.** `test_functions.py` builds layer 3 (the first MoE layer) and layer 0 (dense) of the eager
recipe's model config with random weights, and compares the module path and the Function path as bytes:
every output and gradient of each Function, the router's counter and metric buffers, the quantized operands
handed to the previous layer, the FFN with and without the handoff (shared and dense), the reduce-scatter
input with and without G-007, and the expert operands built fresh and refilled. All exact once two errors,
neither in numerics, were fixed: the handoff test read a leaf tensor's `.grad`, which is a copy (it now
hooks a non-leaf input), and the router check read `freeze_gate`, a config field, instead of the gate
weight's `requires_grad`.


## Step 4: results

All numbers are from one host: torch 3c53c6f, CUDA 13.0, one GB300, fake PG with NGPU=256. The graph runs
substitute the loop's kernels from its `kernel_cache/`, whose keys hit on this host. Every eager variant
is bitwise identical to stock eager (loss, grad_norm and the load-balance loss at every logged step), and
the graph with kernels to the graph baseline; with `local_compile` on, eager with kernels is bitwise
identical to eager with `local_compile`.

### GPU time per step (profiled step 10, CUDA graphs off)

Means of three rounds of alternating runs on one GPU, with no other run of this study on the host (range
in parentheses); the `local_compile` rows are one run each:

| Run | GPU ms | Saving | Numerics |
|---|---:|---:|---|
| Graph baseline | 581.8 (581.3-582.3) | - | reference (graph) |
| **Graph + 20 kernels** | **503.0** (502.3-503.6) | **78.8** | bitwise = graph baseline |
| Eager, `local_compile` off | 545.8 (544.3-547.5) | - | reference (eager) |
| Eager, recipe's `local_compile` | 536.2 | 9.6 | differs (loss region) |
| **Eager + all Functions (16 kernels)** | **472.4** (472.0-473.2) | **73.4** | bitwise = eager reference |
| Eager + all Functions + `local_compile` | 468.8 | 77.0 | bitwise = eager with `local_compile` |

Eager's saving is 93% of the graph's (73.4 vs 78.8 ms). Counted from kernel call counts instead of GPU time,
it is 92% (76.2 vs 82.8 ms, means of the same three rounds). Eager starts 36.0 ms below the graph and stays
30.6 ms below it with the kernels. The difference is mostly the graph's recompute, which eager avoids by
saving: 205 GiB against the graph's 139 GiB.

Adding one Function at a time (a chain of runs on another GPU, saving counted from kernel call counts, so
the clock noise of step 2 moves each by at most 2%):

| Function (added in this order) | Eager saving | Graph kernels it uses | Their graph saving |
|---|---:|---|---:|
| `AttentionInput` | 2.5 | use7, lfu5 x2, 3ybb | |
| `Linear` (wq_b) | 0.6 | lfu5 | |
| `MLAKeyValue` | 6.2 | k6pn | 5.6 |
| `AttentionOutput` | 1.3 | dre3, a6rv | |
| `Router` | 28.9 | gjn6, h4xd, uu33 | 25.1 + 4.7 (h4xd at the recompute site) |
| `SwiGLUFeedForward` | 3.6 | bqhy (+ the handed-over 3ybb and dre3 operands) | |
| subtotal of `AttentionInput`, `Linear`, `AttentionOutput`, `SwiGLUFeedForward` | 8.0 | use7, dre3, a6rv, 3ybb, lfu5, bqhy | 10.0 |
| expert weight prep | 15.5 | x6t6, 5uzo | 7.8 |
| `fsdp_copy_in` | 17.4 | 5nj2, ponk's pack | 14.7 |
| **all** (stock eager -> all; the rows sum to 76.0) | **76.2** | 16 kernels | **63.2 (+ 4.7)** |

The graph column is the loop's per-kernel saving (its section 7: the verifier's isolated measurement of
kernel against reference, times calls per step), not an in-run difference. Where eager differs:
- **Norm and quantization: 8.0 vs 10.0 ms.** Not broken down further; both trainers run quack's RMSNorm
  kernels in their baselines. `AttentionInput` and `AttentionOutput` look small alone because their
  quantized outputs (the handoff and the w13 operands) are consumed only once `SwiGLUFeedForward` is on.
- **Router: 28.9 vs 25.1 ms.** Eager runs `h4xd` in the forward in place of the stock ops that produce the
  saved tensors; the graph saves that at the recompute site (4.7 ms).
- **Expert weight prep: 15.5 vs 7.8 ms.** The quantization part matches (stock 17.1 ms -> 9.1 ms of kernels,
  8.0 saved); the other 7.4 ms are device copies of eager's stock path into FSDP2's operand tensors, which
  the kernels write directly.
- **FSDP copy-in: 17.4 vs 14.7 ms.** G-007: eager's chunk + 256-way cat costs 12.9 ms (61 cats, 3 left for
  the dense layers, which have no G-007 kernel); `5nj2` removes it for 0.8 ms of extra kernel time. G-003:
  `ponk`'s pack replaces a 13.2 ms `_chunk_cat` with 8.0 ms.
- **PyTorch's `reduce_scatter_input_fn_with_native_copy`** alone saves 13.6 ms with no custom kernel (78% of
  `fsdp_copy_in`'s 17.4): it copies each dim-1-sharded gradient straight into the buffer, so the cat
  disappears. With the loop's kernels for everything else it saves 13.8 ms instead of 17.4.

Not reachable from eager, 12.5 ms of the graph's per-kernel saving:
- `z47c`, `c2qd`, `5yin` (3.7 ms): forward copies of G-047 / G-082 that drop what the recompute
  recomputes. Eager does not recompute, so its forward runs `use7` and `dre3`.
- `h4xd` at its recompute site (4.7 ms): eager has no recompute site; it runs `h4xd` in the forward (above).
- `czaf` (4.1 ms): its output layout (w2 then w13 per chunk) is not FSDP2's (w13 then w2), so eager keeps
  the stock `_chunk_cat` for the routed experts (9.9 ms in the run, against `czaf`'s 5.6 ms in the graph).

With all kernels, eager (472.4 ms) against the graph (503.0 ms), from kernel totals by name:
- graph only: about 34 ms of recomputed MXFP8 GEMMs (six per layer, 366 extra calls), 7 ms of recomputed
  MLA RoPE kernels and 3.5 ms of recomputed quantizations;
- eager only: the routed experts' stock `_chunk_cat` (9.9 ms, against `czaf`'s 5.6 ms in the graph), 8.7 ms
  more device-to-device copies and 6.5 ms more dtype casts (`aten::to`), outside the loop's regions.

### Throughput (20 steps, CUDA graphs on)

Means of all runs on this host (range in parentheses): two serial runs on GPU 0 for the stock and
all-Functions eager rows, one for the graph rows, plus two runs each on GPU 3 from earlier, while the other
GPUs ran this study's profiles. The `local_compile` rows are GPU 3 runs only.

| Run | tokens/s | Gain | Memory GiB |
|---|---:|---:|---:|
| Graph baseline | 6,141 (6,139-6,144) | - | 139.4 |
| **Graph + 20 kernels** | **6,968** (6,966-6,971) | +827 (+13.5%) | 139.2 |
| Eager, `local_compile` off | 6,947 (6,907-6,970) | - | 205.3 |
| Eager, recipe's `local_compile` | 7,037 (one run) | +90 (+1.3%) | 202.1 |
| **Eager + all Functions** | **7,778** (7,669-7,870) | +831 (+12.0%) | 205.5 |
| Eager + all Functions + `local_compile` | 7,796 (7,787-7,805) | +849 (+12.2%) | 202.3 |

## Answer: how much of the graph's optimization can eager achieve?

- **93% of the GPU-time saving, bitwise** (73.4 vs 78.8 ms/step), with 16 of the loop's 20 kernels.
  - Throughput gains as many tokens/s as the graph (+831 vs +827; eager's runs spread over 200 tokens/s).
  - Numerics are bitwise identical to stock eager.
  - In absolute terms eager with the kernels is faster than the graph with them (472.4 vs 503.0 ms,
    7,778 vs 6,968 tokens/s) because it saves instead of recomputing, using 66 GiB more memory. The graph
    with all 20 kernels reaches stock eager's throughput (6,968 vs 6,947 tokens/s).
- **The four kernels eager cannot use are artifacts of the graph's setup, not missing fusions.** Three exist
  because the graph recomputes (the forward copies of the norm regions, `h4xd` at its recompute site); eager
  gets the same work done with the recompute kernels in its forward. One (`czaf`) is written for
  GraphTrainer's bucket layout; FSDP2's routed-expert group is ordered differently, and the stock copy-in
  stays (9.9 ms against `czaf`'s 5.6 ms in the graph).
- **Regions that cross module, layer and FSDP boundaries were all reachable.**
  - Cross-module (59% of the saving): Functions owning several modules' forward, as for Qwen3.
  - Cross-layer (3%): a handoff from layer L's backward to layer L-1's, keyed by the gradient tensor; 60 of
    61 lookups hit per step.
  - FSDP glue (35%): FSDP2's experimental `set_reduce_scatter_input_fn` hook and torchtitan's Dist-MoE
    weight hook. Eager's FSDP2 does more glue work than the graph's bucketing, so the same kernels save
    more in eager (32.9 ms, against 22.5 ms in the graph).
- **Eager already had little of it.** The recipe's `local_compile` regions save 9.6 ms (12% of the graph's
  saving; 9.1 ms counted from kernel calls, against 7.4 ms in step 2's separate runs), and not bitwise. The
  recipe's fused MLA and SwiGLU overrides already cover what Inductor compiled for Qwen3 (RoPE), so only
  the aux-loss statistics, the loss and the gate-gradient split remain. On top of the kernels,
  `local_compile` saves another 3.6 ms.

### How hard was it

Given the loop's output and the Qwen3 port, about 50 minutes. The request came at 11:54 (2026-11-09); the
recipes and reading the eager DSv3 block, MLA, router and FSDP2 code took until 12:20; `graph_kernels.py`,
the block override and the unit tests were written 12:20-12:24; the first full-training run with all block
Functions was bitwise at 12:30; the FSDP2 copy-in hook was added and tested by 12:43. No Function needed
a numerics fix (step 3).

What made DSv3 harder than Qwen3:
- **Recompute.** The graph's forward and recompute sites need different kernels, and eager needs the
  recompute kernel's contract in its forward. Picking the right one needs the memory policy's view of the
  graph, which the loop's plan records.
- **A cross-layer region.** Autograd gives no way to pass extra tensors between layers' backward, so the
  handoff is a side channel keyed by tensor identity. It works because the engine passes the same gradient
  tensor through the residual and MoE adds; any op in between (a cast, a hook that replaces the gradient)
  silently turns it into a miss, which is safe (the FFN quantizes itself) but slower.
- **FSDP glue.** Reachable only through an experimental FSDP2 hook, and only where the kernel's layout matches
  FSDP2's (G-003 yes, G-028 no). G-007's kernel reorders the gradient in the FFN's backward, so the copy-in
  must recognize the pre-reordered tensor (by data pointer) and must not reorder it again.
- **The block override replaces the whole block's forward.** The Functions cross the attention, router,
  routed-experts and FFN modules, and overrides of nested config nodes cannot be combined, so the override
  also re-applies the recipe's fused MLA and SwiGLU overrides.

### Caveats

- One model, one layout, fake PG on one GPU. Collectives are emulated, so communication overlap is not
  measured; FSDP glue in particular could overlap with communication on a real cluster.
- Eager and the graph differ numerically from step 1 (step 2), so the two baselines are not the same
  training run; each variant is compared with its own baseline.
- Eager uses 66 GiB more memory than the graph here. With activation remat on, eager would recompute too,
  and the forward / recompute kernel split would apply to it as well.
- The port hard-codes the recipe's layout (FSDP without TP/CP, fused MLA with q LoRA, group-limited sigmoid
  routing with a padding mask, three router pieces, no remat) and raises otherwise.


## Step 5: what carried over from the Qwen3 port

The Qwen3 study asked whether its eager port holds for a sibling model (Qwen3 235B-A22B: mostly yes, the
Functions stay and only the kernels change). DSv3 is a different architecture, so the question here is
which parts of the Qwen3 port it reused.

| Part | Qwen3 30B-A3B port | DSv3 port |
|---|---|---|
| MXFP8 linear helpers (`mxfp8_operands`, `linear_meta`, `scaled_mm`, `wgrad`) | written for F1/F2 | imported unchanged (`wgrad_from_col` split out of `wgrad` for linears that save MXFP8 inputs) |
| Router backward (HiMidLo gate GEMMs on the kernel's bf16 pieces) | F3 | same code |
| Router forward | softmax, round-robin ids, batch-wise aux loss with an all-reduce | new: sigmoid, group-limited top-k, microbatch-wise aux loss, the recompute kernel run in the forward |
| Expert weight prep | patches `_build_operands` (fp32 shard, size-1 expert mesh) | same patch point and refill pattern, on the gathered bf16 weight |
| Attention | F1: fused QKV, q/k norm, RoPE; F2: wo, add, ffn_norm | new: `AttentionInput` (two projections), `Linear`, `MLAKeyValue`, `AttentionOutput` (adds the w13 quantization) |
| FFN | none (no shared experts) | new: `SwiGLUFeedForward` |
| Cross-layer handoff, FSDP2 copy-in | none needed | new |
| Kernel loader, block override, unit-test harness | one specialization per pattern | loader extended to several specializations per pattern; same override and test structure |

The DSv3 port is 956 lines (`graph_kernels.py` 717, `override_dsv3.py` 239). It imports 80 lines of
helpers from the Qwen3 port, and about 100 more are adapted copies (router backward, weight-prep patch,
kernel loader, the block override's skeleton). What carried over in full is the method:
the Function boundary rules, pass-through outputs for engine adds, saving exactly the kernel contract's
tensors, and the bitwise module-vs-Function tests. The Functions themselves follow each model's block, and
the regions the graph finds follow each model's graph (here including recompute and FSDP layout), so a new
architecture needs new Functions.


## This directory

| File | Content |
|---|---|
| `README.md` | this report |
| `results.json` | every run: settings, GPU time, tokens/s, memory, full-precision TensorBoard scalars, bitwise flag, kernel call counts |
| `classify.json` | step 1: per-region sites, phases, layers and modules (from `../qwen3_30b_eager_study/classify.py`) |
| `graph_kernels.py` | the six Functions, the FSDP2 copy-in function, the expert weight-prep patch and the kernel loader |
| `override_dsv3.py` | the `@override` of the DSv3 block that puts them in the model |
| `test_functions.py` | bitwise unit tests against the stock eager modules and FSDP2's default copy-in |
| `eager_recipe.py`, `run.sh` | eager runs |
| `graph_recipe.py`, `run_graph.sh` | GraphTrainer runs without and with the loop's plan and kernel cache |
| `account.py` | GPU time saved between traces, counted from kernel call counts (per-Function savings) |

GPU time and scalars use the Qwen3 study's `busy.py` and `tbdump.py`. The run directories (logs, TensorBoard,
traces) are not committed (`.gitignore`).

## Reproduce

From the repo root, with the loop's kernel cache in place (`graph_kernels.py` loads the kernels from
`../dsv3_671b_distmoe_mxfp8/kernel_cache/`):

```bash
D=torchtitan/experiments/graph_trainer/agentic_compiler/dsv3_671b_eager_study
Q=torchtitan/experiments/graph_trainer/agentic_compiler/qwen3_30b_eager_study
# Bitwise unit tests (one GPU; optionally one test by name, e.g. router)
PYTHONPATH=$D:$D/..:. python $D/test_functions.py
# Eager: stock (local_compile off), then all Functions; GK takes a comma-separated subset
$D/run.sh 0 eager_nolc_prof $D/runs/eager_nolc_prof
GK=all $D/run.sh 0 eager_kernels_prof $D/runs/gk_all_prof
GK=all $D/run.sh 0 eager_kernels_tps $D/runs/gk_all_tps
# PyTorch's experimental FSDP2 copy-in instead of fsdp_copy_in
GK=native_copy_in $D/run.sh 0 eager_kernels_prof $D/runs/native_copy_in_prof
# Eager with the recipe's local_compile regions (or some: LC_REGIONS=moe_aux_loss)
$D/run.sh 0 eager_lc_prof $D/runs/eager_lc_prof
GK=all LOCAL_COMPILE=1 $D/run.sh 0 eager_kernels_prof $D/runs/gk_all_lc_prof
# GraphTrainer, without and with the loop's kernels
$D/run_graph.sh 0 graph_base_prof $D/runs/graph_base_prof
$D/run_graph.sh 0 graph_pack_prof $D/runs/graph_pack_prof
# GPU time and scalars
python $Q/busy.py $D/runs/gk_all_prof/profile/iteration_10/rank0_trace.json.gz
python $Q/tbdump.py $D/runs/eager_nolc_prof $D/runs/gk_all_prof
# Saving counted from kernel call counts, with the 10 largest contributions
python $D/account.py $D/runs/{eager_nolc_prof,gk_all_prof}/profile/iteration_10/rank0_trace.json.gz --top 10
```
