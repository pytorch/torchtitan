# Graph-found optimizations in eager: Qwen3 30B-A3B

The question: optimizations can be applied in both eager and graph mode; what fraction of the
optimizations the agentic compiler found in the graph can eager mode achieve?

Subject: the Qwen3 30B-A3B DistMoE MXFP8 loop (`../qwen3_30b_a3b_distmoe_mxfp8/`): 15 verified CuTe DSL
kernels for 15 lifted regions (G-24 was lifted but got no kernel). In that loop's measurement setup (fake PG, one 32K-token microbatch per step, FA4,
deterministic), GPU time per step excluding emulated collectives went 967.9 -> 679.5 ms (-288.4 ms). The
per-kernel savings add up to 281.8 ms/step.

## Step 1: classification of the patterns

### Method

`classify.py` and `sites.py` (this directory) replay the loop's final plan
(`../qwen3_30b_a3b_distmoe_mxfp8/plans/plan_iter4.txt`) on the graph the lift passes see. The graph comes
from the reference dump of the matcher-robustness experiment (item 1 of `../TODO.md`), which is the same
recipe and layout. That dump is a local workspace file and is not committed; `classify.json` holds the
per-region output. For every site the scripts record:
- **Phase:** the node's `autograd_backward` flag. The recipe recomputes nothing, so there are no
  `MUST_RECOMPUTE` nodes.
- **Layer and module:** `custom.module_fqn`. For backward nodes, the forward node with the same
  `seq_nr` is used.
- **Source line:** from the node's stack trace.
- **What feeds and consumes the region**, with each neighbour's layer relative to the site's layer.

Every region has the expected site count: 48, one per layer; 8 for the loss, one per chunk. The
results:

- **No region crosses a layer.** Every site's members come from one layer. The edges to the
  neighbouring layer are the residual stream: the input `x` and, in backward, its gradient.
- **No region mixes forward and backward.** The finder joins nodes only within one phase.
- **Backward members without a module are autograd-engine accumulation adds.** They are
  `add.Tensor` nodes with seq_nr 6595-6613, far past the forward ops (seq_nr < 840). The engine
  inserts one wherever a forward tensor has two consumers and their gradients must be summed:
  - the residual stream `x` (attention_norm and the residual add);
  - `h` (ffn_norm and the trailing residual add);
  - the ffn_norm output (router gate and routed experts);
  - the RoPE input (`x * cos` and `rotate_half(x) * sin`);
  - the router scores (top-k gather, aux loss).

  No module's forward code contains these adds.

### Classification

Saving = (reference - kernel) x calls, from the loop's verifier measurements (README section 7).

| Region | What | Phase | Scope | Engine adds | Eager today | Saving ms/step |
|---|---|---|---|---:|---|---:|
| G-01 | RoPE bwd q/k, q/k RMSNorm bwd, cat with dv, MXFP8 quant of the wqkv grad | B | cross-module: qkv split, q_norm, k_norm, rope, wqkv MXFP8 bwd | 4 | partial: rope bwd in `local_compile("rope")` | 110.2 |
| G-02 | wqkv output split + contiguous copies, q/k RMSNorm, RoPE with position gather | F | cross-module: qkv split, q_norm, k_norm, rope | - | partial: rope fwd in `local_compile("rope")` | 100.6 |
| G-06 | router + aux-loss bwd, incl. injected aux-loss grad and the gate's grad split | B | cross-module: router, aux_loss, gate (HiMidLo bwd), routed_experts masked_fill | 3 | partial: grad split in `local_compile("fp32_to_bf16_split")` | 17.7 |
| G-07 | loss bwd: nll + log_softmax bwd, fp32 -> bf16 | B | loss function | 0 | yes: `local_compile("loss")` | 16.3 |
| G-90 | loss fwd per chunk: cast, log_softmax, nll, div | F | loss function | - | yes: `local_compile("loss")` | 12.0 |
| G-17 | router: routing map, per-expert counts (int64 + fp32), aux-loss counts | F | cross-module: router, aux_loss | - | none | 5.9 |
| G-19 | sum of MoE-input grads, ffn_norm bwd, residual add, MXFP8 quant of the wo grad | B | cross-module: ffn_norm, residual, wo MXFP8 bwd | 2 | none | 5.4 |
| G-91 | aux-loss fwd: L1-normalize scores, masked column sum | F | aux_loss (one function) | - | none: `moe_aux_loss` does not cover the batch-wise loss | 3.4 |
| G-15 | wqkv dgrad GEMM (unchanged) + attention_norm bwd + residual add | B | cross-module: attention_norm, residual, wqkv MXFP8 bwd | 1 | none | 3.1 |
| G-33 | w13 fp32 -> bf16 cast + `prepare_mxfp8_weight` | F | FSDP parameter path | - | none (same 2 kernels) | 2.1 |
| G-42 | w2, same | F | FSDP parameter path | - | none (same 2 kernels) | 1.2 |
| G-41 | attention_norm RMSNorm + MXFP8 quant of wqkv input | F | cross-module: attention_norm, wqkv MXFP8 fwd | - | none | 1.2 |
| G-45 | residual add + ffn_norm RMSNorm | F | cross-module: residual, ffn_norm | - | none | 1.0 |
| G-28 | wo MXFP8 quant + rearrange, then the unchanged GEMM | F | one autograd.Function (MXFP8 linear fwd) | - | available, unused: torchao's fused CuTe DSL quant + swizzle (TODO in `quantization/mxfp8/linear.py`) | 0.9 |
| G-47 | router softmax, top-k gather, normalize, scale | F | router (one function) | - | none | 0.8 |

By scope (graph saving ms/step, of 281.8):

| Scope | Regions | Saving | Share |
|---|---|---:|---:|
| Cross-layer | none | 0 | 0% |
| Cross-module within a layer | G-01, G-02, G-06, G-17, G-19, G-15, G-41, G-45 | 245.1 | 87% |
| One function / module / autograd.Function | G-07, G-90, G-91, G-28, G-47 | 33.4 | 12% |
| FSDP parameter path | G-33, G-42 | 3.3 | 1% |

By phase: 5 backward regions save 152.7 ms (54%), and 10 forward regions save 129.1 ms (46%).
All four cross-module backward regions (G-01, G-06, G-19, G-15: 136.4 ms, 48%) contain
autograd-engine adds.

### What makes a region hard in eager

Module boundaries are not the obstacle: a forward region that spans modules can be written as one
function. Autograd's boundaries are:

1. **A forward kernel needs a backward.** Replacing forward ops with a kernel removes them from
   autograd. An `autograd.Function` has to provide their backward, either the matching backward
   kernel or the original ops. The graph optimizes forward and backward regions independently, and
   their boundaries do not line up. For example, G-02 covers the qkv split, q/k norm and RoPE, while
   its backward partner G-01 adds the wqkv grad quantization from inside the MXFP8 linear's
   backward.
2. **A backward region must belong to one Function.** It has to own every forward op whose backward
   the region contains. G-01 contains backward ops of four modules plus torchtitan's
   `_MXFP8LinearFunction`, so one Function must span wqkv through RoPE.
3. **Engine adds belong to no Function.** To fuse one, the Function returns the shared tensor once
   per consumer, so that its backward receives the gradients separately and sums them in the kernel.
   Example: an add + RMSNorm Function returns `h` for the residual and the normalized `h` once each
   for router and experts.
4. **Saved tensors are the kernel contract.** A backward kernel reads exactly the forward tensors
   the graph saved (normalized q/k, rstd, cos/sin slices, ...). In eager, the Function's forward must
   save those same tensors.

Applied to this plan, the 15 kernels fit five Functions per layer or step. Each covers a stretch
between GEMMs, attention and collectives:

| Function | Forward | Backward | Regions | Saving ms/step |
|---|---|---|---|---:|
| F1 attention input | attention_norm -> wqkv (MXFP8) -> split -> q/k norm -> RoPE; returns q, k, v and `x` for the residual | G-01 -> wqkv wgrad GEMM -> G-15 | G-41, G-02, G-01, G-15 | 215.1 |
| F2 attention output | wo (MXFP8) -> residual add -> ffn_norm; returns `h` and the normalized `h` twice | G-19 -> wo dgrad/wgrad GEMMs | G-28, G-45, G-19 | 7.3 |
| F3 router | gate -> scores -> top-k gather/normalize, counts, aux-loss stats and injection | G-06 -> gate GEMMs | G-47, G-17, G-91, G-06 | 27.8 |
| F4 loss chunk | log_softmax + nll | G-07 | G-90, G-07 | 28.3 |
| F5 expert weight prep | fp32 -> MXFP8 operands in torchtitan's FSDP2 extension (`fsdp_post_all_gather`) | none (operands carry no grad) | G-33, G-42 | 3.3 |

Ease of porting:
- **Contained in one place:** F4 and F5, plus G-28 and G-17 inside F2 and F3. The loss is already
  one `local_compile` function. G-28 sits inside one Function's forward. G-17 has no gradient.
- **Rewrites spanning modules, with pass-through outputs:** F1, F2 and F3 (250.2 ms, 89% of the
  saving). They replace the forward of the attention block and the router.

### Overlap with what eager already does

- **Existing eager optimizations that cover graph regions:**
  - `local_compile("loss")` compiles exactly G-90 + G-07 (28.3 ms, 10%) with Inductor.
  - `local_compile("rope")` compiles the RoPE part of G-02 and G-01.
  - `local_compile("fp32_to_bf16_split")` compiles the grad split inside G-06.

  GraphTrainer requires `local_compile.regions = []`, so the graph baseline runs all of these as
  plain aten ops. Part of each graph saving is therefore already available in eager.
- **Available but unused in eager:** torchao's fused CuTe DSL MXFP8 quantize + swizzle. It would
  cover the quantization in G-28, G-41, G-19 and G-01, and in G-24, which got no kernel in the loop.
  `quantization/mxfp8/linear.py` measured it 2.4-3.2x faster and bitwise identical, but has not
  switched to it.
- **Named but ineffective:** the eager recipe `qwen3_30b_a3b_dist_moe_mxfp8_8gpu` appends
  `moe_aux_loss`, and its docstring says this compiles the glue GraphTrainer captures. That region
  only wraps `MicrobatchWiseLoadBalanceLoss`'s helper. Qwen3 uses `aux_loss_type="batch_wise"`, so
  G-91 and G-17 run uncompiled in eager.
- **Eager fused kernels that leave no region to find:** FA4 attention, Dist-MoE's grouped GEMMs
  with fused SwiGLU, aten's single-kernel RMSNorm forward and backward, and the MXFP8 `scaled_mm`.
  The lifted regions are the glue between these.

Steps 2-4 measure how much of each overlapping region eager's Inductor regions already save.

## Step 2: eager in the loop's setup

`eager_recipe.py` (this directory) is the eager twin of the graph recipe,
`qwen3_30b_a3b_dist_moe_mxfp8_8gpu`, with the loop's changes: one 32K-token microbatch per step, FA4 varlen
attention, deterministic, seed 42, fake PG with NGPU=8 on one GB300. Two measurements:
- **GPU time per step** (`busy.py`): the union of kernel intervals in profiled step 10, with accelerator graphs
  off. Kernels launched under a collective CPU op (the fake PG's emulated collectives, about 7 ms) are excluded.
- **Throughput:** tokens/s at step 20, with accelerator (CUDA) graphs on as the recipe sets them, no profiler.

**Numerics.** With `local_compile` off, eager runs the same aten ops GraphTrainer traces. Step 1 is bitwise
identical to the graph baseline: loss 12.458515167236328, grad_norm 68.36530303955078. From step 2 on the two
differ (step-10 loss 5.717879 eager, 5.717247 graph), so the difference is outside forward and backward
(optimizer or gradient reduction path). Eager variants are therefore checked against eager with
`local_compile` off.

| Eager | GPU ms/step | tokens/s | Numerics vs `local_compile` off |
|---|---:|---:|---|
| `local_compile` off | 984.9 | 33,991 | reference |
| recipe's `local_compile` regions (loss, rope, fp32_to_bf16_split, ...) | 768.1 | 43,552 | differ from step 1 |

Each `local_compile` region on its own (GPU ms saved against `local_compile` off):

| Region | Saving | Bitwise? |
|---|---:|---|
| `rope` | 187.1 | no |
| `loss` | 27.9 | no |
| `fp32_to_bf16_split` | 2.3 | yes |

`moe_aux_loss` and `fused_binary_activation` cover nothing in this recipe. The batch-wise aux loss is not
the microbatch-wise one the region wraps, and Dist-MoE fuses SwiGLU.

## Step 3: the graph kernels in eager

Each of the 15 verified kernels runs unchanged (the same files, loaded from the loop's `kernel_cache/`) from
one of five eager autograd.Functions (`graph_kernels.py`). Three activation paths:
- **`override_qwen3.graph_kernel_block`**, a `@override` of `Qwen3TransformerBlock.Config`. Its forward calls
  F1, F2 and F3, and hands the ffn_norm output to the router and the routed experts as separate tensors.
  FSDP2 wraps whole blocks, so calling the MoE's children directly keeps their parameters unsharded.
- **`override_loss.graph_kernel_loss`**, a `@override` of `CrossEntropyLoss.Config` for F4.
- **`install_expert_weight_prep()`** for F5, which patches the Dist-MoE weight's FSDP2 extension hooks;
  torchtitan has no config node there.

| Function | Owns (forward) | Kernels | Pass-through for engine adds | Lines |
|---|---|---|---|---:|
| F1 attention input | attention_norm, wqkv (MXFP8), split, q/k norm, RoPE | G-41, G-02 / G-01, G-15 | `x` for the residual | 29 |
| F2 attention output | wo (MXFP8), residual add, ffn_norm | G-28, G-45 / G-19 | `h`, normalized `h` twice | 21 |
| F3 router | gate (HiMidLo), softmax, top-k, counts, aux loss + injection and side effects | G-47, G-17, G-91 / G-06 | none (the Function owns every consumer) | 50 + 13 |
| F4 loss chunk | cast, log_softmax, nll, div | G-90 / G-07 | none | 12 |
| F5 expert weights | fp32 -> MXFP8 operands in the FSDP2 hooks | G-33, G-42 | none | 29 + 7 |

The block override is 74 lines, the loss override 21. The shared helpers (kernel loader, MXFP8 GEMM calls
and WGRAD, copied from `_MXFP8LinearFunction`) are about 60 lines.

**Verification.** `test_functions.py` builds the real eager modules of layer 0 from the recipe's model
config, with random weights. It runs the module path and the Function path on the same inputs and upstream
gradients, and compares every output and gradient as bytes:
- F1: q, k, v and 5 gradients.
- F2: h, normalized h and 4 gradients.
- F3: scores, expert ids, 2 gradients, plus the usage counter, cumulative counts and metric buffers.
- F4: the loss and its gradient.
- F5: all six operand tensors, both freshly built and refilled into FSDP's tensors.

All are exact, each on the first run once the test used the training dtypes. The full training runs are
also bitwise identical to eager with `local_compile` off: loss and grad_norm at steps 1, 10 and 20, for every
combination below.

**What porting took.** Most of the work was reading the eager code to find the forward ops each backward
region depends on. The kernel contracts (`reference.py` and the spec) say exactly what each Function must
save and return. Some details mattered:
- argument orders such as G-01's sin/cos and dk/dq;
- the MXFP8 linear's operands, WGRAD dtype and in-place accumulation;
- the HiMidLo backward layout;
- the router's side effects (usage counter, count all-reduce, metric);
- the refill contract of FSDP's operand tensors.

**Limitations.** The ports reproduce what the loop measured, not a general implementation:
- the kernels are specialized to 32K-token microbatches and this model's sizes;
- the block and router support only the recipe's layout: no TP/CP, no shared experts, round-robin routing,
  a padding mask, no activation remat;
- F5 applies only when the expert FSDP mesh has size 1.

A real-routing router would need `topk` before G-47, which takes the expert ids as input.

## Step 4: results

All numbers are on one host (torch 3c53c6f, CUDA 13.0, one GB300, fake PG with NGPU=8). The graph runs use
the published 30B pack's kernels. On this torch build neither of the pack's cache keys matches: `l2` comes
from AOTAutograd's key, and `l1` hashes torchao's op sources. So `graph_recipe.py` lifts the pack's regions
and substitutes by region id, with the pack's exact call-count check: 640 calls to 15 kernels, bitwise
identical to the graph baseline at steps 1, 10 and 20.

### GPU time per step (profiled step 10, CUDA graphs off)

| Run | GPU ms | Saving | Numerics |
|---|---:|---:|---|
| Graph baseline | 965.8 | - | reference (graph) |
| **Graph + 15 kernels** | **679.1** | **286.7** | bitwise = graph baseline |
| Eager, `local_compile` off | 984.9 | - | reference (eager); step 1 = graph |
| Eager, recipe's `local_compile` (Inductor) | 768.1 | 216.8 | differs |
| Eager + F1 | 763.4 | 221.5 | bitwise = eager reference |
| Eager + F1, F2 | 759.8 | 225.1 | bitwise |
| Eager + F1, F2, F3 | 731.4 | 253.5 | bitwise |
| Eager + F1-F4 | 706.9 | 278.0 | bitwise |
| **Eager + F1-F5 (all 15 kernels)** | **694.8** | **290.1** | bitwise |
| Eager + F1, F2, F3, F5 + Inductor loss | 693.6 | 291.3 | differs (loss region) |

Per Function, the eager saving matches the graph's per-kernel saving:

| Function | Eager saving | Graph per-kernel saving |
|---|---:|---:|
| F1 | 221.5 | 215.1 |
| F2 | 3.6 | 7.3 |
| F3 | 28.4 | 27.8 |
| F4 | 24.5 | 28.3 |
| F5 | 12.1 | 3.3 |

F5 saves more in eager because eager prepares the expert weights twice per step: FSDP2 reshards the
block after forward, and the backward unshard runs the quantizer again (192 calls instead of 96). F5 also
removes eager's separate fp32 -> bf16 cast of the expert weights.

### Throughput (20 steps, CUDA graphs on)

| Run | tokens/s | Gain | Memory GiB |
|---|---:|---:|---:|
| Graph baseline | 34,702 | - | 236.0 |
| **Graph + 15 kernels** | **49,214** | +14,512 (+41.8%) | 231.3 |
| Eager, `local_compile` off | 33,991 | - | 250.3 |
| Eager, recipe's `local_compile` | 43,552 | +9,561 (+28.1%) | 240.3 |
| Eager + F1, F2 | 44,202 | +10,211 | 249.5 |
| Eager + F1-F4 | 47,676 | +13,685 | 244.8 |
| **Eager + F1-F5** | **48,223** | +14,232 (+41.9%) | 244.8 |

## Answer: how much of the graph's optimization can eager achieve?

- **All of it, for this model: 15 of 15 kernels and 101% of the GPU-time saving** (290.1 vs 286.7 ms/step).
  - Throughput gains 98% as many tokens/s as the graph (+14,232 vs +14,512).
  - Numerics are bitwise identical to stock eager, as the graph's are to the graph baseline.
  - No region needed the whole-step graph: none crosses a layer, and each fits one autograd.Function
    between GEMMs, attention and collectives.
- **Eager already had about three quarters of it, but not bitwise.**
  - The recipe's `local_compile` regions save 216.8 ms (76% of the graph's saving) and +9,561 tokens/s
    (66% of the gain).
  - Nearly all of that is the Inductor-compiled RoPE (187 ms of F1's 221 ms) and the loss (28 ms).
  - Both change numerics.
  - The loss is the one region where Inductor matches or beats the graph's kernel. With Inductor's loss
    instead of F4, eager reaches 693.6 ms, but loses bitwise equality.
- **The gap left after optimization is eager's own overhead, not missing fusions.** Eager stays about 16-19
  ms/step behind the graph before and after the kernels (984.9 vs 965.8, 694.8 vs 679.1). With all kernels
  the difference is:
  - device memcpys: 14.5 vs 8.9 ms;
  - copy/cat kernels (FSDP2's copy-in/copy-out vs the graph's bucketing): 11.4 vs 6.5 ms;
  - other kernels: 5.8 ms.

  Memsets are the same (25.4 vs 25.1 ms).

### How hard was it

Given the graph loop's output, porting to eager was mechanical. All five Functions, with bitwise unit tests,
took about 18 minutes (10:12-10:30 on 2026-11-09), including reading the eager code. The first full-training
run with F1 + F2 was bitwise on its first attempt. The loop supplied what eager has no tool for:
- **where to cut:** the regions and their boundaries;
- **verified kernels with exact contracts:** `reference.py` and the spec say what each Function saves and
  returns;
- **a bitwise reference.**

What eager makes harder:
- **Restructured model code instead of a pass.** F1-F3 replace the stock forward of the attention block,
  the MoE and the router with Functions that span several modules. Pass-through outputs are needed to fuse
  the autograd engine's accumulation adds. Side effects must be reproduced by hand (router usage counter,
  count all-reduce, metric).
- **Brittle against model changes.** The graph lifts regions by pattern matching on the traced graph and
  falls back to aten when a pattern changes. The eager ports hard-code the recipe's layout and raise
  otherwise (no TP/CP, no shared experts, round-robin routing, a padding mask, no remat, 32K-token
  microbatches).
- **One port needed a patch.** F5 patches torchtitan's FSDP2 extension hooks, since that layer has no
  override point. It only works when the expert FSDP mesh has size 1, where the graph's region exists too.

### Caveats

- One model, one layout, fake PG on one GPU. Collectives are emulated, so communication overlap, which could
  hide or expose the glue on a real cluster, is not measured. This holds equally for graph and eager.
- The graph's kernels were written for the graph's region boundaries. An eager-first effort could pick other
  boundaries; this study only asks whether the graph's choices are reachable from eager.
- A real top-k router computes the expert ids from the scores, so G-47, which takes the ids as input, would
  need a different region.

### This directory

| File | Content |
|---|---|
| `README.md` | this report |
| `results.json` | every run: settings, GPU time, tokens/s, memory, full-precision TensorBoard scalars, bitwise flag |
| `graph_kernels.py` | the five eager Functions and the kernel loader |
| `override_qwen3.py`, `override_loss.py` | the `@override`s that put them in the model |
| `test_functions.py` | bitwise unit tests against the stock eager modules |
| `eager_recipe.py`, `run.sh` | eager runs |
| `graph_recipe.py`, `run_graph.sh` | GraphTrainer runs with the 30B pack's kernels on this host |
| `busy.py`, `tbdump.py` | GPU time from a profiler trace; TensorBoard scalars |
| `classify.py`, `sites.py`, `classify.json` | step 1 |

The run directories (logs, TensorBoard, traces) are not committed (`runs/` is git-ignored).

### Reproduce

From the repo root, with the 30B loop's kernel cache in place (`graph_kernels.py` loads the kernels from
`../qwen3_30b_a3b_distmoe_mxfp8/kernel_cache/`):

```bash
cd torchtitan/experiments/graph_trainer/agentic_compiler/qwen3_30b_eager_study
# Bitwise unit tests (one GPU)
PYTHONPATH=.:../../../../.. python test_functions.py
# Eager: stock (local_compile off), then all five Functions
./run.sh 0 eager_kernels_prof runs/eager_nolc_prof
GK=attention_input,attention_output,router,loss,expert_weight_prep ./run.sh 0 eager_kernels_prof runs/gk_all5_nolc_prof
GK=attention_input,attention_output,router,loss,expert_weight_prep ./run.sh 0 eager_kernels_tps runs/gk_all5_nolc_tps
# Eager with the recipe's local_compile regions (or one region: LC_REGIONS=rope)
LOCAL_COMPILE=1 ./run.sh 0 eager_kernels_prof runs/eager_lc_prof
# GraphTrainer, without and with the pack's kernels
PACKS=<graph-trainer-kernel-packs checkout> ./run_graph.sh 0 graph_pack_prof runs/graph_pack_prof
# GPU time and scalars
python busy.py runs/gk_all5_nolc_prof/profile/iteration_10/rank0_trace.json.gz
python tbdump.py runs/eager_nolc_prof runs/gk_all5_nolc_prof
```
