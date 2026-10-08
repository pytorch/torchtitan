# Rewrite patterns

Each pattern: what blocks Inductor, the rewrite, and what we measured (GB300, fwd+bwd unless noted). Numbers are examples from DSv4, Kimi K3, DSv3 and Qwen3.5-MoE; the patterns are general.

| # | Pattern | Typical win |
|---|---|---|
| 1 | complex ops -> real arithmetic | RoPE 443.7 -> 199.0 µs |
| 2 | cat / stack lowering per region | 2155 -> 781 µs (or the reverse) |
| 3 | tiny bmm -> explicit sum | 3475 -> 2155 µs |
| 4 | small reductions: unroll | 1354 -> 290 µs |
| 5 | keep small dims static; unbind in region | SwiGLU 1.68-1.75x |
| 6 | gather-sum instead of scatter | combine 13540 -> 1102 µs |
| 7 | scale the narrower tensor in a region | MoE module -12.8% |
| 8 | int16 sort keys | 72.9 -> 30.4 µs |
| 9 | per-entry unroll of tiny loops | 309 -> ~100 µs |
| 10 | router glue as one region | 356.7 -> 135.9 µs |
| 11 | data-dependent shapes, fakes, host syncs | correctness, syncs |
| 12 | graph budget: moved to measurement.md | |
| 13 | fp32 GEMMs, exact bf16 splits | 3262 -> 1291 µs; 0.288 -> 0.065 ms |
| 14 | custom ops with exact fakes | SAC block 302.5 -> 122.6 ms |
| 15 | loss path | -13% / -18% |
| 16 | `autograd.Function` with fwd + bwd regions | 24.3 ms -> 1.56 ms per call |
| 17 | small symbolic reduction -> GEMM | 1435 -> 180 µs |
| 18 | input grad in the next kernel's layout | -3.8% (Triton attention) |
| 19 | per-expert bias: gather + one-hot GEMM grad | GPT-OSS -9.7% |
| 20 | eager hooks around attention as a region | GPT-OSS -3.9% |

## 1. Complex ops -> real arithmetic

Inductor has no codegen for complex multiply. It keeps the ATen complex kernel and cannot fuse the split/cat glue around it ("Torchinductor does not support code generation for complex operators", printed only on a cold compile).

```python
# before: complex RoPE
x_c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
out = torch.view_as_real(x_c * freqs_cis).flatten(-2)

# after (compile branch): same math on the real view of the cache
cos, sin = torch.view_as_real(freqs_cis).unbind(-1)   # [..., D/2] each
x_even, x_odd = x.float().unflatten(-1, (-1, 2)).unbind(-1)
out = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1).flatten(-2)
```

DSv4 RoPE: compiled complex 443.7 µs (7 kernels) -> real arithmetic 199.0 µs (2 kernels). Compiled real arithmetic is not bitwise vs eager complex (FMA, see gating file). At strided call sites (q_pe sliced from a wider q) the real form's backward got slower than compiled complex (DSv3 671B: 1145 vs 1003 µs); a per-half cast plus the ConcatKernel option (pattern 2) brought it to 457 µs. Measure the region as written first: once the surrounding cat/split fuse, a complex op left inside the region (ATen fallback) can tie the real-arithmetic rewrite (seen on DSv3).

## 2. Cat / stack lowering

Inductor lowers a cat either as one masked pointwise kernel (per-element index math; fuses into consumers) or as per-input copies (`ConcatKernel`). Neither is right everywhere.

- Inverse RoPE on `[16384, 64, 512]`: default 2155 µs, ConcatKernel 781 µs (ideal 605).
- Norm + RoPE: default 1026 µs, ConcatKernel 1502 µs (the default lets the cat fuse into the norm's reduction).

Neither lowering won everywhere for us, so it's worth trying both for the region in question; the option is region-scoped:

```python
@local_compile("inverse_rope", batch_invariant=True,
               options={"max_pointwise_cat_inputs": 0, "max_complex_pointwise_cat_inputs": 0})
def inverse_rope(...): ...
```

A masked cat or stack consumed by a persistent reduction re-loads the row once per branch; avoid stacking inside a reduction, or split the reduction.

## 3. Tiny batched matmul -> explicit sum

`[T, 1, 8] @ [T, 8, 7168]` goes to cuBLAS GEMV as an extern, so the bf16 -> fp32 upcast of the big operand is materialized (1.9 GB at 16k) instead of fused.

```python
# before
out = torch.matmul(probs.unsqueeze(1), values.float()).squeeze(1)
# after: Inductor fuses the upcast into the reduction
out = (probs.unsqueeze(-1) * values.float()).sum(1)
```

Kimi residual, T=16384: 3475 -> 2155 µs (ideal 794). Inductor's `decompose_mm_pass` does the same only above 10240 rows and not with BF16x9 fp32 matmuls enabled.

## 4. Small reductions: unroll instead of reduce

Inductor unrolls a reduction into a pointwise kernel only when its size is a **static** int below `unroll_reductions_threshold` (default 8, strict `<`). Otherwise it emits a reduction over a strided dim that runs far below bandwidth.

- Sum over K=16 rows: persistent reduction 1354 µs; Python loop over K (unrolled) 290 µs (ideal 281).
- Weighted sum over N=8: persistent 1574 µs; threshold 9: 353 µs.

```python
# after: a Python loop over a small static K becomes one pointwise kernel
acc = rows[:, 0].float()
for k in range(1, top_k):
    acc = acc + rows[:, k].float()
```

Or scope `options={"unroll_reductions_threshold": K + 1}` to the region. If the small size is symbolic, no threshold applies: see pattern 5 (keep it static), an `autograd.Function` that handles the width explicitly (Kimi residual dynamic width 3642 -> 2652 µs), or pattern 17.

## 5. Keep small dims static

Automatic dynamic shapes make a dim symbolic after it takes a second value. A symbolic hidden size or stack width turns unrolled kernels into loops and adds int64 index math.

```python
@local_compile("swiglu", batch_invariant=True)
def swiglu(x_gate, x_up):
    torch._dynamo.mark_static(x_gate, -1)   # hidden size F is fixed per call site
    ...
```

- **Check first:** newer trees already ship this for SwiGLU/SiTUGLU (`fused_binary_activation`: packed unbind, `mark_static`, recompile limit 16).
- **`mark_static` inside the region:** SwiGLU with static F made dense/routed/shared calls 1.68-1.75x faster, at 5 graphs instead of 3. Marking a view in eager before the call did not stick.
- **Unbind a packed projection inside the region:** pass `[R, 2, F]` and unbind in the region, so the unbind's backward (a `stack`, ~90 µs per MoE layer) fuses instead of running as a separate cat kernel.
- **A Python loop over a dim forces specialization** even under automatic dynamic shapes, one way to keep a small width static.
- **Views get symbolic strides and offsets** even when sizes are static; a packed `[rows, 2, F]` layout avoids it.
- **Mark the token dim dynamic in eager**, before the region (`maybe_mark_dynamic` is forbidden inside a graph), guarded by `is_compiling()` so the marks don't run when the caller is itself compiled:

```python
if not torch.compiler.is_compiling():
    torch._dynamo.maybe_mark_dynamic(x, 0)   # T varies
    torch._dynamo.mark_static(x, -1)         # normalized dim never varies
```

## 6. Deterministic gather-sum instead of scatter / atomics / opaque custom ops

A custom op used to force determinism is opaque to Inductor and blocks fusion. When every output row has a fixed set of contributors (each token owns exactly K expert rows), write combine as a gather-sum:

```python
# combine: out[t] = sum_k scores[t, k] * expert_out[inv_perm[t, k]], accumulated in fp32
rows = expert_out[inv_perm.flatten()].view(T, K, D)
acc = rows[:, 0].float() * scores[:, 0, None]
for k in range(1, K):                    # unrolled over K (pattern 4)
    acc = acc + rows[:, k].float() * scores[:, k, None]
out = acc.to(expert_out.dtype)
```

Wrap it in an `autograd.Function` whose backward is a gather by token. Kimi combine 13540 -> 1102 µs (ideal 827), deterministic, 2-3x more accurate vs fp64. Under compile, Inductor's own scatter can match a gather; measure both eager and compiled before switching.

## 7. Move scaling to the narrower tensor, inside a region

A linear map commutes with per-row scaling: `w2(h * s) == s * w2(h)` when w2 has no bias. Apply routing scores to the SwiGLU output (width F) inside the compiled SwiGLU, not to the w2 output (width D) in eager.

DSv3 MoE module, 4k: 9.36 -> 8.17 ms (-12.8%), peak memory -29% at 16k (the saved `[R, D]` tensor goes away), lower error vs fp64. Guard it exactly:

```python
use_scores_in_act = type(act) is SwiGLU and getattr(w2, "bias", None) is None and output_postprocess is None
```

In eager it is only -3% / -5% (4k / 16k) vs -12.8% / -13% with the region, so about a quarter to a third of the win survives without compile; the region is what makes it pay.

Under all-to-all EP the scores live on the source rank, so this rewrite needs them sent through the all-to-all with the tokens; the gather-sum combine (pattern 6) works on every dispatcher.

## 8. Narrow sort keys

`argsort` of int64 expert ids sorts all 64 key bits (22 kernels). Ids fit in int16 when the expert count does:

```python
keys = expert_ids.to(torch.int16) if num_experts <= torch.iinfo(torch.int16).max else expert_ids
order = torch.argsort(keys, stable=True)   # identical permutation
```

72.9 -> 30.4 µs, 11 kernels.

## 9. Unroll tiny iterative loops per entry

A Sinkhorn loop on `[T, 4, 4]` realizes every iteration (each column sum reads the previous step transposed): 163 kernels, 309 µs. Writing it as 16 per-entry `[T]` tensors gives 37 kernels, ~100 µs, bitwise in fp32/fp64. Cost: much longer cold compile (+~3 min). Present it as a trade-off.

## 10. Router glue as one region

Sigmoid scores, expert bias, group-limited top-k and token sums: compile them as one side-effect-free function (`_route`) and keep counters, bias updates and aux-loss injection in eager. Inductor lowers `topk` as an ATen fallback (no fusion). For group-limited routers, argmax rounds fuse instead; ungrouped routers keep `topk`. Router glue at [T, 256]: 4k 356.7 -> 135.9 µs.

## 11. Data-dependent shapes and fakes

- A custom op whose output size depends on data needs a fake that returns an unbacked size (`torch.library.get_ctx().new_dynamic_size()`), not the input's size. A wrong fake makes make_fx record the wrong shape and Inductor assert at runtime.
- Branches in fakes or backward on that size will guard and fail during tracing; `guard_or_true` avoids that, or better, make the real op return a shape that always matches the fake (e.g. zeros instead of an empty tensor on a rank that received nothing).
- Remove host syncs from the training path: use counts the library already returned, copy from pinned memory with `non_blocking=True` when the data is final, and avoid boolean-mask indexing (sort with a sentinel and slice by a host int instead).

## 13. fp32 GEMMs

Small fp32 GEMMs (DSv4 HcPre/Compressor projections, a router gate) run as slow SIMT kernels unless an emulated fp32 matmul precision (BF16x9) is on, and newer trees route fp32-output projections through dedicated modules (e.g. `HiMidLoLinear`; check what this tree has), and an upcast of a large bf16 input may be materialized first.

- Compute an fp32-output GEMM from bf16 inputs with fp32 accumulation instead of upcasting the input (DSv4 HcPre: 3262 -> 1291 µs).
- For a narrow fp32 backward, split the operand into exact bf16 pieces stacked along the output dim and use one bf16 GEMM with fp32 output (DSv3 router gate backward at [4096, 7168] x [256, 7168]: 0.288 -> 0.065 ms, same error vs fp64 at that K). The error depends on K: at K=16384 the piece split was ~35x worse than BF16x9, so check accuracy at your K.
- Under compile, Inductor removes bf16 round trips (`x.bfloat16().float()`), so a split written as `hi = x.bfloat16(); lo = (x - hi.float()).bfloat16()` collapses. Split by bit masking instead:

```python
hi = (x.view(torch.int32) & -65536).view(torch.float32)   # keep the top 16 bits: exact in bf16
lo = x - hi                                                # exact; repeat for a third piece
```

## 14. Wrap non-traceable kernels as custom ops with exact fakes

A library kernel that Dynamo can't trace causes a graph break, which inside a checkpointed block can make the whole block fall back to eager.

- Register it with `torch.library.custom_op` plus a fake whose output sizes, strides and dtype match the real kernel exactly (a fake that returns contiguous outputs where the kernel returns strided ones breaks downstream ops).
- DSv4 attention as a custom op: SAC block compile 302.5 -> 122.6 ms, because the block no longer graph-broke inside the checkpoint.
- Check libraries that pick a different kernel under `is_compiling()`: one used its fast CuTe path eagerly and a slow Triton path under compile (8.6x slower attention).

## 15. The loss path

- A loss region (final norm + lm_head + cross entropy) beat eager CE by 13% on DSv3 (4k: 34.6 -> 30.1 ms) and saved memory.
- Chunked loss: one lm_head call over the concatenated chunks of all predictions instead of one call per chunk: -18% at 4k. Choose the chunk count from the token count; at large vocab, per-chunk weight-grad accumulation can cost more than full logits.

## 16. `autograd.Function` with separate forward and backward regions

Compile the forward and the backward as two regions inside an `autograd.Function`, and detach inputs before calling the compiled forward. `requires_grad` is part of Dynamo's guards, so detaching the inputs lets a `no_grad` call reuse the same forward graph (no recompile), and you choose exactly what is saved for backward.

```python
_fwd = local_compile("name", batch_invariant=True)(fwd_impl)   # returns out and small saved stats
_bwd = local_compile("name_bwd", batch_invariant=True)(bwd_impl)

class Op(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        out, stats = _fwd(x.detach(), w.detach())   # detached: no_grad reuses this graph
        ctx.save_for_backward(x, w, stats)   # inputs plus small stats; large intermediates are recomputed in _bwd
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return _bwd(grad_out, *ctx.saved_tensors)
```

Kimi K3 attention residual: 24.3 ms eager and 12.2 ms compiled as written -> 1.56 ms per call (ideal 0.68).

## 17. Move a reduction over a small symbolic dim into a GEMM

A weighted sum over a small dim that varies across call sites (Kimi residual width N) lowers to a slow looped reduction when N is symbolic. Expressing it as a batched GEMM, `[T, N] x [N, D]` with the fp32 weights split into exact bf16 pieces (pattern 13), gave 180 µs vs 271 µs (static N) and 1435 µs (dynamic N). Check accuracy vs fp64. Same problem as patterns 4 and 5, different fix.

## 18. Emit an input grad in the layout the next kernel wants

Inductor returns a region's input grads token-major, whatever layout the input had, so the next kernel may pay a transpose copy. Passing a transposed input alone did not change that. An identity `autograd.Function` inside the region whose backward restrides the grad did:

```python
class _GradLayout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.transpose(0, 1).contiguous().transpose(0, 1)   # head-major memory, same logical shape
```

Whether it pays depends on the attention backend that consumes the grad (deceiving-quirks.md: -3.8% with Triton, neutral with FA4).

## 19. Grouped per-expert bias as a gather plus a one-hot GEMM

Eager expanded each expert's bias row by row, and the backward summed bias grads with a bf16 scatter-add (2.65 ms per 2 GPT-OSS layers at 16k). Gather the bias per row in the forward, and compute the bias grad as one GEMM against a one-hot routing matrix:

```python
bias_rows = bias[expert_of_row]                      # forward: [R, F] gathered, fuses into the GEMM epilogue
# backward: one-hot [E + 1, R] (extra row for padding) @ grad_out [R, F] -> [E + 1, F], fp32 accumulation
one_hot = torch.zeros(E + 1, R, device=g.device, dtype=g.dtype).scatter_(0, expert_of_row[None], 1)
bias_grad = (one_hot @ g).float()[:E]
```

Deterministic, accumulated in fp32, and faster in eager too. When one region serves both w13 and w2 (two widths), mark the width static: we saw 6 graphs over the two widths and raised the limit to 16. GPT-OSS 16k: -9.7%.

## 20. Eager hooks between attention and the output projection

Model-specific rescaling between attention and `wo` (GPT-OSS attention sinks) runs as a few eager kernels per layer; a region around it was -3.9% on GPT-OSS.
