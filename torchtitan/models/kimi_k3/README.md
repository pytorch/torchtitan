# Kimi K3

This directory contains the eager numerical reference implementation of Kimi
K3 in TorchTitan. The initial scope is a topology-complete, reduced model for
single-device or FSDP2 training and numerical comparison with the
[released HuggingFace implementation](https://huggingface.co/moonshotai/Kimi-K3).
It is intended to make architecture experiments and model-structure choices
measurable against a stable, inspectable baseline before optimized kernels and
additional parallelisms are introduced.

Every operator outside the KDA recurrence is plain eager PyTorch, which keeps
the model math directly inspectable. KDA itself runs on FLA's chunked Triton
kernel, following the same split Qwen3.5 uses: the kernel is the training path
and a pure-PyTorch recurrence in the unit tests pins its numerics.

## Quick start

```bash
NGPU=1 MODULE=kimi_k3 CONFIG=kimi_k3_debugmodel ./run_train.sh
```

Run the same eager model with two-way FSDP2:

```bash
NGPU=2 MODULE=kimi_k3 CONFIG=kimi_k3_debugmodel ./run_train.sh \
  --parallelism.data_parallel_shard_degree 2
```

Requirements beyond core TorchTitan are listed in `requirements.txt`: KDA needs
`flash-linear-attention` and the multimodal data path needs `torchvision`.

## Reduced model

`debugmodel` preserves each distinct Kimi K3 forward path while reducing
widths, head dimensions, expert count, and depth.

| Component | Released Kimi K3 | `debugmodel` |
|---|---:|---:|
| Parameters | 2.8T | 100M |
| Decoder dimension | 7168 | 256 |
| Vocabulary size | 163840 | 163840 |
| Decoder layers | 93 | 13 |
| Full MLA layers (1-based) | 4, 8, ..., 92, 93 | 4, 8, 12, 13 |
| KDA layers | 69 | 9 |
| Dense FFN layers | 1 | 1 |
| Attention residual block size | 12 | 12 |
| MLA heads | 96 | 4 |
| MLA qk_nope / qk_rope / v head dimension | 128 / 64 / 128 | 32 / 16 / 32 |
| KDA heads / head dimension | 96 / 128 | 4 / 32 |
| Routed experts / top-k | 896 / 16 | 8 / 2 |
| Routed latent / expert hidden dimension | 3584 / 3072 | 128 / 128 |
| Shared experts | 2 | 2 |
| Vision dimension | 1024 | 256 |
| Vision layers | 27 | 4 |
| Vision QKV dimension / heads | 1536 / 12 | 384 / 3 |

The depth sits one layer past a multiple of the full-attention period and of
the attention-residual block size, which reproduces two structural edge cases
of the released 93-layer stack: a final MLA layer immediately after a scheduled
one -- the released `full_attn_layers` ends `..., 88, 92, 93`, so the backbone
always closes on global attention -- and a short trailing residual block.
Neither is expressible as "every n-th layer", so `full_attention_layers` takes
the 1-based list verbatim and a future full-scale flavor can pass the released
24-entry list unchanged.
The released vocabulary size is retained, following other TorchTitan
multimodal debug models and making FSDP state sharding measurable while the
decoder widths and depths remain reduced. In `debugmodel` roughly 84 million
of the 100 million parameters are the token embedding and the separate output
projection over that vocabulary; the transformer itself is correspondingly
small.

## Forward structure

The reference path mirrors the released implementation in these areas:

- FP32-reduction RMSNorm and SiTU activation.
- Gated MLA with low-rank query and KV projections. Kimi K3 sets
  `mla_use_nope=True`, so the RoPE-sized query/key slices are not rotated.
- KDA short causal convolutions, safe decay gate, query/key L2 normalization,
  sigmoid beta, recurrent delta-rule update, and gated output RMSNorm.
- Block-level attention residuals, including the final output residual.
- Stable LatentMoE with sigmoid top-k routing, correction bias, latent
  down/up projections, routed experts, and shared experts.
- MoonViT3d patch embedding, learned spatial positions, 2D RoPE, non-causal
  per-image attention, temporal pooling, 2x2 spatial merge, and
  PatchMergerMLPV2.
- Vision features scattered into runs of media placeholder tokens.

`KimiKDAKernel` is the kernel boundary. It dispatches to FLA's `chunk_kda`
with the gate activation, beta sigmoid, and query/key L2 norm fused in, so a
future backend can replace it while preserving the same input/output contract
and checkpoint schema.

The vision encoder is its own FSDP unit, so its collectives only fire on ranks
that execute it. Because a data-parallel rank can legitimately receive a batch
with no images -- the shared multimodal collator emits `pixel_values=None` for
a text-only batch, and drops images to respect `max_images_per_batch` -- the
model runs the encoder on *every* batch. Batches without images use the
smallest grid the patch merger accepts and contribute its result through
`add_zero_valued_dependency`, which leaves the text embeddings numerically
unchanged while keeping the encoder in the autograd graph. Every rank
therefore issues the same all-gather and reduce-scatter regardless of what its
batch contains, and the encoder correctly receives zero gradients from
text-only ranks.

## Checkpoint conversion

`KimiK3StateDictAdapter` converts between TorchTitan and an unquantized
HuggingFace state dict. It covers:

- dense, MLA, KDA, and LatentMoE decoder layers;
- attention-residual parameters;
- vision patch embedding, fused HuggingFace QKV, transformer blocks, and
  projector;
- the MoE correction bias.

The released checkpoint stores routed expert weights in MXFP4. Loading those
compressed tensors is outside this first change. Numerical comparison should
therefore instantiate the same reduced, unquantized model on both sides and
copy one state dict through the adapter.

`test_kimi_k3_hf_parity.py` freezes the float32 outputs from a deterministic
reduced model evaluated with the released HuggingFace code at commit
`c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721`. The test covers text logits,
router choices, projected vision features, and end-to-end image-text logits.
The source model is loaded strictly from the state dict produced by
`KimiK3StateDictAdapter`; no full checkpoint or network access is required to
run the regression.

### Parity at released head dimensions

`debugmodel` shrinks the head dimensions, so it does not pin the FLA kernel
configuration the full model runs. That was checked separately, out of band, on
a 1.07B configuration built from the same `_kimi_k3_config` builder with the
released head dimensions kept intact: dimension 1024, 25 layers, full attention
at `4, 8, ..., 24, 25`, MLA `qk_nope / qk_rope / v` = `128 / 64 / 128` with 8
heads, KDA 8 heads of 128, 24 routed experts at top-4, vision dimension 512
over 8 layers. Same released commit, one RTX 5080, float32, 128 tokens. Both
sides run FLA's `chunk_kda`; the released code path for MLA and vision
attention was forced to eager, since flash-attn is not required here.

Float32 comparison requires closing **two** independent TF32 switches:

- **cuBLAS**, via `torch.backends.cuda.matmul.allow_tf32 = False`. Torch's own
  default is already off, but NVIDIA's NGC containers set
  `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, which flips
  `float32_matmul_precision` to `HIGH` at process init.
- **Triton**, via `TRITON_F32_DEFAULT=ieee`. The torch flag does not reach
  Triton kernels, and FLA's `chunk_kda` leans on the Triton default for the
  triangular solves in `fla/ops/kda/chunk_intra.py`.

| Quantity | max abs diff | reference max abs |
|---|---:|---:|
| Text logits, 128 tokens | 2.2e-5 | 5.8 |
| Projected vision features, 8x8 patch grid | 4.0e-5 | 4.4 |
| Multimodal logits, 128 tokens | 2.4e-4 | 5.8 |

Text logits differ by 3.9e-6 relative. For roughly 250 sequential dependent
matmuls that is well inside the linear accumulation bound of 1.5e-5 implied by
float32's 6.0e-8 unit roundoff. Per-layer relative drift grows from 4.8e-7
after layer 0 to 3.4e-6 by layer 23, and the per-layer amplification factor
stays in 0.96-1.44 throughout: no layer amplifies, the error only accumulates.
Routed expert IDs are identical for all 3072 token-routings (24 MoE layers x
128 tokens), and the argmax over the vocabulary agrees on every position.

Leaving Triton at its `tf32` default costs a factor of three end to end
(text logits 2.2e-5 -> 7.4e-5) and shows up inside KDA as a 50x step: every
tensor entering `chunk_kda` matches to ~1e-5 relative, while its output matches
only to ~5e-4. Both sides call the same kernel, so this never broke parity --
TF32's error is 99.6% correlated between the two runs and mostly cancels. What
does not cancel is the discontinuity: where the two nearly-identical inputs
land on opposite sides of a rounding boundary, the entry jumps a full TF32
quantum, 50x larger than the input difference that caused it. Under IEEE the
step disappears and `chunk_kda`'s output matches to ~6e-6.

With both switches closed, the eager operators and their FLA counterparts are
numerically indistinguishable: swapping TorchTitan's `Conv1d`+SiLU and eager
gated RMSNorm for FLA's `ShortConvolution` and `FusedRMSNormGated` moves the
logits difference from 2.2e-5 to 2.4e-5, and the two variants differ from each
other by 1.3e-5. None of the three is a privileged reference; what remains is
ordinary reassociation noise, including the routed-expert summation order.

The eager operators are kept deliberately. FLA's `ShortConvolution` and
`FusedRMSNormGated` are Triton-only and raise on CPU tensors; using them here
would make the whole model forward GPU-only and would leave
`test_kimi_k3_hf_parity.py` -- the frozen, network-free, CPU regression -- with
no way to run. They are also not a checkpoint concern either way:
`ShortConvolution` subclasses `nn.Conv1d` with the same `(D, 1, K)` weight, and
`FusedRMSNormGated` carries the same `(D,)` weight, so the state-dict mapping is
unaffected by the choice.

### bfloat16

Comparing the two implementations directly in bfloat16 measures top-k router
ties, not arithmetic. Each side must be compared against its own float32 run.

| bfloat16 vs own float32 | routings changed | logits max abs | argmax kept |
|---|---:|---:|---:|
| TorchTitan | 808 / 3072 | 5.69 | 50.0% |
| Released HuggingFace | 921 / 3072 | 5.96 | 47.7% |

Both implementations lose their own float32 result at the same rate, so the
instability is the model configuration rather than either implementation. It
comes from the router: a randomly initialized gate produces near-tied scores,
bfloat16 rounding reorders them, and a changed expert changes the residual
stream enough to change every later routing decision. Rounding only the gate
of the first MoE layer to bfloat16, with a float32 input, already flips 4 of
128 tokens; those tokens have a median top-4/top-5 score margin of 5.4e-4
against a typical score spread of 5.5e-1.

Routing every token to all 24 experts removes the ties and leaves only the
arithmetic. Nothing then flips on either side, and the two implementations
degrade identically:

| bfloat16 vs own float32, all experts routed | logits max abs | logits mean abs | argmax kept |
|---|---:|---:|---:|
| TorchTitan | 1.72e-1 | 1.66e-2 | 97.7% |
| Released HuggingFace | 1.88e-1 | 1.68e-2 | 94.5% |

Per-layer bfloat16 error grows from 0.8% of the activation scale after layer 0
to roughly 3% by the end of an attention-residual block on both sides, with
neither consistently ahead. That is ordinary bfloat16 accumulation over 25
layers, given a 0.4% unit roundoff.

The practical consequence: validate numerics in float32, as
`.claude/rules` already requires, and do not read a bfloat16 loss difference
against a reference implementation as evidence of a bug until the routing
decisions have been checked.

## Tests

The unit tests cover:

- the reduced layer topology;
- the explicit exact GELU against PyTorch's CPU reference;
- a small text+image model forward and backward;
- exhaustive state-dict round-trip for that small model;
- reduced text, vision, router, and multimodal numerical parity against frozen
  HuggingFace eager outputs;
- single-rank FSDP2 forward and per-parameter gradient parity with a manually
  cast BF16 reference;
- two-rank FSDP2 forward and backward when one rank has an image and the other
  rank is text-only.

`ReferenceKimiKDAKernel` in `tests/unit_tests/test_kimi_k3.py` is the explicit
recurrent formulation of KDA. The tests above build the model with it in place
of the FLA kernel, which is what lets them run on CPU and at head dimensions
FLA cannot compile. A separate CUDA-only test checks the FLA kernel against
that same reference, forward and backward, for both gate activations.

```bash
pytest -q \
  tests/unit_tests/test_kimi_k3.py \
  tests/unit_tests/test_kimi_k3_hf_parity.py
pytest -q tests/unit_tests/test_kimi_k3_fsdp.py
```

## First-version limitations

- FSDP2 data parallelism is supported; HSDP, TP, PP, CP, and EP are rejected.
- No packed documents, activation checkpointing, `torch.compile`, or CPU
  offload.
- Image inputs are supported; video inputs are rejected.
- No generation cache.
- No MXFP4 checkpoint loading.
- No full 2.8T flavor.

These restrictions are explicit so unsupported runtime settings fail instead
of being silently ignored. This first contribution deliberately limits
parallel execution to FSDP2: its purpose is to establish the eager numerical
reference used by Kimi K3 architecture experiments. TP, PP, CP, and EP can be
added independently after that forward contract is locked.
