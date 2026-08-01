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

The `debugmodel` flavor preserves each distinct Kimi K3 forward path while
reducing widths, expert count, and depth.

| Component | Released Kimi K3 | `debugmodel` |
|---|---:|---:|
| Decoder dimension | 7168 | 256 |
| Vocabulary size | 163840 | 163840 |
| Decoder layers | 93 | 13 |
| Full MLA layers (1-based) | 4, 8, ..., 92, 93 | 4, 8, 12 |
| KDA layers | 69 | 10 |
| Dense FFN layers | 1 | 1 |
| Attention residual block size | 12 | 12 |
| Routed experts / top-k | 896 / 16 | 8 / 2 |
| Routed latent / expert hidden dimension | 3584 / 3072 | 128 / 128 |
| Shared experts | 2 | 2 |
| Vision dimension | 1024 | 256 |
| Vision layers | 27 | 4 |
| Vision QKV dimension / heads | 1536 / 12 | 384 / 3 |

Thirteen decoder layers are intentional. They exercise two attention-residual
blocks and preserve the released model's 1-based full-attention cadence.
The released vocabulary size is retained, following other TorchTitan
multimodal debug models and making FSDP state sharding measurable while the
decoder widths and depths remain reduced. The resulting model has about 100
million parameters, of which roughly 84 million are the token embedding and
the separate output projection over that vocabulary; the transformer itself is
correspondingly small.

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
and checkpoint schema. FLA's chunked kernel cannot compile head dimensions
below 16, and the reduced flavor uses 32.

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
