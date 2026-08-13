# Kimi K3

Kimi K3 combines a hybrid **Kimi Delta Attention (KDA) + Multi-head Latent
Attention (MLA)** decoder, **LatentMoE**, and a **MoonViT3d** vision encoder.
TorchTitan currently provides a topology-complete reduced model for architecture
validation, single-device training, and FSDP2 training.

## Prerequisites

Install the additional dependencies:

```bash
pip install av einops pillow torchvision flash-linear-attention
```

## Architecture

- **Decoder** -- hybrid KDA and MLA layers. The MLA layers follow the released
  model's explicit 1-based layer list, including consecutive MLA layers at the
  end of the decoder.
- **Feed-forward layers** -- one dense SiTU feed-forward layer followed by
  LatentMoE layers with sigmoid top-k routing, correction bias, routed experts,
  and shared experts.
- **Attention residuals** -- block-level attention residual connections,
  including the final output residual.
- **KDA backend** -- FLA's chunked Triton kernel, with a pure PyTorch recurrent
  implementation in the unit tests as the numerical reference.
- **Vision encoder** -- MoonViT3d with learned spatial positions, 2D RoPE,
  non-causal attention, temporal pooling, 2x2 spatial merge, and a two-layer
  projector to the decoder dimension.
- **Multimodal forward** -- projected vision embeddings are scattered into runs
  of the shared media placeholder token.

## Model variants

Only `debugmodel` is currently registered. The released Kimi K3 row is included
for architectural comparison and is not a runnable TorchTitan flavor.

| Variant | Parameters | LLM dim | Layers | MLA layers (1-based) | KDA layers | Heads | Experts (top-k) | ViT dim / layers / heads |
|---------|------------|---------|--------|----------------------|------------|-------|-----------------|--------------------------|
| Released Kimi K3 (reference) | 2.8T | 7168 | 93 | 4, 8, ..., 92, 93 | 69 | 96 | 896 (top-16) | 1024 / 27 / 12 |
| debugmodel | 100M | 256 | 13 | 4, 8, 12, 13 | 9 | 4 | 8 (top-2) | 256 / 4 / 3 |

`debugmodel` retains the released vocabulary size of 163840. Its depth also
preserves two structural edge cases from the released model: consecutive final
MLA layers and a short trailing attention-residual block.

## Supported Parallelisms

| Feature | Notes |
|---------|-------|
| FSDP / HSDP | Supported with the default SPMD backend. The decoder is sharded per layer and the vision encoder is a separate FSDP unit |
| Tensor Parallelism (TP) | Not supported |
| Expert Parallelism (EP) | Not supported |
| Pipeline Parallelism (PP) | Not supported |
| Context Parallelism (CP) | Not supported |

`torch.compile`, activation checkpointing, and parameter CPU offload are not
supported by the current Kimi K3 parallelization path.

Run the debug model on one GPU:

```bash
NGPU=1 MODULE=kimi_k3 CONFIG=kimi_k3_debugmodel ./run_train.sh
```

Run it with two-way FSDP2:

```bash
NGPU=2 MODULE=kimi_k3 CONFIG=kimi_k3_debugmodel ./run_train.sh \
  --parallelism.data_parallel_shard_degree 2
```

## Numerical Checks

`scripts/checkpoint_conversion/numerical_tests_kimi_k3.py` loads the released
HuggingFace config, modeling code, processor, and tokenizer from a local model
directory. It reduces the HuggingFace model to the `debugmodel` topology and
transfers the randomly initialized TorchTitan state dict, without loading the
released weights. Each side performs its own image preprocessing before the
full vision-projector-decoder forward. Float32 is the default correctness mode,
and the script does not override the framework's TF32 settings.

The current float32 CUDA validation result is:

- pixel preprocessing: max difference `1.192e-7`, with no values differing
  above `1e-6`;
- projected vision features: cosine similarity `1.000000`, max difference
  `3.152e-3`;
- expert routing: all `3936 / 3936` choices match;
- end-to-end last-token logits: KL `1.8215e-8`, cosine similarity `1.000002`,
  max difference `4.1358e-3`, top-1 match, and top-5 5/5.

Run the comparison with:

```bash
python -m scripts.checkpoint_conversion.numerical_tests_kimi_k3 \
  --hf_model_path ~/hf_assets/moonshotai/Kimi-K3
```

## TODO

- Add the full 2.8T model flavor.
- Add MXFP4 compressed checkpoint loading.
- Add TP, EP, PP, and CP support.
- Add packed-document attention support.
- Add video inputs and a video dataset training pipeline.
- Add `torch.compile`, activation checkpointing, and parameter CPU offload.
- Add generation-cache support.
