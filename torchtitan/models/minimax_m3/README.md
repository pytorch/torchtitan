# MiniMax M3

MiniMax M3 is a natively multimodal Mixture-of-Experts model whose decoder uses
MiniMax Sparse Attention (MSA) and whose vision encoder is a CLIP-style ViT with
3D RoPE.

## Architecture

The first three decoder layers use full GQA attention and a dense feed-forward
network; the other 57 use MSA and MoE. In an MSA layer, an index branch scores
128-token key blocks of each document and every query attends the top 16 blocks
of its KV group, its own block included. The index branch trains with a KL loss
toward the group-averaged main attention, with a stop-gradient teacher and a
detached index input, so the loss updates only the index query and key
projections and their norms.

## Released Model Configuration

The values below follow the
[official MiniMax M3 model card](https://huggingface.co/MiniMaxAI/MiniMax-M3).

| Component | Configuration |
|-----------|---------------|
| Parameters | 428B total, 23B activated |
| Decoder | 60 layers, hidden size 6144, 64 query heads, 4 KV heads, head dim 128, partial RoPE on 64 dims |
| Attention | 3 full-attention layers and 57 MSA layers (index head dim 128, block 128, top-16 blocks), context length 1048576 |
| MoE | 128 experts, top-4 sigmoid routing, hidden size 3072 per expert, 1 shared expert, clamped SwiGLU |
| Vocabulary | 200064 |
| Vision encoder | 32-layer ViT, width 1280, 3D RoPE, 2x2 patch merge |
| Modality | Text, image and video |

The `M3.1-Flash` flavor uses MSA in all 60 layers. The state dict adapter reads
unquantized checkpoints only, so it does not load the released NVFP4 and MXFP8
M3.1-Flash weights.

## Indexer Warmup

`build_model_config(..., sparse_attention=False)` runs every MSA layer as full
causal attention while the index branch keeps training on its KL loss, which is
the warmup the MSA paper uses before switching to sparse attention.
`indexer_loss_coeff` sets the KL coefficient.

## Supported Parallelisms

| Feature | Notes |
|---------|-------|
| FSDP2 / HSDP | Supported |
| Tensor / sequence parallelism | Supported; index query heads shard with their KV groups and the shared index key head replicates |
| Expert parallelism | Supported |
| Pipeline parallelism | Supported |
| Context parallelism | Not yet supported: each query's block selection reads the index keys of its whole document |

## Numerical Parity

The parity script reduces the released Hugging Face configuration to match
TorchTitan's local `debugmodel` configuration before initializing both models,
and loads the TorchTitan weights through Hugging Face's checkpoint conversion.

On a 333-token image and text prompt in bfloat16, the last-token KL divergence
against the Hugging Face implementation is **3.40e-5** with a top-1 match;
92.2% of the sparse attention block selections and 95.5% of the expert choices
match. With Hugging Face's block selections and expert choices imposed, the
KL divergence over all positions is **4.86e-5**.

Vision parity: pixel preprocessing max difference **4.8e-7**; projected vision
features max difference **1.2e-4**.

Test scripts:

- `scripts/checkpoint_conversion/numerical_tests_minimax_m3.py` -- Hugging Face vs.
  TorchTitan comparison
- `tests/unit_tests/cpu/models/minimax_m3/` -- MSA block selection, mask and KL loss
- `tests/unit_tests/gpu/models/minimax_m3/test_minimax_m3.py` -- sparse flex
  attention against masked SDPA
