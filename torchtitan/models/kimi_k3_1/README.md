# Kimi K3.1

Kimi K3.1 Pro keeps the Kimi K3 backbone and replaces its 24 Gated MLA layers
with Gated DeepSeek Sparse Attention (DSA) at the same layer positions.

## Prerequisites

Same as [Kimi K3](../kimi_k3/README.md).

## Architecture

Each DSA layer keeps the Kimi K3 MLA projections and output gate, and adds an
indexer with 32 heads of dimension 128 that selects 2048 tokens per query.
Top-k selection and the indexer loss come from
`torchtitan/models/deepseek_v4/compressor.py`. The indexer is trained only by
the indexer loss. `build_model_config(..., indexer_loss_coeff=...)` sets its
coefficient (default 0.01); 0 turns the loss off.

Sparse attention runs Attention Gym `gather_attn` with `(latent, rope)`
pairs. Its fused kernel needs 16 to 128 heads, so the debug model has 16.

## Supported Parallelisms

| Feature | Notes |
|---------|-------|
| FSDP2 | Tested with EP 2 |
| Expert parallelism | Tested with FSDP 2 |
| Tensor / sequence parallelism | Tested TP 2 with EP 2, with a 32-head debug model; each rank needs 16 to 128 query heads for the Attention Gym pair kernel |
| Context parallel | Not supported |
| Pipeline parallelism | Tested PP 2 (Interleaved1F1B) |

Over 20 steps from the same seed checkpoint, each tested layout matches its
baseline within run-to-run noise: the loss differs by up to 4.6e-3 to 8.0e-3,
and two runs of the same baseline differ by up to 4.9e-3 to 6.0e-3.

## Numerical Parity

`KimiK31StateDictAdapter` maps the released Hugging Face checkpoint, including
the indexer weights.

No Hugging Face model code is released for K3.1, so the forward reference is
vLLM's K3.1 DSA layer. One debug-model DSA layer (random weights, 1024 tokens)
matches it: in fp32 (reference kernel) the top-k is the same and the
relative L2 error is 6.6e-7; in bf16 the top-k overlap is 99.6% and the
relative L2 error is 7.2e-3 with the same top-k.
