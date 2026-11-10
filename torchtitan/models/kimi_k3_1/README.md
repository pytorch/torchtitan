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

Sparse attention runs Attention Gym `gather_attn`. Its kernel does not yet
finish compiling for the Kimi-K3.1-Pro shape (576-wide latent, top-k 2048), so
only the debug model runs today.

## Supported Parallelisms

Only single-device runs have been tested.

## Numerical Parity

`KimiK31StateDictAdapter` maps the released Hugging Face checkpoint, including
the indexer weights.

No Hugging Face model code is released for K3.1.
