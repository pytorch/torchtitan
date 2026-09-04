# Nemotron-3 Model Family

Nemotron-3 is a family of hybrid Mamba-Transformer Mixture-of-Experts (MoE) models combining Mamba-2 SSM and Grouped-Query Attention (GQA) layers with granular routed experts.

## Supported Flavors

- `nemotron_debugmodel`: CI/local testing flavor
- `nemotron_4b`: 4B hybrid Mamba-MoE model (32 experts, top-4 routing)
- `nemotron_31b`: 31.6B total / 3.2B active parameter Nano model (128 experts, top-6 routing)
- `nemotron_120b`: 120B Super hybrid Mamba-MoE model (128 experts, top-8 routing)
- `nemotron_550b`: 550B Ultra hybrid Mamba-MoE model (256 experts, top-8 routing)

## Download Tokenizer

```bash
python scripts/download_hf_assets.py --repo_id nvidia/nemotron-3-nano --assets tokenizer
```

## Training

```bash
# Debug model (used for CI and testing)
MODULE=nemotron_nano CONFIG=nemotron_debugmodel ./run_train.sh

# Nemotron-3 4B
MODULE=nemotron_nano CONFIG=nemotron_4b ./run_train.sh

# Nemotron-3 Nano 31B
MODULE=nemotron_nano CONFIG=nemotron_31b ./run_train.sh

# Nemotron-3 Super 120B
MODULE=nemotron_nano CONFIG=nemotron_120b ./run_train.sh

# Nemotron-3 Ultra 550B
MODULE=nemotron_nano CONFIG=nemotron_550b ./run_train.sh
```

See [`config_registry.py`](./config_registry.py) for available configuration options.

## Checkpoint Conversion

`NemotronStateDictAdapter` supports bidirectional checkpoint conversion between Hugging Face format and TorchTitan DCP via `scripts/checkpoint_conversion/`.
