# Nemotron-3 Nano

Nemotron-3 Nano is a hybrid Mamba-Transformer Mixture-of-Experts (MoE) model combining Mamba-2 SSM and Grouped-Query Attention (GQA) layers with granular routed experts.

## Download Tokenizer

```bash
python scripts/download_hf_assets.py --repo_id nvidia/nemotron-3-nano --assets tokenizer
```

## Training

```bash
# Debug model (used for CI and testing)
MODULE=nemotron_nano CONFIG=nemotron_debugmodel ./run_train.sh

# Nemotron-3 Nano 31B
MODULE=nemotron_nano CONFIG=nemotron_31b ./run_train.sh
```

See [`config_registry.py`](./config_registry.py) for available configuration options.

## Checkpoint Conversion

`NemotronStateDictAdapter` supports bidirectional checkpoint conversion between Hugging Face format and TorchTitan DCP via `scripts/checkpoint_conversion/`.
