# Nemotron-3 Nano

Nemotron-3 Nano is a hybrid Mamba-Transformer Mixture-of-Experts (MoE) model. Architecture features include interleaved Mamba-2 SSM and Grouped-Query Attention (GQA) layers coupled with granular routed experts. Training recipes cover Nemotron-3 Nano 31B (31.6B total / 3.2B active) and a small debug model used for testing.

## Download the tokenizer

```bash
python scripts/download_hf_assets.py --repo_id nvidia/nemotron-3-nano --assets tokenizer
```

## Training

```bash
# Debug model (used for functionality tests)
MODULE=nemotron_nano CONFIG=nemotron_debugmodel ./run_train.sh

# Nemotron-3 Nano 31B
MODULE=nemotron_nano CONFIG=nemotron_31b ./run_train.sh
```

See [`config_registry.py`](./config_registry.py) for full configuration options.

## Supported Parallelisms

| Feature | Status | Notes |
|---------|--------|-------|
| FSDP / HSDP | Supported | Default data-parallel path |
| Tensor Parallel (TP) | Supported | Dense projections & Attention/MLP TP |
| Expert Parallel (EP) | Supported | MoE expert sharding across ranks |
| Sequence Parallel (SP) | Supported | Sequence sharding for long context |
| Pipeline Parallel (PP) | Supported | Uses `pipeline_llm` schedule |
| Activation Checkpointing | Supported | Selective (`SelectiveAC`) and Full (`FullAC`) |
| `torch.compile` | Supported | Compatible with model compile |

## State Dict Adapter

`NemotronStateDictAdapter` supports bidirectional checkpoint conversion between TorchTitan and Hugging Face format via `scripts/checkpoint_conversion/`.
