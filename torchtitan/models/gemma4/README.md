# Gemma-4

Gemma-4 is Google DeepMind's open-weight model family featuring hybrid attention architectures across dense, mixture-of-experts, and edge models. Training recipes cover Gemma-4 E2B, E4B, 12B, 26B A4B (MoE), 31B (Dense), and a debug model used for testing.

## Download the tokenizer

```bash
python scripts/download_hf_assets.py --repo_id google/gemma-4-12b --assets tokenizer
```

The recipes expect tokenizer assets under `./assets/hf/gemma-4-<flavor>` (e.g. `./assets/hf/gemma-4-12b`).

## Training

```bash
# Debug model (used for functionality tests)
MODULE=gemma4 CONFIG=gemma4_debugmodel ./run_train.sh

# Debug model with varlen attention
MODULE=gemma4 CONFIG=gemma4_debugmodel_varlen_attn ./run_train.sh

# Gemma-4 E2B (Edge 2B)
MODULE=gemma4 CONFIG=gemma4_e2b ./run_train.sh

# Gemma-4 E4B (Edge 4B)
MODULE=gemma4 CONFIG=gemma4_e4b ./run_train.sh

# Gemma-4 12B (Dense)
MODULE=gemma4 CONFIG=gemma4_12b ./run_train.sh

# Gemma-4 26B A4B (Mixture-of-Experts)
MODULE=gemma4 CONFIG=gemma4_26b_a4b ./run_train.sh

# Gemma-4 31B (Dense)
MODULE=gemma4 CONFIG=gemma4_31b ./run_train.sh
```

Other recipes include `gemma4_12b_1node_full`, `gemma4_12b_multinode`, `gemma4_12b_long_context`, `gemma4_31b_1node_full`, `gemma4_31b_multinode`, and `gemma4_31b_long_context`. See [`config_registry.py`](./config_registry.py).

## Supported Parallelisms

| Feature | Status | Notes |
|---------|--------|-------|
| FSDP / HSDP | Supported | Default data-parallel path |
| Tensor Parallel (TP) | Supported | Column-parallel QKV/W1/W3, row-parallel Out/W2 |
| Sequence Parallel (SP) | Supported | Enabled via `--parallelism.enable_sequence_parallel` |
| Context Parallel (CP) | Supported | Standard language model CP via trainer context manager |
| Pipeline Parallel (PP) | Supported | Uses `pipeline_llm` schedule |
| Activation Checkpointing | Supported | Selective (`SelectiveAC`) and Full (`FullAC`) |
| `torch.compile` | Supported | Compatible with full model graph compilation |

## State Dict Adapter

`Gemma4StateDictAdapter` supports bidirectional checkpoint conversion between TorchTitan and Hugging Face format via `scripts/checkpoint_conversion/`.
