# Gemma-4

Gemma-4 is Google DeepMind's language model family featuring a hybrid attention architecture that combines sliding-window local attention with global attention on the final layer. Training recipes cover Gemma-4 12B, 31B, and a small debug model used for testing.

## Download the tokenizer

```bash
python scripts/download_hf_assets.py --repo_id google/gemma-4-12b --assets tokenizer
```

The 12B and 31B recipes expect tokenizer assets under `./assets/hf/gemma-4-12b` (and matching `./assets/hf/gemma-4-31b`).

## Training

```bash
# Debug model (used for functionality tests)
MODULE=gemma4 CONFIG=gemma4_debugmodel ./run_train.sh

# Debug model with varlen attention
MODULE=gemma4 CONFIG=gemma4_debugmodel_varlen_attn ./run_train.sh

# Gemma-4 12B
MODULE=gemma4 CONFIG=gemma4_12b ./run_train.sh

# Gemma-4 31B
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
