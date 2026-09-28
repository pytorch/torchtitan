# Gemma-4

Gemma-4 is Google's open-weight model family featuring a hybrid attention architecture (sliding-window FlexAttention with periodic global SDPA attention), GeGLU feed-forward networks, and tied embeddings.

## Model Variants

| CONFIG | Flavor | Architecture | Layers | Dim | Heads (Q / KV) | Global KV Heads | Sliding Window | Context Length |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `gemma4_debugmodel` | `debugmodel` | Dense (Testing) | 6 | 256 | 16 / 16 | N/A | Full | 2,048 |
| `gemma4_e2b` | `e2b` | Edge Dense | 35 | 1,536 | 8 / 1 | 1 | 512 | 262,144 |
| `gemma4_e4b` | `e4b` | Edge Dense | 42 | 2,560 | 8 / 2 | 2 | 512 | 262,144 |
| `gemma4_12b` | `12b` | Dense | 48 | 3,840 | 16 / 8 | 1 | 1,024 | 262,144 |
| `gemma4_26b_a4b` | `26b_a4b` | Mixture-of-Experts | 30 | 2,816 | 16 / 8 | 2 | 1,024 | 262,144 |
| `gemma4_31b` | `31b` | Dense | 60 | 5,376 | 32 / 16 | 4 | 1,024 | 262,144 |

### Instruction-tuned (`-it`) variants

Each flavor also has an instruction-tuned recipe named `gemma4_<flavor>_it`
(`gemma4_debugmodel_it`, `gemma4_e2b_it`, `gemma4_e4b_it`, `gemma4_12b_it`,
`gemma4_26b_a4b_it`, `gemma4_31b_it`). The instruction-tuned checkpoints share
the **exact architecture** of their base counterpart — only the released
weights, tokenizer, and chat template differ — so each `_it` recipe reuses its
base recipe and base model flavor (mirroring how llama3's `sft_debugmodel`
reuses the `debugmodel` flavor). Select the instruction-tuned lineage by
pointing `--hf_assets_path` / `--checkpoint.initial_load_path` at the
corresponding `google/gemma-4-<flavor>-it` weights when training. Checkpoint
conversion uses the base size flavor (e.g. `--model_flavor 12b`).

## Download Tokenizer

```bash
python scripts/download_hf_assets.py --repo_id google/gemma-4-12b --assets tokenizer
```

The recipes expect tokenizer assets under `./assets/hf/gemma-4-<flavor>`.

## Training

```bash
# Debug model (functionality testing)
MODULE=gemma4 CONFIG=gemma4_debugmodel ./run_train.sh

# Full training run (e.g. Gemma-4 12B)
MODULE=gemma4 CONFIG=gemma4_12b ./run_train.sh

# Instruction-tuned (-it) variant (continue from the -it checkpoint)
MODULE=gemma4 CONFIG=gemma4_12b_it ./run_train.sh \
  --hf_assets_path <gemma-4-12b-it_dir> \
  --checkpoint.initial_load_in_hf \
  --checkpoint.initial_load_path <gemma-4-12b-it_dir>
```

### CPT pipeline

The end-to-end continued-pretraining smoke test accepts either lineage; pass the
`-it` suffix to exercise the instruction-tuned checkpoints:

```bash
# Base
./development_artifacts/tests/run_gemma4_cpt_pipeline.sh 12b

# Instruction-tuned (resolves gemma-4-12b-it weights + gemma4_12b_it config)
./development_artifacts/tests/run_gemma4_cpt_pipeline.sh 12b_it
```

## Supported Parallelisms

| Feature | Notes |
|---------|-------|
| FSDP2 / HSDP | Recommended for batch scaling at standard context (e.g. 8k); shards per layer across `dp_shard` with optional replication across `dp_replicate` |
| Context Parallel (CP) | Recommended for long sequences (32k–256k) by sharding the sequence dimension |
| Tensor Parallel (TP) | Supported when `global_kv_heads % tp == 0` (e.g. `tp=2` for E4B/26B-A4B; `tp=2,4` for 31B) |
| Pipeline Parallel (PP) | Standard `pipeline_llm` schedule |
| Activation Checkpointing | Selective (`SelectiveAC`) and Full (`FullAC`) |
| `torch.compile` | Compatible with PyTorch Inductor full-graph compilation |

## HuggingFace Checkpoint Conversion

Convert checkpoints between Hugging Face format and TorchTitan DCP format using [`Gemma4StateDictAdapter`](./state_dict_adapter.py):

```bash
python scripts/checkpoint_conversion/convert_from_hf.py \
  --model_name gemma4 \
  --model_flavor 12b \
  --hf_checkpoints_dir <hf_checkpoints_dir> \
  --dcp_output_dir <dcp_output_dir>
```
