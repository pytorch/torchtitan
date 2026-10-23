## Supported Parallelisms

Coverage below matches the sharding set in [`sharding.py`](./sharding.py),
[`flavors.py`](./flavors.py), and the DeepSeek-V3 jobs in
`tests/integration_tests/models.py`.

| Feature | Notes |
|---------|-------|
| FSDP / HSDP | Default data-parallel path |
| Tensor Parallel (TP) | Including sequence parallel (`enable_sequence_parallel`); TP2 exercised together with CP2 and EP2 |
| Context Parallel (CP) | CP2 exercised with flex attention (`KVAllGatherCPFlexInnerAttention`); composes with FSDP, TP, and EP |
| Expert Parallel (EP) | EP2/EP8 exercised; composes with FSDP, TP, and CP |
| Pipeline Parallel (PP) | 1F1B and Interleaved1F1B schedules |
| Activation checkpointing | Selective (`SelectiveAC`) |
| `torch.compile` | Via `local_compile_regions` |

## Download Tokenizer

```bash
# DeepSeek 671B tokenizer (automatically downloads tokenizer.json and tokenizer_config.json)
python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3.1-Base --assets tokenizer
```

```bash
# DeepSeek 16B tokenizer:
python scripts/download_hf_assets.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer
```

> **Note:** We are reusing the tokenizer from deepseek-ai/deepseek-moe-16b-base to help users test and run the 16B model. This is not the official tokenizer for the DeepSeek-V3-16B model. The DeepSeek-V3 model has a different architecture from the deepseek-moe models (different attention implementation, MoE router implementation, etc.), making it not feasible to load deepseek-moe-16b model weights into DeepSeek-V3-16B.


## Training

```bash
# Quick debug run with small model
MODULE=torchtitan_recipes.tests.models.deepseek_v3 CONFIG=deepseek_v3_debugmodel ./run_train.sh
```

```bash
# 16B parameter model: adapted from older 16B parameter model from https://huggingface.co/deepseek-ai/deepseek-moe-16b-base
MODULE=torchtitan_recipes.tests.models.deepseek_v3 CONFIG=deepseek_v3_16b ./run_train.sh
```

```bash
# 671B parameter model
MODULE=torchtitan_recipes.models.deepseek_v3 CONFIG=deepseek_v3_671b ./run_train.sh
```

### Performance-optimized option

For better performance, DeepSeek-V3 models can opt into fused Triton kernels
for MLA Q/KV assembly, ComplexRoPE, and SwiGLU. The overrides preserve the
existing model parameters and checkpoint layout.

```bash
MODULE=torchtitan_recipes.models.deepseek_v3 CONFIG=deepseek_v3_671b ./run_train.sh \
  --override torchtitan_recipes.overrides.fused_mla.fused_mla \
  --override torchtitan_recipes.overrides.fused_swiglu.fused_swiglu
```

## HuggingFace -> DCP Checkpoint Conversion

We implemented StateDictAdapter to perform HuggingFace safetensor to DCP format conversion. Currently, we only support conversion from HF checkpoints to DCP checkpoints offline (using CPU plain tensor).

Run the offline conversion script:
```bash
python scripts/checkpoint_conversion/convert_from_hf.py <hf_checkpoints_dir> <dcp_output_dir> --model_name deepseek_v3 --model_flavor 671B
```
