# Huggingface Transformers Modeling backend

This enables HF transformers models to be trained with `4D parallelism + torch.compile`

## Quick start

- Requirements `transformers==4.57.1`

- Config: `torchtitan/experiments/transformers_modeling_backend/config_registry.py`
```diff
...
- --module llama3
+ --module transformers_modeling_backend
--config transformers_modeling_backend_debugmodel
...
```
- Train: `LOG_RANK=7 MODEL=transformers_modeling_backend CONFIG=transformers_modeling_backend_debugmodel ./run_train.sh --compile.enable`
    - Make sure you have created the tokenizers beforehand
<img width="1334" height="453" alt="image" src="https://github.com/user-attachments/assets/da459448-027b-4af9-8176-6a3e433a272c" />

## Supported Features

- The following models were tested:
    - Dense (FSDP/CP/TP/PP/`torch.compile`)
        - `meta-llama/Llama-3.2-1B`
        - `microsoft/phi-2`
        - `Qwen/Qwen2.5-7B`
        - `mistralai/Mistral-7B-v0.1`
        - `ByteDance-Seed/Seed-Coder-8B-Instruct`
        - `Qwen/Qwen3-4B-Instruct-2507`
        - `arcee-ai/AFM-4.5B`
        - `ibm-granite/granite-3b-code-base-2k`
        - `baidu/ERNIE-4.5-0.3B-Base-PT`
        - `kyutai/helium-1-preview-2b`
        - `allenai/OLMo-7B-hf`
        - `mistralai/Ministral-8B-Instruct-2410`
    - MoE (upcoming)

## Known issues to address later

- When using HF modeling, the test `FSDP=2 vs FSDP=2 + PP=2`, the `loss` and `grad_norm` not bitwise matching (but converging) while it is the case with Torchtitan modeling. This will be addressed in another PR but the culprit is probably `register_buffer` when loading `seed_checkpoint`
- the HF modeling has lower MFU than Torchtitan MFU

## Further work

- Missing `build_optimizers_with_moe_load_balancing` support for MoE
- Missing TP/PP/EP supports for MoE
- Load HF weights
- Add LORA support
