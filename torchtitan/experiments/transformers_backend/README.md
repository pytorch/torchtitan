# Huggingface Transformers backend

## Quick start

- Requirements `transformers==4.55.4`

- Config: `torchtitan/torchtitan/experiments/transformers_backend/configs/qwen3_fsdp2_tp2_pp2.toml`
```diff
...
[model]
- name = "llama3"
+ name = "Qwen/Qwen3-4B-Instruct-2507"
flavor = "debugmodel"
hf_assets_path = "./tests/assets/tokenizer"
...
```
**Note:** Any model name containing "/" is automatically recognized as a HuggingFace model ID and will use the `transformers_backend`.

- Train: `LOG_RANK=7 CONFIG_FILE=<YOUR_PATH>/torchtitan/experiments/transformers_backend/configs/qwen3_fsdp2_tp2_pp2.toml ./run_train.sh --compile.enable`
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
