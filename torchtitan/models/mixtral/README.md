## Available Features

- Mixtral 8x7B MoE model:
    - Supports FSDP/HSDP, TP, EP, ETP.
    - Supports AC (selective/full), torch.compile.
    - MoE uses Token Choice top-2 routing with auxiliary-loss-free load balancing.
    - Every layer is MoE (no dense FFN fallback).
- HuggingFace checkpoint conversion via StateDictAdapter (bidirectional).
- Configs: `debugmodel` (4 layers, 4 experts), `8x7b` (32 layers, 8 experts).

## Download Tokenizer

```bash
python scripts/download_hf_assets.py --repo_id mistralai/Mixtral-8x7B-v0.1 --assets tokenizer
```

## Training

```bash
# Quick debug run with small model
MODEL=mixtral CONFIG=mixtral_debugmodel ./run_train.sh
```

```bash
# 8x7B parameter model
MODEL=mixtral CONFIG=mixtral_8x7b ./run_train.sh
```

## HuggingFace -> DCP Checkpoint Conversion

```bash
python scripts/checkpoint_conversion/convert_from_hf.py <hf_checkpoints_dir> <dcp_output_dir> --model_name mixtral --model_flavor 8x7b
```

