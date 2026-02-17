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
MODEL=deepseek_v3 CONFIG=deepseek_v3_debugmodel ./run_train.sh
```

```bash
# 16B parameter model: adapted from older 16B parameter model from https://huggingface.co/deepseek-ai/deepseek-moe-16b-base
MODEL=deepseek_v3 CONFIG=deepseek_v3_16b ./run_train.sh
```

```bash
# 671B parameter model
MODEL=deepseek_v3 CONFIG=deepseek_v3_671b ./run_train.sh
```


## HuggingFace -> DCP Checkpoint Conversion

We implemented StateDictAdapter to perform HuggingFace safetensor to DCP format conversion. Currently, we only support conversion from HF checkpoints to DCP checkpoints offline (using CPU plain tensor).

Run the offline conversion script:
```bash
python scripts/checkpoint_conversion/convert_from_hf.py <hf_checkpoints_dir> <dcp_output_dir> --model_name deepseek_v3 --model_flavor 671B
```
