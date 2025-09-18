# Scripts for Downlaoding and Converting HuggingFace Checkpoints

This repository contains scripts to download and convert HuggingFace Llama models to be compatible with TorchTitan.


## Downloading HF Files

### Downloading HuggingFace Llama Checkpoints

You can download HuggingFace model checkpoints using the following command:
```bash
python download_hf_checkpoint.py --model_name <model_name>
# Example: python download_hf_checkpoint.py --model_name meta-llama/Llama-3.2-1B
```

The checkpoints will be downloaded to `HF_HOME` if set, or `~/.cache/huggingface/hub/models--<model_name>/snapshots/<commit_hash>/` otherwise. 
It's recommended to move the downloaded files to your desired location.


### Handling Safetensor Index Files

For large models, weights are split into multiple safetensor chunks, and a `model.safetensors.index.json` file is provided to map weights to their respective files. 
However, small models (e.g., Llama 3.2 1B) may not have this index file since all weights are stored in a single checkpoint file.

To generate a missing `model.safetensors.index.json` file, run:
```bash 
python make_safetensors_index.py --fpath <path_to_model_directory>
```

This is required for metadata when converting HF checkpoints to DCP format.



### Downloading Llama Tokenizer

To use the models, you need to download the corresponding tokenizer. You can do this by running:
```bash
python download_tokenizer.py --repo_id <repo_id> --hf_token <hf_token> --local_dir <local_dir>
# Example: python download_tokenizer.py --repo_id meta-llama/Llama-3.2-1B
```

If `local_dir` is not specified, the tokenizer will be saved under `assets/tokenizer/{model_name}`.




## Converting HF Checkpoints to DCP Format

To convert HF checkpoints to Titan DCP format, follow these steps:

1. Move to the `convert_hf_to_dcp` directory: `cd torchtitan/experiments/evaluation/scripts/convert_hf_to_dcp`.

2. Run the conversion script: `bash convert_hf_to_dcp_with_gpus.sh`.

Make sure to update the checkpoint path in `convert_hf_to_dcp_with_gpus.sh` and the tokenizer path in the TOML files under `evaluation/llama3/train_configs` according to your directory structure. 
Also, adjust the number of GPUs as needed for your experimental setup.


## Sanity Check for Converted Checkpoints

A Jupyter notebook, `sanity_check/compare_hf_titan_models.ipynb`, is provided to compare the outputs from the original HuggingFace model and the converted TorchTitan model. 
Although the architectures are matched, minor differences in logit values may occur due to attention operations. 
These differences are typically marginal (around the 7th decimal place) and do not indicate any bugs in the conversion process.