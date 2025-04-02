## How to convert a Llama 4 checkpoint for use in torchtitan

To continue training from an existing model checkpoint, the checkpoint must be in the DCP format expected by the checkpoint manager.
This folder contains the scripts for converting officially released Llama 4 checkpoints into the expected DCP format, from original Meta format, or from HuggingFace format, using GPUs.

#### Example usage

From Meta format:
```bash
CONFIG_FILE=../train_configs/llama4_16.toml ./convert_meta_to_dcp.sh --checkpoint.enable_checkpoint --checkpoint.convert_path=[checkpoint_folder] --checkpoint.convert_load_every_n_ranks=8
```


From HuggingFace format:
```bash
CONFIG_FILE=../train_configs/llama4_16.toml  ./convert_hf_to_dcp_with_gpus.sh --checkpoint.enable_checkpoint --checkpoint.convert_path=[checkpoint_folder] --checkpoint.convert_load_every_n_ranks=8
```
