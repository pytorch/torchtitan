## How to convert a DeepSeek-V3 checkpoint for use in torchtitan

To continue training from an existing model checkpoint, the checkpoint must be in the DCP format expected by the checkpoint manager.
This folder contains the scripts for converting officially released DeepSeek-v3 checkpoints into the expected DCP format, from original Meta format, or from HuggingFace format, using GPUs.

#### Example usage

From HuggingFace format:
```bash
CONFIG_FILE=../../torchtitan/models/deepseek_v3/train_configs/deepseek_v3_671b.toml  ./convert_hf_to_dcp_with_gpus.sh --checkpoint.enable_checkpoint --checkpoint.convert_path=[checkpoint_folder] --checkpoint.convert_load_every_n_ranks=8
```

Note: Currently, the script will only work with the several layers, not the full 671B model on 8 GPUs. This is because the whole state_dict for 671B model is too huge to fit into 8 GPU.
