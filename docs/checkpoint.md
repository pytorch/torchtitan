# How to use checkpointing in `torchtitan`

You may want to enable checkpointing in `torchtitan` for better fault tolerance during training, or to enable easier importing and exporting of weights between `torchtitan` and other libraries. `torchtitan` offers varying degrees of support for other checkpoint formats which are listed further below.

## A general guide to use checkpoints during training

1. ENABLE CHECKPOINTING
In your `torchtitan` training config, ensure that `enable_checkpoint` is set to True.
```
[checkpoint]
enable_checkpoint = true
folder = "checkpoint"
interval = 500
```
2. SAVE MODEL ONLY
By setting `last_save_model_only` to `True`, the checkpoint will only contain the model and exclude the optimizer state and extra train states, resulting in a smaller checkpoint size.
```
[checkpoint]
enable_checkpoint = true
last_save_model_only = true
```

3. CHOOSE DESIRED EXPORT PRECISION
The default model states are in `float32`. You can choose to export the checkpoint in a lower precision format such as `bfloat16`.
```
[checkpoint]
enable_checkpoint = true
last_save_model_only = true
export_dtype = "bfloat16"
```

4. EXCLUDING SPECIFIC KEYS FROM CHECKPOINT LOADING
In some cases, you may want to partially load from a previous-trained checkpoint and modify certain settings, such as the number of GPUs or the current step. To achieve this, you can use the `exclude_from_loading` parameter to specify which keys should be excluded from loading.
This parameter takes a list of string that should be excluded from loading.
```
[checkpoint]
enable_checkpoint = true
exclude_from_loading = ["data_loader", "lr_scheduler"]
```
When used in command line, the parameter should be a comma-separated list of strings. For example: `--checkpoint.exclude_from_loading data_loader,lr_scheduler`.

5. EXAMPLE CHECKPOINT CONFIGURATION
```
[checkpoint]
enable_checkpoint = true
folder = "checkpoint"
interval = 10
load_step = 5
last_save_model_only = true
export_dtype = "bfloat16"
```

A more exhaustive and up-to-date list of checkpoint config options can be found in `torchtitan/config/job_config.py`

## Creating a seed checkpoint
Sometimes one needs to create a seed checkpoint to initialize a model from step 0.
E.g. it is hard, if not impossible, for meta initialization on multiple devices to reproduce the initialization on a single device.
A seed checkpoint does initialization of the model on a single CPU, and can be loaded from another job on an arbitrary number of GPUs via DCP resharding.

To create a seed checkpoint, use the same model config as you use for training.
e.g.
```bash
NGPU=1 CONFIG=<path_to_model_config> ./run_train.sh --checkpoint.enable_checkpoint --checkpoint.create_seed_checkpoint --parallelism.data_parallel_replicate_degree 1 --parallelism.data_parallel_shard_degree 1 --parallelism.tensor_parallel_degree 1 --parallelism.pipeline_parallel_degree 1 --parallelism.context_parallel_degree 1 --parallelism.expert_parallel_degree 1
```

## Conversion support

### HuggingFace
`torchtitan` offers two ways to work with Hugging Face models: either by directly saving and loading a Hugging Face checkpoint during training, or by using an example conversion script to directly reformat the model weights on cpu.

1. You can directly save huggingface model weights during training by using the `--checkpoint.last_save_in_safetensors_format` and `--checkpoint.last_save_model_only` options together. To directly load a `torchtitan` training session from a huggingface safetensors file, enable `--checkpoint.initial_load_model_only` and `--checkpoint.initial_load_in_hf`, and set `--checkpoint.initial_load_path` to the directory containing the huggingface checkpoint.

2. To directly reformat the weights without the need to run a training loop, run the corresponding conversion script. The naming scheme is `torchtitan`-centric, e.g. convert_from_hf means convert hf->tt.

```bash
python ./scripts/checkpoint_conversion/convert_from_hf.py <input_dir> <output_dir> --model_name <model_name> --model_flavor <model_flavor>
python ./scripts/checkpoint_conversion/convert_to_hf.py <input_dir> <output_dir> --model_name <model_name> --model_flavor <model_flavor>
# e.g.
python ./scripts/convert_from_hf.py ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/ ./initial_load_path/ --model_name llama3 --model_flavor 8B
```

### Torch

This guide will walk you through the steps required to convert a checkpoint from `torchtitan` so that it can be loaded into pt format.

1. CHECKPOINT CONFIGURATION
```
[checkpoint]
enable_checkpoint = true
folder = "checkpoint"
interval = 10
last_save_model_only = true
export_dtype = "bfloat16"
```

2. SAVE THE FINAL CHECKPOINT\
Once the above have been set, the final checkpoint at the end of the training step will consist of model only with the desired export dtype. However, if the final step has not been reached yet, full checkpoints will still be saved so that training can be resumed.

3. CONVERT SHARDED CHECKPOINTS TO A SINGLE FILE\
Finally, once you have obtained the last checkpoint, you can use the following command to convert the sharded checkpoints to a single .pt file.

```bash
python -m torch.distributed.checkpoint.format_utils dcp_to_torch torchtitan/outputs/checkpoint/step-1000 checkpoint.pt
```


That's it. You have now successfully converted a sharded `torchtitan` checkpoint for use with pytorch formats.

### PyTorch Meta Llama

An example script for converting the original Llama3 checkpoints into DCP format to be used with `torchtitan` can be found in `scripts/convert_from_llama.py`.

The script expects a path to the original checkpoint files, and a path to an output directory:
```bash
python -m scripts.convert_from_llama <input_dir> <output_dir>
```
