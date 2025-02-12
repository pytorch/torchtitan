## How to convert a Llama 3 checkpoint for use in torchtitan

If you want to continue training from an existing model checkpoint, the checkpoint must be in the DCP format expected by the checkpoint manager.
An example script for converting the original Llama3 checkpoints into the expected DCP format can be found in `scripts/convert_llama_to_dcp.py`.

The script expects a path to the original checkpoint files, and a path to an output directory:
```bash
python3 scripts/convert_llama_to_dcp.py <input_dir> <output_dir>
```


## How to convert a torchtitan checkpoint for use in torchtune

This guide will walk you through the steps required to convert a checkpoint from torchtitan so that it can be loaded into torchtune.

### Steps
1. ENABLE CHECKPOINTING
In your torchtitan training config, ensure that `enable_checkpoint` is set to True.
```
[checkpoint]
enable_checkpoint = true
folder = "checkpoint"
interval_type = "steps"
interval = 500
```


2. SAVE ONLY MODEL WEIGHTS
By setting `model_weights_only` to `True`, the checkpoint will only contain the model weights and exclude the optimizer state and extra train states, resulting in a smaller checkpoint size.
```
[checkpoint]
enable_checkpoint = true
model_weights_only = true
```

3. CHOOSE DESIRED EXPORT PRECISION
The default model states are in `float32`. You can choose to export the checkpoint in a lower precision format such as `bfloat16`.
```
[checkpoint]
enable_checkpoint = true
model_weights_only = true
export_dtype = "bfloat16"
```

4. EXAMPLE CHECKPOINT CONFIGURATION
```
[checkpoint]
enable_checkpoint = true
folder = "checkpoint"
interval_type = "steps"
interval = 10
load_step = 5
model_weights_only = true
export_dtype = "bfloat16"
```

5. SAVE THE FINAL CHECKPOINT\
Once the above have been set, the final checkpoint at the end of the training step will consist of model weights only with the desired export dtype. However, if the final step has not been reached yet, full checkpoints will still be saved so that training can be resumed.

6. CONVERT SHARDED CHECKPOINTS TO A SINGLE FILE\
Finally, once you have obtained the last checkpoint, you can use the following command to convert the sharded checkpoints to a single .pt file that can be loaded into torchtune:

```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch torchtitan/outputs/checkpoint/step-1000 checkpoint.pt
```

7. EXCLUDING SPECIFIC KEYS FROM CHECKPOINT LOADING
In some cases, you may want to partially load from a previous-trained checkpoint and modify certain settings, such as the number of GPUs or the current step. To achieve this, you can use the `exclude_from_loading` parameter to specify which keys should be excluded from loading.
This parameter takes a comma-separated list of keys that should be excluded from loading.
```
[checkpoint]
enable_checkpoint = true
exclude_from_loading = "data_loader,lr_scheduler"
```

That's it. You have now successfully converted a sharded torchtitan checkpoint for use in torchtune.


## How to create a seed checkpoint
Sometimes one needs to create a seed checkpoint to initialize a model from step 0.
E.g. it is hard, if not impossible, for meta initialization on multiple devices to reproduce the initialization on a single device.
A seed checkpoint does initialization of the model on a single CPU, and can be loaded from another job on an arbitrary number of GPUs via DCP resharding.

To create a seed checkpoint, use the same model config as you use for training.
e.g.
```bash
NGPU=1 CONFIG=<path_to_model_config> ./run_llama_train.sh --checkpoint.enable_checkpoint --checkpoint.create_seed_checkpoint --training.data_parallel_replicate_degree 1 --training.data_parallel_shard_degree 1 --training.tensor_parallel_degree 1 --experimental.pipeline_parallel_degree 1 --experimental.context_parallel_degree 1
```
