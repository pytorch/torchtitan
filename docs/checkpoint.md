# How to Convert a TorchTitan Checkpoint for Use in TorchTune

This guide will walk you through the steps required to convert a checkpoint from TorchTitan so that it can be loaded into TorchTune.

## Steps
1. ENABLE CHECKPOINTING
In your TorchTitan training config, ensure that `enable_checkpoint` is set to True.
```
[checkpoint]
enable_checkpoint = true
folder = "checkpoint"
interval_type = "steps"
interval = 5
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
interval = 5
model_weights_only = true
export_dtype = "bfloat16"
```

5. SAVE THE FINAL CHECKPOINT\
Once the above have been set, the final checkpoint at the end of the training step will consist of model weights only with the desired export dtype. However, if the final step has not been reached yet, full checkpoints will still be saved so that training can be resumed.

6. CONVERT SHARDED CHECKPOINTS TO A SINGLE FILE\
Finally, once you have obtained the last checkpoint, you can use the following command to convert the sharded checkpoints to a single .pt file that can be loaded into TorchTune:

```
python -m torch.distributed.checkpoint.format_utils dcp_to_torch torchtitan/outputs/checkpoint/step-1000 checkpoint.pt
```

That's it. You have now successfully converted a sharded TorchTitan checkpoint for use in TorchTune.
