# How to use checkpointing in `torchtitan`

You may want to enable checkpointing in `torchtitan` for better fault tolerance during training, or to enable easier importing and exporting of weights between `torchtitan` and other libraries. `torchtitan` offers varying degrees of support for other checkpoint formats which are listed further below.

## A general guide to use checkpoints during training

1. ENABLE CHECKPOINTING
In your config_registry function, configure the checkpoint settings:
```python
checkpointer=CheckpointManager.Config(
    interval=500,
),
```
Checkpointing is configured in the config registry rather than on the CLI.

2. SAVE MODEL ONLY
By setting `last_save_model_only` to `True`, the checkpoint will only contain the model and exclude the optimizer state and extra train states, resulting in a smaller checkpoint size.
```python
checkpointer=CheckpointManager.Config(
    interval=500,
    last_save_model_only=True,
),
```

3. CHOOSE DESIRED EXPORT PRECISION
The default model states are in `float32`. You can choose to export the checkpoint in a lower precision format such as `bfloat16`.
```python
checkpointer=CheckpointManager.Config(
    interval=500,
    last_save_model_only=True,
    export_dtype="bfloat16",
),
```

4. EXCLUDING SPECIFIC KEYS FROM CHECKPOINT LOADING
In some cases, you may want to partially load from a previous-trained checkpoint and modify certain settings, such as the number of GPUs or the current step. To achieve this, you can use the `exclude_from_loading` parameter to specify which keys should be excluded from loading.
```python
checkpointer=CheckpointManager.Config(
    exclude_from_loading=["dataloader", "lr_scheduler"],
),
```

Turning on the weight EMA (`ema`) part-way through a run needs the same escape
hatch: the existing checkpoint has no `"ema"` key, so loading it fails until
you exclude it once, after which the EMA cold-starts from the loaded weights.
```python
checkpointer=CheckpointManager.Config(
    exclude_from_loading=["ema"],   # only for the first resume after enabling EMA
),
```
Remove it again afterwards. Left in place it cold-starts the EMA on *every*
later resume, discarding all EMA history each time (this is logged as a
warning).

5. EXAMPLE CHECKPOINT CONFIGURATION
```python
checkpointer=CheckpointManager.Config(
    interval=10,
    load_step=5,
    last_save_model_only=True,
    export_dtype="bfloat16",
),
```

A more exhaustive and up-to-date list of checkpoint config options can be found in `torchtitan/components/checkpointer/base.py` (`BaseCheckpointManager.Config`). DCP-specific `async_mode` is on `CheckpointManager.Config` in `torchtitan/components/checkpointer/dcp.py`.

## Creating a seed checkpoint
Sometimes one needs to create a seed checkpoint to initialize a model from step 0.
E.g. it is hard, if not impossible, for meta initialization on multiple devices to reproduce the initialization on a single device.
A seed checkpoint does initialization of the model on a single CPU, and can be loaded from another job on an arbitrary number of GPUs via DCP resharding.

To create a seed checkpoint, define a config registry entry with
`create_seed_checkpoint=True`, a non-`None` `checkpointer`, and every
parallelism degree set to 1. Then run that configuration on one device.

## Conversion support

### HuggingFace
`torchtitan` offers two ways to work with Hugging Face models: either by directly saving and loading a Hugging Face checkpoint during training, or by using an example conversion script to directly reformat the model weights on cpu.

1. You can directly save Hugging Face model weights during training by setting `checkpointer.last_save_in_hf` and `checkpointer.last_save_model_only` in the config registry. To directly load a `torchtitan` training session from a Hugging Face safetensors file, set `checkpointer.initial_load_in_hf`, and set either `hf_assets_path` or `checkpointer.initial_load_path` to the directory containing the Hugging Face checkpoint. `checkpointer.initial_load_path` overrides `hf_assets_path` if both are set. If `checkpointer.folder` already contains a valid checkpoint, training resumes from that folder and ignores `initial_load_in_hf` / `initial_load_path` (fault-tolerance restart). The first run (empty folder) uses the initial load.

2. To directly reformat the weights without the need to run a training loop, run the corresponding conversion script. The naming scheme is `torchtitan`-centric, e.g. convert_from_hf means convert hf->tt. `convert_ema_to_hf` exports the EMA weights instead of the trained ones, from the same checkpoint; it takes the same arguments and errors out if the checkpoint holds no EMA state. Note that `last_save_model_only` (the default) writes only model weights at the last step, so export the EMA from an interval checkpoint, or set it to `False`.

```bash
python ./scripts/checkpoint_conversion/convert_from_hf.py <input_dir> <output_dir> --model_name <model_name> --model_flavor <model_flavor>
python ./scripts/checkpoint_conversion/convert_to_hf.py <input_dir> <output_dir> --hf_assets_path ./assets/hf/Llama3.1-8B --model_name <model_name> --model_flavor <model_flavor>
python ./scripts/checkpoint_conversion/convert_ema_to_hf.py <input_dir> <output_dir> --hf_assets_path ./assets/hf/Llama3.1-8B --model_name <model_name> --model_flavor <model_flavor>
# e.g.
python ./scripts/checkpoint_conversion/convert_from_hf.py ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/ ./initial_load_path/ --model_name llama3 --model_flavor 8B
```

### Torch

This guide will walk you through the steps required to convert a checkpoint from `torchtitan` so that it can be loaded into pt format.

1. CHECKPOINT CONFIGURATION
```python
checkpointer=CheckpointManager.Config(
    interval=10,
    last_save_model_only=True,
    export_dtype="bfloat16",
),
```

2. SAVE THE FINAL CHECKPOINT\
Once the above have been set, the final checkpoint at the end of the training step will consist of model only with the desired export dtype. However, if the final step has not been reached yet, full checkpoints will still be saved so that training can be resumed.

3. CONVERT SHARDED CHECKPOINTS TO A SINGLE FILE\
Finally, once you have obtained the last checkpoint, you can use the following command to convert the sharded checkpoints to a single .pt file.

```bash
python -m torch.distributed.checkpoint.format_utils dcp_to_torch torchtitan/outputs/checkpoint/step-1000 checkpoint.pt
```


That's it. You have now successfully converted a sharded `torchtitan` checkpoint for use with pytorch formats.
