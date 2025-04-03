# Running DeepSeek in Titan  (experimental)

This folder contains a DeepSeek model supporting v2 and v3 as well as kernels
and scripts needed to run it.

## Inference

### Prerequisites:

You will need to download a DeepSeek model's weights if you want to run a
pre-trained checkpoint.  We provided a script to download the weights from
HuggingFace Model Hub:
```bash
python download.py [vX]
```
where `vX` can be v2 or v3, both are supported. You may be required to create a
HuggingFace account and log in first.

### Running inference:

The inference script is in `generate.py`. You can run it with the following
command:
```bash
torchrun --standalone --nproc-per-node 4 generate.py
```
This will run inference on the `DeepSeek-V2-Lite-Chat` model using 4 GPUs by
default.

Alternatively, you can run inference by using `bash inference.sh`, optionally
followed by your prompt.

## Training

The training script is in `train.py`. You can run it by the following command:
```bash
torchrun --standalone --nproc-per-node 8 train.py
```

This will run training on the `DeepSeek-V2-Lite-Chat` model using 8 GPUs by
default, with pipeline parallel, expert parallel, and data parallel enabled.
