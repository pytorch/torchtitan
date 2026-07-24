## MXFp8 Training on (G)B200 Blackwell - Experimental but available now for testing

MXFP8 training can provide substantial training speedups for models running on Nvidia Blackwell architecture (G and B200s+).  MX FP8 enables fine grained quantization, where 1 x 32 elements are quantized per a single U8ME0 scaling, and this scaling can be done via hardware.

We have tested MXFP8 training at 1856 GPU Scale (Crusoe B200 cluster) and for Llama 3 70B model, we observed ~ 19% speedup with near equal or better convergence loss relative to BF16.

Note that the 19% speedup is a baseline atm - we have work to do to immprove the performance due to bank conflicts in the current kernels.


### Usage steps

Please install latest nightly [TorchAO](https://github.com/pytorch/ao/tree/main/torchao/float8) to support mxfp8 dtype
```
USE_CPP=0 python -m pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu128
```

For mxfp8 with 1x32 scaling, launch training job with the following command (or alternatively set configs in toml files)
```
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" ./run_train.sh --model.converters mx --mx.recipe_name "mxfp8" --training.compile
```
* `--model.converters="mx"`: use mx section for converting the linear layers
* `--mx.recipe_name "mxfp8"`: swap nn.Linears from high precision to mxfp8 for internal computation.
* `--training.compile` (required for competitive performance): use `torch.compile` to fuse the mxfp8 scaling/casting kernels
