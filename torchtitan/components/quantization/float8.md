## Float8 Training on H100s

Float8 training can provide substantial training speedups for models where the majority of GEMMs are sufficiently large enough that the speedup from using float8 tensorcores outweighs the overhead of dynamic quantization. See [here](https://github.com/pytorch/ao/tree/main/torchao/float8#performance) for microbenchmarks detailing the observed speedups for the forward + backward pass of a simple "layer norm => linear => sigmoid" model for different M,N,K sizes, which can help you determine if your model can benefit from float8 training. Note you can also use float8 training for only the subset of layers which will benefit from it by using the `filter_fqns` argument.

### Usage steps

Please install latest [TorchAO](https://github.com/pytorch/ao/tree/main/torchao/float8) to support float8 dtype
```
USE_CPP=0 python -m pip install git+https://github.com/pytorch/ao.git
```

Quantization is applied at config time in your `model_registry()` function via the `quantization` parameter. Each converter walks the model config tree and swaps config types so that quantized modules are built directly.

For float8 with rowwise scaling, configure it in your config_registry function:
```python
from torchtitan.components.quantization import Float8LinearConverter

# In your model_registry call:
model_spec = model_registry(
    "405B",
    quantization=[
        Float8LinearConverter.Config(
            recipe_name="rowwise",
            filter_fqns=["output"],
            model_compile_enabled=True,
        ),
    ],
)
```
* `recipe_name`: Float8 recipe name. Options: `"rowwise"` (default), `"rowwise_with_gw_hp"`.
* `filter_fqns` (optional): a list of fully qualified names of modules not to convert to float8 training. Example: `filter_fqns=["attention.wk", "attention.wv"]`. You can determine which layers to convert by looking at the microbenchmarks in the [performance section](https://github.com/pytorch/ao/tree/main/torchao/float8#performance) of the torchao documentation for the float8 recipe you're using.
    * **Auto-filter**: add `"auto_filter_small_kn"` as one of the `filter_fqns` to enable automatic module filtering, which will automatically not convert linear layers that are not large enough to benefit from float8 training, since the GEMM has to be big enough that the speedup from using FP8 tensorcores is greater than the overhead of creating dynamically quantized inputs. The thresholds for conversion are based on microbenchmarks measured on NVIDIA H100 GPUs, where (K,N) represents the linear layer weight shape. For best performance, you should still manually filter out layers that are too small to benefit from float8 training.
* `model_compile_enabled`: set to `True` when `torch.compile` is enabled for the model (required for competitive performance). `torch.compile` fuses the float8 scaling/casting kernels.

For float8 MoE expert quantization (grouped GEMMs), use `Float8GroupedExpertsConverter`:
```python
from torchtitan.components.quantization import Float8LinearConverter, Float8GroupedExpertsConverter

model_spec = model_registry(
    "671B",
    quantization=[
        Float8LinearConverter.Config(
            filter_fqns=["output", "router.gate"],
            model_compile_enabled=True,
        ),
        Float8GroupedExpertsConverter.Config(
            model_compile_enabled=True,
        ),
    ],
)
```

For parallelisms, for float8 with rowwise scaling, all distributed communication is done in high precision.

For scaling strategy, we support rowwise dynamic scaling (alpha).
