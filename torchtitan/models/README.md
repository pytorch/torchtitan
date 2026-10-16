This note outlines the process of adding a new model in the `torchtitan` repo. In most cases, new models should be added first under the `torchtitan/experiments` folder. For criteria of contributions, please see the [Contributing Guidelines](/torchtitan/experiments/README.md) therein. In general, please adhere to the [Guiding Principles](/README.md#overview) of `torchtitan`.

For offline explorations, we recommend the same steps, unless otherwise noted.

## Adding the model

Please refer to the [Llama 3 folder](llama3) as an example.

The folder should be organized as follows
- `model.py`
  - NOTE: Please adhere to the guiding principles and write single-device model code.
  - NOTE: We prioritize readability over flexibility. The preferred style is to not share modules among different models, except for the most common and complicated ones.
  - Define a Model class inheriting from a base model (e.g. `Decoder` from `torchtitan/models/common/decoder.py`).
  - The model class should contain a nested `Config` dataclass (inheriting from the base model's `Config`) that holds all architecture hyperparameters.
    - `get_nparams_and_flops()` will be used to understand model size and compute throughput.
  - `__init__()` consumes the `Config` to build the model.
  - Parameter initialization is handled by the `param_init` system on each module's `Config`. Set `param_init` (a `dict[str, Callable]` mapping parameter names to init functions) on every sub-config in the model flavor builder. `init_states()` auto-recurses into all submodules, so manual recursive calls are not needed. Override `_init_self_buffers()` for device-aware buffer initialization (e.g., RoPE, MoE).
  - Add additional files to reduce the complexity of `model.py` if it grows too large or complex, e.g. moe.py to host the `MoE`, router, grouped-linear, and expert-activation modules.
- `state_dict_adapter.py`
  - Inherit [`BaseStateDictAdapter`](/torchtitan/models/base/state_dict_adapter.py) to implement state dict mappings between `torchtitan` model definition and other model definitions (e.g. from HuggingFace so that we can save / load model checkpoints in HF formats).
  - There are multiple ways such adapters could be used
    - Checkpoint conversion scripts in `scripts/checkpoint_conversion/` will use them to adapt state dicts containing non-sharded `torch.Tensor` on CPU.
    - During training, [`CheckpointManager`](/torchtitan/components/checkpointer/dcp.py) will use them to adapt state dicts containing (potentially sharded) `DTensor` on GPUs to save / load checkpoints in HF format.
    - In post-training, `to_hf()` helps convert a torchtitan model to HF model, which can be used for inference by other frameworks.
  - This is optional for offline exploration.
- `sharding.py`
  - Define declarative constructors that return the `ShardingConfig` (or `SpmdType`) of the model's own components. TP, SP, EP and inner-attention local SPMD regions are expressed this way instead of a runtime `parallelize_module` plan.
  - Reuse the shared constructors in `torchtitan/models/common/decoder_sharding.py`, `moe/sharding.py` and `vision_encoder_sharding.py` (`token_embedding_sharding_config`, `norm_sharding_config`, `residual_activation_placement`, `colwise_sharding_config`, `rowwise_sharding_config`, `moe_sharding_config`, ...) where possible; skip this file when they cover the whole model.
  - Declare the mesh axes in canonical outer-to-inner SPMD order: `(dp, cp, tp)` for dense (attention/MLP/norm/embed/lm_head) and `(dp_replicate, edp_shard, ep)` for sparse (MoE expert weights). `Module._parallelize` resolves the mesh from the declared axes.
- `model.py`
  - `BaseModel.parallelize()` applies declarative model parallelism, activation checkpointing, and FSDP/HSDP in order.
  - Override `parallelize()` only when the model needs a different lifecycle order, and override `_apply_fsdp()` when it has a model-family-specific FSDP structure.
  - Bind the state dict adapter to the model class with `state_dict_adapter_cls`.
  - Language-model CP goes through `Decoder._cp_shard`, which calls each CP
    attention backend's `prepare_cp_metadata` and then calls
    `context_parallel.shard_tensors` for the declared tensor inputs.
- `pipeline.py` (optional if model size is small)
  - apply PP
- `flavors.py`
  - Define `MODEL_FLAVORS`, which maps flavor names to their config builders and
    capability context lengths.
  - Define `build_model_config(flavor, *, parallelism=...)` to return a concrete, sharded `Model.Config`.
  - Set `sharding_config` on each component config where it is constructed, from the `parallelism` the config is built for (sequence and expert parallelism change the declared layouts). The `make_*` factories in `torchtitan/models/common/config_utils.py` take `enable_sp`/`enable_ep` and shard what they build. Mesh degrees are resolved at runtime, so axes that are not enabled are skipped.
  - A consumer with a different parallelism, such as the RL generator, builds its own model config.
- `__init__.py`
  - Re-export the model class, `MODEL_FLAVORS`, and `build_model_config` as the
    model package's public API.
  - Override the model's pipeline or optimizer hook methods only when it needs model-specific behavior.
  - See [`BaseModel`](base/model.py) for the lifecycle hooks a model can override.
- Training recipes belong in `torchtitan_recipes`, outside the model package.
  Test-only configurations belong in `torchtitan_recipes.tests`.
- `README.md`
  - Include [instructions](/README.md#downloading-a-tokenizer) to download tokenizers / encoders.
  - Include instructions to download model checkpoints for continued pretraining or post training.
  - Update the current status of development, including the supported features and coming features.
  - This is optional for offline exploration.

## Testing and Benchmarking
- Numerics testing
  - One way of doing this E2E is to load the same model checkpoint into the `torchtitan` model and the HF model, and compare the model output given the same input. This assumes
    - HF implementation is correct.
    - The correctness of a `torchtitan` model and the corresponding state dict adapter together indicates the correctness of both.
- Loss converging
  - If there is a verified baseline, compare the loss curves with the baseline.
  - For comparisons within `torchtitan`, see the [guidelines](/docs/debugging.md#validating-a-distributed-training-technique).
- Performance benchmarking
  - Please refer to the [benchmarks](/docs/benchmarks/) folder.
- CI tests
  - Including unit tests and integration tests, see [examples](/tests/).
  - If the model folder is under the experiments folder, put the tests under the model folder. Otherwise, put the tests under the `/tests` folder.
  - Add necessary GitHub [workflows](/.github/workflows/).
