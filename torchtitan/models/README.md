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
  - Implement `Config.set_sharding_()` to populate the model config for a
    specific consumer's `ParallelismConfig`.
  - Keep `build_model_config()` parallelism-independent. It must not accept a
    `ParallelismConfig`, parallelism degrees, or flags such as `enable_sp` and
    `enable_ep`. RL shares one structural model config between trainer and
    generator, but each actor may use different parallelism. Consumer-specific
    state belongs in `Config.set_sharding_()`.
  - Parameter initialization is handled by the `param_init` system on each module's `Config`. Set `param_init` (a `dict[str, Callable]` mapping parameter names to init functions) on every sub-config in the model flavor builder. `init_states()` auto-recurses into all submodules, so manual recursive calls are not needed. Override `_init_self_buffers()` for device-aware buffer initialization (e.g., RoPE, MoE).
  - Add additional files to reduce the complexity of `model.py` if it grows too large or complex, e.g. moe.py to host the `MoE`, router, grouped-linear, and expert-activation modules.
- `state_dict_adapter.py`
  - Inherit [`BaseStateDictAdapter`](/torchtitan/protocols/state_dict_adapter.py) to implement state dict mappings between `torchtitan` model definition and other model definitions (e.g. from HuggingFace so that we can save / load model checkpoints in HF formats).
  - There are multiple ways such adapters could be used
    - Checkpoint conversion scripts in `scripts/checkpoint_conversion/` will use them to adapt state dicts containing non-sharded `torch.Tensor` on CPU.
    - During training, [`CheckpointManager`](/torchtitan/components/checkpointer/dcp.py) will use them to adapt state dicts containing (potentially sharded) `DTensor` on GPUs to save / load checkpoints in HF format.
    - In post-training, `to_hf()` helps convert a torchtitan model to HF model, which can be used for inference by other frameworks.
  - This is optional for offline exploration.
- `sharding.py`
  - Define `set_<model>_sharding_config(config, *, enable_sp, ...)` that populates `sharding_config` on each `Module.Config` in the model config (embeddings, norms, attention, feed-forward, output). TP, SP, and inner-attention local SPMD regions are expressed declaratively via `ShardingConfig` instead of a runtime `parallelize_module` plan.
  - Call the helper from the model config's `set_sharding_()` implementation.
    Runtime consumers copy the shared structural model config and call this
    hook before applying their overrides, validation, and model build.
    This lets independent consumers, such as an RL trainer and generator,
    resolve different sharding layouts from the same model architecture.
  - Reuse shared helpers from `torchtitan/models/common/decoder_sharding.py` (`set_decoder_sharding_config`, `set_dense_ffn_sharding`, `set_gqa_attention_sharding`, `norm_config`, `dense_param_placement`, `dense_activation_placement`) where possible.
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
  - Define `build_model_config(flavor)` to return a concrete `Model.Config`.
- `__init__.py`
  - Re-export the model class, `MODEL_FLAVORS`, and `build_model_config` as the
    model package's public API.
  - Override the model's pipeline or optimizer hook methods only when it needs model-specific behavior.
  - Read [more](/docs/extension.md#models) about the model extension point.
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
  - For comparisons within `torchtitan`, see the [guidelines](/docs/converging.md).
- Performance benchmarking
  - Please refer to the [benchmarks](/benchmarks/) folder.
- CI tests
  - Including unit tests and integration tests, see [examples](/tests/).
  - If the model folder is under the experiments folder, put the tests under the model folder. Otherwise, put the tests under the `/tests` folder.
  - Add necessary GitHub [workflows](/.github/workflows/).
