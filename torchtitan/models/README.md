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
    - `update_from_config()` updates the model config from training configs (e.g. syncing seq_len, handling hardware-specific settings).
  - `__init__()` consumes the `Config` to build the model.
  - Parameter initialization is handled by the `param_init` system on each module's `Config`. Set `param_init` (a `dict[str, Callable]` mapping parameter names to init functions) on every sub-config in the model config registry. `init_states()` auto-recurses into all submodules, so manual recursive calls are not needed. Override `_init_self_buffers()` for device-aware buffer initialization (e.g., RoPE, MoE).
  - Add additional files to reduce the complexity of `model.py` if it grows too large or complex, e.g. moe.py to host the `MoE`, `Router`, and `GroupedExperts` modules.
- `state_dict_adapter.py`
  - Inherit [`BaseStateDictAdapter`](/torchtitan/protocols/state_dict_adapter.py) to implement state dict mappings between `torchtitan` model definition and other model definitions (e.g. from HuggingFace so that we can save / load model checkpoints in HF formats).
  - There are multiple ways such adapters could be used
    - Checkpoint conversion scripts in `scripts/checkpoint_conversion/` will use them to adapt state dicts containing non-sharded `torch.Tensor` on CPU.
    - During training, [`CheckpointManager`](/torchtitan/components/checkpoint.py) will use them to adapt state dicts containing (potentially sharded) `DTensor` on GPUs to save / load checkpoints in HF format.
    - In post-training, `to_hf()` helps convert a torchtitan model to HF model, which can be used for inference by other frameworks.
  - This is optional for offline exploration.
- `sharding.py`
  - Define `set_<model>_sharding_config(config, *, enable_sp, ...)` that populates `sharding_config` on each `Module.Config` in the model config (embeddings, norms, attention, feed-forward, output). TP, SP, and inner-attention `LocalMapConfig` placements are expressed declaratively via `ShardingConfig` instead of a runtime `parallelize_module` plan.
  - Call the helper from `Model.Config.update_from_config()` so placements depend on the trainer's `parallelism` settings.
  - Reuse shared helpers from `torchtitan/models/common/decoder_sharding.py` (`set_decoder_sharding_config`, `set_dense_ffn_sharding`, `set_gqa_attention_sharding`, `norm_config`, `dense_param_placement`, `dense_activation_placement`) where possible.
  - Write the single-device module first, then express its parallelism here -- placements are always written in the `spmd_types` language (`spmd.R/I/V/P`, `spmd.S(dim)`, `PartitionSpec`), and translated into DTensor placements for the `default` and `full_dtensor` backends, so a single set of sharding configs covers all three `--parallelism.spmd_backend` choices.
  - Under `--parallelism.spmd_backend full_dtensor`, declare the mesh axes in canonical outer-to-inner SPMD order: `(dp_replicate, dp_shard, cp, tp)` for dense (attention/MLP/norm/embed/lm_head) and `(dp_replicate, efsdp, ep)` for sparse (MoE expert weights). `Module.parallelize` resolves the mesh by the declared order and validates it matches one of the SPMD meshes; declaring axes out of order raises `ValueError`.
- `parallelize.py`
  - apply training techniques in the following order
    - `model.parallelize(parallel_dims)` — auto-recursive declarative sharding driven by `sharding_config` (TP, SP, attention `local_map`). Replaces per-model `parallelize_module` plan dicts.
    - (MoE models) `apply_moe_ep_tp` for expert-parallel + TP on MoE experts (not yet config-based).
    - activation checkpointing
    - `torch.compile`
    - FSDP /  HSDP
    - NOTE: currently CP support for language models is enabled via a context manager in `torchtitan/train.py`. Ideally no extra work is needed to enable CP.
  - Register the parallelizing function as `parallelize_fn` in the model registry (see `__init__.py` below).
  - NOTE: model inputs need SPMD type annotations as well. If the existing trainers do not cover your input signature, annotate them on the trainer side; any input sharding or splitting belongs in the dataloader, not in the model.
- `pipeline.py` (optional if model size is small)
  - apply PP
- `__init__.py`
  - A dictionary of the actual model configurations, of the type `[str: Model.Config]`.
  - Define `model_registry(flavor)` to return a [`ModelSpec`](/torchtitan/protocols/model_spec.py), consisting of
    - model name and flavor
    - model config (a `Model.Config` dataclass)
    - parallelizing function, pipelining function
    - loss function builder
    - state dict adapter
  - Model name should be the same as the folder name, which should be added to `torchtitan/models/__init__.py` or ``torchtitan/experiments/__init__.py``.
  - Read [more](/docs/extension.md#modelspec) on `ModelSpec`.
- `config_registry.py`
  - Define one function for each training configuration (e.g. `llama3_debugmodel`, `llama3_8b`, `llama3_70b`).
  - Each function returns a `Trainer.Config` (or subclass) instance with all training settings.
  - Functions can derive from each other via mutation for variants (e.g. flex_attn, float8).
  - These are selected at runtime via `--module <model_name> --config <function_name>`.
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
- SPMD typechecking
  - Run with `--parallelism.spmd_backend spmd_types --debug.spmd_typechecking` during development. This invokes the global SPMD typechecker over the trainer's FWD step, catching distributed compute that is unannotated or that violates operator sharding rules. See [`spmd_types` integration](/torchtitan/distributed/SPMD_TYPES.md) for the programming model and for escape hatches when a region is not expressible in global SPMD (`LocalMapConfig`, custom autograd functions, handwritten collectives).
  - A new model should typecheck under every parallelism it claims to support, the same expectation as for DTensor-based models. Pipeline parallelism and SAC + FlexAttention are current gaps, and are rejected with a `ValueError` when typechecking is on.
  - Leave the flag off for real runs -- it adds significant overhead.
- Loss converging
  - If there is a verified baseline, compare the loss curves with the baseline.
  - For comparisons within `torchtitan`, see the [guidelines](/docs/converging.md).
- Performance benchmarking
  - Please refer to the [benchmarks](/benchmarks/) folder.
- CI tests
  - Including unit tests and integration tests, see [examples](/tests/).
  - If the model folder is under the experiments folder, put the tests under the model folder. Otherwise, put the tests under the `/tests` folder.
  - Add necessary GitHub [workflows](/.github/workflows/).
