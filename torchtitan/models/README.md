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
    - `build_flops_estimator()` builds the model-owned callback used to estimate the work of each input batch. See [Batch FLOP estimation](#batch-flop-estimation).
    - `update_from_config()` updates the model config from training configs (e.g. syncing seq_len, handling hardware-specific settings).
  - `__init__()` consumes the `Config` to build the model.
  - Parameter initialization is handled by the `param_init` system on each module's `Config`. Set `param_init` (a `dict[str, Callable]` mapping parameter names to init functions) on every sub-config in the model config registry. `init_states()` auto-recurses into all submodules, so manual recursive calls are not needed. Override `_init_self_buffers()` for device-aware buffer initialization (e.g., RoPE, MoE).
  - Add additional files to reduce the complexity of `model.py` if it grows too large or complex, e.g. moe.py to host the `MoE`, `Router`, and `GroupedExperts` modules.
- `state_dict_adapter.py`
  - Inherit [`BaseStateDictAdapter`](/torchtitan/protocols/state_dict_adapter.py) to implement state dict mappings between `torchtitan` model definition and other model definitions (e.g. from HuggingFace so that we can save / load model checkpoints in HF formats).
  - There are multiple ways such adapters could be used
    - Checkpoint conversion scripts in `scripts/checkpoint_conversion/` will use them to adapt state dicts containing non-sharded `torch.Tensor` on CPU.
    - During training, [`CheckpointManager`](/torchtitan/components/checkpointer/dcp.py) will use them to adapt state dicts containing (potentially sharded) `DTensor` on GPUs to save / load checkpoints in HF format.
    - In post-training, `to_hf()` helps convert a torchtitan model to HF model, which can be used for inference by other frameworks.
  - This is optional for offline exploration.
- `sharding.py`
  - Define `set_<model>_sharding_config(config, *, enable_sp, ...)` that populates `sharding_config` on each `Module.Config` in the model config (embeddings, norms, attention, feed-forward, output). TP, SP, and inner-attention local SPMD regions are expressed declaratively via `ShardingConfig` instead of a runtime `parallelize_module` plan.
  - Call the helper from `Model.Config.update_from_config()` so placements depend on the trainer's `parallelism` settings.
  - Reuse shared helpers from `torchtitan/models/common/decoder_sharding.py` (`set_decoder_sharding_config`, `set_dense_ffn_sharding`, `set_gqa_attention_sharding`, `norm_config`, `dense_param_placement`, `dense_activation_placement`) where possible.
  - Declare the mesh axes in canonical outer-to-inner SPMD order: `(dp, cp, tp)` for dense (attention/MLP/norm/embed/lm_head) and `(dp_replicate, efsdp, ep)` for sparse (MoE expert weights). `Module._parallelize` resolves the mesh from the declared axes.
- `model.py`
  - `BaseModel.parallelize()` applies declarative model parallelism, activation checkpointing, `torch.compile`, and FSDP/HSDP in order.
  - Override `parallelize()` only when the model needs a different lifecycle order, and override `_apply_fsdp()` when it has a model-family-specific FSDP structure.
  - Language-model CP goes through `Decoder.preprocess_inputs` -> `prepare_context_parallel_input`.
- `pipeline.py` (optional if model size is small)
  - apply PP
- `__init__.py`
  - A dictionary of the actual model configurations, of the type `[str: Model.Config]`.
  - Define `model_registry(flavor)` to return a concrete `Model.Config`.
  - Bind the state dict adapter to the model class with `state_dict_adapter_cls`.
  - Override the model's pipeline or optimizer hook methods only when it needs model-specific behavior.
  - Model name should be the same as the folder name, which should be added to `torchtitan/models/__init__.py` or ``torchtitan/experiments/__init__.py``.
  - Read [more](/docs/extension.md#models) about the model extension point.
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

## Batch FLOP estimation

A model config implements `build_flops_estimator(model, *, seq_len)` and
returns a `FlopsEstimator`. The factory may inspect the full meta-device model
once. Its returned callback has this contract:

```python
def estimate_flops(batch: Mapping[str, Any]) -> int:
    ...
```

- `batch` is the raw, unsharded, DP-rank-local input mapping on CPU. The
  callback runs before device transfer, model preprocessing, or context
  parallel sharding.
- The result is a Python `int` containing model-wide logical training FLOPs for
  that batch.
- Capture only precomputed Python scalars in the callback. Do not capture the
  model, parameters, tensors, or bound methods.
- Keep each invocation CPU-only and O(batch metadata). Inspect shapes and small
  metadata such as `grid_thw`, but do not transfer data, call `.item()`, run a
  collective, or enter model execution.
- Access required fields directly. A missing field should raise `KeyError`
  instead of silently guessing a substitute.

Every trainer calls `TrainingEngine.estimate_flops(raw_batch)`. Fixed-shape
decoder models should multiply their cached per-token estimate by
`batch["input"].numel()`. Other workloads should
use the raw CPU field that actually determines their work rather than fabricate
an `input` dependency.

Decoder model configs compose `flops_per_token()` directly from the attention
config owned by each block. This does not require a shared transformer-block base
class.

`BaseModel.Config.get_parameter_counts(model)` provides the default model-structure
accounting path. Backends with a different pre-parallel module representation may
override it while preserving the same total/active semantics.

`get_parameter_counts()` reports architectural total and active parameter
counts. It includes embedding tables and relies on PyTorch's parameter iterator
to count shared parameters only once. `active_parameter_flops_per_unit()` is a
separate compute-oriented helper: it excludes embedding lookups and derives the
conventional `6P` component for matrix-multiplication parameters. Both helpers
weight routed-expert parameters by the expected active fraction
`top_k / num_experts`; this is an expected-use estimate, not an observation of
the router's choices for that batch. A model may instead consume compact CPU
routing metadata when such metadata is already part of the raw batch and is
needed to represent input-dependent work. It must not read routing results back
from an accelerator.

For multimodal models, keep scaling units disjoint. For example, partition
parameter work into per-text-token, per-input-patch or frame, and
per-vision-output-token terms, then add attention contractions separately. For
block-diagonal attention over independently packed examples, count
`sum(length_i**2)`, not `sum(length_i)**2`; the latter incorrectly introduces
cross-example attention. `get_packed_vision_grids()` centralizes selecting
present modalities and traversing their CPU grid metadata once. Each vision
encoder config implements `build_vision_flops_estimator()` because attention segmentation, temporal
pooling, spatial merging, and sparse attention are encoder semantics. The model
config composes that estimator with text and any modules outside the encoder.
Tests should cover fixed-path parity, variable and optional
modalities, closure release, and invocation before device transfer or sharding.

For reporting-window aggregation and TFLOPS/MFU semantics, see
[FLOPs, throughput, and MFU](../../docs/metrics.md#flops-throughput-and-mfu).

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
