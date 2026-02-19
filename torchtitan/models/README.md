This note outlines the process of adding a new model in the `torchtitan` repo. In most cases, new models should be added first under the `torchtitan/experiments` folder. For criteria of contributions, please see the [Contributing Guidelines](/torchtitan/experiments/README.md) therein. In general, please adhere to the [Guiding Principles](/README.md#overview) of `torchtitan`.

For offline explorations, we recommend the same steps, unless otherwise noted.

## Adding the model

Please refer to the [Llama 3 folder](llama3) as an example.

The folder should be organized as follows
- `model` folder: a self-contained folder of model definition
  - `model.py`
    - NOTE: Please adhere to the guiding principles and write single-device model code.
    - NOTE: We prioritize readability over flexibility. The preferred style is to not share modules among different models, except for the most common and complicated ones.
    - Define a Model class inheriting from a base model (e.g. `Decoder` from `torchtitan/models/common/decoder.py`).
    - The model class should contain a nested `Config` dataclass (inheriting from the base model's `Config`) that holds all architecture hyperparameters.
      - `get_nparams_and_flops()` will be used to understand model size and compute throughput.
      - `update_from_config()` updates the model config from training configs (e.g. syncing seq_len, handling hardware-specific settings).
    - `__init__()` consumes the `Config` to build the model.
    - `init_weights()` is used to properly initialize the parameters and buffers in the model. Please define it in a recursive way so that every submodule has its own `init_weights()`.
    - Add additional files to reduce the complexity of `model.py` if it grows too large or complex, e.g. moe.py to host the `MoE`, `Router`, and `GroupedExperts` modules.
  - `state_dict_adapter.py`
    - Inherit [`BaseStateDictAdapter`](/torchtitan/protocols/state_dict_adapter.py) to implement state dict mappings between `torchtitan` model definition and other model definitions (e.g. from HuggingFace so that we can save / load model checkpoints in HF formats).
    - There are multiple ways such adapters could be used
      - Checkpoint conversion scripts in `scripts/checkpoint_conversion/` will use them to adapt state dicts containing non-sharded `torch.Tensor` on CPU.
      - During training, [`CheckpointManager`](/torchtitan/components/checkpoint.py) will use them to adapt state dicts containing (potentially sharded) `DTensor` on GPUs to save / load checkpoints in HF format.
      - In post-training, `to_hf()` helps convert a torchtitan model to HF model, which can be used for inference by other frameworks.
    - This is optional for offline exploration.
- `infra` folder: containing the functions used to parallelize the model using PyTorch native techniques
  - `parallelize.py`
    - apply training techniques in the following order
      - TP (and EP if the model has MoE architecture)
      - activation checkpointing
      - `torch.compile`
      - FSDP /  HSDP
      - NOTE: currently CP support for language models is enabled via a context manager in `torchtitan/train.py`. Ideally no extra work is needed to enable CP.
  - `pipeline.py` (optional if model size is small)
    - apply PP
  - Include other util files if necessary.
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
- Loss converging
  - If there is a verified baseline, compare the loss curves with the baseline.
  - For comparisons within `torchtitan`, see the [guidelines](/docs/converging.md).
- Performance benchmarking
  - Please refer to the [benchmarks](/benchmarks/) folder.
- CI tests
  - Including unit tests and integration tests, see [examples](/tests/).
  - If the model folder is under the experiments folder, put the tests under the model folder. Otherwise, put the tests under the `/tests` folder.
  - Add necessary GitHub [workflows](/.github/workflows/).
