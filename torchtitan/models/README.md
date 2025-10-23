This note outlines the process of adding a new model in the `torchtitan` repo. In most cases, new models should be added first under the `torchtitan/experiments` folder. For criteria of contributions, please see the [Contributing Guidelines](/torchtitan/experiments/README.md) therein. In general, please adhere to the [Guiding Principles](/README.md#overview) of `torchtitan`.

For offline explorations, we recommend the same steps, unless otherwise noted.

## Adding the model

Please refer to the [Llama 3 folder](llama3) as an example.

The folder should be organized as follows
- `model` folder: a self-contained folder of model definition and args
  - `args.py`
    - Inherit [`BaseModelArgs`](/torchtitan/protocols/model.py) and implement the interfaces.
      - `get_nparams_and_flops()` will be used to understand model size and compute throughput.
      - `update_from_config()` updates the model args from training configs. To extend training configs, see the bullet point below on `job_config.py`.
  - `model.py`
    - NOTE: Please adhere to the guiding principles and write single-device model code.
    - NOTE: We prioritize readability over flexibility. The preferred style is to not share modules among different models, except for the most common and complicated ones.
    - Inherit [`ModelProtocol`](/torchtitan/protocols/model.py) and implement the interfaces.
      - `__init__()` consumes a `ModelArgs` input to build the model
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
  - A dictionary of the actual model configurations, of the type `[str: ModelArgs]`.
  - Define `get_train_spec` to return a [`TrainSpec`](/torchtitan/protocols/train_spec.py), consisting a tuple of
    - model class, model args
      - Model name should be the same as the folder name, which should be added to `torchtitan/models/__init__.py` or ``torchtitan/experiments/__init__.py``.
    - parallelizing function, pipelining function
    - builder functions for optimizer, lr scheduler, data loader, tokenizer, and loss function
      - More often than not, existing components can be reused.
      - Adding new datasets requires the `torchtitan` team’s review and legal approval.
      - Try to have minimal dependency on external libraries, if any.
    - state dict adapter
  - If developing outside of torchtitan, one can call `register_train_spec` to register a `TrainSpec` so that `train.py` can be reused.
  - Read [more](/docs/extension.md#trainspec) on `TrainSpec`.
- `README.md`
  - Include [instructions](/README.md#downloading-a-tokenizer) to download tokenizers / encoders.
  - Include instructions to download model checkpoints for continued pretraining or post training.
  - Update the current status of development, including the supported features and coming features.
  - This is optional for offline exploration.
- `job_config.py` (if necessary)
  - Sometimes a new model needs to access additional configs, to be consumed by various training components. Read the [guidance](/docs/extension.md#train-script) on extending `JobConfig`.
- `train.py` (only if absolutely necessary)
  - Sometimes `torchtitan/train.py` may not be enough to run the model. There is a [tradeoff](/docs/extension.md#train-script) between extending the existing one vs. having a new one.
  - Even if a new one needs to be added, it should reuse `torchtitan/train.py` as much as possible. See `torchtitan/experiments/flux/train.py` as an example.
- `train_configs` folder
  - There should be one `.toml` file for each model variant (e.g. Llama 3.1 8B / 70B / 405B) as well as a `debug_model.toml`.
  - They should be verified with real training jobs, in terms of optimized throughput and loss converging.

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
