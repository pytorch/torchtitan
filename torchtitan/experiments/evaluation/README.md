# Evaluation Feature for TorchTitan

## What Is This For?

There was an [RFC](https://github.com/pytorch/torchtitan/issues/1210) to add evaluation for Titan pretrained models. Regarding after-training evaluation, there were two suggested solutions. The first was to convert Titan models to HuggingFace models and then evaluate them, since HuggingFace models are already well-supported in the lm-evaluation-harness framework.

Another solution (indicated as "solution 3" in the RFC) is to integrate `lm_eval` with TorchTitan, following the `lm_eval` guide on [External Library Usage](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage). This requires implementing wrapper classes around TorchTitan models with the specific interfaces required by `lm_eval`. This approach allows us to leverage full n-D parallelism support.

This repository implements **solution 3** and details the necessary steps to evaluate Titan models. Although not perfect, this experimental repository should benefit others who want to evaluate their pretrained models using the TorchTitan framework.


## Version Setup

To match the versions we tested, run the following commands:

```bash
conda create -n titan python=3.12
pip install -r requirements.txt

# Make sure that PyTorch is a Nightly version.
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

pip install transformers

# We recommend cloning lm-eval into a separate directory.
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```


## Evaluation Procedures

To integrate `lm-eval` with TorchTitan and verify performance, we implemented the following features. Please refer to the `README.md` in each subdirectory for more details:

1.  **Converting HF to Titan:** First, we convert HuggingFace models to TorchTitan models. This is necessary to compare the evaluation results between HF-based and Titan-based models. We used Llama 3.2 1B and 3B models as examples. The `llama3` directory contains a perfectly matched Titan Llama class. The `scripts` folder contains utilities to download HF models, convert HF checkpoints to DCP format, and perform a sanity check to ensure their outputs are identical.

2.  **Generation Features:** For few-shot evaluation tasks, generation capabilities are required for TorchTitan models. First, we implemented a Transformer model in the `llama3` directory that integrates a KV caching mechanism using a static cache. Then, referencing the [Lingua](https://github.com/facebookresearch/lingua) codebase, we implemented a generation protocol for packed inputs. This framework uses [FlexAttention](https://pytorch.org/blog/flexattention/) during the prefill phase to pack variable-length sequences and process them in parallel with block-causal masking. During the decode phase, it switches back to SDPA to generate the next token for each sequence in parallel.

3.  **Evaluation Features:** We implemented wrapper classes for TorchTitan models to be used with `lm-eval`. We wrapped the previously implemented generator into a class containing the functions required by `lm-eval`. Using this, we observed that the evaluation results from the Titan-based implementation were identical to those from the original HuggingFace-based model.


## Comparison Between HF and Titan Evaluation Results

### Pre-requisites to Reproduce Results

Refer to the `README.md` file in the `scripts` folder to convert HF checkpoints to Titan DCP.
Then, carefully update your settings accordingly:
- Update `tokenizer_path` in the following configuration files:
  - `llama3/train_configs/llama3.2_1b.toml`
  - `llama3/train_configs/llama3.2_3b.toml`
- Modify the following variables in `generator/utils.py`:
  - `PROJECT_ROOT`
  - `SAVE_ROOT`
  - `job_config.training.dataset_path` (at line #L61)
- In `scripts/sanity_check/compare_hf_titan_models.ipynb`, update:
  - `PROJECT_ROOT` and `SAVE_ROOT` in Cell #1
  - `job_config.training.dataset_path` in Cell #4


### Running Command

For the HuggingFace Llama 3.2 1B model, you can obtain the results from `lm-eval` with the following command:
```bash
lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.2-1B,dtype="bfloat16",add_bos_token=True
        --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,openbookqa
        --device cuda:0
        --batch_size 8
```

For the TorchTitan model, you can run the evaluation with the following command:
```bash
bash evaluation_fewshot.sh llama3_2_1b llama_3.2_1b_dcp
# Or: python evaluate_fewshot.py --model_args dtype=bf16,model_type_size={model_type},exp_name={exp_name},compile_prefilling=False,reduce_generation_overhead=False \
#                               --tasks lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,openbookqa \
#                               --device cuda:0 \
#                               --batch_size 8
```


### Evaluation Results of Llama 3.2 1B

The evaluation results for this implementation are as follows.

| Type | LAMBADA | HellaSwag | PIQA | ARC easy | ARC challenge | OpenBookQA |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **HF** | 62.2% | 64.3% | 74.9% | 61.6% | 36.6% | 37.0% |
| **Titan** | 62.2% | 64.2% | 74.8% | 61.8% | 37.1% | 36.6% |


There were some minor discrepancies. We found that these differences arise from the use of `FlexAttention` in TorchTitan versus `SDPA` in the HuggingFace implementation. We observed very small differences, typically around the 7th decimal place. For more details, please refer to the notebook file in `scripts/sanity_check`.


## Next Steps

This repository has several limitations that should be addressed in future work:

1.  **Single-GPU Support Only:** The current `lm-eval` framework supports multi-GPU evaluation via `accelerate` for HF models. While our implementation supports n-D parallelism for the Titan model itself, its integration with data sharding has not been verified. Thereby, the code is currently restricted to a single GPU, and multi-GPU support needs to be implemented.

2.  **Handling of `add_bos_token`:** A more robust implementation for the `add_bos_token` argument is needed for models like Llama and Gemma that use a BOS token. During our validation with the Llama 3.2 model, we hardcoded the logic to always use a BOS token and adjusted prompt lengths accordingly to ensure accurate metrics. This part requires further investigation and generalization.

3.  **Debugging for Certain Tasks:** Some tasks, such as Wikitext, require debugging. We referenced the Lingua implementation and have encountered some of the same issues present in that codebase. For example, an [issue](https://github.com/facebookresearch/lingua/issues/46) has been reported regarding incorrect metric calculations for the Wikitext benchmark. Rigorous validation is needed for other benchmarks as well.

