# TorchTitan Generation Features

This repository includes the KV caching and text generation functionalities implemented for Transformer models in TorchTitan, referring to [Lingua](https://github.com/facebookresearch/lingua) project.


## Key Features

### Implementation Overview

Our generation implementation is built around input packing and utilizes distinct attention mechanisms for the prefill and decode phases. The core features include:

1.  **Static KV Cache**: We employ a static KV cache, pre-allocating its size based on a maximum sequence length. This cache is incrementally replaced during the generation process and is compatible with `torch.compile`.

2.  **Input Packing**: Multiple prompts are packed into a single sequence. This technique allows for processing multiple requests in a single batch.

3.  **Prefill Phase**: For the initial processing of prompts (prefill), we leverage [FlexAttention](https://pytorch.org/blog/flexattention/). By using block-causal masking, we can process all packed prompts in a single, parallel computation.

4.  **Decode Phase**: In the subsequent token generation (decode) phase, we switch to [Scaled Dot-Product Attention (SDPA)](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html).

To support this packed inference, model components now accept a `tok_idx` argument. This tensor provides the necessary indexing to correctly apply attention masks for each token within the packed sequence.



### How to Test

#### 1. Interactive Generation

You can test text generation with your own prompts using `transformer.py`.

```bash
# Replace {model_type} with the model architecture and {checkpoint_path} with the name of your checkpoint directory.
python transformer.py --model {model_type} --exp_name {checkpoint_path}

# Example: python transformer.py --model llama3_2_1b --exp_name llama_3.2_1b_dcp
```

#### 2. Performance Benchmark

To measure generation throughput and latency, use the `generation_benchmark.py` script.

```bash
# Set the model, checkpoint, and max sequence length for the benchmark.
python generation_benchmark.py --model {model_type} --exp_name {checkpoint_path} --max_tokens {max_sequence_length}

# Example: python generation_benchmark.py --model llama3_2_1b --exp_name llama_3.2_1b_dcp --max_tokens 2048
```

### Generation Example

**Prompt:** "torchtitan is currently in a pre-release state and under extensive development. We showcase training Llama 3.1 LLMs at scale, and are working on other types of generative AI models, including"

**Generated Tokens:** "Llama 3.2 LLMs. We are also working on a new Llama 3.3 LLM, which will be released in the coming months. We are also working on a new Llama 3.4 LLM, which will be released in the coming months ..."