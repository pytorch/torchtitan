# Llama 3.2 Model Class

This folder contains a Llama class that closely matches Hugging Face's [LlamaForCausalLM](https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/llama/modeling_llama.py#L475). The new Llama model is used for sanity checks on generation and few-shot evaluation features.


## Differences from Original Titan Llama 3

The original [Llama3 implementation](https://github.com/pytorch/torchtitan/tree/main/torchtitan/models/llama3) in TorchTitan has several differences compared to the Hugging Face Llama 3 class. As a result, converting Hugging Face checkpoints to Titan DCP may not perfectly reproduce the performance.


### Key Differences

* **RoPE implementation**: TorchTitan Llama 3 follows the recent implementation using `freqs_cis`, similar to HF [Llama4ForCausalLM](https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/llama4/modeling_llama4.py#L610). However, the HF Llama 3 implementation uses [different styles](https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/llama/modeling_llama.py#L112-L136). Although they perform the same operations, extra changes are needed to match the order of RoPE embeddings exactly.

* **Special RoPE Functions**: Hugging Face Llama models (both Llama 3 and Llama 4) use [special RoPE functions](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py#L384-L391). To achieve an exact match, implementing [_compute_llama3_parameters](https://github.com/huggingface/transformers/blob/5a81d7e0b388fb2b86fc1279cdc07d9dc7e84b4c/src/transformers/modeling_rope_utils.py#L340-L378) is necessary.

* **RMSNorm Difference**: Although `nn.RMSNorm` and [LlamaRMSNorm](https://github.com/huggingface/transformers/blob/a5923d4de7df2b1f373dfcfe983216b79b6937/src/transformers/models/llama/modeling_llama.py#L50-L68) are intended to work identically, a slight difference in output logits was observed.

* **Weight Tying**: Hugging Face models use weight tying for word embeddings by default. Applying weight sharing between token embedding and output layers is required.


## Converting Hugging Face Llama 3

Refer to the `../scripts` directory for detailed instructions.

1. Download Hugging Face model checkpoints and tokenizer using files under `../scripts/download_hf`.

2. Convert HF checkpoints to Titan's DCP format using scripts under `../scripts/convert_hf_to_dcp`. 

You can compare the outputs between the original HF and converted Titan models in `../scripts/sanity_check`.



## Added Arguments for Titan Llama Model

To evaluate models, we need to implement generation features first. 
However, the original TorchTitan base architecture does not support key-value caching, which prevents generation.

Therefore, we have incorporated generator and evaluator features by referencing the [Lingua](https://github.com/facebookresearch/lingua) codebase. 

- We implemented KV caching and modified the model forward function to accept additional arguments: token indices and masks.

- Our implementation uses prompt packing, enabling variable-length input sequences by concatenating multiple prompts into a single sequence.

For more details, please refer to the `../generator` and `../evaluator` modules.
