# Testing Checkpoint Conversion for Correctness

When converting checkpoints between file types or model definitions, we need to ensure that the converted checkpoints are correct, i.e. their model definition remains the same, which includes that the converted checkpoint's weights will give the same outputs when loaded in the new intended program context.

This guide provides a general framework on how to test your conversion script for correctness. The example that we will use here is bidirectional conversion between HuggingFace and `torchtitan`.

## Methods

### Sanity Check (Greedy Decode)
A quick way to sanity check if your conversion is correct is to perform greedy decoding inference on both the initial and converted checkpoints and confirm that they are the same. This method doesn't guarantee correctness but will very likely result in a fast **true negative** if the model definitions are not the same. For Llama3, greedy decoding can be achieved using the `generation/test_generate.py` script. Other models may not have an inference script, but the methodology holds the same.

Note that your model definition needs to match your conversion script. For example, if converting from `torchtitan` to HuggingFace, be sure to include the correct `config.json` file that matches the `torchtitan` model architecture. Providing an incorrect `config.json` when loading the model with HuggingFace `transformers` will result in incorrect generations despite a correct weight conversion.

### Comprehensive Check (KL Divergence)
To ensure comprehensive end-to-end correctness we recommend using KL divergence loss to compare the logits between forward passes of both the original and converted model definitions. KL divergence quantifies the "difference" between two probability distributions. A result of zero or a very low KL divergence indicates that the model definitions are equivalent. This method is crucial as it evaluates the entire probability distribution, not just the highest probability at each step.

In our `./scripts/checkpoint_conversion/example.py` this will be performing forward on dcp checkpoints loaded in `torchtitan` and safetensors checkpoints loaded in huggingface `AutoModelForCausalLM`. To convert Llama3 between HuggingFace and `torchtitan` we had to perform a permutation on several of the attention matrices to account for difference between HuggingFace and native Llama RoPE implementations. To demonstrate how a kl divergence test can reveal subtle inaccuracies such as this, we additionally compare the kl divergence between the original and converted model with and without the permutation. The results are as follows:

```
$ python ./scripts/checkpoint_conversion/example.py
Average loss for test from_hf is -4.951488641303202e-14
Average loss for test to_hf is -4.951488641303202e-14
Average loss for test from_hf_no_perm is 6.310602202574955e-06
Average loss for test to_hf_no_perm is 2.0396773834363557e-05
```
