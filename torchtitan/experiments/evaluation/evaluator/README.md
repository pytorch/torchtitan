# TorchTitanLM for Evaluation

To enable the use of our Titan model directly with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework, we have implemented a new `TorchTitanLM` class, referencing the implementation for [Lingua](https://github.com/facebookresearch/lingua).


## Implemented Methods

The `TorchTitanLM` class inherits from the base [`LM`](https://github.com/EleutherAI/lm-evaluation-harness/blob/4f8195f18fbc1b6d212314509d7525e1178e036c/lm_eval/api/model.py#L24) class in `lm-eval` and implements three required methods: `loglikelihood`, `loglikelihood_rolling`, and `generate_until`.

-   The `loglikelihood` methods are used to calculate the likelihood or accuracy of a given continuation within an input prompt.
-   The `generate_until` method performs actual text generation and is used to evaluate metrics on the generated sequences.

While we have confirmed that generation works correctly, and the model should function for most tasks, please be aware that potential bugs may exist for certain tasks or metrics.
