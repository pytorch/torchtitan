# TorchTitanLM for Evaluation

우리는 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)을 Titan model로 직접적으로 사용하기 위해서, 새로운 Class를 구현하였다. Lingua 코드를 참고해서 implement하였다.

## 필요한 Function

lm_eval 에 있는 LM class 를 상속받은 `TorchTitanLM` class를 구현하였다. 총 세가지 method가 구현되었다: `loglikelihood`, `loglikelihood_rolling`, and `generate_until`.
쉽게, `loglikelihood` 계열은 prefill 만을 사용하여 입력받은 prompt의 continuation part에 대한 likelihood 나 accuracy를 얻는 function이다. 반면에 `generate_until` 은 실제 generation을 통해, 생성된 sequence에 대한 Metric을 측정할때 사용된다.

이미 generation 이 잘되는 것을 확인했으므로, 어떤 Task든 동작은 할 것이다. 하지만 몇몇 task와 metric에서는 potential bug가 있을수 있으므로 주의가 필요하다.

# TorchTitanLM for Evaluation

To enable the use of our Titan model directly with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework, we have implemented a new `TorchTitanLM` class, referencing the implementation for [Lingua](https://github.com/facebookresearch/lingua).


## Implemented Methods

The `TorchTitanLM` class inherits from the base [`LM`](https://github.com/EleutherAI/lm-evaluation-harness/blob/4f8195f18fbc1b6d212314509d7525e1178e036c/lm_eval/api/model.py#L24) class in `lm-eval` and implements three required methods: `loglikelihood`, `loglikelihood_rolling`, and `generate_until`.

-   The `loglikelihood` methods are used to calculate the likelihood or accuracy of a given continuation within an input prompt.
-   The `generate_until` method performs actual text generation and is used to evaluate metrics on the generated sequences.

While we have confirmed that generation works correctly, and the model should function for most tasks, please be aware that potential bugs may exist for certain tasks or metrics.