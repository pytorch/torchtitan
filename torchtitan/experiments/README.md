To accelerate contributions to and innovations around `torchtitan`, we are adding this new, experimental folder. Below are the general contributing guidelines, and we look forward to your contributions!

## Contributing Guidelines

We provide this `experiments/` folder to host experiments that add significant value to `torchtitan`, with the following principles. We refer to the part of `torchtitan` outside `experiments` as `core`.
1. Each subfolder in `experiments` will be an experiment, with a clear theme which can be flexible, such as
    - A new model, or preferably a new model architecture, with its training infrastructure including parallelization functions. Please see the [instructions](/torchtitan/models/README.md) on how to contribute a new model.
    - An enhancement or addition to the existing infrastructure of `torchtitan`.
2. It is the contributors' responsibility to justify the value of an experiment. `torchtitan` team will review proposals on a case-by-case basis. As part of the contribution, the contributors should provide documentation that clearly showcases the motivation and innovation of an experiment, including reports on performance and loss convergence.
3. An experiment should reuse existing `torchtitan` code as much as possible, such as modules in [`components/`](../components/) (via a new [`ModelSpec`](../protocols/model_spec.py)) and [`train.py`](../train.py). For a list of extension points we provide, please refer to [docs/extension.md](../../docs/extension.md).
    - The extension points are subject to change. We kindly request that contributors provide feedback if they encounter issues reusing any components, rather than simply using a copy-and-paste approach.
    - The degree to which existing components are reused and whether duplications are legit will also be a criteria of whether an experiment would be accepted.
4. Each experiment is independent from other experiments, and can have its own dependencies (on top of [core dependencies](../../requirements.txt)), and its own tests. An experiment should not contain vendor-specific code, such as kernels written in a proprietary language. Those can be hosted outside as dependency.
5. The dependency from `experiments` to `core` is one-way. Anything in `experiments` is optional for `core` to run successfully. In particular, development in `core` is not blocked by breakage in `experiments`. We will utilize GitHub's [CI mechanism](https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions#onpushpull_requestpull_request_targetpathspaths-ignore) to help test an experiment periodically and only if the experiment itself is affected by a PR.
6. Each experiment needs to have an owner. The owner is responsible to work with `torchtitan` team to maintain the quality and healthiness of an experiment, which includes
    - adapting an experiment to changes in `core` and fix broken tests, no later than the next official `torchtitan` release;
    - responding to GitHub issues and questions in a timely manner.
7. `torchtitan` team reserve the right to remove an experiment. In particular, an experiment should be removed if
    - it has served its purpose (e.g., providing findings, or getting some features upstreamed to `core` or PyTorch, etc.), or
    - it gets stale (e.g. not being maintained).


## Current experiments

| Experiment | Test Status | Owners |
| ----- | ----: | ----: |
| [graph_trainer](./graph_trainer/) | [![GraphTrainer 8 GPU Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml?query=branch%3Amain) [![GraphTrainer H100 8 GPU Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer_h100.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer_h100.yaml?query=branch%3Amain) | [@ruisizhang123](https://github.com/ruisizhang123) [@SherlockNoMad](https://github.com/SherlockNoMad) [@yiming0416](https://github.com/yiming0416) |
| [autoparallel](./autoparallel/) | [![Auto Parallel 8 GPU Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_autoparallel.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_autoparallel.yaml?query=branch%3Amain) | [@wconstab](https://github.com/wconstab) [@xmfan](https://github.com/xmfan) |
| [ft](./ft/) | [![TorchFT 8 GPU Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_torchft.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_torchft.yaml?query=branch%3Amain) | [@tushar00jain](https://github.com/tushar00jain) [@fegin](https://github.com/fegin) |
| [vlm](./vlm/) | [![VLM 8 GPU Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_vlm.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_vlm.yaml?query=branch%3Amain) | [@lkhphuc](https://github.com/lkhphuc) [@shuhuayu](https://github.com/shuhuayu) |
| [transformers_modeling_backend](./transformers_modeling_backend/) | [![Transformers modeling backend 8 GPU Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_transformers_modeling_backend.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_transformers_modeling_backend.yaml?query=branch%3Amain) | [@3outeille](https://github.com/3outeille) |
| [rl](./rl/) | [![RL Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_4gpu_rl.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_4gpu_rl.yaml?query=branch%3Amain) | [@wwwjn](https://github.com/wwwjn) |
| [forge](./forge/) | TBA | [@allenwang28](https://github.com/allenwang28) [@joecummings](https://github.com/joecummings) [@felipemello1](https://github.com/felipemello1) [@daniellepintz](https://github.com/daniellepintz) |
