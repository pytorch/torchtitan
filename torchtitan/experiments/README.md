To accelerate contributions to and innovations around `torchtitan`, we are adding this new, experimental folder. Below are the general contributing guidelines, and we look forward to your contributions!

## Contributing Guidelines

We provide this `experiments/` folder to host experiments that add significant value to `torchtitan`, with the following principles. We refer to the part of `torchtitan` outside `experiments` as `core`.
1. Each subfolder in `experiments` will be an experiment, with a clear theme which can be flexible, such as
    - a new model, or preferably a new model architecture, with its training infrastructure including parallelization functions;
    - an enhancement or addition to the existing infrastructure of `torchtitan`.
2. It is the contributors' responsibility to justify the value of an experiment. `torchtitan` team will review proposals on a case-by-case basis. As part of the contribution, the contributors should provide documentation that clearly showcases the motivation and innovation of an experiment, including reports on performance and loss convergence.
3. An experiment should reuse existing `torchtitan` code as much as possible, such as modules in [`components/`](../components/) (via a new [`TrainSpec`](../protocols/train_spec.py)) and [`train.py`](../train.py). For a list of extension points we provide, please refer to [docs/extension.md](../../docs/extension.md).
    - The extension points are subject to change. We kindly request that contributors provide feedback if they encounter issues reusing any components, rather than simply using a copy-and-paste approach.
    - The degree to which existing components are reused and whether duplications are legit will also be a criteria of whether an experiment would be accepted.
4. Each experiment is independent from other experiments, and can have its own dependencies (on top of [core dependencies](../../requirements.txt)), and its own tests.
5. The dependency from `experiments` to `core` is one-way. Anything in `experiments` is optional for `core` to run successfully. In particular, development in `core` is not blocked by breakage in `experiments`. We will utilize GitHub's [CI mechanism](https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions#onpushpull_requestpull_request_targetpathspaths-ignore) to help test an experiment periodically and only if the experiment itself is affected by a PR.
6. Each experiment needs to have an owner. The owner is responsible to work with `torchtitan` team to maintain the quality and healthiness of an experiment, which includes
    - adapting an experiment to changes in `core` and fix broken tests, no later than the next official `torchtitan` release;
    - responding to GitHub issues and questions in a timely manner.
7. `torchtitan` team reserve the right to remove an experiment. In particular, an experiment should be removed if
    - it has served its purpose (e.g., providing findings, or getting some features upstreamed to `core` or PyTorch, etc.), or
    - it gets stale (e.g. not being maintained).
