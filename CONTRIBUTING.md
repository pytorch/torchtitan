# Contributing to torchtitan
We want to make contributing to this project as easy and transparent as
possible. Contributions should follow the [Contributing Guidelines](#contributing-guidelines) below.

### Setup
```
pip install -r dev-requirements.txt
```

### Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints (`pre-commit run --all-files`).
6. If you haven't already, complete the Contributor License Agreement ("CLA").

### Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

### Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

### License
By contributing to `torchtitan`, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

---

## Contributing Guidelines

### Principles of contribution

- Apply PyTorch-native training techniques.
  - The technique should be of general interests for distributed training.
  - A technique with moderate to large complexity should be sitting in the proper repo (e.g. pytorch/pytorch for a new parallelism, or pytorch/data for a new data loader) instead of `torchtitan`.
  - The main branch of `torchtitan` should have minimal dependency on non-PyTorch libraries. Interesting models/techniques that depend on external libraries can be demonstrated in forks of `torchtitan`.
- Aim for minimum (if not zero) code change to the model. For the Llama model in `torchtitan`, if one has to make (justifiable) model change:
  - After the model change, it should still load the original checkpoint correctly.
  - Document the reasons for the code change, similar to [composability.md](docs/composability.md).
- Keep code modularized, especially for [train.py](train.py), so that it remains easy to copy-paste into a minimal code example. If necessary:
  - Introduce new config options/category in [config_manager.py](torchtitan/config_manager.py).
  - Create separate functions/files.

### Proof of Value

It is the contributor’s responsibility to justify the change. The requirements include, but are not limited to

#### Loss

- If a change does not impact computation results, one should see identical loss before vs. after, with fixed random seeds (`training.seed`) and deterministic algorithms (`training.deterministic`). An example is activation checkpointing.
- If a change is expected to impact computation results, loss converging should be verified via end-to-end training on representable datasets (e.g. Llama 3 models on the C4 dataset). Please refer to the recommended practices in [converging.md](docs/converging.md).

#### Performance
- Memory and WPS / MFU, which are available from logging, should meet expectations.
- It is worth noting that performance expectations vary from case to case. For example, there are cases when a technique targeting memory reduction may cause throughput regression but still be acceptable (e.g. activation checkpointing). Again, it is the contributor's job to justify the feature, whether by achieving hypothetical performance, or by comparing with existing well-known implementations, etc.
- If necessary, verify the numbers on jobs spanning multiple nodes (e.g. on 64 GPUs). Please reach out to the `torchtitan` team for help if you are resource-constrained.
- When appropriate, one should show profile traces and/or memory snapshots to prove the effectiveness.

### Best practices

When appropriate, one should consider

- Adding CPU/GPU unit/integration tests.
  - To add a unit test, put it in the [test](test/) folder and follow the existing test files.
  - To add a GPU integration test, create a new `OverrideDefinitions` in [test_runner.py](test_runner.py). It will override the default config to run on the [debug model](train_configs/debug_model.toml).
- Updating [README](README.md) and writing a new note in the [docs](docs/) folder on installation and usage, similar to [float8.md](docs/float8.md).
- Updating [performance.md](docs/performance.md) with new performance results.
- Creating GitHub issues for things that cannot be addressed at the moment.
- Writing a post on [PyTorch Dev Discussions](https://dev-discuss.pytorch.org/c/distributed/6) forum and linking to it.
