# Contributing to torchtitan
We want to make contributing to this project as easy and transparent as
possible. Contributions should follow the [Contributing Guidelines](#contributing-guidelines) below.

### Setup
```
pip install -r requirements.txt -r requirements-dev.txt
```

### Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. Keep the PR to one concern, and as small as it can be.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints (`pre-commit run --all-files`).
6. Fill in the PR template: the concrete problem with a simple example, what
   changed, numerics/performance evidence, and testing. Reviewers need to be
   convinced, not just informed.
7. Write the "Summary (human-written)" section yourself. See
   [AI-assisted contributions](#ai-assisted-contributions).
8. If you haven't already, complete the Contributor License Agreement ("CLA").

Area experts are requested as reviewers automatically from
[.github/area_reviewers.yml](.github/area_reviewers.yml). Maintainers can ask for
an AI review by commenting `@claude review`; it follows
[.claude/skills/pr-review](.claude/skills/pr-review/SKILL.md).

### AI-assisted contributions
We follow the PyTorch [AI policy](https://github.com/pytorch/pytorch/blob/main/AI_POLICY.md).
You may use AI tools, but a human who understands the change must own it.

- Write the PR's "Summary (human-written)" yourself. A few sentences are
  enough: the problem, why this approach, and what you checked yourself. PRs
  with an empty or AI-written summary may be closed or moved back to draft.
  Write the summary in the PR description on GitHub. The merged commit takes
  its message from the PR description, not from your local commit message. If
  your tool rewrites the PR description from the commit message on each push
  (stack-pr without `--keep-body`, `ghstack -u`), turn that off so the summary
  is kept.
- Other sections may be AI-drafted. You are responsible for every claim in
  them, so check each number and statement against the code and logs.
- Read and simplify the code before you request review. Open the PR as a draft
  until then.
- Don't reply to reviewers with raw AI output. If you quote AI output, put it in
  a code or quote block and add your own view of it.
- We don't accept PRs from fully autonomous agents.

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

Note: To accelerate contributions to and innovations around `torchtitan`, we are adding a new, experimental folder [`torchtitan/experiments`](torchtitan/experiments/), which has its own [Contributing Guidelines](torchtitan/experiments/README.md#contributing-guidelines). The content below is for the core portions of `torchtitan`.

### Principles of contribution

- Apply PyTorch-native training techniques.
  - The technique should be of general interest for distributed training.
  - A technique with moderate to large complexity should be sitting in the proper repo (e.g. pytorch/pytorch for a new parallelism, or pytorch/data for a new data loader) instead of `torchtitan`.
  - The main branch of `torchtitan` should have minimal dependencies on non-PyTorch libraries. Interesting models/techniques that depend on external libraries can be demonstrated in the `experiments` folder, or in forks of `torchtitan`.
- Aim for minimal (if not zero) code change to the model. For the Llama model in `torchtitan`, if one has to make justifiable model change(s):
  - After the model change, it should still load the original checkpoint correctly.
  - Document the reasons for the code change, similar to the model notes in [PIPELINE_PARALLEL.md](torchtitan/distributed/PIPELINE_PARALLEL.md).
- Keep code modularized, especially for [train.py](torchtitan/train.py), so that it remains easy to copy-paste into a minimal code example. If necessary create separate functions/files.
- Training behavior is configured by Python recipes, not by `--section.option` flags. A knob that changes the model goes in the model config dataclass. One that belongs to a component goes in that component's config, and one with no other owner goes in [configs.py](torchtitan/config/configs.py) after checking with the maintainers. See [the configuration doc](torchtitan/config/README.md).

### Proof of Value

It is the contributor’s responsibility to justify the change. The requirements include, but are not limited to

#### Loss

- If a change does not impact computation results, one should see identical loss before vs. after, with fixed random seeds (`config.debug.seed`) and deterministic algorithms (`config.debug.deterministic`). An example is activation checkpointing.
- If a change is expected to impact computation results, loss converging should be verified via end-to-end training on representable datasets (e.g. Llama 3 models on the C4 dataset). Please refer to the recommended practices in [debugging.md](docs/debugging.md#validating-a-distributed-training-technique).

#### Performance
- Memory and TPS / MFU, which are available from logging, should meet expectations.
- It is worth noting that performance expectations vary from case to case. For example, there are cases when a technique targeting memory reduction may cause throughput regression but still be acceptable (e.g. activation checkpointing). Again, it is the contributor's job to justify the feature, whether by achieving hypothetical performance, or by comparing with existing well-known implementations, etc.
- If necessary, verify the numbers on jobs spanning multiple nodes (e.g. on 64 GPUs). Please reach out to the `torchtitan` team for help if you are resource-constrained.
- When appropriate, one should show profile traces and/or memory snapshots to prove the effectiveness.

### Best practices

When appropriate, one should consider

- Adding CPU/GPU unit/integration tests that guard a real invariant. CI capacity is limited, so prefer extending an existing test or integration configuration over adding a new one, and say which regression a new test catches.
  - To add a unit test, put it in the [tests](tests/) folder and follow the existing test files.
  - To add a GPU integration test, add a configuration function and an `IntegrationTestDefinition` naming it to the matching suite module in [tests/integration_tests](tests/integration_tests/). Reusable model debug configurations live in [torchtitan_recipes/tests/models](torchtitan_recipes/tests/models/).
- Updating [README](README.md) and documenting the feature next to its code: usage in the package's `README.md` and design notes in an UPPERCASE `.md` file, similar to [MXFP8](torchtitan/quantization/mxfp8/README.md).
- Following the tensor shape-suffix naming convention for new model code (e.g. `x_BLD`, `q_BLHK`, `out_THV`), with a per-module legend comment as in [attention.py](torchtitan/models/common/attention/attention.py). Capital suffixes name logical tensor dimensions (not sharding layout) and are scoped per file.
- Adding a new file with benchmark results in [benchmarks](docs/benchmarks) folder.
- Creating GitHub issues for things that cannot be addressed at the moment.
- Writing a post on [PyTorch Forums](https://discuss.pytorch.org/c/distributed/torchtitan/44) and linking to it.
