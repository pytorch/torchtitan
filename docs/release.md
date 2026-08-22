# Core release process

This document describes how to stage, validate, and publish a stable
TorchTitan release.

The examples use TorchTitan `0.3.0`, PyTorch `2.14.0`, torchvision `0.29.0`,
and torchao `0.18.0`. Update all version numbers and wheel index URLs for each
release train.

## When to release

- Follow each PyTorch minor release, approximately every two months: cut the
  corresponding TorchTitan release branch shortly after PyTorch creates
  `release/X.Y`, then publish TorchTitan with the PyTorch stable release.
- For an urgent patch, publish `0.Y.(Z+1)` from the existing `release/0.Y`
  branch. A new PyTorch minor release is not required.

## Step 1 - Validate `main` and create the release branch

Complete both CI validation and release-specific testing before cutting the
branch.

### 1A - Confirm all required CI is green

Check the latest `main` or scheduled run for every applicable workflow,
including:

- lint and CPU unit tests;
- core 8-GPU model, feature, and H100 integration tests; and
- integration tests for projects under `torchtitan/experiments/`, including
  GraphTrainer, the Transformers modeling backend, RL, and TorchFT.

Path-filtered experimental workflows might not run on every `main` commit, so
check their latest scheduled run explicitly. Investigate all failures before
cutting the release branch. CI must be green, but CI status alone does not
qualify a release.

### 1B - Create a clean development environment

```bash
cd ~/torchtitan
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e . -r requirements.txt -r requirements-dev.txt
```

Install the PyTorch release-staging packages. Update the versions and CUDA
wheel index for the release being tested.

```bash
uv pip install --force-reinstall --pre \
  --index-url https://download.pytorch.org/whl/test/cu130 \
  "torch==2.14.0" \
  "torchvision==0.29.0" \
  "torchao==0.18.0"
```

For the TorchTitan `0.3.0` release only, install torchdata `0.11.0`. Remove
this special step from later release instructions as it is no longer needed.

```bash
uv pip install "torchdata==0.11.0"
```

### 1C - Run unit and smoke tests

```bash
pre-commit run --all-files
pytest tests/ -x
```

Smoke-test the example and tutorial commands documented in the repository.
Confirm that the instructions are accurate and that each selected job completes.

### 1D - Validate representative model numerics

Run real training, not only startup or import checks. The checked-in CUDA golden
losses are validated on the standard 8x A10G CI environment, so run these
comparisons on the same runner class.

Llama 3:

```bash
python3 scripts/loss_compare.py . . \
  --baseline-options="--parallelism.data_parallel_replicate_degree=1" \
  --baseline-ngpus=8 \
  --steps=100 \
  --job-dump-folder=/tmp/tt-release-214-llama3 \
  --import-result=tests/assets/losses/llama3_cuda.txt \
  --assert-equal
```

Qwen3 MoE:

```bash
python3 scripts/loss_compare.py . . \
  --baseline-module=qwen3 \
  --baseline-config=qwen3_moe_debug \
  --baseline-options="--parallelism.tensor_parallel_degree=2 --parallelism.expert_parallel_degree=4 --parallelism.spmd_backend=spmd_types --training.disable_cuda_graphs" \
  --baseline-ngpus=8 \
  --steps=100 \
  --job-dump-folder=/tmp/tt-release-214-qwen3-moe \
  --import-result=tests/assets/losses/qwen3_moe_cuda.txt \
  --assert-equal
```

### 1E - Select the version and create the branch

Use `0.Y.0rc1` for the first release candidate and `0.Y.0` for the final
release. Check the latest published version in the
[PyPI release history](https://pypi.org/project/torchtitan/#history).

```bash
git checkout main
git pull origin main
git checkout -b release/0.3
git push -u origin release/0.3
```

## Step 2 - Set the RC version

Edit [`assets/version.txt`](../assets/version.txt) to the RC version, then open
the PR against the release branch, not `main`.

Use [Semantic Versioning](https://semver.org/) for the base `0.Y.Z` version:

- increment `Y` for a feature release, for example `0.3.0` -> `0.4.0`;
- increment `Z` for a patch release containing fixes, for example `0.3.0` ->
  `0.3.1`; and
- append the Python RC suffix `rcN` for release candidates, for example
  `0.3.0rc1`, then `0.3.0rc2` if another candidate is required.

```bash
git checkout release/0.3
git pull origin release/0.3
echo "0.3.0rc1" > assets/version.txt
```

Open a PR targeting `release/0.3`, wait for CI, and merge it.

## Step 3 - Stage the RC on TestPyPI

### 3A - Update pinned release dependencies

Before staging a new release:

1. In
   [`.github/workflows/validate_rc.yaml`](../.github/workflows/validate_rc.yaml),
   update the pinned `torch`, `torchvision`, `torchao`, and `triton` versions.
   Set Triton to the version required by the selected PyTorch wheel.
2. In `.github/workflows/validate_release_gpu.yml`, update the pinned `torch`,
   `torchvision`, and `torchao` versions and the PyTorch wheel index. Triton is
   installed transitively by the GPU PyTorch wheel and is verified at runtime
   rather than pinned separately.
3. Merge both workflow updates into the release branch before staging the RC.

### 3B - Publish the RC to TestPyPI

The
[`test_release.yml`](../.github/workflows/test_release.yml) workflow builds the
wheel and source distribution, runs `twine check --strict`, and uploads the
artifacts to TestPyPI.

To run it:

1. Open **GitHub Actions -> Publish a Release to TestPyPI**.
2. Click **Run workflow**.
3. Select the release branch, for example `release/0.3`.
4. Start the workflow.

### 3C - Confirm the staged release

Confirm that the RC appears in the
[TestPyPI release history](https://test.pypi.org/project/torchtitan/#history).
Do not proceed until the package is visible and installable.

## Step 4 - Validate the RC installation

### 4A - Automatic CPU package validation

The TestPyPI staging workflow automatically invokes the `validate-rc` job after
publishing. It does not run on push or on a schedule.

For Python 3.11 and 3.12, the job creates a clean virtual environment and:

- installs the pinned `torch`, `torchvision`, `torchao`, and `triton` packages
  from the PyTorch test CPU channel;
- installs the exact TorchTitan RC from TestPyPI;
- verifies the TorchTitan version and confirms that it was imported from
  `site-packages`;
- runs a short CPU forward, backward, and optimizer smoke test; and
- verifies that the loss is finite and decreases.

Confirm that every `validate-rc` matrix job is green.

Equivalent manual installation check:

```bash
python -m pip install --pre \
  --index-url https://download.pytorch.org/whl/test/cpu \
  "torch==2.14.0" \
  "torchvision==0.29.0" \
  "torchao==0.18.0" \
  "triton==3.8.0"

python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "torchtitan==0.3.0rc1"

python -c "import torchtitan; print(torchtitan.__version__, torchtitan.__file__)"
```

### 4B - On-demand GPU RC validation

After the RC is available on TestPyPI, run the **Validate a TestPyPI Release
Candidate on GPUs** workflow from the release branch. Every job installs the
exact TorchTitan RC from TestPyPI together with the pinned PyTorch
release-staging packages.

To run it:

1. Open **GitHub Actions -> Validate a TestPyPI Release Candidate on GPUs**.
2. Click **Run workflow**.
3. Select the release branch, for example `release/0.3`.
4. Start the workflow and wait for every validation job to finish.

The workflow covers:

- `validate-standard-gpu` on 8x A10G: CUDA golden losses, models, features,
  Flux, and the standard GraphTrainer suite;
- `validate-h100` on 8x H100: the core H100 suite, including DeepSeek
  HybridEP/DeepEP; and
- `validate-graph-trainer-h100` on 8x H100: GraphTrainer DeepSeek and Qwen3 MoE
  tests that require H100-class hardware.

RL, TorchFT, and the Transformers modeling backend are not included in this RC
GPU workflow.

### 4C - Release acceptance criteria

Before finalizing the version, confirm that:

- the automatic CPU validation matrix is green for Python 3.11 and 3.12;
- every GPU validation job is green;
- logs show the exact TorchTitan RC loaded from `site-packages`;
- logs show the expected `torch`, `torchvision`, and `torchao` versions;
- the A10G golden-loss comparisons match exactly; and every model and feature test completes successfully.

If the staged package needs a code fix, follow Step 5, increment the RC version,
publish the new RC, and repeat the validation. Never reuse an RC version.

## Step 5 - Triage an RC validation failure

Identify the root cause before changing the release. The fix path depends on
whether the problem is in TorchTitan, PyTorch, torchao, or the validation
environment.

### 5A - The bug is in TorchTitan

Fix the issue on `main` first, then cherry-pick the merged fix to the release
branch.

1. Land the fix PR on `main` and record its merge commit SHA.
2. Cherry-pick the commit onto the release branch:

   ```bash
   git checkout release/0.3
   git pull origin release/0.3
   git cherry-pick <sha>
   # Resolve conflicts if necessary, then push the branch.
   git push origin release/0.3
   ```

3. Update `assets/version.txt` to the next RC, for example `0.3.0rc2`.
4. Repeat Steps 3-4 to publish and validate the new RC. Never reuse an RC
   version.

If an urgent fix cannot wait for `main` CI, open the fix PR directly against
`release/0.3`, then forward-port the identical change to `main`.

### 5B - The bug is in PyTorch

#### PyTorch has not been released yet

1. Land the fix in `pytorch/pytorch` `main`, or work with a PyTorch developer to
   land it.
2. Mark it as a release blocker for `X.Y` and request a cherry-pick into
   `pytorch/pytorch` `release/X.Y`. The release manager must approve it before
   the cherry-pick deadline.
3. When PyTorch publishes the next RC, update the pinned version and repeat
   Steps 3-4.
4. If the upstream fix cannot land in time, use a documented TorchTitan
   workaround and remove it after the upstream fix is available.

#### PyTorch is already generally available

The published wheel cannot be changed. File the upstream issue, wait for a
PyTorch patch release such as `X.Y.1`, then update the pin and revalidate. If
the release cannot wait, use a documented TorchTitan workaround.

The same policy applies to torchao issues.

## Step 6 - Finalize the version

After all CPU and GPU RC validations are green:

1. Update [`assets/version.txt`](../assets/version.txt) on the release branch:

   ```bash
   git checkout release/0.3
   git pull origin release/0.3
   echo "0.3.0" > assets/version.txt
   ```

2. Open a PR targeting `release/0.3`.
3. Confirm CI is green and merge the PR.

## Step 7 - Cut the GitHub Release and publish to PyPI

1. Open the [new release page](https://github.com/pytorch/torchtitan/releases/new).
2. Set the tag to `v0.3.0` and the target branch to `release/0.3`.
3. Click **Generate release notes**. Verify that the **Full Changelog** compares
   against the previous release tag, then organize the changes and add the
   pinned torch and torchao versions plus a short highlight summary.
4. Select **Set as a pre-release**, following the current TorchTitan convention.
5. Click **Publish**. This triggers
   [`.github/workflows/release.yml`](../.github/workflows/release.yml).
6. When prompted, approve the protected `release` environment deployment.

## Step 8 - Verify the production PyPI release

Use a clean environment and verify the actual installed wheels:

```bash
python -m venv /tmp/torchtitan-release-verify
source /tmp/torchtitan-release-verify/bin/activate
python -m pip install --upgrade pip

python -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.14.0" \
  "torchvision==0.29.0" \
  "torchao==0.18.0"

python -m pip install "torchtitan==0.3.0"
python -c "import torchtitan; print(torchtitan.__version__, torchtitan.__file__)"
```

Then:

1. Confirm the release appears in the
   [PyPI release history](https://pypi.org/project/torchtitan/#history).
2. Confirm the imported version is `0.3.0` and the package was loaded from
   `site-packages`.
3. Run a short debug-model training check against the installed PyPI wheel.
   Verify that the loss is finite and decreases. An import check alone is not
   sufficient.
