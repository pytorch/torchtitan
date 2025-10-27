## Stable Releases
Currently we follow a lightweight release process.
- Update the version number in `assets/version.txt` with a PR. The version numbering should follow https://semver.org/.
  - E.g. for a pre-release `0.y.z`
    - if major features are added, increment `y`
    - if minor fixes are added, increment `z`
- Create a new release at https://github.com/pytorch/torchtitan/releases/new
  - In the tag section, add a new tag for the release. The tag should use the version number with a `v` prefix (for example, `v0.1.0`). Make sure to select the `main` branch as the target.
  - In the release notes
    - include proper nightly versions for `torch` and `torchao`, which can be found in [latest CI](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu.yaml) test log "Run script in container" section. E.g.
        - "Successfully installed ... `torch-2.8.0.dev20250605+cu126`"
        - "Successfully installed `torchao-0.12.0.dev20250605+cu126`"
    - describe the release at a high level compared to the last release, e.g.
      - "added an experiment for multimodal LLM training"
      - or simply state "this is a regular release"
  - For now, choose "Set as a pre-release".
- As we set up the GitHub workflow [release.yml](/.github/workflows/release.yml), it should trigger a [GitHub action](https://github.com/pytorch/torchtitan/actions/workflows/release.yml) to update the [torchtitan package on PyPI](https://pypi.org/project/torchtitan/), which requires approval from one of the maintainers to run.

The general instruction on managing releases can be found [here](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).


## Nightly Builds
Nightly builds are automatically triggered by a [nightly GitHub workflow](/.github/workflows/build_whl_and_publish.yaml) and can be installed by
```bash
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu126
```
You can replace `cu126` with another version of cuda (e.g. `cu128`) or an AMD GPU (e.g. `rocm6.3`).
