# DeepSeek V3 671B DistMoE MTP1 256-GPU Runbook

## Scope and acceptance gate

This runbook reproduces four matched performance-and-profile runs on 256 GB300
GPUs:

1. Baseline Chien-Chin: eager PP1 MTP1.
2. Test GraphTrainer Chien-Chin: GraphTrainer PP1 MTP1.
3. Baseline Sanket: eager PP2/VPP8 MTP1.
4. Test GraphTrainer Sanket: GraphTrainer PP2/VPP8 MTP1.

All four use the same published TorchTitan stack, PyTorch runtime, DistMoE
extension, objective, and MAST tenant. Inputs match within each eager/GraphTrainer
pair, not across the two pairs. Chien-Chin uses the committed test tokenizer and
`tests/assets/c4_test/data.json`; Sanket uses the packaged DeepSeek V3.1 tokenizer
and `/mnt/mffuse/c4`. All four replace learned routing with deterministic
round-robin routing. These are execution comparisons, not historical convergence
reproductions. Every job enables the profiler and memory history. A result
enters the report only when its console-rounded loss and grad norm are finite,
its full-step CUDA graph captures and replays, and its profile artifacts are
readable. Performance statistics exclude every metrics interval that overlaps
profiler warmup, recording, trace export, or memory-snapshot export.
Failed or dead jobs are not report links.

The local EDP1 numerical defect is fixed and has exact pre-rebase acceptance
evidence. Before packaging, freeze the tested source as immutable commit S and
push the predeclared runtime-source tag below. Build and run every final artifact
from S. After the packaged gates and MAST runs complete, amend only this runbook
and its report in the same top `[not-for-land]` commit to produce report tip R
on the final report branch. Replace each pending value only at its stated stage:

```bash
export FINAL_REPORT_BRANCH=dist-moe-256gpu-repro-final-20261006
export FINAL_REPORT_REF=refs/tags/dist-moe-256gpu-repro-report-20261006-v5
export RUNTIME_SOURCE_REF=refs/tags/dist-moe-256gpu-repro-runtime-source-20261006-v5
export FORK_PUSH_REPOSITORY=git@github.com:IvanKobzarev/torchtitan.git
export FINAL_BASE_COMMIT=948d65c868c5fa8f0290bcf9e54b69f004721f54
export PREVIOUS_RUNTIME_SOURCE_COMMIT=8f8337e566c0b32822b9bd2fcd22efff859ad72d
export RUNTIME_SOURCE_COMMIT=TO_BE_FROZEN
export FINAL_RUNTIME_FBPKG_ID=TO_BE_PUBLISHED
export FINAL_LAUNCHER_FBPKG_ID=TO_BE_PUBLISHED
```

Both S and R are based directly on `FINAL_BASE_COMMIT` and contain exactly 21
commits above it. The first 20 are individually scoped, landable commits from
`880029205` through the singleton-FSDP fix `85de4b5b9`; the only remaining
commit is the top `[not-for-land]` reproduction commit isolating configs,
benchmarking, profiling, debugging, packaging, documentation, and the pinned
runtime compatibility revert described below. S is retained at
`RUNTIME_SOURCE_REF`; R is the later docs-only amendment at
`FINAL_REPORT_BRANCH` and is retained at `FINAL_REPORT_REF` after publication.

The publisher archives `HEAD` and records that commit in the package provenance.
Embedding a runtime or launcher identifier into S would therefore change S and
the package that produced the identifier. Freeze and tag S before packaging,
substitute S into the launcher, and record artifact identifiers and successful
run results only in R. A commit also cannot contain its own hash, so
`RUNTIME_SOURCE_REF` identifies S while `FINAL_REPORT_BRANCH` and the immutable
`FINAL_REPORT_REF` identify R.

The `v5` refs are the current retry namespace; the abandoned `v1` publication
plan must never be moved or reused. The immutable `v2` source ref records a
package attempt that failed before publication because its relocated build
environment could not import the unpinned build backend. The immutable `v3`
source ref records the next package attempt, which passed that build then
failed before publication because a host CUDA 12.8 compiler was selected for a
CUDA 13.0 PyTorch runtime. Neither attempt produced a runtime package or MAST
jobs. The immutable `v4` source produced runtime package
`torchtitan_conda_dist_moe_256gpu_sm103_final:2f07dc452cd7453b984d0e3352221320`,
but its recursive provenance manifests included mutable `.pyc` files. Normal
relocation and import changed those bytecode hashes, so v4 was superseded before
any MAST job. Tags and published fbpkg identifiers are immutable. If a source
change is required after S is tagged, increment both refs to `v6`, rerun all five source
gates, and rebuild every downstream artifact. Do not force-update a release tag
or relabel an existing package.

Both final fbpkgs are ephemeral and expire 28 days after they are built. The
report is an exact historical record, not a promise that those packages remain
fetchable. Rebuilding is possible only while the exact base-runtime,
launcher-template, and launcher-dependency UUIDs remain fetchable. This runbook
does not contain enough source or build provenance to reconstruct those binary
inputs after they expire. Before expiration, their owners must preserve them or
publish authoritative immutable rebuild recipes; otherwise this report remains
historical and cannot be executed self-sufficiently. Never substitute a mutable
alias for an unavailable input. Any rebuilt input requires new full identifiers,
all five packaged gates, both smokes, all eight dryruns, and a new report release.

The resource tenant is
`gen_ai/msl/fair_research/fair_prod/Alignement/MuseSpark_1_2_Safety_DCT`. <!-- codespell:ignore alignement -->

## Exact configurations

The launcher imports these functions from
`scripts/dsv3_671b_dist_moe_256gpu/mast_configs.py` inside the runtime package:

| Report line | Combined config in the published package | Equivalent alias in this source |
| --- | --- | --- |
| Baseline Chien-Chin | `deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile` | `deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance` |
| Test GraphTrainer Chien-Chin | `graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile` | `graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance` |
| Baseline Sanket | `deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile` | `deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance` |
| Test GraphTrainer Sanket | `graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile` | `graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance` |

Both pairs enable MTP depth 1, data target depth 1, MTP loss scale 0.1, and
complementary sequence-wise auxiliary loss coefficient 0.01 at all 59 routed
depths. Both use an outer full-step CUDA graph. The Chien-Chin pair uses PP1,
DP256, EP64, 16 effective microbatches, 16,777,216 tokens/step, and seed 42.
The Sanket pair uses PP2/VPP8, DP128, EP64, 120 microbatches, 62,914,560
tokens/step, and seed 14536.

The PP1 config inherits `num_pp_microbatches=240`, but pipeline scheduling is
inactive at PP1. Its effective local accumulation count is 16, derived from
tokens per step divided by sequence length and DP256. PP2 uses 120 actual
pipeline microbatches. The Sanket recipe matches historical geometry, arithmetic,
and seed but does not include the unavailable warmed checkpoint, AirStore
validation stream, or validation pass.

The base factory activates fused MLA and fused SwiGLU config overrides for every
derived recipe. Config loading applies 124 overrides: 62 MLA sites and 62
SwiGLU sites, including the MTP depth. No extra launcher flag is required.

This source makes all eight aliases enable the profiler and memory snapshots
and resolve to four workloads. The already published v6 package predates that
alias cleanup, so its four `*_profile` names are mandatory; its
`*_performance` names must not be used for this final matrix. PP1 profiles
iteration 41 and records its memory snapshot at iteration 41. PP2 uses profiler
frequency 43, warmup 3, active 2, and records its memory snapshot at iteration
40. Every workload runs 60 training steps and logs metrics every ten steps.

## Validate and freeze the runtime source

Run the five source gates from the candidate top commit before creating S. The
commands below reject a dirty tree, the wrong fork, an incorrect base or commit
count, more than one `[not-for-land]` commit, or a reproduction commit that is
not the stack tip. They also record the exact interpreter, dependency set, and
machine-readable result from each distinct gate.

The source gates use the explicitly pinned development PyTorch, torchao, and
DistMoE revisions below and compare eager with GraphTrainer only within that
environment. They are not artifact-parity evidence. The later packaged gates
are repeated with the final runtime's separately pinned dependency composition
and are authoritative for the published artifact.

```bash
set -euo pipefail
export FORK_REMOTE=fork
export FORK_REPOSITORY=https://github.com/IvanKobzarev/torchtitan.git
export FINAL_REPORT_CHECKOUT=/path/to/clean/torchtitan
export RELEASE_ARTIFACTS=/path/to/artifacts/release-v5
export SOURCE_GATE_ROOT=/path/to/large/tmp/source-gates-v5
export SOURCE_GATE_PYTHON=/absolute/path/to/pinned-development-python
export SOURCE_GATE_PYTORCH_SOURCE=/path/to/clean/pytorch
export SOURCE_GATE_DIST_MOE_SOURCE=/path/to/clean/dist_moe
export EXPECTED_SOURCE_GATE_PYTORCH_BINARY_REVISION=dc0efe32b9ee884dfe0b56e24b3f49a954cfc543
export EXPECTED_SOURCE_GATE_PYTORCH_SOURCE_REVISION=4e6137a2d34b1ac1d646a1b7362ab79ec51ecf4d
export EXPECTED_SOURCE_GATE_TORCHAO_REVISION=6ded493033ebba47e2631b087cb17f01992d51f0
export EXPECTED_SOURCE_GATE_DIST_MOE_REVISION=18b4f4887ab9a97e35193d0921cff51a249202ee
export PYTHONNOUSERSITE=1
export PYTHONPATH="$SOURCE_GATE_DIST_MOE_SOURCE"

cd "$FINAL_REPORT_CHECKOUT"
test "$(git branch --show-current)" = "$FINAL_REPORT_BRANCH"
test "$(git remote get-url "$FORK_REMOTE")" = "$FORK_REPOSITORY"
test "$(git remote get-url --push "$FORK_REMOTE")" = "$FORK_REPOSITORY"
test -z "$(git status --porcelain)"
RUNTIME_SOURCE_COMMIT=$(git rev-parse HEAD)
export RUNTIME_SOURCE_COMMIT
[[ "$RUNTIME_SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
test "$(git merge-base "$FINAL_BASE_COMMIT" "$RUNTIME_SOURCE_COMMIT")" = \
  "$FINAL_BASE_COMMIT"
test "$(git rev-list --count \
  "$FINAL_BASE_COMMIT..$RUNTIME_SOURCE_COMMIT")" -eq 21
test "$(git rev-list --count --merges \
  "$FINAL_BASE_COMMIT..$RUNTIME_SOURCE_COMMIT")" -eq 0
test "$(git rev-list --reverse \
  "$FINAL_BASE_COMMIT..$RUNTIME_SOURCE_COMMIT" | sed -n '1p')" = \
  8800292058c7c6fb274206e816cc5600852c09a6
test "$(git rev-parse "$RUNTIME_SOURCE_COMMIT^")" = \
  85de4b5b9eef78b6cf668e22a8d93625eced5818
test "$(git log -1 --format=%s "$RUNTIME_SOURCE_COMMIT")" = \
  "[not-for-land] Add DistMoE 256-GPU reproduction package"
test "$(git log --format=%s \
  "$FINAL_BASE_COMMIT..$RUNTIME_SOURCE_COMMIT" \
  | awk '/^\[not-for-land\]/{count++} END{print count+0}')" -eq 1

SOURCE_GATE_PYTHON=$(readlink -f "$SOURCE_GATE_PYTHON")
test -x "$SOURCE_GATE_PYTHON"
test -z "$(git -C "$SOURCE_GATE_PYTORCH_SOURCE" status --porcelain)"
test "$(git -C "$SOURCE_GATE_PYTORCH_SOURCE" rev-parse HEAD)" = \
  "$EXPECTED_SOURCE_GATE_PYTORCH_SOURCE_REVISION"
test -z "$(git -C "$SOURCE_GATE_DIST_MOE_SOURCE" status --porcelain)"
test "$(git -C "$SOURCE_GATE_DIST_MOE_SOURCE" rev-parse HEAD)" = \
  "$EXPECTED_SOURCE_GATE_DIST_MOE_REVISION"
for revision in \
  "$EXPECTED_SOURCE_GATE_PYTORCH_BINARY_REVISION" \
  "$EXPECTED_SOURCE_GATE_PYTORCH_SOURCE_REVISION" \
  "$EXPECTED_SOURCE_GATE_TORCHAO_REVISION" \
  "$EXPECTED_SOURCE_GATE_DIST_MOE_REVISION"; do
  [[ "$revision" =~ ^[0-9a-f]{40}$ ]]
done
test ! -e "$SOURCE_GATE_ROOT"
test ! -e "$RELEASE_ARTIFACTS/source-gates"
mkdir -p "$SOURCE_GATE_ROOT" "$RELEASE_ARTIFACTS/source-gates"
"$SOURCE_GATE_PYTHON" -m pytest -q \
  tests/unit_tests/cpu/test_runtime_tree_manifest.py \
  2>&1 | tee "$RELEASE_ARTIFACTS/source-gates/runtime-tree-manifest-test.log"
"$SOURCE_GATE_PYTHON" - "$PWD" <<'PY' \
  >"$RELEASE_ARTIFACTS/source-gates/interpreter.json"
import importlib.metadata as metadata
import json
import os
import platform
from pathlib import Path
import sys

import dist_moe
import torch
import torchao
import torchtitan

source_root = Path(sys.argv[1]).resolve()
torchtitan_path = Path(torchtitan.__file__).resolve()
assert torchtitan_path.is_relative_to(source_root), torchtitan_path
dist_moe_source = Path(os.environ["SOURCE_GATE_DIST_MOE_SOURCE"]).resolve()
dist_moe_path = Path(dist_moe.__file__).resolve()
assert dist_moe_path.is_relative_to(dist_moe_source), dist_moe_path
torch_source = Path(os.environ["SOURCE_GATE_PYTORCH_SOURCE"]).resolve()
torch_path = Path(torch.__file__).resolve()
assert torch_path.is_relative_to(torch_source), torch_path
assert torch.version.git_version == os.environ[
    "EXPECTED_SOURCE_GATE_PYTORCH_BINARY_REVISION"
]
torchao_direct_url = json.loads(
    metadata.distribution("torchao").read_text("direct_url.json")
)
torchao_revision = torchao_direct_url["vcs_info"]["commit_id"]
assert torchao_revision == os.environ["EXPECTED_SOURCE_GATE_TORCHAO_REVISION"]

def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None

print(json.dumps({
    "executable": sys.executable,
    "module_files": {
        "dist_moe": dist_moe.__file__,
        "torch": torch.__file__,
        "torchao": torchao.__file__,
        "torchtitan": torchtitan.__file__,
    },
    "packages": {
        name: package_version(name)
        for name in ("dist-moe", "torch", "torchao", "torchtitan")
    },
    "platform": platform.platform(),
    "python": sys.version,
    "dist_moe_revision": os.environ["EXPECTED_SOURCE_GATE_DIST_MOE_REVISION"],
    "pytorch_binary_revision": torch.version.git_version,
    "pytorch_source_revision": os.environ[
        "EXPECTED_SOURCE_GATE_PYTORCH_SOURCE_REVISION"
    ],
    "torchao_revision": torchao_revision,
}, indent=2, sort_keys=True))
PY
"$SOURCE_GATE_PYTHON" -m pip freeze --all \
  >"$RELEASE_ARTIFACTS/source-gates/pip-freeze.txt"
git rev-parse HEAD >"$RELEASE_ARTIFACTS/source-gates/torchtitan-commit.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$SOURCE_GATE_PYTHON" \
  scripts/dsv3_671b_dist_moe_256gpu/run_paired_numerics.py pp1 \
  --pp1-accumulation-steps 2 --capture-steps 1,2 \
  --output-dir "$SOURCE_GATE_ROOT/pp1-ga2" \
  2>&1 | tee "$RELEASE_ARTIFACTS/source-gates/pp1-ga2.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$SOURCE_GATE_PYTHON" \
  scripts/dsv3_671b_dist_moe_256gpu/run_paired_numerics.py pp1 \
  --pp1-accumulation-steps 4 --capture-steps 1,2 \
  --output-dir "$SOURCE_GATE_ROOT/pp1-ga4" \
  2>&1 | tee "$RELEASE_ARTIFACTS/source-gates/pp1-ga4.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$SOURCE_GATE_PYTHON" \
  scripts/dsv3_671b_dist_moe_256gpu/run_paired_numerics.py pp2 \
  --capture-steps 1,2 --output-dir "$SOURCE_GATE_ROOT/pp2" \
  2>&1 | tee "$RELEASE_ARTIFACTS/source-gates/pp2.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$SOURCE_GATE_PYTHON" \
  scripts/dsv3_671b_dist_moe_256gpu/run_paired_numerics.py pp1 \
  --pp1-accumulation-steps 4 --capture-steps 1,2 --cuda-graphs \
  --output-dir "$SOURCE_GATE_ROOT/pp1-ga4-cg" \
  2>&1 | tee "$RELEASE_ARTIFACTS/source-gates/pp1-ga4-cg.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$SOURCE_GATE_PYTHON" \
  scripts/dsv3_671b_dist_moe_256gpu/run_paired_numerics.py pp2 \
  --capture-steps 1,2 --cuda-graphs \
  --output-dir "$SOURCE_GATE_ROOT/pp2-cg" \
  2>&1 | tee "$RELEASE_ARTIFACTS/source-gates/pp2-cg.log"

: >"$RELEASE_ARTIFACTS/source-gates/results.jsonl"
for gate in pp1-ga2 pp1-ga4 pp2 pp1-ga4-cg pp2-cg; do
  result="$SOURCE_GATE_ROOT/$gate/paired_numerics_result.json"
  jq -e '.equal == true' "$result" >/dev/null
  install -m 0644 "$result" \
    "$RELEASE_ARTIFACTS/source-gates/$gate-result.json"
  jq -c --arg gate "$gate" '{gate: $gate, result: .}' "$result" \
    >>"$RELEASE_ARTIFACTS/source-gates/results.jsonl"
done
sha256sum "$RELEASE_ARTIFACTS/source-gates"/* \
  >"$RELEASE_ARTIFACTS/source-gates-sha256.txt"
test -z "$(git status --porcelain)"
unset PYTHONPATH
```

Only after all five results are exactly equal, create and push S. A pre-existing
local or remote release tag is a hard failure; never replace it. The report
branch currently names the superseded v4 source. Replace that sibling NFR
tip only with an exact force-with-lease while atomically creating the immutable
v5 tag. `GIT_CONFIG_COUNT=0` disables the Meta host's injected SSH-to-HTTPS
rewrite; HTTPS OAuth credentials without the `workflow` scope cannot publish a
commit that contains workflow files.

```bash
if git show-ref --verify --quiet "$RUNTIME_SOURCE_REF"; then
  echo "Ref already exists locally: $RUNTIME_SOURCE_REF" >&2
  exit 1
fi
if [[ -n "$(GIT_CONFIG_COUNT=0 git ls-remote \
  "$FORK_PUSH_REPOSITORY" "$RUNTIME_SOURCE_REF")" ]]; then
  echo "Ref already exists remotely: $RUNTIME_SOURCE_REF" >&2
  exit 1
fi
test "$(GIT_CONFIG_COUNT=0 git ls-remote "$FORK_PUSH_REPOSITORY" \
  "refs/heads/$FINAL_REPORT_BRANCH" | awk '{print $1}')" = \
  "$PREVIOUS_RUNTIME_SOURCE_COMMIT"
git tag -a "${RUNTIME_SOURCE_REF#refs/tags/}" "$RUNTIME_SOURCE_COMMIT" \
  -m "DistMoE 256-GPU immutable runtime source v5"
GIT_CONFIG_COUNT=0 git push --atomic \
  --force-with-lease="refs/heads/$FINAL_REPORT_BRANCH:$PREVIOUS_RUNTIME_SOURCE_COMMIT" \
  "$FORK_PUSH_REPOSITORY" \
  "$RUNTIME_SOURCE_COMMIT:refs/heads/$FINAL_REPORT_BRANCH" \
  "$RUNTIME_SOURCE_REF"
test "$(GIT_CONFIG_COUNT=0 git ls-remote "$FORK_PUSH_REPOSITORY" \
  "refs/heads/$FINAL_REPORT_BRANCH" | awk '{print $1}')" = \
  "$RUNTIME_SOURCE_COMMIT"
test "$(GIT_CONFIG_COUNT=0 git ls-remote "$FORK_PUSH_REPOSITORY" \
  "$RUNTIME_SOURCE_REF^{}" | awk '{print $1}')" = \
  "$RUNTIME_SOURCE_COMMIT"
test "$(GIT_CONFIG_COUNT=0 git ls-remote "$FORK_PUSH_REPOSITORY" \
  'refs/tags/dist-moe-256gpu-repro-runtime-source-20261006-v4^{}' \
  | awk '{print $1}')" = "$PREVIOUS_RUNTIME_SOURCE_COMMIT"
```

## Build and publish the runtime fbpkg

The following dependency revisions are the validated build inputs. The base
runtime is an input image, not the final artifact:

```bash
export FINAL_SOURCE_REPOSITORY=https://github.com/IvanKobzarev/torchtitan.git
export PYTORCH_SOURCE_REPOSITORY=https://github.com/IvanKobzarev/pytorch.git
export PYTORCH_BINARY_REVISION=93e8717146a92b354a21fd346bdbffb106fe7586
export PYTORCH_COMMIT=8f709e9258aff0d5f3e74b40b7631ed30fe1c6ca
export DIST_MOE_COMMIT=18b4f4887ab9a97e35193d0921cff51a249202ee
export TORCHAO_COMMIT=a701b6a6058720c21f95908b7ae4a24bf0cae1b6
export BASE_RUNTIME_FBPKG_ID=torchtitan_conda_ivankobzarev_dist_moe_256gpu_sm103_20261006_v9:910dab27fa354591a9e0d5119feabdb5
```

Use full `PACKAGE:UUID` identifiers for every fbpkg variable. A bare UUID is not
an acceptable recorded artifact identifier. Materialize clean detached sources:

```bash
export SOURCE_ROOT=/path/to/large/tmp/final-sources
test ! -e "$SOURCE_ROOT"
mkdir -p "$SOURCE_ROOT"

git clone --filter=blob:none "$FINAL_SOURCE_REPOSITORY" "$SOURCE_ROOT/torchtitan"
git -C "$SOURCE_ROOT/torchtitan" fetch origin "$RUNTIME_SOURCE_REF"
test "$(git -C "$SOURCE_ROOT/torchtitan" rev-parse 'FETCH_HEAD^{commit}')" = \
  "$RUNTIME_SOURCE_COMMIT"
git -C "$SOURCE_ROOT/torchtitan" checkout --detach "$RUNTIME_SOURCE_COMMIT"

git clone --filter=blob:none "$PYTORCH_SOURCE_REPOSITORY" "$SOURCE_ROOT/pytorch"
git -C "$SOURCE_ROOT/pytorch" fetch origin \
  "$PYTORCH_BINARY_REVISION" "$PYTORCH_COMMIT"
git -C "$SOURCE_ROOT/pytorch" checkout --detach "$PYTORCH_COMMIT"
git -C "$SOURCE_ROOT/pytorch" submodule update --init --depth 1 \
  third_party/flash-attention

git clone --filter=blob:none https://github.com/meta-pytorch/dist_moe.git "$SOURCE_ROOT/dist_moe"
git -C "$SOURCE_ROOT/dist_moe" checkout --detach "$DIST_MOE_COMMIT"

git clone --filter=blob:none https://github.com/pytorch/ao.git "$SOURCE_ROOT/torchao"
git -C "$SOURCE_ROOT/torchao" checkout --detach "$TORCHAO_COMMIT"

for source in torchtitan pytorch dist_moe torchao; do
  test -z "$(git -C "$SOURCE_ROOT/$source" status --porcelain)"
done
test "$(git -C "$SOURCE_ROOT/torchtitan" rev-parse HEAD)" = \
  "$RUNTIME_SOURCE_COMMIT"
```

The base provides PyTorch binaries built at `PYTORCH_BINARY_REVISION`, including
the NCCL2 CUDA-graph cleanup, plus the relocatable Python/CUDA environment,
compiler, tokenizer, and matching MXFP8 extension. The validated target is a
descendant that adds the pipeline metadata-restoration change. Its only
production-source delta is `torch/distributed/pipelining/stage.py`; its other
delta is the matching test. The publisher verifies that ancestry and allowlist
before overlaying the tracked Python source, and rejects every other
production-source difference.
Reinstall all source dependencies with `PRENORMALIZED_RUNTIME=0`; do not merely
relabel the packages already installed in the base runtime.

```bash
export BASE_DEST=/path/to/large/tmp/final-base-runtime
test ! -e "$BASE_DEST"
mkdir -p "$BASE_DEST"
fbpkg fetch --dest "$BASE_DEST" --extract --verify \
  --unexpected-fails-verify "$BASE_RUNTIME_FBPKG_ID"

export BASE_ROOT="$BASE_DEST/conda"
"$BASE_ROOT/bin/python" "$BASE_ROOT/bin/conda-unpack-fb"

export TORCHTITAN_SOURCE="$SOURCE_ROOT/torchtitan"
export PYTORCH_SOURCE="$SOURCE_ROOT/pytorch"
export DIST_MOE_SOURCE="$SOURCE_ROOT/dist_moe"
export TORCHAO_SOURCE="$SOURCE_ROOT/torchao"
export RUNTIME_PREFIX="$BASE_ROOT"
export COMPILER_PREFIX="$BASE_ROOT"
export BASE_RUNTIME_FBPKG_ID
export HF_TOKENIZER_SOURCE="$BASE_ROOT/src/torchtitan/assets/hf/DeepSeek-V3.1-Base"
export TORCHAO_MXFP8_EXTENSION="$BASE_ROOT/lib/python3.12/site-packages/torchao/_C_mxfp8.cpython-312-aarch64-linux-gnu.so"
export PRENORMALIZED_RUNTIME=0
export EXPECTED_CUDA_ARCH=sm_103
export EXPECTED_TORCHTITAN_REVISION="$RUNTIME_SOURCE_COMMIT"
export EXPECTED_PYTORCH_BINARY_REVISION="$PYTORCH_BINARY_REVISION"
export EXPECTED_PYTORCH_SOURCE_REVISION="$PYTORCH_COMMIT"
export EXPECTED_DIST_MOE_REVISION="$DIST_MOE_COMMIT"
export EXPECTED_TORCHAO_REVISION="$TORCHAO_COMMIT"
export FBPKG_NAME=torchtitan_conda_dist_moe_256gpu_sm103_final
export STAGING_PARENT=/path/to/large/tmp/torchtitan-fbpkg-staging

cd "$TORCHTITAN_SOURCE"
scripts/publish_dist_moe_fbpkg.sh
```

Record the returned full identifier as `FINAL_RUNTIME_FBPKG_ID`. The current
publisher refuses to stage or publish unless all five expected 40-character
revisions exactly match the clean TorchTitan, PyTorch binary, PyTorch source,
DistMoE, and torchao inputs. Immediately set the runtime artifact's absolute
lifetime and retain its metadata:

```bash
export FINAL_RUNTIME_FBPKG_ID=PACKAGE:UUID_FROM_PUBLISHER
fbpkg expire --exact-update-only "$FINAL_RUNTIME_FBPKG_ID" 28d
fbpkg info --json "$FINAL_RUNTIME_FBPKG_ID" \
  | tee "$RELEASE_ARTIFACTS/final-runtime-fbpkg-info.json"
test "$(jq -r .package \
  "$RELEASE_ARTIFACTS/final-runtime-fbpkg-info.json")" = "$FBPKG_NAME"
test "$(jq -r .uuid \
  "$RELEASE_ARTIFACTS/final-runtime-fbpkg-info.json")" = \
  "${FINAL_RUNTIME_FBPKG_ID#*:}"
```

The top `[not-for-land]` package commit removes
`MixedPrecisionPolicy.param_dtype_override_fn` forwarding because this pinned
runtime predates that API. This package-only revert must not land and must be
dropped when the PyTorch runtime implements the forwarding contract.

Fetch the published artifact into a new directory and verify it on a GB300 host:

```bash
set -euo pipefail
export RUNTIME_DEST=/path/to/large/tmp/final-runtime-fetch
test ! -e "$RUNTIME_DEST"
mkdir -p "$RUNTIME_DEST"
fbpkg fetch --dest "$RUNTIME_DEST" --extract --verify \
  --unexpected-fails-verify "$FINAL_RUNTIME_FBPKG_ID"

export RUNTIME_ROOT="$RUNTIME_DEST/conda"
"$RUNTIME_ROOT/bin/python" "$RUNTIME_ROOT/bin/conda-unpack-fb"
export PATH="$RUNTIME_ROOT/bin:/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin"
export LD_LIBRARY_PATH="$RUNTIME_ROOT/lib:/usr/local/cuda-13.0/lib64"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
cd "$RUNTIME_ROOT/src/torchtitan"
runtime_manifest_tool=scripts/dsv3_671b_dist_moe_256gpu/runtime_tree_manifest.py
provenance_dir="$RUNTIME_ROOT/torchtitan_fbpkg_provenance"
provenance_artifacts="$RELEASE_ARTIFACTS/runtime-tree-provenance"
test ! -e "$provenance_artifacts"
mkdir -p "$provenance_artifacts"
install -m 0644 \
  "$provenance_dir/pack_policy.txt" \
  "$provenance_dir/dist_moe_installed_sha256.txt" \
  "$provenance_dir/flash_attn_runtime_sha256.txt" \
  "$provenance_artifacts/"
{
  grep -Fx 'runtime_tree_manifests_exclude=*.pyc' \
    "$provenance_dir/pack_policy.txt"
  "$RUNTIME_ROOT/bin/python" "$runtime_manifest_tool" verify --unprefixed \
    "$provenance_dir/dist_moe_installed_sha256.txt" \
    "$RUNTIME_ROOT/lib/python3.12/site-packages/dist_moe"
  "$RUNTIME_ROOT/bin/python" "$runtime_manifest_tool" verify \
    "$provenance_dir/flash_attn_runtime_sha256.txt" \
    "$RUNTIME_ROOT/lib/python3.12/site-packages/flash_attn/cute" \
    "$RUNTIME_ROOT/lib/python3.12/site-packages/quack" \
    "$RUNTIME_ROOT/lib/python3.12/site-packages/torch_c_dlpack_ext"
} 2>&1 | tee "$provenance_artifacts/verification.log"
"$RUNTIME_ROOT/bin/python" - <<'PY'
from pathlib import Path

import dist_moe._blockscaled  # noqa: F401
import torch
import torchtitan

source_root = Path.cwd().resolve()
assert Path(torchtitan.__file__).resolve().is_relative_to(source_root)
assert torch.cuda.get_device_capability() == (10, 3)
assert torch.cuda.get_arch_list() == ["sm_103"]
assert hasattr(torch.ops.aten, "_scaled_addmm_")
assert hasattr(torch.ops.dist_moe, "block_scaled_backward_accumulate_")
assert hasattr(torch.ops.dist_moe, "bf16_backward_accumulate_")
print(torch.__version__, torch.version.git_version, torchtitan.__file__)
PY

grep -Fx "pytorch=$PYTORCH_COMMIT" \
  "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/revisions.txt"
grep -Fx "pytorch_binary_base=$PYTORCH_BINARY_REVISION" \
  "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/revisions.txt"
grep -Fx "base_runtime_fbpkg=$BASE_RUNTIME_FBPKG_ID" \
  "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/revisions.txt"
grep -Fx "torchtitan=$RUNTIME_SOURCE_COMMIT" \
  "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/revisions.txt"
grep -Fx "dist_moe=$DIST_MOE_COMMIT" \
  "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/revisions.txt"
grep -Fx "torchao=$TORCHAO_COMMIT" \
  "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/revisions.txt"
grep -Fx "build_setuptools=78.1.0" \
  "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/revisions.txt"
grep -Fx "build_cython=0.29.37" \
  "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/revisions.txt"
grep -Fx "cuda_toolkit=13.0" \
  "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/revisions.txt"
(cd "$RUNTIME_ROOT/src/torchtitan" && \
  sha256sum -c "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/tokenizer_sha256.txt")
```

Run `conda-unpack-fb` exactly once after extraction at the final absolute path,
before any packaged console script. Do not activate or move the runtime first.

Landable commit `85de4b5b9` resolves the singleton-FSDP EDP1 accumulation defect.
Without an expert-gradient collective, GraphPP previously left the terminal
BF16-to-FP32 persistent-gradient cast in every repeated microbatch compute
action. GraphTrainer therefore accumulated in FP32 instead of matching eager
FSDP's BF16 accumulation followed by one cast at the once-per-step reduction
boundary. The fix explicitly marks that boundary and extracts the cast as a
zero-collective reduction epilogue with strict provenance validation.

The fix has exact local evidence from the pre-rebase source. PP1 DP4/EP4,
EDP1, GA4 produced loss `13.099918365478516` on both implementations and
matched all 340 parameter and gradient records on each rank. PP2/VPP8 DP2/EP2,
EDP1 produced exact eager/GraphTrainer losses `13.09231185913086` and
`8.543712615966797` at steps 1 and 2; its parameter and gradient manifests
matched 166/166 records on ranks 0-1 and 175/175 on ranks 2-3. These results
exercise the formerly failing no-collective path but remain internal acceptance
evidence. Final publication still requires the tracked deterministic comparison
for PP1 and PP2 on the rebased source and packaged runtime; finite loss or
convergence alone is insufficient.

Use the tracked paired gate for that check. It delegates both training runs and
full-precision TensorBoard scalar extraction to `scripts/loss_compare.py`, then
requires exact loss, grad norm, config contract, and pre-optimizer SHA-256 for
every local parameter gradient on every rank. The eager controls explicitly set
`model.local_compile_regions=[]`. Both controls use seed 42,
`debug.deterministic=True`, and `debug.deterministic_warn_only=False`.

Repeat the same five distinct gates with the fetched package interpreter. Each
output path must be new. The default disables CUDA graphs to isolate the
training implementations. The last two gates exercise the publication path
with the existing outer full-step CUDA graph enabled. Record the package
interpreter, complete dependency set, provenance, logs, and all five results
separately from the source-gate artifacts:

```bash
export PACKAGED_GATE_ROOT=/path/to/large/tmp/packaged-gates-v5
export PACKAGED_GATE_PYTHON="$RUNTIME_ROOT/bin/python"
test ! -e "$PACKAGED_GATE_ROOT"
test ! -e "$RELEASE_ARTIFACTS/packaged-gates"
mkdir -p "$PACKAGED_GATE_ROOT" "$RELEASE_ARTIFACTS/packaged-gates"
test "$(readlink -f "$PACKAGED_GATE_PYTHON")" = \
  "$(readlink -f "$RUNTIME_ROOT/bin/python")"
"$PACKAGED_GATE_PYTHON" - "$RUNTIME_ROOT/src/torchtitan" <<'PY' \
  >"$RELEASE_ARTIFACTS/packaged-gates/interpreter.json"
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import sys

import dist_moe
import torch
import torchao
import torchtitan

packages = ("dist-moe", "torch", "torchao", "torchtitan")
source_root = Path(sys.argv[1]).resolve()
torchtitan_path = Path(torchtitan.__file__).resolve()
assert torchtitan_path.is_relative_to(source_root), torchtitan_path
print(json.dumps({
    "executable": sys.executable,
    "module_files": {
        "dist_moe": dist_moe.__file__,
        "torch": torch.__file__,
        "torchao": torchao.__file__,
        "torchtitan": torchtitan.__file__,
    },
    "packages": {name: metadata.version(name) for name in packages},
    "platform": platform.platform(),
    "python": sys.version,
    "pytorch_revision": torch.version.git_version,
}, indent=2, sort_keys=True))
PY
"$PACKAGED_GATE_PYTHON" -m pip freeze --all \
  >"$RELEASE_ARTIFACTS/packaged-gates/pip-freeze.txt"
install -m 0644 \
  "$RUNTIME_ROOT/torchtitan_fbpkg_provenance/revisions.txt" \
  "$RELEASE_ARTIFACTS/packaged-gates/revisions.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PACKAGED_GATE_PYTHON" \
  scripts/dsv3_671b_dist_moe_256gpu/run_paired_numerics.py pp1 \
  --pp1-accumulation-steps 2 --capture-steps 1,2 \
  --output-dir "$PACKAGED_GATE_ROOT/pp1-ga2" \
  2>&1 | tee "$RELEASE_ARTIFACTS/packaged-gates/pp1-ga2.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PACKAGED_GATE_PYTHON" \
  scripts/dsv3_671b_dist_moe_256gpu/run_paired_numerics.py pp1 \
  --pp1-accumulation-steps 4 --capture-steps 1,2 \
  --output-dir "$PACKAGED_GATE_ROOT/pp1-ga4" \
  2>&1 | tee "$RELEASE_ARTIFACTS/packaged-gates/pp1-ga4.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PACKAGED_GATE_PYTHON" \
  scripts/dsv3_671b_dist_moe_256gpu/run_paired_numerics.py pp2 \
  --capture-steps 1,2 --output-dir "$PACKAGED_GATE_ROOT/pp2" \
  2>&1 | tee "$RELEASE_ARTIFACTS/packaged-gates/pp2.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PACKAGED_GATE_PYTHON" \
  scripts/dsv3_671b_dist_moe_256gpu/run_paired_numerics.py pp1 \
  --pp1-accumulation-steps 4 --capture-steps 1,2 --cuda-graphs \
  --output-dir "$PACKAGED_GATE_ROOT/pp1-ga4-cg" \
  2>&1 | tee "$RELEASE_ARTIFACTS/packaged-gates/pp1-ga4-cg.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$PACKAGED_GATE_PYTHON" \
  scripts/dsv3_671b_dist_moe_256gpu/run_paired_numerics.py pp2 \
  --capture-steps 1,2 --cuda-graphs \
  --output-dir "$PACKAGED_GATE_ROOT/pp2-cg" \
  2>&1 | tee "$RELEASE_ARTIFACTS/packaged-gates/pp2-cg.log"

: >"$RELEASE_ARTIFACTS/packaged-gates/results.jsonl"
for gate in pp1-ga2 pp1-ga4 pp2 pp1-ga4-cg pp2-cg; do
  result="$PACKAGED_GATE_ROOT/$gate/paired_numerics_result.json"
  jq -e '.equal == true' "$result" >/dev/null
  install -m 0644 "$result" \
    "$RELEASE_ARTIFACTS/packaged-gates/$gate-result.json"
  jq -c --arg gate "$gate" '{gate: $gate, result: .}' "$result" \
    >>"$RELEASE_ARTIFACTS/packaged-gates/results.jsonl"
done
sha256sum "$RELEASE_ARTIFACTS/packaged-gates"/* \
  >"$RELEASE_ARTIFACTS/packaged-gates-sha256.txt"
```

Each command exits nonzero on any mismatch. Preserve
`paired_numerics_result.json`, `comparison_statistics.txt`, the two training
logs, TensorBoard events, config contracts, and `gradient_sha256/` manifests
with the final report artifacts. Digests cover rank-local contiguous bytes; a
DTensor is never gathered by the harness.

## Publish the MAST launcher

The launcher text sources are based on the validated v20 launcher. Binary RDMA
libraries are copied from this immutable template package and checked before the
tracked final overrides are installed:

```bash
export LAUNCHER_TEMPLATE_FBPKG_ID=torchtitan_muse_spark_launcher_ivankobzarev_20261005_v1:925c00775066448cb3fcdcb8f43b395f
export LAUNCHER_TEMPLATE_DIR=/path/to/large/tmp/launcher-template
export LAUNCHER_SOURCE=/path/to/large/tmp/final-launcher-source
export FINAL_LAUNCHER_NAME=torchtitan_muse_spark_launcher_dist_moe_mtp1_final
export CONDA_MAST_CORE_FBPKG_ID=conda_mast_core:bd79ff6a03ba48989f1b142418c1f0b0
export TORCHX_TORCHRUN_FBPKG_ID=torchx_torchrun:5b856b7bf0ff472baad17a127e73dc3a
export FOLLY_SYMBOLIZER_FBPKG_ID=folly.symbolizer:7fd465c396be46fabed56a0c54aee61b
export MANIFOLDFS_FBPKG_ID=manifold.manifoldfs:e3844a236d124df08a162a4580416fbc
export OILFS_FBPKG_ID=oil.oilfs:bbf8d01e29dd477eb42f2bfb58dce254
export ADDITIONAL_PACKAGES_FBPKG_ID=torchtitan_additional_packages:29d8d805d6c4403480902a0a0a637898
test ! -e "$LAUNCHER_TEMPLATE_DIR"
test ! -e "$LAUNCHER_SOURCE"
mkdir -p "$LAUNCHER_TEMPLATE_DIR" "$LAUNCHER_SOURCE"
fbpkg fetch --dest "$LAUNCHER_TEMPLATE_DIR" --extract --verify \
  --unexpected-fails-verify "$LAUNCHER_TEMPLATE_FBPKG_ID"

(cd "$LAUNCHER_TEMPLATE_DIR" && sha256sum -c <<'EOF'
bc63487705712767881ae6b86726a78f2de6c25e4de8b833806e03862b0cd587  bha_four_nvl_domains.json
a16f8c148610a9fbfb885702795a402f12e3a39baa6f8f5f730e9fbd987d5b0b  cc_wrapper.sh
def046ce753bfa1607e2c3fabd563724fc81552b6e8278489cbd5534c11c74f3  cxx_wrapper.sh
9b39044206368071953f64317a9c2c4d6433379eb662322771666bcdb7f502af  lib/libibverbs.so.1
b11ded57c4121c3b87da4c714c2cad8a87e828475311022dc0ef22e7be39fcff  lib/libmlx5-rdmav57.so
96326054434e04de706a13f6bc2f7a631558b45d8b497b0ec6642b0674d02e75  mast.py
21b468bdced46d87a751d119179ba1c6805dd1deec6dfb5c39f148666a23093e  mount.sh
EOF
)

install -m 0644 "$LAUNCHER_TEMPLATE_DIR/bha_four_nvl_domains.json" "$LAUNCHER_SOURCE/"
install -m 0755 "$LAUNCHER_TEMPLATE_DIR/cc_wrapper.sh" "$LAUNCHER_SOURCE/"
install -m 0755 "$LAUNCHER_TEMPLATE_DIR/cxx_wrapper.sh" "$LAUNCHER_SOURCE/"
install -m 0644 "$LAUNCHER_TEMPLATE_DIR/mast.py" "$LAUNCHER_SOURCE/"
install -m 0755 "$LAUNCHER_TEMPLATE_DIR/mount.sh" "$LAUNCHER_SOURCE/"
cp -a "$LAUNCHER_TEMPLATE_DIR/lib" "$LAUNCHER_SOURCE/"
python3 - "$LAUNCHER_SOURCE/mast.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
replacements = {
    'conda_mast_core:stable': (
        'conda_mast_core:bd79ff6a03ba48989f1b142418c1f0b0'
    ),
    'folly.symbolizer:stable': (
        'folly.symbolizer:7fd465c396be46fabed56a0c54aee61b'
    ),
    'manifold.manifoldfs': (
        'manifold.manifoldfs:e3844a236d124df08a162a4580416fbc'
    ),
    'oil.oilfs:stable': 'oil.oilfs:bbf8d01e29dd477eb42f2bfb58dce254',
}
for mutable_id, immutable_id in replacements.items():
    if text.count(mutable_id) != 1:
        raise RuntimeError(
            f"expected one {mutable_id!r} in {path}, found "
            f"{text.count(mutable_id)}"
        )
    text = text.replace(mutable_id, immutable_id)
role_needle = "    role = job_spec.roles[0]\n"
role_replacement = role_needle + """    runner_prefix = "torchx_torchrun:"
    runner_id = "torchx_torchrun:5b856b7bf0ff472baad17a127e73dc3a"
    image_parts = role.image.split(";")
    runner_indexes = [
        index
        for index, image_part in enumerate(image_parts)
        if image_part.startswith(runner_prefix)
    ]
    if len(runner_indexes) != 1:
        raise RuntimeError(
            f"expected one {runner_prefix} dependency, got {role.image!r}"
        )
    image_parts[runner_indexes[0]] = runner_id
    role.image = ";".join(image_parts)
"""
if text.count(role_needle) != 1:
    raise RuntimeError(
        f"expected one role assignment in {path}, found {text.count(role_needle)}"
    )
text = text.replace(role_needle, role_replacement)
path.write_text(text)
PY
printf '%s\n' \
  "$CONDA_MAST_CORE_FBPKG_ID" \
  "$TORCHX_TORCHRUN_FBPKG_ID" \
  "$FOLLY_SYMBOLIZER_FBPKG_ID" \
  "$MANIFOLDFS_FBPKG_ID" \
  "$OILFS_FBPKG_ID" \
  "$ADDITIONAL_PACKAGES_FBPKG_ID" \
  >"$LAUNCHER_SOURCE/launcher_fbpkg_dependencies.txt"
while IFS= read -r dependency_id; do
  package_name=${dependency_id%%:*}
  package_uuid=${dependency_id#*:}
  dependency_info="$RELEASE_ARTIFACTS/launcher-dependency-$package_name-$package_uuid.json"
  fbpkg info --json "$dependency_id" \
    | tee "$dependency_info" \
    | jq -e --arg package "$package_name" --arg uuid "$package_uuid" \
      '.package == $package and .uuid == $uuid' >/dev/null
done <"$LAUNCHER_SOURCE/launcher_fbpkg_dependencies.txt"
if rg -n 'conda_mast_core:stable|torchx_torchrun:stable|folly\.symbolizer:stable|oil\.oilfs:stable|"manifold\.manifoldfs"' \
  "$LAUNCHER_SOURCE/mast.py"; then
  echo "Mutable launcher fbpkg dependency remains" >&2
  exit 1
fi
install -m 0644 \
  "$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/launcher/mast_configs.py" \
  "$LAUNCHER_SOURCE/mast_configs.py"
sed "s|@FINAL_RUNTIME_FBPKG_ID@|$FINAL_RUNTIME_FBPKG_ID|g" \
  "$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/launcher/.torchxconfig" \
  >"$LAUNCHER_SOURCE/.torchxconfig"
sed -e "s|@FINAL_SOURCE_COMMIT@|$RUNTIME_SOURCE_COMMIT|g" \
    -e "s|@BASE_RUNTIME_FBPKG_ID@|$BASE_RUNTIME_FBPKG_ID|g" \
    -e "s|@PYTORCH_BINARY_REVISION@|$PYTORCH_BINARY_REVISION|g" \
    -e "s|@PYTORCH_COMMIT@|$PYTORCH_COMMIT|g" \
    -e "s|@DIST_MOE_COMMIT@|$DIST_MOE_COMMIT|g" \
    -e "s|@TORCHAO_COMMIT@|$TORCHAO_COMMIT|g" \
  "$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/launcher/run.sh.in" \
  >"$LAUNCHER_SOURCE/run.sh"
chmod 0755 "$LAUNCHER_SOURCE/run.sh"
if rg -n '@[A-Z][A-Z0-9_]+@|TO_BE_(FROZEN|PUBLISHED)' \
  "$LAUNCHER_SOURCE/.torchxconfig" \
  "$LAUNCHER_SOURCE/bha_four_nvl_domains.json" \
  "$LAUNCHER_SOURCE/cc_wrapper.sh" \
  "$LAUNCHER_SOURCE/cxx_wrapper.sh" \
  "$LAUNCHER_SOURCE/launcher_fbpkg_dependencies.txt" \
  "$LAUNCHER_SOURCE/mast.py" \
  "$LAUNCHER_SOURCE/mast_configs.py" \
  "$LAUNCHER_SOURCE/mount.sh" \
  "$LAUNCHER_SOURCE/run.sh"; then
  echo "Unresolved launcher substitution token remains" >&2
  exit 1
fi
python3 \
  "$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/launcher_manifest.py" \
  create "$LAUNCHER_SOURCE" "$LAUNCHER_SOURCE/launcher_manifest.json"
python3 \
  "$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/launcher_manifest.py" \
  verify "$LAUNCHER_SOURCE" "$LAUNCHER_SOURCE/launcher_manifest.json"
LAUNCHER_MANIFEST_SHA256=$(sha256sum \
  "$LAUNCHER_SOURCE/launcher_manifest.json" | awk '{print $1}')
export LAUNCHER_MANIFEST_SHA256
printf '%s  launcher_manifest.json\n' "$LAUNCHER_MANIFEST_SHA256" \
  | tee "$RELEASE_ARTIFACTS/launcher-manifest-sha256.txt"

fbpkg build "$FINAL_LAUNCHER_NAME" \
  --local-config-path \
    "$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/launcher/launcher_build_config.json" \
  --repo-path "$LAUNCHER_SOURCE" --json --yes --unclean-clowntown \
  --ephemeral --expire 28d | tee "$RELEASE_ARTIFACTS/launcher-build.json"
```

Record the returned full `PACKAGE:UUID` as `FINAL_LAUNCHER_FBPKG_ID`. The
build is intentionally ephemeral with an absolute 28-day lifetime; retain
its immutable metadata beside the build JSON:

```bash
export FINAL_LAUNCHER_FBPKG_ID=PACKAGE:UUID_FROM_BUILD
fbpkg info --json "$FINAL_LAUNCHER_FBPKG_ID" \
  | tee "$RELEASE_ARTIFACTS/final-launcher-fbpkg-info.json"
test "$(jq -r .package \
  "$RELEASE_ARTIFACTS/final-launcher-fbpkg-info.json")" = \
  "$FINAL_LAUNCHER_NAME"
test "$(jq -r .uuid \
  "$RELEASE_ARTIFACTS/final-launcher-fbpkg-info.json")" = \
  "${FINAL_LAUNCHER_FBPKG_ID#*:}"
```

The launcher uses `run_as_root=True`, `activate_conda=False`, mounts
`/mnt/mffuse`, changes to the packaged TorchTitan source, and sets
`PYTHONPATH` only to the launcher workspace so `mast_configs.py` is importable.

The scheduler contract must request 64 nodes with four GB300 GPUs each, Normal
priority, strict single-region placement, and the MuseSpark tenant. Use this
comma-separated scheduler configuration for every launch:

```bash
export CHECKPOINT_AGENT_FBPKG_ID=checkpoint_agent:0d3fa4116fd446ea931c0699c748dedb
fbpkg info --json "$CHECKPOINT_AGENT_FBPKG_ID" \
  | tee "$RELEASE_ARTIFACTS/checkpoint-agent-fbpkg-info.json" \
  | jq -e \
    --arg package "${CHECKPOINT_AGENT_FBPKG_ID%%:*}" \
    --arg uuid "${CHECKPOINT_AGENT_FBPKG_ID#*:}" \
    '.package == $package and .uuid == $uuid' >/dev/null
export SCHEDULER_CFG="conda_fbpkg_id=$FINAL_RUNTIME_FBPKG_ID,workspace_fbpkg_id=$FINAL_LAUNCHER_FBPKG_ID,hpcIdentity=pytorch_distributed,hpcJobOncall=meta_conda,hpcClusterUuid=MastGenAICluster,rmAttribution=MuseSpark_1_2_Safety_DCT,flexPoolId=MuseSpark_1_2_Safety_DCT,jobPriority=NORMAL,hpcJobPriorityBand=REGULAR,modelTypeName=gen_ai_conda,jobType=OFFLINE_TRAINING,maxJobFailures=0,opecTag=DEDICATED_ONLY,monitoringConfig=GENAI,localityConstraints=region;lco,forceSingleRegion=True,useStrictName=True,use_caf=False,checkpointAgentPkg=$CHECKPOINT_AGENT_FBPKG_ID"
```

## Preflight

Run the tracked configuration-only preflight with CUDA hidden. It explicitly
bypasses the MXFP8 capability probe only while constructing config objects; the
launcher separately validates an actual GB300 before training.

```bash
cd "$RUNTIME_ROOT/src/torchtitan"
CUDA_VISIBLE_DEVICES='' "$RUNTIME_ROOT/bin/python" \
  scripts/dsv3_671b_dist_moe_256gpu/preflight_configs.py \
  | tee /path/to/artifacts/config-preflight.json

cmp \
  "$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/mast_configs.py" \
  "$RUNTIME_ROOT/src/torchtitan/scripts/dsv3_671b_dist_moe_256gpu/mast_configs.py"
cmp \
  "$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/launcher/mast_configs.py" \
  "$LAUNCHER_SOURCE/mast_configs.py"
sha256sum \
  "$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/mast_configs.py" \
  "$RUNTIME_ROOT/src/torchtitan/scripts/dsv3_671b_dist_moe_256gpu/mast_configs.py" \
  "$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/launcher/mast_configs.py" \
  "$LAUNCHER_SOURCE/mast_configs.py" \
  | tee /path/to/artifacts/recipe-sha256.txt
```

The JSON validates all geometry, MTP and auxiliary-loss depths, round-robin
routing, input paths, outer CUDA graphs, profiler schedules, eager deferred
reduction, and GraphTrainer WGrad/reduction ownership. It also exposes the
ignored PP1 `num_pp_microbatches=240` value and the effective count of 16.

## Verify the launcher and submit the four final jobs

Fetch the launcher into an empty directory. Verify every packaged payload entry,
the manifest's separately recorded digest, all pinned launcher dependencies,
and the absence of unresolved substitution tokens before invoking TorchX:

```bash
export LAUNCHER_DIR=/path/to/large/tmp/final-launcher-fetch
test ! -e "$LAUNCHER_DIR"
mkdir -p "$LAUNCHER_DIR"
fbpkg fetch --dest "$LAUNCHER_DIR" --extract --verify \
  --unexpected-fails-verify "$FINAL_LAUNCHER_FBPKG_ID"
test "$(sha256sum "$LAUNCHER_DIR/launcher_manifest.json" | awk '{print $1}')" = \
  "$LAUNCHER_MANIFEST_SHA256"
cmp "$LAUNCHER_SOURCE/launcher_manifest.json" \
  "$LAUNCHER_DIR/launcher_manifest.json"
python3 \
  "$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/launcher_manifest.py" \
  verify "$LAUNCHER_DIR" "$LAUNCHER_DIR/launcher_manifest.json" \
  --allow-fbpkg-fetch-metadata
cmp \
  "$TORCHTITAN_SOURCE/scripts/dsv3_671b_dist_moe_256gpu/launcher/mast_configs.py" \
  "$LAUNCHER_DIR/mast_configs.py"
cmp "$LAUNCHER_SOURCE/launcher_fbpkg_dependencies.txt" \
  "$LAUNCHER_DIR/launcher_fbpkg_dependencies.txt"
while IFS= read -r dependency_id; do
  rg -F --quiet "$dependency_id" "$LAUNCHER_DIR/mast.py"
done <"$LAUNCHER_DIR/launcher_fbpkg_dependencies.txt"
EXPECTED_ROLE_IMAGE_JSON=$(
  {
    printf '%s\n' "$FINAL_LAUNCHER_FBPKG_ID" "$FINAL_RUNTIME_FBPKG_ID"
    cat "$LAUNCHER_DIR/launcher_fbpkg_dependencies.txt"
  } | jq -Rsc 'split("\n") | map(select(length > 0)) | sort'
)
export EXPECTED_ROLE_IMAGE_JSON
if rg -n '@[A-Z][A-Z0-9_]+@|TO_BE_(FROZEN|PUBLISHED)' \
  "$LAUNCHER_DIR/.torchxconfig" \
  "$LAUNCHER_DIR/bha_four_nvl_domains.json" \
  "$LAUNCHER_DIR/cc_wrapper.sh" \
  "$LAUNCHER_DIR/cxx_wrapper.sh" \
  "$LAUNCHER_DIR/launcher_fbpkg_dependencies.txt" \
  "$LAUNCHER_DIR/launcher_manifest.json" \
  "$LAUNCHER_DIR/mast.py" \
  "$LAUNCHER_DIR/mast_configs.py" \
  "$LAUNCHER_DIR/mount.sh" \
  "$LAUNCHER_DIR/run.sh"; then
  echo "Fetched launcher contains an unresolved substitution token" >&2
  exit 1
fi
cd "$LAUNCHER_DIR"
```

Materialize and retain eight separate dryruns. The component and JSON checks
enforce 64 replicas, four GB300s per replica, the requested config, and zero
retries before any 256-GPU job is submitted:

```bash
export RUN_ATTEMPT=1
LAUNCH_USER=$(python3 -c 'import getpass; print(getpass.getuser())')
export LAUNCH_USER
test ! -e "$RELEASE_ARTIFACTS/dryruns-r$RUN_ATTEMPT"
mkdir -p "$RELEASE_ARTIFACTS/dryruns-r$RUN_ATTEMPT"
while IFS='|' read -r label config_name; do
  job_name="final-$label-r$RUN_ATTEMPT"
  dryrun="$RELEASE_ARTIFACTS/dryruns-r$RUN_ATTEMPT/$label.json"
  test ! -e "$dryrun"
  torchx run --dryrun --json -s mast_conda -cfg "$SCHEDULER_CFG" \
    mast.py:train --module_name mast_configs --config_name "$config_name" \
    --nodes 64 --nproc_per_node 4 --retries 0 --h gb300 \
    --name "$job_name" | tee "$dryrun"
  jq -e --arg config "$config_name" \
    --arg checkpoint_agent "$CHECKPOINT_AGENT_FBPKG_ID" \
    --arg runtime "$FINAL_RUNTIME_FBPKG_ID" \
    --arg launcher "$FINAL_LAUNCHER_FBPKG_ID" \
    --arg app_name "$job_name-256-$LAUNCH_USER" \
    --argjson expected_role_image "$EXPECTED_ROLE_IMAGE_JSON" \
    '.scheduler == "mast_conda"
     and .app.name == $app_name
     and .cfg.conda_fbpkg_id == $runtime
     and .cfg.workspace_fbpkg_id == $launcher
     and .cfg.conda_path_in_fbpkg == "conda"
     and .cfg.activate_conda == false
     and .cfg.git == false
     and .cfg.hpcIdentity == "pytorch_distributed"
     and .cfg.hpcJobOncall == "meta_conda"
     and .cfg.hpcClusterUuid == "MastGenAICluster"
     and .cfg.rmAttribution == "MuseSpark_1_2_Safety_DCT"
     and .cfg.flexPoolId == "MuseSpark_1_2_Safety_DCT"
     and .cfg.jobPriority == "NORMAL"
     and .cfg.hpcJobPriorityBand == "REGULAR"
     and .cfg.modelTypeName == "gen_ai_conda"
     and .cfg.jobType == "OFFLINE_TRAINING"
     and .cfg.maxJobFailures == 0
     and .cfg.opecTag == "DEDICATED_ONLY"
     and .cfg.monitoringConfig == "GENAI"
     and .cfg.localityConstraints == ["region", "lco"]
     and .cfg.forceSingleRegion == true
     and .cfg.useStrictName == true
     and .cfg.use_caf == false
     and .cfg.checkpointAgentPkg == $checkpoint_agent
     and .app.roles[0].num_replicas == 64
     and .app.roles[0].resource.gpu == 4
     and .app.roles[0].resource.tags["torchx/named_resources.name"] == "gb300"
     and .app.roles[0].max_retries == 0
     and ([.app.roles[0].image | split(";")[]] | sort)
         == $expected_role_image
     and ([.app.roles[0].image | split(";")[]
           | test(":(stable|default|latest)$")] | any | not)
     and .app.roles[0].env.CONFIG == $config' "$dryrun" >/dev/null
done <<'EOF'
perf-eager-cc-mtp1|deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance
perf-gt-cc-mtp1|graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance
perf-eager-sanket-mtp1|deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance
perf-gt-sanket-mtp1|graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_performance
profile-eager-cc-mtp1|deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile
profile-gt-cc-mtp1|graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile
profile-eager-sanket-mtp1|deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile
profile-gt-sanket-mtp1|graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile
EOF
sha256sum "$RELEASE_ARTIFACTS/dryruns-r$RUN_ATTEMPT"/*.json \
  >"$RELEASE_ARTIFACTS/dryruns-r$RUN_ATTEMPT-sha256.txt"
```

The launcher's existing small-run paths are safe because neither initializes the
671B model. The 1x4 preflight constructs all eight configs and validates the
package on one GB300 host; the 2x4 distributed smoke repeats that validation and
performs only an eight-rank NCCL all-reduce. These are package and fabric checks,
not training or numerical-success claims. Submit and require both to succeed
before the full jobs:

```bash
submit_job() {
  local label=$1
  local config_name=$2
  local nodes=$3
  local env_value=${4:-}
  local job_name="final-$label-r$RUN_ATTEMPT"
  local output="$RELEASE_ARTIFACTS/$label-r$RUN_ATTEMPT-submit.json"
  local -a env_args=()
  test ! -e "$output"
  if [[ -n "$env_value" ]]; then
    env_args=(--env "$env_value")
  fi
  torchx run --json -s mast_conda -cfg "$SCHEDULER_CFG" mast.py:train \
    --module_name mast_configs --config_name "$config_name" \
    --nodes "$nodes" --nproc_per_node 4 --retries 0 --h gb300 \
    --name "$job_name" "${env_args[@]}" | tee "$output" >&2
  local handle app_id ui_url
  handle=$(jq -er '.handle | select(startswith("mast_conda://"))' "$output")
  app_id=$(jq -er '.app_id | select(type == "string" and length > 0)' "$output")
  ui_url=$(jq -er '.ui_url | select(startswith("https://"))' "$output")
  jq -e --arg handle "$handle" --arg app_id "$app_id" --arg ui_url "$ui_url" \
    '.handle == $handle
     and .handle == ("mast_conda://torchx/" + .app_id)
     and .app_id == $app_id
     and .ui_url == $ui_url' "$output" >/dev/null
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$label" "$config_name" "$job_name" "$handle" "$app_id" "$ui_url"
}

monitor_handles() {
  local handles_file=$1
  local status_dir=$2
  mkdir -p "$status_dir"
  while true; do
    local remaining=0
    while IFS=$'\t' read -r label config_name job_name handle app_id ui_url; do
      local latest="$status_dir/$label-latest.json"
      if ! torchx status --json "$handle" >"$latest.tmp"; then
        remaining=$((remaining + 1))
        continue
      fi
      mv "$latest.tmp" "$latest"
      jq -e --arg handle "$handle" --arg app_id "$app_id" --arg ui_url "$ui_url" \
        '.handle == $handle and .app_id == $app_id and .ui_url == $ui_url' \
        "$latest" >/dev/null
      local state
      state=$(jq -er '.state' "$latest")
      jq -c --arg observed_at "$(date --iso-8601=seconds)" \
        --arg label "$label" --arg handle "$handle" --arg app_id "$app_id" \
        --arg ui_url "$ui_url" \
        '{observed_at: $observed_at, label: $label, handle: $handle,
          app_id: $app_id, ui_url: $ui_url, status: .}' \
        "$latest" >>"$status_dir/history.jsonl"
      case "$state" in
        SUCCEEDED) ;;
        FAILED|CANCELLED)
          echo "$label reached $state: $handle" >&2
          return 1
          ;;
        *) remaining=$((remaining + 1)) ;;
      esac
    done <"$handles_file"
    if [[ "$remaining" -eq 0 ]]; then
      return 0
    fi
    sleep 60
  done
}

SMOKE_HANDLES="$RELEASE_ARTIFACTS/smoke-handles-r$RUN_ATTEMPT.tsv"
test ! -e "$SMOKE_HANDLES"
submit_job preflight-1x4 \
  deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance \
  1 PREFLIGHT_ONLY=1 >>"$SMOKE_HANDLES"
submit_job distributed-smoke-2x4 \
  deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_performance \
  2 DISTRIBUTED_SMOKE_ONLY=1 >>"$SMOKE_HANDLES"
monitor_handles "$SMOKE_HANDLES" \
  "$RELEASE_ARTIFACTS/smoke-status-r$RUN_ATTEMPT"
```

Submit each combined performance-and-profile job once and capture its exact app
handle from TorchX's JSON.
Every name carries the explicit `rN` attempt. If a failure requires a code,
config, runtime, or launcher change, create a new immutable source/artifact
release as applicable, increment `RUN_ATTEMPT`, regenerate all eight dryruns,
and use a new handles file. Never overwrite an earlier attempt or resubmit an
unchanged deterministic failure.

```bash
FINAL_HANDLES="$RELEASE_ARTIFACTS/final-handles-r$RUN_ATTEMPT.tsv"
test ! -e "$FINAL_HANDLES"
while IFS='|' read -r label config_name; do
  submit_job "$label" "$config_name" 64 >>"$FINAL_HANDLES"
done <<'EOF'
profile-eager-cc-mtp1|deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile
profile-gt-cc-mtp1|graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_chien_chin_mtp1_256gpu_profile
profile-eager-sanket-mtp1|deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile
profile-gt-sanket-mtp1|graph_trainer_deepseek_v3_671b_dist_moe_mxfp8_sanket_final_mtp1_256gpu_profile
EOF
```

TorchX appends the GPU count and user to each MAST name. Poll the four exact
captured handles, not a mutable scheduler listing, at one-minute cadence. The
monitor stops immediately on a terminal failure so the first failing rank can
be inspected before any retry:

```bash
monitor_handles "$FINAL_HANDLES" \
  "$RELEASE_ARTIFACTS/final-status-r$RUN_ATTEMPT"
```

## Collect results and profiles

For each successful run, save:

- MAST run URL, app handle, source commit, full runtime and launcher fbpkg
  identifiers, and
  recipe SHA-256;
- every console-rounded loss and grad norm, tokens/s/GPU, TFLOP/s/GPU, and
  memory sample;
- the exact clean step window used for mean and standard deviation;
- CUDA-graph capture/replay evidence;
- rank-0 trace for PP1;
- rank-0 and rank-128 traces for PP2.

The final MAST recipes do not emit TensorBoard scalars, and their console output
rounds loss and grad norm. Treat those values only as finite training-health
evidence. Full-precision loss/grad-norm equality and exact gradient evidence
come from the five source and five packaged paired gates above, never from the
MAST logs.

Predeclare one identical clean window for each eager/GraphTrainer pair. For PP1,
use intervals ending at steps 20, 30, 40, and 60; interval 50 includes the
iteration-41 profile and memory snapshot. For PP2, use intervals ending at steps
20, 30, and 60; interval 40 includes profiler warmup and the memory snapshot,
and interval 50 includes the remaining warmup, recording, and trace export.
Step 10 is startup. Do not change either window after seeing results. Compute
the reported variability as sample standard deviation (denominator `n - 1`).

Report aggregate tokens/s as per-GPU tokens/s times 256. TorchTitan's GB300 BF16
peak is 2.5 PFLOP/s, so if MFU is absent derive it as
`100 * TFLOP/s/GPU / 2500`. Never mix historical CoreWeave values into the
same-cluster baseline delta.

Keep raw traces for measurement and compacted copies only for visualization.
Use four distinct source paths so the eager and GraphTrainer runs can never
overwrite or masquerade as one another. The commands below retain hashes
for both PP1 rank-0 traces and all four PP2 rank-0/rank-128 traces, compact each
one independently, and make separate eager and GraphTrainer PP timelines:

```bash
export COMPACTOR="$TORCHTITAN_SOURCE/.claude/skills/cuda_graph_trace_compaction/scripts/compact_cuda_graph_trace.py"
export EAGER_PP1_TRACE_DIR=/path/to/final-profile-eager-cc-mtp1/profiling/traces/iteration_41
export GT_PP1_TRACE_DIR=/path/to/final-profile-gt-cc-mtp1/profiling/traces/iteration_41
export EAGER_PP2_TRACE_DIR=/path/to/final-profile-eager-sanket-mtp1/profiling/traces/iteration_43
export GT_PP2_TRACE_DIR=/path/to/final-profile-gt-sanket-mtp1/profiling/traces/iteration_43
export TRACE_ARTIFACTS=/path/to/artifacts/traces
test ! -e "$TRACE_ARTIFACTS"
mkdir -p \
  "$TRACE_ARTIFACTS/eager-pp2-pair" \
  "$TRACE_ARTIFACTS/gt-pp2-pair"

sha256sum \
  "$EAGER_PP1_TRACE_DIR/rank0_trace.json.gz" \
  "$GT_PP1_TRACE_DIR/rank0_trace.json.gz" \
  "$EAGER_PP2_TRACE_DIR/rank0_trace.json.gz" \
  "$EAGER_PP2_TRACE_DIR/rank128_trace.json.gz" \
  "$GT_PP2_TRACE_DIR/rank0_trace.json.gz" \
  "$GT_PP2_TRACE_DIR/rank128_trace.json.gz" \
  | tee "$TRACE_ARTIFACTS/raw-trace-sha256.txt"

python3 "$COMPACTOR" "$EAGER_PP1_TRACE_DIR/rank0_trace.json.gz" \
  -o "$TRACE_ARTIFACTS/eager-pp1-rank0-compacted.json.gz" \
  | tee "$TRACE_ARTIFACTS/eager-pp1-compaction.json"
python3 "$COMPACTOR" "$GT_PP1_TRACE_DIR/rank0_trace.json.gz" \
  -o "$TRACE_ARTIFACTS/gt-pp1-rank0-compacted.json.gz" \
  | tee "$TRACE_ARTIFACTS/gt-pp1-compaction.json"
python3 "$COMPACTOR" "$EAGER_PP2_TRACE_DIR/rank0_trace.json.gz" \
  -o "$TRACE_ARTIFACTS/eager-pp2-rank0-compacted.json.gz" \
  | tee "$TRACE_ARTIFACTS/eager-pp2-rank0-compaction.json"
python3 "$COMPACTOR" "$EAGER_PP2_TRACE_DIR/rank128_trace.json.gz" \
  -o "$TRACE_ARTIFACTS/eager-pp2-rank128-compacted.json.gz" \
  | tee "$TRACE_ARTIFACTS/eager-pp2-rank128-compaction.json"
python3 "$COMPACTOR" "$GT_PP2_TRACE_DIR/rank0_trace.json.gz" \
  -o "$TRACE_ARTIFACTS/gt-pp2-rank0-compacted.json.gz" \
  | tee "$TRACE_ARTIFACTS/gt-pp2-rank0-compaction.json"
python3 "$COMPACTOR" "$GT_PP2_TRACE_DIR/rank128_trace.json.gz" \
  -o "$TRACE_ARTIFACTS/gt-pp2-rank128-compacted.json.gz" \
  | tee "$TRACE_ARTIFACTS/gt-pp2-rank128-compaction.json"

ln -s "$EAGER_PP2_TRACE_DIR/rank0_trace.json.gz" \
  "$TRACE_ARTIFACTS/eager-pp2-pair/rank0_trace.json.gz"
ln -s "$EAGER_PP2_TRACE_DIR/rank128_trace.json.gz" \
  "$TRACE_ARTIFACTS/eager-pp2-pair/rank128_trace.json.gz"
ln -s "$GT_PP2_TRACE_DIR/rank0_trace.json.gz" \
  "$TRACE_ARTIFACTS/gt-pp2-pair/rank0_trace.json.gz"
ln -s "$GT_PP2_TRACE_DIR/rank128_trace.json.gz" \
  "$TRACE_ARTIFACTS/gt-pp2-pair/rank128_trace.json.gz"
python3 "$COMPACTOR" "$TRACE_ARTIFACTS/eager-pp2-pair" --merge-pp-ranks \
  -o "$TRACE_ARTIFACTS/eager-pp2-ranks0-128-merged-compacted.json.gz" \
  | tee "$TRACE_ARTIFACTS/eager-pp2-merge.json"
python3 "$COMPACTOR" "$TRACE_ARTIFACTS/gt-pp2-pair" --merge-pp-ranks \
  -o "$TRACE_ARTIFACTS/gt-pp2-ranks0-128-merged-compacted.json.gz" \
  | tee "$TRACE_ARTIFACTS/gt-pp2-merge.json"
```

Publish the retained traces with the internal Perfetto uploader. Capture its
stdout because it contains the share link:

```bash
export SHARE_TRACE="$HOME/fbsource/arvr/scripts/perfetto/share_trace.py"
test -f "$SHARE_TRACE"
for trace in \
  "$EAGER_PP1_TRACE_DIR/rank0_trace.json.gz" \
  "$GT_PP1_TRACE_DIR/rank0_trace.json.gz" \
  "$EAGER_PP2_TRACE_DIR/rank0_trace.json.gz" \
  "$EAGER_PP2_TRACE_DIR/rank128_trace.json.gz" \
  "$GT_PP2_TRACE_DIR/rank0_trace.json.gz" \
  "$GT_PP2_TRACE_DIR/rank128_trace.json.gz" \
  "$TRACE_ARTIFACTS/eager-pp1-rank0-compacted.json.gz" \
  "$TRACE_ARTIFACTS/gt-pp1-rank0-compacted.json.gz" \
  "$TRACE_ARTIFACTS/eager-pp2-rank0-compacted.json.gz" \
  "$TRACE_ARTIFACTS/eager-pp2-rank128-compacted.json.gz" \
  "$TRACE_ARTIFACTS/gt-pp2-rank0-compacted.json.gz" \
  "$TRACE_ARTIFACTS/gt-pp2-rank128-compacted.json.gz" \
  "$TRACE_ARTIFACTS/eager-pp2-ranks0-128-merged-compacted.json.gz" \
  "$TRACE_ARTIFACTS/gt-pp2-ranks0-128-merged-compacted.json.gz"; do
  python3 "$SHARE_TRACE" "$trace" | tee -a "$TRACE_ARTIFACTS/share-links.txt"
done
```

Only links from successful final jobs belong in the report.

## Publish the docs-only report tip

After all gates and successful runs are recorded, edit only this runbook and
its report. Amend the single top `[not-for-land]` commit so R still has exactly
21 commits above the same base and the same 20 landable commits as S. The
explicit force-with-lease permits only the expected S-to-R report-branch
replacement; it cannot overwrite an independently changed remote branch.

```bash
cd "$FINAL_REPORT_CHECKOUT"
test "$(git branch --show-current)" = "$FINAL_REPORT_BRANCH"
test "$(git rev-parse HEAD)" = "$RUNTIME_SOURCE_COMMIT"
test "$(git remote get-url --push "$FORK_REMOTE")" = "$FORK_REPOSITORY"
git diff --check
git diff --name-only "$RUNTIME_SOURCE_COMMIT" \
  | LC_ALL=C sort >"$RELEASE_ARTIFACTS/report-changed-paths.txt"
diff -u - "$RELEASE_ARTIFACTS/report-changed-paths.txt" <<'EOF'
docs/dsv3_671b_dist_moe_256gpu_report.md
docs/dsv3_671b_dist_moe_256gpu_runbook.md
EOF
git add \
  docs/dsv3_671b_dist_moe_256gpu_report.md \
  docs/dsv3_671b_dist_moe_256gpu_runbook.md
git diff --cached --check
git commit --amend --no-edit
FINAL_REPORT_COMMIT=$(git rev-parse HEAD)
export FINAL_REPORT_COMMIT
test "$FINAL_REPORT_COMMIT" != "$RUNTIME_SOURCE_COMMIT"
test "$(git rev-parse "$FINAL_REPORT_COMMIT^")" = \
  "$(git rev-parse "$RUNTIME_SOURCE_COMMIT^")"
test "$(git merge-base "$FINAL_BASE_COMMIT" "$FINAL_REPORT_COMMIT")" = \
  "$FINAL_BASE_COMMIT"
test "$(git rev-list --count \
  "$FINAL_BASE_COMMIT..$FINAL_REPORT_COMMIT")" -eq 21
test "$(git log -1 --format=%s "$FINAL_REPORT_COMMIT")" = \
  "[not-for-land] Add DistMoE 256-GPU reproduction package"
test "$(git log --format=%s "$FINAL_BASE_COMMIT..$FINAL_REPORT_COMMIT" \
  | awk '/^\[not-for-land\]/{count++} END{print count+0}')" -eq 1
git diff --name-only "$RUNTIME_SOURCE_COMMIT" "$FINAL_REPORT_COMMIT" \
  | LC_ALL=C sort >"$RELEASE_ARTIFACTS/report-commit-changed-paths.txt"
diff -u - "$RELEASE_ARTIFACTS/report-commit-changed-paths.txt" <<'EOF'
docs/dsv3_671b_dist_moe_256gpu_report.md
docs/dsv3_671b_dist_moe_256gpu_runbook.md
EOF
test -z "$(git status --porcelain)"

if git show-ref --verify --quiet "$FINAL_REPORT_REF"; then
  echo "Ref already exists locally: $FINAL_REPORT_REF" >&2
  exit 1
fi
if [[ -n "$(GIT_CONFIG_COUNT=0 git ls-remote \
  "$FORK_PUSH_REPOSITORY" "$FINAL_REPORT_REF")" ]]; then
  echo "Ref already exists remotely: $FINAL_REPORT_REF" >&2
  exit 1
fi
git tag -a "${FINAL_REPORT_REF#refs/tags/}" "$FINAL_REPORT_COMMIT" \
  -m "DistMoE 256-GPU immutable report v5"
GIT_CONFIG_COUNT=0 git push --atomic \
  --force-with-lease="refs/heads/$FINAL_REPORT_BRANCH:$RUNTIME_SOURCE_COMMIT" \
  "$FORK_PUSH_REPOSITORY" \
  "$FINAL_REPORT_COMMIT:refs/heads/$FINAL_REPORT_BRANCH" \
  "$FINAL_REPORT_REF"
test "$(GIT_CONFIG_COUNT=0 git ls-remote "$FORK_PUSH_REPOSITORY" \
  "refs/heads/$FINAL_REPORT_BRANCH" | awk '{print $1}')" = \
  "$FINAL_REPORT_COMMIT"
test "$(GIT_CONFIG_COUNT=0 git ls-remote "$FORK_PUSH_REPOSITORY" \
  "$FINAL_REPORT_REF^{}" | awk '{print $1}')" = \
  "$FINAL_REPORT_COMMIT"
test "$(GIT_CONFIG_COUNT=0 git ls-remote "$FORK_PUSH_REPOSITORY" \
  "$RUNTIME_SOURCE_REF^{}" | awk '{print $1}')" = \
  "$RUNTIME_SOURCE_COMMIT"
```

## Final report checklist

The report is complete only when it contains:

1. exactly four clearly named performance lines: Baseline Chien-Chin, Test
   GraphTrainer Chien-Chin, Baseline Sanket, and Test GraphTrainer Sanket;
2. matched same-cluster eager deltas for both pairs;
3. paired-gate full-precision loss/grad-norm and gradient-hash evidence, MAST
   finite rounded health values, TPS, sample standard deviation, aggregate TPS,
   TFLOP/s/GPU, MFU, and peak memory;
4. successful MAST links only;
5. final `share_trace.py` links only;
6. final branch, commit stack, full runtime and launcher identifiers, recipe
   hash, and all
   launch commands needed by a reader without access to the build machine.
