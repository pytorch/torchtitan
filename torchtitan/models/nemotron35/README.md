# Nemotron-3 Model Family

Nemotron-3 is a family of hybrid Mamba-Transformer Mixture-of-Experts (MoE) models combining Mamba-2 SSM and Grouped-Query Attention (GQA) layers with granular routed experts.

## Supported Flavors

- `nemotron_debugmodel`: CI/local testing flavor
- `nemotron_4b`: 3.97B dense hybrid model, 42 layers of Mamba-2 / MLP / attention (no MoE)
- `nemotron_31b`: 31.6B total / ~3.2B active Nano MoE model, 52 layers (23 Mamba / 23 MoE / 6 attention, 128 experts, top-6 routing)

## Download Tokenizer

```bash
python scripts/download_hf_assets.py --repo_id nvidia/nemotron-3-nano --assets tokenizer
```

## Training

```bash
# Debug model (used for CI and testing)
MODULE=nemotron3 CONFIG=nemotron_debugmodel ./run_train.sh

# Nemotron-3 4B
MODULE=nemotron3 CONFIG=nemotron_4b ./run_train.sh

# Nemotron-3 Nano 31B
MODULE=nemotron3 CONFIG=nemotron_31b ./run_train.sh
```

## Fused Mamba-2 scan

The Mamba-2 chunked scan uses `mamba-ssm`'s Triton kernel when it is installed
and the model is on GPU, falling back to a pure-PyTorch reference otherwise
(CPU, or no `mamba-ssm`). The kernel is Triton-only, so it runs on both CUDA
and ROCm -- measured on 8x MI355X (gfx950), 31B at 8192 context:

| | PyTorch reference | Fused Triton |
|---|---|---|
| MFU | 0.63% | **2.65%** |
| tokens/sec | 724 | **3,042** |
| peak memory | 114.3 GiB | **67.1 GiB** |

Install it without the Mamba-1 CUDA extension (which is unrelated and does not
build on ROCm):

```bash
MAMBA_FORCE_BUILD=FALSE MAMBA_SKIP_CUDA_BUILD=TRUE pip install --no-build-isolation mamba-ssm
```

Set `NEMOTRON_DISABLE_FUSED_MAMBA=1` to force the reference path (for A/B
timing or bisecting numerics).

## Building `causal-conv1d` on ROCm

This port does **not** use `causal-conv1d` -- the depthwise causal convolution
is a plain `nn.Conv1d`. It is documented here because it is a hard requirement
of HF `transformers`' fully-fused NemotronH path, so it is needed to benchmark
this port against that reference, and because the build failure is badly
misleading.

`pip install causal-conv1d` fails against `torch >= 2.14` on any platform, ROCm
included. The real error is a C++ standard mismatch:

```
torch/all.h:5:2: error: #error C++20 or later compatible compiler is required to use PyTorch.
c10/util/intrusive_ptr.h:  error: 'strong_ordering' in namespace 'std' does not name a type
c10/util/TypeIndex.h:      error: 'string_view' has no member named 'starts_with'
```

`setup.py` hardcodes `-std=c++17` in both the `cxx` and `nvcc`/`hipcc` flag
lists, and `torch.utils.cpp_extension` only appends its own `-std=c++20` when
the extension supplies no `-std` flag -- so the pin wins and the host compile of
`causal_conv1d.cpp` fails on ATen headers, which have required C++20 since
torch 2.14 (pytorch/pytorch#178150).

Two traps when reading that build log. It emits hundreds of
`note: candidate: ... c10::impl::DictIterator ... operator!=` lines, which are
GCC listing candidate overloads *after* the real error -- they are not a C10 API
break, and chasing them leads nowhere. And the failure is **not** ROCm-related:
`hipcc` compiles `causal_conv1d_fwd.hip` and `causal_conv1d_bwd.hip` for gfx950
without complaint. Only the host compile fails. Downgrading torch is therefore
both unnecessary and destructive on ROCm, where the working wheel is a nightly.

The fix is to *remove* the pins rather than raise them, so that
`BuildExtension` keeps choosing the standard that matches the installed torch:

```bash
git clone https://github.com/Dao-AILab/causal-conv1d.git /tmp/ccv1d-build
cd /tmp/ccv1d-build
sed -i 's/, *"-std=c++17"//; /^ *"-std=c++17",$/d' setup.py

export PYTORCH_ROCM_ARCH=gfx950 ROCM_HOME=/opt/rocm MAX_JOBS=32
pip install . --no-build-isolation --no-deps
```

`--no-build-isolation` is required: without it the build environment installs a
fresh **CUDA** torch and the extension compiles against the wrong one (the tell
is `torch.__version__ = 2.14.0+cu130` in the log). `--no-deps` then guarantees
the install phase cannot replace a ROCm torch build. Set `PYTORCH_ROCM_ARCH` to
your own target; `gfx950` is MI355X.

Verified on 8x MI355X (gfx950), `causal_conv1d_fn` with `activation="silu"`
against a `F.conv1d` + `F.silu` reference, forward and backward:

| dtype | forward rel err | input-grad rel err |
|---|---|---|
| float32 | 1.43e-07 | 2.02e-07 |
| bfloat16 | 4.69e-03 | 6.45e-03 |

The bfloat16 deltas sit at bfloat16 epsilon (2^-8 ~ 3.9e-3) and are accumulation
order, not error -- compare in float32 when checking this kernel, or an absolute
tolerance will read as a false failure.

See [`config_registry.py`](./config_registry.py) for available configuration options.

## Checkpoint Conversion

`NemotronStateDictAdapter` supports bidirectional checkpoint conversion between Hugging Face format and TorchTitan DCP via `scripts/checkpoint_conversion/`.
