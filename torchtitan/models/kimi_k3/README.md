# Kimi K3

Kimi K3 combines a hybrid Kimi Delta Attention (KDA) and Multi-head Latent
Attention (MLA) decoder with LatentMoE and a MoonViT-V2 vision encoder.

## Prerequisites

Install the additional dependencies:

```bash
pip install -r .ci/docker/requirements-vlm.txt
```

## Architecture

Kimi K3 is built on Kimi Delta Attention (KDA) and Attention Residuals
(AttnRes), with 69 KDA layers and 24 Gated MLA layers. Stable LatentMoE selects
16 of 896 experts per token, and MoonViT-V2 provides native vision input.

## Released Model Configuration

The values below follow the
[official Kimi K3 model card](https://huggingface.co/moonshotai/Kimi-K3) and
describe the released model.

| Component | Configuration |
|-----------|---------------|
| Architecture | Mixture-of-Experts (MoE) |
| Parameters | 2.8T total, 104B activated |
| Decoder | 93 layers, 1 dense layer, hidden size 7168, 96 attention heads |
| Attention | 69 KDA layers and 24 Gated MLA layers, context length 1048576 |
| LatentMoE | Dimension 3584, hidden size 3072 per expert, 896 experts, top-16 routing, 2 shared experts |
| Vocabulary | 160K |
| Activation | SiTU-GLU |
| Vision encoder | MoonViT-V2, 401M parameters |
| Quantization | MXFP4 weights and MXFP8 activations with quantization-aware training |
| Modality | Text and image |

## Supported Parallelisms

| Feature | Notes |
|---------|-------|
| FSDP2 / HSDP | Decoder sharded per layer; vision encoder sharded as a separate unit |

## Numerical Parity

The parity script reduces the released Hugging Face configuration to match
TorchTitan's local `debugmodel` configuration before initializing both models.

End-to-end KL divergence against the Hugging Face implementation (multimodal
inputs): **6.7634e-7**, with **100% top-1 and top-5 match**.

Vision parity: pixel preprocessing max difference **1.192e-7**; projected vision
features cosine similarity **1.000000** and max difference **2.730e-3**.

Test scripts:

- `scripts/checkpoint_conversion/numerical_tests_kimi_k3.py` -- Hugging Face vs.
  TorchTitan comparison
- `tests/unit_tests/gpu/test_kimi_k3.py` -- KDA and FSDP2 correctness

## Running MX QAT

MX QAT uses the same launcher, trainer, data configuration, and optimizer as the
ordinary Kimi debug recipe. Install a compatible TorchAO version that provides
`MXFakeQuantizeConfig` and `mx_fake_quantized_grouped_mm`, then select the QAT
recipe:

```bash
NGPU=1 MODULE=kimi_k3 CONFIG=kimi_k3_debugmodel_mx_qat ./run_train.sh
```

The debug recipe starts from random initialization and uses `EMULATED` by
default. Without a checkpoint it selects all config-eligible expert and dense
projections, keeps BF16 master parameters, and applies MXFP8 activation fake
quantization to grouped experts. No manual expert replacement or parameter-name
list is needed. Prepare the normal Kimi tokenizer and dataset dependencies as
for `kimi_k3_debugmodel`; QAT does not replace that data setup.

For packed checkpoint initialization, write a normal Python run configuration.
Use a checkpoint matching the debug architecture, such as the fixture generated
by `scripts/checkpoint_conversion/create_kimi_k3_mxfp4_fixture.py`; the full
released checkpoint does not match the debug model.

```python
# my_kimi_runs.py
from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel_mx_qat


def qat():
    config = kimi_k3_debugmodel_mx_qat(
        checkpoint_path="/absolute/path/to/packed-debug-checkpoint",
    )
    config.training.steps = 100
    # Set tokenizer, data, and parallelism here, just as for a normal run.
    return config
```

```bash
NGPU=1 MODULE=my_kimi_runs CONFIG=qat ./run_train.sh
```

`checkpoint_path` sets the existing HF and quantized-load options together and
reads `config.json` plus `model.safetensors.index.json` before selecting QAT
modules. The config defines eligible weights; only actual packed/scale pairs in
the index select QAT. Eligible weights stored in BF16 remain ordinary weights.
The released checkpoint packs routed expert matrices while residual and fused
projection weights remain BF16. The synthetic debug fixture deliberately packs
all eligible weights, so its selected set can differ from the release.

The loader validates the same manifest-derived policy against the selected QAT
modules and actual tensor headers. Missing pairs, orphan scales, and pairs
outside the config policy remain errors. A grouped parameter mixing packed and
BF16 experts can be imported, but cannot be selected for grouped QAT.

For native grouped execution, pass the existing TorchAO configurations to the
same recipe function; both `kernel_preference` values must agree:

```python
import torch
from torchao.prototype.qat import MXFakeQuantizeConfig
from torchao.quantization.quantize_.common import KernelPreference

config = kimi_k3_debugmodel_mx_qat(
    weight_fake_quant_config=MXFakeQuantizeConfig(
        dtype=torch.float4_e2m1fn_x2, kernel_preference=KernelPreference.AUTO,
    ),
    activation_fake_quant_config=MXFakeQuantizeConfig(
        dtype=torch.float8_e4m3fn, kernel_preference=KernelPreference.AUTO,
    ),
)
```

`AUTO` requires supported SM100 CUDA kernels and currently at most 32 local
experts. `EMULATED` uses dequantized operands for GEMM. Backend configuration
changes belong in the recipe; the model launch command stays the same.

## Released checkpoint preflight

Before allocating a training job, check the safetensors headers against the
model's HF tensor shapes, including ordinary non-quantized tensors:

```bash
python -m scripts.checkpoint_conversion.validate_kimi_k3_mxfp4_checkpoint \
  --checkpoint /absolute/path/to/Kimi-K3 --model-flavor Kimi-K3
```

The preflight constructs the model on the meta device and uses the same storage
reader as training. It validates packed/scale pairs and every expected logical
tensor shape. It reads only headers and the small padding tails, not full model
weights. Use `--model-flavor debugmodel` for a matching synthetic fixture, or
`--unquantized` for ordinary HF weights.

The released 96-head KDA checkpoint stores `A_log` as a 128-element vector.
Loading accepts either the canonical 96-element vector or that exact padded
shape, and requires all 32 discarded values to be zero. The reader exposes 96
elements before DCP planning; model parameters and HF export remain canonical.
This rule also applies to unquantized HF imports. Other unexpected shapes or
nonzero padding are errors. See the [release inspection](https://huggingface.co/moonshotai/Kimi-K3/discussions/150).

Shape preflight does not establish numerical equivalence or successful
released-model training. Those require separate evaluation runs.
