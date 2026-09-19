# Kimi K3

Kimi K3 combines a hybrid Kimi Delta Attention (KDA) and Multi-head Latent
Attention (MLA) decoder with LatentMoE and a MoonViT-V2 vision encoder.

## Prerequisites

Install the additional dependencies:

```bash
pip install -r .ci/docker/requirements-vlm.txt
pip install -r scripts/checkpoint_conversion/requirements_kimi_k3.txt
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
| Tensor and sequence parallelism | Validated with FSDP2/TP2/EP2 in the B200 integration suite |
| Expert parallelism | Standard PyTorch all-to-all; validated with FSDP2/EP2 |

## Numerical Parity

The following manual checks cover the reduced Kimi K3 model:

| Check | Coverage |
|------|----------|
| Moonshot BF16 forward parity | Pinned Kimi K3 revision, seed 42, independent multimodal preprocessing, vision features, MoE routes, and final logits over the debug model |
| KDA forward and backward parity | BF16 fused kernel and eager implementation against an FP64 recurrence at 1, 63, 64, and 65 tokens |
| Training loss and gradient norm | Ten deterministic FSDP2/EP2 steps checked by `loss_compare.py` in the B200 integration suite |

Run the checks on an SM100 or SM103 system:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.checkpoint_conversion.numerical_tests_kimi_k3
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tests/unit_tests/gpu/test_kimi_k3.py::TestKimiK3::test_attention_gym_kda_kernel_matches_recurrent_reference
```

The Moonshot comparison reduces the released configuration to TorchTitan's
`debugmodel` dimensions and transfers one initialized state dict into both
implementations. It pins the upstream code revision and all reference-only
Python packages and reports preprocessing, vision, routing, and final-logit
metrics. At full random-initialized depth, small BF16 differences can change
near-tied MoE routes and compound across layers.
