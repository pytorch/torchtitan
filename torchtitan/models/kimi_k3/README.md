# Kimi K3

Kimi K3 combines a hybrid Kimi Delta Attention (KDA) and Multi-head Latent
Attention (MLA) decoder with LatentMoE and a MoonViT-V2 vision encoder.

## Prerequisites

Install the additional dependencies:

```bash
pip install av einops pillow torchvision flash-linear-attention
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

`scripts/checkpoint_conversion/numerical_tests_kimi_k3.py` reduces the released
Hugging Face config to the debug model and compares both implementations on the
same text+image prompt, each side doing its own preprocessing. Float32 results:

| Stage | Result |
|-------|--------|
| Pixel preprocessing | max difference `1.192e-7` |
| Projected vision features | cosine `1.000000`, max difference `3.152e-3` |
| MoE routing | `3936 / 3936` expert choices match |
| Last-token logits | KL `1.8215e-8`, top-1 match, top-5 `5 / 5` |

Routed experts always run through the bf16 grouped GEMM, so the float32 logit
difference is bounded by bf16 rather than by float32.

`tests/unit_tests/test_kimi_k3.py` covers the KDA kernel against a recurrent
reference, the HuggingFace state-dict round trip, a multimodal forward, and
FSDP2 forward/backward parity against non-distributed execution.
