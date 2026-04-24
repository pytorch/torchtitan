# FLUX.2

**Frontier Visual Intelligence** — State-of-the-art image generation and editing from [Black Forest Labs](https://bfl.ai).

---

<p align="center">
<a href="https://docs.bfl.ai">API Docs</a> •
<a href="https://huggingface.co/black-forest-labs">Hugging Face</a> •
<a href="https://bfl.ai/blog">Blog</a>
</p>

This repo contains minimal inference code to run image generation & editing with our FLUX.2 open-weight models.

## News

- **[15.01.2026]** Today, we release the FLUX.2 [klein] family of models, our fastest models yet. Sub-second generation on consumer GPUs. Read more about it in our [blog post](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence).
- **[25.11.2025]** We are releasing FLUX.2 [dev], a 32B parameter model for text-to-image generation, and image editing (single reference image and multiple reference images).

## Model Overview

| Name | Step-distilled | Guidance-distilled | Text-to-Image | Image Editing (Single reference) | Image Editing (Multi-reference) | License |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| [FLUX.2 [klein] 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) | ✅ | ✅ | ✅ | ✅ | ✅ | [apache-2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| [FLUX.2 [klein] 9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | ✅ | ✅ | ✅ | ✅ | ✅ | [FLUX Non-Commercial License](model_licenses/LICENSE-FLUX-NON-COMMERICAL) |
| [FLUX.2 [klein] 4B Base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) | ❌ | ❌ | ✅ | ✅ | ✅ | [apache-2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) |
| [FLUX.2 [klein] 9B Base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) | ❌ | ❌ | ✅ | ✅ | ✅ | [FLUX Non-Commercial License](model_licenses/LICENSE-FLUX-NON-COMMERICAL) |
| [FLUX.2 [dev]](https://huggingface.co/black-forest-labs/FLUX.2-dev) | ❌ | ✅ | ✅ | ✅ | ✅ | [FLUX Non-Commercial License](model_licenses/LICENSE-FLUX-NON-COMMERICAL) |

**All models support**: Text-to-Image ✅ | Single-ref Editing ✅ | Multi-ref Editing ✅

## Which Model Should I Use?

| Need | Recommended |
|------|-------------|
| Real-time apps, interactive workflows | [klein] 4B or 9B (distilled) |
| Consumer GPU (e.g. RTX 3090/4070) | [klein] 4B |
| Fine-tuning, LoRA training | [klein] Base or FLUX.2 [dev] |
| Maximum quality, no latency constraints | FLUX.2 [dev] |

## `FLUX.2 [klein]`

FLUX.2 [klein] is our fastest model family — generating and editing (multiple) images in under a second without sacrificing quality. Built for real-time applications, creative iteration, and deployment on consumer hardware.

### Key Capabilities
- **Sub-second inference** — Generate or edit images under a second on modern hardware
- **Unified generation & editing** — Text-to-image, image editing, and multi-reference in one model
- **Runs on consumer GPUs** — Klein 4B fits in ~8GB VRAM (RTX 3090/4070 and up)
- **Apache 2.0 on 4B** — Open-source, fine-tuning, and customization

### Performance

Klein models define the Pareto frontier for quality vs. latency and VRAM across text-to-image, single-reference editing, and multi-reference generation:

<p align="center">
<img src="assets/klein_benchmark.jpg" alt="FLUX.2 [klein] vs Baselines — Elo vs Latency and VRAM" width="800"/>
</p>
<sub>Higher Elo + Lower Latency/VRAM = Better.</sub>

### The Klein Family

| Model | Best For |
|:---|:---|
| **[klein] 4B** | Maximum speed, consumer hardware, edge deployment |
| **[klein] 9B** | Best quality-to-latency ratio, production apps |
| **[klein] 4B Base** | Fine-tuning on limited hardware, full customization |
| **[klein] 9B Base** | Research, LoRA training, maximum output diversity |

**Distilled vs Base:**
- Use **Distilled** (4-step) for production apps and real-time generation
- Use **Base** (50-step) for fine-tuning, LoRA training, and maximum flexibility

**Licensing:** 4B models are [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md). 9B models use the [FLUX.2-dev Non-Commercial License](model_licenses/LICENSE-FLUX-DEV).

### Text-to-image examples

Example focused on realism 
![t2i-klein-grid](assets/t2i_klein_realism.jpg)

Example focused on output diversity
![t2i-klein-others](assets/t2i_klein_others.jpg)

### Editing examples

![i2i-klein](assets/i2i_klein.jpg)

## `FLUX.2 [dev]`

`FLUX.2 [dev]` is a 32B parameter flow matching transformer model capable of generating and editing (multiple) images. The model is released under the [FLUX.2-dev Non-Commercial License](model_licenses/LICENSE-FLUX-DEV) and can be found [here](https://huggingface.co/black-forest-labs/FLUX.2-dev).

Note that the below script for `FLUX.2 [dev]` needs considerable amount of VRAM (H100-equivalent GPU). We partnered with Hugging Face to make quantized versions that run on consumer hardware; below you can find instructions on how to run it on a RTX 4090 with a remote text encoder, for other quantization sizes and combinations, check the [diffusers quantization guide here](docs/flux2_dev_hf.md).

### Text-to-image examples

![t2i-grid](assets/teaser_generation.png)

### Editing examples

![edit-grid](assets/teaser_editing.png)

### Prompt upsampling

`FLUX.2 [dev]` benefits significantly from prompt upsampling. The inference script below offers the option to use both local prompt upsampling with the same model we use for text encoding ([`Mistral-Small-3.2-24B-Instruct-2506`](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506)), or alternatively, use any model on [OpenRouter](https://openrouter.ai/) via an API call.

See the [upsampling guide](docs/flux2_with_prompt_upsampling.md) for additional details and guidance on when to use upsampling.

## `FLUX.2` autoencoder

The FLUX.2 autoencoder has considerably improved over the [FLUX.1 autoencoder](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/ae.safetensors). The autoencoder is released under [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) and can be found [here](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/ae.safetensors). For more information, see our [technical blogpost](https://bfl.ai/research/representation-comparison).

## Local installation

The inference code was tested on GB200 using CUDA 12.9 and Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu129 --no-cache-dir
```

## Run the CLI

Before running the CLI, you may download the weights from [here](https://huggingface.co/black-forest-labs/FLUX.2-dev) and set the following environment variables.

```bash
export FLUX2_MODEL_PATH="<flux2_path>"
export AE_MODEL_PATH="<ae_path>"
export KLEIN_4B_MODEL_PATH="<klein_4b_path>"
export KLEIN_4B_BASE_MODEL_PATH="<klein_4b_base_path>"
export KLEIN_9B_MODEL_PATH="<klein_9b_path>"
export KLEIN_9B_BASE_MODEL_PATH="<klein_9b_base_path>"
```

If you don't set the environment variables, the weights will be downloaded automatically.

You can start an interactive session to do both text to image generation as well as editing (one or multiple) images with the following command:

```bash
PYTHONPATH=src python scripts/cli.py
```

## Watermarking

We've added an option to embed invisible watermarks directly into the generated images
via the [invisible watermark library](https://github.com/ShieldMnt/invisible-watermark).

Additionally, we are recommending implementing a solution to mark the metadata of your outputs, such as [C2PA](https://c2pa.org/)

## Citation

If you find the provided code or models useful for your research, consider citing them as:

```bib
@misc{flux-2-2025,
    author={Black Forest Labs},
    title={{FLUX.2: Frontier Visual Intelligence}},
    year={2025},
    howpublished={\url{https://bfl.ai/blog/flux-2}},
}
```
