![Teaser](../assets/teaser_generation.png)
![Teaser](../assets/teaser_editing.png)

`FLUX.2 [dev]` is a 32 billion parameter rectified flow transformer capable of generating, editing and combining images based on text instructions.
For more information, please read our [blog post](https://bfl.ai/blog/flux-2).

# Key Features
1. State of the art in open text-to-image generation, single-reference editing and multi-reference editing.
2. No need for finetuning: character, object and style reference without additional training in one model.
4. Trained using guidance distillation, making `FLUX.2 [dev]` more efficient.
5. Open weights to drive new scientific research, and empower artists to develop innovative workflows.
6. Generated outputs can be used for personal, scientific, and commercial purposes, as described in the [FLUX \[dev\] Non-Commercial License](https://github.com/black-forest-labs/flux/blob/main/model_licenses/LICENSE-FLUX1-dev).

# Usage
We provide a reference implementation of `FLUX.2 [dev]`, as well as sampling code, in a dedicated [github repository](https://github.com/black-forest-labs/flux2).
Developers and creatives looking to build on top of `FLUX.2 [dev]` are encouraged to use this as a starting point.

`FLUX.2 [dev]` is also available in both [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [Diffusers](https://github.com/huggingface/diffusers).

### Using with diffusers 🧨

For local deployment on a consumer type graphics card, like an RTX 4090 or an RTX 5090, please see the [diffusers docs](https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_dev_hf.md) on our GitHub page.

As an example, here's a way to load a 4-bit quantized model with a remote text-encoder on an RTX 4090:

```python
import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel
from diffusers.utils import load_image
from huggingface_hub import get_token
import requests
import io

repo_id = "diffusers/FLUX.2-dev-bnb-4bit"
device = "cuda:0"
torch_dtype = torch.bfloat16

def remote_text_encoder(prompts):
    response = requests.post(
        "https://remote-text-encoder-flux-2.huggingface.co/predict",
        json={"prompt": prompts},
        headers={
            "Authorization": f"Bearer {get_token()}",
            "Content-Type": "application/json"
        }
    )
    prompt_embeds = torch.load(io.BytesIO(response.content))

    return prompt_embeds.to(device)

pipe = Flux2Pipeline.from_pretrained(
    repo_id, transformer=transformer, text_encoder=None, torch_dtype=torch_dtype
).to(device)

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

image = pipe(
    prompt_embeds=remote_text_encoder(prompt),
    #image=load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png") #optional image input
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50, #28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
```


---

# Risks

Black Forest Labs is committed to the responsible development and deployment of our models. Prior to releasing the FLUX.2 family of models, we evaluated and mitigated a number of risks in our model checkpoints and hosted services, including the generation of unlawful content such as child sexual abuse material (CSAM) and nonconsensual intimate imagery (NCII). We implemented a series of pre-release mitigations to help prevent misuse by third parties, with additional post-release mitigations to help address residual risks:
1. Pre-training mitigation. We filtered pre-training data for multiple categories of “not safe for work” (NSFW) and known child sexual abuse material (CSAM) to help prevent a user generating unlawful content in response to text prompts or uploaded images. We have partnered with the Internet Watch Foundation, an independent nonprofit organization dedicated to preventing online abuse, to filter known CSAM from the training data.
2. Post-training mitigation. Subsequently, we undertook multiple rounds of targeted fine-tuning to provide additional mitigation against potential abuse, including both text-to-image (T2I) and image-to-image (I2I) attacks. By inhibiting certain behaviors and suppressing certain concepts in the trained model, these techniques can help to prevent a user generating synthetic CSAM or NCII from a text prompt, or transforming an uploaded image into synthetic CSAM or NCII.
3. Ongoing evaluation. Throughout this process, we conducted multiple internal and external third-party evaluations of model checkpoints to identify further opportunities for mitigation. External third-party evaluations focused on eliciting CSAM and NCII through adversarial testing with (i) text-only prompts, (ii) a single uploaded reference image with text prompts, and (iii) multiple uploaded reference images with text prompts. Based on this feedback, we conducted further safety fine-tuning to produce our open-weight model (FLUX.2 [dev]).
4. Release decision. After safety fine-tuning and prior to release, we conducted a final third-party evaluation of the proposed release checkpoint, focused on T2I and I2I generation of synthetic CSAM and NCII, including a comparison with other open-weight T2I and I2I models (total prompts n≈2,800). The final FLUX.2 [dev] checkpoint demonstrated high resilience against violative inputs in complex generation and editing tasks, and demonstrated higher resilience than leading open-weight models across these risk categories. Based on these findings, we approved the release of the FLUX.2 Pro model via API and the release of the open-weight FLUX.2 [dev] model under a non-commercial license to support third-party research and development.
5. Inference filters. The repository for the FLUX.2 [dev] model includes filters for NSFW and IP-infringing content at input and output. Filters or manual review must be used with the model under the terms of the FLUX.2 [dev] Non-Commercial License. We may approach known deployers of the FLUX.2 [dev] model at random to verify that filters or manual review processes are in place. Additionally, we apply multiple filters to intercept text prompts, uploaded images, and output images on the API for FLUX.2 [pro]. We utilize both in-house and third-party supplied filters to prevent CSAM and NCII outputs, including filters provided by Hive and Microsoft. We provide filters for other categories of potentially harmful content, including gore, which can be adjusted by developers based on their specific risk profile and legitimate use cases.
6. Content provenance. Content provenance features can help users and platforms better identify, label, and interpret AI-generated content online. The inference code for FLUX.2 [dev] implements an example of pixel-layer watermarking, and this repository includes links to the Coalition for Content Provenance and Authenticity (C2PA) standard for metadata. The API for FLUX.2 Pro applies cryptographically-signed C2PA metadata to output content to indicate that images were produced with our model.
7. Policies. Use of our models and access to our API are governed by our FLUX [dev] Non-Commercial License (for our non-commercial open-weight users); Developer Terms of Service, Self-Hosted Commercial License Terms, and Usage Policy (for our commercial open-weight model users); and Developer Terms of Service, FLUX API Service Terms, and Usage Policy (for our API users). These prohibit the generation of unlawful content or the use of generated content for unlawful, defamatory, or abusive purposes. Developers and users must consent to these conditions to access the FLUX.2 [dev] model on Hugging Face.
8. Monitoring. We are monitoring for patterns of violative use after release. We continue to issue and escalate takedown requests to websites, services, or businesses that misuse our models. Additionally, we may ban users or developers who we detect intentionally and repeatedly violate our policies via the FLUX API. Additionally, we provide a dedicated email address (safety@blackforestlabs.ai) to solicit feedback from the community. We maintain a reporting relationship with organizations such as the Internet Watch Foundation and the National Center for Missing and Exploited Children, and welcome ongoing engagement with authorities, developers, and researchers to share intelligence about emerging risks and develop effective mitigations.


# License
This model falls under the [FLUX \[dev\] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.2-dev/blob/main/LICENSE.txt).
