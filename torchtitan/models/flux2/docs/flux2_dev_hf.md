# 🧨 Running the model with diffusers

## Getting started 

Install diffusers from `main` and upgrade your `transformers`, `accelerate` and `bitsandbytes` dependencies to latest

```sh
pip install git+https://github.com/huggingface/diffusers.git
pip install --upgrade transformers accelerate bitsandbytes
```

After accepting the gating on the [FLUX.2-dev repository](https://huggingface.co/black-forest-labs/FLUX.2-dev), login with Hugging Face on your terminal
```sh
hf auth login
```

See below for inference instructions on different GPUs.

---

## 💾 Lower VRAM (~24-32G) - RTX 4090 and 5090

Those with 24-32GB of VRAM can use the model with **4-bit quantization**

### 4-bit transformer and remote text-encoder (~18G of VRAM)

The diffusers team is introducing a remote text-encoder for this release.
The text-embeddings are calculated in bf16 in the cloud and you only load the transformer into VRAM (this setting can get as low as ~18G of VRAM)

```py
import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from huggingface_hub import get_token
import requests
import io

repo_id = "diffusers/FLUX.2-dev-bnb-4bit" #quantized text-encoder and DiT. VAE still in bf16
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
    repo_id, text_encoder=None, torch_dtype=torch_dtype
).to(device)

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

#cat_image = load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png")
image = pipe(
    prompt_embeds=remote_text_encoder(prompt),
    #image=[cat_image] #optional multi-image input
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50, #28 steps can be a good trade-off
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
```

### 4-bit transformer and 4-bit text-encoder (~20G of VRAM)

Load both the text-encoder and the transformer in 4-bit. 
The text-encoder is offloaded from VRAM for the transformer to run with `pipe.enable_model_cpu_offload()`, making sure both will fit.

```py
import torch
from diffusers import Flux2Pipeline, AutoModel
from transformers import Mistral3ForConditionalGeneration
from diffusers.utils import load_image

repo_id = "diffusers/FLUX.2-dev-bnb-4bit" #quantized text-encoder and DiT. VAE still in bf16
device = "cuda:0"
torch_dtype = torch.bfloat16

text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
    repo_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, device_map="cpu"
)
dit = AutoModel.from_pretrained(
    repo_id, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cpu"
)
pipe = Flux2Pipeline.from_pretrained(
    repo_id, text_encoder=text_encoder, transformer=dit, torch_dtype=torch_dtype
)
pipe.enable_model_cpu_offload()

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL + Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."
#cat_image = load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png")
image = pipe(
    prompt=prompt,
    #image=[cat_image] #multi-image input
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50,
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
``` 

To understand how different quantizations affect the model's abilities and quality, access the [FLUX.2 on diffusers](https://huggingface.co/blog/flux-2) blog

---

## 💿 More VRAM (80G+)

Even an H100 can't hold the text-encoder, transormer and VAE at the same time. However, as they each fit individually, it is a matter of activating the `pipe.enable_model_cpu_offload()`
For H200, B200 or larger cards, everything fits.

```py
import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image

repo_id = "black-forest-labs/FLUX.2-dev"
device = "cuda:0"
torch_dtype = torch.bfloat16

pipe = Flux2Pipeline.from_pretrained(
    repo_id, torch_dtype=torch_dtype
)
pipe.enable_model_cpu_offload() #no need to do cpu offload for >80G VRAM carts like H200, B200, etc. and do a `pipe.to(device)` instead

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

#cat_image = load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png")
image = pipe(
    prompt=prompt,
    #image=[cat_image] #multi-image input
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50,
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
```

### Remote text-encoder + H100
`pipe.enable_model_cpu_offload()` slows you down a bit. You can move as fast as possible on the H100 with the remote text-encoder 
```py
import torch
from diffusers import Flux2Pipeline
from diffusers.utils import load_image
from huggingface_hub import get_token
import requests
import io

repo_id = "black-forest-labs/FLUX.2-dev"
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
    assert response.status_code == 200, f"{response.status_code=}"
    prompt_embeds = torch.load(io.BytesIO(response.content))

    return prompt_embeds.to(device)

pipe = Flux2Pipeline.from_pretrained(
    repo_id, text_encoder=None, torch_dtype=torch_dtype
).to(device)

prompt = "Realistic macro photograph of a hermit crab using a soda can as its shell, partially emerging from the can, captured with sharp detail and natural colors, on a sunlit beach with soft shadows and a shallow depth of field, with blurred ocean waves in the background. The can has the text `BFL + Diffusers` on it and it has a color gradient that start with #FF5733 at the top and transitions to #33FF57 at the bottom."

#cat_image = load_image("https://huggingface.co/spaces/zerogpu-aoti/FLUX.1-Kontext-Dev-fp8-dynamic/resolve/main/cat.png")
image = pipe(
    prompt_embeds=remote_text_encoder(prompt),
    #image=[cat_image] #optional multi-image input
    generator=torch.Generator(device=device).manual_seed(42),
    num_inference_steps=50,
    guidance_scale=4,
).images[0]

image.save("flux2_output.png")
```

## 🧮 Other VRAM sizes

If you have different GPU sizes, you can experiment with different quantizations, for example, for 40-48G VRAM GPUs, (8-bit) quantization instead of 4-bit can be a good trade-off. You can learn more on the [diffusers FLUX.2 release blog](https://huggingface.co/blog/flux-2)
