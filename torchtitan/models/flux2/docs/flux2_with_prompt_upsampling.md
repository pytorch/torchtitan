# Prompt upsampling with FLUX.2

Prompt upsampling uses a large vision language model to expand and enrich your prompts before generation, which can significantly improve results for reasoning-heavy and complex generation tasks.

## When to use prompt upsampling

Prompt upsampling is particularly effective for prompts requiring reasoning or complex interpretation:

- **Text generation in images**: Creating memes, posters, or images where the model needs to generate creative or contextually appropriate text
- **Image-based instructions**: Prompts where the input image contains overlaid text, arrows, or annotations that need to be interpreted (e.g., "follow the instructions in the image", "read the diagram and generate the result")
- **Code and math reasoning**: Generating visualizations of algorithms, mathematical concepts, or code flow diagrams where logical structure is important

For simple, direct prompts (e.g., "a red car"), prompt upsampling may not provide significant benefits.

## Methods

We provide two methods for prompt upsampling:

### 1. API-based prompt upsampling (recommended)

API-based prompt upsampling via [OpenRouter](https://openrouter.ai/) generally produces better results by leveraging more capable models.

Set your API key as an environment variable:

```bash
export OPENROUTER_API_KEY="<api_key>"
```

Then run the CLI with upsampling enabled:
```bash

export PYTHONPATH=src
python scripts/cli.py --upsample_prompt_mode=openrouter
```

You can switch between different models using `--openrouter_model=<model_name>`.

Alternatively, you can just start the CLI via

```bash
export PYTHONPATH=src
python scripts/cli.py
```

and choose your prompt upsampling model interactively.

**Example output:**

| Prompt: "Make a meme about generating memes with this model" |
|:---:|
| <img src="../assets/t2i_upsample_example.png" alt="Output" width="512"> |

### 2. Local prompt upsampling

Local prompt upsampling uses [`Mistral-Small-3.2-24B-Instruct-2506`](https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506), which is the model we use for text encoding in `FLUX.2 [dev]`. This option requires no API keys but may produce less detailed expansions. 

To enable local prompt upsampling, use `--upsample_prompt_mode=local`.

**Example:**

<table>
  <tr>
    <th colspan="2" style="text-align: center;">Prompt: "Describe what the red arrow is seeing"</th>
  </tr>
  <tr>
    <th>Input</th>
    <th>Output</th>
  </tr>
  <tr>
    <td align="center"><img src="../assets/i2i_upsample_input.png" alt="Input image"></td>
    <td align="center"><img src="../assets/i2i_upsample_example.png" alt="Output image"></td>
  </tr>
</table>