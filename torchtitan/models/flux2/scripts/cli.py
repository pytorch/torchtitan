import json
import os
import random
import shlex
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import torch
from einops import rearrange
from PIL import ExifTags, Image

from flux2.openrouter_api_client import DEFAULT_SAMPLING_PARAMS, OpenRouterAPIClient
from flux2.sampling import (
    batched_prc_img,
    batched_prc_txt,
    denoise,
    denoise_cfg,
    encode_image_refs,
    get_schedule,
    scatter_ids,
)
from flux2.util import FLUX2_MODEL_INFO, load_ae, load_flow_model, load_text_encoder

# from flux2.watermark import embed_watermark


@dataclass
class Config:
    prompt: str = "a photo of a forest with mist swirling around the tree trunks. The word 'FLUX.2' is painted over it in big, red brush strokes with visible texture"
    seed: Optional[int] = None
    width: int = 1360
    height: int = 768
    num_steps: int = 50
    guidance: float = 4.0
    input_images: List[Path] = field(default_factory=list)
    match_image_size: Optional[int] = None  # Index of input_images to match size from
    upsample_prompt_mode: Literal["none", "local", "openrouter"] = "none"
    openrouter_model: str = "mistralai/pixtral-large-2411"  # OpenRouter model name

    def copy(self) -> "Config":
        return Config(
            prompt=self.prompt,
            seed=self.seed,
            width=self.width,
            height=self.height,
            num_steps=self.num_steps,
            guidance=self.guidance,
            input_images=list(self.input_images),
            match_image_size=self.match_image_size,
            upsample_prompt_mode=self.upsample_prompt_mode,
            openrouter_model=self.openrouter_model,
        )


DEFAULTS = Config()

INT_FIELDS = {"width", "height", "seed", "num_steps", "match_image_size"}
FLOAT_FIELDS = {"guidance"}
LIST_FIELDS = {"input_images"}
UPSAMPLING_MODE_FIELDS = ("none", "local", "openrouter")
STR_FIELDS = {"openrouter_model"}


def coerce_value(key: str, raw: str):
    """Convert a raw string to the correct field type."""
    if key in INT_FIELDS:
        if raw.lower() == "none" or raw == "":
            return None
        return int(raw)

    if key in FLOAT_FIELDS:
        return float(raw)

    if key in STR_FIELDS:
        return raw.strip().strip('"').strip("'")

    if key in LIST_FIELDS:
        # Handle empty list cases
        if raw == "" or raw == "[]":
            return []
        # Accept comma-separated or space-separated; strip quotes.
        items = []
        # If user passed a single token that contains commas, split on commas.
        tokens = [raw] if ("," in raw and " " not in raw) else shlex.split(raw)
        for tok in tokens:
            for part in tok.split(","):
                part = part.strip()
                if part:
                    if os.path.exists(part):
                        items.append(Path(part))
                    else:
                        print(f"File {part} not found. Skipping for now. Please check your path")
        return items

    if key == "upsample_prompt_mode":
        v = str(raw).strip().strip('"').strip("'").lower()
        if v in UPSAMPLING_MODE_FIELDS:
            return v
        raise ValueError(
            f"invalid upsample_prompt_mode: {v}. Must be one of: {', '.join(UPSAMPLING_MODE_FIELDS)}"
        )

    # plain strings
    return raw


def apply_updates(cfg: Config, updates: Dict[str, Any]) -> None:
    for k, v in updates.items():
        if not hasattr(cfg, k):
            print(f"  ! unknown key: {k}", file=sys.stderr)
            continue
        # Validate upsample_prompt_mode
        if k == "upsample_prompt_mode":
            valid_modes = {"none", "local", "openrouter"}
            if v not in valid_modes:
                print(
                    f"  ! Invalid upsample_prompt_mode: {v}. Must be one of: {', '.join(valid_modes)}",
                    file=sys.stderr,
                )
                continue
        setattr(cfg, k, v)


def parse_key_values(line: str) -> Dict[str, Any]:
    """
    Parse shell-like 'key=value' pairs. Values can be quoted.
    Example: prompt="a dog" width=768 input_images="in1.png,in2.jpg"
    """
    updates: Dict[str, Any] = {}
    for token in shlex.split(line):
        if "=" not in token:
            # Allow bare commands like: run, show, reset, quit
            updates[token] = True
            continue
        key, val = token.split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            updates[key] = coerce_value(key, val)
        except Exception as e:
            print(f"  ! could not parse {key}={val!r}: {e}", file=sys.stderr)
    return updates


def print_config(cfg: Config):
    d = asdict(cfg)
    d["input_images"] = [str(p) for p in cfg.input_images]
    print("Current config:")
    for k in [
        "prompt",
        "seed",
        "width",
        "height",
        "num_steps",
        "guidance",
        "input_images",
        "match_image_size",
        "upsample_prompt_mode",
        "openrouter_model",
    ]:
        print(f"  {k}: {d[k]}")
    print()


def print_help():
    print("""
Available commands:
  [Enter]           - Run generation with current config
  <any text>        - Set as prompt (then press Enter to generate)
  run               - Run generation with current config
  show              - Show current configuration
  reset             - Reset configuration to defaults
  help, h, ?        - Show this help message
  quit, q, exit     - Exit the program

Setting parameters:
  key=value         - Update a config parameter (shows updated config, doesn't run)

  Examples:
    prompt="a cat in a hat"
    width=768 height=768
    seed=42
    num_steps=30
    guidance=3.5
    input_images="img1.jpg,img2.jpg"
    match_image_size=0    (use dimensions from first input image)
    upsample_prompt_mode="none"  (prompt upsampling mode: "none", "local", or "openrouter")
    openrouter_model="mistralai/pixtral-large-2411"  (OpenRouter model name)

You can combine parameter updates:
  prompt="sunset" width=1920 height=1080

Parameters:
  prompt            - Text prompt for generation (string)
  seed              - Random seed (integer or 'none' for random)
  width             - Output width in pixels (integer)
  height            - Output height in pixels (integer)
  num_steps         - Number of denoising steps (integer)
  guidance          - Guidance scale (float)
  input_images      - Comma-separated list of input image paths (list)
  match_image_size  - Index of input image to match dimensions from (integer, 0-based)
  upsample_prompt_mode - Prompt upsampling mode: "none" (default), "local", or "openrouter" (string)
  openrouter_model  - OpenRouter model name (string, default: "mistralai/pixtral-large-2411")
                         Examples: "mistralai/pixtral-large-2411", "qwen/qwen3-vl-235b-a22b-instruct", etc.
                         Note: For "openrouter" mode, set OPENROUTER_API_KEY environment variable
""")


def validate_model_params(model_name: str, cfg: Config) -> bool:
    """Validate that config parameters match model requirements. Returns True if valid."""
    model_info = FLUX2_MODEL_INFO[model_name]
    defaults = model_info.get("defaults", {})
    fixed_params = model_info.get("fixed_params", set())

    errors = []
    if "num_steps" in fixed_params and cfg.num_steps != defaults["num_steps"]:
        errors.append(
            f"Model '{model_name}' requires num_steps={defaults['num_steps']}, "
            f"but you specified num_steps={cfg.num_steps}"
        )

    if "guidance" in fixed_params and cfg.guidance != defaults["guidance"]:
        errors.append(
            f"Model '{model_name}' requires guidance={defaults['guidance']}, "
            f"but you specified guidance={cfg.guidance}"
        )

    if errors:
        print("\nERROR: Invalid parameters for selected model:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print("\nPlease adjust your parameters and try again.", file=sys.stderr)
        return False

    return True


# ---------- Main Loop ----------


def main(
    model_name: str | None = None,
    single_eval: bool = False,
    prompt: str | None = None,
    debug_mode: bool = False,
    cpu_offloading: bool = False,
    **overwrite,
):
    # Prompt for model selection if not provided
    if model_name is None:
        available_models = list(FLUX2_MODEL_INFO.keys())
        print("Available models:")
        for i, name in enumerate(available_models, 1):
            print(f"  {i}. {name}")
        while True:
            try:
                choice = input(f"\nSelect a model [default: {available_models[0]}]: ").strip()
                if choice == "":
                    model_name = available_models[0]
                    break
                elif choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(available_models):
                        model_name = available_models[idx]
                        break
                    print(f"Please enter a number between 1 and {len(available_models)}")
                elif choice.lower() in FLUX2_MODEL_INFO:
                    model_name = choice.lower()
                    break
                else:
                    print(f"Invalid choice. Available models: {', '.join(available_models)}")
            except (EOFError, KeyboardInterrupt):
                print("\nbye!")
                return

    assert (
        model_name.lower() in FLUX2_MODEL_INFO
    ), f"{model_name} is not available, choose from {FLUX2_MODEL_INFO.keys()}"

    model_info = FLUX2_MODEL_INFO[model_name]
    torch_device = torch.device("cuda")

    text_encoder = load_text_encoder(model_name, device=torch_device)
    if "klein" in model_name:
        mod_and_upsampling_model = load_text_encoder("flux.2-dev")
    else:
        mod_and_upsampling_model = text_encoder

    model = load_flow_model(
        model_name, debug_mode=debug_mode, device="cpu" if cpu_offloading else torch_device
    )
    ae = load_ae(model_name)
    ae.eval()
    text_encoder.eval()

    # API client will be initialized lazily when needed
    openrouter_api_client: Optional[OpenRouterAPIClient] = None

    cfg = DEFAULTS.copy()

    # Apply model defaults if not overridden
    defaults = model_info.get("defaults", {})
    if "num_steps" in defaults and "num_steps" not in overwrite:
        cfg.num_steps = defaults["num_steps"]
    if "guidance" in defaults and "guidance" not in overwrite:
        cfg.guidance = defaults["guidance"]

    changes = [f"{key}={value}" for key, value in overwrite.items()]
    updates = parse_key_values(" ".join(changes))
    apply_updates(cfg, updates)
    if prompt is not None:
        cfg.prompt = prompt

    # Validate initial config
    if not validate_model_params(model_name, cfg):
        sys.exit(1)
    print_config(cfg)

    while True:
        if not single_eval:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye!")
                break

            if not line:
                # Empty -> run with current config
                cmd = "run"
                updates = {}
            else:
                # Check if this is plain text (no key=value pairs and not a known command)
                known_commands = {"run", "show", "reset", "quit", "q", "exit", "help", "h", "?"}
                if "=" not in line and line.lower() not in known_commands:
                    # Treat the entire line as a prompt
                    updates = {"prompt": line}
                    cmd = None
                else:
                    try:
                        updates = parse_key_values(line)
                    except Exception as e:  # noqa: BLE001
                        print(f"  ! Failed to parse command: {type(e).__name__}: {e}", file=sys.stderr)
                        print(
                            "  ! Please check your syntax (e.g., matching quotes) and try again.\n",
                            file=sys.stderr,
                        )
                        continue

                if "prompt" in updates and mod_and_upsampling_model.test_txt(updates["prompt"]):
                    print(
                        "Your prompt has been flagged for potential copyright or public personas concerns. Please choose another."
                    )
                    updates.pop("prompt")

                if "input_images" in updates:
                    flagged = False
                    for image in updates["input_images"]:
                        if mod_and_upsampling_model.test_image(image):
                            print(f"The image {image} has been flagged as unsuitable. Please choose another.")
                            flagged = True
                    if flagged:
                        updates.pop("input_images")

                # If the line was only 'run' / 'show' / ... it will appear as {cmd: True}
                # If it had key=val pairs, there may be no bare command -> just update config
                bare_cmds = [k for k, v in updates.items() if v is True and k.isalpha()]
                cmd = bare_cmds[0] if bare_cmds else None

                # Remove bare commands from updates so they don't get applied as fields
                for c in bare_cmds:
                    updates.pop(c, None)

            if cmd in ("quit", "q", "exit"):
                print("bye!")
                break
            elif cmd == "reset":
                cfg = DEFAULTS.copy()
                # Re-apply model defaults
                if "num_steps" in defaults:
                    cfg.num_steps = defaults["num_steps"]
                if "guidance" in defaults:
                    cfg.guidance = defaults["guidance"]
                print_config(cfg)
                continue
            elif cmd == "show":
                print_config(cfg)
                continue
            elif cmd in ("help", "h", "?"):
                print_help()
                continue

            # Apply key=value changes
            if updates:
                # Create a temporary copy to test the updates
                temp_cfg = cfg.copy()
                apply_updates(temp_cfg, updates)

                # Validate the temporary config
                if not validate_model_params(model_name, temp_cfg):
                    continue

                # Only apply to actual config if validation passed
                cfg = temp_cfg
                print_config(cfg)
                continue

            # Only run if explicitly requested (empty line or 'run' command)
            if cmd != "run":
                if cmd is not None:
                    print(f"  ! Unknown command: '{cmd}'", file=sys.stderr)
                    print("  ! Type 'help' to see available commands.\n", file=sys.stderr)
                continue

        try:
            # Load input images first to potentially match dimensions
            img_ctx = [Image.open(input_image) for input_image in cfg.input_images]

            # Apply match_image_size if specified
            width = cfg.width
            height = cfg.height
            if cfg.match_image_size is not None:
                if cfg.match_image_size < 0 or cfg.match_image_size >= len(img_ctx):
                    print(
                        f"  ! match_image_size={cfg.match_image_size} is out of range (0-{len(img_ctx)-1})",
                        file=sys.stderr,
                    )
                    print(f"  ! Using default dimensions: {width}x{height}", file=sys.stderr)
                else:
                    ref_img = img_ctx[cfg.match_image_size]
                    width, height = ref_img.size
                    print(f"  Matched dimensions from image {cfg.match_image_size}: {width}x{height}")

            seed = cfg.seed if cfg.seed is not None else random.randrange(2**31)
            dir = Path("output")
            dir.mkdir(exist_ok=True)
            output_name = dir / f"sample_{len(list(dir.glob('*')))}.png"

            with torch.no_grad():
                ref_tokens, ref_ids = encode_image_refs(ae, img_ctx)

                if cfg.upsample_prompt_mode == "openrouter":
                    try:
                        # Ensure API key is available, otherwise prompt the user
                        api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
                        if not api_key:
                            try:
                                entered = input(
                                    "OPENROUTER_API_KEY not set. Enter it now (leave blank to skip OpenRouter upsampling): "
                                ).strip()
                            except (EOFError, KeyboardInterrupt):
                                entered = ""
                            if entered:
                                os.environ["OPENROUTER_API_KEY"] = entered
                            else:
                                print(
                                    "  ! No API key provided; disabling OpenRouter upsampling",
                                    file=sys.stderr,
                                )
                                cfg.upsample_prompt_mode = "none"
                                prompt = cfg.prompt
                                # Skip OpenRouter flow

                        # Only proceed if still in openrouter mode (not disabled above)
                        if cfg.upsample_prompt_mode == "openrouter":
                            # Let user specify sampling params, or use model defaults if available
                            sampling_params_input = ""
                            try:
                                sampling_params_input = input(
                                    "Enter OpenRouter sampling params as JSON or key=value (blank to use defaults): "
                                ).strip()
                            except (EOFError, KeyboardInterrupt):
                                sampling_params_input = ""

                            sampling_params: Dict[str, Any] = {}
                            if sampling_params_input:
                                # Try JSON first
                                parsed_ok = False
                                try:
                                    parsed = json.loads(sampling_params_input)
                                    if isinstance(parsed, dict):
                                        sampling_params = parsed
                                        parsed_ok = True
                                except Exception:
                                    parsed_ok = False
                                if not parsed_ok:
                                    # Fallback: parse key=value pairs separated by spaces or commas
                                    tokens = [
                                        tok
                                        for tok in sampling_params_input.replace(",", " ").split(" ")
                                        if tok
                                    ]
                                    for tok in tokens:
                                        if "=" not in tok:
                                            continue
                                        k, v = tok.split("=", 1)
                                        v_str = v.strip()
                                        v_low = v_str.lower()
                                        if v_low in {"true", "false"}:
                                            val: Any = v_low == "true"
                                        else:
                                            try:
                                                if "." in v_str:
                                                    num = float(v_str)
                                                    val = int(num) if num.is_integer() else num
                                                else:
                                                    val = int(v_str)
                                            except Exception:
                                                val = v_str
                                        sampling_params[k.strip()] = val
                                print(f"  Using custom OpenRouter sampling params: {sampling_params}")
                            else:
                                model_key = cfg.openrouter_model
                                default_params = DEFAULT_SAMPLING_PARAMS.get(model_key)
                                if default_params:
                                    sampling_params = default_params
                                    print(
                                        f"  Using default OpenRouter sampling params for {model_key}: {sampling_params}"
                                    )
                                else:
                                    print(
                                        f"  Setting no OpenRouter sampling params: not set for this model ({model_key})"
                                    )

                            # Initialize or reinitialize client if model changed
                            if (
                                openrouter_api_client is None
                                or openrouter_api_client.model != cfg.openrouter_model
                                or getattr(openrouter_api_client, "sampling_params", None) != sampling_params
                            ):
                                openrouter_api_client = OpenRouterAPIClient(
                                    model=cfg.openrouter_model,
                                    sampling_params=sampling_params,
                                )
                            else:
                                # Ensure client uses latest sampling params
                                openrouter_api_client.sampling_params = sampling_params
                            upsampled_prompts = openrouter_api_client.upsample_prompt(
                                [cfg.prompt], img=[img_ctx] if img_ctx else None
                            )
                            prompt = upsampled_prompts[0] if upsampled_prompts else cfg.prompt
                    except Exception as e:
                        print(f"  ! Failed to upsample prompt via OpenRouter API: {e}", file=sys.stderr)
                        print(
                            "  ! Disabling OpenRouter upsampling and falling back to original prompt",
                            file=sys.stderr,
                        )
                        cfg.upsample_prompt_mode = "none"
                        prompt = cfg.prompt
                elif cfg.upsample_prompt_mode == "local":
                    # Use local model for upsampling
                    upsampled_prompts = mod_and_upsampling_model.upsample_prompt(
                        [cfg.prompt], img=[img_ctx] if img_ctx else None
                    )
                    prompt = upsampled_prompts[0] if upsampled_prompts else cfg.prompt
                else:
                    # upsample_prompt_mode == "none" or invalid value
                    prompt = cfg.prompt

                print("Generating with prompt: ", prompt)

                if model_info["guidance_distilled"]:
                    ctx = text_encoder([prompt]).to(torch.bfloat16)
                else:
                    ctx_empty = text_encoder([""]).to(torch.bfloat16)
                    ctx_prompt = text_encoder([prompt]).to(torch.bfloat16)
                    ctx = torch.cat([ctx_empty, ctx_prompt], dim=0)
                ctx, ctx_ids = batched_prc_txt(ctx)

                if cpu_offloading:
                    text_encoder = text_encoder.cpu()
                    torch.cuda.empty_cache()
                    model = model.to(torch_device)
                    if "klein" in model_name:
                        mod_and_upsampling_model = mod_and_upsampling_model.cpu()

                # Create noise
                shape = (1, 128, height // 16, width // 16)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                randn = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device="cuda")
                x, x_ids = batched_prc_img(randn)

                timesteps = get_schedule(cfg.num_steps, x.shape[1])
                if model_info["guidance_distilled"]:
                    x = denoise(
                        model,
                        x,
                        x_ids,
                        ctx,
                        ctx_ids,
                        timesteps=timesteps,
                        guidance=cfg.guidance,
                        img_cond_seq=ref_tokens,
                        img_cond_seq_ids=ref_ids,
                    )
                else:
                    x = denoise_cfg(
                        model,
                        x,
                        x_ids,
                        ctx,
                        ctx_ids,
                        timesteps=timesteps,
                        guidance=cfg.guidance,
                        img_cond_seq=ref_tokens,
                        img_cond_seq_ids=ref_ids,
                    )
                x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
                x = ae.decode(x).float()
                # x = embed_watermark(x)

                if cpu_offloading:
                    model = model.cpu()
                    torch.cuda.empty_cache()
                    text_encoder = text_encoder.to(torch_device)

                    if "klein" in model_name:
                        mod_and_upsampling_model = mod_and_upsampling_model.to(torch_device)

            x = x.clamp(-1, 1)
            x = rearrange(x[0], "c h w -> h w c")

            img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())

            if mod_and_upsampling_model.test_image(img):
                print("Your output has been flagged. Please choose another prompt / input image combination")
            else:
                exif_data = Image.Exif()
                exif_data[ExifTags.Base.Software] = "AI generated;flux2"
                exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                img.save(output_name, exif=exif_data, quality=95, subsampling=0)
                print(f"Saved {output_name}")

        except Exception as e:  # noqa: BLE001
            print(f"\n  ERROR: {type(e).__name__}: {e}", file=sys.stderr)
            print("  The model is still loaded. Please fix the error and try again.\n", file=sys.stderr)

        if single_eval:
            break


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
