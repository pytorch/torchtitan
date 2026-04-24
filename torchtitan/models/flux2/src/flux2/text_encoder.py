import os
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    pipeline,
)
try:
    from accelerate import init_empty_weights
except Exception:
    from transformers.modeling_utils import init_empty_weights

from .hf_cache import is_hf_offline, resolve_hf_cache_dir
from .sampling import cap_pixels, concatenate_images
from .system_messages import (
    PROMPT_IMAGE_INTEGRITY,
    PROMPT_IMAGE_INTEGRITY_FOLLOW_UP,
    PROMPT_TEXT_INTEGRITY,
    SYSTEM_MESSAGE,
    SYSTEM_MESSAGE_UPSAMPLING_I2I,
    SYSTEM_MESSAGE_UPSAMPLING_T2I,
    SYSTEM_PROMPT_CONTENT_FILTER,
)

OUTPUT_LAYERS_MISTRAL = [10, 20, 30]
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512
NSFW_THRESHOLD = 0.85
UPSAMPLING_MAX_IMAGE_SIZE = 768**2
_QWEN3_PATH_ENV_TEMPLATES = (
    "FLUX2_QWEN3_{variant}_MODEL_PATH",
    "FLUX2_QWEN3_{variant}_PATH",
    "QWEN3_{variant}_MODEL_PATH",
    "QWEN3_{variant}_PATH",
)


def _find_repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "torchtitan").exists():
            return parent
    return None


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        try:
            normalized = path.expanduser()
        except Exception:
            continue
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        unique.append(normalized)
    return unique


def _resolve_local_pretrained_dir(path: Path) -> Path | None:
    if not path.exists():
        return None
    if path.is_file():
        return path.parent if path.name == "config.json" else None
    if (path / "config.json").exists():
        return path

    snapshots_dir = path / "snapshots"
    main_ref = path / "refs" / "main"
    if main_ref.is_file() and snapshots_dir.is_dir():
        revision = main_ref.read_text().strip()
        if revision:
            snapshot_path = snapshots_dir / revision
            if (snapshot_path / "config.json").exists():
                return snapshot_path

    if snapshots_dir.is_dir():
        snapshot_paths = sorted(
            candidate
            for candidate in snapshots_dir.iterdir()
            if candidate.is_dir() and (candidate / "config.json").exists()
        )
        if snapshot_paths:
            return snapshot_paths[0]

    return None


def _qwen3_path_env_vars(variant: str) -> tuple[str, ...]:
    variant = variant.upper()
    return tuple(template.format(variant=variant) for template in _QWEN3_PATH_ENV_TEMPLATES)


def _resolve_qwen3_model_spec(model_spec: str, cache_dir: str | None) -> tuple[str, tuple[str, ...]]:
    model_name = model_spec.split("/")[-1]
    variant = model_name.removeprefix("Qwen3-").removesuffix("-FP8")
    env_vars = _qwen3_path_env_vars(variant)

    candidates: list[Path] = []
    for env_var in env_vars:
        path = os.environ.get(env_var)
        if path:
            candidates.append(Path(path))

    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        hf_home = cache_path.parent if cache_path.name == "hub" else cache_path
        candidates.extend(
            [
                hf_home / model_name,
                hf_home / "Qwen" / model_name,
                cache_path / f"models--Qwen--{model_name}",
            ]
        )

    repo_root = _find_repo_root()
    if repo_root is not None:
        hf_assets_dir = repo_root / "assets" / "hf"
        candidates.extend(
            [
                hf_assets_dir / model_name,
                hf_assets_dir / "Qwen" / model_name,
                hf_assets_dir / "hub" / f"models--Qwen--{model_name}",
            ]
        )

    for candidate in _unique_paths(candidates):
        resolved_path = _resolve_local_pretrained_dir(candidate)
        if resolved_path is not None:
            return str(resolved_path), env_vars

    return model_spec, env_vars


class Mistral3SmallEmbedder(nn.Module):
    @classmethod
    def init_empty(
        cls,
        model_spec: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        model_spec_processor: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        torch_dtype: str = "bfloat16",
    ) -> "Mistral3SmallEmbedder":
        cache_dir = resolve_hf_cache_dir()
        local_files_only = is_hf_offline()
        config = AutoConfig.from_pretrained(
            model_spec,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        with init_empty_weights():
            model = Mistral3ForConditionalGeneration(config)
        model = model.to(getattr(torch, torch_dtype))
        processor = AutoProcessor.from_pretrained(
            model_spec_processor,
            use_fast=False,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.model = model
        self.processor = processor
        self.yes_token, self.no_token = processor.tokenizer.encode(
            ["yes", "no"], add_special_tokens=False
        )
        self.max_length = MAX_LENGTH
        self.upsampling_max_image_size = UPSAMPLING_MAX_IMAGE_SIZE
        self._hf_cache_dir = cache_dir
        self._hf_local_files_only = local_files_only
        self.nsfw_classifier = None
        return self

    def __init__(
        self,
        model_spec: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        model_spec_processor: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        torch_dtype: str = "bfloat16",
    ):
        super().__init__()

        cache_dir = resolve_hf_cache_dir()
        local_files_only = is_hf_offline()

        self.model: Mistral3ForConditionalGeneration = Mistral3ForConditionalGeneration.from_pretrained(
            model_spec,
            torch_dtype=getattr(torch, torch_dtype),
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_spec_processor,
            use_fast=False,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.yes_token, self.no_token = self.processor.tokenizer.encode(
            ["yes", "no"], add_special_tokens=False
        )

        self.max_length = MAX_LENGTH
        self.upsampling_max_image_size = UPSAMPLING_MAX_IMAGE_SIZE

        self._hf_cache_dir = cache_dir
        self._hf_local_files_only = local_files_only
        self.nsfw_classifier = None

    def _model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _model_dtype(self) -> torch.dtype:
        return next(self.model.parameters()).dtype


    def _validate_and_process_images(
        self, img: list[list[Image.Image]] | list[Image.Image]
    ) -> list[list[Image.Image]]:
        # Simple validation: ensure it's a list of PIL images or list of lists of PIL images
        if not img:
            return []

        # Check if it's a list of lists or a list of images
        if isinstance(img[0], Image.Image):
            # It's a list of images, convert to list of lists
            img = [[im] for im in img]

        # potentially concatenate multiple images to reduce the size
        img = [[concatenate_images(img_i)] if len(img_i) > 1 else img_i for img_i in img]

        # cap the pixels
        img = [[cap_pixels(img_i, self.upsampling_max_image_size) for img_i in img_i] for img_i in img]
        return img

    def format_input(
        self,
        txt: list[str],
        system_message: str = SYSTEM_MESSAGE,
        img: list[Image.Image] | list[list[Image.Image]] | None = None,
    ) -> list[list[dict]]:
        """
        Format a batch of text prompts into the conversation format expected by apply_chat_template.
        Optionally, add images to the input.

        Args:
            txt: List of text prompts
            system_message: System message to use (default: CREATIVE_SYSTEM_MESSAGE)
            img: List of images to add to the input.

        Returns:
            List of conversations, where each conversation is a list of message dicts
        """
        # Remove [IMG] tokens from prompts to avoid Pixtral validation issues
        # when truncation is enabled. The processor counts [IMG] tokens and fails
        # if the count changes after truncation.
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in txt]

        if img is None or len(img) == 0:
            return [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}],
                    },
                    {"role": "user", "content": [{"type": "text", "text": prompt}]},
                ]
                for prompt in cleaned_txt
            ]
        else:
            assert len(img) == len(txt), "Number of images must match number of prompts"
            img = self._validate_and_process_images(img)

            messages = [
                [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_message}],
                    },
                ]
                for _ in cleaned_txt
            ]

            for i, (el, images) in enumerate(zip(messages, img)):
                # optionally add the images per batch element.
                if images is not None:
                    el.append(
                        {
                            "role": "user",
                            "content": [{"type": "image", "image": image_obj} for image_obj in images],
                        }
                    )
                # add the text.
                el.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": cleaned_txt[i]}],
                    }
                )

            return messages

    def tokenize(self, txt: list[str]) -> dict[str, torch.Tensor]:
        messages_batch = self.format_input(txt=txt)
        inputs = self.processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": inputs["input_ids"].cpu(),
            "attention_mask": inputs["attention_mask"].cpu(),
        }

    @torch.no_grad()
    def upsample_prompt(
        self,
        txt: list[str],
        img: list[Image.Image] | list[list[Image.Image]] | None = None,
        temperature: float = 0.15,
    ) -> list[str]:
        """
        Upsample prompts using the model's generate method.

        Args:
            txt: List of input prompts to upsample
            img: Optional list of images or list of lists of images. If None or all None, uses t2i mode, otherwise i2i mode.

        Returns:
            List of upsampled prompts
        """
        # Set system message based on whether images are provided
        if img is None or len(img) == 0 or img[0] is None:
            system_message = SYSTEM_MESSAGE_UPSAMPLING_T2I
        else:
            system_message = SYSTEM_MESSAGE_UPSAMPLING_I2I

        # Format input messages
        messages_batch = self.format_input(txt=txt, system_message=system_message, img=img)

        # Process all messages at once
        # with image processing a too short max length can throw an error in here.
        try:
            inputs = self.processor.apply_chat_template(
                messages_batch,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=2048,
            )
        except ValueError as e:
            print(
                f"Error processing input: {e}, your max length is probably too short, when you have images in the input."
            )
            raise e

        # Move to device
        model_device = self._model_device()
        inputs["input_ids"] = inputs["input_ids"].to(model_device)
        inputs["attention_mask"] = inputs["attention_mask"].to(model_device)

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(model_device, self._model_dtype())

        # Generate text using the model's generate method
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=temperature,
                use_cache=True,
            )

            # Decode only the newly generated tokens (skip input tokens)
            # Extract only the generated portion
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = generated_ids[:, input_length:]

            raw_txt = self.processor.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            return raw_txt
        except Exception as e:
            print(f"Error generating upsampled prompt: {e}, returning original prompt")
            return txt

    @torch.no_grad()
    def forward(self, txt: list[str]):
        # Format input messages
        messages_batch = self.format_input(txt=txt)

        # Process all messages at once
        # with image processing a too short max length can throw an error in here.
        inputs = self.processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        # Move to device
        model_device = self._model_device()
        input_ids = inputs["input_ids"].to(model_device)
        attention_mask = inputs["attention_mask"].to(model_device)

        # Forward pass through the model
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_MISTRAL], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")

    def yes_no_logit_processor(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Sets all tokens but yes/no to the minimum.
        """
        scores_yes_token = scores[:, self.yes_token].clone()
        scores_no_token = scores[:, self.no_token].clone()
        scores_min = scores.min()
        scores[:, :] = scores_min - 1
        scores[:, self.yes_token] = scores_yes_token
        scores[:, self.no_token] = scores_no_token
        return scores

    def _get_nsfw_classifier(self):
        if self.nsfw_classifier is None:
            model_kwargs = {"local_files_only": self._hf_local_files_only}
            if self._hf_cache_dir:
                model_kwargs["cache_dir"] = self._hf_cache_dir
            self.nsfw_classifier = pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection",
                model_kwargs=model_kwargs,
            )
        return self.nsfw_classifier

    def test_image(self, image: Image.Image | str | Path | torch.Tensor) -> bool:
        if isinstance(image, torch.Tensor):
            image = rearrange(image[0].clamp(-1.0, 1.0), "c h w -> h w c")
            image = Image.fromarray((127.5 * (image + 1.0)).cpu().byte().numpy())
        elif isinstance(image, (str, Path)):
            image = Image.open(image)

        classifier = self._get_nsfw_classifier()
        classification = next(c for c in classifier(image) if c["label"] == "nsfw")
        if classification["score"] > NSFW_THRESHOLD:
            return True

        # 512^2 pixels are enough for checking
        w, h = image.size
        f = (512**2 / (w * h)) ** 0.5
        image = image.resize((int(f * w), int(f * h)))

        chat = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_CONTENT_FILTER,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_IMAGE_INTEGRITY,
                    },
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": PROMPT_IMAGE_INTEGRITY_FOLLOW_UP,
                    },
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model_device())
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self._model_dtype())

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.yes_no_logit_processor],
            do_sample=False,
        )

        return generate_ids[0, -1].item() == self.yes_token

    def test_txt(self, txt: str) -> bool:
        chat = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_CONTENT_FILTER,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_TEXT_INTEGRITY.format(prompt=txt),
                    },
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model_device())

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.yes_no_logit_processor],
            do_sample=False,
        )
        return generate_ids[0, -1].item() == self.yes_token


class Qwen3Embedder(nn.Module):
    def __init__(
        self,
        model_spec: str,
        device: str | torch.device = "cuda",
    ):
        super().__init__()

        cache_dir = resolve_hf_cache_dir()
        local_files_only = is_hf_offline()
        resolved_model_spec, model_env_vars = _resolve_qwen3_model_spec(model_spec, cache_dir)
        model_name = model_spec.split("/")[-1]

        model_kwargs = {
            "torch_dtype": None,
            "device_map": str(device),
            "local_files_only": local_files_only,
        }
        tokenizer_kwargs = {"local_files_only": local_files_only}
        if Path(resolved_model_spec).exists():
            print(f"Loading {resolved_model_spec} for the Qwen3 text encoder")
        else:
            model_kwargs["cache_dir"] = cache_dir
            tokenizer_kwargs["cache_dir"] = cache_dir

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                resolved_model_spec,
                **model_kwargs,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                resolved_model_spec,
                **tokenizer_kwargs,
            )
        except OSError as exc:
            if local_files_only and resolved_model_spec == model_spec:
                raise OSError(
                    f"Offline Hugging Face lookup failed for {model_spec}. "
                    f"Download the model into assets/hf/{model_name} or set one of "
                    f"{', '.join(model_env_vars)} to a local model directory or HF snapshot."
                ) from exc
            raise
        self.max_length = MAX_LENGTH
    def _model_device(self) -> torch.device:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device("cpu")


    @torch.no_grad()
    def forward(self, txt: list[str]):
        all_input_ids = []
        all_attention_masks = []

        for prompt in txt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self._model_device())
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self._model_device())

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")

    def tokenize(self, txt: list[str]) -> dict[str, torch.Tensor]:
        all_input_ids = []
        all_attention_masks = []

        for prompt in txt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).cpu()
        attention_mask = torch.cat(all_attention_masks, dim=0).cpu()

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def test_txt(self, txt: str) -> bool:
        raise NotImplementedError("Qwen3Embedder does not support text testing")

    def test_image(self, image) -> bool:
        raise NotImplementedError("Qwen3Embedder does not support image testing")

    def upsample_prompt(self, txt: list[str], img=None, **kwargs) -> list[str]:
        raise NotImplementedError("Qwen3Embedder does not support upsampling")


def load_mistral_small_embedder(device: str | torch.device = "cuda") -> Mistral3SmallEmbedder:
    return Mistral3SmallEmbedder().to(device)


def load_qwen3_embedder(variant: str, device: str | torch.device = "cuda"):
    return Qwen3Embedder(model_spec=f"Qwen/Qwen3-{variant}-FP8", device=device)


def load_mistral_small_state_dict(
    model_spec: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
    torch_dtype: str = "bfloat16",
) -> dict[str, torch.Tensor]:
    cache_dir = resolve_hf_cache_dir()
    local_files_only = is_hf_offline()
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_spec,
        torch_dtype=getattr(torch, torch_dtype),
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return model.state_dict()
