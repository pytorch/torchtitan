from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Any, Literal

import einops
import torch

from torchtitan.components.tokenizer import BaseTokenizer


class WorldModelTokenizer(BaseTokenizer):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseTokenizer.Config):
        compressor_model: str = ""
        compressor_in_channels: Literal[3, 6, "auto"] = "auto"

    def __init__(
        self,
        config: Config,
        *,
        tokenizer_path: str | None = None,
    ) -> None:
        del tokenizer_path
        super().__init__()
        self.config = config
        self._encoder: torch.nn.Module | None = None
        self._encoder_key: tuple[torch.device, torch.dtype] | None = None

    def encode(
        self,
        inputs: dict[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if "latents" in inputs:
            return inputs["latents"].to(device=device, dtype=dtype)

        encoder = self._encoder_on(device=device, dtype=dtype)
        imgs = inputs["imgs"]
        big_imgs = inputs["big_imgs"]
        batch, timesteps = imgs.shape[:2]
        in_channels = self._compressor_in_channels(encoder)
        if in_channels == 3:
            rearrange_spec = "nc b t h w c -> (nc b t) c h w"
            inverse_spec = "(nc b t) c h w -> b t (nc c) h w"
        elif in_channels == 6:
            rearrange_spec = "nc b t h w c -> (b t) (nc c) h w"
            inverse_spec = "(b t) (nc c) h w -> b t (nc c) h w"
        else:
            raise ValueError(f"unsupported compressor input channels: {in_channels}")

        with torch.inference_mode():
            x = einops.rearrange(
                [imgs, big_imgs],
                rearrange_spec,
                nc=2,
                b=batch,
                t=timesteps,
            ).to(device=device, dtype=dtype)
            x = x.div(255.0).mul(2).sub(1).clamp(-1, 1)
            latents = encoder(x)
            if isinstance(latents, tuple):
                latents = latents[0]
            return einops.rearrange(
                latents,
                inverse_spec,
                nc=2,
                b=batch,
                t=timesteps,
            )

    def decode(self, *args: Any, **kwargs: Any) -> str:
        return ""

    def get_vocab_size(self) -> int:
        return 0

    def _encoder_on(self, *, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
        if not self.config.compressor_model:
            raise ValueError("inputs contain images, but tokenizer.compressor_model is empty")
        key = (device, dtype)
        if self._encoder is None:
            self._encoder = self._load_encoder()
        if self._encoder_key != key:
            self._encoder = self._encoder.to(device=device, dtype=dtype)
            self._encoder_key = key
        return self._encoder

    def _load_encoder(self) -> torch.nn.Module:
        model = self.config.compressor_model
        if os.path.isdir(model):
            model = os.path.join(model, "encoder.pt2")
        if os.path.exists(model):
            return torch.export.load(model).module()
        if "/" in model:
            from huggingface_hub import hf_hub_download

            return torch.export.load(hf_hub_download(model, "encoder.pt2")).module()

        from xx.training.lib.checkpoint import Checkpoint

        return torch.export.load(io.BytesIO(Checkpoint(model)["encoder.pt2"])).module()

    def _compressor_in_channels(self, encoder: torch.nn.Module) -> int:
        configured = self.config.compressor_in_channels
        if configured != "auto":
            return configured
        try:
            return int(encoder.get_buffer("example_shapes").tolist()[0][1])
        except Exception:
            return 6
