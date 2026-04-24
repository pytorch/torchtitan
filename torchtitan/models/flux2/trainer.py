# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

import torch

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.trainer import Trainer

from torchtitan.models.flux2.configs import Encoder, Inference, Validation
from torchtitan.models.flux2.src.flux2.util import (
    FLUX2_MODEL_INFO,
    load_ae,
    load_text_encoder,
)


class Flux2Trainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        encoder: Encoder = field(default_factory=Encoder)
        validation: Validation = field(default_factory=Validation)
        inference: Inference = field(default_factory=Inference)

    def __init__(self, config: Config):
        super().__init__(config)

        # Distinct seed across FSDP ranks to keep stochastic CFG dropout unique.
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["fsdp", "dp_replicate"],
        )

        self._dtype = (
            TORCH_DTYPE_MAP[config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        model_name = config.encoder.model_name.lower()

        self.autoencoder = load_ae(model_name, device=self.device).to(dtype=self._dtype)
        self.autoencoder = self.autoencoder.eval().requires_grad_(False)

        self._text_cache_namespace = model_name
        self._text_cache_mode = config.encoder.text_encoder_cache_mode.lower()
        if self._text_cache_mode not in {"off", "read"}:
            raise ValueError(
                f"Invalid text_encoder_cache_mode: {self._text_cache_mode}."
                " Expected 'off' or 'read'."
            )
        cache_dir = config.encoder.text_encoder_cache_dir
        self._text_cache_dir = Path(cache_dir) if cache_dir else None
        if self._text_cache_mode != "off":
            if self._text_cache_dir is None:
                raise ValueError("text_encoder_cache_dir must be set when cache is enabled.")
            if self._text_cache_mode == "read" and not self._text_cache_dir.exists():
                raise ValueError(
                    f"text_encoder_cache_dir does not exist: {self._text_cache_dir}"
                )

        self.text_encoder = None
        if self._text_cache_mode != "read":
            text_encoder_device = config.encoder.text_encoder_device
            self.text_encoder = load_text_encoder(model_name, device=text_encoder_device)
            self.text_encoder = self.text_encoder.eval().requires_grad_(False)

        self._default_guidance = config.encoder.guidance
        if self._default_guidance is None and model_name in FLUX2_MODEL_INFO:
            self._default_guidance = FLUX2_MODEL_INFO[model_name]["defaults"].get("guidance", 1.0)
        if self._default_guidance is None:
            self._default_guidance = 1.0

    @staticmethod
    def _latents_to_seq(latents: torch.Tensor) -> torch.Tensor:
        bsz, ch, h, w = latents.shape
        return latents.permute(0, 2, 3, 1).reshape(bsz, h * w, ch)

    @staticmethod
    def _build_img_ids(
        bsz: int, h: int, w: int, device: torch.device
    ) -> torch.Tensor:
        t = torch.zeros(1, dtype=torch.int64, device=device)
        h_ids = torch.arange(h, dtype=torch.int64, device=device)
        w_ids = torch.arange(w, dtype=torch.int64, device=device)
        l = torch.zeros(1, dtype=torch.int64, device=device)
        ids = torch.cartesian_prod(t, h_ids, w_ids, l)
        return ids.unsqueeze(0).expand(bsz, -1, -1)

    @staticmethod
    def _build_txt_ids(bsz: int, seq_len: int, device: torch.device) -> torch.Tensor:
        t = torch.zeros(1, dtype=torch.int64, device=device)
        h = torch.zeros(1, dtype=torch.int64, device=device)
        w = torch.zeros(1, dtype=torch.int64, device=device)
        l = torch.arange(seq_len, dtype=torch.int64, device=device)
        ids = torch.cartesian_prod(t, h, w, l)
        return ids.unsqueeze(0).expand(bsz, -1, -1)

    def _text_cache_path(self, prompt: str) -> Path:
        h = hashlib.sha1(f"{self._text_cache_namespace}\n{prompt}".encode("utf-8")).hexdigest()
        assert self._text_cache_dir is not None
        return self._text_cache_dir / h[:2] / f"{h}.pt"

    def _load_cached_encoding(self, prompt: str) -> torch.Tensor | None:
        if self._text_cache_dir is None:
            return None
        path = self._text_cache_path(prompt)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict):
            cached_prompt = payload.get("prompt")
            if cached_prompt is not None and cached_prompt != prompt:
                return None
            cached_ns = payload.get("namespace")
            if cached_ns is not None and cached_ns != self._text_cache_namespace:
                return None
            enc = payload.get("encoding")
            return enc if torch.is_tensor(enc) else None
        if torch.is_tensor(payload):
            return payload
        return None

    def _get_text_encodings(self, prompts: list[str]) -> torch.Tensor:
        if self._text_cache_mode == "off":
            assert self.text_encoder is not None
            return self.text_encoder(prompts)

        cached: list[torch.Tensor | None] = [None] * len(prompts)
        missing_idx: list[int] = []
        for i, prompt in enumerate(prompts):
            enc = self._load_cached_encoding(prompt)
            if enc is None:
                missing_idx.append(i)
            else:
                cached[i] = enc

        if missing_idx:
            sample = ", ".join(repr(prompts[i]) for i in missing_idx[:3])
            raise FileNotFoundError(
                f"Missing precomputed text encodings for prompts: {sample}"
            )

        if any(enc is None for enc in cached):
            raise RuntimeError("Text encoding cache returned empty entries.")
        return torch.stack([enc for enc in cached], dim=0)

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform a single forward and backward pass through the FLUX.2 model.
        """

        assert (
            global_valid_tokens is None
        ), "FLUX.2 model doesn't rescale loss by number of global valid tokens"

        prompts = input_dict.get("prompt")
        if prompts is None:
            raise KeyError("Flux2Trainer expects 'prompt' in input_dict.")
        if isinstance(prompts, str):
            prompts = [prompts]

        images = labels.to(device=self.device, dtype=self._dtype)

        with torch.no_grad():
            clean_latents = self.autoencoder.encode(images)
            text_encodings = self._get_text_encodings(prompts)

        text_encodings = text_encodings.to(device=self.device, dtype=self._dtype)

        local_valid_tokens = torch.tensor(
            clean_latents.numel(), dtype=torch.float32, device=self.device
        )

        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.float()

        bsz, _, latent_h, latent_w = clean_latents.shape

        with torch.no_grad(), torch.device(self.device):
            timesteps = torch.rand((bsz,), device=self.device, dtype=clean_latents.dtype)
            if "timestep" in input_dict:
                timestep_val = input_dict["timestep"]
                if isinstance(timestep_val, torch.Tensor):
                    timesteps = timestep_val.to(
                        device=self.device, dtype=clean_latents.dtype
                    )
                else:
                    timesteps = torch.tensor(
                        timestep_val, device=self.device, dtype=clean_latents.dtype
                    )

            noise = torch.randn_like(clean_latents)
            sigmas = timesteps.view(-1, 1, 1, 1)
            latents = (1 - sigmas) * clean_latents + sigmas * noise

        img_seq = self._latents_to_seq(latents)
        target_seq = self._latents_to_seq(noise - clean_latents)
        img_ids = self._build_img_ids(bsz, latent_h, latent_w, device=self.device)
        txt_ids = self._build_txt_ids(bsz, text_encodings.shape[1], device=self.device)

        if self.parallel_dims.cp_enabled:
            from torchtitan.distributed.context_parallel import cp_shard

            (
                img_seq,
                img_ids,
                text_encodings,
                txt_ids,
                target_seq,
            ), _ = cp_shard(
                self.parallel_dims.get_mesh("cp"),
                (img_seq, img_ids, text_encodings, txt_ids, target_seq),
                None,  # FLUX.2 does not use explicit attention masks.
                load_balancer_type=None,
            )

        guidance = None
        model = self.model_parts[0]
        if getattr(model, "use_guidance_embed", False):
            guidance = torch.full(
                (bsz,), float(self._default_guidance), device=self.device, dtype=timesteps.dtype
            )

        with self.train_context():
            with self.maybe_enable_amp:
                pred = model(
                    x=img_seq,
                    x_ids=img_ids,
                    timesteps=timesteps,
                    ctx=text_encodings,
                    ctx_ids=txt_ids,
                    guidance=guidance,
                )

                loss = self.loss_fn(pred, target_seq) / global_valid_tokens

            del pred, noise, target_seq
            loss.backward()

        return loss

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        parallel_dims = self.parallel_dims

        if self.gradient_accumulation_steps > 1:
            raise ValueError("FLUX.2 doesn't support gradient accumulation for now.")

        input_dict, labels = next(data_iterator)

        loss = self.forward_backward_step(input_dict=input_dict, labels=labels)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(loss, loss_mesh),
                dist_utils.dist_max(loss, loss_mesh),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                ),
            )
        else:
            global_avg_loss = global_max_loss = loss.detach().item()
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics=extra_metrics,
        )
