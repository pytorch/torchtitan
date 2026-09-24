# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, cast, Self

import spmd_types as spmd
import torch
from torch import nn, Tensor
from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.distributed.spmd_types import annotate_replicated_parameters
from torchtitan.models.common.linear import Linear
from torchtitan.models.flux.model.autoencoder import AutoEncoder
from torchtitan.models.flux.model.hf_embedder import FluxEmbedder

from torchtitan.models.flux.model.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    local_split_text_image,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from torchtitan.models.flux.sharding import annotate_flux_forward_inputs
from torchtitan.models.flux.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_data,
)
from torchtitan.models.utils import quadratic_attention_flops_per_token
from torchtitan.protocols import BaseModel
from torchtitan.protocols.module import ModuleList

from .state_dict_adapter import FluxStateDictAdapter


class FluxModel(BaseModel):
    state_dict_adapter_cls = FluxStateDictAdapter
    supports_pipeline_parallel = False

    """
    Transformer model for flow matching on sequences.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        img_in: Linear.Config
        txt_in: Linear.Config
        in_channels: int = 64
        out_channels: int = 64
        vec_in_dim: int = 768
        context_in_dim: int = 512
        hidden_size: int = 3072
        mlp_ratio: float = 4.0
        num_heads: int = 24
        depth: int = 19
        depth_single_blocks: int = 38
        axes_dim: tuple = (16, 56, 56)
        theta: int = 10_000
        qkv_bias: bool = True
        autoencoder: AutoEncoder.Config = field(default_factory=AutoEncoder.Config)

        # Text encoder configs, set by the model registry. The trainer can
        # override version and random_init when it builds the encoders.
        clip_encoder: FluxEmbedder.Config
        t5_encoder: FluxEmbedder.Config

        # Sub-component configs (all required — set by the model registry)
        pe_config: EmbedND.Config
        time_in_config: MLPEmbedder.Config
        vector_in_config: MLPEmbedder.Config
        final_layer_config: LastLayer.Config
        double_blocks: list[DoubleStreamBlock.Config]
        single_blocks: list[SingleStreamBlock.Config]

        def update_from_config(self, *, config, **kwargs) -> None:
            from torchtitan.models.flux.sharding import set_flux_sharding_config

            set_flux_sharding_config(self)

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int
        ) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())

            # Base: 6 FLOPs per parameter per token (fwd + bwd for linear
            # layers). This assumes every token passes through every parameter.
            num_flops_per_token = 6 * nparams

            # Correction 1: DoubleStreamBlocks have symmetric img/txt streams;
            # each token only passes through one side. Subtract one side's
            # per-token linear params per block (excluding modulation, which
            # is per-sample and handled separately below).
            #
            # Per-side per-token weight params:
            #   attn.qkv:  h * 3h       = 3h²
            #   attn.proj: h * h         =  h²
            #   mlp:       2 * h * h*r   = 2rh²
            #   Total: h² * (4 + 2r)
            db_h = self.double_blocks[0].hidden_size
            db_r = self.double_blocks[0].mlp_ratio
            nparams_db_one_side_per_token = int(db_h * db_h * (4 + 2 * db_r))
            num_flops_per_token -= 6 * nparams_db_one_side_per_token * self.depth

            # Correction 2: Modulation layers operate on vec (per-sample
            # conditioning from CLIP + timestep), not per-token. The 6*nparams
            # base counts them as per-token; replace with amortized per-token
            # cost (once per sample / seq_len tokens).
            #
            # Per-sample modulation weight params:
            #   DoubleStreamBlock: img_mod(6h²) + txt_mod(6h²) = 12h² per block
            #   SingleStreamBlock: modulation(3h²) per block
            #   LastLayer: adaLN_modulation(2h²)
            sb_h = self.single_blocks[0].hidden_size
            fl_h = self.final_layer_config.hidden_size
            nparams_mod_per_sample = (
                12 * db_h * db_h * self.depth
                + 3 * sb_h * sb_h * self.depth_single_blocks
                + 2 * fl_h * fl_h
            )
            num_flops_per_token -= 6 * nparams_mod_per_sample * (seq_len - 1) // seq_len

            # Add non-parameterized self-attention FLOPs (QK^T and attn*V)
            # on the same convention as the other models: the factor of 6
            # covers forward + backward and multiply-adds, and the two
            # contractions are carried by (qk_head_dim + v_head_dim).
            db_heads = self.double_blocks[0].num_heads
            sb_heads = self.single_blocks[0].num_heads
            db_head_dim = db_h // db_heads
            sb_head_dim = sb_h // sb_heads
            num_flops_per_token += (
                quadratic_attention_flops_per_token(
                    num_heads=sb_heads,
                    qk_head_dim=sb_head_dim,
                    v_head_dim=sb_head_dim,
                    seq_len=seq_len,
                )
                * self.depth_single_blocks
                + quadratic_attention_flops_per_token(
                    num_heads=db_heads,
                    qk_head_dim=db_head_dim,
                    v_head_dim=db_head_dim,
                    seq_len=seq_len,
                )
                * self.depth
            )

            return nparams, num_flops_per_token

    def __init__(self, config: Config):
        super().__init__()

        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}"
            )
        pe_dim = config.hidden_size // config.num_heads
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {config.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.pe_embedder = config.pe_config.build()
        self.img_in = config.img_in.build()
        self.time_in = config.time_in_config.build()
        self.vector_in = config.vector_in_config.build()
        self.txt_in = config.txt_in.build()

        self.double_blocks = ModuleList([cfg.build() for cfg in config.double_blocks])

        self.single_blocks = ModuleList([cfg.build() for cfg in config.single_blocks])

        self.final_layer = config.final_layer_config.build()

    def parallelize(
        self,
        *,
        parallel_dims: ParallelDims,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
        compile_config: CompileConfig | None,
        ac_config: ActivationCheckpointingConfig | None,
        dump_folder: str,
    ) -> Self:
        """Apply Flux's AC-before-SPMD parallelization lifecycle."""
        from torchtitan.distributed.utils import get_spmd_context

        with get_spmd_context(parallel_dims=parallel_dims):
            if ac_config is not None:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                )

                for blocks in (self.double_blocks, self.single_blocks):
                    for layer_id, block in blocks.named_children():
                        blocks.register_module(
                            layer_id,
                            checkpoint_wrapper(block, preserve_rng_state=True),
                        )

            self._parallelize(parallel_dims)
            annotate_replicated_parameters(self, parallel_dims)

            if compile_config is not None and "model" in compile_config.components:
                for block in (*self.double_blocks, *self.single_blocks):
                    block.compile(backend=compile_config.backend, fullgraph=True)

            self._apply_fsdp(
                parallel_dims=parallel_dims,
                training=training,
                parallelism=parallelism,
            )
        return self

    def _apply_fsdp(
        self,
        *,
        parallel_dims: ParallelDims,
        training: TrainingConfig,
        parallelism: ParallelismConfig,
    ) -> None:
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            fully_shard,
            MixedPrecisionPolicy,
        )

        from torchtitan.distributed.fsdp import (
            disable_fsdp_gradient_division,
            enable_fsdp_symm_mem,
            resolve_fsdp_mesh,
        )

        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        fsdp_config: dict[str, Any] = {
            "mesh": dp_mesh,
            "mp_policy": MixedPrecisionPolicy(
                param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
                reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            ),
        }
        if dp_mesh_dims is not None:
            fsdp_config["dp_mesh_dims"] = dp_mesh_dims
        if training.enable_cpu_offload:
            fsdp_config["offload_policy"] = CPUOffloadPolicy()

        for module in (self.img_in, self.time_in, self.vector_in, self.txt_in):
            fully_shard(module, **fsdp_config)
        for block in (*self.double_blocks, *self.single_blocks):
            fully_shard(block, **fsdp_config)
        fully_shard(self.final_layer, **fsdp_config, reshard_after_forward=False)
        fully_shard(self, **fsdp_config)
        enable_fsdp_symm_mem(self, parallelism.fsdp_symm_mem_scope)
        disable_fsdp_gradient_division(self)

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims,
        parallelism: ParallelismConfig,
        max_num_documents: int | None = None,
        max_context_length: int | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Encode a raw image-text batch and prepare flow-matching inputs."""
        del parallelism, max_num_documents, max_context_length
        autoencoder = cast(AutoEncoder | None, kwargs["autoencoder"])
        clip_encoder = cast(FluxEmbedder, kwargs["clip_encoder"])
        t5_encoder = cast(FluxEmbedder, kwargs["t5_encoder"])
        dtype = cast(torch.dtype, kwargs["dtype"])
        batch = dict(input_dict)
        batch["image"] = batch.pop("labels")
        batch = preprocess_data(
            device=batch["image"].device,
            dtype=dtype,
            autoencoder=autoencoder,
            clip_encoder=clip_encoder,
            t5_encoder=t5_encoder,
            batch=batch,
        )

        image_encodings = batch["img_encodings"]
        clip_encodings = batch["clip_encodings"]
        t5_encodings = batch["t5_encodings"]
        batch_size = image_encodings.shape[0]

        with torch.no_grad(), torch.device(image_encodings.device):
            noise = torch.randn_like(image_encodings)
            timesteps = torch.rand((batch_size,))
            sigmas = timesteps.view(-1, 1, 1, 1)
            latents = (1 - sigmas) * image_encodings + sigmas * noise

            _, _, latent_height, latent_width = latents.shape
            position_dim = 3
            latent_pos_enc = create_position_encoding_for_latents(
                batch_size, latent_height, latent_width, position_dim
            )
            text_pos_enc = torch.zeros(batch_size, t5_encodings.shape[1], position_dim)
            latents = pack_latents(latents)
            target = pack_latents(noise - image_encodings)

        if parallel_dims.cp_enabled:
            from torchtitan.distributed.context_parallel import cp_shard

            (
                latents,
                latent_pos_enc,
                t5_encodings,
                text_pos_enc,
                target,
            ), _ = cp_shard(
                parallel_dims.get_mesh("cp"),
                (latents, latent_pos_enc, t5_encodings, text_pos_enc, target),
                None,
                load_balancer_type=None,
                input_seq_dims=1,
            )

        return (
            latents,
            target,
            {
                "img_ids": latent_pos_enc,
                "txt": t5_encodings,
                "txt_ids": text_pos_enc,
                "y": clip_encodings,
                "timesteps": timesteps,
                "loss_target": target,
            },
        )

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        loss_target: Tensor | None = None,
    ) -> Tensor:
        annotate_flux_forward_inputs(
            latents=img,
            latent_pos_enc=img_ids,
            t5_encodings=txt,
            text_pos_enc=txt_ids,
            target=loss_target,
            clip_encodings=y,
            timesteps=timesteps,
        )

        @spmd.local_map(
            in_types=(
                spmd.PartitionSpec("dp", "cp", None),
                spmd.PartitionSpec("dp", "cp", None),
            ),
            out_types=spmd.PartitionSpec("dp", "cp", None),
        )
        def _local_concat_text_image(text: Tensor, image: Tensor) -> Tensor:
            return torch.cat((text, image), dim=1)

        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = _local_concat_text_image(txt_ids, img_ids)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = _local_concat_text_image(txt, img)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        _, img = local_split_text_image(img, txt.shape[1])

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img
