from dataclasses import dataclass, field

import torch

from torch import nn, Tensor
from torchtitan.components.tokenizer import Tokenizer
from torchtitan.config_manager import JobConfig

from torchtitan.experiments.flux.model.modules.autoencoder import AutoEncoderParams
from torchtitan.experiments.flux.model.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)

from torchtitan.protocols.train_spec import BaseModelArgs, ModelProtocol
from torchtitan.tools.logging import logger


@dataclass
class FluxModelArgs(BaseModelArgs):
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
    guidance_embed: bool = True
    autoencoder_params: AutoEncoderParams = field(default_factory=AutoEncoderParams)

    def update_from_config(self, job_config: JobConfig, tokenizer: Tokenizer) -> None:
        # context_in_dim is the same as the T5 embedding dimension
        self.context_in_dim = job_config.encoder.max_t5_encoding_len

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        # TODO(jianiw): Add the number of flops for the autoencoder
        logger.warning("FLUX model haven't implement get_nparams_and_flops() function")
        return 0, 0


class FluxModel(nn.Module, ModelProtocol):
    """
    Transformer model for flow matching on sequences.

    Agrs:
        model_args: FluxModelArgs.

    Attributes:
        model_args (TransformerModelArgs): Model configuration arguments.
    """

    def __init__(self, model_args: FluxModelArgs):
        super().__init__()

        self.model_args = model_args
        self.in_channels = model_args.in_channels
        self.out_channels = model_args.out_channels
        if model_args.hidden_size % model_args.num_heads != 0:
            raise ValueError(
                f"Hidden size {model_args.hidden_size} must be divisible by num_heads {model_args.num_heads}"
            )
        pe_dim = model_args.hidden_size // model_args.num_heads
        if sum(model_args.axes_dim) != pe_dim:
            raise ValueError(
                f"Got {model_args.axes_dim} but expected positional dim {pe_dim}"
            )
        self.hidden_size = model_args.hidden_size
        self.num_heads = model_args.num_heads
        self.pe_embedder = EmbedND(
            dim=pe_dim, theta=model_args.theta, axes_dim=model_args.axes_dim
        )
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(model_args.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if model_args.guidance_embed
            else nn.Identity()
        )
        self.txt_in = nn.Linear(model_args.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=model_args.mlp_ratio,
                    qkv_bias=model_args.qkv_bias,
                )
                for _ in range(model_args.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    self.hidden_size, self.num_heads, mlp_ratio=model_args.mlp_ratio
                )
                for _ in range(model_args.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    def init_weights(self, buffer_device=None):
        # TODO(jianiw): replace placeholder with real weight init
        for param in self.parameters():
            param.data.uniform_(0, 0.1)

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256))
        if self.model_args.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
        vec = vec + self.vector_in(y)
        txt = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe)

        img = torch.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img

    @classmethod
    def from_model_args(cls, model_args: FluxModelArgs) -> "FluxModel":
        """
        Initialize a Flux model from a FluxModelArgs object.

        Args:
            model_args (FluxModelArgs): Model configuration arguments.

        Returns:
            FluxModel: FluxModel model.

        """
        return cls(model_args)
