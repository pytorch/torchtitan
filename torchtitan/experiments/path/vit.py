from __future__ import annotations

from dataclasses import dataclass
from xx.ml_tools.constants.model import ModelInputs

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.models.common import Embedding, LayerNorm, Linear, RMSNorm
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleList
from torchtitan.tools.logging import logger

from .model import PathMLP, PathSelfAttention, PathTransformerBlock


class PatchEmbed(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        proj: Linear.Config
        patch_size: tuple[int, int, int]

    def __init__(self, config: Config):
        super().__init__()
        self.patch_size = config.patch_size
        self.proj = config.proj.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pt, ph, pw = self.patch_size
        x = rearrange(
            x, "b (t pt) c (h ph) (w pw) -> b (t h w) (pt c ph pw)", pt=pt, ph=ph, pw=pw
        )
        return self.proj(x.to(self.proj.weight.dtype))


class PlanHead(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm: LayerNorm.Config | RMSNorm.Config
        head: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm = config.norm.build()
        self.head = config.head.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(x))


class PlanViT(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        input_size: tuple[int, int, int]
        patch_size: tuple[int, int, int]
        in_channels: int
        n_embd: int
        output_mult: float
        patch_embed: PatchEmbed.Config
        pos_embedding: Embedding.Config
        blocks: list[PathTransformerBlock.Config]
        norm: LayerNorm.Config | RMSNorm.Config
        plan_head: PlanHead.Config

        @property
        def num_patches(self) -> int:
            t, h, w = self.input_size
            pt, ph, pw = self.patch_size
            return (t // pt) * (h // ph) * (w // pw)

        def update_from_config(self, *, config, **kwargs) -> None:
            parallelism = config.parallelism
            for name, degree in {
                "tensor parallel": parallelism.tensor_parallel_degree,
                "context parallel": parallelism.context_parallel_degree,
                "pipeline parallel": parallelism.pipeline_parallel_degree,
                "expert parallel": parallelism.expert_parallel_degree,
            }.items():
                if degree > 1:
                    raise ValueError(f"PlanViT does not support {name}")

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())
            return nparams, 6 * nparams

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patch_embed = config.patch_embed.build()
        self.pos_embedding = config.pos_embedding.build()
        self.blocks = ModuleList([block.build() for block in config.blocks])
        self.norm = config.norm.build()
        self.plan_head = config.plan_head.build()

    def verify_module_protocol(self) -> None:
        pass

    def _frames(self, inputs: dict[str, torch.Tensor] | torch.Tensor) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            return inputs
        img, big = inputs[ModelInputs.IMG], inputs[ModelInputs.BIG_IMG]
        frame = torch.cat([img[:, -1], big[:, -1]], dim=1).unsqueeze(1)
        return (frame.float() - 127.5) / 63.75

    def forward(
        self, inputs: dict[str, torch.Tensor] | torch.Tensor
    ) -> dict[str, torch.Tensor]:
        x = self.patch_embed(self._frames(inputs))
        pos = self.pos_embedding(torch.arange(x.shape[1], device=x.device))
        x = x + rearrange(pos, "t c -> () t c")
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return {"plan": self.plan_head(x.mean(dim=1)) * self.config.output_mult}


def parallelize_vit(
    model: PlanViT,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
) -> PlanViT:
    if (
        parallel_dims.tp_enabled
        or parallel_dims.cp_enabled
        or parallel_dims.pp_enabled
        or parallel_dims.ep_enabled
    ):
        raise ValueError("PlanViT supports data parallelism only")
    names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    dp_mesh: DeviceMesh = parallel_dims.get_mesh(names)
    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        cast_forward_inputs=True,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    for idx, block in enumerate(model.blocks):
        fully_shard(
            block, **fsdp_config, reshard_after_forward=(idx < len(model.blocks) - 1)
        )
    fully_shard(model, **fsdp_config)
    logger.info("Applied FSDP to PlanViT")
    return model
