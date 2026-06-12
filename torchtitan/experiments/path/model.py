from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, distribute_tensor

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import _apply_ac_to_transformer_block
from torchtitan.distributed.fsdp import enable_fsdp_symm_mem, get_fsdp_reshard_after_forward_policy
from torchtitan.models.common import Embedding, LayerNorm, Linear, RMSNorm, SiLU
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict, ModuleList, Sequential
from torchtitan.tools.logging import logger
from xx.ml_tools.constants.model import ModelInputs

from . import convnext


@dataclass(frozen=True)
class PathHead:
    name: str
    output_size: int
    mlp: bool
    scale: bool


class ScaleLayer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_features: int

    def __init__(self, config: Config):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(config.n_features))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class PathMLP(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm: LayerNorm.Config | RMSNorm.Config
        c_fc: Linear.Config
        c_proj: Linear.Config
        act: str
        dropout: float

    def __init__(self, config: Config):
        super().__init__()
        self.norm = config.norm.build()
        self.c_fc = config.c_fc.build()
        self.act = nn.GELU(approximate="tanh") if config.act == "gelu_tanh" else nn.GELU()
        self.c_proj = config.c_proj.build()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.act(self.c_fc(self.norm(x)))))


class PathSelfAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm: LayerNorm.Config | RMSNorm.Config
        q_norm: LayerNorm.Config | RMSNorm.Config | None
        k_norm: LayerNorm.Config | RMSNorm.Config | None
        c_attn: Linear.Config
        c_proj: Linear.Config
        inner_attention: ScaledDotProductAttention.Config
        n_head: int
        head_dim: int
        dropout: float

    def __init__(self, config: Config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.norm = config.norm.build()
        self.q_norm = config.q_norm.build() if config.q_norm is not None else nn.Identity()
        self.k_norm = config.k_norm.build() if config.k_norm is not None else nn.Identity()
        self.c_attn = config.c_attn.build()
        self.c_proj = config.c_proj.build()
        self.inner_attention = config.inner_attention.build()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        qkv = self.c_attn(self.norm(x)).view(b, t, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(2)
        q, k = self.q_norm(q), self.k_norm(k)
        x = self.inner_attention(q, k, v, is_causal=True)
        return self.dropout(self.c_proj(x.reshape(b, t, self.n_head * self.head_dim)))


class PathTransformerBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: PathSelfAttention.Config
        mlp: PathMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.attention = config.attention.build()
        self.mlp = config.mlp.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        return x + self.mlp(x)


class PathTransformer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layers: list[PathTransformerBlock.Config]

    def __init__(self, config: Config):
        super().__init__()
        self.layers = ModuleList([layer.build() for layer in config.layers])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def apply_activation_checkpointing(self, wrap, base_fqn: str) -> None:
        for layer_id, layer in enumerate(self.layers):
            self.layers[layer_id] = wrap(layer, f"{base_fqn}.layers.{layer_id}")

    def apply_fsdp(self, shard, reshard_after_forward: bool) -> None:
        for layer in self.layers:
            shard(layer, reshard_after_forward)


class PointSummarizer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        mlp1: PathMLP.Config
        mlp2: PathMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.mlp1 = config.mlp1.build()
        self.mlp2 = config.mlp2.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x) + x
        return self.mlp2(x) + x


class LinearEncoder(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_layer: Linear.Config
        out_layer: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.net = Sequential(config.in_layer.build(), SiLU.Config().build(), config.out_layer.build())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalSummarizer(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        mlp1: PathMLP.Config
        mlp2: PathMLP.Config
        desire_encoder: LinearEncoder.Config
        traffic_encoder: LinearEncoder.Config
        action_t_encoder: LinearEncoder.Config
        transformer: PathTransformer.Config
        pos_embedding: Embedding.Config
        block_size: int
        dense_training_outputs: bool

    def __init__(self, config: Config):
        super().__init__()
        self.block_size = config.block_size
        self.dense_training_outputs = config.dense_training_outputs
        self.mlp1 = config.mlp1.build()
        self.mlp2 = config.mlp2.build()
        self.desire_encoder = config.desire_encoder.build()
        self.traffic_encoder = config.traffic_encoder.build()
        self.action_t_encoder = config.action_t_encoder.build()
        self.transformer = config.transformer.build()
        self.pos_embedding = config.pos_embedding.build()

    def forward(
        self,
        feats: torch.Tensor,
        desire: torch.Tensor,
        traffic_convention: torch.Tensor,
        action_t: torch.Tensor,
    ) -> torch.Tensor:
        feats = self.mlp1(feats) + feats
        feats = self.mlp2(feats) + feats
        desire = rearrange(self.desire_encoder(desire), "b c -> b () c")
        traffic_convention = rearrange(self.traffic_encoder(traffic_convention), "b c -> b () c")
        action_t = rearrange(self.action_t_encoder(action_t), "b c -> b () c")
        pos = self.pos_embedding(torch.arange(self.block_size, device=feats.device))
        x = feats + rearrange(pos, "t c -> () t c") + desire + traffic_convention + action_t
        x = self.transformer(x)
        return x if self.dense_training_outputs else x[:, self.block_size - 1]


class Hydra(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        heads: tuple[PathHead, ...]
        head_mlps: dict[str, PathMLP.Config]
        final_layers: dict[str, Linear.Config]
        scale_layers: dict[str, ScaleLayer.Config]

    def __init__(self, config: Config):
        super().__init__()
        self.heads = config.heads
        self.head_mlp = ModuleDict({name: cfg.build() for name, cfg in config.head_mlps.items()})
        self.final_layer = ModuleDict({name: cfg.build() for name, cfg in config.final_layers.items()})
        self.scale_layer = ModuleDict({name: cfg.build() for name, cfg in config.scale_layers.items()})

    def forward(self, in_feats: torch.Tensor) -> dict[str, torch.Tensor]:
        ret = {}
        for name, layer in self.final_layer.items():
            feats = self.head_mlp[name](in_feats) + in_feats if name in self.head_mlp else in_feats
            ret[name] = layer(feats)
        for name, layer in self.scale_layer.items():
            ret[name] = layer(ret[name])
        return ret


class Policy(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        summarizer: PointSummarizer.Config
        hydra: Hydra.Config

    def __init__(self, config: Config):
        super().__init__()
        self.summarizer = config.summarizer.build()
        self.hydra = config.hydra.build()

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.hydra(self.summarizer(features))


class TemporalPolicy(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        temporal_summarizer: TemporalSummarizer.Config
        temporal_hydra: Hydra.Config
        history_idxs: tuple[int, ...]

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.temporal_summarizer = config.temporal_summarizer.build()
        self.temporal_hydra = config.temporal_hydra.build()
        self.register_buffer("history_idxs", torch.empty(len(config.history_idxs), dtype=torch.long), persistent=False)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device if buffer_device is not None else self.history_idxs.device
        self.history_idxs = torch.tensor(self.config.history_idxs, dtype=torch.long, device=device)

    def forward(
        self,
        features: torch.Tensor,
        desire_pulse: torch.Tensor,
        traffic_convention: torch.Tensor,
        action_t: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        dtype = features.dtype
        stacked_desire = rearrange(desire_pulse.to(dtype), "b t c -> b (t c)")
        summary = self.temporal_summarizer(
            features[:, self.history_idxs],
            stacked_desire,
            traffic_convention[:, -1].to(dtype),
            action_t[:, -1].to(dtype),
        )
        return self.temporal_hydra(summary)


class Vision(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        flavor: str
        input_frame_names: tuple[str, ...]
        in_channels: int
        vision_features: int
        pretrained: bool
        drop_path_rate: float
        mean: float
        std: float

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = convnext.create_convnext(
            config.flavor,
            pretrained=False,
            in_chans=config.in_channels,
            num_classes=config.vision_features,
            drop_path_rate=config.drop_path_rate,
        )
        self.register_buffer("_mean", torch.empty(1, config.in_channels, 1, 1), persistent=True)
        self.register_buffer("_std", torch.empty(1, config.in_channels, 1, 1), persistent=True)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        device = buffer_device if buffer_device is not None else self._mean.device
        self._mean = torch.full((1, self.config.in_channels, 1, 1), self.config.mean, device=device)
        self._std = torch.full((1, self.config.in_channels, 1, 1), self.config.std, device=device)

    def load_pretrained(self) -> None:
        if not self.config.pretrained:
            return

        state_dict = self._pretrained_state_dict()
        target_state = self.encoder.state_dict()
        load_state = {}
        for name, value in state_dict.items():
            if name.startswith("head."):
                continue
            target = target_state.get(name)
            if target is None:
                continue
            value = self._move_pretrained_value(value, target)
            if tuple(value.shape) != tuple(target.shape):
                continue
            load_state[name] = value

        missing, unexpected = self.encoder.load_state_dict(load_state, strict=False)
        pretrained_name = convnext.pretrained_name(self.config.flavor)
        logger.info(
            f"Loaded {len(load_state)} ConvNeXt tensors from {pretrained_name} "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )

    def _pretrained_state_dict(self) -> dict[str, torch.Tensor]:
        from timm.models._builder import adapt_input_conv, load_state_dict_from_hf

        state_dict = load_state_dict_from_hf(
            f"timm/{convnext.pretrained_name(self.config.flavor)}",
            weights_only=True,
        )
        state_dict = convnext.checkpoint_filter_fn(state_dict, self.encoder)
        if self.config.in_channels != 3:
            state_dict["stem.0.weight"] = adapt_input_conv(self.config.in_channels, state_dict["stem.0.weight"])
        return state_dict

    @staticmethod
    def _move_pretrained_value(value: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if isinstance(target, DTensor):
            return distribute_tensor(value.to(dtype=target.dtype), target.device_mesh, list(target.placements))
        return value.to(device=target.device, dtype=target.dtype)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat([inputs[name] for name in self.config.input_frame_names], dim=1)
        dtype = next(self.encoder.parameters()).dtype
        x = x.to(dtype)
        return self.encoder((x - self._mean.to(dtype)) / self._std.to(dtype))


class PathModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        vision: Vision.Config
        point_policy: Policy.Config
        temporal_policy: TemporalPolicy.Config

        def update_from_config(self, *, config, **kwargs) -> None:
            parallelism = config.parallelism
            if parallelism.full_dtensor:
                raise ValueError("path v1 does not support full DTensor")
            unsupported = {
                "tensor parallel": parallelism.tensor_parallel_degree,
                "context parallel": parallelism.context_parallel_degree,
                "pipeline parallel": parallelism.pipeline_parallel_degree,
                "expert parallel": parallelism.expert_parallel_degree,
            }
            for name, degree in unsupported.items():
                if degree > 1:
                    raise ValueError(f"path v1 does not support {name}")

        def get_nparams_and_flops(self, model: Module, seq_len: int) -> tuple[int, int]:
            nparams = sum(p.numel() for p in model.parameters())
            return nparams, 2 * nparams

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.vision = config.vision.build()
        self.point_policy = config.point_policy.build()
        self.temporal_policy = config.temporal_policy.build()

    def verify_module_protocol(self) -> None:
        pass

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        super().init_states(buffer_device=buffer_device)
        self._init_plain_modules()
        self.vision.load_pretrained()

    def _init_plain_modules(self) -> None:
        for module in self.modules():
            if isinstance(module, Module):
                continue
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        self.vision.encoder.init_path_weights()

    def forward(
        self,
        inputs: dict[str, torch.Tensor] | torch.Tensor,
        big_img: torch.Tensor | None = None,
        desire_pulse: torch.Tensor | None = None,
        traffic_convention: torch.Tensor | None = None,
        action_t: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if isinstance(inputs, torch.Tensor):
            inputs = {
                ModelInputs.IMG: inputs,
                ModelInputs.BIG_IMG: big_img,
                ModelInputs.DESIRE: desire_pulse,
                ModelInputs.TRAFFIC: traffic_convention,
                ModelInputs.ACTION_T: action_t,
            }
        img = inputs[ModelInputs.IMG]
        b, t, *_ = img.shape
        vision_inputs = {
            name: rearrange(inputs[name], "b t c h w -> (b t) c h w", b=b, t=t)
            for name in self.config.vision.input_frame_names
        }
        features = self.vision(vision_inputs)
        features = rearrange(features, "(b t) c -> b t c", b=b, t=t)
        return self.point_policy(features) | self.temporal_policy(
            features,
            inputs[ModelInputs.DESIRE],
            inputs[ModelInputs.TRAFFIC],
            inputs[ModelInputs.ACTION_T],
        )


def parallelize_path(
    model: PathModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
) -> PathModel:
    if parallelism.full_dtensor:
        raise ValueError("path v1 does not support full DTensor")
    if parallel_dims.tp_enabled or parallel_dims.cp_enabled or parallel_dims.pp_enabled or parallel_dims.ep_enabled:
        raise ValueError("path v1 supports data parallelism only")

    model_compile_enabled = compile_config.enable and "model" in compile_config.components
    if ac_config.mode != "none":
        _apply_activation_checkpointing(model, ac_config)

    if model_compile_enabled:
        _apply_compile(model, compile_config)

    names = ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    _apply_fsdp(
        model,
        parallel_dims.get_mesh(names),
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    logger.info("Applied HSDP to the path model" if parallel_dims.dp_replicate_enabled else "Applied FSDP to the path model")
    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the path model")
    return model


def _apply_activation_checkpointing(
    model: PathModel,
    ac_config: ActivationCheckpointConfig,
) -> None:
    def wrap(module: nn.Module, fqn: str) -> nn.Module:
        return _apply_ac_to_transformer_block(module, ac_config, base_fqn=fqn)

    model.vision.encoder.apply_activation_checkpointing(
        wrap,
        ac_config.mode,
        "vision.encoder",
    )
    model.temporal_policy.temporal_summarizer.transformer.apply_activation_checkpointing(
        wrap,
        "temporal_policy.temporal_summarizer.transformer",
    )

    logger.info(f"Applied {ac_config.mode} activation checkpointing to the path model")


def _apply_compile(model: PathModel, compile_config: CompileConfig) -> None:
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True
    model.vision.encoder.compile(backend=compile_config.backend)
    model.point_policy.compile(backend=compile_config.backend)
    model.temporal_policy.compile(backend=compile_config.backend)
    logger.info("Compiling path model components with torch.compile")


def _apply_fsdp(
    model: PathModel,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    enable_symm_mem: bool = False,
) -> None:
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=True,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy,
        pp_enabled,
    )

    def shard(module: nn.Module, reshard: bool) -> None:
        fully_shard(module, **fsdp_config, reshard_after_forward=reshard)

    model.vision.encoder.apply_fsdp(
        shard,
        reshard_after_forward,
        reshard_after_forward_policy == "always",
    )
    model.temporal_policy.temporal_summarizer.transformer.apply_fsdp(
        shard,
        reshard_after_forward,
    )
    shard(model.vision.encoder, reshard_after_forward)
    shard(model.point_policy, reshard_after_forward)
    shard(model.temporal_policy, reshard_after_forward)
    fully_shard(model, **fsdp_config)

    if enable_symm_mem:
        enable_fsdp_symm_mem(model)
