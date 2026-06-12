from __future__ import annotations

import math
import os
from functools import partial

import torch.nn as nn

from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.tokenizer import NoOpTokenizer
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.models.common import Embedding, LayerNorm, Linear
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.protocols.model_spec import ModelSpec
from xx.datasets.helpers import DEFAULT_BIG_TRAIN_LIST
from xx.ml_tools.constants.model import (
    FRAME_TYPE,
    INPUT_FRAMES_NAMES,
    N_FRAMES,
    TEMPORAL_INPUTS,
    ModelInputs,
    frame_constants_from_fps,
)
from xx.training.path.config import DatasetConfig as XXPathDatasetConfig
from xx.training.path.hydra_configs import DRIVING_HEADS, META_HEADS, POSE_HEADS, TEMPORAL_META_HEADS

from .dataset import PathDataLoader
from .model import (
    Hydra,
    LinearEncoder,
    PathHead,
    PathMLP,
    PathModel,
    PathSelfAttention,
    PathTransformer,
    PathTransformerBlock,
    PointSummarizer,
    Policy,
    ScaleLayer,
    TemporalPolicy,
    TemporalSummarizer,
    Vision,
    parallelize_path,
)
from .loss import PathLoss
from .onnx_checkpoint import PathOnnxCheckpointManager
from .trainer import PathTrainer
from .validate import PathValidator


_LINEAR_INIT = {"weight": partial(nn.init.normal_, mean=0.0, std=0.02), "bias": nn.init.zeros_}
_NORM_INIT = {"weight": nn.init.ones_, "bias": nn.init.zeros_}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="path",
        flavor=flavor,
        model=_model_config(flavor),
        parallelize_fn=parallelize_path,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def convnext_tiny() -> PathTrainer.Config:
    return _path("convnext_tiny")


def convnext_small() -> PathTrainer.Config:
    return _path("convnext_small")


def convnext_base() -> PathTrainer.Config:
    return _path("convnext_base")


def convnext_xxlarge() -> PathTrainer.Config:
    return _path("convnext_xxlarge")


def _path(flavor: str) -> PathTrainer.Config:
    steps = 512*10
    mixed_precision_param = "bfloat16"
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", str(local_world_size)))
    num_nodes = int(os.environ.get("GROUP_WORLD_SIZE", str(world_size // local_world_size)))
    return PathTrainer.Config(
        loss=PathLoss.Config(),
        model_spec=model_registry(flavor),
        tokenizer=NoOpTokenizer.Config(),
        dataloader=_dataloader_config(split="train"),
        optimizer=_optimizer_config(),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=round(steps * 0.),
            total_steps=steps,
            decay_ratio=0.,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=16,
            global_batch_size=-1,
            seq_len=1,
            steps=steps,
            max_norm=1.0,
            dtype="float32",
            mixed_precision_param=mixed_precision_param,
            mixed_precision_reduce="float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=num_nodes,
            data_parallel_shard_degree=local_world_size,
            tensor_parallel_degree=1,
            context_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
            enable_sequence_parallel=False,
        ),
        checkpoint=_checkpoint_config(),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        compile=CompileConfig(enable=True, components=["model"]),
        metrics=MetricsProcessor.Config(log_freq=10, enable_reporterv2=True),
        validator=PathValidator.Config(
            enable=True,
            freq=512,
            steps=32,
            dataloader=_dataloader_config(split="val"),
            mixed_precision_param=mixed_precision_param,
        ),
        debug=DebugConfig(seed=0),
    )


def _model_config(flavor: str) -> PathModel.Config:
    vision_features = 512
    frame_constants = frame_constants_from_fps(n_frames=N_FRAMES, frame_type=FRAME_TYPE)
    in_channels = sum(frame_constants["frame_shapes"][name][0] for name in INPUT_FRAMES_NAMES)
    block_size = len(frame_constants["history_idxs"])
    temporal_len = frame_constants["temporal_len"]
    dim = vision_features

    return PathModel.Config(
        vision=Vision.Config(
            flavor=flavor,
            input_frame_names=tuple(INPUT_FRAMES_NAMES),
            in_channels=in_channels,
            vision_features=vision_features,
            pretrained=False,
            drop_path_rate=0.2,
            mean=255 / 2,
            std=255 / 4,
        ),
        point_policy=Policy.Config(
            summarizer=PointSummarizer.Config(
                mlp1=_mlp(dim, mlp_mult=2, bias=False, dropout=0.0),
                mlp2=_mlp(dim, mlp_mult=2, bias=False, dropout=0.0),
            ),
            hydra=_hydra(_heads(META_HEADS + POSE_HEADS), in_features=dim, mlp_mult=2),
        ),
        temporal_policy=TemporalPolicy.Config(
            temporal_summarizer=TemporalSummarizer.Config(
                mlp1=_mlp(dim, mlp_mult=2, bias=False, dropout=0.0),
                mlp2=_mlp(dim, mlp_mult=2, bias=False, dropout=0.0),
                desire_encoder=_encoder(TEMPORAL_INPUTS[ModelInputs.DESIRE][0] * temporal_len, dim),
                traffic_encoder=_encoder(TEMPORAL_INPUTS[ModelInputs.TRAFFIC][0], dim),
                action_t_encoder=_encoder(TEMPORAL_INPUTS[ModelInputs.ACTION_T][0], dim),
                transformer=PathTransformer.Config(
                    layers=[
                        PathTransformerBlock.Config(
                            attention=_attention(dim=dim, n_head=8, dropout=0.1),
                            mlp=_mlp(dim, mlp_mult=2, bias=True, dropout=0.1),
                        )
                        for _ in range(4)
                    ]
                ),
                pos_embedding=Embedding.Config(num_embeddings=block_size, embedding_dim=dim, param_init=_LINEAR_INIT),
                block_size=block_size,
                dense_training_outputs=True,
            ),
            temporal_hydra=_hydra(_heads(DRIVING_HEADS + TEMPORAL_META_HEADS), in_features=dim, mlp_mult=2),
            history_idxs=tuple(int(x) for x in frame_constants["history_idxs"]),
        ),
    )


def _dataloader_config(*, split: str) -> PathDataLoader.Config:
    base = XXPathDatasetConfig()
    return PathDataLoader.Config(
        dataset=DEFAULT_BIG_TRAIN_LIST,
        split=split,
        shuffle_size=_si_int(base.shuffle_size),
        min_mixing=base.min_mixing,
        num_writers=base.num_writers,
        num_readers=base.num_readers,
        fps=base.fps,
        pipeline_dir=base.pipeline_dir,
        plan_only=base.plan_only,
        limit=base.limit,
        n_frames=base.n_frames,
        rgb=base.rgb,
        unvision=base.unvision,
    )


def _checkpoint_config() -> PathOnnxCheckpointManager.Config:
    frame_constants = frame_constants_from_fps(n_frames=N_FRAMES, frame_type=FRAME_TYPE)
    temporal_len = frame_constants["temporal_len"]
    vision_input_names = [ModelInputs.IMG, ModelInputs.BIG_IMG]
    temporal_policy_input_names = [
        ModelInputs.FEATURES,
        ModelInputs.DESIRE,
        ModelInputs.TRAFFIC,
        ModelInputs.ACTION_T,
    ]
    input_names = [
        *vision_input_names,
        *temporal_policy_input_names,
    ]
    input_shapes = [
        [1, *frame_constants["frame_shapes"][ModelInputs.IMG]],
        [1, *frame_constants["frame_shapes"][ModelInputs.BIG_IMG]],
        [1, temporal_len, TEMPORAL_INPUTS[ModelInputs.FEATURES][0]],
        [1, temporal_len, TEMPORAL_INPUTS[ModelInputs.DESIRE][0]],
        [1, temporal_len, TEMPORAL_INPUTS[ModelInputs.TRAFFIC][0]],
        [1, temporal_len, TEMPORAL_INPUTS[ModelInputs.ACTION_T][0]],
    ]
    return PathOnnxCheckpointManager.Config(
        enable=True,
        save_model_state_dict=True,
        export_onnx=True,
        enable_first_step_checkpoint=True,
        interval=512,
        input_names=input_names,
        input_shapes=input_shapes,
        input_dtypes=["float16"] * len(input_names),
        onnx_model_dtype="float16",
        vision_input_names=vision_input_names,
        temporal_policy_input_names=temporal_policy_input_names,
    )


def _si_int(value: str | int) -> int:
    suffixes = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    value = str(value).strip().lower()
    return int(float(value[:-1]) * suffixes[value[-1]]) if value[-1] in suffixes else int(value)


def _optimizer_config() -> OptimizersContainer.Config:
    common = {"lr": 1e-3, "betas": (0.9, 0.95), "eps": 1e-8}
    no_decay = r"(point_policy\.hydra|temporal_policy\.temporal_hydra)\.(final_layer|scale_layer)"
    return OptimizersContainer.Config(
        implementation="fused_opt_states_bf16",
        param_groups=[
            ParamGroupConfig(
                pattern=no_decay,
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "weight_decay": 0.0},
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "weight_decay": 3e-2},
            ),
        ],
    )


def _heads(heads) -> tuple[PathHead, ...]:
    return tuple(PathHead(head.name, head.output_size, head.mlp, head.scale) for head in heads)


def _hidden_dim(dim: int, mlp_mult: float, multiple_of: int = 256) -> int:
    hidden = int(dim * mlp_mult)
    return multiple_of * math.ceil(hidden / multiple_of)


def _mlp(dim: int, *, mlp_mult: float, bias: bool, dropout: float) -> PathMLP.Config:
    hidden = _hidden_dim(dim, mlp_mult)
    return PathMLP.Config(
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        c_fc=Linear.Config(in_features=dim, out_features=hidden, bias=bias, param_init=_LINEAR_INIT),
        c_proj=Linear.Config(in_features=hidden, out_features=dim, bias=bias, param_init=_LINEAR_INIT),
        act="gelu_tanh",
        dropout=dropout,
    )


def _encoder(in_features: int, dim: int) -> LinearEncoder.Config:
    return LinearEncoder.Config(
        in_layer=Linear.Config(in_features=in_features, out_features=dim, bias=True, param_init=_LINEAR_INIT),
        out_layer=Linear.Config(in_features=dim, out_features=dim, bias=False, param_init=_LINEAR_INIT),
    )


def _attention(*, dim: int, n_head: int, dropout: float) -> PathSelfAttention.Config:
    head_dim = dim // n_head
    return PathSelfAttention.Config(
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        q_norm=LayerNorm.Config(normalized_shape=head_dim, param_init=_NORM_INIT),
        k_norm=LayerNorm.Config(normalized_shape=head_dim, param_init=_NORM_INIT),
        c_attn=Linear.Config(in_features=dim, out_features=3 * dim, bias=True, param_init=_LINEAR_INIT),
        c_proj=Linear.Config(in_features=dim, out_features=dim, bias=True, param_init=_LINEAR_INIT),
        inner_attention=ScaledDotProductAttention.Config(),
        n_head=n_head,
        head_dim=head_dim,
        dropout=dropout,
    )


def _hydra(heads: tuple[PathHead, ...], *, in_features: int, mlp_mult: float) -> Hydra.Config:
    return Hydra.Config(
        heads=heads,
        head_mlps={head.name: _mlp(in_features, mlp_mult=mlp_mult, bias=False, dropout=0.0) for head in heads if head.mlp},
        final_layers={
            head.name: Linear.Config(
                in_features=in_features,
                out_features=head.output_size,
                bias=True,
                param_init=_LINEAR_INIT,
            )
            for head in heads
        },
        scale_layers={head.name: ScaleLayer.Config(n_features=head.output_size) for head in heads if head.scale},
    )
