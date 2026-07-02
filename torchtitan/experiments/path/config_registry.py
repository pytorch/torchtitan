# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import dataclasses
import math
import os
from functools import partial
from xx.common.basedir import XX_BASEDIR
from xx.datasets.constants import BASE_DIR_GT_10M
from xx.datasets.helpers import DEFAULT_BIG_TRAIN_LIST
from xx.ml_tools.constants.model import (
    frame_constants_from_fps,
    FRAME_TYPE,
    INPUT_FRAMES_NAMES,
    ModelInputs,
    N_FRAMES,
    SUPERCOMBO_FPS,
    TEMPORAL_INPUTS,
)
from xx.training.path.config import DatasetConfig as XXPathDatasetConfig
from xx.training.path.hydra_configs import (
    DRIVING_HEADS,
    META_HEADS,
    PLAN_HEAD_SIZE,
    POSE_HEADS,
    TEMPORAL_META_HEADS,
)

import torch.nn as nn

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.tokenizer import NoOpTokenizer
from torchtitan.config import (
    CompileConfig,
    DebugConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed.activation_checkpoint import FullAC
from torchtitan.models.common import Embedding, LayerNorm, Linear
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.protocols.model_spec import ModelSpec

from .dataset import PathDataLoader
from .loss import PathLoss
from .model import (
    Hydra,
    LinearEncoder,
    parallelize_path,
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
)
from .onnx_checkpoint import PathOnnxCheckpointManager
from .trainer import PathTrainer
from .validate import PathValidator
from .vit import parallelize_vit, PatchEmbed, PlanHead, PlanViT, PlanViTLoss


_LINEAR_INIT = {
    "weight": partial(nn.init.normal_, mean=0.0, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_, "bias": nn.init.zeros_}

VIT_HEAD_DIM = 64
VIT_NUM_LAYERS = 8
VIT_INPUT_SIZE = (1, 128, 256)
VIT_PATCH_SIZE = (1, 16, 8)
VIT_IN_CHANNELS = 24
VIT_BASE_WIDTH = 256
VIT_WIDTHS = {
    f"w{d}": d
    for d in (
        64,
        128,
        192,
        256,
        320,
        384,
        448,
        512,
        640,
        896,
        1024,
        1280,
        1536,
        1792,
        2048,
        3072,
    )
}
VIT_STEPS = 512
MUP_PATTERN = (
    r"^(blocks\.\d+\.attention\.c_attn|blocks\.\d+\.attention\.c_proj"
    r"|blocks\.\d+\.mlp\.c_fc|blocks\.\d+\.mlp\.c_proj)\.weight$"
)


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


def _dp_degrees() -> tuple[int, int]:
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    world_size = int(os.environ.get("WORLD_SIZE", str(local_world_size)))
    num_nodes = int(
        os.environ.get("GROUP_WORLD_SIZE", str(world_size // local_world_size))
    )
    return num_nodes, local_world_size


dp_degrees = _dp_degrees


def _path(flavor: str) -> PathTrainer.Config:
    steps = 1024 * 100
    validation_freq = 1024
    reports = {
        name: [validation_freq, steps // 2, steps]
        for name in (
            "analyse_driving",
            "analyse_lat_no_noise",
            "analyse_cones",
            "analyse_lights",
            "analyse_stop",
            "analyse_hard_brake",
        )
    }
    reports["analyse_dataset"] = [validation_freq]
    mixed_precision_param = "bfloat16"
    num_nodes, local_world_size = _dp_degrees()
    reporterv2_host = os.getenv("REPORTERV2_HOST")
    reporterv2_training_id = os.getenv("REPORTERV2_TRAINING_ID")
    checkpoint_base_folder = (
        f"{reporterv2_host.rstrip('/')}/checkpoint" if reporterv2_host else ""
    )
    fps = SUPERCOMBO_FPS
    plan_only = False
    return PathTrainer.Config(
        loss=PathLoss.Config(),
        model_spec=model_registry(flavor),
        tokenizer=NoOpTokenizer.Config(),
        dataloader=_dataloader_config(split="train", fps=fps, plan_only=plan_only),
        optimizer=_optimizer_config(),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=round(steps * 0.01),
            total_steps=steps,
            decay_ratio=0.2,
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
        checkpoint=_checkpoint_config(
            folder=reporterv2_training_id or "checkpoint",
            base_folder=checkpoint_base_folder,
            interval=validation_freq,
        ),
        fps=fps,
        activation_checkpoint=FullAC.Config(),
        compile=CompileConfig(enable=True, components=["model"]),
        metrics=MetricsProcessor.Config(
            log_freq=16, enable_reporterv2=True, save_freq=validation_freq
        ),
        validator=PathValidator.Config(
            enable=True,
            freq=validation_freq,
            steps=32,
            dataloader=_dataloader_config(split="val", fps=fps, plan_only=plan_only),
            mixed_precision_param=mixed_precision_param,
            reports=reports,
        ),
        debug=DebugConfig(seed=0),
    )


def _model_config(flavor: str) -> PathModel.Config:
    vision_features = 512
    n_frames_input = N_FRAMES
    input_frame_names = INPUT_FRAMES_NAMES
    input_frame_type = FRAME_TYPE
    frame_constants = frame_constants_from_fps(
        n_frames=n_frames_input, frame_type=input_frame_type
    )
    in_channels = sum(
        frame_constants["frame_shapes"][name][0] for name in input_frame_names
    )
    block_size = len(frame_constants["history_idxs"])
    temporal_len = frame_constants["temporal_len"]
    dim = vision_features

    return PathModel.Config(
        n_frames_input=n_frames_input,
        input_frame_names=tuple(input_frame_names),
        frame_type=input_frame_type,
        vision=Vision.Config(
            flavor=flavor,
            input_frame_names=tuple(input_frame_names),
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
                desire_encoder=_encoder(
                    TEMPORAL_INPUTS[ModelInputs.DESIRE][0] * temporal_len, dim
                ),
                traffic_encoder=_encoder(TEMPORAL_INPUTS[ModelInputs.TRAFFIC][0], dim),
                action_t_encoder=_encoder(
                    TEMPORAL_INPUTS[ModelInputs.ACTION_T][0], dim
                ),
                transformer=PathTransformer.Config(
                    layers=[
                        PathTransformerBlock.Config(
                            attention=_attention(dim=dim, n_head=8, dropout=0.1),
                            mlp=_mlp(dim, mlp_mult=2, bias=True, dropout=0.1),
                        )
                        for _ in range(4)
                    ]
                ),
                pos_embedding=Embedding.Config(
                    num_embeddings=block_size,
                    embedding_dim=dim,
                    param_init=_LINEAR_INIT,
                ),
                block_size=block_size,
                dense_training_outputs=True,
            ),
            temporal_hydra=_hydra(
                _heads(DRIVING_HEADS + TEMPORAL_META_HEADS), in_features=dim, mlp_mult=2
            ),
            history_idxs=tuple(int(x) for x in frame_constants["history_idxs"]),
        ),
    )


def _dataloader_config(
    *, split: str, fps: int, plan_only: bool
) -> PathDataLoader.Config:
    base = XXPathDatasetConfig(fps=fps, plan_only=plan_only)
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


def _checkpoint_config(
    folder: str, base_folder: str, interval: int
) -> PathOnnxCheckpointManager.Config:
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
        keep_latest_k=0,  # keep all checkpoints
        enable=True,
        checkpoint_base_folder=base_folder,
        save_model_state_dict=True,  # another copy of full state dict
        export_onnx=True,
        enable_first_step_checkpoint=True,
        folder=folder,
        interval=interval,
        input_names=input_names,
        input_shapes=input_shapes,
        input_dtypes=["float16"] * len(input_names),
        onnx_model_dtype="float16",  # WIP: test if fp16 doesn't degrade performance
        vision_input_names=vision_input_names,
        temporal_policy_input_names=temporal_policy_input_names,
    )


def _si_int(value: str | int) -> int:
    suffixes = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    value = str(value).strip().lower()
    return (
        int(float(value[:-1]) * suffixes[value[-1]])
        if value[-1] in suffixes
        else int(value)
    )


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
    return tuple(
        PathHead(head.name, head.output_size, head.mlp, head.scale) for head in heads
    )


def _hidden_dim(dim: int, mlp_mult: float, multiple_of: int = 256) -> int:
    hidden = int(dim * mlp_mult)
    return multiple_of * math.ceil(hidden / multiple_of)


def _mlp(dim: int, *, mlp_mult: float, bias: bool, dropout: float) -> PathMLP.Config:
    hidden = _hidden_dim(dim, mlp_mult)
    return PathMLP.Config(
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        c_fc=Linear.Config(
            in_features=dim, out_features=hidden, bias=bias, param_init=_LINEAR_INIT
        ),
        c_proj=Linear.Config(
            in_features=hidden, out_features=dim, bias=bias, param_init=_LINEAR_INIT
        ),
        act="gelu_tanh",
        dropout=dropout,
    )


def _encoder(in_features: int, dim: int) -> LinearEncoder.Config:
    return LinearEncoder.Config(
        in_layer=Linear.Config(
            in_features=in_features,
            out_features=dim,
            bias=True,
            param_init=_LINEAR_INIT,
        ),
        out_layer=Linear.Config(
            in_features=dim, out_features=dim, bias=False, param_init=_LINEAR_INIT
        ),
    )


def _attention(*, dim: int, n_head: int, dropout: float) -> PathSelfAttention.Config:
    head_dim = dim // n_head
    return PathSelfAttention.Config(
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        q_norm=LayerNorm.Config(normalized_shape=head_dim, param_init=_NORM_INIT),
        k_norm=LayerNorm.Config(normalized_shape=head_dim, param_init=_NORM_INIT),
        c_attn=Linear.Config(
            in_features=dim, out_features=3 * dim, bias=True, param_init=_LINEAR_INIT
        ),
        c_proj=Linear.Config(
            in_features=dim, out_features=dim, bias=True, param_init=_LINEAR_INIT
        ),
        inner_attention=ScaledDotProductAttention.Config(),
        n_head=n_head,
        head_dim=head_dim,
        dropout=dropout,
    )


def _hydra(
    heads: tuple[PathHead, ...], *, in_features: int, mlp_mult: float
) -> Hydra.Config:
    return Hydra.Config(
        heads=heads,
        head_mlps={
            head.name: _mlp(in_features, mlp_mult=mlp_mult, bias=False, dropout=0.0)
            for head in heads
            if head.mlp
        },
        final_layers={
            head.name: Linear.Config(
                in_features=in_features,
                out_features=head.output_size,
                bias=True,
                param_init=_LINEAR_INIT,
            )
            for head in heads
        },
        scale_layers={
            head.name: ScaleLayer.Config(n_features=head.output_size)
            for head in heads
            if head.scale
        },
    )


def _lin(in_f: int, out_f: int, *, std: float, bias: bool = True) -> Linear.Config:
    return Linear.Config(
        in_features=in_f,
        out_features=out_f,
        bias=bias,
        param_init={
            "weight": partial(nn.init.normal_, mean=0.0, std=std),
            "bias": nn.init.zeros_,
        },
    )


def _hidden_std(fan_in: int, *, mup: bool) -> float:
    return fan_in**-0.5 if mup else VIT_BASE_WIDTH**-0.5


def _vit_attention(dim: int, *, n_head: int, mup: bool) -> PathSelfAttention.Config:
    head_dim = dim // n_head
    return PathSelfAttention.Config(
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        q_norm=LayerNorm.Config(normalized_shape=head_dim, param_init=_NORM_INIT),
        k_norm=LayerNorm.Config(normalized_shape=head_dim, param_init=_NORM_INIT),
        c_attn=_lin(dim, 3 * dim, std=_hidden_std(dim, mup=mup)),
        c_proj=_lin(
            dim, dim, std=_hidden_std(dim, mup=mup) / math.sqrt(2 * VIT_NUM_LAYERS)
        ),
        inner_attention=ScaledDotProductAttention.Config(),
        n_head=n_head,
        head_dim=head_dim,
        dropout=0.0,
        is_causal=False,
    )


def _vit_mlp(dim: int, *, mup: bool, mult: float = 4.0) -> PathMLP.Config:
    hidden = _hidden_dim(dim, mult)
    return PathMLP.Config(
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        c_fc=_lin(dim, hidden, std=_hidden_std(dim, mup=mup)),
        c_proj=_lin(
            hidden,
            dim,
            std=_hidden_std(hidden, mup=mup) / math.sqrt(2 * VIT_NUM_LAYERS),
        ),
        act="gelu_tanh",
        dropout=0.0,
    )


def _vit_model_config(flavor: str, *, mup: bool) -> PlanViT.Config:
    dim = VIT_WIDTHS[flavor]
    if dim % VIT_HEAD_DIM != 0:
        raise ValueError(
            f"vit width {dim} must be a multiple of head_dim {VIT_HEAD_DIM}"
        )
    n_head = dim // VIT_HEAD_DIM
    pt, ph, pw = VIT_PATCH_SIZE
    patch_dim = pt * VIT_IN_CHANNELS * ph * pw
    t, h, w = VIT_INPUT_SIZE
    num_patches = (t // pt) * (h // ph) * (w // pw)
    return PlanViT.Config(
        output_mult=(VIT_BASE_WIDTH / dim) if mup else 1.0,
        mean=255 / 2,
        std=255 / 4,
        patch_embed=PatchEmbed.Config(
            proj=_lin(patch_dim, dim, std=patch_dim**-0.5),
            patch_size=VIT_PATCH_SIZE,
        ),
        pos_embedding=Embedding.Config(
            num_embeddings=num_patches, embedding_dim=dim, param_init=_LINEAR_INIT
        ),
        blocks=[
            PathTransformerBlock.Config(
                attention=_vit_attention(dim, n_head=n_head, mup=mup),
                mlp=_vit_mlp(dim, mup=mup),
            )
            for _ in range(VIT_NUM_LAYERS)
        ],
        norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
        plan_head=PlanHead.Config(
            norm=LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT),
            head=_lin(dim, PLAN_HEAD_SIZE, std=VIT_BASE_WIDTH**-0.5),
        ),
    )


vit_model_config = _vit_model_config


def vit_model_registry(flavor: str, *, mup: bool) -> ModelSpec:
    return ModelSpec(
        name="path",
        flavor=flavor,
        model=_vit_model_config(flavor, mup=mup),
        parallelize_fn=parallelize_vit,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


def _vit_dataloader_config(*, split: str) -> PathDataLoader.Config:
    dataset = (
        "datasets/lists/prune10m_val.txt"
        if split == "val"
        else "datasets/lists/prune10m_random100k_seed0.txt"
    )
    return dataclasses.replace(
        _dataloader_config(split=split, fps=SUPERCOMBO_FPS, plan_only=True),
        dataset=os.path.join(XX_BASEDIR, dataset),
        pipeline_dir=BASE_DIR_GT_10M,
    )


vit_dataloader_config = _vit_dataloader_config


def _vit_optimizer_config(
    flavor: str, *, mup: bool, lr: float, wd: float
) -> OptimizersContainer.Config:
    m = VIT_WIDTHS[flavor] / VIT_BASE_WIDTH
    common = {"betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": wd}
    catch_all = ParamGroupConfig(
        pattern=r".*",
        optimizer_name="AdamW",
        optimizer_kwargs=common,
    )
    mup_group = ParamGroupConfig(
        pattern=MUP_PATTERN,
        optimizer_name="AdamW",
        lr_mult=1.0 / m,
        optimizer_kwargs={**common, "weight_decay": wd * m},
    )
    groups = [mup_group, catch_all] if mup else [catch_all]
    return OptimizersContainer.Config(
        implementation="fused_opt_states_bf16", lr=lr, param_groups=groups
    )


def _vit(
    flavor: str, *, mup: bool, lr: float = 3e-4, wd: float = 0.0125
) -> PathTrainer.Config:
    num_nodes, local_world_size = _dp_degrees()
    return PathTrainer.Config(
        loss=PlanViTLoss.Config(),
        model_spec=vit_model_registry(flavor, mup=mup),
        tokenizer=NoOpTokenizer.Config(),
        dataloader=_vit_dataloader_config(split="train"),
        optimizer=_vit_optimizer_config(flavor, mup=mup, lr=lr, wd=wd),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=round(VIT_STEPS * 0.1),
            total_steps=None,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=16,
            global_batch_size=-1,
            seq_len=1,
            steps=VIT_STEPS,
            max_norm=1.0,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=num_nodes,
            data_parallel_shard_degree=local_world_size,
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        metrics=MetricsProcessor.Config(
            log_freq=10, enable_reporterv2=True, save_freq=VIT_STEPS
        ),
        validator=PathValidator.Config(
            enable=True,
            freq=1024,
            steps=32,
            dataloader=_vit_dataloader_config(split="val"),
            mixed_precision_param="bfloat16",
        ),
        fps=SUPERCOMBO_FPS,
        debug=DebugConfig(seed=0),
    )


vit = _vit


def vit_standard_w256() -> PathTrainer.Config:
    return _vit("w256", mup=False)


def vit_standard_w512() -> PathTrainer.Config:
    return _vit("w512", mup=False)


def vit_standard_w1024() -> PathTrainer.Config:
    return _vit("w1024", mup=False)


def vit_standard_w2048() -> PathTrainer.Config:
    return _vit("w2048", mup=False)


def vit_mup_w256() -> PathTrainer.Config:
    return _vit("w256", mup=True)


def vit_mup_w512() -> PathTrainer.Config:
    return _vit("w512", mup=True)


def vit_mup_w1024() -> PathTrainer.Config:
    return _vit("w1024", mup=True)


def vit_mup_w2048() -> PathTrainer.Config:
    return _vit("w2048", mup=True)


def vit_mup_w64() -> PathTrainer.Config:
    return _vit("w64", mup=True)


def vit_mup_w128() -> PathTrainer.Config:
    return _vit("w128", mup=True)


def vit_mup_w192() -> PathTrainer.Config:
    return _vit("w192", mup=True)


def vit_mup_w320() -> PathTrainer.Config:
    return _vit("w320", mup=True)


def vit_mup_w384() -> PathTrainer.Config:
    return _vit("w384", mup=True)


def vit_mup_w448() -> PathTrainer.Config:
    return _vit("w448", mup=True)


def vit_mup_w640() -> PathTrainer.Config:
    return _vit("w640", mup=True)


def vit_mup_w896() -> PathTrainer.Config:
    return _vit("w896", mup=True)


def vit_mup_w1280() -> PathTrainer.Config:
    return _vit("w1280", mup=True)


def vit_mup_w1536() -> PathTrainer.Config:
    return _vit("w1536", mup=True)


def vit_mup_w1792() -> PathTrainer.Config:
    return _vit("w1792", mup=True)


def vit_mup_w3072() -> PathTrainer.Config:
    return _vit("w3072", mup=True)
