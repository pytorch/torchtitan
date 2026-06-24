# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Config assembly + flavors for plan_vit, mirroring path/config_registry.py.

Width flavors scale n_head at fixed head_dim=64 (the clean muP axis); base = w256. Two cameras are
channel-stacked into in_channels=24 (no VAE). The trainer-side config functions live below the model side.
"""

from __future__ import annotations

import math
import os
from functools import partial
from xx.ml_tools.constants.model import SUPERCOMBO_FPS

import torch.nn as nn

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig
from torchtitan.components.tokenizer import NoOpTokenizer
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.experiments.path.dataset import PathDataLoader
from torchtitan.experiments.path.loss import PathLoss
from torchtitan.models.common import Embedding, LayerNorm, Linear
from torchtitan.models.common.attention import ScaledDotProductAttention
from torchtitan.protocols.model_spec import ModelSpec
from .model import (
    parallelize_plan_vit,
    PatchEmbed,
    PlanHead,
    PlanViT,
    PlanViTAttention,
    PlanViTBlock,
    PlanViTMLP,
)
from .trainer import PlanViTTrainer

_LINEAR_INIT = {
    "weight": partial(nn.init.normal_, mean=0.0, std=0.02),
    "bias": nn.init.zeros_,
}
_NORM_INIT = {"weight": nn.init.ones_, "bias": nn.init.zeros_}

HEAD_DIM = 64
N_LAYER = 8
INPUT_SIZE = (
    1,
    128,
    256,
)  # current frame; spatial ViT (temporal history is a later variant)
PATCH_SIZE = (1, 16, 8)
IN_CHANNELS = 24  # two cameras (IMG + BIG_IMG), 12 YUV channels each, channel-stacked
PLAN_SIZE = 15 * 33 * 2  # 990, laplacian mu+log-sigma
BASE_WIDTH = 256
PLAN_VIT_WIDTHS = {"w128": 128, "w256": 256, "w512": 512, "w1024": 1024, "w2048": 2048}


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
    # muP shrinks hidden/output init to 1/sqrt(fan_in) so pre-activations stay O(1) as width grows;
    # standard param holds the base-width variance 1/sqrt(BASE_WIDTH), so it fans out with width.
    return fan_in**-0.5 if mup else BASE_WIDTH**-0.5


def _ln(dim: int) -> LayerNorm.Config:
    return LayerNorm.Config(normalized_shape=dim, param_init=_NORM_INIT)


def _hidden(dim: int, mult: float, multiple_of: int = 256) -> int:
    return multiple_of * math.ceil(int(dim * mult) / multiple_of)


def _attention(
    dim: int, n_head: int, *, mup: bool, qk_norm: bool = True
) -> PlanViTAttention.Config:
    head_dim = dim // n_head
    return PlanViTAttention.Config(
        norm=_ln(dim),
        q_norm=_ln(head_dim) if qk_norm else None,
        k_norm=_ln(head_dim) if qk_norm else None,
        c_attn=_lin(dim, 3 * dim, std=_hidden_std(dim, mup=mup)),
        c_proj=_lin(dim, dim, std=_hidden_std(dim, mup=mup) / math.sqrt(2 * N_LAYER)),
        inner_attention=ScaledDotProductAttention.Config(),
        n_head=n_head,
        head_dim=head_dim,
        dropout=0.0,
    )


def _mlp(dim: int, *, mup: bool, mult: float = 4.0) -> PlanViTMLP.Config:
    hidden = _hidden(dim, mult)
    return PlanViTMLP.Config(
        norm=_ln(dim),
        c_fc=_lin(dim, hidden, std=_hidden_std(dim, mup=mup)),
        c_proj=_lin(
            hidden, dim, std=_hidden_std(hidden, mup=mup) / math.sqrt(2 * N_LAYER)
        ),
        act="gelu_tanh",
        dropout=0.0,
    )


def _model_config(flavor: str, *, mup: bool, qk_norm: bool = True) -> PlanViT.Config:
    n_embd = PLAN_VIT_WIDTHS[flavor]
    n_head = n_embd // HEAD_DIM
    pt, ph, pw = PATCH_SIZE
    patch_dim = pt * IN_CHANNELS * ph * pw
    t, h, w = INPUT_SIZE
    num_patches = (t // pt) * (h // ph) * (w // pw)
    return PlanViT.Config(
        input_size=INPUT_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
        n_embd=n_embd,
        output_mult=1.0,  # no readout multiplier: the 1/m mult broke output width-stability (coord check)
        patch_embed=PatchEmbed.Config(
            proj=_lin(
                patch_dim, n_embd, std=patch_dim**-0.5
            ),  # input embed: width-independent
            patch_size=PATCH_SIZE,
        ),
        pos_embedding=Embedding.Config(
            num_embeddings=num_patches, embedding_dim=n_embd, param_init=_LINEAR_INIT
        ),
        blocks=[
            PlanViTBlock.Config(
                attention=_attention(n_embd, n_head, mup=mup, qk_norm=qk_norm),
                mlp=_mlp(n_embd, mup=mup),
            )
            for _ in range(N_LAYER)
        ],
        norm=_ln(n_embd),
        plan_head=PlanHead.Config(
            norm=_ln(n_embd),
            head=_lin(n_embd, PLAN_SIZE, std=_hidden_std(n_embd, mup=mup)),
        ),
    )


def model_registry(flavor: str, *, mup: bool) -> ModelSpec:
    return ModelSpec(
        name="plan_vit",
        flavor=flavor,
        model=_model_config(flavor, mup=mup),
        parallelize_fn=parallelize_plan_vit,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )


STEPS = 512  # per-run step budget; override with training.steps=N on the CLI
# learning rate is the muTransfer sweep axis: one run per (flavor, lr); set with `-e PLAN_VIT_LR=...`
SWEEP_LR = float(os.getenv("PLAN_VIT_LR", "3e-4"))
# hidden + output matrix weights get muP lr eta/m; input embed, norms, biases get eta
MUP_PATTERN = (
    r"^(blocks\.\d+\.attention\.c_attn|blocks\.\d+\.attention\.c_proj"
    r"|blocks\.\d+\.mlp\.c_fc|blocks\.\d+\.mlp\.c_proj|plan_head\.head)\.weight$"
)


def _si_int(value: str | int) -> int:
    suffixes = {"k": 1_000, "m": 1_000_000, "g": 1_000_000_000}
    value = str(value).strip().lower()
    return (
        int(float(value[:-1]) * suffixes[value[-1]])
        if value[-1] in suffixes
        else int(value)
    )


def _dataloader_config(*, split: str) -> PathDataLoader.Config:
    from xx.common.basedir import XX_BASEDIR
    from xx.datasets.constants import BASE_DIR_GT_10M
    from xx.training.path.config import DatasetConfig as XXPathDatasetConfig

    base = XXPathDatasetConfig(fps=SUPERCOMBO_FPS, plan_only=True)
    return PathDataLoader.Config(
        # prune-10M study data: a seeded random 10k sample of the 10M store (training_2026_02)
        dataset=os.path.join(XX_BASEDIR, "datasets/lists/prune10m_random10k_seed0.txt"),
        split=split,
        shuffle_size=_si_int(base.shuffle_size),
        min_mixing=base.min_mixing,
        num_writers=base.num_writers,
        num_readers=base.num_readers,
        fps=base.fps,
        pipeline_dir=BASE_DIR_GT_10M,  # the 10M store, not the 2.5M big-train list
        plan_only=base.plan_only,
        limit=base.limit,
        n_frames=base.n_frames,
        rgb=base.rgb,
        unvision=base.unvision,
    )


def _optimizer_config(
    flavor: str, *, mup: bool, lr: float, wd: float
) -> OptimizersContainer.Config:
    m = PLAN_VIT_WIDTHS[flavor] / BASE_WIDTH
    common = {"betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": wd}
    if mup:
        groups = [
            ParamGroupConfig(
                pattern=MUP_PATTERN,
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "lr": lr / m},
            ),
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "lr": lr},
            ),
        ]
    else:
        groups = [
            ParamGroupConfig(
                pattern=r".*",
                optimizer_name="AdamW",
                optimizer_kwargs={**common, "lr": lr},
            )
        ]
    return OptimizersContainer.Config(
        implementation="fused_opt_states_bf16", param_groups=groups
    )


def _plan_vit(
    flavor: str, *, mup: bool, lr: float = SWEEP_LR, wd: float = 3e-2
) -> PlanViTTrainer.Config:
    return PlanViTTrainer.Config(
        loss=PathLoss.Config(),
        model_spec=model_registry(flavor, mup=mup),
        tokenizer=NoOpTokenizer.Config(),
        dataloader=_dataloader_config(split="train"),
        optimizer=_optimizer_config(flavor, mup=mup, lr=lr, wd=wd),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=round(STEPS * 0.1),
            total_steps=STEPS,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
        training=TrainingConfig(
            local_batch_size=16,
            global_batch_size=-1,
            seq_len=1,
            steps=STEPS,
            max_norm=1.0,
            dtype="float32",
            mixed_precision_param="bfloat16",
            mixed_precision_reduce="float32",
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1, data_parallel_shard_degree=8
        ),
        checkpoint=CheckpointManager.Config(enable=False),
        metrics=MetricsProcessor.Config(
            log_freq=10, enable_reporterv2=True, save_freq=STEPS
        ),
        debug=DebugConfig(seed=0),
    )


def plan_vit_standard_w256() -> PlanViTTrainer.Config:
    return _plan_vit("w256", mup=False)


def plan_vit_standard_w512() -> PlanViTTrainer.Config:
    return _plan_vit("w512", mup=False)


def plan_vit_standard_w1024() -> PlanViTTrainer.Config:
    return _plan_vit("w1024", mup=False)


def plan_vit_standard_w2048() -> PlanViTTrainer.Config:
    return _plan_vit("w2048", mup=False)


def plan_vit_mup_w256() -> PlanViTTrainer.Config:
    return _plan_vit("w256", mup=True)


def plan_vit_mup_w512() -> PlanViTTrainer.Config:
    return _plan_vit("w512", mup=True)


def plan_vit_mup_w1024() -> PlanViTTrainer.Config:
    return _plan_vit("w1024", mup=True)


def plan_vit_mup_w2048() -> PlanViTTrainer.Config:
    return _plan_vit("w2048", mup=True)


def plan_vit() -> PlanViTTrainer.Config:
    return plan_vit_mup_w256()
