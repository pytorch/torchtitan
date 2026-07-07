# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import os
from dataclasses import replace

from xx.common.basedir import XX_BASEDIR
from xx.datasets.constants import DEFAULT_10M_TRAIN_LIST

from torchtitan.components.lr_scheduler import LRSchedulersContainer

from ..path.config_registry import dp_degrees as _dp_degrees, vit as _vit
from ..path.trainer import final_checkpoint_config, PathTrainer

BIG_FLAVOR = "w2048"
BIG_STEPS = 15360
BIG_LR = 1e-2
DEPLOY_WARMUP_STEPS = 1024
DEPLOY_DECAY_RATIO = 0.1
RANDOM1M_LIST = os.path.join(XX_BASEDIR, "datasets/lists/prune10m_random1m_seed0.txt")


def _ckpt(seed: int, dataset: str | None = None) -> PathTrainer.Config:
    cfg = _vit(BIG_FLAVOR, mup=True, lr=BIG_LR)
    dataset = dataset or cfg.dataloader.dataset
    stem = os.path.splitext(os.path.basename(dataset))[0]
    return replace(
        cfg,
        training=replace(cfg.training, steps=BIG_STEPS),
        dataloader=replace(cfg.dataloader, dataset=dataset),
        debug=replace(cfg.debug, seed=seed),
        checkpoint=final_checkpoint_config(
            flavor=BIG_FLAVOR, stem=stem, seed=seed, steps=BIG_STEPS
        ),
    )


def vit_mup_w2048_ckpt() -> PathTrainer.Config:
    return _ckpt(seed=0)


def vit_mup_w2048_ckpt_rand1m_s0() -> PathTrainer.Config:
    return _ckpt(0, RANDOM1M_LIST)


def vit_mup_w2048_ckpt_rand1m_s1() -> PathTrainer.Config:
    return _ckpt(1, RANDOM1M_LIST)


def vit_mup_w2048_ckpt_rand1m_s2() -> PathTrainer.Config:
    return _ckpt(2, RANDOM1M_LIST)


def vit_mup_w2048_ckpt_full10m_s0() -> PathTrainer.Config:
    return _ckpt(0, DEFAULT_10M_TRAIN_LIST)


def vit_mup_w2048_ckpt_full10m_s1() -> PathTrainer.Config:
    return _ckpt(1, DEFAULT_10M_TRAIN_LIST)


def vit_mup_w2048_ckpt_full10m_s2() -> PathTrainer.Config:
    return _ckpt(2, DEFAULT_10M_TRAIN_LIST)


def vit_mup_w256_deploy_schedule() -> PathTrainer.Config:
    cfg = _vit("w256", mup=True, lr=BIG_LR)
    num_nodes, local_world_size = _dp_degrees()
    world_size = num_nodes * local_world_size
    return replace(
        cfg,
        training=replace(
            cfg.training,
            steps=BIG_STEPS,
            global_batch_size=cfg.training.local_batch_size * world_size * 2,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=DEPLOY_WARMUP_STEPS,
            total_steps=BIG_STEPS,
            decay_ratio=DEPLOY_DECAY_RATIO,
            decay_type="cosine",
            min_lr_factor=0.0,
        ),
    )
