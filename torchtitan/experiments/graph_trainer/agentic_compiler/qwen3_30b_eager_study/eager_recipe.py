# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager Qwen3 30B-A3B DistMoE MXFP8 recipes in the graph loop's measurement setup.

The base is ``qwen3_30b_a3b_dist_moe_mxfp8_8gpu`` (the eager twin of the graph loop's
``graph_trainer_qwen3_30b_a3b_dist_moe_mxfp8_8gpu``) with the loop's changes: one
32K-token microbatch per step, FA4 varlen attention, deterministic, seed 42.

    RUN=<dir> PYTHONPATH=<this dir>:. NGPU=8 LOCAL_RANK=0 \\
        python -m torchtitan.train --module eager_recipe --config <name> --comm-backend fake

Configs:
- eager_lc: the recipe's local_compile regions (loss, rope, fp32_to_bf16_split, ...).
- eager_nolc: local_compile.regions = [], the aten ops GraphTrainer traces.
- *_prof: 10 steps, step 10 profiled, accelerator graphs off (as the graph loop measured).
- *_tps: 20 steps, accelerator graphs as the recipe sets them, no profiler.
"""

import os

from torchtitan.models.common.attention.attention import VarlenInnerAttention
from torchtitan.observability.metrics import TensorBoardLogger

from torchtitan_recipes.graph_trainer.megatron_bridge_benchmarks import (
    qwen3_30b_a3b_dist_moe_mxfp8_8gpu,
)


def _base(*, local_compile: bool):
    config = qwen3_30b_a3b_dist_moe_mxfp8_8gpu()
    config.training.num_tokens_per_train_step = -1
    for layer in config.model.layers:
        inner = layer.attention.inner_attention
        assert isinstance(inner, VarlenInnerAttention.Config), type(inner)
        inner.sdpa_backends = ["FLASH_ATTENTION"]
    if not local_compile:
        config.model.local_compile.regions = []
    if os.environ.get("LC_REGIONS"):
        # Only the listed local_compile regions (e.g. LC_REGIONS=rope).
        config.model.local_compile.regions = os.environ["LC_REGIONS"].split(",")
    config.debug.seed = 42
    config.debug.deterministic = True
    run = os.path.abspath(os.environ["RUN"])
    os.makedirs(run, exist_ok=True)
    config.dump_folder = run
    if not any(isinstance(c, TensorBoardLogger.Config) for c in config.metrics.loggers):
        config.metrics.loggers = [*config.metrics.loggers, TensorBoardLogger.Config()]
    config.metrics.save_tb_folder = os.path.join(run, "tb")
    return config


def _prof(config):
    config.training.steps = 10
    config.training.disable_accelerator_graphs = True
    config.profiler.enable_profiling = True
    config.profiler.enable_memory_snapshot = False
    config.profiler.save_traces_folder = os.path.join(config.dump_folder, "profile")
    config.profiler.profile_freq = 10
    config.profiler.profiler_warmup = 3
    config.profiler.profiler_active = 1
    config.profiler.profiler_repeat = 1
    return config


def _tps(config):
    config.training.steps = 20
    config.profiler.enable_profiling = False
    return config


def eager_lc_prof():
    return _prof(_base(local_compile=True))


def eager_nolc_prof():
    return _prof(_base(local_compile=False))


def eager_lc_tps():
    return _tps(_base(local_compile=True))


def eager_nolc_tps():
    return _tps(_base(local_compile=False))


def _kernels(config):
    """Graph-loop kernels in eager. GK: comma-separated Functions (attention_input,
    attention_output, router, loss, expert_weight_prep); LOCAL_COMPILE=1 keeps the recipe's local_compile regions."""
    names = [n for n in os.environ.get("GK", "").split(",") if n]
    flags = {n: n in names for n in ("attention_input", "attention_output", "router")}
    imports = []
    if any(flags.values()):
        imports.append(("override_qwen3.graph_kernel_block", flags))
    if "loss" in names:
        imports.append("override_loss.graph_kernel_loss")
    if "expert_weight_prep" in names:
        import graph_kernels

        graph_kernels.install_expert_weight_prep()
    config.override.imports = imports
    return config


def eager_kernels_prof():
    lc = os.environ.get("LOCAL_COMPILE", "0") == "1"
    return _kernels(_prof(_base(local_compile=lc)))


def eager_kernels_tps():
    lc = os.environ.get("LOCAL_COMPILE", "0") == "1"
    return _kernels(_tps(_base(local_compile=lc)))
